"""Audit domain-overlap banks without opening task query outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
from sklearn.metrics import roc_auc_score

import shared_domain_overlap_banks_v167 as bank
from prepare_data import ROOT, atomic_json, protocol


def audit_case(case):
    source = bank.source
    assert source.sha(case['input_file']) == case['input_sha256']
    with np.load(case['input_file']) as inputs:
        assert 'y_target' not in inputs.files
        assert not set(inputs['source_indices']) & set(inputs['target_indices'])
        X0, y0, Q0, numeric = (inputs[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    task, tid = case['task'], case['task_id']
    results = []
    for path in sorted((bank.OUT/'banks'/task/str(tid)).glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert source.sha(path) == meta['output_sha256']
        assert meta['original_input_sha256'] == case['input_sha256']
        assert meta['final_query_labels_opened'] is False
        assert meta['new_constructor_receives_generator_or_strength'] is False
        if meta['rho']:
            X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, meta['family'], meta['rho'], tid)
        else:
            X, y, Q = X0, y0, Q0
        assert protocol.array_sha256(y) == meta['source_y_sha256']
        a, b, partitions = bank.domain_rule.quality(X, Q, numeric)
        assert partitions == meta['domain_crossfit_partitions']
        with np.load(path) as data, np.load(meta['original_bank']) as original:
            np.testing.assert_array_equal(a, data['Domain_source_probability'])
            np.testing.assert_array_equal(b, data['Domain_target_probability'])
            np.testing.assert_array_equal(data['Base'], original['Base'])
            slots = meta['replacement_slots']
            for slot in set(range(5))-set(slots):
                np.testing.assert_array_equal(data['members'][slot], original['members'][slot])
            for ids, fraction in zip(meta['selected_source_rows'], (.75, .9)):
                expected = bank.selection.select(y, a, task, fraction)
                np.testing.assert_array_equal(ids, expected)
                assert len(ids) <= 256 and len(set(ids)) == len(ids)
                if task == 'classification':
                    assert set(np.unique(y[ids])) == set(np.unique(y))
        results.append({'task': task, 'data_id': case['data_id'], 'family': meta['family'], 'rho': meta['rho'],
            'domain_auc': float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))})
    return results


def main(workers):
    source = bank.source
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert source.sha(bank.__file__) == plan['script_sha256']
    assert source.sha(bank.domain_rule.__file__) == plan['domain_rule_sha256']
    assert source.sha(bank.selection.__file__) == plan['selection_sha256']
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(audit_case, c) for c in cases]):
            rows.extend(future.result())
    assert len(rows) == 1327
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(bank.OUT/'domain_crossfit_diagnostics.json', rows)
    atomic_json(bank.OUT/'independent_constructor_audit.json', {
        'status': 'PASS', 'conditions_replayed': len(rows), 'task_query_labels_opened': False,
        'domain_source_target_features_and_domain_labels_only': True,
        'crossfit_domain_models_recomputed_exactly': True, 'selected_rows_recomputed': True,
        'Base_and_three_original_members_bitwise_unchanged': True, 'Markdown_bytes': 0})
    print('DOMAIN_OVERLAP_AUDIT_PASS', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args().workers)
