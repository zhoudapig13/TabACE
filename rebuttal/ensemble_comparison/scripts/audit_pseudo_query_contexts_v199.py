"""Replay pseudo-label contexts without reading any final query outcome."""
import json
import numpy as np
import shared_pseudo_query_banks_v198 as bank
from prepare_data import ROOT, atomic_json, protocol


def main():
    source = bank.source
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert source.sha(bank.__file__) == plan['script_sha256']
    assert source.sha(bank.constructor.__file__) == plan['constructor_sha256']
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    count, contexts = 0, 0
    for c in cases:
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as data:
            assert 'y_target' not in data.files
            assert not set(data['source_indices']) & set(data['target_indices'])
            X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        for path in sorted((bank.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert source.sha(path) == meta['output_sha256']
            assert source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            assert meta['final_query_outcome_files_opened'] is False
            assert meta['learned_query_label_teacher_used'] is False
            if meta['rho']:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, c['task'], meta['family'], meta['rho'], c['task_id'])
            else:
                X, y, Q = X0, y0, Q0
            with np.load(path) as data, np.load(meta['parent_bank']) as old:
                np.testing.assert_array_equal(data['Base'], old['Base'])
                np.testing.assert_array_equal(data['Canonical_native8'], old['Canonical_native8'])
                np.testing.assert_array_equal(data['members'][:3], old['members'][:3])
                P = np.concatenate([old['Base'][None], old['members']], axis=0)
                plans, teacher = bank.constructor.contexts(X, y, Q, P, c['task'])
                np.testing.assert_array_equal(teacher, data['Teacher'])
            assert protocol.array_sha256(y) == meta['true_source_labels_sha256']
            assert len(plans) == len(meta['query_crossfit_contexts']) == 2
            actual_calls = 0
            for branch, recorded in zip(plans, meta['query_crossfit_contexts']):
                seen = []
                for p, r in zip(branch, recorded):
                    for name in ('query_rows', 'source_rows', 'pseudo_query_rows'):
                        np.testing.assert_array_equal(p[name], r[name])
                    assert not set(p['query_rows']) & set(p['pseudo_query_rows'])
                    assert protocol.array_sha256(p['X']) == r['context_X_sha256']
                    assert protocol.array_sha256(p['y']) == r['context_y_sha256']
                    assert protocol.array_sha256(p['pseudo_labels']) == r['pseudo_label_sha256']
                    assert len(p['y']) == r['context_rows'] == len(y) <= 256
                    np.testing.assert_array_equal(p['y'][:len(p['source_rows'])], y[p['source_rows']])
                    seen.extend(p['query_rows'].tolist())
                    actual_calls += int(not p['no_pseudo_fallback'])
                    contexts += 1
                assert sorted(seen) == list(range(len(Q)))
            assert actual_calls == meta['new_TFM_calls']
            count += 1
    assert count == 1327
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count, 'query_fold_contexts_replayed': contexts,
        'query_outcome_files_opened': False, 'teacher_recomputed_from_raw_predictions_only': True,
        'each_query_absent_from_own_pseudo_context': True, 'context_row_cap_256': True,
        'no_new_true_labels': True, 'source_class_support_and_original_label_values_preserved': True,
        'Base_and_first_three_experts_and_Native8_control_byte_equal': True,
        'explicit_transductive_constructor_extension': True, 'Markdown_bytes': 0}
    atomic_json(bank.OUT/'independent_constructor_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
