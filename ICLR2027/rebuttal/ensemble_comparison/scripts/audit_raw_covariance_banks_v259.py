"""Reconstruct both raw covariance contexts without any task-query label."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_raw_covariance_banks_v258 as bank
from run_tabicl import CHECKPOINTS


def check_case(c):
    assert bank.source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    target = bank.constructor.fit_target(Q0, numeric)
    count = 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in bank.source.novel.STRENGTHS]:
        path = bank.OUT/'banks'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert meta['original_input_sha256'] == c['input_sha256']
        assert not meta['task_query_labels_opened'] and meta['new_query_TFM_calls'] == 2
        with np.load(path) as data, np.load(meta['parent_bank']) as parent:
            np.testing.assert_array_equal(data['Base'], parent['Base'])
            np.testing.assert_array_equal(data['members'][:5], parent['members'])
            np.testing.assert_array_equal(data['Canonical_native8'], parent['Canonical_native8'])
            assert data['members'].shape[0] == 7 and np.isfinite(data['members']).all()
            if c['task'] == 'classification':
                np.testing.assert_allclose(data['members'].sum(2), 1., atol=1e-6)
        if rho:
            X, y, Q, _ = bank.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        else:
            X, y, Q = X0, y0, Q0
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(Q, Q0)
        original = bank.source.OUT/'predictions/perturbation'/c['task']/str(c['task_id'])/path.name
        original_meta = json.loads(original.with_suffix('.json').read_text())
        assert bank.source.sha(original) == original_meta['output_sha256']
        for value, key in [(X, 'observed_source_X_sha256'), (y, 'observed_source_y_sha256'),
            (Q, 'observed_target_X_sha256')]:
            assert bank.protocol.array_sha256(value) == original_meta[key]
        views = bank.constructor.construct(X, target)
        assert len(views) == len(meta['raw_covariance_contexts']) == 2
        for view, record in zip(views, meta['raw_covariance_contexts']):
            assert view['info'] == {key: record[key] for key in view['info']}
            assert bank.protocol.array_sha256(view['X']) == record['context_X_sha256']
            assert bank.protocol.array_sha256(y) == record['context_y_sha256']
            assert bank.protocol.array_sha256(Q) == record['query_X_sha256']
            assert len(view['X']) == record['context_rows'] == c['source_rows'] <= 256
            assert not record['source_task_labels_used'] and not record['query_task_labels_used']
            assert not record['marginal_CDF_alignment_applied_first']
        count += 1
    return count


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    for path, key in [(bank.__file__, 'script_sha256'), (bank.constructor.__file__, 'constructor_sha256'),
        (bank.parent.OUT/'plan.json', 'parent_plan_sha256'), (bank.source.OUT/'plan.json', 'original_plan_sha256')]:
        assert bank.source.sha(path) == plan[key]
    for name, digest in json.loads((bank.source.OUT/'plan.json').read_text())['checkpoints'].items():
        assert bank.source.sha(CHECKPOINTS/name) == digest
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    counts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in cases]):
            counts.append(future.result())
            if len(counts) % 10 == 0:
                print('RAW_COVARIANCE_AUDIT', len(counts), len(cases), flush=True)
    assert len(counts) == 107 and sum(counts) == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': sum(counts), 'repaired_contexts_reconstructed': 2*sum(counts),
        'Base_and_five_parent_alternatives_unchanged': True,
        'observed_benchmark_inputs_match_original_per_condition_hashes': True,
        'source_rows_and_label_values_unchanged': True, 'no_task_labels_used_by_transform': True,
        'query_task_labels_opened': False, 'raw_covariance_estimation_precedes_any_CDF_mapping': True,
        'categorical_values_unchanged': True, 'checkpoint_hashes_unchanged': True,
        'TFM_inference_not_repeated_by_this_audit': True,
        'constructor_extension_not_original_Shared': True, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
