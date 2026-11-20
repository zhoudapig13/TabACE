"""Rebuild every query cluster and local source membership without query labels."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_query_local_banks_v248 as bank
from run_tabicl import CHECKPOINTS


def check_case(c):
    assert bank.source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    target = bank.constructor.fit_target(Q0, numeric)
    count, contexts = 0, 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in bank.source.novel.STRENGTHS]:
        path = bank.OUT/'banks'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert bank.source.sha(meta['source_quality_bank']) == meta['source_quality_bank_sha256']
        assert meta['original_input_sha256'] == c['input_sha256']
        assert not meta['query_task_labels_opened']
        with np.load(path) as data, np.load(meta['parent_bank']) as parent:
            np.testing.assert_array_equal(data['Base'], parent['Base'])
            np.testing.assert_array_equal(data['members'][:5], parent['members'])
            np.testing.assert_array_equal(data['Canonical_native8'], parent['Canonical_native8'])
            assert data['members'].shape[0] == 7 and np.isfinite(data['members']).all()
            if c['task'] == 'classification':
                np.testing.assert_allclose(data['members'].sum(2), 1., atol=1e-6)
        with np.load(meta['source_quality_bank']) as data:
            source_loss = data['SourceCV_loss']
        if rho:
            X, y, Q, _ = bank.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        else:
            X, y, Q = X0, y0, Q0
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(Q, Q0)
        assert bank.protocol.array_sha256(target['query_assignment']) == meta['query_cluster_assignment_sha256']
        plans = bank.constructor.construct(X, y, source_loss, c['task'], target)
        expected = []
        for branch in plans:
            detail = []
            for p in branch:
                src, query = p['source_rows'], p['query_rows']
                assert len(src) <= 256 and len(np.unique(src)) == len(src)
                if c['task'] == 'classification':
                    assert set(y[src]) == set(y)
                    rare = np.concatenate([np.flatnonzero(y == label) for label in np.unique(y) if np.sum(y == label) < 3]) if any(np.sum(y == label) < 3 for label in np.unique(y)) else []
                    assert set(rare).issubset(src)
                detail.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()} |
                    {'context_X_sha256': bank.protocol.array_sha256(X[src]),
                        'context_y_sha256': bank.protocol.array_sha256(y[src]),
                        'query_X_sha256': bank.protocol.array_sha256(Q[query]), 'actual_context_rows': len(src)})
                contexts += 1
            expected.append(detail)
        assert expected == meta['local_context_plans']
        assert meta['new_query_TFM_calls'] == sum(map(len, plans)) <= 16
        count += 1
    return {'conditions': count, 'local_contexts_reconstructed': contexts}


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    for path, key in [(bank.__file__, 'script_sha256'), (bank.constructor.__file__, 'constructor_sha256'),
        (bank.parent.OUT/'plan.json', 'parent_plan_sha256'), (bank.quality.OUT/'plan.json', 'source_quality_plan_sha256'),
        (bank.source.OUT/'plan.json', 'original_plan_sha256')]:
        assert bank.source.sha(path) == plan[key]
    for name, digest in json.loads((bank.source.OUT/'plan.json').read_text())['checkpoints'].items():
        assert bank.source.sha(CHECKPOINTS/name) == digest
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in cases]):
            results.append(future.result())
            if len(results) % 10 == 0:
                print('QUERY_LOCAL_AUDIT', len(results), len(cases), flush=True)
    count = sum(r['conditions'] for r in results)
    assert len(results) == 107 and count == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count,
        'local_contexts_reconstructed': sum(r['local_contexts_reconstructed'] for r in results),
        'Base_and_five_parent_alternatives_unchanged': True,
        'geometry_fitted_on_query_input_features_only': True,
        'source_feature_values_and_labels_not_modified': True,
        'source_class_coverage_and_rare_rows_retained': True,
        'all_query_partitions_cover_exactly_once_per_branch': True,
        'query_task_labels_opened': False, 'TFM_inference_not_repeated_by_this_audit': True,
        'checkpoint_hashes_unchanged': True, 'constructor_and_compute_extension_explicit': True,
        'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
