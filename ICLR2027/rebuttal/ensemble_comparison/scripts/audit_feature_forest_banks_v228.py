"""Refit all query-feature pretext models and reconstruct every repaired view.

No true query task label is loaded. This verifies constructor inputs and parent
prediction provenance, not a second GPU inference or a generalization claim.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_feature_forest_banks_v227 as bank
from run_tabicl import CHECKPOINTS


def check_case(c):
    assert bank.source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    fitted = bank.constructor.fit_target(Q0, numeric)
    info = fitted['info'] | {'query_X_sha256': bank.protocol.array_sha256(Q0),
        'original_input_sha256': c['input_sha256'], 'query_task_labels_used': False}
    model_path = bank.OUT/'feature_models'/c['task']/f'{c["task_id"]}.json'
    assert info == json.loads(model_path.read_text())
    all_valid = []
    for fold in info['feature_CV_splits']:
        a, b = set(fold['train']), set(fold['valid'])
        assert a.isdisjoint(b) and a|b == set(range(len(Q0)))
        all_valid += fold['valid']
    assert sorted(all_valid) == list(range(len(Q0)))
    for r in info['responses_considered']:
        if 'skip' in r:
            continue
        assert not set(r['input_encoded_columns']) & set(r['own_encoding_columns'])
        assert r['used'] == (r['reliability'] >= bank.constructor.MIN_RELIABILITY)
    count = 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in bank.source.novel.STRENGTHS]:
        path = bank.OUT/'banks'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert bank.source.sha(meta['feature_model_info']) == meta['feature_model_info_sha256']
        assert meta['feature_model_info'] == str(model_path)
        assert meta['original_input_sha256'] == c['input_sha256']
        assert meta['new_TFM_calls'] == 2 and not meta['query_task_labels_opened']
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
        contexts = bank.constructor.repair_source(X, fitted)
        assert len(contexts) == len(meta['feature_repair_contexts']) == 2
        for context, record in zip(contexts, meta['feature_repair_contexts']):
            assert context['info'] == {k: record[k] for k in context['info']}
            assert bank.protocol.array_sha256(context['X']) == record['context_X_sha256']
            assert bank.protocol.array_sha256(y) == record['context_y_sha256']
            assert bank.protocol.array_sha256(Q) == record['query_X_sha256']
            assert len(context['X']) == len(y) == c['source_rows'] <= 256
            for j in fitted['categorical']:
                allowed = np.asarray(np.r_[X[:, j], Q[:, j]], dtype=np.float32)
                assert np.isin(context['X'][:, j], allowed).all()
        count += 1
    return {'task': c['task'], 'task_id': c['task_id'], 'conditions': count,
        'feature_responses_considered': len(info['responses_considered']),
        'feature_models_used': len(fitted['models'])}


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    for path, key in [(bank.__file__, 'script_sha256'), (bank.constructor.__file__, 'constructor_sha256'),
        (bank.parent.OUT/'plan.json', 'parent_plan_sha256'), (bank.source.OUT/'plan.json', 'original_plan_sha256')]:
        assert bank.source.sha(path) == plan[key]
    assert bank.sklearn.__version__ == plan['sklearn_version']
    for name, digest in json.loads((bank.source.OUT/'plan.json').read_text())['checkpoints'].items():
        assert bank.source.sha(CHECKPOINTS/name) == digest
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in cases]):
            results.append(future.result())
            if len(results) % 10 == 0:
                print('FEATURE_REPAIR_AUDIT', len(results), len(cases), flush=True)
    conditions = sum(r['conditions'] for r in results)
    assert len(results) == 107 and conditions == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'cases': len(results), 'conditions': conditions,
        'pretext_feature_models_refitted_on_all_cases': True, 'repaired_contexts_reconstructed': 2*conditions,
        'query_task_labels_opened': False, 'own_feature_encoding_group_excluded': True,
        'feature_CV_partitions_checked': True, 'all_five_parent_alternatives_and_Base_unchanged': True,
        'source_rows_and_labels_and_query_features_unchanged': True,
        'new_TFM_inference_not_repeated_by_this_audit': True,
        'checkpoint_hashes_unchanged': True, 'constructor_extension_not_original_Shared': True,
        'case_results': results, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'case_results'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
