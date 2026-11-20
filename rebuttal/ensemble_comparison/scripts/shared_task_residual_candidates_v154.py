"""Source-only per-dataset residual adapters on the original Shared bank."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

import unseen_shared_robustness_v117 as source
import shared_source_crossfit_banks_v135 as source_cv
import source_task_residual_v153 as adapter
from prepare_data import RUN, atomic_json, protocol

OUT = RUN/'results/shared_task_residual_v154'


def freeze():
    assert json.loads((source_cv.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'PER_DATASET_SOURCE_LABEL_RESIDUAL_ADAPTATION', 'seed': 0,
        'context_cap': 256, 'original_Base_and_five_Shared_predictions_unchanged': True,
        'benchmark_and_query_cohort_unchanged': True, 'TFM_weights_updated': False,
        'new_per_dataset_fitted_head': 'kernel ridge on source-OOF residuals',
        'uses_own_observed_source_labels_for_task_adaptation': True,
        'extra_true_query_labels_or_clean_source_features_used': False,
        'source_OOF_scoring_additional_TFM_calls': 'up to3; cached from v135',
        'residual_training_labels': 'observed source outcomes minus source-held-out model predictions',
        'feature_encoder_and_kernel_fitted_on_source_only': True,
        'query_features_used_for_prediction_only_by_residual_adapter': True,
        'candidate_specifications': adapter.SPECIFICATIONS,
        'global_regularization_and_gain_selection': 'subsequent other-dataset nested calibration only',
        'new_TFM_inference_in_this_stage': False, 'expected_conditions': 1327,
        'source_cv_plan_sha256': source.sha(source_cv.OUT/'plan.json'),
        'script_sha256': source.sha(__file__), 'adapter_helper_sha256': source.sha(adapter.__file__),
        'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def worker(c):
    assert source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as z:
        assert 'y_target' not in z.files
        X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        assert set(z['source_indices']).isdisjoint(z['target_indices'])
    n = 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
        dest = OUT/'corrections'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        if dest.with_suffix('.json').exists():
            assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
            n += 1
            continue
        parent = source_cv.OUT/'banks'/c['task']/str(c['task_id'])/dest.name
        parent_meta = json.loads(parent.with_suffix('.json').read_text())
        assert source.sha(parent) == parent_meta['output_sha256']
        with np.load(parent) as z:
            cv = z['SourceCV_prediction']
        if family == 'clean':
            X, y, Q = X0.copy(), y0.copy(), Q0.copy()
        else:
            X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        assert protocol.array_sha256(y) == parent_meta['source_y_sha256']
        correction, details = adapter.fit_predict(X, y, Q, numeric, cv, c['task'])
        source.npwrite(dest, task_corrections=correction)
        original = source.OUT/'predictions/perturbation'/c['task']/str(c['task_id'])/dest.name
        atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
            {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
            'source_cv_bank': str(parent), 'source_cv_bank_sha256': source.sha(parent),
            'original_prediction_bank': str(original), 'original_prediction_bank_sha256': source.sha(original),
            'input_sha256': c['input_sha256'], 'query_labels_opened': False,
            'own_observed_source_labels_used_for_adapter_fitting': True,
            'source_feature_values_are_observed_corrupted_values': True,
            'correction_shape': list(correction.shape), 'adapter_details': details})
        n += 1
    return {'task': c['task'], 'task_id': c['task_id'], 'conditions': n}


def launch(workers):
    freeze()
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker, c) for c in cases]):
            results.append(future.result())
            if len(results)%10 == 0:
                print('TASK_RESIDUAL_PROGRESS', len(results), len(cases), flush=True)
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'cases': results})
    assert sum(c['conditions'] for c in results) == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'cases': results})
    print('TASK_RESIDUAL_COMPLETE', len(results), 1327, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    launch(p.parse_args().workers)
