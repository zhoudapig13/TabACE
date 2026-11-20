"""Acquire source-local fusion features while retaining original contexts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

import unseen_shared_robustness_v117 as source
import shared_source_crossfit_banks_v135 as source_cv
import source_local_support_v141 as support
from prepare_data import RUN, atomic_json, protocol

OUT = RUN/'results/shared_local_support_v142'


def freeze():
    assert json.loads((source_cv.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'NEW_SOURCE_LOCAL_FUSION_FEATURES_ORIGINAL_CONTEXTS_RETAINED',
        'seed': 0, 'context_cap': 256, 'original_five_Shared_context_predictions_retained': True,
        'Base_and_benchmark_unchanged': True, 'source_CV_scoring_calls_added': 'up to3; cached from v135',
        'new_features': 'source-label neighborhood votes and source-OOF residuals;8/32/128 neighbors',
        'uses_existing_observed_source_labels': True, 'uses_unlabeled_query_features': True,
        'uses_query_labels_or_clean_source_features_or_generator_identity': False,
        'no_additional_TFM_inference_in_this_stage': True,
        'new_head_fitting_or_query_scoring_in_this_stage': False,
        'source_CV_bank_plan_sha256': source.sha(source_cv.OUT/'plan.json'),
        'original_plan_sha256': source.sha(source.OUT/'plan.json'),
        'script_sha256': source.sha(__file__), 'feature_helper_sha256': source.sha(support.__file__),
        'expected_conditions': 1327, 'Markdown_written': False}
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
        dest = OUT/'features'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        if dest.with_suffix('.json').exists():
            assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
            n += 1
            continue
        parent = source_cv.OUT/'banks'/c['task']/str(c['task_id'])/dest.name
        meta = json.loads(parent.with_suffix('.json').read_text())
        assert source.sha(parent) == meta['output_sha256']
        with np.load(parent) as z:
            prediction, loss = z['SourceCV_prediction'], z['SourceCV_loss']
        if family == 'clean':
            X, y, Q = X0.copy(), y0.copy(), Q0.copy()
        else:
            X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        assert protocol.array_sha256(y) == meta['source_y_sha256']
        features, details = support.features(X, y, Q, numeric, prediction, loss, c['task'])
        source.npwrite(dest, local_support=features)
        original = source.OUT/'predictions/perturbation'/c['task']/str(c['task_id'])/dest.name
        atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
            {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
            'source_cv_bank': str(parent), 'source_cv_bank_sha256': source.sha(parent),
            'original_prediction_bank': str(original), 'original_prediction_bank_sha256': source.sha(original),
            'input_sha256': c['input_sha256'], 'query_labels_opened': False,
            'source_labels_are_existing_observed_context_labels': True,
            'feature_shape': list(features.shape), 'feature_details': details})
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
                print('LOCAL_SUPPORT_PROGRESS', len(results), len(cases), flush=True)
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'cases': results})
    assert sum(c['conditions'] for c in results) == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'cases': results})
    print('LOCAL_SUPPORT_COMPLETE', len(results), 1327, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    launch(p.parse_args().workers)
