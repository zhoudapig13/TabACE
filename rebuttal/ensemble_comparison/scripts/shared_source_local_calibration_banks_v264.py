"""Replace two repair outputs by SOURCE-trained combinations of existing outputs.

No new query inference, context, or true label row. Reuses honest SourceProxyCV
forecasts (v234), whose source-fold models exclude their scored source labels.
Source labels fit only small fixed-regularization ensemble heads.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_feature_forest_banks_v227 as parent
import source_proxy_repair_features_v234 as source_cv
import source_local_fusion_v263 as local
from prepare_data import RUN, atomic_json, protocol

source = parent.source
OUT = RUN/'results/shared_source_local_calibration_v264'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'independent_membership_audit.json').read_text())['status'] == 'PASS'
    assert json.loads((source_cv.OUT/'independent_source_CV_audit.json').read_text())['status'] == 'PASS'
    plan = {'status': 'SOURCE_LOCAL_FUSION_SUPERVISION_ADAPTATION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'source_rows_cap': 256, 'predictor_outputs_including_Base': 8,
        'first_five_parent_alternatives_unchanged': True,
        'local_experts': ['Base1', 'forest_repair_mild', 'forest_repair_full'],
        'source_meta_labels': 'same existing source training labels; no extra label rows',
        'local_fit_inputs': 'source-OOF expert forecasts on common full-repair source validation proxy',
        'source_probe_models_exclude_scored_source_labels': True,
        'local_heads': {'convex': {'regularization': .01}, 'ridge_or_loglinear': {'regularization': .1}},
        'minimum_scored_source_rows': 16, 'insufficient_support_fallback': 'original Base',
        'local_hyperparameters_tuned_on_task_queries': False,
        'task_query_labels_opened': False, 'new_TFM_inference': False,
        'two_outputs_are_calibrated_combinations_not_additional_TFM_forwards': True,
        'maximum_query_TFM_views': 15, 'source_proxy_CV_TFM_views': 9,
        'additional_parent_source_CV_views': {'classification': 0, 'regression': 3},
        'maximum_all_views': {'classification': 24, 'regression': 27},
        'compute_matched_to_Native1_Base': False,
        'Base_and_benchmark_and_source_pool_and_query_cohort_unchanged': True,
        'source_label_values_unchanged': True, 'extra_true_labeled_rows': 0,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'script_sha256': source.sha(__file__), 'local_head_sha256': source.sha(local.__file__),
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'source_CV_plan_sha256': source.sha(source_cv.OUT/'plan.json'),
        'source_CV_audit_sha256': source.sha(source_cv.OUT/'independent_source_CV_audit.json'),
        'expected_conditions': 1327, 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(c):
    assert source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        y = data['y_source']
    count = 0
    for old in sorted((parent.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
        meta = json.loads(old.with_suffix('.json').read_text())
        assert source.sha(old) == meta['output_sha256']
        cvpath = source_cv.OUT/'features'/c['task']/str(c['task_id'])/old.name
        cvmeta = json.loads(cvpath.with_suffix('.json').read_text())
        assert source.sha(cvpath) == cvmeta['output_sha256']
        assert cvmeta['feature_bank_sha256'] == meta['output_sha256']
        assert protocol.array_sha256(y) == cvmeta['source_labels_sha256']
        with np.load(cvpath) as data:
            source_prediction, mask = data['predictions'], data['scored']
        with np.load(old) as data:
            base, members = data['Base'], data['members'].copy()
            raw_repair = members[5:].copy()
            P = np.concatenate([base[None], raw_repair], axis=0)
            mu, sigma = float(data['source_mean']), max(float(data['source_scale']), 1e-12)
            policies = []
            for i, kind in enumerate(('convex', 'stack')):
                policy = local.fit(source_prediction[:, mask], y[mask], c['task'], mu, sigma, kind)
                members[5+i] = local.predict(P, policy, c['task'], mu, sigma)
                policies.append(policy)
            dest = OUT/'banks'/c['task']/str(c['task_id'])/old.name
            source.npwrite(dest, Base=base, members=members, UncalibratedRepair_predictions=raw_repair,
                Canonical_native8=data['Canonical_native8'], FrozenFusion=data['FrozenFusion'],
                source_mean=mu, source_scale=sigma)
        atomic_json(dest.with_suffix('.json'), {key: meta[key] for key in ('task', 'task_id', 'data_id', 'fold', 'family', 'rho', 'original_input_sha256')} |
            {'output_sha256': source.sha(dest), 'parent_bank': str(old), 'parent_bank_sha256': meta['output_sha256'],
                'source_CV_file': str(cvpath), 'source_CV_sha256': cvmeta['output_sha256'],
                'source_label_sha256': protocol.array_sha256(y), 'local_models': policies,
                'scored_source_rows': int(mask.sum()), 'source_labels_used_for_local_fusion_fit': True,
                'task_query_labels_opened': False, 'no_new_query_TFM_forwards': True})
        count += 1
    return {'task': c['task'], 'task_id': c['task_id'], 'conditions': count}


def main():
    freeze()
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(worker, c) for c in cases]):
            results.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'cases': results})
    assert sum(r['conditions'] for r in results) == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': 1327, 'cases': results})
    print('SOURCE_LOCAL_CALIBRATION_BANK_COMPLETE', 1327, flush=True)


if __name__ == '__main__':
    main()
