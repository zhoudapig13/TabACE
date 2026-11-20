"""Fixed five-rule generic context expansion from already cached predictions.

This is a full constructor-bank adaptation, not the original Shared bank.
No benchmark changes, outcome-based dataset filtering, or new TFM inference.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

import unseen_shared_robustness_v117 as source
from prepare_data import RUN, atomic_json

OUT = RUN/'results/shared_generic_pool_v163'
PARENTS = {
    'transport': RUN/'results/shared_robust_context_adaptation_v122',
    'smooth': RUN/'results/shared_label_smoothing_v124',
    'hampel': RUN/'results/shared_conditional_hampel_v126',
    'crossfit': RUN/'results/shared_source_crossfit_v135',
    'graphical': RUN/'results/shared_graphical_v147',
}
RULES = {
    'classification': [('marginal_transport', 'transport', 1),
        ('covariance_transport', 'transport', 4), ('source_label_local_smooth', 'smooth', 1),
        ('target_graphical_repair', 'graphical', 4), ('source_CV_keep75', 'crossfit', 1)],
    'regression': [('covariance_transport', 'transport', 4),
        ('source_label_local_smooth', 'smooth', 3), ('source_label_Hampel', 'hampel', 3),
        ('source_CV_keep75', 'crossfit', 3), ('source_CV_keep90', 'crossfit', 4)],
}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    for parent in PARENTS.values():
        assert json.loads((parent/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'FULL_GENERIC_CONSTRUCTOR_BANK_ADAPTATION', 'seed': 0,
        'context_cap': 256, 'alternative_contexts': 5, 'estimators_per_context': 1,
        'all_deployment_TFM_predictions_including_Base': 6,
        'Base_and_benchmark_and_query_cohort_unchanged': True,
        'context_rules_by_task': {task: [list(v) for v in rules] for task, rules in RULES.items()},
        'rules_fixed_before_new_fusion_calibration': True,
        'no_family_or_strength_specific_context_selection': True,
        'not_original_Shared_constructor_bank': True,
        'source_labels_used_by_some_adapted_rules': True,
        'source_CV_contexts_require_extra_training_scoring': 'up to3 source-CV TFM calls',
        'no_extra_true_query_labels_or_clean_source_features': True,
        'new_TFM_inference_in_this_stage': False, 'parent_plan_sha256': {
            key: source.sha(path/'plan.json') for key, path in PARENTS.items()},
        'script_sha256': source.sha(__file__), 'expected_conditions': 1327,
        'previously_evaluated_development_cohort': True, 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(c):
    task, tid, n = c['task'], c['task_id'], 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
        dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
        if dest.with_suffix('.json').exists():
            assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
            n += 1
            continue
        original = source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
        with np.load(original) as z:
            base, mu, sigma = z['Base'], float(z['source_mean']), float(z['source_scale'])
        arrays, metadata, members, references = {}, {}, [], []
        rows = []
        for rule, parent, index in RULES[task]:
            if parent not in arrays:
                path = PARENTS[parent]/'banks'/task/str(tid)/dest.name
                meta = json.loads(path.with_suffix('.json').read_text())
                assert source.sha(path) == meta['output_sha256']
                assert meta['original_input_sha256'] == c['input_sha256']
                assert meta['family'] == family and meta['rho'] == rho
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['Base'], base)
                    assert np.isclose(float(z['source_mean']), mu)
                    assert np.isclose(float(z['source_scale']), sigma)
                    arrays[parent] = z['members']
                metadata[parent] = meta
            path = PARENTS[parent]/'banks'/task/str(tid)/dest.name
            members.append(arrays[parent][index])
            references.append({'rule': rule, 'parent_bank': str(path),
                'parent_sha256': metadata[parent]['output_sha256'], 'member_index': index})
            if parent == 'crossfit':
                fraction_index = 0 if rule.endswith('75') else 1
                rows.append(len(metadata[parent]['selected_source_rows'][fraction_index]))
            else:
                rows.append(c['source_rows'])
        members = np.stack(members)
        assert len(members) == 5 and max(rows) <= 256 and np.isfinite(members).all()
        policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
        frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
            'mu': mu, 'sigma': sigma}, policy['Shared'])
        source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
            source_mean=mu, source_scale=sigma)
        atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
            {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
            'original_bank': str(original), 'original_bank_sha256': source.sha(original),
            'original_input_sha256': c['input_sha256'], 'member_provenance': references,
            'context_rows': rows, 'outcome_query_labels_opened': False,
            'context_rules_selected_by_task_only_not_by_family': True})
        n += 1
    return {'task': task, 'task_id': tid, 'conditions': n}


def launch(workers):
    freeze()
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker, c) for c in cases]):
            results.append(future.result())
            if len(results)%10 == 0:
                print('GENERIC_POOL_PROGRESS', len(results), len(cases), flush=True)
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'cases': results})
    assert sum(r['conditions'] for r in results) == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'cases': results})
    print('GENERIC_POOL_COMPLETE', len(results), 1327, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    launch(parser.parse_args().workers)
