"""Nested dataset-OOF adaptation of an unchanged Shared context bank.

Two separately reported settings: LOMO excludes the tested new mechanism
from all fitting and inner validation; pooled sees all three new mechanisms
on other datasets. Neither fits this outer dataset's query labels. Previously
scored cohorts remain exploratory development data, not fresh confirmation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch

import adapt_shared_heads_v119 as heads
import unseen_shared_robustness_v117 as original
from prepare_data import RUN, ROOT, atomic_json
from run_tabicl import metrics

OUT = RUN / 'results/unseen_shared_adaptation_v119'
FAMILIES = ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        'status': 'FROZEN_BEFORE_FUSION_HEAD_FITTING', 'seed': 0,
        'base_bank': str(original.OUT), 'frozen_Base_and_benchmark_unchanged': True,
        'no_new_context_or_TFM_training': True, 'extra_target_labels': 0,
        'same_five_alternative_contexts_and_256_row_cap': True,
        'target_fit_labels_from_other_datasets_only': True,
        'data_id_outer_folds': 5, 'inner_dataset_validation_folds': 4,
        'calibration_modes': {
            'lomo': 'exclude the evaluated new family in both calibration fitting and inner selection',
            'pooled': 'include all three new families on other datasets; explicitly known-family adaptation',
        },
        'classification_loss': 'half cross entropy plus half classwise pair-ranking for adaptive/convex heads; CE for loglinear head',
        'regression_loss': 'macro dataset/family/strength calibration-target-variance normalized squared error',
        'validation_primary': {'classification': 'AUC', 'regression': 'R2'},
        'class_candidate_families': ['Base', 'Mean6', 'Median6', 'convex', 'adaptive probability weights', 'loglinear'],
        'regression_candidate_families': ['Base', 'Mean6', 'Median6', 'ridge', 'adaptive weights', 'median winsorized adaptive weights'],
        'regularizations': heads.REGS, 'candidates_each_selection': 12,
        'clean_is_one_equally_weighted_family': True,
        'new_head_is_not_original_paper_fusion': True,
        'classification_gain_minimum_not_enforced_in_fit': True,
        'significance_not_a_hyperparameter_selection_objective': True,
        'all_evaluation_cohorts_previously_scored_development': True,
        'formal_independent_confirmation': False,
        'planned_primary_tests': 'two modes times three families times two tasks, Holm12',
        'primary_success': 'positive mean and95CI lower>0 and Holm12 p<0.05 for each of six endpoints within one mode',
        'secondary_direction_check': 'Acc increases and nRMSE decreases; all metrics reported',
        'all_candidates_and_all_null_or_negative_outcomes_preserved': True,
        'script_sha256': original.sha(__file__), 'head_sha256': original.sha(heads.__file__),
        'upstream_plan_sha256': original.sha(original.OUT / 'plan.json'),
        'Markdown_written': False,
    }
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(plan))
    else:
        atomic_json(p, plan)


def load_cells(task, exclude_fold=None):
    cases = json.loads((original.OUT / 'case_manifest.json').read_text())
    out = []
    for case in cases:
        if case['task'] != task or case['track'] != 'perturbation' or case['fold'] == exclude_fold:
            continue
        # The outer-fold filter above executes BEFORE opening any truth file.
        with np.load(case['truth_file']) as z:
            y = z['y_target']
        for path in sorted((original.OUT / 'predictions/perturbation' / task / str(case['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as z:
                c = {'P': np.concatenate([z['Base'][None], z['members']], axis=0),
                    'mu': float(z['source_mean']), 'sigma': max(float(z['source_scale']), 1e-12)}
            out.append(c | {'task': task, 'data_id': case['data_id'], 'task_id': case['task_id'],
                'fold': case['fold'], 'family': meta['family'], 'rho': meta['rho'], 'y': y,
                'source_file': str(path), 'source_file_sha256': meta['output_sha256']})
    return out


def choose(source):
    folds = sorted({c['fold'] for c in source})
    assert len(folds) == 4
    trials = []
    for spec in heads.candidates(source[0]['task']):
        inner = []
        for fold in folds:
            train = [c for c in source if c['fold'] != fold]
            validation = [c for c in source if c['fold'] == fold]
            train_ids, val_ids = {c['data_id'] for c in train}, {c['data_id'] for c in validation}
            assert train_ids.isdisjoint(val_ids)
            model = heads.fit(train, spec)
            score = heads.score(validation, model)
            inner.append({'inner_fold': fold, 'model': model, 'score': score,
                'training_data_ids': sorted(train_ids), 'validation_data_ids': sorted(val_ids)})
        denominator = sum(x['score']['datasets'] for x in inner)
        value = {key: sum(x['score']['datasets']*x['score'][key] for x in inner)/denominator
            for key in ('primary', 'secondary')}
        trials.append({'specification': spec, 'inner': inner, 'score': value})
    best = max(trials, key=lambda x: (x['score']['primary'], x['score']['secondary']))
    return heads.fit(source, best['specification']), best, trials


def worker(task, fold):
    torch.set_num_threads(1)
    source = load_cells(task, exclude_fold=fold)
    outer_ids = {c['data_id'] for c in source}
    scenarios = [('pooled', 'all')] + [('lomo', family) for family in FAMILIES]
    for mode, family in scenarios:
        destination = OUT / 'policies' / task / str(fold) / mode / f'{family}.json'
        if destination.exists():
            continue
        values = [c for c in source if mode == 'pooled' or c['family'] != family]
        if mode == 'lomo':
            assert all(c['family'] != family for c in values)
        print('ADAPT_START', task, fold, mode, family, len(values), flush=True)
        model, best, trials = choose(values)
        atomic_json(destination, {'task': task, 'outer_fold': fold, 'mode': mode,
            'excluded_new_family': family if mode == 'lomo' else None,
            'target_new_family_seen_in_calibration': mode == 'pooled',
            'model': model, 'selected_specification': best['specification'],
            'selection_score': best['score'], 'trials': trials,
            'training_data_ids': sorted(outer_ids),
            'training_families': sorted({c['family'] for c in values}),
            'own_outer_query_labels_opened_for_fitting': False,
            'raw_contexts_or_Base_or_operator_changed': False})
        print('ADAPT_POLICY', task, fold, mode, family, best['specification'], best['score'], flush=True)
    return {'task': task, 'fold': fold}


def train():
    freeze()
    results = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(worker, task, fold) for task in ('classification', 'regression') for fold in range(5)]
        for future in as_completed(futures):
            results.append(future.result())
            print('ADAPT_COMPLETED', results[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'completed_jobs': results})
    paths = list((OUT / 'policies').glob('*/*/*/*.json'))
    assert len(paths) == 40
    atomic_json(OUT / 'policy_seal.json', {str(p.relative_to(OUT)): original.sha(p) for p in paths})
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE', 'completed_jobs': results})


def report():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert original.sha(__file__) == plan['script_sha256'] and original.sha(heads.__file__) == plan['head_sha256']
    assert json.loads((OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    seal = json.loads((OUT / 'policy_seal.json').read_text())
    assert all(original.sha(OUT / p) == h for p, h in seal.items())
    rows = []
    for task in ('classification', 'regression'):
        for c in load_cells(task):
            for mode in ('pooled', 'lomo'):
                families = (FAMILIES if c['family'] == 'clean' and mode == 'lomo' else (c['family'],))
                for family in families:
                    if family != 'clean' and family not in FAMILIES:
                        continue
                    key = 'all' if mode == 'pooled' else family
                    p = OUT / 'policies' / task / str(c['fold']) / mode / f'{key}.json'
                    saved = json.loads(p.read_text())
                    assert c['data_id'] not in saved['training_data_ids']
                    if mode == 'lomo':
                        assert family not in saved['training_families']
                    predictions = {'Base': c['P'][0], 'Shared_adapted': heads.predict(c, saved['model'])}
                    for method, q in predictions.items():
                        value = metrics(c['y'], q, task, max(float(c['y'].std()), 1e-12))
                        rows.append({'task': task, 'data_id': c['data_id'], 'task_id': c['task_id'],
                            'fold': c['fold'], 'mode': mode, 'family': family if c['family'] != 'clean' else 'clean',
                            'lomo_clean_policy_family': family if c['family'] == 'clean' and mode == 'lomo' else '',
                            'rho': c['rho'], 'method': method, **value})
                    dest = OUT / 'predictions' / task / mode / str(c['task_id']) / f'{family}_{c["rho"]}.npz'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest, prediction=predictions['Shared_adapted'])
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'detail.csv', index=False)
    tables, tests = [], []
    for (task, mode, family), group in detail[detail.rho > 0].groupby(['task', 'mode', 'family']):
        cols = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        values = group.groupby(['data_id', 'method'])[cols].mean()
        for method in ('Base', 'Shared_adapted'):
            v = values.xs(method, level='method')
            tables.append({'task': task, 'mode': mode, 'family': family, 'method': method,
                'datasets': len(v), **{k: float(v[k].mean()) for k in cols}})
        primary = 'auc' if task == 'classification' else 'r2'
        delta = values.xs('Shared_adapted', level='method')[primary]-values.xs('Base', level='method')[primary]
        tests.append({'task': task, 'mode': mode, 'family': family, 'metric': primary,
            **original.paired_statistics(delta)})
    stats = pd.DataFrame(tests)
    assert len(stats) == 12
    last = 0.
    for rank, i in enumerate(sorted(stats.index, key=lambda j: stats.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (12-rank)*stats.loc[i, 'p_two_sided']))
        stats.loc[i, 'p_holm_12'] = last
    table = pd.DataFrame(tables)
    table.to_csv(OUT / 'family_averages.csv', index=False)
    stats.to_csv(OUT / 'paired_statistics.csv', index=False)
    success = {mode: bool(((v.gain > 0) & (v.ci_low > 0) & (v.p_holm_12 < .05)).all())
        for mode, v in stats.groupby('mode')}
    old = pd.read_csv(original.OUT / 'family_averages.csv')
    for row in table[table.method == 'Base'].to_dict('records'):
        before = old[(old.task == row['task']) & (old.family == row['family']) & (old.method == 'Base')].iloc[0]
        for key in (('accuracy', 'auc') if row['task'] == 'classification' else ('r2', 'nrmse')):
            assert np.isclose(before[key], row[key], atol=1e-12)
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS', 'Base_metrics_identical_to_v117': True,
        'policy_files_unchanged_after_seal': True, 'own_dataset_not_in_new_head_fit': True,
        'lomo_excludes_evaluated_mechanism': True, 'pooled_explicitly_sees_new_mechanisms': True,
        'candidate_selection_uses_inner_other_datasets_only': True,
        'all_six_primary_endpoints_positive_significant_by_mode': success,
        'exploratory_previously_evaluated_development_cohort': True,
        'formal_independent_confirmation': False, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Improve weak or negative frozen-Shared perturbation performance without changing Base or contexts',
        'Setting': plan, 'Results': json.loads(table.to_json(orient='records', double_precision=15)),
        'Analysis': {'all_six_primary_endpoints_significant': success,
            'original_frozen_Shared_remains_available': True,
            'new_shared_fusion_heads_are_adaptations': True},
        'Next Steps': 'Inspect failures and require separate held-out-row confirmation before confirmatory claims; never modify benchmark to force a ranking.',
    })
    print(table.to_string(index=False), flush=True)
    print(stats.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['train', 'report'])
    a = p.parse_args()
    train() if a.phase == 'train' else report()
