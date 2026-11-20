"""OOF fusion of explicit transductive source-CV pseudo-label repair contexts.

Report frozen old fusion, LOMO adaptation and pooled known-family adaptation
separately. All hypotheses remain exploratory on the already scored cohort;
all earlier failures remain included in cumulative multiple-test reporting.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')

import numpy as np
import pandas as pd
import torch

import unseen_shared_robustness_v117 as original
import shared_group_risk_v127 as calibration
import adapt_shared_heads_v119 as heads
import shared_source_manifold_banks_v138 as bank
from prepare_data import ROOT, atomic_json, RUN
from run_tabicl import metrics

OUT = bank.OUT / 'calibration'
FAMILIES = calibration.FAMILIES


def freeze():
    assert json.loads((bank.OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        'status': 'FROZEN_BEFORE_ADAPTED_CONTEXT_FUSION_FIT', 'seed': 0,
        'explicit_constructor_adaptation': True, 'original_Shared_rule_claimed': False,
        'Base_and_benchmark_unchanged': True, 'five_contexts_and_Native1_each': True,
        'new_context_bank': str(bank.OUT), 'source_crossfit_pseudolabel_manifold_repair': True,
        'source_labels_assumed_reasonably_reliable': True, 'source_label_values_preserved': True, 'bank_plan_sha256': original.sha(bank.OUT / 'plan.json'),
        'modes': {'frozen_fusion': 'old v77 weights on new transductive pseudo-label repair contexts',
            'lomo': 'exclude tested family from fitting and selection; other datasets only',
            'pooled': 'new three families included in other-dataset calibration'},
        'extra_labels_on_evaluated_dataset': 0, 'whole_evaluated_dataset_excluded_from_fit': True,
        'outer_dataset_folds': 5, 'inner_dataset_folds': 4,
        'head_candidates': '12 fixed/stacking/group-risk candidates; original three regularizations',
        'training_risk': 'softmax worst-family excess over Base, temperature0.02, clean risk weight0.1',
        'inner_selection': 'maximize worst-family OOF gain then macro family gain; no test metric used',
        'head_sha256': original.sha(heads.__file__),
        'calibration_selection_implementation_sha256': original.sha(calibration.__file__),
        'script_sha256': original.sha(__file__),
        'new_branch_maps_use_unlabeled_target_features': True,
        'new_branch_maps_use_Base_query_pseudo_labels': True,
        'deployment_is_transductive': True,
        'true_query_labels_enter_new_constructor': False,
        'same_old_three_perturbations_and_four_positive_strengths': True,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'primary_tests': 'three modes times three perturbations times two tasks;18 tests',
        'cumulative_tests': '129 earlier perturbation tests plus18 new reports;Holm147',
        'significant_means': 'positive gain,95CI lower>0, and Holm147<.05 for each of six endpoints',
        'secondary_directions': 'Acc higher and nRMSE lower', 'Markdown_written': False,
    }
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def load_cells(task, exclude_fold=None):
    cases = json.loads((original.OUT / 'case_manifest.json').read_text())
    result = []
    for c in cases:
        if c['track'] != 'perturbation' or c['task'] != task or c['fold'] == exclude_fold:
            continue
        with np.load(c['truth_file']) as z:
            y = z['y_target']
        for path in sorted((bank.OUT / 'banks' / task / str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as z:
                values = {'P': np.concatenate([z['Base'][None], z['members']], axis=0),
                    'frozen': z['FrozenFusion'], 'mu': float(z['source_mean']),
                    'sigma': max(float(z['source_scale']), 1e-12)}
            result.append(values | {'task': task, 'task_id': c['task_id'], 'data_id': c['data_id'],
                'family': meta['family'], 'rho': meta['rho'], 'fold': c['fold'], 'y': y,
                'path': str(path), 'bank_sha256': meta['output_sha256']})
    return result


def worker(task, fold):
    torch.set_num_threads(1)
    cells = load_cells(task, exclude_fold=fold)
    for mode, family in [('pooled', 'all')] + [('lomo', f) for f in FAMILIES]:
        path = OUT / 'policies' / task / str(fold) / mode / f'{family}.json'
        if path.exists():
            continue
        values = [c for c in cells if mode == 'pooled' or c['family'] != family]
        print('ROBUST_CAL_START', task, fold, mode, family, flush=True)
        model, best, trials = calibration.choose(values)
        atomic_json(path, {'task': task, 'outer_fold': fold, 'mode': mode,
            'excluded_family': family if mode == 'lomo' else None,
            'constructor_adaptation': True, 'model': model,
            'selected_specification': best['specification'], 'selection_score': best['score'],
            'trials': trials, 'training_data_ids': sorted({c['data_id'] for c in values}),
            'training_families': sorted({c['family'] for c in values}),
            'own_query_labels_opened_for_fitting': False})
        print('ROBUST_CAL_POLICY', task, fold, mode, family, best['specification'], flush=True)
    return {'task': task, 'fold': fold}


def train():
    freeze()
    results = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(worker, task, fold) for task in ('classification', 'regression') for fold in range(5)]
        for future in as_completed(jobs):
            results.append(future.result())
            print('ROBUST_CAL_COMPLETE', results[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'jobs': results})
    paths = list((OUT / 'policies').glob('*/*/*/*.json'))
    assert len(paths) == 40
    atomic_json(OUT / 'policy_seal.json', {str(p.relative_to(OUT)): original.sha(p) for p in paths})
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE', 'jobs': results})


def report():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert original.sha(__file__) == plan['script_sha256']
    assert json.loads((OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    seal = json.loads((OUT / 'policy_seal.json').read_text())
    assert all(original.sha(OUT / p) == h for p, h in seal.items())
    rows = []
    policy_cache = {}
    for task in ('classification', 'regression'):
        for c in load_cells(task):
            for mode in ('frozen_fusion', 'pooled', 'lomo'):
                if mode == 'frozen_fusion':
                    prediction = c['frozen']
                else:
                    if c['family'] == 'clean' and mode == 'lomo':
                        continue
                    key = 'all' if mode == 'pooled' else c['family']
                    p = OUT / 'policies' / task / str(c['fold']) / mode / f'{key}.json'
                    if p not in policy_cache:
                        policy_cache[p] = json.loads(p.read_text())
                    saved = policy_cache[p]
                    assert c['data_id'] not in saved['training_data_ids']
                    if mode == 'lomo':
                        assert c['family'] not in saved['training_families']
                    prediction = heads.predict(c, saved['model'])
                for method, q in (('Base', c['P'][0]), ('Shared_adapted', prediction)):
                    rows.append({k: c[k] for k in ('task', 'task_id', 'data_id', 'family', 'rho')}
                        | {'mode': mode, 'method': method,
                            **metrics(c['y'], q, task, max(float(c['y'].std()), 1e-12))})
                dest = OUT / 'predictions' / task / mode / str(c['task_id']) / f'{c["family"]}_{c["rho"]}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, prediction=prediction)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'detail.csv', index=False)
    tables, tests, directions = [], [], []
    for (task, mode, family), group in detail[detail.rho > 0].groupby(['task', 'mode', 'family']):
        cols = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        units = group.groupby(['data_id', 'method'])[cols].mean()
        for method in ('Base', 'Shared_adapted'):
            v = units.xs(method, level='method')
            tables.append({'task': task, 'mode': mode, 'family': family, 'method': method,
                'datasets': len(v), **{k: float(v[k].mean()) for k in cols}})
        primary = 'auc' if task == 'classification' else 'r2'
        secondary = 'accuracy' if task == 'classification' else 'nrmse'
        a = units.xs('Shared_adapted', level='method')
        b = units.xs('Base', level='method')
        tests.append({'task': task, 'mode': mode, 'family': family, 'metric': primary,
            **original.paired_statistics(a[primary]-b[primary])})
        value = float((a[secondary]-b[secondary]).mean())*(1 if task == 'classification' else -1)
        directions.append({'task': task, 'mode': mode, 'family': family, 'oriented_secondary_gain': value})
    table, stats = pd.DataFrame(tables), pd.DataFrame(tests)
    assert len(stats) == 18
    stats['version'] = 'source_manifold_'+stats['mode']
    old = pd.read_csv(RUN / 'results/shared_source_crossfit_v135/calibration/cumulative_exploratory_statistics.csv')
    combined = pd.concat([old, stats], ignore_index=True)
    assert len(combined) == 147
    last = 0.
    for rank, i in enumerate(sorted(combined.index, key=lambda j: combined.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (147-rank)*combined.loc[i, 'p_two_sided']))
        combined.loc[i, 'p_holm_cumulative147'] = last
    outcomes = {}
    for mode in ('frozen_fusion', 'pooled', 'lomo'):
        g = combined[combined.version == 'source_manifold_'+mode]
        primary = bool(((g.gain > 0) & (g.ci_low > 0) & (g.p_holm_cumulative147 < .05)).all())
        secondary = all(x['oriented_secondary_gain'] > 0 for x in directions if x['mode'] == mode)
        outcomes[mode] = {'all_six_primary_positive_significant': primary,
            'all_six_secondary_positive': bool(secondary), 'all_requirements_met': bool(primary and secondary)}
    table.to_csv(OUT / 'family_averages.csv', index=False)
    stats.to_csv(OUT / 'paired_statistics.csv', index=False)
    combined.to_csv(OUT / 'cumulative_exploratory_statistics.csv', index=False)
    pd.DataFrame(directions).to_csv(OUT / 'secondary_changes.csv', index=False)
    old_table = pd.read_csv(original.OUT / 'family_averages.csv')
    for row in table[table.method == 'Base'].to_dict('records'):
        original_row = old_table[(old_table.task == row['task']) & (old_table.family == row['family']) &
            (old_table.method == 'Base')].iloc[0]
        for metric in (('accuracy', 'auc') if row['task'] == 'classification' else ('r2', 'nrmse')):
            assert np.isclose(row[metric], original_row[metric], atol=1e-12)
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS_EXPLICIT_CONSTRUCTOR_ADAPTATION',
        'Base_metrics_unchanged': True, 'original_benchmark_and_query_cohort_unchanged': True,
        'new_contexts_are_not_original_Shared': True,
        'all_null_negative_positive_results_preserved': True,
        'own_dataset_absent_from_head_calibration': True,
        'lomo_excludes_evaluated_generator': True, 'goal_checks': outcomes,
        'previously_evaluated_development_cohort': True, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Test source-CV gated transductive pseudo-label feature repair with OOF fusion',
        'Setting': plan, 'Results': json.loads(table.to_json(orient='records', double_precision=15)),
        'Analysis': {'explicit_constructor_adaptation': True, 'goal_checks': outcomes},
        'Next Steps': 'Confirm any selected adaptation on separately reserved rows; never describe this as the original frozen Shared.',
    })
    print(table.to_string(index=False), flush=True)
    print(stats.to_string(index=False), flush=True)
    print(json.dumps(outcomes, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['train', 'report'])
    a = p.parse_args()
    train() if a.phase == 'train' else report()


