"""Other-dataset recalibration of the reconstructed original Shared gate and five-context bank.

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
import original_shared_gate_adaptation_v203 as calibration
import original_shared_gate_adaptation_v203 as heads
import shared_native8_adaptation_v120 as bank
from prepare_data import ROOT, atomic_json, RUN
from run_tabicl import metrics

OUT = RUN/'results/original_shared_gate_adaptation_v204'
FAMILIES = calibration.FAMILIES


def freeze():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'ORIGINAL_RECONSTRUCTED_GATE_PARAMETER_RECALIBRATION', 'seed': 0,
        'original_five_Shared_constructor_roles_retained': True,
        'original_reconstructed_fusion_formula_retained': True,
        'new_context_rules_or_pseudo_labels_or_meta_features': False,
        'Base_and_benchmark_and_source_pool_and_query_cohort_unchanged': True,
        'context_row_cap': 256, 'alternative_contexts': 5,
        'backends': {'native1': 6, 'native8_bank_native1_gate': 41, 'native8_all': 48},
        'budget_units': 'configured internal estimator views, not exact FLOPs',
        'backbone_weights_updated': False, 'new_TFM_inference': False,
        'new_context_bank': str(bank.OUT),
        'bank_plan_sha256': original.sha(bank.OUT/'plan.json'),
        'original_plan_sha256': original.sha(original.OUT/'plan.json'),
        'original_constructor_sha256': original.sha(original.constructors.__file__),
        'head_sha256': original.sha(heads.__file__),
        'calibration_selection_implementation_sha256': original.sha(calibration.__file__),
        'fusion_implementation_sha256': original.sha(heads.fusion.__file__),
        'fast_metrics_sha256': original.sha(heads.fast.__file__),
        'script_sha256': original.sha(__file__),
        'outer_dataset_folds': 5, 'inner_dataset_folds': 4,
        'head_candidates': '21:3 backends times Base fallback or6 original gain/regularization choices',
        'gate_variant': 'original entropy gate, unchanged formula',
        'fit_labels': 'query labels of other dataset IDs only; evaluated whole ID excluded',
        'own_final_query_labels_excluded_from_all_fitting': True,
        'fit_objective': 'original classification NLL or query-variance-normalized regression MSE',
        'inner_selection': 'maximize worst-family OOF primary gain then macro gain over unchanged Native1 Base',
        'modes': {'pooled': 'other datasets; three new generators included in calibration',
            'lomo': 'exclude tested generator from all fitting and selection'},
        'primary_tests': 'two modes times3 perturbations times2 tasks;12 new exploratory tests',
        'cumulative_tests': '351 earlier tests plus12 new;Holm363',
        'significant_means': 'each endpoint positive,95CI lower>0,Holm363<.05',
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'not_bit_exact_archived_paper_reproduction': True, 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def load_cells(task, exclude_fold=None):
    cases = json.loads((original.OUT/'case_manifest.json').read_text())
    result = []
    for c in cases:
        if c['track'] != 'perturbation' or c['task'] != task or c['fold'] == exclude_fold:
            continue
        with np.load(c['truth_file']) as truth:
            y = truth['y_target']
        for path in sorted((original.OUT/'predictions/perturbation'/task/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert original.sha(path) == meta['output_sha256']
            assert meta['input_sha256'] == c['input_sha256']
            npath = bank.OUT/'predictions'/task/str(c['task_id'])/path.name
            nmeta = json.loads(npath.with_suffix('.json').read_text())
            assert original.sha(npath) == nmeta['output_sha256']
            assert nmeta['original_bank_sha256'] == meta['output_sha256']
            assert nmeta['source_input_sha256'] == c['input_sha256']
            with np.load(path) as data, np.load(npath) as native:
                np.testing.assert_array_equal(data['Base'], native['Base1'])
                values = {'P': np.concatenate([data['Base'][None], data['members']], axis=0),
                    'native8_base': native['Native8'], 'native8_bank': native['members'],
                    'mu': float(data['source_mean']), 'sigma': max(float(data['source_scale']), 1e-12)}
            result.append(values | {'task': task, 'task_id': c['task_id'], 'data_id': c['data_id'],
                'family': meta['family'], 'rho': meta['rho'], 'fold': c['fold'], 'y': y,
                'path': str(path), 'bank_sha256': meta['output_sha256'],
                'native8_path': str(npath), 'native8_sha256': nmeta['output_sha256']})
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
            'constructor_adaptation': False, 'original_gate_parameter_recalibration': True, 'model': model,
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
            for mode in ('pooled', 'lomo'):
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
                for method, q in (('Base', c['P'][0]), ('Shared_adapted', prediction), ('Native8_only', c['native8_base'])):
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
        for method in ('Base', 'Shared_adapted', 'Native8_only'):
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
    assert len(stats) == 12
    stats['version'] = 'original_gate_'+stats['mode']
    old = pd.read_csv(RUN / 'results/shared_source_evidence_v193/calibration/cumulative_exploratory_statistics.csv')
    combined = pd.concat([old, stats], ignore_index=True)
    assert len(combined) == 363
    last = 0.
    for rank, i in enumerate(sorted(combined.index, key=lambda j: combined.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (363-rank)*combined.loc[i, 'p_two_sided']))
        combined.loc[i, 'p_holm_cumulative363'] = last
    outcomes = {}
    for mode in ('pooled', 'lomo'):
        g = combined[combined.version == 'original_gate_'+mode]
        primary = bool(((g.gain > 0) & (g.ci_low > 0) & (g.p_holm_cumulative363 < .05)).all())
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
    atomic_json(OUT / 'audit.json', {'status': 'PASS_ORIGINAL_RECONSTRUCTED_GATE_PARAMETER_ADAPTATION',
        'Base_metrics_unchanged': True, 'original_benchmark_and_query_cohort_unchanged': True,
        'original_reconstructed_Shared_contexts_and_gate_formula_retained': True,
        'all_null_negative_positive_results_preserved': True,
        'own_dataset_absent_from_global_head_calibration': True,  'own_query_labels_excluded_from_all_fitting': True,
        'lomo_excludes_evaluated_generator': True, 'goal_checks': outcomes,
        'previously_evaluated_development_cohort': True, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Recalibrate only the original reconstructed Shared gate parameters and internal ensemble budget on other dataset IDs',
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


