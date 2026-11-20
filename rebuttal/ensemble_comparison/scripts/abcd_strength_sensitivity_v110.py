"""Development-only inference-strength sweep; not a tuned baseline comparison.

Every saved A-D predictor is evaluated at every predeclared strength. A
frozen-at-one comparison is separately and explicitly labeled asymmetric.
Evaluation labels only compute metrics; no model/strength is selected.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import abcd_transfer_calibration_v109 as previous
import abcd_fresh_confirmation_v101 as fresh
from prepare_data import RUN, atomic_json
from run_tabicl import metrics

OUT = RUN / 'results/abcd_strength_sensitivity_v110'
STRENGTHS = (.25, .5, .75, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = previous.OUT
    seal = json.loads((source / 'policy_seal.json').read_text())
    assert all(digest(source / p) == h for p, h in seal.items())
    plan = {
        'code_sha256': digest(__file__), 'seed': 0, 'strengths': STRENGTHS,
        'formula': 'prediction(t) = Base + t * (saved_prediction - Base)',
        'source_results': str(source), 'context_cap': 256,
        'new_TFM_fits': 0, 'new_training': False,
        'source_policy_seal_sha256': digest(source / 'policy_seal.json'),
        'evaluation_labels_used_for_fitting_or_strength_selection': False,
        'strength_selected': None, 'all_grid_points_reported': True,
        'cohorts': ['old_development', 'additional_development'],
        'original_paper_replication': False, 'independent_confirmation': False,
        'comparison_modes': ['all_methods_same_strength', 'A_frozen_others_at_strength'],
        'asymmetric_mode_is_not_a_fair_best_tuned_mechanism_comparison': True,
        'lower_strength_does_not_guarantee_lower_accuracy_or_R2': True,
        'Markdown_written': False,
    }
    target = OUT / 'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == json.loads(json.dumps(plan))
    else:
        atomic_json(target, plan)
    rows = []

    def record(task, cohort, tid, did, rho, method, y, base, prediction):
        assert np.isfinite(base).all() and np.isfinite(prediction).all()
        for strength in STRENGTHS:
            q = prediction if strength == 1. else base + strength * (prediction - base)
            if task == 'classification':
                assert np.all(q >= -1e-10)
                assert np.allclose(q.sum(axis=1), 1., atol=1e-5)
            rows.append({'task': task, 'cohort': cohort, 'task_id': tid,
                'data_id': did, 'rho': rho, 'method': method, 'strength': strength,
                **metrics(y, q, task, max(float(y.std()), 1e-9))})

    for cohort in plan['cohorts']:
        folder = source / 'classification_predictions' / cohort
        for directory in sorted(folder.iterdir()):
            tid = int(directory.name)
            input_file = previous.classification.cls.banks.split.OUT / 'inputs' / f'{tid}.json'
            did = json.loads(input_file.read_text())['data_id']
            truth = (previous.classification.cls.banks.split.OUT / 'evaluation_truth' / f'{tid}.npz'
                if cohort == 'old_development' else fresh.OUT / 'truth/classification' / f'{tid}.npz')
            with np.load(truth) as z:
                y = z['y_evaluation']
            for rho in (0, 20, 50, 80, 100):
                with np.load(directory / f'{rho}_Base.npz') as z:
                    base = z['prediction']
                for method in ('A', 'B', 'C', 'D'):
                    with np.load(directory / f'{rho}_{method}.npz') as z:
                        prediction = z['prediction']
                    record('classification', cohort, tid, did, rho, method, y, base, prediction)
        print('COMPLETED classification', cohort, flush=True)

    policies = {(fold, method): json.loads((source / 'regression' / str(fold) /
        f'{method}_policy.json').read_text())['policy']
        for fold in range(5) for method in ('A', 'B', 'C', 'D')}
    for matched, methods in ((True, ('A', 'B')), (False, ('C', 'D'))):
        for cell in previous.inputs.load_cells(matched):
            m = cell['meta']
            if m['rho'] and m['family'] != 'concept':
                continue
            tid, did, rho = m['task_id'], m['data_id'], m['rho']
            for cohort in plan['cohorts']:
                c = cell.copy()
                if cohort == 'old_development':
                    y = c['y']
                else:
                    if 'extra_base' not in cell:
                        continue
                    c['base'], c['bank'] = cell['extra_base'], cell['extra_bank']
                    with np.load(fresh.OUT / 'truth/regression' / f'{tid}.npz') as z:
                        y = z['y_evaluation']
                for method in methods:
                    prediction = previous.algorithm.predict(c, policies[(cell['fold'], method)])
                    record('regression', cohort, tid, did, rho, method, y, c['base'], prediction)
        print('COMPLETED regression matched=', matched, flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'detail.csv', index=False)
    value_columns = ['accuracy', 'auc', 'r2', 'nrmse']
    summaries, contrasts = [], []
    for (task, cohort), group in detail.groupby(['task', 'cohort']):
        cols = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        by_dataset = group[group.rho > 0].groupby(['strength', 'method', 'data_id'])[cols].mean()
        clean = group[group.rho == 0].groupby(['strength', 'method'])[cols].mean()
        for strength in STRENGTHS:
            for method in ('A', 'B', 'C', 'D'):
                v = by_dataset.loc[(strength, method)]
                summaries.append({'task': task, 'cohort': cohort, 'strength': strength,
                    'method': method, 'datasets': len(v), **v.mean().to_dict(),
                    **{'clean_' + k: float(clean.loc[(strength, method), k]) for k in cols}})
            primary = cols[0]
            for mode in plan['comparison_modes']:
                for a, b in previous.classification.CONTRASTS:
                    sa = 1. if mode == 'A_frozen_others_at_strength' and a == 'A' else strength
                    delta = by_dataset.loc[(sa, a), primary] - by_dataset.loc[(strength, b), primary]
                    assert np.isfinite(delta).all()
                    contrasts.append({'task': task, 'cohort': cohort, 'strength': strength,
                        'mode': mode, 'contrast': a + '-' + b, 'datasets': len(delta),
                        'mean_difference': float(delta.mean())})
    tables = pd.DataFrame(summaries)
    tables.to_csv(OUT / 'tables.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'contrasts.csv', index=False)
    # Strength one must reproduce every original task/method aggregate.
    for task, filename in (('classification', 'classification_tables.csv'), ('regression', 'tables.csv')):
        old = pd.read_csv(source / filename)
        for row in tables[(tables.task == task) & (tables.strength == 1.)].to_dict('records'):
            baseline = old[(old.cohort == row['cohort']) & (old.method == row['method'])].iloc[0]
            for metric in value_columns:
                if metric in old.columns:
                    assert np.isclose(row[metric], baseline[metric], atol=1e-12)
    md = RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md'
    assert md.stat().st_size == 0
    assert all(digest(source / p) == h for p, h in seal.items())
    atomic_json(OUT / 'audit.json', {'status': 'PASS', 'original_policies_unchanged': True,
        'strength_one_matches_previous_results': True, 'all_strengths_reported': True,
        'fitting_or_strength_selection_from_evaluation_labels': False,
        'original_dataset_counts': {'classification': 72, 'regression': 33},
        'additional_dataset_counts': {'classification': 69, 'regression': 22},
        'no_new_TFM_inference_or_training': True, 'asymmetric_comparison_explicit': True,
        'Markdown_bytes': 0})
    print(tables[tables.cohort == 'old_development'].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
