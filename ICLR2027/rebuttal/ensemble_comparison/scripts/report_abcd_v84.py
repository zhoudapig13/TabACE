"""Independent metric/statistic report for the frozen v84 mechanism study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import abcd_mechanism_v84 as study
from prepare_data import atomic_json
from run_tabicl import metrics


def paired_statistics(delta):
    d = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    ci = np.quantile(means, [.025, .975])
    observed = abs(float(d.mean()))
    extreme = 0
    for _ in range(100):
        signs = 2 * rng.integers(0, 2, size=(2000, len(d))) - 1
        extreme += int(np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= observed - 1e-15))
    return {'gain': float(d.mean()), 'gain_ci_low': float(ci[0]), 'gain_ci_high': float(ci[1]),
            'p_two_sided': (extreme + 1) / 200001,
            'wins': int(np.sum(d > 1e-6)), 'ties': int(np.sum(np.abs(d) <= 1e-6)),
            'losses': int(np.sum(d < -1e-6))}


def baseline_report():
    plan = json.loads((study.OUT / 'plan.json').read_text())
    rows = []
    for task, (track, family) in study.TARGETS.items():
        for path in study.source_paths(task):
            meta = json.loads(path.with_suffix('.json').read_text())
            if meta['track'] != track or (meta['rho'] and meta['family'] != family):
                continue
            with np.load(path) as f:
                y, pred = f['y_target'], f['base']
            values = metrics(y, pred, task, max(float(y.std()), 1e-12) if task == 'regression' else 1.)
            rows.append({'task': task, 'data_id': meta['data_id'], 'rho': meta['rho'], **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(study.OUT / 'baseline_detail.csv', index=False)
    reports = []
    for task, group in frame.groupby('task'):
        keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        for block, sub in [('clean', group[group.rho == 0]), ('perturbed', group[group.rho > 0])]:
            actual = sub.groupby('data_id')[keys].mean().mean()
            ref = plan['paper_reference'][task]
            if block == 'clean':
                severity = 'rho' if task == 'classification' else 'strength'
                historical = next(row for row in ref['rows'] if row[severity] == 0)
            else:
                historical = ref['positive_strength_mean']
            row = {'task': task, 'block': block, 'datasets': sub.data_id.nunique()}
            for key in keys:
                target = historical[f'mean_base_{key}']
                row.update({key: float(actual[key]), f'paper_{key}': target, f'difference_{key}': float(actual[key] - target)})
            reports.append(row)
    table = pd.DataFrame(reports)
    table.to_csv(study.OUT / 'baseline_reference_comparison.csv', index=False)
    print(table.to_string(index=False), flush=True)


def report(audit=False):
    files = sorted((study.OUT / 'folds').glob('*/*/detail.csv'))
    if not files:
        print('NO_COMPLETED_FOLDS', flush=True)
        return
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert not frame.duplicated(['task', 'data_id', 'rho', 'method']).any()
    frame.to_csv(study.OUT / 'detail.csv', index=False)
    output = []
    for task, group in frame.groupby('task'):
        primary = 'auc' if task == 'classification' else 'r2'
        secondary = 'accuracy' if task == 'classification' else 'nrmse'
        perturbed = group[group.rho > 0].groupby(['data_id', 'method'])[[primary, secondary]].mean()
        clean = group[group.rho == 0].set_index(['data_id', 'method'])[[primary, secondary]]
        base = perturbed.xs('Base', level='method')
        clean_base = clean.xs('Base', level='method')
        for method in ['Base', 'Single-matched', 'A', 'B', 'C', 'D']:
            values = perturbed.xs(method, level='method')
            ids = base.index.intersection(values.index)
            delta = values.loc[ids, primary] - base.loc[ids, primary]
            c = clean.xs(method, level='method').loc[ids]
            output.append({'task': task, 'method': method, 'datasets': len(ids),
                primary: float(values.loc[ids, primary].mean()), secondary: float(values.loc[ids, secondary].mean()),
                'clean_primary_change': float((c[primary] - clean_base.loc[ids, primary]).mean()),
                'clean_secondary_change': float((c[secondary] - clean_base.loc[ids, secondary]).mean()),
                **paired_statistics(delta)})
    table = pd.DataFrame(output)
    tested = table.index[table.method.isin(['A', 'B', 'C', 'D'])].tolist()
    ordered = sorted(tested, key=lambda i: table.loc[i, 'p_two_sided'])
    previous = 0.
    for rank, i in enumerate(ordered):
        adjusted = max(previous, min(1., (len(ordered) - rank) * table.loc[i, 'p_two_sided']))
        table.loc[i, 'p_holm'] = adjusted
        previous = adjusted
    table['positive_significant'] = (table.gain_ci_low > 0) & (table.p_holm < .05)
    table.to_csv(study.OUT / 'tables.csv', index=False)
    complete = len(files) == 10 and len(frame) == 3150
    atomic_json(study.OUT / 'statistical_status.json', {
        'status': 'COMPLETE' if complete else 'PARTIAL', 'outer_fold_files': len(files),
        'rows': len(frame), 'all_eight_ABCD_positive_significant': bool(complete and table.loc[tested, 'positive_significant'].all()),
        'bootstrap_unit': 'dataset after averaging its four positive severities',
        'inference_scope': 'conditional on fitted cross-validated policies; no pipeline refit bootstrap',
        'multiple_testing': f'Holm over {len(tested)} currently completed A-D comparisons; final family is eight',
        'original_paper_replication': False, 'base_alignment_gate_passed': False,
        'regression_is_adapted_dagger': True, 'seed': 0, 'new_seed_variance_claim': False})
    if audit:
        assert complete, (len(files), len(frame))
        errors = []
        for row in frame.to_dict('records'):
            path = study.OUT / 'folds' / row['task'] / str(row['fold']) / 'predictions' / str(row['task_id']) / f'{row["rho"]}_{row["method"]}.npz'
            with np.load(path) as f:
                p, y = f['prediction'], f['y']
            v = metrics(y, p, row['task'], max(float(y.std()), 1e-12) if row['task'] == 'regression' else 1.)
            for key, value in v.items():
                assert np.isclose(value, row[key], rtol=1e-11, atol=1e-11), (path, key)
            if row['method'] == 'Base':
                track, family = study.TARGETS[row['task']]
                name = f'{family}_{row["rho"]}.npz' if row['rho'] else ('original_0.npz' if row['task'] == 'classification' else 'covariate_0.npz')
                origin = study.RUN / 'results/reference/tabicl' / track / row['task'] / str(row['task_id']) / name
                with np.load(origin) as f:
                    assert np.array_equal(y, f['y_target']) and np.array_equal(p, f['base']), origin
        for path in (study.OUT / 'folds').glob('*/*/*_policy.json'):
            policy = json.loads(path.read_text())
            assert set(policy['training_data_ids']).isdisjoint(policy['excluded_data_ids']), path
            assert policy['unseen_mechanism_audit'], path
        for record in json.loads((study.OUT / 'input_manifest.json').read_text()):
            assert study.sha(record['path']) == record['sha256'], record['path']
        md = study.ROOT / 'REBUTTAL_EXPERIMENTS.md'
        assert md.stat().st_size == 0
        atomic_json(study.OUT / 'audit.json', {'status': 'PASS', 'metric_rows_recomputed': len(frame),
            'Base_and_truth_bitwise_unchanged': True, 'outer_dataset_disjointness': True,
            'input_manifest_unchanged': True, 'Markdown_bytes': 0,
            'scope': 'cache-based reconstruction diagnostic; not original-protocol or fresh-backbone replay verification'})
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['baseline', 'report', 'audit'])
    args = parser.parse_args()
    if args.phase == 'baseline':
        baseline_report()
    else:
        report(audit=args.phase == 'audit')
