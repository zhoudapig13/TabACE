"""Paired report of prediction-only adaptation, preserving v84 controls."""
import json
import numpy as np
import pandas as pd

import abcd_fusion_adaptation_v86 as adapted
import abcd_mechanism_v84 as original
from prepare_data import atomic_json
from report_abcd_v84 import paired_statistics
from run_tabicl import metrics


def main():
    files = sorted((adapted.OUT / 'folds/classification').glob('*/detail.csv'))
    assert len(files) == 5, 'five adapted classification folds required'
    classification = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    prior = pd.read_csv(original.OUT / 'detail.csv')
    regression = prior[prior.task == 'regression'].copy()
    data = pd.concat([classification, regression], ignore_index=True)
    assert len(data) == 3150
    assert not data.duplicated(['task', 'data_id', 'rho', 'method']).any()
    rows = []
    for task, group in data.groupby('task'):
        primary, secondary = ('auc', 'accuracy') if task == 'classification' else ('r2', 'nrmse')
        scored = group[group.rho > 0].groupby(['data_id', 'method'])[[primary, secondary]].mean()
        clean = group[group.rho == 0].set_index(['data_id', 'method'])[[primary, secondary]]
        base = scored.xs('Base', level='method'); clean_base = clean.xs('Base', level='method')
        for method in ['Base', 'Single-matched', 'A', 'B', 'C', 'D']:
            values = scored.xs(method, level='method').loc[base.index]
            c = clean.xs(method, level='method').loc[base.index]
            rows.append({'task': task, 'method': method, 'datasets': len(base),
                primary: values[primary].mean(), secondary: values[secondary].mean(),
                'clean_primary_change': (c[primary] - clean_base[primary]).mean(),
                'clean_secondary_change': (c[secondary] - clean_base[secondary]).mean(),
                **paired_statistics(values[primary] - base[primary])})
    table = pd.DataFrame(rows)
    tested = table.index[table.method.isin(['A', 'B', 'C', 'D'])].tolist()
    ordered = sorted(tested, key=lambda i: table.loc[i, 'p_two_sided'])
    last = 0.
    for rank, i in enumerate(ordered):
        last = max(last, min(1., (8 - rank) * table.loc[i, 'p_two_sided']))
        table.loc[i, 'p_holm'] = last
    table['positive_significant'] = (table.gain_ci_low > 0) & (table.p_holm < .05)
    # Recompute all newly adapted metric rows and the complete mixture mapping.
    cells = {}
    for matched in (True, False):
        cells[matched] = {(c['meta']['task_id'], c['meta']['track'], c['meta']['family'], c['meta']['rho']): c
                          for c in original.load('classification', matched)}
    audited = 0
    for record in classification.to_dict('records'):
        folder = adapted.OUT / 'folds/classification' / str(record['fold'])
        p = folder / 'predictions' / str(record['task_id']) / f'{record["rho"]}_{record["method"]}.npz'
        with np.load(p) as z:
            y, prediction = z['y'], z['prediction']
        for key, value in metrics(y, prediction, 'classification', 1.).items():
            assert np.isclose(value, record[key], rtol=1e-11, atol=1e-11), p
        matched = record['method'] not in ['C', 'D']
        family = 'shortcut' if record['rho'] else 'original'
        cell = cells[matched][(record['task_id'], 'five_corrected', family, record['rho'])]
        assert np.array_equal(y, cell['y'])
        if record['method'] == 'Base':
            expected = cell['base']
        elif record['method'] == 'Single-matched':
            expected = cell['bank'][3]
        else:
            policy = json.loads((folder / f'{record["method"]}_policy.json').read_text())
            assert set(policy['training_data_ids']).isdisjoint(policy['excluded_data_ids'])
            expected = adapted.predict(cell, policy['model'])
            best = max(t['inner_dataset_auc'] for t in policy['trials'])
            selected = next(t for t in policy['trials'] if t['candidate'] == policy['selected'])
            assert abs(selected['inner_dataset_auc'] - best) < 1e-12
        assert np.allclose(prediction, expected, atol=1e-12, rtol=1e-12), p
        audited += 1
    data.to_csv(adapted.OUT / 'detail.csv', index=False)
    table.to_csv(adapted.OUT / 'tables.csv', index=False)
    assert (original.ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(adapted.OUT / 'audit.json', {'status': 'PASS', 'adapted_classification_metric_rows': audited,
        'predictions_and_policies_recomputed': True, 'Base_unchanged': True,
        'regression_rows_reused': len(regression), 'regression_audit': str(original.OUT / 'audit.json'),
        'all_eight_ABCD_positive_significant': bool(table.loc[tested, 'positive_significant'].all()),
        'original_paper_replication': False, 'Base_alignment_gate_passed': False,
        'independent_confirmation': False, 'Markdown_bytes': 0})
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
