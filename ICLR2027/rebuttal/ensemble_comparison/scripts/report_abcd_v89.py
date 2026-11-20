"""Keep selected and unselected support-filter outcomes, with paired audits."""
import json
import numpy as np
import pandas as pd

import abcd_support_filter_v89 as study
import abcd_concept_calibrated_v87 as concept
import tabace_fusion_v28 as rule
from prepare_data import atomic_json
from report_abcd_v84 import paired_statistics
from run_tabicl import metrics


def aggregate(data):
    output = []
    for task, group in data.groupby('task'):
        primary, secondary = ('auc', 'accuracy') if task == 'classification' else ('r2', 'nrmse')
        g = group[group.rho > 0].groupby(['data_id', 'method'])[[primary, secondary]].mean()
        c = group[group.rho == 0].set_index(['data_id', 'method'])[[primary, secondary]]
        base = g.xs('Base', level='method'); clean_base = c.xs('Base', level='method')
        for method in ['Base', 'Single-matched', 'A', 'B', 'C', 'D']:
            values = g.xs(method, level='method').loc[base.index]
            clean = c.xs(method, level='method').loc[base.index]
            output.append({'task': task, 'method': method, 'datasets': len(base),
                primary: values[primary].mean(), secondary: values[secondary].mean(),
                'clean_primary_change': (clean[primary] - clean_base[primary]).mean(),
                'clean_secondary_change': (clean[secondary] - clean_base[secondary]).mean(),
                **paired_statistics(values[primary] - base[primary])})
    result = pd.DataFrame(output)
    tested = result.index[result.method.isin(['A', 'B', 'C', 'D'])].tolist()
    last = 0.
    for rank, i in enumerate(sorted(tested, key=lambda j: result.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (8 - rank) * result.loc[i, 'p_two_sided']))
        result.loc[i, 'p_holm'] = last
    result['positive_significant'] = (result.gain_ci_low > 0) & (result.p_holm < .05)
    return result


def main():
    files = list((study.OUT / 'folds/classification').glob('*/detail.csv'))
    assert len(files) == 5
    all_data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(all_data) == 5040
    methods = ['Base', 'Single-matched', 'A', 'B', 'C', 'D']
    selected = all_data[all_data.method.isin(methods)].copy()
    assert len(selected) == 2160
    regression = pd.read_csv(concept.OUT / 'detail.csv')
    combined = pd.concat([selected, regression], ignore_index=True)
    table = aggregate(combined)
    all_data.to_csv(study.OUT / 'all_candidate_detail.csv', index=False)
    selected.to_csv(study.OUT / 'selected_detail.csv', index=False)
    combined.to_csv(study.OUT / 'combined_detail.csv', index=False)
    table.to_csv(study.OUT / 'combined_tables.csv', index=False)
    candidate_data = all_data[~all_data.method.isin(['Base', 'Single-matched'])].copy()
    candidate_data['method_role'] = candidate_data.method.str.split('_').str[0]
    candidate_data.groupby(['variant', 'method_role', 'rho'])[['accuracy', 'auc']].mean().to_csv(study.OUT / 'all_candidate_strength_means.csv')
    candidate_data[candidate_data.rho > 0].groupby(['variant', 'method_role', 'data_id'])[['accuracy', 'auc']].mean().groupby(['variant', 'method_role']).mean().to_csv(study.OUT / 'all_candidate_means.csv')
    cells = {(matched, variant): {(c['meta']['task_id'], c['meta']['track'], c['meta']['family'], c['meta']['rho']): c
                for c in study.load(matched, variant)} for matched in (True, False)
                for variant in ('original', 'fraction_only', 'support_filter')}
    policies = {}
    for path in (study.OUT / 'folds/classification').glob('*/*_policy.json'):
        policy = json.loads(path.read_text()); fold = int(path.parent.name); role = policy['method']
        assert set(policy['training_data_ids']).isdisjoint(policy['excluded_data_ids'])
        assert policy['selected_variant'] == min(policy['candidates'], key=lambda c: c['inner_CE'])['variant']
        policies[(fold, role)] = policy
    for r in all_data.to_dict('records'):
        role = r['method'] if r['method'] in ('Base', 'Single-matched') else r['method'].split('_')[0]
        matched = role not in ('C', 'D')
        cell = cells[(matched, r['variant'])][(r['task_id'], 'five_corrected', 'shortcut' if r['rho'] else 'original', r['rho'])]
        path = study.OUT / 'folds/classification' / str(r['fold']) / 'predictions' / str(r['task_id']) / f'{r["rho"]}_{r["method"]}.npz'
        with np.load(path) as z:
            prediction, y = z['prediction'], z['y']
        assert np.array_equal(y, cell['y'])
        if role == 'Base':
            expected = cell['base']
        elif role == 'Single-matched':
            expected = cell['bank'][3]
        else:
            policy = policies[(r['fold'], role)]
            candidate = next(c for c in policy['candidates'] if c['variant'] == r['variant'])
            expected = rule.predict(cell, candidate['policy'])
            assert (r['method'] == role) == (r['variant'] == policy['selected_variant'])
        assert np.allclose(prediction, expected, atol=1e-12, rtol=1e-12), path
        for key, value in metrics(y, prediction, 'classification', 1.).items():
            assert np.isclose(value, r[key], atol=1e-11, rtol=1e-11), path
    md = study.previous.ROOT / 'REBUTTAL_EXPERIMENTS.md'; assert md.stat().st_size == 0
    atomic_json(study.OUT / 'audit.json', {'status': 'PASS', 'metric_and_policy_prediction_rows_recomputed': len(all_data),
        'selected_rows': len(selected), 'all_selected_and_unselected_candidates_retained': True,
        'selection_uses_only_inner_validation_CE': True, 'Base_unchanged': True,
        'regression_source': str(concept.OUT / 'audit.json'), 'original_paper_replication': False,
        'independent_confirmation': False, 'Markdown_bytes': 0,
        'all_eight_ABCD_positive_significant': bool(table[table.method.isin(['A', 'B', 'C', 'D'])].positive_significant.all())})
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
