"""Audited Concept table on the explicitly calibrated development protocol."""
import json
import numpy as np
import pandas as pd

import abcd_concept_calibrated_v87 as experiment
import abcd_mechanism_v84 as original
import tabace_fusion_v28 as rule
from prepare_data import atomic_json
from report_abcd_v84 import paired_statistics
from run_tabicl import metrics


def main():
    files = list((experiment.OUT / 'folds/regression').glob('*/detail.csv'))
    assert len(files) == 5
    regression = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(regression) == 990
    prior = pd.read_csv(original.OUT / 'detail.csv')
    combined = pd.concat([prior[prior.task == 'classification'], regression], ignore_index=True)
    output = []
    for task, group in combined.groupby('task'):
        primary, secondary = ('auc', 'accuracy') if task == 'classification' else ('r2', 'nrmse')
        scored = group[group.rho > 0].groupby(['data_id', 'method'])[[primary, secondary]].mean()
        clean = group[group.rho == 0].set_index(['data_id', 'method'])[[primary, secondary]]
        base = scored.xs('Base', level='method'); clean_base = clean.xs('Base', level='method')
        for method in ['Base', 'Single-matched', 'A', 'B', 'C', 'D']:
            values = scored.xs(method, level='method').loc[base.index]
            delta = values[primary] - base[primary]
            c = clean.xs(method, level='method').loc[base.index]
            output.append({'task': task, 'method': method, 'datasets': len(base),
                primary: values[primary].mean(), secondary: values[secondary].mean(),
                'clean_primary_change': (c[primary] - clean_base[primary]).mean(),
                'clean_secondary_change': (c[secondary] - clean_base[secondary]).mean(),
                'adaptation_dagger': task == 'regression', **paired_statistics(delta)})
    table = pd.DataFrame(output)
    tested = table.index[table.method.isin(['A', 'B', 'C', 'D'])].tolist()
    last = 0.
    for rank, i in enumerate(sorted(tested, key=lambda j: table.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (8 - rank) * table.loc[i, 'p_two_sided']))
        table.loc[i, 'p_holm'] = last
    table['positive_significant'] = (table.gain_ci_low > 0) & (table.p_holm < .05)
    cells = {matched: {(c['meta']['task_id'], c['meta']['family'], c['meta']['rho']): c
                        for c in original.load('regression', matched)} for matched in (True, False)}
    replay_differences = []
    for record in regression.to_dict('records'):
        folder = experiment.OUT / 'folds/regression' / str(record['fold'])
        path = folder / 'predictions' / str(record['task_id']) / f'{record["rho"]}_{record["method"]}.npz'
        with np.load(path) as z:
            prediction, y = z['prediction'], z['y']
        scale = max(float(y.std()), 1e-12)
        for key, value in metrics(y, prediction, 'regression', scale).items():
            assert np.isclose(value, record[key], rtol=1e-11, atol=1e-11), path
        matched = record['method'] not in ('C', 'D')
        cell = dict(cells[matched][(record['task_id'], 'concept' if record['rho'] else 'covariate', record['rho'])])
        if record['rho']:
            bank = experiment.OUT / 'banks' / str(record['task_id']) / f'concept_{record["rho"]}.npz'
            with np.load(bank) as z:
                cell.update(base=z['base'], bank=z['members'].copy(), mu=float(z['source_mean']), sigma=float(z['source_scale']))
                assert np.array_equal(y, z['y_target'])
                if matched:
                    cell['bank'][0] = z['repair']
        else:
            assert np.array_equal(y, cell['y'])
        if record['method'] == 'Base':
            expected = cell['base']
            if record['rho']:
                prior_path = original.RUN / 'results/abcd_base_source_sensitivity_v85/predictions/regression' / str(record['task_id']) / f'0.5_{record["rho"]}.npz'
                with np.load(prior_path) as z:
                    replay_differences.append(float(np.max(np.abs(z['prediction'] - prediction))))
        elif record['method'] == 'Single-matched':
            expected = cell['bank'][0]
        else:
            p = json.loads((folder / f'{record["method"]}_policy.json').read_text())
            assert set(p['training_data_ids']).isdisjoint(p['excluded_data_ids'])
            assert p['unseen_mechanism_audit']
            expected = rule.predict(cell, p['policy'])
            if record['method'] in ('B', 'D'):
                old = json.loads((original.OUT / 'folds/regression' / str(record['fold']) / f'{record["method"]}_policy.json').read_text())
                assert np.array_equal(p['policy']['parameters'], old['policy']['parameters'])
        assert np.allclose(prediction, expected, rtol=1e-12, atol=1e-12), path
    regression.to_csv(experiment.OUT / 'detail.csv', index=False)
    combined.to_csv(experiment.OUT / 'combined_detail.csv', index=False)
    table.to_csv(experiment.OUT / 'combined_tables.csv', index=False)
    table[table.task == 'regression'].to_csv(experiment.OUT / 'tables.csv', index=False)
    assert (original.ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(experiment.OUT / 'audit.json', {'status': 'PASS', 'rows_recomputed': len(regression),
        'paired_Base_and_method_environment': True, 'unseen_B_D_policies_identical_to_original_excluded_mechanism_training': True,
        'maximum_Base_replay_difference_from_v85': max(replay_differences),
        'independent_confirmation': False, 'original_paper_replication': False,
        'classification_Base_still_unaligned': True, 'all_eight_ABCD_positive_significant': bool(table.loc[tested, 'positive_significant'].all()),
        'Markdown_bytes': 0})
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
