"""Descriptive fixed-rule diagnostics; no selection or new hypothesis tests."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import shared_domain_overlap_banks_v167 as bank
from shared_task_residual_heads_v155 import metric


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    rows, domains = [], []
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    for c in cases:
        with np.load(c['truth_file']) as truth:
            y = truth['y_target']
        for p in sorted((bank.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(p.with_suffix('.json').read_text())
            assert bank.source.sha(p) == meta['output_sha256']
            if meta['rho'] == 0:
                continue
            with np.load(p) as data:
                slots = meta['replacement_slots']
                predictions = {'Base': data['Base'], 'Domain_keep75': data['members'][slots[0]],
                    'Domain_keep90': data['members'][slots[1]], 'Mean6':
                        np.concatenate([data['Base'][None], data['members']]).mean(axis=0)}
                for method, value in predictions.items():
                    primary, secondary = metric({'task': c['task'], 'y': y}, value)
                    values = {'auc': primary, 'accuracy': secondary} if c['task'] == 'classification' else {
                        'r2': primary, 'nrmse': -secondary}
                    rows.append({'task': c['task'], 'data_id': c['data_id'], 'family': meta['family'],
                        'rho': meta['rho'], 'method': method, **values})
                a, b = data['Domain_source_probability'], data['Domain_target_probability']
                domains.append({'task': c['task'], 'data_id': c['data_id'], 'family': meta['family'],
                    'rho': meta['rho'], 'domain_auc': roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])})
    frame = pd.DataFrame(rows)
    frame.to_csv(bank.OUT/'branch_diagnostic_detail.csv', index=False)
    table = frame.groupby(['task', 'family', 'method', 'data_id']).mean(numeric_only=True).groupby(
        ['task', 'family', 'method']).mean(numeric_only=True).drop(columns='rho').reset_index()
    table.to_csv(bank.OUT/'branch_diagnostic_averages.csv', index=False)
    pd.DataFrame(domains).to_csv(bank.OUT/'domain_diagnostic_detail.csv', index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
