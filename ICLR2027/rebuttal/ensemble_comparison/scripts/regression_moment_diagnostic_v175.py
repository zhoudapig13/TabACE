"""Describe prediction dispersion; oracle slopes never enter deployed models.

Query outcomes are read for diagnosis only. Any later deployable moment map
must use source-label moments and query predictions, not these oracle values.
"""
import json
import numpy as np
import pandas as pd

import shared_native8_generic_banks_v171 as bank
from prepare_data import atomic_json


def main():
    rows = []
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text())
        if c['track'] == 'perturbation' and c['task'] == 'regression']
    for c in cases:
        with np.load(c['truth_file']) as truth:
            y = truth['y_target']
        for path in sorted((bank.OUT/'banks/regression'/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert bank.source.sha(path) == meta['output_sha256']
            with np.load(path) as data:
                mu, sigma = float(data['source_mean']), max(float(data['source_scale']), 1e-12)
                values = {'Base': data['Base'], 'Native8': data['Canonical_native8'],
                    'Mean6_generic': np.concatenate([data['Base'][None], data['members']]).mean(axis=0)}
                for method, p in values.items():
                    variance = float(np.var(p))
                    slope = float(np.mean((p-p.mean())*(y-y.mean()))/variance) if variance > 1e-20 else np.nan
                    rows.append({'data_id': c['data_id'], 'family': meta['family'], 'rho': meta['rho'], 'method': method,
                        'prediction_std_over_source_label_std': float(p.std()/sigma),
                        'query_label_std_over_source_label_std_diagnostic_only': float(y.std()/sigma),
                        'query_label_minus_source_label_mean_over_source_std_diagnostic_only': float((y.mean()-mu)/sigma),
                        'prediction_minus_source_label_mean_over_source_std': float((p.mean()-mu)/sigma),
                        'oracle_slope_diagnostic_only_not_deployable': slope})
    frame = pd.DataFrame(rows)
    frame.to_csv(bank.OUT/'moment_diagnostic_detail.csv', index=False)
    grouped = frame.groupby(['data_id', 'family', 'method']).mean(numeric_only=True).reset_index()
    summary = grouped.groupby(['family', 'method']).median(numeric_only=True).drop(columns=['data_id', 'rho']).reset_index()
    summary.to_csv(bank.OUT/'moment_diagnostic_medians.csv', index=False)
    atomic_json(bank.OUT/'moment_diagnostic_scope.json', {'diagnostic_only': True,
        'oracle_slope_not_a_deployable_model_or_hyperparameter': True,
        'later_adaptation_must_not_load_this_diagnostic_file': True,
        'all_conditions_retained': True, 'no_new_hypothesis_tests': True})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
