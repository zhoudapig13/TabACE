"""Descriptive teacher/student controls on the development cohort; no fitting."""
import json
import numpy as np
import pandas as pd
import shared_pseudo_query_banks_v198 as bank
from shared_task_residual_heads_v155 import metric
from prepare_data import atomic_json


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    rows = []
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    for c in cases:
        with np.load(c['truth_file']) as data:
            y = data['y_target']
        for path in sorted((bank.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert bank.source.sha(path) == meta['output_sha256']
            with np.load(path) as data:
                predictions = {'Base': data['Base'], 'Teacher': data['Teacher'],
                    'Pseudo25': data['members'][3], 'Pseudo50': data['members'][4],
                    'Mean6': np.concatenate([data['Base'][None], data['members']]).mean(axis=0)}
                for method, value in predictions.items():
                    primary, secondary = metric({'task': c['task'], 'y': y}, value)
                    values = {'auc': primary, 'accuracy': secondary} if c['task'] == 'classification' else {
                        'r2': primary, 'nrmse': -secondary}
                    rows.append({'task': c['task'], 'data_id': c['data_id'], 'family': meta['family'],
                        'rho': meta['rho'], 'method': method, **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(bank.OUT/'branch_diagnostic_detail.csv', index=False)
    table = frame[frame.rho > 0].groupby(['task', 'family', 'method', 'data_id']).mean(numeric_only=True).groupby(
        ['task', 'family', 'method']).mean(numeric_only=True).drop(columns='rho').reset_index()
    table.to_csv(bank.OUT/'branch_diagnostic_averages.csv', index=False)
    atomic_json(bank.OUT/'branch_diagnostic_scope.json', {'diagnostic_only': True,
        'no_model_fitting_or_family_specific_selection': True,
        'all_teacher_and_student_controls_retained': True, 'no_new_significance_tests': True})
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
