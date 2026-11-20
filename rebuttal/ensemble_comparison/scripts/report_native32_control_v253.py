"""Audit and descriptively score the frozen Native32 control, with no new tests."""
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd

import official_native32_control_v252 as run
from run_tabicl import CHECKPOINTS, metrics


def main():
    assert json.loads((run.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((run.OUT/'plan.json').read_text())
    assert run.source.sha(run.__file__) == plan['script_sha256']
    assert run.source.sha(run.sys.modules[run.Predictor.__module__].__file__) == plan['predictor_sha256']
    assert run.source.sha(run.source.OUT/'plan.json') == plan['original_plan_sha256']
    for name, digest in json.loads((run.source.OUT/'plan.json').read_text())['checkpoints'].items():
        assert run.source.sha(CHECKPOINTS/name) == digest
    rows, actual_counts, conditions = [], [], 0
    for c in json.loads((run.source.OUT/'case_manifest.json').read_text()):
        if c['track'] != 'perturbation':
            continue
        assert run.source.sha(c['input_file']) == c['input_sha256']
        assert run.source.sha(c['truth_file']) == c['truth_sha256']
        with np.load(c['input_file']) as data:
            X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        with np.load(c['truth_file']) as truth:
            target = truth['y_target']
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in run.source.novel.STRENGTHS]:
            path = run.OUT/'predictions'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert run.source.sha(path) == meta['output_sha256']
            assert run.source.sha(meta['original_bank']) == meta['original_bank_sha256']
            assert meta['configured_estimators'] == 32 and not meta['query_task_labels_opened']
            if rho:
                X, y, Q, _ = run.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
            else:
                X, y, Q = X0, y0, Q0
            assert [run.protocol.array_sha256(v) for v in (X, y, Q)] == meta['input_hashes']
            assert meta['original_input_sha256'] == c['input_sha256']
            actual = meta['actual_estimator_views']
            assert sum(meta['actual_normalization_views'].values()) == actual
            assert set(meta['actual_normalization_views']).issubset({'none', 'power'})
            assert actual == 0 if meta['degenerate_input_fallback'] else 1 <= actual <= 32
            actual_counts.append(actual)
            with np.load(path) as data, np.load(meta['original_bank']) as original:
                np.testing.assert_array_equal(data['Base'], original['Base'])
                if meta['degenerate_input_fallback']:
                    np.testing.assert_array_equal(data['Native32'], data['Base'])
                for method in ('Base', 'Native32'):
                    prediction = data[method]
                    assert np.isfinite(prediction).all()
                    if c['task'] == 'classification':
                        np.testing.assert_allclose(prediction.sum(1), 1., atol=1e-6)
                    value = metrics(target, prediction, c['task'], max(float(target.std()), 1e-12))
                    if c['task'] == 'regression':
                        assert np.isclose(value['r2'], 1-value['nrmse']**2)
                    rows.append({k: c[k] for k in ('task', 'task_id', 'data_id')} |
                        {'family': family, 'rho': rho, 'method': method, **value})
            conditions += 1
    detail = pd.DataFrame(rows)
    summaries = []
    for (task, family, method), frame in detail[detail.rho > 0].groupby(['task', 'family', 'method']):
        columns = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        units = frame.groupby('data_id')[columns].mean()
        summaries.append({'task': task, 'family': family, 'method': method, 'datasets': len(units),
            **{k: float(units[k].mean()) for k in columns}})
    table = pd.DataFrame(summaries)
    assert len(table) == 12 and conditions == 1327 and len(detail) == 2654
    old = pd.read_csv(run.source.OUT/'family_averages.csv')
    for row in table[table.method == 'Base'].to_dict('records'):
        previous = old[(old.task == row['task']) & (old.family == row['family']) & (old.method == 'Base')].iloc[0]
        for metric in (('accuracy', 'auc') if row['task'] == 'classification' else ('r2', 'nrmse')):
            assert np.isclose(row[metric], previous[metric], atol=1e-12)
    detail.to_csv(run.OUT/'detail.csv', index=False)
    table.to_csv(run.OUT/'family_averages.csv', index=False)
    assert (run.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    audit = {'status': 'PASS', 'conditions': conditions, 'macro_rows': len(table),
        'observed_inputs_and_Base_byte_unchanged': True, 'checkpoint_hashes_unchanged': True,
        'official_native32_configuration_checked_during_inference': True,
        'minimum_actual_views': min(actual_counts), 'maximum_actual_views': max(actual_counts),
        'control_descriptive_no_new_primary_hypotheses': True,
        'query_labels_opened_only_for_scoring_after_inference_complete': True,
        'not_exact_FLOPs_or_latency_matching': True, 'goal_achievement_not_asserted': True, 'Markdown_bytes': 0}
    run.atomic_json(run.OUT/'independent_control_audit.json', audit)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
