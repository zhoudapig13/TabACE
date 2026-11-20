"""Independent operator replay and aggregate audit of the frozen v117 run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import unseen_shared_robustness_v117 as study
from prepare_data import atomic_json, protocol


def main():
    out = study.OUT
    plan = json.loads((out / 'plan.json').read_text())
    assert study.sha(study.__file__) == plan['code_sha256']
    assert study.sha(study.novel.__file__) == plan['generator_sha256']
    manifest = json.loads((out / 'case_manifest.json').read_text())
    expected = set()
    checked = 0
    categorical_checks = 0
    for c in manifest:
        path = Path(c['input_file'])
        assert study.sha(path) == c['input_sha256']
        assert study.sha(c['truth_file']) == c['truth_sha256']
        with np.load(path) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
            assert len(set(z['source_indices']) & set(z['target_indices'])) == 0
        categories = np.array(sorted(set(range(X0.shape[1]))-set(numeric)), dtype=int)
        policy_path = out / 'policies' / c['task'] / f'{c["fold"]}.json'
        policy = json.loads(policy_path.read_text())
        assert c['data_id'] not in policy['training_data_ids']
        assert c['data_id'] in policy['excluded_data_ids']
        conditions = [('clean', 0)] + [(f, r) for f in c['families'] for r in study.novel.STRENGTHS]
        for family, rho in conditions:
            pred = out / 'predictions' / c['track'] / c['task'] / str(c['task_id']) / f'{family}_{rho}.npz'
            expected.add(pred)
            m = json.loads(pred.with_suffix('.json').read_text())
            assert study.sha(pred) == m['output_sha256']
            if family == 'clean':
                X, y, Q, info = X0, y0, Q0, None
            else:
                X, y, Q, info = study.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
                assert info == m['intervention']
            for arr, key in ((X, 'observed_source_X_sha256'), (y, 'observed_source_y_sha256'),
                    (Q, 'observed_target_X_sha256')):
                assert protocol.array_sha256(arr) == m[key]
            assert not m['query_labels_opened_for_inference']
            assert not m['new_mechanism_id_or_strength_passed_to_Shared']
            assert m['policy_sha256'] == study.sha(policy_path)
            assert m['context_rows'] <= 256 and max(m['branch_rows']) <= 256
            assert m['actual_TFM_calls'] <= 6
            if family in ('cell_heavy_tail', 'feature_quantization', 'nonlinear_sensor', 'cross_feature_rotation'):
                assert np.array_equal(X[:, categories], X0[:, categories])
                assert np.array_equal(Q[:, categories], Q0[:, categories])
                categorical_checks += 1
            if family == 'column_desynchronization':
                assert np.array_equal(np.sort(X, axis=0), np.sort(X0, axis=0))
            if family == 'interaction_sampling':
                ids = info['selected_source_positions']
                assert np.array_equal(X, X0[ids]) and np.array_equal(y, y0[ids])
            elif family != 'clean':
                assert np.array_equal(y, y0)
            if family in ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization', 'interaction_sampling'):
                assert np.array_equal(Q, Q0)
            checked += 1
    actual = set((out / 'predictions').glob('*/*/*/*.npz'))
    assert expected == actual and checked == plan['expected_conditions']
    detail = pd.read_csv(out / 'detail.csv')
    table = pd.read_csv(out / 'family_averages.csv')
    assert len(detail) == 3*checked
    for row in table.to_dict('records'):
        g = detail[(detail.task == row['task']) & (detail.track == row['track']) &
            (detail.family == row['family']) & (detail.method == row['method']) & (detail.rho > 0)]
        assert g.groupby('data_id').size().eq(4).all()
        cols = ['accuracy', 'auc'] if row['task'] == 'classification' else ['r2', 'nrmse']
        units = g.groupby('data_id')[cols].mean()
        assert len(units) == row['datasets']
        for metric in cols:
            assert np.isclose(units[metric].mean(), row[metric], atol=1e-12)
    reg = detail[detail.task == 'regression']
    assert np.allclose(reg.nrmse**2, 1-reg.r2, rtol=1e-8, atol=1e-8)
    stats = pd.read_csv(out / 'paired_statistics.csv')
    assert len(stats) == 12 and stats.p_holm_12.between(0, 1).all()
    assert (study.ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(out / 'independent_audit.json', {
        'status': 'PASS', 'operator_conditions_replayed': checked,
        'input_and_truth_hashes_unchanged': True,
        'numeric_maps_preserve_categorical_values_checked_conditions': categorical_checks,
        'desynchronization_preserves_column_marginals': True,
        'interaction_selection_resamples_whole_source_pairs_only': True,
        'evaluation_row_ids_disjoint_from_source': True,
        'no_undeclared_skipped_or_added_conditions': True,
        'macro_averages_recomputed': len(table),
        'conditionwise_regression_R2_nRMSE_consistency': True,
        'not_an_independent_new_dataset_or_paper_replication_claim': True,
        'Markdown_bytes': 0,
    })
    print('PASS_INDEPENDENT_REPLAY', checked, 'conditions;', len(table), 'macro rows', flush=True)


if __name__ == '__main__':
    main()
