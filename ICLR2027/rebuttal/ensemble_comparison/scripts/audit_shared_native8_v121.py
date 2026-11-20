"""Verify all Native-8 adaptations and exact preservation of Base-1."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import shared_native8_adaptation_v120 as study
from prepare_data import atomic_json
from run_tabicl import normalize


def main():
    out = study.OUT
    plan = json.loads((out / 'plan.json').read_text())
    assert study.source.sha(study.__file__) == plan['script_sha256']
    assert json.loads((out / 'execution.json').read_text())['status'] == 'COMPLETE'
    count = 0
    for path in (out / 'predictions').glob('*/*/*.npz'):
        m = json.loads(path.with_suffix('.json').read_text())
        assert study.source.sha(path) == m['output_sha256']
        assert m['native_estimators_each_call'] == 8 and m['actual_TFM_context_calls'] <= 6
        assert max(m['branch_rows']) <= 256 and m['canonical_context_rows'] <= 256
        assert m['extra_target_labels'] == 0 and not m['target_labels_loaded_for_inference']
        with np.load(path) as z:
            base, members = z['Native8'], z['members']
            shared, conservative, mean = z['Shared_Native8'], z['Shared_Native8_conservative'], z['Mean6_Native8']
            original_base = z['Base1']
            cell = {'task': m['task'], 'base': base, 'bank': members,
                'mu': float(z['source_mean']), 'sigma': float(z['source_scale'])}
        policy = json.loads(Path(m['policy_file']).read_text())
        assert m['data_id'] not in policy['training_data_ids']
        assert np.array_equal(study.source.fusion.predict(cell, policy['Shared']), shared)
        expected_conservative = base+.25*(shared-base)
        expected_mean = np.concatenate([base[None], members], axis=0).mean(0)
        if m['task'] == 'classification':
            expected_conservative = normalize(expected_conservative)
            expected_mean = normalize(expected_mean)
        assert np.array_equal(expected_conservative, conservative)
        assert np.array_equal(expected_mean, mean)
        with np.load(m['original_bank']) as z:
            assert np.array_equal(original_base, z['Base'])
        count += 1
    assert count == plan['expected_conditions'] == 1327
    detail = pd.read_csv(out / 'detail.csv')
    table = pd.read_csv(out / 'family_averages.csv')
    for row in table.to_dict('records'):
        g = detail[(detail.task == row['task']) & (detail.family == row['family']) &
            (detail.method == row['method']) & (detail.rho > 0)]
        cols = ['accuracy', 'auc'] if row['task'] == 'classification' else ['r2', 'nrmse']
        units = g.groupby('data_id')[cols].mean()
        assert len(units) == row['datasets']
        assert g.groupby('data_id').size().eq(4).all()
        for metric in cols:
            assert np.isclose(units[metric].mean(), row[metric], atol=1e-12)
    combined = pd.read_csv(out / 'cumulative_exploratory_statistics.csv')
    assert len(combined) == 30
    previous = 0.
    for rank, i in enumerate(sorted(combined.index, key=lambda j: combined.loc[j, 'p_two_sided'])):
        previous = max(previous, min(1., (30-rank)*combined.loc[i, 'p_two_sided']))
        assert np.isclose(previous, combined.loc[i, 'p_holm_cumulative30'], atol=1e-12)
    achievements = {}
    secondary = []
    for method in ('Shared_Native8', 'Shared_Native8_conservative'):
        g = combined[combined.version == method]
        primary = bool(((g.gain > 0) & (g.ci_low > 0) & (g.p_holm_cumulative30 < .05)).all())
        directions = []
        for (task, family), rows in table.groupby(['task', 'family']):
            v = rows[rows.method == method].iloc[0]
            base = rows[rows.method == 'Base1'].iloc[0]
            gain = float(v.accuracy-base.accuracy) if task == 'classification' else float(base.nrmse-v.nrmse)
            directions.append(gain > 0)
            secondary.append({'method': method, 'task': task, 'family': family,
                'metric': 'accuracy' if task == 'classification' else 'negative_nrmse', 'gain': gain})
        achievements[method] = {'all_six_primary_cumulative_significant': primary,
            'all_six_secondary_directions_positive': bool(all(directions)),
            'all_primary_and_secondary_requirements_met': bool(primary and all(directions))}
    pd.DataFrame(secondary).to_csv(out / 'secondary_metric_changes.csv', index=False)
    assert (study.ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(out / 'independent_audit.json', {
        'status': 'PASS', 'prediction_conditions_verified': count,
        'Base1_predictions_bitwise_equal_to_original': True,
        'Native8_frozen_fusion_mean_and_conservative_predictions_recomputed': True,
        'thirty_family_macro_metric_rows_independently_recomputed': True,
        'cumulative30_Holm_correction_independently_recomputed': True,
        'goal_endpoint_checks': achievements, 'Markdown_bytes': 0,
        'budget_and_native_ensemble_confound_must_remain_disclosed': True,
    })
    print(json.dumps(achievements, indent=2), flush=True)


if __name__ == '__main__':
    main()
