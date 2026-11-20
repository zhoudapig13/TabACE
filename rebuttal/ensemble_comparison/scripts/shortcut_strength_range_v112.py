"""Check bounded attenuation range and separately report a supervision control.

No predictor or label changes. The source-only Base reference is not D and
has a different target-supervision budget. All grid points are preserved.
"""
import json
import hashlib

import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json

SOURCE = RUN / 'results/abcd_transfer_calibration_v109'
EXAMPLE = RUN / 'results/abcd_asymmetric_development_v111'
OUT = RUN / 'results/shortcut_strength_range_v112'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    strengths = np.linspace(0., 1., 101)
    atomic_json(OUT / 'plan.json', {
        'status': 'DEVELOPMENT_GRID_DIAGNOSTIC_WITH_SEPARATE_SUPERVISION_REFERENCE',
        'strengths': strengths.tolist(), 'no_training': True,
        'evaluation_labels_used_to_compute_metrics_only': True,
        'no_strength_selected': True, 'seed': 0, 'context_cap': 256,
        'source_only_Base_is_not_D': True,
        'source_only_reference_uses_zero_target_labels_rather_than_64': True,
        'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'Markdown_written': False,
    })
    truth_dir = RUN / 'results/target_calibration_split_v97/evaluation_truth'
    rows = []
    for folder in sorted((SOURCE / 'classification_predictions/old_development').iterdir()):
        tid = int(folder.name)
        with np.load(truth_dir / f'{tid}.npz') as z:
            y = z['y_evaluation']
        for rho in (20, 50, 80, 100):
            with np.load(folder / f'{rho}_Base.npz') as z:
                base = z['prediction']
            for method in ('B', 'C', 'D'):
                with np.load(folder / f'{rho}_{method}.npz') as z:
                    prediction = z['prediction']
                q = base[None] + strengths[:, None, None] * (prediction - base)[None]
                accuracy = (q.argmax(2) == y[None]).mean(1)
                rows.extend({'task_id': tid, 'rho': rho, 'method': method,
                    'strength': float(strength), 'accuracy': float(value)}
                    for strength, value in zip(strengths, accuracy))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'detail.csv', index=False)
    ds = detail.groupby(['method', 'strength', 'task_id']).accuracy.mean()
    summary = ds.groupby(['method', 'strength']).agg(['mean', 'count']).reset_index()
    summary.to_csv(OUT / 'strength_grid.csv', index=False)
    assert summary['count'].eq(72).all()
    old = pd.read_csv(SOURCE / 'classification_tables.csv')
    example = pd.read_csv(EXAMPLE / 'tables.csv')
    comparison = []
    for method in ('A', 'B', 'C', 'D'):
        r = example[(example.task == 'classification') & (example.cohort == 'old_development') & (example.method == method)].iloc[0]
        comparison.append({'method': method, 'status': 'post_hoc_asymmetric_development_example',
            'target_calibration_labels': 64, 'strength': r.strength,
            'accuracy': r.accuracy, 'auc': r.auc, 'datasets': 72})
    r = old[(old.cohort == 'old_development') & (old.method == 'Base_zero')].iloc[0]
    comparison.append({'method': 'Base_zero_target_labels',
        'status': 'separate_supervision_reference_NOT_D', 'target_calibration_labels': 0,
        'strength': None, 'accuracy': r.accuracy, 'auc': r.auc, 'datasets': 72})
    pd.DataFrame(comparison).to_csv(OUT / 'comparison_with_separate_reference.csv', index=False)
    assert r.accuracy < .7
    ranges = {m: {'minimum_grid_mean_accuracy': float(g['mean'].min()),
        'maximum_grid_mean_accuracy': float(g['mean'].max()),
        'number_of_grid_means_below_0_7': int((g['mean'] < .7).sum())}
        for m, g in summary.groupby('method')}
    assert (RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {
        'grid_ranges': ranges,
        'scope': '101 global strengths in [0,1], classification main development cohort only',
        'no_claim_of_continuous_interval_impossibility': True,
        'threshold_below_0_7_met_only_by_separate_zero_target_Base_reference': bool(r.accuracy < .7 and not (summary['mean'] < .7).any()),
        'original_A_B_C_D_results_not_overwritten': True, 'no_fitting': True,
        'no_independent_confirmation_or_mechanism_validation': True, 'Markdown_bytes': 0,
    })
    print(json.dumps(ranges, indent=2))
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == '__main__':
    main()
