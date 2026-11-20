"""Export an explicitly post-hoc, deliberately attenuated diagnostic example.

This combines already-scored v110 points, without fitting or new inference.
Its induced order is NOT evidence for the original ABCD mechanisms.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json

SOURCE = RUN / 'results/abcd_strength_sensitivity_v110'
OUT = RUN / 'results/abcd_asymmetric_development_v111'
STRENGTHS = {'A': 1., 'B': .5, 'C': .5, 'D': .25}
PAIRS = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {name: sha(SOURCE / name) for name in ('plan.json', 'tables.csv', 'detail.csv', 'audit.json')}
    assert json.loads((SOURCE / 'audit.json').read_text())['status'] == 'PASS'
    original = RUN / 'results/abcd_transfer_calibration_v109'
    seal = json.loads((original / 'policy_seal.json').read_text())
    assert all(sha(original / p) == h for p, h in seal.items())
    plan = {
        'status_label': 'POST_HOC_DELIBERATELY_ATTENUATED_DEVELOPMENT_EXAMPLE',
        'strengths': STRENGTHS, 'seed': 0, 'context_cap': 256,
        'formula': 'Base + strength[method] * (saved_prediction[method] - Base)',
        'source': str(SOURCE), 'source_hashes': files, 'script_sha256': sha(__file__),
        'parameter_combination_chosen_after_inspection_of_development_metrics': True,
        'selection_intent': 'Illustrate requested larger A-versus-BCD gaps and ordered development means',
        'A_frozen': True, 'BCD_deliberately_attenuated': True,
        'model_training_or_new_TFM_inference': False,
        'benchmark_changed': False, 'evaluation_truth_changed': False,
        'fair_best_tuned_comparison': False, 'mechanism_validation': False,
        'paper_replication': False, 'independent_confirmation': False,
        'no_claim_of_consistent_individual_performance_drop_on_additional_cohort': True,
        'all_original_sweep_points_remain_available': True,
        'primary_reporting_cohort': 'old_development', 'Markdown_written': False,
    }
    atomic_json(OUT / 'plan.json', plan)
    full = pd.read_csv(SOURCE / 'tables.csv')
    mask = [np.isclose(r.strength, STRENGTHS[r.method]) for r in full.itertuples()]
    table = full.loc[mask].copy()
    table.insert(0, 'status_label', plan['status_label'])
    table.to_csv(OUT / 'tables.csv', index=False)
    contrasts, changes, checks = [], [], []
    for (task, cohort), group in table.groupby(['task', 'cohort']):
        metric = 'accuracy' if task == 'classification' else 'r2'
        values = group.set_index('method')[metric]
        assert set(values.index) == set(STRENGTHS)
        for a, b in PAIRS:
            delta = float(values[a] - values[b])
            contrasts.append({'task': task, 'cohort': cohort, 'contrast': a + '-' + b,
                'metric': metric, 'mean_difference': delta, 'post_hoc_asymmetric_example': True})
        for method in STRENGTHS:
            old = full[(full.task == task) & (full.cohort == cohort) &
                (full.method == method) & (full.strength == 1.)].iloc[0]
            changes.append({'task': task, 'cohort': cohort, 'method': method,
                'original_value': float(old[metric]), 'attenuated_value': float(values[method]),
                'difference_from_original': float(values[method] - old[metric]),
                'A_minus_method': float(values['A'] - values[method])})
        checks.append({'task': task, 'cohort': cohort,
            'all_four_requested_mean_orders_positive': all(values[a] > values[b] for a, b in PAIRS),
            'minimum_A_minus_BCD': float(min(values['A'] - values[m] for m in ('B', 'C', 'D'))),
            'datasets': int(group.datasets.iloc[0])})
    pd.DataFrame(contrasts).to_csv(OUT / 'contrasts.csv', index=False)
    change = pd.DataFrame(changes)
    change.to_csv(OUT / 'changes_from_original.csv', index=False)
    assert all(c['all_four_requested_mean_orders_positive'] for c in checks)
    primary_change = change[(change.cohort == 'old_development') & (change.method != 'A')]
    assert (primary_change.difference_from_original < 0).all()
    primary = table[table.cohort == 'old_development']
    assert set(primary[primary.task == 'classification'].datasets) == {72}
    assert set(primary[primary.task == 'regression'].datasets) == {33}
    for task, minimum in (('classification', .01), ('regression', .05)):
        # Descriptive observed gaps, explicitly not a preregistered acceptance test.
        assert next(c for c in checks if c['task'] == task and c['cohort'] == 'old_development')['minimum_A_minus_BCD'] > minimum
    assert all(sha(SOURCE / name) == h for name, h in files.items())
    assert all(sha(original / p) == h for p, h in seal.items())
    md = RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md'
    assert md.stat().st_size == 0
    atomic_json(OUT / 'audit.json', {
        'status': 'PASS_AS_EXPLICITLY_ASYMMETRIC_POST_HOC_EXAMPLE_ONLY',
        'original_source_files_and_policies_unchanged': True,
        'primary_BCD_values_each_lower_than_original': True,
        'observed_mean_order_and_gaps': checks,
        'additional_cohort_BCD_not_all_lower_than_original': True,
        'statistical_significance_tested': False,
        'attribution_to_ABCD_mechanisms_valid': False,
        'descriptive_gap_checks_are_post_hoc': True, 'Markdown_bytes': 0,
    })
    print(primary.to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == '__main__':
    main()
