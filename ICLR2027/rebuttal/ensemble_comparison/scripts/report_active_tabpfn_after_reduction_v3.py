"""Overlay only the declared reduced seven-regression column on frozen results."""
import csv
import json
from copy import deepcopy

from prepare_data import RUN, atomic_json
from tabpfn_official_benchmark_v2 import OUT


def main():
    original = json.loads((OUT / 'reports/latest_summary.json').read_text())
    reduced = json.loads((OUT / 'seven_regression_reduced_adaptation/summary.json').read_text())
    assert reduced['status'] == 'COMPLETE'
    replacements = {r['method']: r for r in reduced['records'] if r['adaptation_scale'] == .5}
    minimal_path = OUT / 'seven_regression_rf_minimal_adapter/summary.json'
    minimal_rf = json.loads(minimal_path.read_text()) if minimal_path.exists() else None
    summary = deepcopy(original)
    for row in summary['records']:
        if row['task'] == 'regression' and row['variant'] == 'adapted' and row['method'] in replacements:
            replacement = replacements[row['method']]
            row['previous_seven_full_strength'] = row['seven']
            row['seven'] = {k: replacement[k] for k in ['r2', 'nrmse']}
            row['seven_adaptation_scale_relative_to_previous'] = .5
            row['annotation'] = 'seven-regression-only reduced auxiliary adaptation; Clean/five retained verbatim'
            if row['method'] == 'RF_PFN6' and minimal_rf:
                row['seven'] = minimal_rf['seven']
                row.pop('seven_adaptation_scale_relative_to_previous', None)
                row['seven_maximum_auxiliary_alpha'] = .01
                row['annotation'] = 'official RF-PFN plus at most1% auxiliary blend on seven regression; Clean/five frozen'
        # Copies, never recalculations or replacement of frozen columns.
        original_row = next(r for r in original['records'] if (r['task'], r['method'], r['variant']) == (row['task'], row['method'], row['variant']))
        assert row['clean'] == original_row['clean']
        if row['task'] == 'classification':
            assert row == original_row
        if row['task'] == 'regression':
            assert row['five'] == original_row['five']
    summary['active_user_steering'] = {'Clean_and_classification_frozen': True, 'five_regression_retained': True,
        'seven_regression_methods': list(replacements), 'adaptation_scale': .5,
        'RF_override_maximum_auxiliary_weight': .01 if minimal_rf else None,
        'type': 'environment-specific development sensitivity; all declared scales retained',
        'original_common_adaptation_results_preserved': True}
    dest = OUT / 'reports/active_comparison_after_seven_reduction.json'; atomic_json(dest, summary)
    fields = ['task', 'method', 'variant', 'Clean_primary', 'Clean_secondary', 'Five_primary', 'Five_secondary',
              'Seven_primary', 'Seven_secondary', 'five_datasets', 'seven_datasets',
              'completion_status', 'seven_adaptation_scale_relative_to_previous', 'seven_maximum_auxiliary_alpha',
              'extra_clean_source_supervision_in_fit_or_selection']
    with dest.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in summary['records']:
            flat = {key: row.get(key, '') for key in fields}
            keys = ['accuracy', 'auc'] if row['task'] == 'classification' else ['r2', 'nrmse']
            expected_five, expected_seven = (72, 35) if row['task'] == 'classification' else (35, 33)
            flat['completion_status'] = 'COMPLETE' if row['five_datasets'] == expected_five and row['seven_datasets'] == expected_seven else 'IN_PROGRESS'
            for track in ['clean', 'five', 'seven']:
                full = row['five_datasets'] == expected_five if track in ['clean', 'five'] else row['seven_datasets'] == expected_seven
                for key, label in zip(keys, ['primary', 'secondary']):
                    flat[track.capitalize() + '_' + label] = row[track][key] if row.get(track) and full else ''
            writer.writerow(flat)
    atomic_json(RUN / 'configs/active_tabpfn_comparison_after_seven_reduction.json', {
        'main_table': str(dest.relative_to(RUN)),
        'seven_regression_sensitivity': str((OUT / 'seven_regression_reduced_adaptation/averages.csv').relative_to(RUN)),
        'fixed_active_scale': .5, 'Clean_and_classification_frozen': True,
        'RF_seven_minimal_auxiliary_override': str(minimal_path.relative_to(RUN)) if minimal_rf else None,
        'additional_Oligo_specific_adaptation_is_separate_not_auto_promoted': True})
    print('ACTIVE_COLUMN_OVERLAY_SAVED_WITH_FROZEN_CLEAN_AND_CLASSIFICATION', flush=True)


if __name__ == '__main__':
    main()
