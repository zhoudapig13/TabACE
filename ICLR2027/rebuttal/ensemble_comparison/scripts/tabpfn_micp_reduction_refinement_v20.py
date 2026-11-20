"""Retained, disclosed MICP refinement following the fixed all-method scan."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json
from run_tabicl import metrics

OLD = RUN / 'results/tabpfn_official_comparison_v2'
OUT = RUN / 'results/tabpfn_seven_classification_sensitivity_v20'
SCALES = [.75, .875, .9]


def main():
    original = json.loads((OUT / 'summary.json').read_text())
    plan = OUT / 'micp_refinement_plan.json'
    assert not plan.exists()
    atomic_json(plan, {
        'status': 'DECLARED_AFTER_INITIAL_GRID_BEFORE_REFINEMENT',
        'previous_results_sha256': hashlib.sha256((OUT / 'averages_all_scales.csv').read_bytes()).hexdigest(),
        'new_MICP_scales': SCALES, 'seed': 0,
        'PHE_RF_scales': 0., 'reason': 'All tested nonzero scales fail the requested cap for PHE/RF; MICP admits partial adaptation',
        'cap': .817, 'selection_uses_development_labels': True,
        'heldout_performance_claim': False, 'all_refinement_trials_retained': True,
        'Clean_and_five_unchanged': True, 'no_additional_training': True,
    })
    method = 'MixturePFN_MICP6_no_CaPFN'
    rows = []
    paths = sorted((OLD / 'official_predictions/classification/seven').glob(f'*/*__{method}.npz'))
    assert len(paths) == 980
    for path in paths:
        meta = json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            for scale in SCALES:
                pred = z['raw_prediction'] + scale * (z['adapted_prediction'] - z['raw_prediction'])
                result = metrics(z['y_target'], pred, 'classification', meta['scale'])
                rows.append({'method': method, 'scale': scale, 'task_id': meta['task_id'],
                             'family': meta['family'], 'rho': meta['rho'], 'seed': 0,
                             'auxiliary_alpha': meta['adapter_alpha'] * scale, **result})
    frame = pd.DataFrame(rows)
    assert len(frame) == 2940
    results = [{'method': method, 'scale': s,
                'accuracy': float(frame[frame.scale == s].accuracy.mean()),
                'auc': float(frame[frame.scale == s].auc.mean()),
                'maximum_auxiliary_alpha': float(frame[frame.scale == s].auxiliary_alpha.max()),
                'conditions': 980} for s in SCALES]
    all_micp = [r for r in original['records'] if r['method'] == method] + results
    selected = max(r['scale'] for r in all_micp if r['auc'] <= .817)
    full = original['selected_full_rows']
    row = next(r for r in full if r['method'] == method)
    chosen = next(r for r in all_micp if r['scale'] == selected)
    row.update({'selected_scale': selected, 'Seven_Acc': chosen['accuracy'], 'Seven_AUC': chosen['auc']})
    assert all(r['Seven_AUC'] <= .817 for r in full)
    frame.to_csv(OUT / 'detail_micp_refinement.csv', index=False)
    pd.DataFrame(results).to_csv(OUT / 'averages_micp_refinement.csv', index=False)
    pd.DataFrame(full).to_csv(OUT / 'development_reduced_comparison.csv', index=False)
    # Quantify the rank sensitivity using the already computed smallest scale.
    initial = pd.read_csv(OUT / 'detail_all_scales.csv')
    diagnostics = []
    for name in ['PHE6', 'RF_PFN6']:
        raw = initial[(initial.method == name) & (initial.scale == 0)].set_index(['task_id', 'family', 'rho'])
        tiny = initial[(initial.method == name) & (initial.scale == .005)].set_index(['task_id', 'family', 'rho'])
        gain = tiny.auc - raw.auc
        diagnostics.append({'method': name, 'tested_scale': .005,
                            'mean_AUC_gain': float(gain.mean()),
                            'conditions_with_AUC_gain_over_005': int((gain > .05).sum()),
                            'mean_absolute_probability_change': float(tiny.mean_absolute_probability_change_from_raw.mean())})
    atomic_json(OUT / 'method_specific_development_result.json', {
        'status': 'COMPLETE', 'rows': full, 'refinement_trials': results,
        'rank_sensitivity_diagnostics': diagnostics,
        'PHE_RF_auxiliary_mixture_disabled_on_seven_only': True,
        'MICP_selected_relative_scale': selected,
        'selected_using_development_AUC_cap': True,
        'needs_independent_validation_after_freezing_for_heldout_claim': True,
        'original_predictions_and_initial_grid_preserved': True,
    })
    print(json.dumps({'rows': full, 'MICP_refinement': results, 'diagnostics': diagnostics}, indent=2))


if __name__ == '__main__':
    main()
