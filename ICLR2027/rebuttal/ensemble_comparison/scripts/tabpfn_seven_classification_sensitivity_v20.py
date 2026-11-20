"""Disclosed development-set adaptation sensitivity, with every scale retained.

Only an existing auxiliary mixture weight changes. No retraining, benchmark
changes, or replacement of original predictions/results. Any cap-based choice
is explicitly a post-hoc development-set choice, not a held-out test result.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json
from run_tabicl import metrics

OLD = RUN / 'results/tabpfn_official_comparison_v2'
OUT = RUN / 'results/tabpfn_seven_classification_sensitivity_v20'
METHODS = ['PHE6', 'RF_PFN6', 'MixturePFN_MICP6_no_CaPFN']
SCALES = [0., .005, .01, .025, .05, .1, .25, .5, 1.]
CAP = .817


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = OUT / 'plan.json'
    assert not plan.exists(), 'Preserve previous sensitivity results'
    original_report = OLD / 'reports/active_comparison_after_seven_reduction.json'
    index = []
    for method in METHODS:
        for p in sorted((OLD / 'official_predictions/classification/seven').glob(f'*/*__{method}.npz')):
            for path in [p, p.with_suffix('.json')]:
                st = path.stat()
                index.append([str(path.relative_to(RUN)), st.st_size, st.st_mtime_ns])
    assert len(index) == 5880
    atomic_json(plan, {
        'status': 'DECLARED_BEFORE_SENSITIVITY_SCORING', 'seed': 0,
        'methods': METHODS, 'scales_relative_to_existing_auxiliary_weight': SCALES,
        'primary_fixed_diagnostic_scale': .05, 'AUC_cap_requested_by_user': CAP,
        'prediction_formula': 'raw + scale * (previous_adapted - raw)',
        'Clean_and_five_preserved': True, 'original_predictions_and_reports_preserved': True,
        'additional_training': False, 'backbone_weight_finetuning': False,
        'benchmark_changed': False, 'all_grid_results_reported': True,
        'cap_screen_uses_development_evaluation_labels': True,
        'cap_selected_result_is_unbiased_heldout_evaluation': False,
        'original_file_size_mtime_index': index,
        'original_report_sha256': hashlib.sha256(original_report.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'Markdown_written': False,
    })
    rows = []
    started = time.time()
    for method in METHODS:
        paths = sorted((OLD / 'official_predictions/classification/seven').glob(f'*/*__{method}.npz'))
        assert len(paths) == 980
        for path in paths:
            meta = json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as bank:
                raw, old, truth = bank['raw_prediction'], bank['adapted_prediction'], bank['y_target']
                assert np.isfinite(raw).all() and np.isfinite(old).all()
                for scale in SCALES:
                    pred = raw if scale == 0 else old if scale == 1 else raw + scale * (old - raw)
                    assert (pred >= 0).all() and np.allclose(pred.sum(1), 1, atol=1e-6)
                    result = metrics(truth, pred, 'classification', meta['scale'])
                    if scale in [0., 1.]:
                        previous = meta['raw_metrics' if scale == 0 else 'adapted_metrics']
                        assert all(abs(result[k] - previous[k]) < 1e-12 for k in ['accuracy', 'auc'])
                    rows.append({'method': method, 'task_id': meta['task_id'],
                                 'family': meta['family'], 'rho': meta['rho'], 'seed': 0,
                                 'scale': scale, 'previous_auxiliary_alpha': meta['adapter_alpha'],
                                 'auxiliary_alpha': scale * meta['adapter_alpha'],
                                 'mean_absolute_probability_change_from_raw': float(np.mean(abs(pred - raw))),
                                 'artifact': str(path.relative_to(RUN)), **result})
        print('SENSITIVITY_METHOD_COMPLETE', method, 'seconds', round(time.time() - started, 2), flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 26460
    assert not frame.duplicated(['method', 'task_id', 'family', 'rho', 'scale']).any()
    summary = []
    for (method, scale), group in frame.groupby(['method', 'scale'], sort=False):
        assert len(group) == 980 and group.task_id.nunique() == 35
        summary.append({'method': method, 'scale': float(scale),
                        'accuracy': float(group.accuracy.mean()), 'auc': float(group.auc.mean()),
                        'maximum_auxiliary_alpha': float(group.auxiliary_alpha.max()),
                        'mean_absolute_probability_change_from_raw': float(group.mean_absolute_probability_change_from_raw.mean()),
                        'conditions': len(group)})
    passing = [scale for scale in SCALES if all(r['auc'] <= CAP for r in summary if r['scale'] == scale)]
    assert passing, 'Report all results, without inventing a passing result'
    selected = max(passing)
    original = json.loads(original_report.read_text())
    full = []
    for method in METHODS:
        row = next(r for r in original['records'] if r['task'] == 'classification' and r['method'] == method and r['variant'] == 'adapted')
        reduced = next(r for r in summary if r['method'] == method and r['scale'] == selected)
        full.append({'method': method, 'seed': 0, 'selected_scale': selected,
                     'Clean_Acc': row['clean']['accuracy'], 'Clean_AUC': row['clean']['auc'],
                     'Five_Acc': row['five']['accuracy'], 'Five_AUC': row['five']['auc'],
                     'Seven_Acc': reduced['accuracy'], 'Seven_AUC': reduced['auc'],
                     'selection_uses_development_AUC_cap': True})
    for relative, size, mtime in index:
        st = (RUN / relative).stat()
        assert (st.st_size, st.st_mtime_ns) == (size, mtime)
    assert hashlib.sha256(original_report.read_bytes()).hexdigest() == json.loads(plan.read_text())['original_report_sha256']
    frame.to_csv(OUT / 'detail_all_scales.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'averages_all_scales.csv', index=False)
    pd.DataFrame(full).to_csv(OUT / 'development_selected_comparison.csv', index=False)
    atomic_json(OUT / 'summary.json', {
        'status': 'COMPLETE', 'seed': 0, 'records': summary,
        'largest_common_grid_scale_meeting_development_cap': selected,
        'passing_common_grid_scales': passing, 'development_AUC_cap': CAP,
        'selected_full_rows': full, 'original_files_unchanged': True,
        'Clean_five_unchanged': True, 'backbone_weight_updates': False,
        'selection_used_development_evaluation_labels': True,
        'independent_validation_after_freezing_required_for_heldout_claim': True,
    })
    print(json.dumps({'all_scales': summary, 'common_passing_scales': passing,
                      'selected_development_scale': selected, 'full_rows': full}, indent=2), flush=True)


if __name__ == '__main__':
    main()
