"""Finalize v114 reporting after a numpy-scalar JSON serialization failure.

No inference, policy, benchmark or metric changes. The 48-row display is
explicitly a post-hoc development highlight of the already complete grid.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json

OUT = RUN / 'results/shortcut_context_label_budget_v114'
POLICIES = RUN / 'results/abcd_transfer_calibration_v109'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    code = Path(__file__).with_name('shortcut_context_label_budget_v114.py')
    assert sha(code) == plan['code_sha256']
    assert json.loads((OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    before = {f: sha(OUT / f) for f in ('plan.json', 'detail.csv', 'tables.csv', 'audit.json')}
    table = pd.read_csv(OUT / 'tables.csv')
    detail = pd.read_csv(OUT / 'detail.csv')
    for r in table.to_dict('records'):
        g = detail[(detail.context_target_rows == r['context_target_rows']) &
            (detail.method == r['method'])]
        pert = g[g.rho > 0].groupby('data_id')[['accuracy', 'auc']].mean()
        assert len(pert) == r['datasets'] == 72
        for metric in ('accuracy', 'auc'):
            assert np.isclose(pert[metric].mean(), r[metric], atol=1e-12)
            assert np.isclose(g[g.rho == 0][metric].mean(), r['clean_' + metric], atol=1e-12)
    checked = 0
    for directory in (OUT / 'banks').iterdir():
        for rho in (0, 20, 50, 80, 100):
            records = [json.loads((directory / str(n) / f'{rho}.json').read_text())
                for n in (16, 24, 32, 48, 64)]
            for field in ('source_X_sha256', 'source_y_sha256', 'evaluation_X_sha256', 'input_sha256'):
                assert len({m[field] for m in records}) == 1
            for a, b in zip(records, records[1:]):
                assert set(a['selected_target_positions']).issubset(b['selected_target_positions'])
            for m in records:
                assert len(m['selected_target_positions']) == m['target_rows_in_context']
                assert m['native_context_rows'] == 256
                assert m['source_rows_in_native_context'] == 256 - m['target_rows_in_context']
                assert max(m['branch_context_rows']) <= 256
                assert not m['evaluation_labels_loaded_for_inference']
                checked += 1
    assert checked == 1800
    seal = json.loads((POLICIES / 'policy_seal.json').read_text())
    assert all(sha(POLICIES / p) == h for p, h in seal.items())
    highlight = table[table.context_target_rows == 48].copy()
    highlight['display_status'] = 'POST_HOC_48_ROW_DEVELOPMENT_HIGHLIGHT_ALL_GRID_POINTS_RETAINED'
    highlight.to_csv(OUT / 'highlight_48_context_rows.csv', index=False)
    display = []
    for method, budget, tag in [('A', 64, 'original_64_row_reference'),
            ('A', 48, 'same_48_row_budget_reference'), ('B', 48, '48_row_development_variant'),
            ('C', 48, '48_row_development_variant'), ('D', 48, '48_row_development_variant')]:
        v = table[(table.method == method) & (table.context_target_rows == budget)].iloc[0]
        display.append({'method': method, 'target_rows_in_inference_context': budget,
            'total_context_cap': 256, 'target_labels_used_for_fusion_calibration': 64,
            'status': tag, 'datasets': 72, 'seed': 0,
            'accuracy': float(v.accuracy), 'auc': float(v.auc),
            'clean_accuracy': float(v.clean_accuracy), 'clean_auc': float(v.clean_auc)})
    pd.DataFrame(display).to_csv(OUT / 'comparison_with_original_A_reference.csv', index=False)
    effect = []
    for method in ('A', 'B', 'C', 'D'):
        a = table[(table.method == method) & (table.context_target_rows == 64)].iloc[0]
        b = table[(table.method == method) & (table.context_target_rows == 48)].iloc[0]
        effect.append({'method': method, 'original_accuracy': float(a.accuracy),
            'context48_accuracy': float(b.accuracy), 'accuracy_change': float(b.accuracy - a.accuracy),
            'auc_change': float(b.auc - a.auc)})
    pd.DataFrame(effect).to_csv(OUT / 'changes_64_to_48.csv', index=False)
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Diagnose the abrupt source-only removal using gradual inference-context label budgets',
        'Setting': plan,
        'Results': json.loads(table.to_json(orient='records', double_precision=15)),
        'Analysis': {
            '48_row_highlight_is_post_hoc': True,
            'target_calibration_supervision_stays_64_not_48': True,
            'BCD_accuracy_changes_from64_to48': effect[1:],
            'D48_accuracy_above0_7': float(highlight[highlight.method == 'D'].accuracy.iloc[0] - .7),
            'same48_budget_A_reference_included': True,
            'no_independent_significance_or_paper_mechanism_validation_claim': True,
        },
        'Next Steps': 'User clarified approximate targets are sufficient. Preserve all five budgets without further threshold-driven search.',
    })
    atomic_json(OUT / 'reporting_amendment.json', {
        'previous_failure': 'numpy.int64 in experiment-log JSON serialization after all metrics and audit completed',
        'fix': 'convert already computed table to native JSON records; no inference or metric rerun',
        'script_sha256': sha(__file__), 'original_inference_script_unchanged': True,
        'original_source_files_unchanged': before,
    })
    assert all(sha(OUT / f) == h for f, h in before.items())
    assert (RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'report_completion.json', {
        'status': 'COMPLETE', 'all_25_aggregate_rows_independently_recomputed': True,
        'nested_budget_and_input_invariants_verified_conditions': checked,
        'full_grid_metrics_unchanged': True, 'frozen_original_policies_unchanged': True,
        'highlight_is_post_hoc_development_only': True, 'Markdown_bytes': 0,
    })
    print(pd.DataFrame(display).to_string(index=False), flush=True)
    print(pd.DataFrame(effect).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
