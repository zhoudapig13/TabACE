"""Replay provenance and descriptive branch quality of the fixed generic bank.

No fitting or family-specific selection. Branch tables are diagnostics on the
previously evaluated development cohort, not additional confirmatory tests.
"""
import json
import numpy as np
import pandas as pd

import shared_generic_pool_banks_v163 as bank
from prepare_data import ROOT, atomic_json
from run_tabicl import metrics


def main():
    source = bank.source
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert source.sha(bank.__file__) == plan['script_sha256']
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text())
        if c['track'] == 'perturbation']
    rows, count, references = [], 0, 0
    for c in cases:
        task, tid = c['task'], c['task_id']
        with np.load(c['input_file']) as inputs:
            assert 'y_target' not in inputs.files
            assert not set(inputs['source_indices']) & set(inputs['target_indices'])
        for path in sorted((bank.OUT/'banks'/task/str(tid)).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert source.sha(path) == meta['output_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            assert meta['context_rules_selected_by_task_only_not_by_family']
            assert meta['outcome_query_labels_opened'] is False
            with np.load(path) as data, np.load(meta['original_bank']) as original:
                np.testing.assert_array_equal(data['Base'], original['Base'])
                assert len(data['members']) == 5 and max(meta['context_rows']) <= 256
                for i, (rule, parent, index) in enumerate(bank.RULES[task]):
                    reference = meta['member_provenance'][i]
                    assert reference['rule'] == rule and reference['member_index'] == index
                    assert source.sha(reference['parent_bank']) == reference['parent_sha256']
                    assert str(bank.PARENTS[parent]) in reference['parent_bank']
                    with np.load(reference['parent_bank']) as p:
                        np.testing.assert_array_equal(data['members'][i], p['members'][index])
                    references += 1
                predictions = {'Base': data['Base'], 'Mean6':
                    np.concatenate([data['Base'][None], data['members']]).mean(axis=0)}
                predictions.update({rule: data['members'][i]
                    for i, (rule, _, _) in enumerate(bank.RULES[task])})
                # Outcome labels are opened only after provenance checks for scoring.
                with np.load(c['truth_file']) as truth:
                    y = truth['y_target']
                for name, value in predictions.items():
                    rows.append({'task': task, 'data_id': c['data_id'], 'family': meta['family'],
                        'rho': meta['rho'], 'method': name,
                        **metrics(y, value, task, max(float(np.std(y)), 1e-12))})
            count += 1
    assert count == 1327 and references == 6635
    frame = pd.DataFrame(rows)
    frame.to_csv(bank.OUT/'branch_diagnostic_detail.csv', index=False)
    output = []
    for (task, family, method), group in frame[frame.rho > 0].groupby(['task', 'family', 'method']):
        keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        units = group.groupby('data_id')[keys].mean()
        output.append({'task': task, 'family': family, 'method': method,
            'datasets': len(units), **{key: float(units[key].mean()) for key in keys}})
    table = pd.DataFrame(output)
    table.to_csv(bank.OUT/'branch_diagnostic_averages.csv', index=False)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(bank.OUT/'independent_provenance_audit.json', {
        'status': 'PASS', 'banks': count, 'parent_branches_replayed': references,
        'Base_bitwise_unchanged': True, 'rule_order_task_specific_not_family_specific': True,
        'extra_source_CV_compute_disclosed': True, 'all_descriptive_branch_results_retained': True,
        'diagnostic_is_not_independent_confirmation': True, 'Markdown_bytes': 0})
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
