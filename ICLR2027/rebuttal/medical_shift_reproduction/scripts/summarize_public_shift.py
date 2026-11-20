"""Summarize frozen clinical comparisons; keep threshold diagnostics separate."""
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_public_shift import OUT, DATA, ROOT, metrics, paired, probability, write_json

TASKS = ['readmission_other_to_ER', 'sepsis_A_to_B', 'sepsis_B_to_A']


def holm(tests):
    ordered = sorted(tests, key=lambda item: item[1]['mcnemar_p'])
    previous = 0.
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1., max(previous, (len(ordered)-rank)*value['mcnemar_p']))
        value['holm_adjusted_p'] = adjusted
        value['significant_positive_holm05'] = bool(value['accuracy_delta'] > 0 and adjusted < .05)
        previous = adjusted


def main():
    rows, tests, diagnostics, diagnostic_tests = [], [], [], []
    complete = [task for task in TASKS if (OUT/task/'metrics.json').exists()]
    for task in complete:
        result = json.loads((OUT/task/'metrics.json').read_text())
        frozen = json.loads((OUT/task/'frozen_policies.json').read_text())
        policy = frozen['policies']
        for domain, data in result['domains'].items():
            for method, score in data['methods'].items():
                rows.append(dict(task=task, domain=domain, method=method,
                                 n=data['n'], positives=data['positives'],
                                 always_negative_accuracy=data['always_negative_accuracy'],
                                 specific_fallback_to_shared=policy['Specific'] == policy['Shared'], **score))
            if domain == 'ood':
                for name, values in data['adjacent_comparisons'].items():
                    tests.append((task+'/'+name, values))
        source = DATA/task
        with np.load(source/'member_predictions.npz') as z:
            bank = {k: z[k] for k in z.files}
        with np.load(source/'test_truth.npz') as z:
            truth = {k: z[k] for k in z.files}
        diagnostic = dict(role='DIAGNOSTIC_ONLY; not used for model or threshold selection',
                          threshold=.5, domains={})
        for domain in ['id', 'ood']:
            y = truth['y_'+domain]
            ps = {k: probability(bank['base_'+domain], bank['bank_'+domain], p) for k, p in policy.items()}
            scores = {k: metrics(y, v, .5) for k, v in ps.items()}
            comparisons = {n: paired(y, ps[a] >= .5, ps[b] >= .5) for n, a, b in
                           [('Shared_minus_Base', 'Base', 'Shared'),
                            ('Specific_minus_Shared', 'Shared', 'Specific')]}
            diagnostic['domains'][domain] = dict(methods=scores, comparisons=comparisons)
            for method, score in scores.items():
                diagnostics.append(dict(task=task, domain=domain, method=method,
                                        n=len(y), positives=int(y.sum()), **score))
            if domain == 'ood':
                for name, values in comparisons.items():
                    diagnostic_tests.append((task+'/'+name, values))
        write_json(OUT/task/'fixed_threshold_diagnostic.json', diagnostic)
    # Unfinished predeclared comparisons are conservatively included with p=1.
    for task in set(TASKS)-set(complete):
        for name in ['Shared_minus_Base', 'Specific_minus_Shared']:
            placeholder = dict(mcnemar_p=1., accuracy_delta=0., pending=True)
            tests.append((task+'/'+name, placeholder))
            diagnostic_tests.append((task+'/'+name, placeholder.copy()))
    holm(tests)
    holm(diagnostic_tests)
    pd.DataFrame(rows).to_csv(OUT/'aggregate_primary.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'aggregate_fixed_threshold_diagnostic.csv', index=False)
    write_json(OUT/'paired_primary_tests.json', dict(tests))
    write_json(OUT/'paired_fixed_threshold_diagnostic_tests.json', dict(diagnostic_tests))
    write_json(OUT/'status.json', dict(
        status='COMPLETE' if len(complete) == len(TASKS) else 'PARTIAL',
        completed_tasks=complete, remaining_tasks=sorted(set(TASKS)-set(complete)),
        original_paper_replication=False, development_result_rank_enforcement=False,
        heldout_target_labels_used_for_fitting=False, requested_ranking_guaranteed=False,
        requested_ranking_passed_primary=bool(len(complete) == len(TASKS) and all(
            row['significant_positive_holm05'] for _, row in tests)),
        software={p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'tabicl']},
        markdown_written=False))
    print(pd.DataFrame(rows)[['task', 'domain', 'method', 'accuracy', 'auc']].to_string(index=False))


if __name__ == '__main__':
    main()
