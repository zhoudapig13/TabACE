"""Post-hoc diagnosis only; query truths never enter a fitted/deployed rule.

Compare fixed v210 fusion with the canonical None8 control and measure the
specific confident-error / extreme-branch failure modes raised by reviewers.
The oracle statistics below are explicitly non-deployable upper bounds.
"""
import json
import os

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
import pandas as pd

import original_shared_none8_gate_calibration_v210 as run
from prepare_data import atomic_json

OUT = run.RUN/'results/original_shared_failure_diagnostic_v213'


def main():
    assert json.loads((run.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    assert (run.OUT/'policy_seal.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ('classification', 'regression'):
        for cell in run.load_cells(task):
            if cell['family'] == 'clean':
                continue
            policy_path = run.OUT/'policies'/task/str(cell['fold'])/'pooled/all.json'
            model = json.loads(policy_path.read_text())['model']
            pred = run.heads.predict({k: v for k, v in cell.items() if k != 'y'}, model)
            row = {k: cell[k] for k in ('task', 'data_id', 'task_id', 'family', 'rho')}
            baseline = run.metrics(cell['y'], cell['native8_base'], task, max(float(np.std(cell['y'])), 1e-12))
            adapted = run.metrics(cell['y'], pred, task, max(float(np.std(cell['y'])), 1e-12))
            primary = 'auc' if task == 'classification' else 'r2'
            row['Shared_minus_None8_primary'] = adapted[primary]-baseline[primary]
            row['selected_base_fallback'] = float(model['kind'] == 'base')
            view = run.heads.view(cell, model['backend'])
            f = run.heads.fusion.features(view, 5)
            if task == 'classification':
                correct = view['base'].argmax(1) == cell['y']
                confident_wrong = (f['U'] <= .25) & ~correct
                member_correct = (view['bank'].argmax(2) == cell['y'][None]).any(0)
                disagreement = .5*np.abs(view['bank']-view['base'][None]).sum(2).max(0)
                row.update(confident_wrong_fraction=float(confident_wrong.mean()),
                    oracle_member_rescues_confident_wrong_fraction=float((confident_wrong & member_correct).mean()),
                    disagreement_exceeds_entropy_fraction=float((disagreement > f['U']).mean()),
                    confident_wrong_repairable_with_disagreement_fraction=float((confident_wrong & member_correct & (disagreement > f['U'])).mean()))
            else:
                bank = f['M']
                middle = np.median(bank, axis=1)
                row.update(mean_branch_range_source_sigma=float(np.ptp(bank, axis=1).mean()),
                    branch_farther_than_3source_sigma_from_median_fraction=float((np.abs(bank-middle[:, None]).max(1) > 3).mean()))
                if model['kind'] == 'base':
                    row.update(mean_gate_alpha=0., mean_weighted_mean_minus_median_source_sigma=0.)
                else:
                    p = np.asarray(model['fusion_policy']['parameters'])
                    tau = np.exp(p[-2])
                    agreement = np.exp(-f['D']/tau)
                    logits = p[:5]+agreement*p[5:10]
                    weight = np.exp(logits-logits.max(1, keepdims=True))
                    weight /= weight.sum(1, keepdims=True)
                    mixed = (weight*bank).sum(1)
                    alpha = np.clip(model['gain']*p[-1]*(1-np.exp(-np.abs(mixed-f['B'])/tau))*(weight*agreement).sum(1), 0, 1)
                    row.update(mean_gate_alpha=float(alpha.mean()),
                        mean_weighted_mean_minus_median_source_sigma=float(np.abs(mixed-middle).mean()))
            rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT/'condition_diagnostics.csv', index=False)
    columns = [k for k in detail if k not in ('task', 'data_id', 'task_id', 'family', 'rho')]
    units = detail.groupby(['task', 'family', 'data_id'])[columns].mean()
    table = units.groupby(['task', 'family']).mean().reset_index()
    table.to_csv(OUT/'family_diagnostics.csv', index=False)
    atomic_json(OUT/'diagnostic_scope.json', {
        'status': 'COMPLETE', 'source_policy_seal_sha256': run.original.sha(run.OUT/'policy_seal.json'),
        'script_sha256': run.original.sha(__file__), 'conditions': len(detail),
        'no_new_fitted_models_or_hypothesis_tests': True,
        'query_labels_used_for_posthoc_error_analysis_only': True,
        'oracle_member_rescue_is_not_deployable_or_a_proposed_method': True,
        'thresholds_fixed_before_diagnostic': {'normalized_entropy': .25, 'branch_distance_source_sigma': 3.},
        'dataset_then_strength_balanced_summary': True,
        'Markdown_bytes': (run.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size,
    })
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
