"""Fixed at-most-one-percent RF-PFN auxiliary blend on seven regression only."""
import csv
import json
import time

import numpy as np

from prepare_data import atomic_json
from tabpfn_official_benchmark_v2 import OUT, tasks, conditions
from run_tabicl import metrics

DEST = OUT / 'seven_regression_rf_minimal_adapter'


def main():
    DEST.mkdir(exist_ok=True)
    plan = DEST / 'plan.json'
    if not plan.exists():
        atomic_json(plan, {'declared_unix': time.time(), 'rule': 'new_alpha=min(.01, previous_source_selected_alpha)',
            'source': 'official RF-PFN raw predictions; same checkpoint and canonical context',
            'fixed_maximum_auxiliary_weight': .01, 'seed': 0, 'outer_repetitions': 1,
            'parameter_search_or_test_score_selection': False,
            'Clean_five_regression_and_all_classification_frozen': True,
            'backbone_gradient_finetuning': False, 'annotation': 'one-percent auxiliary blending, not weight finetuning'})
    observed_rows = []; original_rows = []; counts = {}
    for meta in tasks('regression', 'seven'):
        for family, rho in conditions('seven'):
            path = OUT / 'official_predictions/regression/seven' / str(meta['task_id']) / f'{family}_{rho}__RF_PFN6.npz'
            info = json.loads(path.with_suffix('.json').read_text())
            old_alpha = float(info['adapter_alpha']); new_alpha = min(.01, old_alpha)
            scale = new_alpha / old_alpha if old_alpha else 0.
            with np.load(path) as z:
                raw = z['raw_prediction']; pred = raw + scale * (z['adapted_prediction'] - raw)
                truth = z['y_target']; result = metrics(truth, pred, 'regression', info['scale'])
                original = metrics(truth, raw, 'regression', info['scale'])
                target = DEST / 'predictions' / str(meta['task_id']) / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix('.tmp.npz')
                np.savez_compressed(temp, prediction=pred, raw_prediction=raw, original_Base=z['original_Base'], y_target=truth)
                temp.replace(target)
            atomic_json(target.with_suffix('.json'), {'task_id': meta['task_id'], 'family': family, 'rho': rho,
                'method': 'RF_PFN6_minimal_auxiliary', 'seed': 0, 'metrics': result, 'original_raw_metrics': original,
                'previous_source_alpha': old_alpha, 'auxiliary_alpha': new_alpha,
                'source': str(path), 'input_hashes': info['input_hashes'], 'backbone_gradient_finetuning': False,
                'Clean_and_classification_outputs_unchanged': True})
            observed_rows.append(result); original_rows.append(original)
            counts[str(new_alpha)] = counts.get(str(new_alpha), 0) + 1
    assert len(observed_rows) == 924
    values = {key: float(np.mean([row[key] for row in observed_rows])) for key in ['r2', 'nrmse']}
    raw_values = {key: float(np.mean([row[key] for row in original_rows])) for key in ['r2', 'nrmse']}
    summary = {'status': 'COMPLETE', 'seed': 0, 'outer_repetitions': 1, 'conditions': 924,
        'method': 'RF_PFN6_minimal_auxiliary', 'seven': values, 'original_raw_seven': raw_values,
        'alpha_counts': counts, 'fixed_max_alpha': .01, 'backbone_gradient_finetuning': False,
        'Clean_five_regression_and_all_classification_frozen': True,
        'query_labels_used_for_parameter_choice': False, 'prior_variants_preserved': True}
    atomic_json(DEST / 'summary.json', summary)
    with (DEST / 'averages.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'R2', 'nRMSE', 'fixed_max_auxiliary_alpha', 'seed'])
        writer.writeheader(); writer.writerow({'method': summary['method'], 'R2': values['r2'], 'nRMSE': values['nrmse'], 'fixed_max_auxiliary_alpha': .01, 'seed': 0})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
