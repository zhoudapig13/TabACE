"""Fixed weaker-adaptation sensitivity; existing Clean/classification stay frozen."""
import csv
import json
from pathlib import Path

import numpy as np

from prepare_data import atomic_json
from tabpfn_official_benchmark_v2 import OUT, tasks, conditions
from run_tabicl import metrics

DEST = OUT / 'seven_regression_reduced_adaptation'
METHODS = ['Native6', 'PHE6', 'RF_PFN6']
SCALES = [0., .25, .5]


def main():
    plan = json.loads((DEST / 'plan_and_freeze.json').read_text())
    assert plan['primary_reduction_scale'] == .5
    rows = []; count = 0; grouped = {(method, scale): [] for method in METHODS for scale in SCALES}
    for meta in tasks('regression', 'seven'):
        for family, rho in conditions('seven'):
            for method in METHODS:
                source = OUT / 'official_predictions/regression/seven' / str(meta['task_id']) / f'{family}_{rho}__{method}.npz'
                observed = json.loads(source.with_suffix('.json').read_text())
                with np.load(source) as z:
                    raw, old, truth = z['raw_prediction'], z['adapted_prediction'], z['y_target']
                    arrays = {'raw_prediction': raw, 'previous_adapted_prediction': old,
                              'original_Base': z['original_Base'], 'y_target': truth}
                    values = {}
                    for scale in SCALES:
                        pred = raw + scale * (old - raw)
                        arrays[f'prediction_scale_{scale:g}'] = pred
                        result = metrics(truth, pred, 'regression', observed['scale'])
                        values[str(scale)] = result
                        grouped[(method, scale)].append(result)
                    target = DEST / 'predictions' / str(meta['task_id']) / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix('.tmp.npz'); np.savez_compressed(temp, **arrays); temp.replace(target)
                    atomic_json(target.with_suffix('.json'), {'task_id': meta['task_id'], 'family': family,
                        'rho': rho, 'method': method, 'seed': 0, 'metrics_by_scale': values,
                        'previous_auxiliary_alpha': observed['adapter_alpha'],
                        'new_auxiliary_alpha_by_scale': {str(s): s * observed['adapter_alpha'] for s in SCALES},
                        'source': str(source), 'input_hashes': observed['input_hashes'],
                        'original_Clean_and_classification_unchanged': True,
                        'query_labels_used_to_choose_scale': False, 'all_fixed_scales_preserved': True})
                count += 1
    for method in METHODS:
        for scale in SCALES:
            values = grouped[(method, scale)]
            assert len(values) == 924
            row = {'method': method, 'adaptation_scale': scale, 'primary_requested_variant': scale == .5,
                   'r2': float(np.mean([v['r2'] for v in values])),
                   'nrmse': float(np.mean([v['nrmse'] for v in values])), 'conditions': len(values)}
            rows.append(row); print(json.dumps(row), flush=True)
    # Original outputs are read only. Verify every existing classification file
    # from the user's freeze point retained its recorded size and modification time.
    for relative, size, mtime in plan['classification_completed_file_index']:
        stat = (OUT / relative).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (size, mtime), relative
    assert count == 2772
    atomic_json(DEST / 'summary.json', {'status': 'COMPLETE', 'seed': 0, 'outer_repetitions': 1,
        'records': rows, 'fixed_primary_scale': .5,
        'classification_freeze_index_verified': True, 'original_outputs_overwritten': False,
        'interpretation': 'Post-hoc seven-regression adaptation-strength sensitivity, all declared scales reported',
        'historical_Shared_not_a_parameter_selection_rule': True})
    with (DEST / 'averages.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('SEVEN_ONLY_REDUCED_ADAPTATION_COMPLETE', count, flush=True)


if __name__ == '__main__':
    main()
