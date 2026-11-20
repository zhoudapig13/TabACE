"""Inspect DID-level gains of sealed inner-selected models using calibration IDs.

No newly fitted model or primary hypothesis. The evaluated outer dataset IDs
are excluded before loading labels for each diagnostic policy.
"""
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd

import shared_raw_covariance_calibration_v260 as run

OUT = run.OUT/'inner_gain_stability_v262'


def main():
    assert json.loads((run.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ('classification', 'regression'):
        for fold in range(5):
            cells = run.load_cells(task, exclude_fold=fold)
            for mode, family in [('pooled', 'all')]+[('lomo', f) for f in run.FAMILIES]:
                path = run.OUT/'policies'/task/str(fold)/mode/f'{family}.json'
                policy = json.loads(path.read_text())
                values = [c for c in cells if mode == 'pooled' or c['family'] != family]
                trial = next(t for t in policy['trials'] if t['specification'] == policy['selected_specification'])
                by_family = {}
                for inner in trial['inner']:
                    validation = [c for c in values if c['fold'] == inner['fold'] and c['family'] != 'clean']
                    for c in validation:
                        assert c['data_id'] not in inner['training_data_ids']
                        pred = run.heads.predict({k: v for k, v in c.items() if k != 'y'}, inner['model'])
                        a, b = run.heads.fast.metric(c, pred)[0], run.heads.fast.metric(c, c['P'][0])[0]
                        by_family.setdefault(c['family'], {}).setdefault(c['data_id'], []).append(a-b)
                for f, units in by_family.items():
                    gains = np.asarray([np.mean(v) for v in units.values()])
                    mean = float(gains.mean())
                    se = float(gains.std(ddof=1)/np.sqrt(len(gains))) if len(gains) > 1 else 0.
                    assert np.isclose(mean, trial['family_gains'][f], atol=1e-12)
                    rows.append({'task': task, 'outer_fold': fold, 'mode': mode, 'family': f,
                        'excluded_family': family, 'datasets': len(gains), 'mean_gain': mean,
                        'gain_SE': se, 'one_SE_adjusted_gain': mean-se,
                        'DID_positive_fraction': float((gains > 0).mean()),
                        'minimum_DID_gain': float(gains.min()), 'maximum_DID_gain': float(gains.max())})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT/'selected_model_inner_stability.csv', index=False)
    summary = []
    for (task, mode), frame in detail.groupby(['task', 'mode']):
        summary.append({'task': task, 'mode': mode, 'policy_family_cells': len(frame),
            'positive_mean_cells': int((frame.mean_gain > 0).sum()),
            'positive_one_SE_cells': int((frame.one_SE_adjusted_gain > 0).sum()),
            'mean_gain_SE': float(frame.gain_SE.mean())})
    run.atomic_json(OUT/'diagnostic_scope.json', {
        'status': 'COMPLETE', 'script_sha256': run.original.sha(__file__),
        'source_policy_seal_sha256': run.original.sha(run.OUT/'policy_seal.json'),
        'source_fold_exclusions_respected_for_each_policy': True,
        'no_new_fits_or_primary_hypothesis_tests': True,
        'one_SE_adjustment_is_descriptive_not_a_confidence_interval': True,
        'summary': summary, 'Markdown_bytes': (run.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
