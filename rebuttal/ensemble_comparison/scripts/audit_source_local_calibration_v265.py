"""Refit source-only ensemble heads and replay derived query forecasts.

No final-query outcome is opened. Source labels are explicitly used for the
small local head fit; the shared source-CV artifact is checked by sealed hash.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_source_local_calibration_banks_v264 as bank


def check_case(c):
    assert bank.source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        y = data['y_source']
    count = 0
    for path in sorted((bank.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert bank.source.sha(meta['source_CV_file']) == meta['source_CV_sha256']
        assert meta['source_label_sha256'] == bank.protocol.array_sha256(y)
        assert not meta['task_query_labels_opened'] and meta['no_new_query_TFM_forwards']
        with np.load(meta['source_CV_file']) as data:
            Psource, mask = data['predictions'], data['scored']
        assert mask.sum() == meta['scored_source_rows']
        with np.load(path) as data, np.load(meta['parent_bank']) as parent:
            np.testing.assert_array_equal(data['Base'], parent['Base'])
            np.testing.assert_array_equal(data['members'][:5], parent['members'][:5])
            np.testing.assert_array_equal(data['UncalibratedRepair_predictions'], parent['members'][5:])
            np.testing.assert_array_equal(data['Canonical_native8'], parent['Canonical_native8'])
            assert data['members'].shape[0] == 7 and np.isfinite(data['members']).all()
            mu, sigma = float(data['source_mean']), float(data['source_scale'])
            Pquery = np.concatenate([parent['Base'][None], parent['members'][5:]], axis=0)
            for i, kind in enumerate(('convex', 'stack')):
                fitted = bank.local.fit(Psource[:, mask], y[mask], c['task'], mu, sigma, kind)
                assert fitted == meta['local_models'][i]
                prediction = bank.local.predict(Pquery, meta['local_models'][i], c['task'], mu, sigma)
                np.testing.assert_array_equal(prediction, data['members'][5+i])
            if c['task'] == 'classification':
                np.testing.assert_allclose(data['members'].sum(2), 1., atol=1e-6)
        count += 1
    return count


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    for path, key in [(bank.__file__, 'script_sha256'), (bank.local.__file__, 'local_head_sha256'),
        (bank.parent.OUT/'plan.json', 'parent_plan_sha256'), (bank.source_cv.OUT/'plan.json', 'source_CV_plan_sha256'),
        (bank.source_cv.OUT/'independent_source_CV_audit.json', 'source_CV_audit_sha256')]:
        assert bank.source.sha(path) == plan[key]
    assert json.loads((bank.source_cv.OUT/'independent_source_CV_audit.json').read_text())['status'] == 'PASS'
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    counts = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in cases]):
            counts.append(future.result())
    assert len(counts) == 107 and sum(counts) == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': sum(counts), 'locally_calibrated_outputs_replayed': 2*sum(counts),
        'source_only_heads_refitted': 2*sum(counts), 'Base_and_first_five_parent_outputs_unchanged': True,
        'local_training_labels_are_existing_source_labels': True, 'extra_true_label_rows': 0,
        'task_query_labels_opened': False, 'new_query_TFM_forwards': 0,
        'local_hyperparameters_fixed_before_query_scoring': True,
        'local_source_supervision_is_explicit': True, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
