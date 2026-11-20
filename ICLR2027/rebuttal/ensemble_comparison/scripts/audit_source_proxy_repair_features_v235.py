"""Replay source-CV data flow and source-label losses, without query labels."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import source_proxy_repair_features_v234 as run
from run_tabicl import CHECKPOINTS


def check_case(c):
    assert run.source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    model = run.bank.constructor.fit_target(Q0, numeric)
    conditions, scored_rows = 0, 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in run.source.novel.STRENGTHS]:
        path = run.OUT/'features'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert run.source.sha(path) == meta['output_sha256']
        assert not meta['task_query_labels_opened'] and not meta['task_query_predictions_produced']
        assert meta['original_input_sha256'] == c['input_sha256']
        if rho:
            X, y, Q, _ = run.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        else:
            X, y, Q = X0, y0, Q0
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(Q, Q0)
        views = [X]+[v['X'] for v in run.bank.constructor.repair_source(X, model)]
        assert [run.protocol.array_sha256(v) for v in views] == meta['source_context_X_sha256']
        assert run.protocol.array_sha256(y) == meta['source_labels_sha256']
        partitions, mandatory = run.folds(y, c['task'])
        expected_mask = np.ones(len(y), dtype=bool)
        expected_mask[mandatory] = False
        records = []
        for train, valid in partitions:
            assert not set(train) & set(valid)
            assert set(mandatory).issubset(train)
            for i, view in enumerate(views):
                records.append({'context_index': i, 'training_rows': train.tolist(), 'validation_rows': valid.tolist(),
                    'training_X_sha256': run.protocol.array_sha256(view[train]),
                    'training_y_sha256': run.protocol.array_sha256(y[train]),
                    'validation_proxy_X_sha256': run.protocol.array_sha256(views[2][valid]),
                    'callback_receives_validation_labels': False})
        assert records == meta['source_CV_expected_call_records']
        with np.load(path) as data:
            np.testing.assert_array_equal(data['scored'], expected_mask)
            predictions, loss = data['predictions'], data['loss']
            assert np.isnan(predictions[:, ~expected_mask]).all()
            assert np.isnan(loss[:, ~expected_mask]).all()
            assert np.isfinite(predictions[:, expected_mask]).all()
            if c['task'] == 'classification':
                np.testing.assert_allclose(predictions[:, expected_mask].sum(2), 1., atol=1e-6)
                truth_probability = predictions[:, np.flatnonzero(expected_mask), y[expected_mask].astype(int)]
                expected_loss = -np.log(np.clip(truth_probability, 1e-12, 1.))
            else:
                expected_loss = ((predictions[:, expected_mask]-y[expected_mask])/max(float(y.std()), 1e-12))**2
            np.testing.assert_allclose(loss[:, expected_mask], expected_loss, atol=1e-12, rtol=1e-12)
            average = loss[:, expected_mask].mean(1) if expected_mask.any() else np.zeros(3)
            gains = average[0]-average[1:]
            np.testing.assert_array_equal(gains, data['source_proxy_gain'])
            np.testing.assert_array_equal(average, meta['source_proxy_loss_mean_observed_mild_full'])
            np.testing.assert_array_equal(gains, meta['source_proxy_gain_mild_full'])
            if meta['pilot_reuse'] is not None:
                old = meta['pilot_reuse']
                assert run.source.sha(old['path']) == old['sha256']
                with np.load(old['path']) as cached:
                    for key in ('predictions', 'loss', 'scored'):
                        np.testing.assert_array_equal(data[key], cached[key])
        assert meta['source_validation_TFM_calls_required'] == len(records)
        assert meta['source_validation_TFM_calls_new'] == (0 if meta['pilot_reuse'] else len(records))
        conditions += 1
        scored_rows += int(expected_mask.sum())
    return {'conditions': conditions, 'scored_source_rows_over_conditions': scored_rows}


def main():
    assert json.loads((run.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((run.OUT/'plan.json').read_text())
    for path, key in [(run.__file__, 'script_sha256'), (run.proxy.__file__, 'proxy_sha256'),
        (run.folds.__code__.co_filename, 'source_fold_implementation_sha256'),
        (run.bank.constructor.__file__, 'constructor_sha256'), (run.bank.OUT/'plan.json', 'feature_bank_plan_sha256'),
        (run.PILOT/'plan.json', 'pilot_plan_sha256'), (run.source.OUT/'plan.json', 'original_plan_sha256')]:
        assert run.source.sha(path) == plan[key]
    for name, digest in json.loads((run.source.OUT/'plan.json').read_text())['checkpoints'].items():
        assert run.source.sha(CHECKPOINTS/name) == digest
    manifest = [c for c in json.loads((run.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in manifest]):
            results.append(future.result())
            if len(results) % 10 == 0:
                print('SOURCE_PROXY_AUDIT', len(results), len(manifest), flush=True)
    count = sum(r['conditions'] for r in results)
    assert len(results) == 107 and count == 1327
    assert (run.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'cases': len(results), 'conditions': count,
        'all_three_contexts_share_each_source_validation_input': True,
        'held_out_source_labels_excluded_from_TFM_callback': True,
        'source_label_losses_and_gains_independently_recomputed': True,
        'all_source_fold_training_and_proxy_validation_hashes_replayed': True,
        'rare_source_rows_always_training_and_unscored': True,
        'task_query_labels_opened': False, 'TFM_inference_not_repeated_by_this_audit': True,
        'checkpoint_hashes_unchanged': True, 'Markdown_bytes': 0}
    run.atomic_json(run.OUT/'independent_source_CV_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
