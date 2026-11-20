"""Reconstruct every source-label edit decision and source-probe partition."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_label_row_guard_banks_v242 as bank
from run_tabicl import CHECKPOINTS


def check_case(c):
    assert bank.source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    fitted = bank.forest.constructor.fit_target(Q0, numeric)
    count, edits = 0, 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in bank.source.novel.STRENGTHS]:
        path = bank.OUT/'banks'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert meta['original_input_sha256'] == c['input_sha256']
        assert not meta['query_task_labels_opened'] and meta['new_query_TFM_calls'] == 2
        if rho:
            X, y, Q, _ = bank.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        else:
            X, y, Q = X0, y0, Q0
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(Q, Q0)
        repaired = [v['X'] for v in bank.forest.constructor.repair_source(X, fitted)]
        assert [bank.protocol.array_sha256(v) for v in repaired] == meta['raw_repaired_X_sha256']
        partitions, mandatory = bank.guard.folds(y, c['task'])
        mask = np.ones(len(y), dtype=bool)
        mask[mandatory] = False
        with np.load(path) as data, np.load(meta['parent_bank']) as parent:
            np.testing.assert_array_equal(data['Base'], parent['Base'])
            np.testing.assert_array_equal(data['members'][:5], parent['members'])
            np.testing.assert_array_equal(data['Canonical_native8'], parent['Canonical_native8'])
            assert data['members'].shape[0] == 7 and np.isfinite(data['members']).all()
            np.testing.assert_array_equal(data['SourceProbe_scorable'], mask)
            prediction, loss = data['SourceProbe_predictions'], data['SourceProbe_loss']
            assert np.isnan(prediction[:, ~mask]).all() and np.isnan(loss[:, ~mask]).all()
            assert np.isfinite(prediction[:, mask]).all()
            if c['task'] == 'classification':
                np.testing.assert_allclose(prediction[:, mask].sum(2), 1., atol=1e-6)
                expected = -np.log(np.clip(prediction[:, np.flatnonzero(mask), y[mask].astype(int)], 1e-12, 1.))
            else:
                expected = ((prediction[:, mask]-y[mask])/max(float(y.std()), 1e-12))**2
            np.testing.assert_allclose(loss[:, mask], expected, atol=1e-12, rtol=1e-12)
            accepted = (loss[1:] < loss[0][None]) & mask[None]
            np.testing.assert_array_equal(accepted, data['SourceRow_edit_accepted'])
            for i, flag in enumerate(accepted):
                guarded = X.copy()
                guarded[flag] = repaired[i][flag]
                assert bank.protocol.array_sha256(guarded) == meta['context_X_sha256'][i]
                assert np.flatnonzero(flag).tolist() == meta['accepted_source_row_ids'][i]
                np.testing.assert_array_equal(guarded[mandatory], X[mandatory])
                edits += int(flag.sum())
        expected_records = []
        for train, valid in partitions:
            assert not set(train) & set(valid) and set(mandatory).issubset(train)
            probe = np.concatenate([v[valid] for v in [X]+repaired], axis=0)
            expected_records.append({'training_rows': train.tolist(), 'validation_rows': valid.tolist(),
                'probe_order': ['observed', 'mild_repair', 'full_repair'],
                'training_X_sha256': bank.protocol.array_sha256(X[train]),
                'training_y_sha256': bank.protocol.array_sha256(y[train]),
                'probe_X_sha256': bank.protocol.array_sha256(probe),
                'callback_received_validation_labels': False})
        assert expected_records == meta['source_probe_partitions']
        assert meta['source_probe_TFM_calls'] == len(partitions)
        assert bank.protocol.array_sha256(y) == meta['source_y_sha256']
        assert bank.protocol.array_sha256(Q) == meta['query_X_sha256']
        count += 1
    return {'conditions': count, 'accepted_source_row_edits': edits}


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    for path, key in [(bank.__file__, 'script_sha256'), (bank.guard.__file__, 'row_guard_sha256'),
        (bank.guard.folds.__code__.co_filename, 'source_fold_sha256'),
        (bank.forest.constructor.__file__, 'forest_constructor_sha256'),
        (bank.parent.OUT/'plan.json', 'parent_plan_sha256'), (bank.source.OUT/'plan.json', 'original_plan_sha256')]:
        assert bank.source.sha(path) == plan[key]
    for name, digest in json.loads((bank.source.OUT/'plan.json').read_text())['checkpoints'].items():
        assert bank.source.sha(CHECKPOINTS/name) == digest
    manifest = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in manifest]):
            results.append(future.result())
            if len(results) % 10 == 0:
                print('ROW_GUARD_AUDIT', len(results), len(manifest), flush=True)
    count = sum(r['conditions'] for r in results)
    assert len(results) == 107 and count == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count, 'repaired_contexts_reconstructed': 2*count,
        'accepted_source_row_edits_recomputed': sum(r['accepted_source_row_edits'] for r in results),
        'Base_and_all_five_parent_alternatives_unchanged': True,
        'source_probe_training_excludes_each_scored_row_and_its_label': True,
        'source_training_features_observed_not_label_edited': True,
        'source_label_losses_and_row_edit_decisions_recomputed': True,
        'all_three_probe_input_views_verified': True,
        'query_task_labels_opened': False, 'TFM_inference_not_repeated_by_this_audit': True,
        'checkpoint_hashes_unchanged': True, 'constructor_extension_not_original_Shared': True,
        'source_labels_explicitly_used_for_training_feature_edits': True, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
