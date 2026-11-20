"""Disjoint extra labeled development references, with unchanged v117 evaluation.

Classification references are original rows outside both fixed benchmark splits.
Regression references are unused rows from the existing clean training cache.
Final evaluation labels are never opened by this preparation script.
"""
import json
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import unseen_shared_robustness_v117 as source
from prepare_data import LOCAL_CC18, ROOT, RUN, atomic_json, protocol

OUT = RUN/'results/clean_reference_manifest_v268'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        'seed': 0, 'reference_rows_requested': 64,
        'explicit_extra_labeled_supervision': True,
        'benchmark_and_Base_unchanged': True,
        'evaluation_labels_opened': False,
        'context_row_cap': 256,
        'classification_reference_pool': 'original labeled rows excluding fixed source and evaluation row identities; only source-supported classes',
        'regression_reference_pool': 'unused rows of original clean source training cache',
        'reference_selection': 'uniform without replacement, seed0; no metric-based selection',
        'acceptance': {'each_of_three_perturbation_families': True,
                       'classification_accuracy_gain_at_least': .05,
                       'regression_r2_gain_at_least': .05},
        'AUC_and_nRMSE': 'reported without minimum improvement constraint',
        'development_adaptation_not_frozen_Shared_generalization': True,
        'script_sha256': source.sha(__file__),
    }
    existing = OUT/'plan.json'
    if existing.exists():
        assert json.loads(existing.read_text()) == plan
    else:
        atomic_json(existing, plan)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text())
             if c['track'] == 'perturbation']
    rows = []
    for c in cases:
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X, y, Q, sid, qid = (z[k] for k in
                ('X_source', 'y_source', 'X_target', 'source_indices', 'target_indices'))
        if c['task'] == 'classification':
            folder = LOCAL_CC18/str(c['task_id'])
            frame = pd.read_parquet(folder/'X.parquet')
            labels = pd.read_parquet(folder/'y.parquet').iloc[:, 0]
            valid = labels.notna().to_numpy()
            ids = np.flatnonzero(valid)
            allX, _ = protocol._numeric_matrix(frame.loc[valid])
            pos = {int(row): i for i, row in enumerate(ids)}
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in sid]], X)
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in qid]], Q)
            encoder = LabelEncoder().fit(labels.iloc[sid].to_numpy())
            np.testing.assert_array_equal(encoder.transform(labels.iloc[sid].to_numpy()), y)
            candidates = np.setdiff1d(ids, np.concatenate([sid, qid]))
            candidates = candidates[labels.iloc[candidates].isin(encoder.classes_).to_numpy()]
            count = min(64, len(candidates))
            selected = np.random.default_rng(0).choice(candidates, count, replace=False)
            RX = allX[[pos[int(i)] for i in selected]]
            Ry = encoder.transform(labels.iloc[selected].to_numpy()).astype(y.dtype)
            provenance = str(folder)
        else:
            assert source.sha(c['original_raw_cache']) == c['original_cache_sha256']
            with np.load(c['original_raw_cache']) as z:
                ids, allX, ally = z['source_indices'], z['X_source'], z['y_source']
            pos = {int(row): i for i, row in enumerate(ids)}
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in sid]], X)
            np.testing.assert_array_equal(ally[[pos[int(i)] for i in sid]], y)
            candidates = np.setdiff1d(ids, np.concatenate([sid, qid]))
            count = min(64, len(candidates))
            selected = np.random.default_rng(0).choice(candidates, count, replace=False)
            positions = [pos[int(i)] for i in selected]
            RX, Ry = allX[positions], ally[positions]
            provenance = c['original_raw_cache']
        assert count == 64, (c['task'], c['task_id'], count)
        assert set(map(int, selected)).isdisjoint(map(int, sid))
        assert set(map(int, selected)).isdisjoint(map(int, qid))
        assert len(set(map(int, selected))) == count
        output = OUT/'references'/c['task']/f"{c['task_id']}.npz"
        source.npwrite(output, X_reference=RX, y_reference=Ry, reference_indices=selected)
        meta = {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')}
        meta.update(reference_rows=count, reference_file=str(output),
                    reference_sha256=source.sha(output), provenance=provenance,
                    reference_row_ids_sha256=protocol.array_sha256(selected),
                    benchmark_input_sha256=c['input_sha256'],
                    disjoint_from_source_and_final_evaluation=True,
                    clean_source_features_reconstructed_exactly=True,
                    extra_labels=64, final_evaluation_labels_opened=False)
        atomic_json(output.with_suffix('.json'), meta)
        rows.append(meta)
        print('REFERENCE_READY', c['task'], c['task_id'], count, flush=True)
    atomic_json(OUT/'reference_manifest.json', rows)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT/'audit.json', {'status': 'PASS', 'datasets': len(rows),
        'classification': sum(r['task']=='classification' for r in rows),
        'regression': sum(r['task']=='regression' for r in rows),
        'reference_rows_per_dataset': 64, 'all_row_identities_disjoint': True,
        'original_clean_source_features_replayed': True,
        'final_evaluation_labels_opened': False, 'Markdown_bytes': 0})


if __name__ == '__main__':
    main()
