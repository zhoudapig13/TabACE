"""Single-seed OligoICP: grouped subset sampling and label-free IQR selection.

The paper's molecular target groups are unavailable in this generic benchmark.
We explicitly adapt them to twenty source-only KMeans groups. Keep both the
60->6 matched-retained-count variant and the original-count 400->40 variant.
"""
import argparse
import json
import logging
import time
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans

from prepare_data import atomic_json, protocol
from tabpfn_official_benchmark_v2 import OUT, tasks, conditions, frozen_input
from tabpfn_official_methods_v2 import adapter
from run_tabpfn import model, aligned
from run_tabicl import metrics


def grouping(X):
    Z = (X - np.median(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-6)
    labels = KMeans(n_clusters=min(20, len(X)), n_init=10, random_state=0).fit_predict(Z)
    active = np.unique(labels)
    return labels, active


def candidate_spec(task_id, track, member, labels, active):
    streams = np.random.SeedSequence([0, int(task_id), int(track == 'seven'), member]).spawn(2)
    rng = np.random.default_rng(streams[0])
    count = int(rng.integers(1, len(active) + 1))
    groups = rng.choice(active, count, replace=False)
    rows = np.flatnonzero(np.isin(labels, groups))
    seed = int(streams[1].generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))
    return rows, groups, seed


def infer(estimator, X, y, Q, seed):
    if float(np.std(y)) < 1e-12 or np.all(np.ptp(X.astype(float), axis=0) == 0):
        return np.full(len(Q), float(np.mean(y))), np.full(len(Q), float(np.quantile(y, .85) - np.quantile(y, .15))), 0
    estimator.random_state = seed
    estimator.fit(X, y)
    values = estimator.predict(Q, output_type='main', quantiles=[.15, .85])
    pred = np.asarray(values['mean'], dtype=float).reshape(-1)
    low, high = (np.asarray(v, dtype=float).reshape(-1) for v in values['quantiles'])
    return pred, np.maximum(high - low, 0), 1


def run_cell(meta, track, family, rho, estimator, auxiliary, policy, total):
    folder = OUT / 'oligo_predictions' / track / str(meta['task_id']) / f'{family}_{rho}'
    done_path = folder / f'results_{total}.json'
    if done_path.exists():
        return
    data, info, base = frozen_input('regression', track, meta, family, rho)
    X, y, Q, truth = (data[k] for k in ['X_source', 'y_source', 'X_target', 'y_target'])
    hashes = {key: protocol.array_sha256(value) for key, value in data.items()}
    folder.mkdir(parents=True, exist_ok=True)
    progress = folder / 'progress.json'
    if progress.exists():
        state = json.loads(progress.read_text())
        assert state['input_hashes'] == hashes and state['seed'] == 0
        start = state['completed_candidates']
        labels = np.load(folder / 'group_labels.npy')
        active = np.unique(labels)
        arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r+') for name in ['means', 'iqrs', 'seeds', 'group_masks', 'row_counts', 'seconds', 'calls']}
    else:
        labels, active = grouping(X)
        np.save(folder / 'group_labels.npy', labels)
        specs = {'means': (np.float64, (400, len(Q))), 'iqrs': (np.float64, (400, len(Q))),
            'seeds': (np.uint32, (400,)), 'group_masks': (np.bool_, (400, 20)),
            'row_counts': (np.int32, (400,)), 'seconds': (np.float64, (400,)), 'calls': (np.int32, (400,))}
        arrays = {name: np.lib.format.open_memmap(folder / (name + '.npy'), mode='w+', dtype=dtype, shape=shape) for name, (dtype, shape) in specs.items()}
        start = 0
    for member in range(start, total):
        rows, groups, seed = candidate_spec(meta['task_id'], track, member, labels, active)
        tick = time.perf_counter()
        pred, iqr, calls = infer(estimator, X[rows], y[rows], Q, seed)
        assert np.isfinite(pred).all() and np.isfinite(iqr).all()
        arrays['means'][member], arrays['iqrs'][member] = pred, iqr
        arrays['seeds'][member] = seed
        arrays['group_masks'][member] = False
        arrays['group_masks'][member, groups] = True
        arrays['row_counts'][member] = len(rows)
        arrays['seconds'][member] = time.perf_counter() - tick
        arrays['calls'][member] = calls
        if (member + 1) % 20 == 0 or member + 1 == total:
            for value in arrays.values():
                value.flush()
            atomic_json(progress, {'seed': 0, 'outer_repetitions': 1, 'completed_candidates': member + 1,
                'input_hashes': hashes, 'query_labels_used_for_candidate_selection': False})
            print('OLIGO_CANDIDATES', track, meta['dataset'], family, rho, member + 1, flush=True)
    selected = np.argsort(arrays['iqrs'][:total].mean(1), kind='stable')[:total // 10]
    raw = arrays['means'][selected].mean(0)
    full = arrays['means'][:total].mean(0)
    auxiliary_pred = aligned(auxiliary, Q, 'regression', None)
    adapted = (1 - policy['alpha']) * raw + policy['alpha'] * auxiliary_pred
    mean_adapted = (1 - policy['alpha']) * full + policy['alpha'] * auxiliary_pred
    path = folder / f'results_{total}.npz'
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, raw_prediction=raw, adapted_prediction=adapted,
        full_ensemble_mean=full, full_ensemble_mean_adapted=mean_adapted,
        original_Base=base, y_target=truth, selected_indices=selected)
    temp.replace(path)
    atomic_json(done_path, info | {'status': 'COMPLETE', 'seed': 0, 'outer_repetitions': 1,
        'method': f'OligoICP_{total}to{total // 10}_groupadapted', 'candidate_evaluations': total,
        'retained_members': total // 10, 'raw_metrics': metrics(truth, raw, 'regression', info['scale']),
        'adapted_metrics': metrics(truth, adapted, 'regression', info['scale']),
        'equal_candidate_budget_full_mean_metrics': metrics(truth, full, 'regression', info['scale']),
        'equal_candidate_budget_full_mean_adapted_metrics': metrics(truth, mean_adapted, 'regression', info['scale']),
        'selected_indices': selected.tolist(), 'input_hashes': hashes, 'quantiles': [.15, .85],
        'candidate_groups': 'source-only KMeans, at most20; k uniform1..20 then select k groups without replacement',
        'grouping_is_generic_tabular_adaptation': True, 'original_molecular_target_groups_used': False,
        'TabPFN_v3_instead_of_paper_v2': True, 'benchmark_query_labels_used_for_training_or_selection': False,
        'adapter_alpha': policy['alpha'], 'extra_clean_source_TRAIN_supervision': True,
        'auxiliary_prediction_weight_nonzero': policy['alpha'] > 0,
        'backbone_gradient_finetuning': False, 'actual_predict_api_calls': int(arrays['calls'][:total].sum()),
        'candidate_seconds': float(arrays['seconds'][:total].sum()), 'includes_all_rejected_candidate_costs': True})
    print('OLIGO_CONDITION_COMPLETE', total, track, meta['dataset'], family, rho, flush=True)


def run(shard, shards, total, limit=0, per_cell_limit=0):
    torch.set_num_threads(1); logging.disable(logging.INFO); warnings.filterwarnings('ignore')
    work = [(track, meta) for track in ['five', 'seven'] for meta in tasks('regression', track)][shard::shards]
    if limit:
        work = work[:limit]
    estimator = model('regression', 'cuda:0')
    for track, meta in work:
        # The main official-method workers create this common policy first.
        # Waiting avoids concurrent writes/retraining of one source adapter.
        policy_path = OUT / 'source_adapters/regression' / track / str(meta['task_id']) / 'policy.json'
        deadline = time.monotonic() + 1800
        while not policy_path.exists():
            worker_state = OUT / 'regression_methods_workers.json'
            if worker_state.exists() and json.loads(worker_state.read_text())['status'] == 'FAILED':
                raise RuntimeError('Resolve official regression worker failures before continuing dependent OligoICP adapters')
            if time.monotonic() > deadline:
                raise TimeoutError(f'Source adapter not ready: {track} {meta["dataset"]}')
            time.sleep(5)
        auxiliary, policy = adapter('regression', track, meta)
        grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
        if per_cell_limit:
            grid = grid[:per_cell_limit]
        for family, rho in grid:
            run_cell(meta, track, family, rho, estimator, auxiliary, policy, total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--candidates', type=int, choices=[60, 400], default=400)
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--per-cell-limit', type=int, default=0)
    args = parser.parse_args()
    run(args.shard, args.shards, args.candidates, args.limit, args.per_cell_limit)
