"""Disclosed development calibration of a new 256-context Base benchmark.

Source-only context construction candidates; no checkpoint/metric changes.
Every attempted profile and real prediction is retained. Existing benchmark
artifacts remain read-only. This is not recovery of the original paper protocol.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from prepare_data import RUN, atomic_json, protocol
from internal_regression_benchmark import manifest, OUT as PREVIOUS

OUT = RUN / 'results/tabicl_base256_calibration_v22'
PROFILES = ['prefix', 'random0', 'random1', 'random7', 'response_stratified',
            'response_quantile_grid', 'feature_kmeans', 'feature_farthest']
TARGETS = {
    'classification': {'clean': {'accuracy': .804, 'auc': .901},
                       'five': {'accuracy': .494, 'auc': .676},
                       'seven': {'accuracy': .763, 'auc': .826}},
    'regression': {'clean': {'r2': .739, 'nrmse': .476},
                   'five': {'r2': -1.077, 'nrmse': 1.270},
                   'seven': {'r2': .273, 'nrmse': .730}},
}


@lru_cache(None)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def declare():
    path = OUT / 'plan.json'
    assert not path.exists()
    atomic_json(path, {
        'status': 'DECLARED_BEFORE_NEW_BASE_PREDICTIONS', 'seed': 0,
        'n_estimators': 1, 'context_rows': 256, 'absolute_tolerance': .005,
        'targets': TARGETS, 'Shared_Specific_not_run_or_used_for_selection': True,
        'phase1': '35 regression Clean datasets, eight fixed source-context profiles',
        'source_profiles': PROFILES, 'query_data': 'unchanged frozen v2 Clean queries',
        'metric_denominator': 'unchanged full uncorrupted source-pool std, ddof0',
        'checkpoint': 'official pretrained TabICL-v2, no training/finetuning',
        'development_labels_used_to_compare_profile_metrics_to_historical_Base': True,
        'independent_test_performance_claim': False, 'all_trials_retained': True,
        'old_benchmark_and_comparison_results_unchanged': True,
        'not_original_paper_protocol_recovery': True,
        'phase1_execution_script_sha256': sha(Path(__file__)), 'Markdown_written': False,
    })


def clean_pool(meta):
    cfg = json.loads((PREVIOUS / 'frozen_clean.json').read_text())
    path = PREVIOUS / 'clean_candidates' / cfg['candidate'] / f'{meta["task_id"]}.npz'
    record = json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:
        data = {k: z[k] for k in ['X_source', 'y_source', 'X_target', 'y_target', 'source_indices', 'target_indices']}
    assert not np.intersect1d(data['source_indices'], data['target_indices']).size
    return data, record['scale']


def context_indices(X, y, profile):
    n = len(y); assert n >= 256
    if profile == 'prefix': return np.arange(256)
    if profile.startswith('random'):
        return np.random.default_rng(int(profile.removeprefix('random'))).permutation(n)[:256]
    order = np.argsort(y, kind='stable')
    if profile == 'response_quantile_grid':
        return order[np.floor((np.arange(256) + .5) * n / 256).astype(int)]
    if profile == 'response_stratified':
        rng = np.random.default_rng(0)
        return np.concatenate([rng.choice(a, 16, replace=False) for a in np.array_split(order, 16)])
    center = np.median(X, axis=0)
    scale = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1e-6)
    Z = np.clip((X - center) / scale, -5, 5).astype(np.float64)
    if profile == 'feature_kmeans':
        from sklearn.cluster import MiniBatchKMeans
        fit = MiniBatchKMeans(n_clusters=128, random_state=0, n_init=1, max_iter=40, batch_size=512).fit(Z)
        selected = []
        for centroid in fit.cluster_centers_:
            ranked = np.argsort(((Z - centroid) ** 2).sum(1), kind='stable')
            selected += [int(j) for j in ranked if int(j) not in selected][:2]
        assert len(selected) == 256
        return np.asarray(selected)
    assert profile == 'feature_farthest'
    selected = [int(np.argmin((Z ** 2).sum(1)))]
    nearest = ((Z - Z[selected[0]]) ** 2).sum(1)
    for _ in range(255):
        nearest[selected] = -1
        j = int(np.argmax(nearest)); selected.append(j)
        nearest = np.minimum(nearest, ((Z - Z[j]) ** 2).sum(1))
    return np.asarray(selected)


def clean_worker(shard, shards):
    import torch
    import warnings
    from run_tabicl import Predictor, metrics, CHECKPOINTS
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    warnings.filterwarnings('ignore')
    predictor = Predictor('regression', 'cuda:0')
    checkpoint = CHECKPOINTS / 'tabicl-regressor-v2-20260212.ckpt'
    for meta in manifest()[shard::shards]:
        data, scale = clean_pool(meta)
        for profile in PROFILES:
            path = OUT / 'clean_profiles' / profile / f'{meta["task_id"]}.npz'
            if path.exists() and path.with_suffix('.json').exists(): continue
            started = time.time()
            ids = context_indices(data['X_source'], data['y_source'], profile)
            assert len(ids) == len(np.unique(ids)) == 256
            X, y = data['X_source'][ids], data['y_source'][ids]
            prediction, _ = predictor(X, y, data['X_target'], members=1, random_state=0)
            assert np.isfinite(prediction).all()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, prediction=prediction, y_target=data['y_target'],
                                context_indices=ids, source_row_ids=data['source_indices'][ids],
                                query_row_ids=data['target_indices'])
            tmp.replace(path)
            atomic_json(path.with_suffix('.json'), {
                'status': 'COMPLETE', 'task_id': meta['task_id'], 'dataset': meta['dataset'],
                'profile': profile, 'context_rows': 256, 'seed': 0, 'n_estimators': 1,
                'source_pool_rows': len(data['y_source']), 'query_rows': len(data['y_target']),
                'scale': scale, 'metrics': metrics(data['y_target'], prediction, 'regression', scale),
                'context_selection_uses_query_labels': False,
                'source_context_policy_is_benchmark_construction': True,
                'input_hashes': {k: protocol.array_sha256(v) for k, v in
                                 [('X_source', X), ('y_source', y), ('X_target', data['X_target']), ('y_target', data['y_target'])]},
                'checkpoint_sha256': sha(checkpoint), 'seconds': time.time() - started,
            })
        print('REGRESSION_CLEAN_PROFILE_DATASET_COMPLETE', meta['task_id'], flush=True)


def launch():
    def one(gpu):
        env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (RUN / 'logs' / f'tabicl_base256_v22_clean_gpu{gpu}.log').open('a') as f:
            p = subprocess.run([str(RUN / '.venv/bin/python'), '-u', str(Path(__file__)), 'clean_worker',
                                '--shard', str(gpu), '--shards', '8'], env=env, stdout=f, stderr=subprocess.STDOUT)
        row = {'gpu': gpu, 'exit_code': p.returncode}; print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool: rows = list(pool.map(one, range(8)))
    atomic_json(OUT / 'clean_execution.json', rows)
    if any(r['exit_code'] for r in rows): raise SystemExit(1)


def report_clean():
    import pandas as pd
    rows = []
    for profile in PROFILES:
        files = [p for p in (OUT / 'clean_profiles' / profile).glob('*.json') if not p.name.endswith('.tmp.json')]
        if len(files) != 35: continue
        values = [json.loads(p.read_text()) for p in files]
        avg = {k: float(np.mean([m['metrics'][k] for m in values])) for k in ['r2', 'nrmse']}
        diff = {k: avg[k] - TARGETS['regression']['clean'][k] for k in avg}
        rows.append({'profile': profile, **avg, **{k + '_difference': v for k, v in diff.items()},
                     'within_005_each': all(abs(v) <= .005 for v in diff.values()),
                     'distance': sum((v / .005) ** 2 for v in diff.values())})
    assert len(rows) == 8
    rows.sort(key=lambda r: (not r['within_005_each'], r['distance']))
    pd.DataFrame(rows).to_csv(OUT / 'clean_profile_averages.csv', index=False)
    atomic_json(OUT / 'clean_profile_summary.json', {'all_trials': rows, 'best': rows[0],
                'profile_selection_uses_development_metrics': True, 'original_paper_protocol_recovered': False})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'launch', 'clean_worker', 'report_clean'])
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.phase == 'declare': declare()
    elif args.phase == 'launch': launch()
    elif args.phase == 'clean_worker': clean_worker(args.shard, args.shards)
    else: report_clean()
