"""Materialize the existing five-perturbation protocol without changing it."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'fair_budget_experiments/limix_2m_ovr/src/protocol.py'
spec = importlib.util.spec_from_file_location('original_five_protocol', PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)
LOCAL_CC18 = ROOT.parent / 'PFN-Research/data/lucos_cc18'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def local_data(item):
    path = LOCAL_CC18 / str(item.task_id)
    if not (path / 'source.json').exists():
        return original_task_data(item)
    frame = pd.read_parquet(path / 'X.parquet')
    labels = pd.read_parquet(path / 'y.parquet').iloc[:, 0]
    valid = ~labels.isna().to_numpy()
    original = np.flatnonzero(valid)
    source = json.loads((path / 'source.json').read_text())
    categorical = tuple(i for i, x in enumerate(source['categorical']) if x)

    class CachedTask:
        def get_train_test_split_indices(self, repeat=0, fold=0, sample=0):
            assert repeat == 0 and sample == 0
            with np.load(path / 'official_folds.npz') as f:
                return f[f'train_{fold}'], f[f'test_{fold}']

    return frame.loc[valid], labels.to_numpy()[valid], categorical, (CachedTask(), original)


original_task_data = protocol._task_data
protocol._task_data = local_data


def prepare_one(task, task_id):
    output = RUN / 'cache/five' / task / f'{task_id}.npz'
    if output.exists() and output.with_suffix('.json').exists():
        return json.loads(output.with_suffix('.json').read_text())
    local = LOCAL_CC18 / str(task_id) / 'source.json'
    if task == 'classification' and local.exists():
        m = json.loads(local.read_text())
        item = protocol.DatasetSpec(task_id, m['dataset_id'], m['name'], 'target')
    else:
        import openml
        ot = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(ot.dataset_id, download_data=False,
                                             download_qualities=False, download_features_meta_data=False)
        item = protocol.DatasetSpec(task_id, int(ot.dataset_id), dataset.name, ot.target_name)
    split = (protocol.load_cc18_split(item) if task == 'classification'
             else protocol.load_ctr23_split(item))
    metadata = {'track': 'five', 'task': task, 'task_id': task_id,
                'data_id': item.data_id, 'dataset': item.name,
                'source_rows': len(split.y_source), 'target_rows': len(split.y_target),
                'features': split.X_source.shape[1],
                'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                'source_indices_sha256': protocol.array_sha256(split.source_indices),
                'target_indices_sha256': protocol.array_sha256(split.target_indices),
                'X_source_sha256': protocol.array_sha256(split.X_source),
                'y_source_sha256': protocol.array_sha256(split.y_source),
                'X_target_sha256': protocol.array_sha256(split.X_target),
                'y_target_sha256': protocol.array_sha256(split.y_target),
                'scale': max(float(np.std(split.y_source)), 1e-12)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp.npz')
    np.savez_compressed(temp, X_source=split.X_source, y_source=split.y_source,
                        X_target=split.X_target, y_target=split.y_target,
                        source_indices=split.source_indices, target_indices=split.target_indices)
    temp.replace(output)
    atomic_json(output.with_suffix('.json'), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['classification', 'regression', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    import openml
    openml.config.set_root_cache_directory(str(RUN / 'cache/openml'))
    jobs = []
    for task, suite in [('classification', 99), ('regression', 353)]:
        if args.task not in ['both', task]:
            continue
        study = json.loads((RUN / f'cache/suite_{suite}.json').read_text())['study']
        ids = [int(x) for x in study['tasks']['task_id']]
        if args.limit:
            ids = ids[:args.limit]
        jobs.extend((task, i) for i in ids)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(prepare_one, *job): job for job in jobs}
        for future in as_completed(pending):
            task, task_id = pending[future]
            try:
                result = future.result()
                print('READY', task, task_id, result['dataset'], result['source_rows'], result['features'], flush=True)
            except Exception as exc:
                row = {'task': task, 'task_id': task_id, 'error': repr(exc), 'traceback': traceback.format_exc()}
                atomic_json(RUN / 'results/preparation_failures' / f'{task}_{task_id}.json', row)
                failures.append(row)
                print('FAILED', task, task_id, repr(exc), flush=True)
    atomic_json(RUN / 'results/preparation_status.json',
                {'expected': len(jobs), 'complete': len(jobs)-len(failures), 'failed': failures})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
