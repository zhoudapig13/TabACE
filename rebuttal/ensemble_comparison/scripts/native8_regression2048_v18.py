"""Matched Native-1/Native-8 regression comparison, full old source cap 2048."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from prepare_data import RUN, atomic_json, protocol

OUT = RUN / 'results/native8_regression2048_seed0_v18'
PLAN = RUN / 'configs/native8_regression2048_seed0_v18.json'
BENCH = RUN / 'results/internal_regression_benchmark_v2'


@lru_cache(None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    return [json.loads(p.read_text()) for p in sorted((RUN / 'cache/five/regression').glob('*.json'))]


def declare():
    assert not PLAN.exists()
    atomic_json(PLAN, {
        'status': 'FROZEN_BEFORE_INFERENCE', 'seed': 0, 'n_estimators': 8,
        'context_cap': 2048, 'source_selection': 'entire unchanged old Native-1 source context',
        'same_source_query_and_metric_scale_as_previous_Native1': True,
        'task': 'regression', 'blocks': ['Clean', 'Five'], 'datasets': 35,
        'conditions': 735, 'strengths': [.2, .5, .8, 1.],
        'finetuning': False, 'adaptation': False, 'seven_not_rerun': True,
        'classification_not_rerun': True, '512_not_run': True,
        'benchmark': 'internal_regression_benchmark_v2', 'exact_original_paper_protocol': False,
        'frozen_files_sha256': {str(p.relative_to(RUN)): sha(p) for p in
            [BENCH / 'frozen_clean.json', BENCH / 'frozen_five.json']},
        'execution_script_sha256': sha(Path(__file__)), 'Markdown_written': False,
    })
    print('DECLARED', PLAN, flush=True)


def worker(shard, shards):
    import torch
    import warnings
    from run_tabicl import Predictor, metrics, CHECKPOINTS
    from recalibrate_tabicl_regression_v2 import load_frozen_five
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    plan = json.loads(PLAN.read_text())
    for p, expected in plan['frozen_files_sha256'].items():
        assert sha(RUN / p) == expected
    predictor = Predictor('regression', 'cuda:0')
    checkpoint = CHECKPOINTS / 'tabicl-regressor-v2-20260212.ckpt'
    cache, failures = {}, []
    for meta in manifest()[shard::shards]:
        for family, rho in [('original', 0)] + [(f, s) for f in protocol.FAMILIES for s in [20, 50, 80, 100]]:
            path = OUT / 'predictions' / str(meta['task_id']) / f'{family}_{rho}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                continue
            try:
                start = time.time()
                data, info, base = load_frozen_five(meta, family, rho, cache)
                X, y, Q = data['X_source'], data['y_source'], data['X_target']
                assert 0 < len(y) <= 2048
                prediction, _ = predictor(X, y, Q, members=8, random_state=0)
                assert prediction.shape == base.shape == data['y_target'].shape
                assert np.isfinite(prediction).all()
                source = BENCH / 'base_predictions/five' / str(meta['task_id']) / path.name
                previous = json.loads(source.with_suffix('.json').read_text())
                hashes = {k: protocol.array_sha256(v) for k, v in data.items()}
                assert hashes == previous['input_hashes']
                observed = {name: metrics(data['y_target'], p, 'regression', info['scale'])
                            for name, p in [('Native-1', base), ('Native-8', prediction)]}
                for k in ['r2', 'nrmse']:
                    assert abs(observed['Native-1'][k] - previous['metrics'][k]) < 1e-10
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, Native1=base, Native8=prediction, y_target=data['y_target'])
                tmp.replace(path)
                atomic_json(path.with_suffix('.json'), {
                    'status': 'COMPLETE', 'task_id': meta['task_id'], 'dataset': meta['dataset'],
                    'family': family, 'rho': rho, 'seed': 0, 'context_rows': len(y), 'context_cap': 2048,
                    'query_rows': len(Q), 'scale': info['scale'], 'metrics': observed,
                    'n_estimators': {'Native-1': 1, 'Native-8': 8},
                    'input_hashes': hashes, 'Native1_reused_from': str(source.relative_to(RUN)),
                    'checkpoint_sha256': sha(checkpoint), 'checkpoint': str(checkpoint),
                    'plan_sha256': sha(PLAN), 'same_input_for_both_methods': True,
                    'finetuning': False, 'adaptation': False,
                    'test_labels_used_for_fitting_or_selection': False,
                    'seconds': time.time() - start,
                })
            except Exception as exc:
                record = {'task_id': meta['task_id'], 'family': family, 'rho': rho,
                          'error': repr(exc), 'traceback': traceback.format_exc()}
                failures.append(record)
                atomic_json(OUT / 'failures' / f'{meta["task_id"]}_{family}_{rho}.json', record)
                print('FAILED', json.dumps(record), flush=True)
                torch.cuda.empty_cache()
        print('DATASET_FINISHED', meta['task_id'], meta['dataset'], flush=True)
    atomic_json(OUT / f'worker_{shard}.json', {'failures': failures, 'status': 'FAILED' if failures else 'COMPLETE'})
    if failures:
        raise SystemExit(1)


def launch():
    def one(gpu):
        env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        log = RUN / 'logs' / f'native8_regression2048_v18_gpu{gpu}.log'
        with log.open('a') as f:
            p = subprocess.run([str(RUN / '.venv/bin/python'), '-u', str(Path(__file__)), 'worker',
                                '--shard', str(gpu), '--shards', '8'], env=env, stdout=f, stderr=subprocess.STDOUT)
        row = {'gpu': gpu, 'exit_code': p.returncode}
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(8)))
    atomic_json(OUT / 'execution.json', rows)
    if any(r['exit_code'] for r in rows):
        raise SystemExit(1)


def report(complete):
    import pandas as pd
    from run_tabicl import metrics
    records = []
    files = [p for p in sorted((OUT / 'predictions').glob('*/*.json')) if not p.name.endswith('.tmp.json')]
    for path in files:
        m = json.loads(path.read_text())
        assert m['seed'] == 0 and m['same_input_for_both_methods']
        assert not m['finetuning'] and not m['adaptation']
        assert m['plan_sha256'] == sha(PLAN)
        with np.load(path.with_suffix('.npz')) as z:
            for name, key in [('Native-1', 'Native1'), ('Native-8', 'Native8')]:
                observed = metrics(z['y_target'], z[key], 'regression', m['scale'])
                for metric in ['r2', 'nrmse']:
                    assert abs(observed[metric] - m['metrics'][name][metric]) < 1e-12
                records.append({'method': name, 'task_id': m['task_id'], 'family': m['family'],
                                'rho': m['rho'], 'context_rows': m['context_rows'], **observed})
    frame = pd.DataFrame(records)
    assert not frame.duplicated(['method', 'task_id', 'family', 'rho']).any()
    rows = []
    for name in ['Native-1', 'Native-8']:
        for block, clean, expected in [('Clean', True, 35), ('Five', False, 700)]:
            part = frame[(frame.method == name) & ((frame.rho == 0) if clean else (frame.rho > 0))]
            rows.append({'method': name, 'block': block, 'context_cap': 2048, 'seed': 0,
                         'completed': len(part), 'expected': expected,
                         'r2': float(part.r2.mean()), 'nrmse': float(part.nrmse.mean())})
    if complete:
        assert len(files) == 735 and all(row['completed'] == row['expected'] for row in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'detail.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'averages.csv', index=False)
    atomic_json(OUT / 'summary.json', {'complete': len(files) == 735, 'rows': rows,
                'actual_context_rows_min': int(frame.context_rows.min()),
                'actual_context_rows_max': int(frame.context_rows.max()),
                'same_input_for_both_methods': True, 'seven_not_rerun': True})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'launch', 'worker', 'report', 'live'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.phase == 'declare': declare()
    elif args.phase == 'launch': launch()
    elif args.phase == 'worker': worker(args.shard, args.shards)
    else: report(args.phase == 'report')
