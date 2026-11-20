"""Native TabICL-v2, n_estimators=8, context cap 256, seed 0 only.

Replays the current frozen development benchmarks. No score-based adaptation,
benchmark changes, or 512-context inference. Existing exact-input Native-8
classification predictions are reused, with a fresh numerical cross-check.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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

OUT = RUN / 'results/native8_context256_seed0_v16'
PLAN = RUN / 'configs/native8_context256_seed0_v16.json'
STRENGTHS = [20, 50, 80, 100]


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grids():
    from seven_shifts import registry
    cases = []
    for task in ['classification', 'regression']:
        cases += [('five', json.loads(p.read_text()))
                  for p in sorted((RUN / 'cache/five' / task).glob('*.json'))]
    cases += [('seven', m) for m in registry()]
    return cases


def declare():
    assert not PLAN.exists(), 'Preserve the existing declared plan'
    frozen = [RUN / 'configs/feature_protocol_frozen.json',
              RUN / 'results/internal_seven_fixed_strength_v2/frozen_benchmark.json']
    frozen += [RUN / 'results/internal_regression_benchmark_v2' / f'frozen_{s}.json'
               for s in ['clean', 'five', 'seven']]
    atomic_json(PLAN, {
        'status': 'FROZEN_BEFORE_INFERENCE', 'seed': 0, 'n_estimators': 8,
        'context_rows': 256, 'run_512': False, 'finetuning': False,
        'adaptation': False, 'backbone': 'TabICL-v2 official frozen checkpoint',
        'strengths': [.2, .5, .8, 1.],
        'tracks': {'classification_five': 'five_feature_calibrated',
                   'classification_seven': 'internal_seven_fixed_strength_v2',
                   'regression_clean_five': 'internal_regression_benchmark_v2',
                   'regression_seven': 'seven_calibrated_v2'},
        'cohorts': {'classification_clean_five': 72, 'classification_seven': 35,
                    'regression_clean_five': 35, 'regression_seven': 33},
        'expected_cells': 4151,
        'context_selection': 'first 256 rows of the already materialized source context',
        'regression_five_original_source_cap': 2048,
        'queries_and_metric_denominators_unchanged': True,
        'native_options': 'official defaults, except n_estimators=8, batch_size=1, random_state=0',
        'reuse': 'only exact source/query/label hashes with a saved native_8 prediction',
        'reuse_cross_check': 'fresh inference for first reused cell of each worker',
        'benchmark_parameters_previously_selected_on_development_data': True,
        'exact_original_paper_protocol': False, 'same_protocol_Shared_run': False,
        'all_outcomes_reported_without_ranking_selection': True,
        'frozen_input_files_sha256': {str(p.relative_to(RUN)): file_sha(p) for p in frozen},
        'script_sha256': file_sha(Path(__file__)), 'Markdown_written': False,
    })
    print('DECLARED', PLAN, flush=True)


def condition(track, meta, family, rho, cache):
    from shared_reference import load_condition
    if track == 'five' and meta['task'] == 'classification':
        return load_condition(meta | {'track': 'five_feature_calibrated'}, family, rho)
    if track == 'five':
        from recalibrate_tabicl_regression_v2 import load_frozen_five
        data, info, _ = load_frozen_five(meta, family, rho, cache.setdefault('regression', {}))
        return data, info
    if meta['task'] == 'regression':
        return load_condition(meta | {'track': 'seven_calibrated_v2'}, family, rho)
    from internal_seven_fixed_strength import make_condition, SELECTED
    from seven_shifts import classification_clean
    if meta['task_id'] not in cache:
        cache[meta['task_id']] = classification_clean(meta)
    frozen = json.loads(SELECTED.read_text())
    return make_condition(meta, family, rho, frozen, cache[meta['task_id']])


def cached_native(meta, family, rho, hashes):
    old = RUN / 'results/tabicl/five_feature_calibrated/classification' / str(meta['task_id']) / f'{family}_{rho}.npz'
    if not old.exists() or not old.with_suffix('.json').exists():
        return None, None
    record = json.loads(old.with_suffix('.json').read_text())
    expected = {'X_source': record.get('X_context_sha256'),
                'y_source': record.get('y_context_sha256'),
                'X_target': record.get('X_shifted_target_sha256'),
                'y_target': record.get('y_target_sha256')}
    if expected != hashes or record.get('seed') != 0 or 'native_8' not in record.get('methods', {}):
        return None, None
    with np.load(old) as z:
        prediction = z['prediction__native_8'].copy()
    return prediction, str(old.relative_to(RUN))


def worker(shard, shards, limit=0):
    import torch
    import warnings
    from run_tabicl import Predictor, metrics, CHECKPOINTS
    from seven_shifts import FAMILIES
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    plan = json.loads(PLAN.read_text())
    assert plan['context_rows'] == 256 and plan['seed'] == 0 and not plan['run_512']
    for path, expected in plan['frozen_input_files_sha256'].items():
        assert file_sha(RUN / path) == expected
    predictors, cache, failures = {}, {}, []
    checked_reuse = False
    cases = grids()[shard::shards]
    if limit:
        cases = cases[:limit]
    for track, meta in cases:
        task = meta['task']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        families = protocol.FAMILIES if track == 'five' else FAMILIES
        cells = ([('original', 0)] if track == 'five' else []) + [(f, s) for f in families for s in STRENGTHS]
        for family, rho in cells:
            path = OUT / 'predictions' / task / track / str(meta['task_id']) / f'{family}_{rho}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                continue
            try:
                start = time.time()
                data, info = condition(track, meta, family, rho, cache)
                X, y = data['X_source'][:256], data['y_source'][:256]
                Q = data['X_target']
                assert len(y) == 256
                n_classes = info.get('n_classes', len(np.unique(data['y_source']))) if task == 'classification' else None
                inputs = {'X_source': X, 'y_source': y, 'X_target': Q, 'y_target': data['y_target']}
                hashes = {k: protocol.array_sha256(a) for k, a in inputs.items()}
                prediction, reused = (cached_native(meta, family, rho, hashes)
                                      if task == 'classification' and track == 'five' else (None, None))
                cross_check = None
                if prediction is None or not checked_reuse:
                    fresh, _ = predictor(X, y, Q, members=8, random_state=0, n_classes=n_classes)
                    if prediction is not None:
                        cross_check = float(np.max(np.abs(fresh - prediction)))
                        assert np.allclose(fresh, prediction, rtol=1e-5, atol=1e-6), cross_check
                        checked_reuse = True
                    else:
                        prediction = fresh
                assert len(prediction) == len(Q) and np.isfinite(prediction).all()
                scale = max(float(data['y_target'].std()), 1e-12) if task == 'regression' and track == 'seven' else info['scale']
                checkpoint = CHECKPOINTS / f"tabicl-{'classifier' if task == 'classification' else 'regressor'}-v2-20260212.ckpt"
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, prediction=prediction, y_target=data['y_target'], context_positions=np.arange(256))
                temp.replace(path)
                atomic_json(path.with_suffix('.json'), {
                    'status': 'COMPLETE', 'task': task, 'task_id': meta['task_id'],
                    'dataset': meta['dataset'], 'track': track, 'family': family, 'rho': rho,
                    'strength': rho / 100., 'method': 'TabICL Native-8', 'seed': 0,
                    'context_rows': len(y), 'original_materialized_source_rows': len(data['y_source']),
                    'n_estimators': 8, 'query_rows': len(Q), 'scale': scale,
                    'input_hashes': hashes, 'checkpoint': str(checkpoint),
                    'checkpoint_sha256': file_sha(checkpoint), 'benchmark_track': info['track'],
                    'plan_sha256': file_sha(PLAN), 'source_only_selection': True,
                    'test_labels_used_for_training_or_method_selection': False,
                    'adaptation': False, 'finetuning': False,
                    'reused_from': reused, 'reuse_fresh_max_error': cross_check,
                    'metrics': metrics(data['y_target'], prediction, task, scale),
                    'seconds': time.time() - start,
                })
            except Exception as exc:
                failure = {'task': task, 'track': track, 'task_id': meta['task_id'],
                           'family': family, 'rho': rho, 'error': repr(exc), 'traceback': traceback.format_exc()}
                failures.append(failure)
                atomic_json(OUT / 'failures' / f'{task}_{track}_{meta["task_id"]}_{family}_{rho}.json', failure)
                print('FAILED', json.dumps(failure), flush=True)
                torch.cuda.empty_cache()
        print('DATASET_FINISHED', task, track, meta['task_id'], flush=True)
    atomic_json(OUT / f'worker_{shard}.json', {'status': 'FAILED' if failures else 'COMPLETE', 'failures': failures})
    if failures:
        raise SystemExit(1)


def launch():
    python = RUN / '.venv/bin/python'
    def one(gpu):
        log = RUN / 'logs' / f'native8_context256_v16_gpu{gpu}.log'
        env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with log.open('a') as f:
            p = subprocess.run([str(python), '-u', str(Path(__file__)), 'worker',
                                '--shard', str(gpu), '--shards', '8'], env=env, stdout=f, stderr=subprocess.STDOUT)
        row = {'gpu': gpu, 'exit_code': p.returncode, 'log': str(log)}
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(8)))
    atomic_json(OUT / 'execution.json', rows)
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


def report(require_complete):
    import pandas as pd
    from run_tabicl import metrics
    rows = []
    for p in sorted((OUT / 'predictions').glob('*/*/*/*.json')):
        if p.name.endswith('.tmp.json'):
            continue
        m = json.loads(p.read_text())
        assert m['status'] == 'COMPLETE' and m['context_rows'] == 256 and m['n_estimators'] == 8 and m['seed'] == 0
        with np.load(p.with_suffix('.npz')) as z:
            observed = metrics(z['y_target'], z['prediction'], m['task'], m['scale'])
            for key, val in observed.items():
                if val is not None:
                    assert abs(val - m['metrics'][key]) < 1e-12
            assert np.array_equal(z['context_positions'], np.arange(256))
        rows.append({k: m[k] for k in ['task', 'track', 'task_id', 'family', 'rho', 'context_rows', 'seed', 'reused_from']} | observed)
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['task', 'track', 'task_id', 'family', 'rho']).any()
    summary = []
    for task, clean_expected, seven_expected in [('classification', 72, 35), ('regression', 35, 33)]:
        for block, track, clean, expected in [('Clean', 'five', True, clean_expected),
                                             ('Five', 'five', False, clean_expected * 20),
                                             ('Seven', 'seven', False, seven_expected * 28)]:
            part = frame[(frame.task == task) & (frame.track == track) & ((frame.rho == 0) if clean else (frame.rho > 0))]
            keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
            row = {'task': task, 'block': block, 'context_rows': 256, 'seed': 0,
                   'completed': len(part), 'expected': expected, 'complete': len(part) == expected}
            row.update({key: float(part[key].mean()) if len(part) else None for key in keys})
            summary.append(row)
    if require_complete:
        assert len(frame) == 4151 and all(r['complete'] for r in summary)
    frame.to_csv(OUT / 'detail.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'averages.csv', index=False)
    atomic_json(OUT / 'summary.json', {'complete': all(r['complete'] for r in summary),
                'context_rows': 256, 'seed': 0, 'n_estimators': 8, 'rows': summary,
                'cached_cells': int(frame.reused_from.notna().sum()),
                'benchmark_is_internal_development_reconstruction': True, 'run_512': False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['declare', 'worker', 'launch', 'report', 'live'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.phase == 'declare': declare()
    elif a.phase == 'worker': worker(a.shard, a.shards, a.limit)
    elif a.phase == 'launch': launch()
    else: report(a.phase == 'report')
