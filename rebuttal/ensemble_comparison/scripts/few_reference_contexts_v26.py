"""Use a small, disjoint clean TRAIN reference pool with registered operators.

This changes adaptation inputs, never the frozen benchmark. It measures a
limited unique-example budget rather than a fraction of replaced context rows.
All observed-only and row-replacement experiments remain preserved.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import paperish_fixed_benchmark_v24 as core
import numpy as np
import pandas as pd
import torch
from prepare_data import atomic_json, protocol
from run_tabicl import Predictor, metrics

REFERENCE_BUDGET = int(os.environ.get('TABACE_REFERENCE_BUDGET', '64'))
OUT = core.RUN / ('results/few_reference_contexts_v26' if REFERENCE_BUDGET == 64
                  else f'results/few_reference_contexts_v27_ref{REFERENCE_BUDGET}')
VARIANT = f'reference{REFERENCE_BUDGET:03d}'
RAW_SCALE = False


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists():
        return
    old = json.loads((core.OUT / 'plan.json').read_text())
    atomic_json(path, old | dict(
        status='DECLARED_BEFORE_NEW_INFERENCE_AND_EVALUATION',
        adaptation='registered context operators on a limited clean FT TRAIN reference pool',
        source_reference_pool_cap=REFERENCE_BUDGET,
        class_coverage_exception='raise the pool size only when needed to retain one example per observed class',
        per_context_cap=256,
        actual_rows='at most the limited clean pool size; actual counts recorded per branch',
        variants=[VARIANT], reference_fraction_per_view_upper_bound=None,
        Shared_alternatives=5, Specific_alternatives=10,
        Specific_pool_scope='five Shared plus five reconstructed family-directed operators on the same small clean pool',
        regression_normalization='mean/std of the same limited clean reference labels',
        normalization_ablation='regression also replays the exact same predictions using cached observed-source normalization',
        additional_control='one Native-1 prediction using the identical limited reference pool',
        final_query_labels_used_for_selection=False,
        paper_values_not_used_for_selection=True,
        unique_extra_clean_label_budget=f'{REFERENCE_BUDGET}, or the observed class count if greater',
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    ))


def make(job, role, block, family, rho, predictor, pool, split, ref, cache=None):
    sub = Path('calibration') if role == 'validation' else Path('query') / block
    dest = OUT / core.rel(job) / sub / f'{family}_{rho}.npz'
    if dest.exists() and dest.with_suffix('.json').exists():
        return
    raw = core.RAW / core.rel(job) / sub / f'{family}_{rho}.npz'
    meta = json.loads(raw.with_suffix('.json').read_text())
    if role == 'query':
        data, _ = core.source.bank(job['task'], block, job['meta'], family, rho)
    else:
        data = core.source.materialize(job, pool, np.r_[split['train'], split['val']], split['cal'], family, rho)
    assert meta['input_hashes'] == {k: protocol.array_sha256(v) for k, v in data.items()}
    with np.load(raw) as z:
        base = z['base']
    A, b, ids = ref
    Q = data['X_target']
    cap = len(b)
    common = core.main.contexts.shared_contexts(A, b, Q, cap, job['task'], base)
    extra = core.main.contexts.specific_contexts(A, b, Q, cap, job['task'], block, family, base)
    assert len(common) == len(extra) == 5
    cache = {} if cache is None else cache
    costs, predictions = [], []
    calls_before = len(cache)
    started = time.monotonic()

    def predict(C, d, V):
        key = tuple(protocol.array_sha256(a) for a in (C, d, V))
        if key not in cache:
            p, _ = predictor(C, d, V, members=1, random_state=0,
                             n_classes=base.shape[1] if job['task'] == 'classification' else None)
            assert np.isfinite(p).all()
            cache[key] = p
        return cache[key], key

    reference, _ = predict(A, b, Q)
    for name, C, d, V in common + extra:
        assert len(d) <= cap <= 256
        p, hashes = predict(C, d, V)
        predictions.append(p)
        costs.append(dict(role=name, rows=len(d), input_hashes=hashes))
    assert {k: protocol.array_sha256(v) for k, v in data.items()} == meta['input_hashes']
    core.main.scaling.npwrite(dest, members=np.stack(predictions), reference=reference,
                             mu=np.array(float(b.mean())), sigma=np.array(max(float(b.std()), 1e-8)))
    atomic_json(dest.with_suffix('.json'), dict(
        frozen_input_hashes=meta['input_hashes'], roles=costs,
        clean_reference_examples=len(b), clean_reference_ids_hash=protocol.array_sha256(ids),
        reference_disjoint_from_calibration_and_final_queries=True,
        additional_clean_TRAIN_label_supervision=True, pretrained_backbone_only=True,
        query_labels_used_by_constructors=False, seconds=time.monotonic() - started,
        new_unique_TFM_inputs=len(cache)-calls_before))


def worker(shard, limit, shards=8):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    predictors = {}
    jobs = core.main.jobs()[shard::shards]
    if limit:
        jobs = jobs[:limit]
    for i, job in enumerate(jobs):
        dest = OUT / core.rel(job)
        if (dest / 'inference_complete.json').exists():
            continue
        try:
            pool = core.main.scaling.prepare(job['task'], job['track'], job['meta'])
            with np.load(core.FT / core.rel(job) / 'split.npz') as z:
                split = {k: z[k] for k in z.files}
            A, b, ids = core.references.references(job, pool, split)
            count = min(len(b), max(REFERENCE_BUDGET, len(np.unique(b)) if job['task'] == 'classification' else REFERENCE_BUDGET))
            ref = A[:count], b[:count], ids[:count]
            if job['task'] not in predictors:
                predictors[job['task']] = Predictor(job['task'], 'cuda:0')
            predictor = predictors[job['task']]
            cache = {}
            for family, rho in [('original', 0)] + [(f, r) for f in core.main.families(job['track']) for r in [50, 100]]:
                make(job, 'validation', 'clean' if rho == 0 else job['track'], family, rho,
                     predictor, pool, split, ref, cache)
            for block in (['clean', 'five'] if job['track'] == 'five' else ['seven']):
                for family, rho in core.main.contexts.case_grid(block):
                    make(job, 'query', block, family, rho, predictor, pool, split, ref, cache)
            atomic_json(dest / 'inference_complete.json', dict(task=job['task'], track=job['track'],
                        task_id=job['meta']['task_id'], clean_reference_examples=count))
            (dest / 'error.json').unlink(missing_ok=True)
            print('INFERENCE_COMPLETE', shard, i + 1, len(jobs), job['task'], job['track'], job['meta']['task_id'], flush=True)
        except Exception:
            error = traceback.format_exc()
            atomic_json(dest / 'error.json', dict(error=error))
            print(error, flush=True)


def load_cells(task, variant, role):
    cells = []
    for job in [j for j in core.main.jobs() if j['task'] == task]:
        root = core.RAW / core.rel(job)
        paths = sorted((root / 'calibration').glob('*.npz')) if role == 'validation' else sorted((root / 'query').glob('*/*.npz'))
        for p in paths:
            c = core.read_cell(p, job, role)
            with np.load(OUT / core.rel(job) / p.relative_to(root)) as z:
                c.update(bank=z['members'], reference=z['reference'])
                if not RAW_SCALE:
                    c.update(mu=float(z['mu']), sigma=float(z['sigma']))
            c.update(path=str(p), group=(job['track'], c['meta']['family']))
            cells.append(c)
    return cells


def report_reference_control(task):
    rows = []
    for c in load_cells(task, VARIANT, 'query'):
        rows.append({k: c['meta'][k] for k in ['task', 'block', 'task_id', 'dataset', 'family', 'rho']} |
                    dict(method='Reference1', **metrics(c['y'], c['reference'], task, c['meta']['scale'])))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f'reference1_detail_{task}.csv', index=False)
    keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
    avg = frame.groupby(['task', 'block', 'method'], as_index=False)[keys].mean()
    avg.to_csv(OUT / f'reference1_averages_{task}.csv', index=False)
    print('SAME_SUPERVISION_REFERENCE_CONTROL', flush=True)
    print(avg.to_string(index=False), flush=True)


def launch():
    (OUT / 'logs').mkdir(exist_ok=True)
    workers = []
    for gpu in range(8):
        log = OUT / 'logs' / f'inference_{gpu}.log'
        with log.open('a') as stream:
            p = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--shard', str(gpu)],
                                 env=os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        workers.append(dict(gpu=gpu, pid=p.pid, log=str(log)))
    atomic_json(OUT / 'workers.json', workers)
    print(workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'worker', 'launch', 'fit'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--task', choices=['classification', 'regression'])
    parser.add_argument('--raw-scale', action='store_true')
    args = parser.parse_args()
    declare()
    if args.phase == 'worker':
        worker(args.shard, args.limit, args.shards)
    elif args.phase == 'launch':
        launch()
    elif args.phase == 'fit':
        RAW_SCALE = args.raw_scale
        core.OUT = OUT
        core.load_cells = load_cells
        core.fit_worker(args.task, VARIANT + ('_rawscale' if RAW_SCALE else ''))
        if not RAW_SCALE:
            report_reference_control(args.task)
