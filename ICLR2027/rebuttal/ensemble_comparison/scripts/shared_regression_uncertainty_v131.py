"""Acquire official predictive intervals for fixed v126 regression contexts.

This changes no context or stored mean prediction. It records additional
unlabeled predictive features for a possible subsequent fusion adaptation.
No evaluation labels are opened and no effect is claimed by this script.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
import torch

import unseen_shared_robustness_v117 as source
import shared_conditional_hampel_banks_v126 as parent
import conditional_hampel_context_v126 as clipping
import shared_quantile_contexts_v122 as alignment
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN/'results/shared_regression_uncertainty_v131'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'FROZEN_BEFORE_UNCERTAINTY_ACQUISITION', 'seed': 0,
        'task': 'regression', 'context_cap': 256, 'contexts': 'canonical plus five v126 branches',
        'official_Native_estimators_per_context': 1, 'quantiles': [.15, .85],
        'new_information': 'official predictive interval widths on existing unlabeled query features',
        'Base_or_contexts_or_benchmark_modified': False, 'query_labels_opened': False,
        'source_labels_are_existing_context_labels': True,
        'stored_v126_means_retained_even_if_numeric_roundoff_differs': True,
        'mean_replay_rtol': 1e-4, 'mean_replay_atol_source_std': 1e-5,
        'expected_conditions': 455, 'datasets': 35,
        'script_sha256': source.sha(__file__), 'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'clipping_sha256': source.sha(clipping.__file__),
        'alignment_sha256': source.sha(alignment.__file__), 'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text())
        if c['task'] == 'regression' and c['track'] == 'perturbation']
    order = np.random.default_rng(131).permutation(len(cases))
    predictor = Predictor('regression', 'cuda:0')
    for i, position in enumerate(order):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'features'/str(c['task_id'])/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, 'regression', family, rho, c['task_id'])
            bank_path = parent.OUT/'banks/regression'/str(c['task_id'])/dest.name
            with np.load(bank_path) as z:
                P = np.concatenate([z['Base'][None], z['members']], axis=0)
                sigma = max(float(z['source_scale']), 1e-12)
            original_views = source.constructors.shared_contexts(X, y, Q, len(y), 'regression', P[0])
            repaired, labels, _ = clipping.repair(X, y, Q, numeric, 'regression')
            _, cov, _ = alignment.align(X, Q, numeric)
            views = [('canonical', X, y, Q)]+original_views[:3]+[
                ('conditional_hampel', repaired, labels, Q), ('quantile_covariance', cov, y, Q)]
            widths, replayed, cache = [], [], {}
            for role, A, b, T in views:
                key = tuple(protocol.array_sha256(v) for v in (A, b, T))
                if key not in cache:
                    cache[key] = predictor(A, b, T, members=1, random_state=0, quantiles=True)
                mean, uncertainty = cache[key]
                assert np.isfinite(mean).all() and np.isfinite(uncertainty).all()
                assert (uncertainty >= 0).all()
                replayed.append(mean)
                widths.append(uncertainty)
            replayed, widths = np.stack(replayed), np.stack(widths)
            np.testing.assert_allclose(replayed, P, rtol=1e-4, atol=1e-5*sigma)
            source.npwrite(dest, interval_width=widths, replayed_mean=replayed)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'parent_bank': str(bank_path), 'parent_bank_sha256': source.sha(bank_path),
                'source_input_sha256': c['input_sha256'], 'query_labels_opened': False,
                'context_rows': [len(b) for _, _, b, _ in views],
                'roles': [role for role, _, _, _ in views], 'TFM_calls': len(cache),
                'maximum_mean_difference_over_source_std': float(np.max(np.abs(replayed-P))/sigma),
                'official_interval_alphas': [.15, .85]})
        print('UNCERTAINTY_COMPLETE', c['task_id'], flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[name] = '1'
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            process = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': process.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            results.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results})
            print('UNCERTAINTY_WORKER', results[-1], flush=True)
    success = all(r['returncode'] == 0 for r in results)
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    launch(a.shards) if a.phase == 'launch' else worker(a.shard, a.shards)
