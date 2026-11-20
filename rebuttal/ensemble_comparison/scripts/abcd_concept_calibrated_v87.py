"""Concept source benchmark development using the diagnosed response amplitude.

Reuses untouched non-response conditions, regenerates Concept and Temporal
responses and all their paired Base/member predictions. This is a declared
development configuration, not an exact historical-generator recovery.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')

import numpy as np
import torch

import abcd_mechanism_v84 as origin
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor
from shared_reference import contexts
import seven_shifts as source

ORIGIN = origin.OUT
OUT = RUN / 'results/abcd_concept_calibrated_v87'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == origin.sha(__file__)
        return
    atomic_json(p, {'status': 'FROZEN_BEFORE_NEW_ABCD_MEMBER_INFERENCE', 'seed': 0,
        'task': 'regression', 'datasets': 33, 'context_rows': 256, 'reference_rows': 128,
        'source_pool_cap': 1024, 'estimators_per_branch': 1, 'alternative_branches': 5,
        'response_drift_sd': 1., 'original_reconstructed_response_drift_sd': 2.,
        'nominal_strengths': [.2, .5, .8, 1.],
        'all_other_source_parameters_unchanged': True, 'target_cohorts_and_labels_unchanged': True,
        'calibration_changes_both_concept_and_temporal_response_component': True,
        'amplitude_choice': 'fixed 0.5 multiplier from declared v85 development grid; not independent confirmation',
        'development_Base_reference': {'r2': .507276, 'nrmse': .664169},
        'paper_Base_reference': {'r2': .529880, 'nrmse': .639541},
        'matching_constructor': 'observed-source smooth-component repair replaces bootstrap slot; dagger adaptation',
        'ABCD_protocol': str(ORIGIN / 'plan.json'),
        'unseen_calibration_excludes': ['concept', 'temporal'],
        'all_four_policies_refitted_after_benchmark_change': True,
        'original_paper_replication': False, 'independent_test': False,
        'reference_mean_selected_on_development_metrics': True,
        'Markdown_written': False, 'code_sha256': origin.sha(__file__)})


def prepare(shard, shards):
    torch.set_num_threads(1); torch.manual_seed(0)
    source.PARAMETERS['regression_response_drift_sd'] = 1.
    dataset_meta = [m for m in source.registry() if m['task'] == 'regression']
    predictor = Predictor('regression', 'cuda:0')
    for i, meta in enumerate(dataset_meta):
        if i % shards != shard:
            continue
        clean = source.regression_clean(meta)
        for family in ['concept', 'temporal']:
            for rho in (20, 50, 80, 100):
                p = OUT / 'banks' / str(meta['task_id']) / f'{family}_{rho}.npz'
                if p.with_suffix('.json').exists():
                    continue
                data, info = source.materialize(meta, family, rho, clean)
                assert np.array_equal(data['X_target'], clean[-2].astype(np.float32))
                assert np.array_equal(data['y_target'], clean[-1])
                X, y, Q = data['X_source'], data['y_source'], data['X_target']
                base, _ = predictor(X, y, Q)
                views = contexts(data['X_pool'], data['y_pool'], Q, X, y, 256, 'regression')
                members = [predictor(A, b, T)[0] for _, A, b, T in views]
                A, b, details = origin.repaired_context(X, y)
                repaired, _ = predictor(A, b, Q)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, base=base, members=np.stack(members), repair=repaired,
                    y_target=data['y_target'], source_mean=float(data['y_pool'].mean()),
                    source_scale=max(float(data['y_pool'].std()), 1e-9))
                tmp.replace(p)
                atomic_json(p.with_suffix('.json'), info | {'status': 'COMPLETE', 'details': details,
                    'X_source_sha256': protocol.array_sha256(X), 'y_source_sha256': protocol.array_sha256(y),
                    'X_target_sha256': protocol.array_sha256(Q), 'y_target_sha256': protocol.array_sha256(data['y_target']),
                    'context_roles': [v[0] for v in views], 'context_rows': [len(v[2]) for v in views],
                    'replacement_slot': 0, 'repair_receives_clean_y_or_target_y': False,
                    'development_benchmark': True, 'benchmark_generator_uses_target_labels': False})
        print('CONCEPT_BANK_TASK', meta['task_id'], flush=True)


def worker(fold):
    saved = {}
    for matched in (True, False):
        cells = origin.load('regression', matched)
        for c in cells:
            m = c['meta']
            if m['rho'] and m['family'] in ('concept', 'temporal'):
                path = OUT / 'banks' / str(m['task_id']) / f'{m["family"]}_{m["rho"]}.npz'
                with np.load(path) as z:
                    assert np.array_equal(z['y_target'], c['y'])
                    c['base'], c['bank'] = z['base'].copy(), z['members'].copy()
                    c['mu'], c['sigma'] = float(z['source_mean']), float(z['source_scale'])
                    if matched:
                        c['bank'][0] = z['repair']
                c['path'] = path
        saved[matched] = cells
    origin.OUT = OUT
    origin.load = lambda task, matched: saved[matched]
    origin.fit_worker('regression', fold)


def launch(phase, shards):
    freeze()
    def execute(i):
        args = ['prepare-worker', '--shard', str(i), '--shards', str(shards)] if phase == 'prepare' else ['fit-worker', '--fold', str(i)]
        log = OUT / f'{phase}_{i}.log'
        env = os.environ.copy()
        if phase == 'prepare':
            env['CUDA_VISIBLE_DEVICES'] = str(i)
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, *args], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'job': i, 'returncode': p.returncode, 'log': str(log)}
    jobs = range(shards if phase == 'prepare' else 5)
    records = []
    with ThreadPoolExecutor(max_workers=shards if phase == 'prepare' else 4) as pool:
        for future in as_completed([pool.submit(execute, i) for i in jobs]):
            r = future.result(); records.append(r); print(r, flush=True)
            atomic_json(OUT / f'{phase}_execution.json', {'status': 'RUNNING', 'workers': records})
    atomic_json(OUT / f'{phase}_execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records) else 'FAILED', 'workers': records})
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'prepare-worker', 'fit', 'fit-worker'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8); p.add_argument('--fold', type=int)
    a = p.parse_args()
    if a.phase == 'prepare-worker':
        prepare(a.shard, a.shards)
    elif a.phase == 'fit-worker':
        worker(a.fold)
    else:
        launch(a.phase, a.shards)
