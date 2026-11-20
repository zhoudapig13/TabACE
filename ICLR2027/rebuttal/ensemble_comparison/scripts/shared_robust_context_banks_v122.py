"""Explicit constructor adaptation: replace two of Shared's five contexts.

Class slots1/4 and regression alignment slots3/4 become unlabeled quantile
transport and covariance correction. Original Base and remaining three
members stay bit-identical. No generator, source label, or query label enters
the new feature maps. Deployment still uses Base+five Native-1 contexts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')

import numpy as np
import torch

import unseen_shared_robustness_v117 as source
import shared_quantile_contexts_v122 as adaptation
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN / 'results/shared_robust_context_adaptation_v122'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [c for c in json.loads((source.OUT / 'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    plan = {
        'status': 'CONSTRUCTOR_ADAPTATION_NOT_ORIGINAL_PAPER_SHARED',
        'seed': 0, 'context_cap': 256, 'estimators_per_context': 1,
        'deployment_contexts_including_Base': 6, 'alternative_contexts': 5,
        'Base_and_three_original_alternatives_reused_bit_identical': True,
        'replaced_slots': {'classification': [1, 4], 'regression': [3, 4]},
        'new_contexts': ['numeric marginal quantile transport', 'quantile transport plus unlabeled covariance excess correction'],
        'new_alignment_receives_labels_or_mechanism_id': False,
        'categorical_value_interpolation': False,
        'numeric_only_target_PCA_rank_cap': 32, 'target_covariance_ridge_fraction': .01,
        'positive_excess_covariance_noise': True, 'unmodeled_orthogonal_residual_retained': True,
        'extra_target_labels_or_clean_source_measurements': False,
        'original_Source_X_y_and_Target_X_y_and_severities_unchanged': True,
        'new_branch_inputs_are_explicit_method_adaptations': True,
        'old_bad_and_null_results_not_overwritten': True,
        'scoring_or_new_fusion_fitting_in_this_script': False,
        'expected_conditions': sum(1+4*len(c['families']) for c in cases),
        'script_sha256': source.sha(__file__), 'alignment_sha256': source.sha(adaptation.__file__),
        'upstream_plan_sha256': source.sha(source.OUT / 'plan.json'), 'Markdown_written': False,
    }
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT / 'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    order = np.random.default_rng(122).permutation(len(cases))
    models = {}
    for i, position in enumerate(order):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in models:
            models[task] = Predictor(task, 'cuda:0')
        predictor = models[task]
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        K = len(np.unique(y0)) if task == 'classification' else None
        slots = [1, 4] if task == 'classification' else [3, 4]
        for family, rho in [('clean', 0)] + [(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT / 'banks' / task / str(tid) / f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                continue
            old = source.OUT / 'predictions/perturbation' / task / str(tid) / dest.name
            m = json.loads(old.with_suffix('.json').read_text())
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            for value, key in ((X, 'observed_source_X_sha256'), (y, 'observed_source_y_sha256'), (Q, 'observed_target_X_sha256')):
                assert protocol.array_sha256(value) == m[key]
            with np.load(old) as z:
                base, members = z['Base'], z['members'].copy()
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            aligned, corrected, details = adaptation.align(X, Q, numeric)
            cache = {protocol.array_sha256(X): base}
            calls = 0
            for slot, A in zip(slots, (aligned, corrected)):
                key = protocol.array_sha256(A)
                if key not in cache:
                    for model in predictor.models.values():
                        root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    cache[key] = predictor(A, y, Q, members=1, random_state=0, n_classes=K)[0]
                    calls += 1
                members[slot] = cache[key]
            cell = {'task': task, 'base': base, 'bank': members, 'mu': mu, 'sigma': sigma}
            policy = json.loads((source.OUT / 'policies' / task / f'{c["fold"]}.json').read_text())
            frozen_fusion = source.fusion.predict(cell, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen_fusion,
                source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_bank': str(old), 'original_bank_sha256': source.sha(old),
                'original_input_sha256': c['input_sha256'], 'replacement_slots': slots,
                'new_context_rows': [len(y), len(y)], 'new_TFM_calls': calls,
                'aligned_source_sha256': protocol.array_sha256(aligned),
                'corrected_source_sha256': protocol.array_sha256(corrected),
                'source_labels_sha256': protocol.array_sha256(y),
                'new_alignment_uses_source_or_target_labels': False,
                'evaluation_labels_loaded_for_inference': False, 'alignment_details': details})
        print('ROBUST_CONTEXT_TASK', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        log = OUT / f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}
    records = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            records.append(future.result())
            print('ROBUST_CONTEXT_WORKER', records[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'workers': records})
    success = all(x['returncode'] == 0 for x in records)
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': records})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    worker(a.shard, a.shards) if a.phase == 'worker' else launch(a.shards)
