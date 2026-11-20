"""Replace v122's quantile-only branch with source-label local smoothing.

Keep the covariance alignment branch, three original contexts and exact Base.
This is explicitly a supervised constructor adaptation with reliable-source-
label assumptions, not the original paper Shared. Five contexts remain.
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
import shared_robust_context_banks_v122 as previous
import label_aware_context_v124 as smoothing
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN / 'results/shared_label_smoothing_v124'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((previous.OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'SUPERVISED_CONTEXT_ADAPTATION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_cap': 256, 'alternative_contexts': 5, 'estimators_per_context': 1,
        'Base_unchanged': True, 'benchmark_unchanged': True,
        'one_branch_replaced': {'classification': 1, 'regression': 3},
        'covariance_alignment_branch_from_v122_retained': True,
        'three_original_branches_retained': True,
        'new_branch': 'source-label conditional nearest-neighbor feature smoothing',
        'source_label_values_modified': False, 'source_feature_values_modified_by_method': True,
        'uses_observed_source_labels': True, 'source_labels_assumed_reasonably_reliable': True,
        'extra_target_labels_or_clean_source_features': False,
        'numeric_smoothing_fraction': .5, 'conditional_neighbors_including_self': 5,
        'categorical_mode_threshold': .8, 'regression_response_rank_strata': 8,
        'no_severity_or_generator_identity_given_to_new_constructor': True,
        'source_bank': str(previous.OUT), 'expected_conditions': 1327,
        'script_sha256': source.sha(__file__), 'smoothing_sha256': source.sha(smoothing.__file__),
        'previous_bank_plan_sha256': source.sha(previous.OUT / 'plan.json'), 'Markdown_written': False}
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(plan))
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT / 'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    order = np.random.default_rng(124).permutation(len(cases))
    models = {}
    for i, position in enumerate(order):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in models:
            models[task] = Predictor(task, 'cuda:0')
        predictor = models[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        K = len(np.unique(y0)) if task == 'classification' else None
        slot = 1 if task == 'classification' else 3
        for family, rho in [('clean', 0)] + [(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT / 'banks' / task / str(tid) / f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                continue
            old = previous.OUT / 'banks' / task / str(tid) / dest.name
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            repaired, labels, details = smoothing.smooth(X, y, numeric, task)
            assert np.array_equal(labels, y)
            with np.load(old) as z:
                base, members = z['Base'], z['members'].copy()
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            if np.array_equal(repaired, X):
                prediction = base
            else:
                for model in predictor.models.values():
                    root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                prediction = predictor(repaired, labels, Q, members=1, random_state=0, n_classes=K)[0]
            members[slot] = prediction
            policy = json.loads((source.OUT / 'policies' / task / f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
                source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'previous_bank': str(old), 'previous_bank_sha256': source.sha(old),
                'original_input_sha256': c['input_sha256'], 'replacement_slot': slot,
                'context_rows': len(y), 'repaired_source_sha256': protocol.array_sha256(repaired),
                'source_labels_sha256': protocol.array_sha256(labels), 'source_labels_unchanged': True,
                'constructor_uses_source_labels': True, 'constructor_uses_query_labels': False,
                'query_labels_opened_for_inference': False, 'smoothing_details': details})
        print('LABEL_SMOOTH_TASK', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        env['OPENBLAS_NUM_THREADS'] = '1'
        log = OUT / f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            results.append(future.result())
            print('LABEL_SMOOTH_WORKER', results[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'workers': results})
    success = all(x['returncode'] == 0 for x in results)
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    worker(a.shard, a.shards) if a.phase == 'worker' else launch(a.shards)
