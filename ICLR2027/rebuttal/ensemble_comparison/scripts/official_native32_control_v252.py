"""Official Native32 descriptive control on the unchanged v117 source inputs.

Configured-estimator views approximate the local-context full-batch budget;
they are not FLOPs or latency matching. No query labels or Shared adaptation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import torch

import unseen_shared_robustness_v117 as source
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN/'results/official_native32_control_v252'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'OFFICIAL_NATIVE32_DESCRIPTIVE_COMPUTE_CONTROL', 'seed': 0,
        'configured_estimators': 32, 'official_default_normalization': True,
        'same_observed_source_rows_and_labels_as_Base': True, 'source_cap': 256,
        'no_Shared_constructor_or_fusion_adaptation': True, 'query_task_labels_opened': False,
        'backbone_weights_updated': False, 'new_primary_hypothesis_tests': 0,
        'control_is_descriptive': True, 'not_exact_FLOPs_or_latency_matching': True,
        'actual_views_can_be_fewer_with_limited_feature_or_class_permutations': True,
        'degenerate_input_predictor_fallback_has_zero_TFM_views': True,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'predictor_sha256': source.sha(sys.modules[Predictor.__module__].__file__),
        'original_plan_sha256': source.sha(source.OUT/'plan.json'),
        'previously_scored_development_cohort': True, 'independent_confirmation': False, 'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    predictors = {}
    for i, position in enumerate(np.random.default_rng(252).permutation(len(cases))):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as data:
            assert 'y_target' not in data.files
            X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
            meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            for A, key in [(X, 'observed_source_X_sha256'), (y, 'observed_source_y_sha256'), (Q, 'observed_target_X_sha256')]:
                assert protocol.array_sha256(A) == meta[key]
            for estimator in predictor.models.values():
                root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                if root is not None and hasattr(root, 'root'):
                    root.root = None
            torch.cuda.empty_cache()
            prediction = predictor(X, y, Q, members=32, random_state=0, n_classes=K)[0]
            fallback = X.shape[1] == 0 or np.all(np.ptp(X.astype(np.float64), axis=0) == 0)
            fallback = fallback or (len(np.unique(y)) == 1 if task == 'classification' else float(y.std()) < 1e-12)
            if fallback:
                counts, actual = {}, 0
            else:
                model = predictor.models[32]
                assert model.get_params()['n_estimators'] == 32 and model.get_params()['norm_methods'] is None
                counts = {key: len(value) for key, value in model.ensemble_generator_.ensemble_configs_.items()}
                actual = sum(counts.values())
                assert 1 <= actual <= 32 and set(counts).issubset({'none', 'power'})
            with np.load(old) as data:
                base, mu, sigma = data['Base'], data['source_mean'], data['source_scale']
            source.npwrite(dest, Base=base, Native32=prediction, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                    'original_input_sha256': c['input_sha256'], 'original_bank': str(old), 'original_bank_sha256': meta['output_sha256'],
                    'input_hashes': [protocol.array_sha256(v) for v in (X, y, Q)],
                    'configured_estimators': 32, 'actual_estimator_views': actual, 'actual_normalization_views': counts,
                    'degenerate_input_fallback': bool(fallback), 'query_task_labels_opened': False})
        print('OFFICIAL_NATIVE32_CASE_COMPLETE', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)],
                env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            results.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results})
            print('OFFICIAL_NATIVE32_WORKER', results[-1], flush=True)
    count = len(list((OUT/'predictions').glob('*/*/*.json')))
    success = all(r['returncode'] == 0 for r in results) and count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results, 'conditions': count})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    launch(args.shards) if args.phase == 'launch' else worker(args.shard, args.shards)
