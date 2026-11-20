"""Adapt two Shared contexts using source-only cross-fitted reliability.

Base, the benchmark, and three original alternatives remain unchanged.
Three source cross-validation inference calls add compute, not new labels.
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
import source_crossfit_context_v134 as selection
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN/'results/shared_source_crossfit_v135'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'SOURCE_CROSSFIT_CONSTRUCTOR_ADAPTATION_NOT_ORIGINAL_SHARED',
        'seed': 0, 'context_cap': 256, 'alternative_contexts': 5,
        'estimators_per_context': 1, 'Base_and_three_original_alternatives_unchanged': True,
        'benchmark_and_query_cohort_and_source_pool_unchanged': True,
        'source_cross_validation_folds': 3, 'source_scoring_TFM_calls_added': 'up to3 per condition',
        'source_loss': 'held-out negative log likelihood or source-standardized squared error',
        'selection': 'keep lowest-loss75%/90% within source class or8 source-response rank strata',
        'no_source_feature_or_source_label_value_modifications': True,
        'source_rare_classes_below3': 'never source-held-out; retain all in final selected contexts',
        'replaced_slots': {'classification': [1, 4], 'regression': [3, 4]},
        'no_final_query_labels_or_clean_source_counterfactual_used': True,
        'constructor_does_not_receive_generator_or_severity': True,
        'source_crossfit_uses_only_existing_source_labels': True,
        'already_evaluated_development_cohort': True, 'independent_confirmation': False,
        'new_parameter_tuning_on_own_final_query_labels': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'selection_sha256': source.sha(selection.__file__),
        'original_plan_sha256': source.sha(source.OUT/'plan.json'), 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text())
        if c['track'] == 'perturbation']
    order = np.random.default_rng(135).permutation(len(cases))
    predictors = {}
    for i, position in enumerate(order):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        K = len(np.unique(y0)) if task == 'classification' else None
        slots = [1, 4] if task == 'classification' else [3, 4]
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            with np.load(old) as z:
                base, members = z['Base'], z['members'].copy()
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            cache = {tuple(protocol.array_sha256(v) for v in (X, y, Q)): base}
            calls = 0
            def infer(A, b, T):
                nonlocal calls
                key = tuple(protocol.array_sha256(v) for v in (A, b, T))
                if key not in cache:
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    cache[key] = predictor(A, b, T, members=1, random_state=0, n_classes=K)[0]
                    calls += 1
                assert np.isfinite(cache[key]).all()
                return cache[key]
            loss, source_prediction, mandatory, partitions = selection.quality(X, y, task, infer)
            selected = []
            for slot, fraction in zip(slots, (.75, .9)):
                ids = selection.select(y, loss, task, fraction)
                assert len(np.unique(ids)) == len(ids) and len(ids) <= 256
                assert set(mandatory).issubset(ids)
                members[slot] = infer(X[ids], y[ids], Q)
                selected.append(ids.tolist())
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
                source_mean=mu, source_scale=sigma, SourceCV_loss=loss,
                SourceCV_prediction=source_prediction)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_bank': str(old), 'original_bank_sha256': source.sha(old),
                'original_input_sha256': c['input_sha256'], 'replacement_slots': slots,
                'new_context_rows': [len(ids) for ids in selected], 'selected_source_rows': selected,
                'source_crossfit_partitions': partitions, 'mandatory_unscored_rare_rows': mandatory.tolist(),
                'source_CV_prediction_NaN_rows_are_unscored_mandatory_rare_rows': True,
                'source_features_and_labels_preserved_in_selected_contexts': True,
                'source_y_sha256': protocol.array_sha256(y), 'TFM_calls_added': calls,
                'source_CV_uses_source_labels': True, 'final_query_labels_opened': False,
                'new_constructor_receives_generator_or_strength': False})
        print('SOURCE_CROSSFIT_COMPLETE', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '1'
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            process = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': process.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            results.append(future.result())
            print('SOURCE_CROSSFIT_WORKER', results[-1], flush=True)
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results})
    success = all(r['returncode'] == 0 for r in results)
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    launch(args.shards) if args.phase == 'launch' else worker(args.shard, args.shards)
