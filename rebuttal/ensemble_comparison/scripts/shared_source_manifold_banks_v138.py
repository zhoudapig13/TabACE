"""Explicit transductive context adaptation using source-CV and pseudo-labels."""
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
import shared_source_crossfit_banks_v135 as parent
import source_crossfit_manifold_v137 as adaptation
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN/'results/shared_source_manifold_v138'


def freeze():
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'TRANSDUCTIVE_PSEUDOLABEL_CONTEXT_ADAPTATION_NOT_ORIGINAL_SHARED',
        'seed': 0, 'context_cap': 256, 'alternative_contexts': 5, 'estimators_per_context': 1,
        'Base_and_three_original_context_predictions_unchanged': True,
        'benchmark_source_pool_and_query_cohort_unchanged': True,
        'new_contexts': 'source-CV gated target-neighbor feature repair; maximum blends0.5/0.9',
        'new_context_rows': 'all existing source rows; original source label values unchanged',
        'uses_existing_source_labels': True, 'uses_unlabeled_final_query_features': True,
        'uses_Base_pseudo_labels_on_query_features': True,
        'uses_true_final_query_labels_or_clean_source_counterfactual': False,
        'deployment_is_transductive': True, 'inductive_original_Shared_claimed': False,
        'source_crossfit_scoring_TFM_calls_added': 'up to3; cached from v135',
        'TFM_calls_for_new_contexts': 'up to2 additional; replace2 of five contexts',
        'replaced_slots': {'classification': [1, 4], 'regression': [3, 4]},
        'source_label_reliability_assumption': True,
        'own_evaluation_y_used_for_choosing_pseudo_labels_or_blends': False,
        'generator_or_strength_given_to_constructor': False,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'adaptation_sha256': source.sha(adaptation.__file__),
        'source_crossfit_bank_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    order = np.random.default_rng(138).permutation(len(cases))
    predictors = {}
    for index, position in enumerate(order):
        if index % shards != shard:
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
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            assert source.sha(old) == json.loads(old.with_suffix('.json').read_text())['output_sha256']
            with np.load(old) as z:
                base, members = z['Base'], z['members'].copy()
                loss = z['SourceCV_loss']
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            repaired, labels, details = adaptation.repair(X, y, Q, base, numeric, loss, task)
            np.testing.assert_array_equal(labels, y)
            cache = {protocol.array_sha256(X): base}
            for slot, A in zip(slots, repaired):
                key = protocol.array_sha256(A)
                if key not in cache:
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    cache[key] = predictor(A, labels, Q, members=1, random_state=0, n_classes=K)[0]
                members[slot] = cache[key]
            assert np.isfinite(members).all()
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
                source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'source_crossfit_bank': str(old), 'source_crossfit_bank_sha256': source.sha(old),
                'original_input_sha256': c['input_sha256'], 'replacement_slots': slots,
                'context_rows': len(y), 'source_labels_sha256': protocol.array_sha256(labels),
                'source_labels_unchanged': True, 'constructor_uses_true_query_labels': False,
                'constructor_uses_Base_query_pseudo_labels': True, 'deployment_is_transductive': True,
                'constructor_receives_generator_or_strength': False,
                'new_source_feature_hashes': [protocol.array_sha256(A) for A in repaired],
                'adaptation_details': details})
        print('SOURCE_MANIFOLD_COMPLETE', task, tid, flush=True)


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
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results})
            print('SOURCE_MANIFOLD_WORKER', results[-1], flush=True)
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
