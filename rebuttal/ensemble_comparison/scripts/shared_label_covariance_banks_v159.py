"""Source-label-assisted posterior repair replaces the graphical context."""
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
import shared_graphical_banks_v147 as parent
import source_label_covariance_v158 as denoising
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN/'results/shared_label_covariance_v159'


def freeze():
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'EXPLICIT_SOURCE_LABEL_COVARIANCE_CONSTRUCTOR_ADAPTATION',
        'seed': 0, 'context_cap': 256, 'alternative_contexts': 5, 'estimators_per_context': 1,
        'Base_and_three_original_alternatives_unchanged': True,
        'v122_marginal_quantile_transport_branch_retained': True,
        'parent_graphical_branch_replaced': True, 'replacement_slot': 4,
        'new_constructor': 'source-label-assisted Gaussian measurement-error posterior',
        'source_task_labels_used_by_new_feature_map': True, 'categorical_columns_modified': False,
        'target_covariance_and_source_label_joint_fit': True,
        'true_query_outcomes_or_clean_source_counterfactual_used': False,
        'unlabeled_query_feature_batch_required': True, 'deployment_is_transductive': True,
        'generator_or_strength_given_to_new_constructor': False,
        'source_label_values_unchanged': True, 'benchmark_or_query_cohort_changed': False,
        'same_six_Native1_context_predictions_for_deployment': True,
        'feature_model_computation_additional': 'CPU source-label and unlabeled-target covariance fit; same source labels as Base; no query outcomes',
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'denoising_helper_sha256': source.sha(denoising.__file__),
        'parent_bank_plan_sha256': source.sha(parent.OUT/'plan.json'), 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    order = np.random.default_rng(159).permutation(len(cases))
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
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            previous = parent.OUT/'banks'/task/str(tid)/dest.name
            assert source.sha(previous) == json.loads(previous.with_suffix('.json').read_text())['output_sha256']
            with np.load(previous) as z:
                base, members = z['Base'], z['members'].copy()
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            repaired, details = denoising.repair(X, y, Q, numeric, task)
            if np.array_equal(repaired, X):
                prediction = base
            else:
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                prediction = predictor(repaired, y, Q, members=1, random_state=0, n_classes=K)[0]
            members[4] = prediction
            assert np.isfinite(members).all()
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
                source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'parent_bank': str(previous), 'parent_bank_sha256': source.sha(previous),
                'original_input_sha256': c['input_sha256'], 'replacement_slot': 4,
                'context_rows': len(y), 'source_labels_sha256': protocol.array_sha256(y),
                'new_context_source_features_sha256': protocol.array_sha256(repaired),
                'new_feature_constructor_uses_source_task_labels': True,
                'query_outcome_labels_opened': False, 'deployment_is_transductive': True,
                'denoising_details': details})
        print('LABEL_COVARIANCE_COMPLETE', task, tid, flush=True)


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
            print('LABEL_COVARIANCE_WORKER', results[-1], flush=True)
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
