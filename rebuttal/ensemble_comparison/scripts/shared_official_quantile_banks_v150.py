"""Original Shared memberships with official quantile preprocessing adaptation.

No TFM weight updates. Original Native-1 Base is reused unchanged. A canonical
Native-1 quantile control is saved to separate normalization from fusion gains.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import torch

import unseen_shared_robustness_v117 as source
import official_quantile_predictor_v149 as configured
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import TabICLClassifier, TabICLRegressor

OUT = RUN/'results/shared_official_quantile_v150'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    library = Path(inspect.getfile(TabICLClassifier))
    files = [library, Path(inspect.getfile(TabICLRegressor)), library.with_name('preprocessing.py')]
    plan = {'status': 'OFFICIAL_NORMALIZATION_ADAPTATION_NOT_WEIGHT_FINETUNING',
        'seed': 0, 'context_cap': 256, 'Shared_alternatives': 5, 'estimators_per_context': 1,
        'Base_is_original_Native1_none_normalization': True, 'Base_predictions_unchanged': True,
        'original_five_Shared_context_memberships_and_values_retained': True,
        'adapted_official_parameter': {'norm_methods': 'quantile'},
        'normalizer_fitting': 'official per-context source-only preprocessing fit',
        'extra_control': 'same canonical context and official Native1 quantile preprocessing',
        'TFM_weights_updated': False, 'benchmark_or_query_cohort_modified': False,
        'extra_outcome_labels_or_clean_source_features_used': False,
        'new_constructor_or_extra_query_feature_reference_pool': False,
        'deployment_TFM_contexts_including_original_Base': 6,
        'extra_control_model_is_not_part_of_Shared_bank': True,
        'outlier_threshold_and_other_estimator_defaults_unchanged': True,
        'original_paper_frozen_Shared_claimed': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'predictor_configuration_sha256': source.sha(configured.__file__),
        'official_implementation_sha256': {str(p): source.sha(p) for p in files},
        'original_plan_sha256': source.sha(source.OUT/'plan.json'), 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    order = np.random.default_rng(150).permutation(len(cases))
    predictors = {}
    for i, position in enumerate(order):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = configured.create(task)
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
            old = source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
            original_meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == original_meta['output_sha256']
            with np.load(old) as z:
                base = z['Base']
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            for A, name in ((X, 'observed_source_X_sha256'), (y, 'observed_source_y_sha256'), (Q, 'observed_target_X_sha256')):
                assert protocol.array_sha256(A) == original_meta[name]
            cache = {}
            def infer(A, b, T):
                key = tuple(protocol.array_sha256(v) for v in (A, b, T))
                if key not in cache:
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    cache[key] = predictor(A, b, T, members=1, random_state=0, n_classes=K)[0]
                    generator = getattr(predictor.models[1], 'ensemble_generator_', None)
                    if generator is not None:
                        assert set(generator.ensemble_configs_) == {'quantile'}
                        assert sum(len(v) for v in generator.ensemble_configs_.values()) == 1
                assert np.isfinite(cache[key]).all()
                return cache[key]
            canonical = infer(X, y, Q)
            views = source.constructors.shared_contexts(X, y, Q, len(y), task, base)
            assert [role for role, _, _, _ in views] == original_meta['Shared_roles']
            assert [len(b) for _, _, b, _ in views] == original_meta['branch_rows']
            assert len(views) == 5 and max(len(b) for _, _, b, _ in views) <= 256
            members = np.stack([infer(A, b, T) for _, A, b, T in views])
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
                Canonical_quantile=canonical, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_bank': str(old), 'original_bank_sha256': source.sha(old),
                'original_input_sha256': c['input_sha256'], 'Shared_roles': original_meta['Shared_roles'],
                'branch_rows': original_meta['branch_rows'], 'normalization_method': 'quantile',
                'estimators_per_context': 1, 'query_outcome_labels_opened': False,
                'original_Shared_memberships_retained': True,
                'branch_input_hashes': [[protocol.array_sha256(v) for v in (A, b, T)] for _, A, b, T in views],
                'normalizer_control_saved': True, 'new_inference_calls_including_control': len(cache),
                'no_original_None_Base_used_as_quantile_alias': True})
        print('OFFICIAL_QUANTILE_COMPLETE', task, tid, flush=True)


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
            print('OFFICIAL_QUANTILE_WORKER', results[-1], flush=True)
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
