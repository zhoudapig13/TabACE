"""Append two unlabeled-query feature-repair contexts to all five v182 views.

This is a new Shared constructor extension, not original Shared. Each query
feature model is trained once and reused across the fixed source-only
perturbations. True query outcomes are neither loaded nor accepted.
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
import sklearn
import torch

import shared_dual_canonical_banks_v182 as parent
import target_feature_forest_repair_v226 as constructor
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = parent.source
OUT = RUN/'results/shared_feature_forest_v227'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'UNLABELED_QUERY_FEATURE_REPAIR_EXTENSION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_row_cap': 256, 'raw_predictor_outputs_including_Base': 8,
        'all_five_parent_alternatives_retained': True, 'two_extra_alternatives': ['forest_feature_repair_mild', 'forest_feature_repair_full'],
        'maximum_prediction_estimator_passes': 15, 'extra_source_CV_TFM_calls': {'classification': 0, 'regression': 3},
        'compute_matched_to_Native1_Base': False,
        'feature_repair_training': 'query input features only; each input feature predicted from other input features',
        'true_task_labels_used_for_feature_repair': False, 'source_labels_modified': False,
        'extra_true_labeled_rows': 0, 'query_features_used_transductively': True,
        'query_task_outcome_files_opened': False, 'constructor_receives_family_or_strength': False,
        'conditional_repair_uses_three_fold_feature_CV': True,
        'feature_forest_estimators': constructor.TREES, 'maximum_feature_responses': constructor.MAX_RESPONSES,
        'maximum_encoded_predictors_per_response': constructor.MAX_PREDICTORS,
        'new_TFM_calls_each_condition': 2, 'TFM_estimators_each_new_context': 1,
        'numeric_gains': [.5, 1.], 'categorical_codes_never_interpolated': True,
        'Base_and_benchmark_and_query_cohort_unchanged': True, 'backbone_weights_updated': False,
        'feature_models_reused_only_when_query_features_bitwise_unchanged': True,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'sklearn_version': sklearn.__version__, 'expected_conditions': 1327,
        'script_sha256': source.sha(__file__), 'constructor_sha256': source.sha(constructor.__file__),
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'original_plan_sha256': source.sha(source.OUT/'plan.json'), 'Markdown_written': False}
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
    for i, position in enumerate(np.random.default_rng(227).permutation(len(cases))):
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
        print('FEATURE_REPAIR_FIT_START', task, tid, Q0.shape, flush=True)
        fitted = constructor.fit_target(Q0, numeric)
        model_info = fitted['info'] | {'query_X_sha256': protocol.array_sha256(Q0),
            'original_input_sha256': c['input_sha256'], 'query_task_labels_used': False}
        model_path = OUT/'feature_models'/task/f'{tid}.json'
        if model_path.exists():
            assert json.loads(model_path.read_text()) == model_info
        else:
            atomic_json(model_path, model_info)
        model_hash = source.sha(model_path)
        print('FEATURE_REPAIR_FIT_DONE', task, tid, len(fitted['models']), flush=True)
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            with np.load(old) as raw:
                base, members = raw['Base'], raw['members'].copy()
                native8, frozen = raw['Canonical_native8'], raw['FrozenFusion']
                mu, sigma = float(raw['source_mean']), float(raw['source_scale'])
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            np.testing.assert_array_equal(Q, Q0)
            np.testing.assert_array_equal(y, y0)
            predictions, details = [], []
            for context in constructor.repair_source(X, fitted):
                assert len(context['X']) == len(y) <= 256
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                predictions.append(predictor(context['X'], y, Q, members=1, random_state=0, n_classes=K)[0])
                details.append(context['info'] | {'context_rows': len(y),
                    'context_X_sha256': protocol.array_sha256(context['X']),
                    'context_y_sha256': protocol.array_sha256(y), 'query_X_sha256': protocol.array_sha256(Q)})
            members = np.concatenate([members, np.stack(predictions)], axis=0)
            assert len(members) == 7 and np.isfinite(members).all()
            source.npwrite(dest, Base=base, members=members, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                    'original_input_sha256': c['input_sha256'], 'parent_bank': str(old),
                    'parent_bank_sha256': meta['output_sha256'], 'feature_repair_contexts': details,
                    'feature_model_info': str(model_path), 'feature_model_info_sha256': model_hash,
                    'new_TFM_calls': 2, 'query_task_labels_opened': False,
                    'source_label_values_unchanged': True})
        print('FEATURE_REPAIR_CASE_COMPLETE', task, tid, flush=True)


def launch(shards):
    freeze()

    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '1'
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)],
                env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}

    values = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            values.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': values})
            print('FEATURE_REPAIR_WORKER', values[-1], flush=True)
    count = len(list((OUT/'banks').glob('*/*/*.json')))
    success = all(p['returncode'] == 0 for p in values) and count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': values, 'conditions': count})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    launch(args.shards) if args.phase == 'launch' else worker(args.shard, args.shards)
