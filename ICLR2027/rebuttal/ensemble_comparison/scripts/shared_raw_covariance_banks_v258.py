"""Two raw excess-covariance contexts supplement the preserved v182 bank.

Same source rows and labels, frozen TFM weights, unchanged benchmark. The
source feature values change only inside the two explicit new contexts.
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

import shared_dual_canonical_banks_v182 as parent
import raw_excess_covariance_contexts_v257 as constructor
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = parent.source
OUT = RUN/'results/shared_raw_covariance_v258'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'RAW_EXCESS_COVARIANCE_CONTEXT_EXTENSION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_row_cap': 256, 'predictor_outputs_including_Base': 8,
        'all_five_parent_alternatives_unchanged': True,
        'new_rules': 'raw excess covariance contraction, gains0.5 and1.0',
        'CDF_mapping_before_noise_covariance_estimation': False,
        'raw_source_covariance_versus_unlabeled_query_covariance': True,
        'numeric_target_PCA_rank_cap': 32, 'covariance_floor_fraction': .01,
        'source_task_labels_used_by_transform': False, 'query_task_labels_used': False,
        'source_label_values_and_source_row_membership_unchanged': True,
        'categorical_feature_values_unchanged': True, 'extra_true_labeled_rows': 0,
        'query_features_used_transductively': True, 'constructor_receives_family_or_strength': False,
        'new_query_TFM_calls_each_condition': 2, 'maximum_query_estimator_views': 15,
        'parent_source_CV_TFM_views': {'classification': 0, 'regression': 3},
        'maximum_total_views_including_parent_source_CV': {'classification': 15, 'regression': 18},
        'compute_matched_to_Native1_Base': False, 'backbone_weights_updated': False,
        'Base_and_benchmark_and_query_cohort_unchanged': True,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'constructor_sha256': source.sha(constructor.__file__),
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
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
    predictors = {}
    for i, position in enumerate(np.random.default_rng(258).permutation(len(cases))):
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
        target_model = constructor.fit_target(Q0, numeric)
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            with np.load(old) as data:
                base, members = data['Base'], data['members'].copy()
                native8, frozen = data['Canonical_native8'], data['FrozenFusion']
                mu, sigma = float(data['source_mean']), float(data['source_scale'])
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            np.testing.assert_array_equal(y, y0)
            np.testing.assert_array_equal(Q, Q0)
            predictions, records = [], []
            for context in constructor.construct(X, target_model):
                assert len(context['X']) == len(y) <= 256
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                predictions.append(predictor(context['X'], y, Q, members=1, random_state=0, n_classes=K)[0])
                records.append(context['info'] | {'context_rows': len(y),
                    'context_X_sha256': protocol.array_sha256(context['X']),
                    'context_y_sha256': protocol.array_sha256(y), 'query_X_sha256': protocol.array_sha256(Q)})
            members = np.concatenate([members, np.stack(predictions)], axis=0)
            assert members.shape[0] == 7 and np.isfinite(members).all()
            source.npwrite(dest, Base=base, members=members, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                    'original_input_sha256': c['input_sha256'], 'parent_bank': str(old), 'parent_bank_sha256': meta['output_sha256'],
                    'raw_covariance_contexts': records, 'new_query_TFM_calls': 2,
                    'task_query_labels_opened': False, 'source_label_values_preserved': True})
        print('RAW_COVARIANCE_CASE_COMPLETE', task, tid, flush=True)


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
            print('RAW_COVARIANCE_WORKER', results[-1], flush=True)
    count = len(list((OUT/'banks').glob('*/*/*.json')))
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
