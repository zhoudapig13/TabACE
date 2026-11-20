"""A fixed five-alternative bank with two source-supervised feature contexts.

This is a constructor extension, not a reproduction of original frozen Shared.
Existing observed source labels only; no query outcomes are loaded here.
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
import supervised_projection_contexts_v214 as constructor
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = parent.source
OUT = RUN/'results/shared_supervised_projection_v215'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'SOURCE_SUPERVISED_PROJECTION_EXTENSION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_row_cap': 256, 'final_predictor_outputs_including_Base': 6,
        'replaced_slots': [3, 4], 'first_three_alternative_predictions_unchanged': True,
        'constructor_rules': ['robust_scaled_source_PLS', 'marginal_rank_source_PLS'],
        'source_response': 'classification one-hot or regression standardized existing source labels',
        'latent_dimensions': 'min(4, active source dimensions, source rows minus1)',
        'source_rows_and_labels_unchanged': True, 'extra_true_labeled_rows': 0,
        'query_features_used_transductively': True, 'query_task_labels_used': False,
        'constructor_receives_family_or_strength': False,
        'backbone_weights_updated': False, 'Base_and_benchmark_unchanged': True,
        'source_CV_TFM_calls_required_by_retained_contexts': 0,
        'maximum_prediction_estimator_passes': 13, 'compute_matched_to_Native1_Base': False,
        'new_TFM_calls_per_condition': 2, 'configured_estimators_each_new_context': 1,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'constructor_sha256': source.sha(constructor.__file__),
        'encoding_implementation_sha256': source.sha(constructor.encode.__code__.co_filename),
        'parent_bank_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'original_plan_sha256': source.sha(source.OUT/'plan.json'),
        'Markdown_written': False}
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
    for i, position in enumerate(np.random.default_rng(215).permutation(len(cases))):
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
            X0, y0, Q0, numeric = (z[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            old_meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == old_meta['output_sha256']
            assert old_meta['original_input_sha256'] == c['input_sha256']
            with np.load(old) as z:
                base, members = z['Base'], z['members'].copy()
                native8 = z['Canonical_native8']
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            contexts = constructor.construct(X, y, Q, numeric, task)
            details = []
            for slot, context in zip((3, 4), contexts):
                np.testing.assert_array_equal(context['y'], y)
                assert len(context['y']) <= 256
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                members[slot] = predictor(context['X'], context['y'], context['Q'], members=1,
                    random_state=0, n_classes=K)[0]
                details.append(context['info'] | {'context_rows': len(context['y']),
                    'context_X_sha256': protocol.array_sha256(context['X']),
                    'context_y_sha256': protocol.array_sha256(context['y']),
                    'projected_query_X_sha256': protocol.array_sha256(context['Q'])})
            assert np.isfinite(members).all()
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_input_sha256': c['input_sha256'], 'parent_bank': str(old),
                'parent_bank_sha256': source.sha(old), 'supervised_contexts': details,
                'new_TFM_calls': 2, 'source_labels_sha256': protocol.array_sha256(y),
                'final_query_outcome_files_opened': False, 'source_labels_used_for_projection': True,
                'new_true_labeled_rows': 0})
        print('SUPERVISED_PROJECTION_CASE', task, tid, flush=True)


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
            print('SUPERVISED_PROJECTION_WORKER', values[-1], flush=True)
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
