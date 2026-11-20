"""Extract source-only task reliability on a common target-like CV proxy.

No final-query task label or prediction is generated. Reuses sealed v233
source diagnostics when their context hashes and fold rules match exactly.
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

import shared_feature_forest_banks_v227 as bank
import source_proxy_repair_quality_v231 as proxy
from source_crossfit_context_v134 import folds
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = bank.source
OUT = RUN/'results/source_proxy_repair_features_v234'
PILOT = RUN/'results/source_proxy_repair_pilot_v233'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((bank.OUT/'independent_membership_audit.json').read_text())['status'] == 'PASS'
    plan = {'status': 'SOURCE_PROXY_RELIABILITY_EXTRACTION_NOT_QUERY_EVALUATION', 'seed': 0,
        'context_cap': 256, 'source_context_variants': ['observed', 'mild_repair', 'full_repair'],
        'validation_features': 'identical full-repair source-held-out input for all variants',
        'source_supervision': 'existing observed source labels only; held-out labels excluded from TFM calls',
        'no_extra_label_rows': True, 'task_query_labels_opened': False, 'new_task_query_predictions': False,
        'source_CV_folds': 3, 'rare_source_classes_below3_always_in_training_and_unscored': True,
        'source_validation_TFM_calls': 'up to9 configured-Native1 calls each condition',
        'TFM_weights_updated': False, 'benchmark_and_Base_unchanged': True,
        'feature_inputs_intended_for_next_stage': 'paired source-CV loss gain of each repair versus observed context',
        'proposed_local_branch_guard': 'enable repair only if its source-CV mean loss is lower than observed context; otherwise Base fallback',
        'guard_threshold_fixed_before_full_extraction': 0.,
        'guard_not_evaluated_on_task_query_outcomes_in_this_stage': True,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'proxy_sha256': source.sha(proxy.__file__), 'source_fold_implementation_sha256': source.sha(folds.__code__.co_filename),
        'constructor_sha256': source.sha(bank.constructor.__file__),
        'feature_bank_plan_sha256': source.sha(bank.OUT/'plan.json'),
        'pilot_plan_sha256': source.sha(PILOT/'plan.json'),
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
    for i, position in enumerate(np.random.default_rng(234).permutation(len(cases))):
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
        model = bank.constructor.fit_target(Q0, numeric)
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'features'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            np.testing.assert_array_equal(Q, Q0)
            repaired = [v['X'] for v in bank.constructor.repair_source(X, model)]
            view_hashes = [protocol.array_sha256(v) for v in [X]+repaired]
            existing = bank.OUT/'banks'/task/str(tid)/dest.name
            existing_meta = json.loads(existing.with_suffix('.json').read_text())
            assert view_hashes[1:] == [r['context_X_sha256'] for r in existing_meta['feature_repair_contexts']]
            pilot = PILOT/'features'/task/str(tid)/dest.name
            calls = 0
            reused = None
            if pilot.with_suffix('.json').exists():
                meta = json.loads(pilot.with_suffix('.json').read_text())
                assert source.sha(pilot) == meta['output_sha256']
                assert meta['source_context_X_sha256'] == view_hashes
                assert meta['source_labels_sha256'] == protocol.array_sha256(y)
                with np.load(pilot) as data:
                    result = {key: data[key] for key in ('predictions', 'loss', 'scored')}
                result['partitions'] = meta['source_partitions']
                result['mandatory_unscored_rows'] = np.asarray(meta['mandatory_unscored_rows'], dtype=int)
                reused = {'path': str(pilot), 'sha256': meta['output_sha256']}
            else:
                def infer(A, b, T):
                    nonlocal calls
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    calls += 1
                    return predictor(A, b, T, members=1, random_state=0, n_classes=K)[0]
                result = proxy.quality(X, y, repaired, task, infer)
            scored = result['scored']
            average = result['loss'][:, scored].mean(1) if scored.any() else np.zeros(3)
            gains = average[0]-average[1:]
            source.npwrite(dest, predictions=result['predictions'], loss=result['loss'], scored=scored, source_proxy_gain=gains)
            records = []
            partitions, mandatory = folds(y, task)
            assert [p.tolist() for _, p in partitions] == [r['validation_rows'] for r in result['partitions']]
            for train, valid in partitions:
                for index, view in enumerate([X]+repaired):
                    records.append({'context_index': index, 'training_rows': train.tolist(), 'validation_rows': valid.tolist(),
                        'training_X_sha256': protocol.array_sha256(view[train]), 'training_y_sha256': protocol.array_sha256(y[train]),
                        'validation_proxy_X_sha256': protocol.array_sha256(repaired[1][valid]),
                        'callback_receives_validation_labels': False})
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest), 'original_input_sha256': c['input_sha256'],
                    'feature_bank_meta': str(existing.with_suffix('.json')), 'feature_bank_sha256': existing_meta['output_sha256'],
                    'source_context_X_sha256': view_hashes, 'source_labels_sha256': protocol.array_sha256(y),
                    'source_partitions': result['partitions'], 'source_CV_expected_call_records': records,
                    'mandatory_unscored_rows': mandatory.tolist(), 'scored_source_rows': int(scored.sum()),
                    'source_proxy_loss_mean_observed_mild_full': average.tolist(), 'source_proxy_gain_mild_full': gains.tolist(),
                    'source_validation_TFM_calls_new': calls, 'source_validation_TFM_calls_required': len(records),
                    'pilot_reuse': reused, 'task_query_labels_opened': False,
                    'task_query_predictions_produced': False, 'proxy_validation_features_shared_across_contexts': True})
        print('SOURCE_PROXY_FEATURE_CASE_COMPLETE', task, tid, flush=True)


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
            print('SOURCE_PROXY_FEATURE_WORKER', results[-1], flush=True)
    count = len(list((OUT/'features').glob('*/*/*.json')))
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
