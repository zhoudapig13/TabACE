"""Two query-local, unmodified-row contexts appended to all five v182 views.

The 64-row geometric and 128-row reliability-aware contexts can grow to retain
mandatory rare source classes. Each query cluster gets its own source subset.
This is an explicit constructor and compute-budget extension, not original
frozen Shared. Query task labels are never read.
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
import shared_source_crossfit_banks_v135 as quality
import query_cluster_contexts_v247 as constructor
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = parent.source
OUT = RUN/'results/shared_query_local_v248'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((quality.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'QUERY_LOCAL_SOURCE_SELECTION_EXTENSION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'source_pool_cap': 256, 'context_row_cap': 256, 'predictor_outputs_including_Base': 8,
        'all_five_parent_alternatives_unchanged': True,
        'new_rules': ['query_cluster_geometric64', 'query_cluster_reliability128'],
        'requested_rows_can_increase_for_rare_class_coverage_but_never_exceed_source_pool': True,
        'maximum_query_clusters': 8, 'geometry_fit': 'unlabeled query features only',
        'numeric_scaling': 'query median/IQR, std fallback, clip5',
        'high_dimensional_geometry': 'query-only randomized PCA16 if encoded dimension>32',
        'source_off_manifold_penalty': .25, 'source_loss_penalties': [0., .25],
        'source_CV_loss': 'cached v135 source-held-out NLL or source-standardized squared error',
        'source_labels_used': 'existing labels for class/response coverage and source-only CV quality',
        'source_feature_values_and_label_values_unchanged': True, 'extra_true_labeled_rows': 0,
        'query_features_used_transductively': True, 'query_task_labels_opened': False,
        'constructor_receives_family_or_severity': False, 'backbone_weights_updated': False,
        'Base_and_benchmark_and_query_cohort_unchanged': True,
        'maximum_new_query_TFM_calls_full_batch': 16,
        'maximum_query_TFM_views_full_batch_including_parents': 29,
        'maximum_TFM_views_contributing_to_each_query': 15,
        'source_CV_scoring_views': 3, 'maximum_total_views_full_batch_including_source_CV': 32,
        'compute_matched_to_Native1_Base': False,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'constructor_sha256': source.sha(constructor.__file__),
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'source_quality_plan_sha256': source.sha(quality.OUT/'plan.json'),
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
    for i, position in enumerate(np.random.default_rng(248).permutation(len(cases))):
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
            qpath = quality.OUT/'banks'/task/str(tid)/dest.name
            qmeta = json.loads(qpath.with_suffix('.json').read_text())
            assert source.sha(qpath) == qmeta['output_sha256']
            assert qmeta['original_input_sha256'] == c['input_sha256']
            with np.load(qpath) as data:
                loss = data['SourceCV_loss']
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            np.testing.assert_array_equal(y, y0)
            np.testing.assert_array_equal(Q, Q0)
            assert protocol.array_sha256(y) == qmeta['source_y_sha256']
            plans = constructor.construct(X, y, loss, task, target_model)
            extra, records, calls = [], [], 0
            for branch in plans:
                prediction = np.empty_like(base)
                details = []
                for p in branch:
                    src, query = p['source_rows'], p['query_rows']
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    prediction[query] = predictor(X[src], y[src], Q[query], members=1,
                        random_state=0, n_classes=K)[0]
                    calls += 1
                    details.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()} |
                        {'context_X_sha256': protocol.array_sha256(X[src]),
                            'context_y_sha256': protocol.array_sha256(y[src]),
                            'query_X_sha256': protocol.array_sha256(Q[query]), 'actual_context_rows': len(src)})
                assert np.isfinite(prediction).all()
                extra.append(prediction)
                records.append(details)
            members = np.concatenate([members, np.stack(extra)], axis=0)
            assert members.shape[0] == 7
            source.npwrite(dest, Base=base, members=members, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                    'original_input_sha256': c['input_sha256'], 'parent_bank': str(old), 'parent_bank_sha256': meta['output_sha256'],
                    'source_quality_bank': str(qpath), 'source_quality_bank_sha256': qmeta['output_sha256'],
                    'query_cluster_assignment_sha256': protocol.array_sha256(target_model['query_assignment']),
                    'local_context_plans': records, 'new_query_TFM_calls': calls,
                    'source_feature_values_and_labels_unchanged': True, 'query_task_labels_opened': False})
        print('QUERY_LOCAL_CASE_COMPLETE', task, tid, flush=True)


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
            print('QUERY_LOCAL_WORKER', results[-1], flush=True)
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
