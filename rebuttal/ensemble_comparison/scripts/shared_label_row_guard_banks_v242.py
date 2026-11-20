"""Append two source-label-consistent repaired contexts to all five v182 views.

Held-out SOURCE labels decide individual source feature edits. They never enter
the corresponding source-fold model fit. No true task-query label is loaded.
This is a new supervised constructor adaptation, not original frozen Shared.
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

import shared_feature_forest_banks_v227 as forest
import source_label_row_repair_guard_v241 as guard
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

parent, source = forest.parent, forest.source
OUT = RUN/'results/shared_label_row_guard_v242'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((forest.OUT/'independent_membership_audit.json').read_text())['status'] == 'PASS'
    plan = {'status': 'SOURCE_LABEL_ROW_GUARD_CONSTRUCTOR_EXTENSION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_row_cap': 256, 'predictor_outputs_including_Base': 8,
        'all_five_parent_alternatives_unchanged': True,
        'feature_repair_fit': 'unlabeled query input features only; frozen v226 forest parameters',
        'source_row_guard': 'keep repaired row iff it lowers held-out source-label loss under the SAME observed-context source-fold model',
        'source_probe_inputs': 'concatenate observed,mild,full views of same source validation rows',
        'source_probe_training_features': 'observed source only; no label-dependent edited rows',
        'source_probe_training_labels': 'source-fold training labels only; held-out source labels excluded',
        'source_labels_used_after_probe': 'choose source feature edits; existing train labels only',
        'rare_source_classes_below3': 'always in source-probe training; no scored edits',
        'label_values_and_row_membership_preserved': True, 'extra_true_labeled_rows': 0,
        'task_query_labels_opened': False, 'unlabeled_query_features_used_transductively': True,
        'row_guard_threshold': 0., 'threshold_selected_from_task_query_results': False,
        'new_query_TFM_calls_per_condition': 2, 'source_probe_TFM_calls_each_condition': 'up to3',
        'maximum_query_TFM_estimator_views': 15,
        'additional_parent_source_CV_views': {'classification': 0, 'regression': 3},
        'maximum_all_TFM_views_including_source_CV': {'classification': 18, 'regression': 21},
        'backbone_weights_updated': False, 'compute_matched_to_Native1_Base': False,
        'Base_and_benchmark_and_query_cohort_unchanged': True,
        'constructor_receives_family_or_strength': False,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'row_guard_sha256': source.sha(guard.__file__), 'source_fold_sha256': source.sha(guard.folds.__code__.co_filename),
        'forest_constructor_sha256': source.sha(forest.constructor.__file__),
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
    for i, position in enumerate(np.random.default_rng(242).permutation(len(cases))):
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
        fitted = forest.constructor.fit_target(Q0, numeric)
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
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
            repaired = [v['X'] for v in forest.constructor.repair_source(X, fitted)]
            raw_forest_meta = json.loads((forest.OUT/'banks'/task/str(tid)/dest.name).with_suffix('.json').read_text())
            assert [protocol.array_sha256(v) for v in repaired] == [r['context_X_sha256'] for r in raw_forest_meta['feature_repair_contexts']]
            calls = 0
            def infer(A, b, T):
                nonlocal calls
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                calls += 1
                return predictor(A, b, T, members=1, random_state=0, n_classes=K)[0]
            result = guard.construct(X, y, repaired, task, infer)
            probe_calls = calls
            extra = [infer(context, y, Q) for context in result['contexts']]
            members = np.concatenate([members, np.stack(extra)], axis=0)
            assert len(members) == 7 and np.isfinite(members).all()
            source.npwrite(dest, Base=base, members=members, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma,
                SourceProbe_predictions=result['predictions'], SourceProbe_loss=result['loss'],
                SourceProbe_scorable=result['scorable'], SourceRow_edit_accepted=result['accepted'])
            records = []
            for partition in result['partitions']:
                train, valid = np.asarray(partition['training_rows']), np.asarray(partition['validation_rows'])
                probe = np.concatenate([v[valid] for v in [X]+repaired], axis=0)
                records.append(partition | {'training_X_sha256': protocol.array_sha256(X[train]),
                    'training_y_sha256': protocol.array_sha256(y[train]),
                    'probe_X_sha256': protocol.array_sha256(probe),
                    'callback_received_validation_labels': False})
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                    'original_input_sha256': c['input_sha256'], 'parent_bank': str(old), 'parent_bank_sha256': meta['output_sha256'],
                    'context_X_sha256': [protocol.array_sha256(v) for v in result['contexts']],
                    'raw_repaired_X_sha256': [protocol.array_sha256(v) for v in repaired],
                    'source_y_sha256': protocol.array_sha256(y), 'query_X_sha256': protocol.array_sha256(Q),
                    'source_probe_partitions': records, 'mandatory_unscored_rows': result['mandatory_unscored_rows'].tolist(),
                    'accepted_source_row_ids': [np.flatnonzero(v).tolist() for v in result['accepted']],
                    'source_probe_TFM_calls': probe_calls, 'new_query_TFM_calls': calls-probe_calls,
                    'source_labels_used_for_feature_edit_choice': True, 'query_task_labels_opened': False})
        print('LABEL_ROW_GUARD_CASE_COMPLETE', task, tid, flush=True)


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
            print('LABEL_ROW_GUARD_WORKER', results[-1], flush=True)
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
