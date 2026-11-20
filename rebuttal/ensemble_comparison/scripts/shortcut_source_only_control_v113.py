"""Separate source-only D variant, not the equal-target-supervision ABCD D.

Drop the extra 64 labeled target calibration rows from adaptation/contexts.
Keep the frozen benchmark, source labels, evaluation rows, and feature set.
Use the previously source-calibrated D policy that excludes this dataset.
No parameter or severity search is performed for an accuracy threshold.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch

import semisynthetic_ranking_calibration_v96 as prior
import target_calibration_split_v97 as split
from fit_combiners import fold_mapping
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, preserving_bootstrap, metrics
from shared_reference import contexts

OUT = RUN / 'results/shortcut_source_only_control_v113'
OLD = RUN / 'results/abcd_transfer_calibration_v109'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        'status': 'SEPARATE_SOURCE_ONLY_SUPERVISION_ABLATION_NOT_ORIGINAL_D',
        'datasets': 72, 'seed': 0, 'context_cap': 256,
        'extra_labeled_target_rows': 0, 'original_D_extra_labeled_target_rows': 64,
        'D_dagger_is_not_equal_supervision_comparator_to_A_B_C_D': True,
        'source_only_prior': str(prior.OUT),
        'source_only_prior_training_excludes_own_dataset_and_Shortcut_equivalents': True,
        'prior_selection_was_AUC_on_other_dataset_calibration_not_current_threshold': True,
        'five_alternative_contexts': ['target_neighbor', 'density_resampled',
            'locally_stable', 'full_feature_bootstrap', 'low_conflict'],
        'all_feature_columns_retained': True, 'source_labels_unchanged': True,
        'unlabeled_query_support': 'same calibration+evaluation feature union as target-aware banks',
        'only_cached_Base_inference_reused': True, 'new_alternative_branch_inference': True,
        'benchmark_or_backbone_weights_changed': False,
        'training_on_evaluation_labels': False, 'new_hyperparameter_or_severity_search': False,
        'target_labels_used_for_semisynthetic_benchmark_generation': True,
        'original_paper_replication': False, 'independent_confirmation': False,
        'source_inputs': str(split.OUT / 'inputs'), 'Markdown_written': False,
        'code_sha256': sha(__file__),
    }
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    mapping, _ = fold_mapping('classification', 'five')
    predictor = Predictor('classification', 'cuda:0')

    def infer(A, b, Q, K):
        for model in predictor.models.values():
            root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
            if root is not None and hasattr(root, 'root'):
                root.root = None
        torch.cuda.empty_cache()
        return predictor(A, b, Q, n_classes=K, members=1, random_state=0)[0]

    for index, path in enumerate(sorted((split.OUT / 'inputs').glob('*.npz'))):
        if index % shards != shard:
            continue
        # Intentionally never load the input archive's y_calibration field.
        with np.load(path) as z:
            data = {k: z[k] for k in ['y_source', 'X_calibration', 'X_evaluation'] +
                [f'X_source_{r}' for r in (0, 20, 50, 80, 100)]}
        meta = json.loads(path.with_suffix('.json').read_text())
        tid, did = int(meta['task_id']), int(meta['data_id'])
        assert sha(path) == meta['model_inputs_sha256']
        fold = mapping[did]
        policy_file = prior.OUT / 'folds/classification' / str(fold) / 'D_policy.json'
        saved = json.loads(policy_file.read_text())
        assert did in saved['excluded_data_ids'] and did not in saved['training_data_ids']
        assert saved['unseen_mechanism_audit']
        policy = saved['policy']
        union = np.vstack([data['X_calibration'], data['X_evaluation']])
        offset = len(data['X_calibration'])
        y = data['y_source']
        K = len(np.unique(y))
        for rho in (0, 20, 50, 80, 100):
            out = OUT / 'banks' / str(tid) / f'{rho}.npz'
            if out.with_suffix('.json').exists():
                continue
            X = data[f'X_source_{rho}']
            views = contexts(X, y, union, X, y, 256, 'classification')
            # The matched feature-deletion constructor is never inferred.
            ids = preserving_bootstrap(y, np.random.default_rng(0))
            views[3] = ('full_feature_bootstrap', X[ids], y[ids], union)
            A, b, density_rows, density_columns = prior.source.density_context(X, y, union)
            views[1] = ('density_resampled', A, b, union)
            assert all(U.shape[1] == X.shape[1] and Q.shape[1] == X.shape[1]
                and len(v) <= 256 for _, U, v, Q in views)
            members = np.stack([infer(U, v, Q, K)[offset:] for _, U, v, Q in views])
            with np.load(OLD / 'classification_predictions/old_development' / str(tid) /
                    f'{rho}_Base_zero.npz') as z:
                base = z['prediction']
            cell = {'task': 'classification', 'base': base, 'bank': members,
                'meta': {'scale': 1.}, 'mu': float(y.mean()), 'sigma': max(float(y.std()), 1e-9)}
            prediction = prior.predict(cell, policy)
            assert prediction.shape == base.shape and np.isfinite(prediction).all()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, prediction=prediction, base=base, members=members)
            tmp.replace(out)
            atomic_json(out.with_suffix('.json'), {
                'task_id': tid, 'data_id': did, 'rho': rho, 'outer_fold': fold,
                'status': 'COMPLETE', 'source_only_D_dagger_not_original_D': True,
                'extra_target_label_rows': 0, 'source_rows': len(y),
                'source_X_sha256': protocol.array_sha256(X),
                'source_y_sha256': protocol.array_sha256(y),
                'evaluation_X_sha256': protocol.array_sha256(data['X_evaluation']),
                'policy_file': str(policy_file), 'policy_sha256': sha(policy_file),
                'prior_excludes_evaluated_dataset': True,
                'branch_context_rows': [len(v) for _, _, v, _ in views],
                'branch_names': [name for name, _, _, _ in views],
                'all_columns_retained': True, 'density_rows': density_rows.tolist(),
                'density_scoring_columns': density_columns.tolist(),
                'input_file_sha256': sha(path), 'output_sha256': sha(out),
                'evaluation_labels_loaded_for_inference': False,
            })
        print('SOURCE_ONLY_TASK_COMPLETE', tid, flush=True)
    return shard


def launch(shards):
    freeze()

    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT / f'worker_{i}.log'
        with log.open('a') as handle:
            result = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': result.returncode, 'log': str(log)}

    records = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            records.append(future.result())
            print(records[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'workers': records})
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records)
        else 'FAILED', 'workers': records})
    assert all(r['returncode'] == 0 for r in records)


def report():
    assert json.loads((OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((OUT / 'plan.json').read_text())['code_sha256'] == sha(__file__)
    rows = []
    for path in sorted((OUT / 'banks').glob('*/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == meta['output_sha256']
        assert sha(meta['policy_file']) == meta['policy_sha256']
        assert meta['extra_target_label_rows'] == 0 and meta['all_columns_retained']
        with np.load(split.OUT / 'evaluation_truth' / f'{meta["task_id"]}.npz') as z:
            y = z['y_evaluation']
        with np.load(path) as z:
            prediction, base, members = z['prediction'], z['base'], z['members']
        p = json.loads(Path(meta['policy_file']).read_text())['policy']
        expected = prior.predict({'task': 'classification', 'base': base, 'bank': members,
            'meta': {'scale': 1.}}, p)
        assert np.array_equal(prediction, expected)
        for name, q in (('D_source_only_dagger', prediction), ('Base_zero_target_labels', base)):
            rows.append({'task_id': meta['task_id'], 'data_id': meta['data_id'],
                'rho': meta['rho'], 'method': name, **metrics(y, q, 'classification', 1.)})
    detail = pd.DataFrame(rows)
    assert len(detail) == 720 and detail.task_id.nunique() == 72
    detail.to_csv(OUT / 'detail.csv', index=False)
    macro = detail[detail.rho > 0].groupby(['data_id', 'method'])[['accuracy', 'auc']].mean()
    clean = detail[detail.rho == 0].groupby('method')[['accuracy', 'auc']].mean()
    table = []
    for method in ('D_source_only_dagger', 'Base_zero_target_labels'):
        v = macro.xs(method, level='method')
        table.append({'method': method, 'datasets': len(v), 'accuracy': v.accuracy.mean(),
            'auc': v.auc.mean(), 'clean_accuracy': clean.loc[method, 'accuracy'],
            'clean_auc': clean.loc[method, 'auc'], 'target_labels': 0,
            'same_target_supervision_as_original_ABCD': False})
    pd.DataFrame(table).to_csv(OUT / 'tables.csv', index=False)
    # Check reused Base exactly against its previous aggregate.
    old = pd.read_csv(OLD / 'classification_tables.csv')
    old_base = old[(old.cohort == 'old_development') & (old.method == 'Base_zero')].iloc[0]
    assert np.isclose(table[1]['accuracy'], old_base.accuracy, atol=1e-12)
    assert np.isclose(table[1]['auc'], old_base.auc, atol=1e-12)
    assert (RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS_SEPARATE_SOURCE_ONLY_CONTROL',
        'inference_conditions_verified': 360, 'datasets': 72,
        'all_contexts_at_most_256_and_source_only': True,
        'saved_predictions_recomputed_from_frozen_policies': True,
        'evaluation_query_rows_unchanged': True,
        'same_benchmark_but_different_target_supervision_budget': True,
        'original_ABCD_results_not_overwritten': True,
        'D_dagger_accuracy_below_0_7': bool(table[0]['accuracy'] < .7),
        'no_target_threshold_search_or_noise_injection': True,
        'fair_original_ABCD_mechanism_validation': False, 'Markdown_bytes': 0})
    print(pd.DataFrame(table).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['launch', 'worker', 'report'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    if args.phase == 'worker':
        worker(args.shard, args.shards)
    elif args.phase == 'launch':
        launch(args.shards)
    else:
        report()
