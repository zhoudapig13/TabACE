"""Inference target-label context-budget sensitivity with fixed 64-label fits.

This is a development diagnostic, not the original ABCD comparison. All
policies keep their existing 64-label calibration. Only target labels used
inside TFM inference contexts are reduced; source rows fill the native cap.
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

import abcd_metric_aligned_v100 as algorithm
import target_supervised_banks_v98 as construction
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, preserving_bootstrap, metrics
from shared_reference import contexts

OUT = RUN / 'results/shortcut_context_label_budget_v114'
POLICIES = RUN / 'results/abcd_transfer_calibration_v109'
BUDGETS = (16, 24, 32, 48, 64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    seal = json.loads((POLICIES / 'policy_seal.json').read_text())
    assert all(sha(POLICIES / p) == h for p, h in seal.items())
    plan = {
        'status': 'DEVELOPMENT_INFERENCE_CONTEXT_LABEL_BUDGET_ABLATION',
        'datasets': 72, 'seed': 0, 'total_context_cap': 256,
        'target_rows_in_inference_context': BUDGETS,
        'target_labels_used_by_existing_calibration_fit': 64,
        'fusion_policy_parameters_fixed_across_budgets': True,
        'policies': str(POLICIES), 'policy_seal_sha256': sha(POLICIES / 'policy_seal.json'),
        'all_A_B_C_D_reported_at_every_budget': True,
        'subset_selection': 'nested class-balanced seed0 ordering, same for all methods and strengths',
        'unlabeled_query_support': 'fixed union of original calibration and evaluation features',
        'native_context': 'source(256-n)+target(n)',
        'alternative_contexts': ['target_neighbor', 'target_labeled_with_missing_class_anchors',
            'locally_stable', 'matched_feature_drop_A_B_or_full_feature_bootstrap_C_D', 'low_conflict'],
        'both_native_and_alternative_contexts_use_only_selected_target_labels': True,
        '64_budget_reuses_original_v98_predictions': True,
        'benchmark_severities_and_query_rows_changed': False,
        'backbone_or_fusion_parameters_retrained': False,
        'noise_injection_or_query_label_training': False,
        'context_budget_is_not_total_supervision_budget': True,
        'previously_evaluated_development_cohort': True,
        'not_original_paper_replication_or_independent_confirmation': True,
        'no_A_B_C_D_order_or_accuracy_target_used_for_fitting': True,
        'budget_selected_for_final_comparison': None,
        'any_later_selection_on_this_grid_must_be_labeled_post_hoc': True,
        'code_sha256': sha(__file__), 'Markdown_written': False,
    }
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(plan))
    else:
        atomic_json(p, plan)


def target_order(labels):
    rng = np.random.default_rng(0)
    classes = rng.permutation(np.unique(labels))
    queues = {int(c): list(rng.permutation(np.flatnonzero(labels == c))) for c in classes}
    order = []
    while any(queues.values()):
        for c in classes:
            if queues[int(c)]:
                order.append(int(queues[int(c)].pop()))
    assert sorted(order) == list(range(len(labels)))
    return np.asarray(order, dtype=int)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = Predictor('classification', 'cuda:0')

    def infer(X, y, Q, K):
        for estimator in model.models.values():
            root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
            if root is not None and hasattr(root, 'root'):
                root.root = None
        torch.cuda.empty_cache()
        return model(X, y, Q, n_classes=K, members=1, random_state=0)[0]

    for index, path in enumerate(sorted((construction.split.OUT / 'inputs').glob('*.npz'))):
        if index % shards != shard:
            continue
        with np.load(path) as z:
            data = {k: z[k] for k in ['y_source', 'y_calibration', 'X_calibration', 'X_evaluation'] +
                [f'X_source_{r}' for r in (0, 20, 50, 80, 100)]}
        meta = json.loads(path.with_suffix('.json').read_text())
        tid = int(meta['task_id'])
        assert sha(path) == meta['model_inputs_sha256']
        order = target_order(data['y_calibration'])
        union = np.vstack([data['X_calibration'], data['X_evaluation']])
        offset = len(data['y_calibration'])
        K = len(np.unique(data['y_source']))
        policies = {name: json.loads((POLICIES / 'classification' / str(tid) /
            f'{name}_policy.json').read_text())['policy'] for name in ('A', 'B', 'C', 'D')}
        for budget in BUDGETS:
            selected = np.sort(order[:budget])
            T, t = data['X_calibration'][selected], data['y_calibration'][selected]
            for rho in (0, 20, 50, 80, 100):
                out = OUT / 'banks' / str(tid) / str(budget) / f'{rho}.npz'
                if out.with_suffix('.json').exists():
                    continue
                X, y = data[f'X_source_{rho}'], data['y_source']
                A, b = construction.native_context(X, y, T, t)
                pool_X, pool_y = np.vstack([X, T]), np.r_[y, t]
                views = contexts(pool_X, pool_y, union, A, b, 256, 'classification')
                C, d = construction.target_context(X, y, T, t)
                views[1] = ('target_labeled', C, d, union)
                bootstrap_ids = preserving_bootstrap(b, np.random.default_rng(0))
                assert len(b) == 256 and all(len(v) <= 256 for _, _, v, _ in views)
                if budget == 64:
                    family = 'clean' if rho == 0 else 'shortcut'
                    cache = construction.OUT / 'banks' / str(tid) / f'{family}_{rho}' / 'evaluation.npz'
                    with np.load(cache) as z:
                        base, members, bootstrap = z['base_target'], z['members'], z['bootstrap']
                else:
                    base = infer(A, b, union, K)[offset:]
                    members = np.stack([infer(U, v, Q, K)[offset:] for _, U, v, Q in views])
                    bootstrap = infer(A[bootstrap_ids], b[bootstrap_ids], union, K)[offset:]
                values = {'base': base, 'members': members, 'bootstrap': bootstrap,
                    'target_calibration_positions_used_in_context': selected}
                for method in ('A', 'B', 'C', 'D'):
                    bank = members.copy()
                    if method in ('C', 'D'):
                        bank[3] = bootstrap
                    cell = {'task': 'classification', 'base': base, 'bank': bank, 'meta': {'scale': 1.}}
                    values['prediction_' + method] = algorithm.predict(cell, policies[method])
                    if budget == 64:
                        with np.load(POLICIES / 'classification_predictions/old_development' / str(tid) /
                                f'{rho}_{method}.npz') as z:
                            assert np.array_equal(values['prediction_' + method], z['prediction'])
                out.parent.mkdir(parents=True, exist_ok=True)
                temp = out.with_suffix('.tmp.npz')
                np.savez_compressed(temp, **values)
                temp.replace(out)
                atomic_json(out.with_suffix('.json'), {
                    'status': 'COMPLETE', 'task_id': tid, 'data_id': meta['data_id'], 'rho': rho,
                    'target_rows_in_context': budget, 'target_calibration_labels_for_existing_fusion': 64,
                    'native_context_rows': len(b), 'source_rows_in_native_context': 256-budget,
                    'branch_context_rows': [len(v) for _, _, v, _ in views],
                    'target_only_source_anchors': len(d)-len(t),
                    'source_X_sha256': protocol.array_sha256(X),
                    'source_y_sha256': protocol.array_sha256(y),
                    'evaluation_X_sha256': protocol.array_sha256(data['X_evaluation']),
                    'selected_target_positions': selected.tolist(), 'input_sha256': sha(path),
                    'policy_sha256': {name: sha(POLICIES / 'classification' / str(tid) /
                        f'{name}_policy.json') for name in policies},
                    'output_sha256': sha(out), '64_cached_predictions_verified_identical': budget == 64,
                    'evaluation_labels_loaded_for_inference': False,
                })
            print('CONTEXT_BUDGET_TASK', tid, budget, flush=True)


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
    for path in sorted((OUT / 'banks').glob('*/*/*.npz')):
        m = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == m['output_sha256']
        tid, n = m['task_id'], m['target_rows_in_context']
        with np.load(construction.split.OUT / 'evaluation_truth' / f'{tid}.npz') as z:
            y = z['y_evaluation']
        with np.load(path) as z:
            for method in ('A', 'B', 'C', 'D', 'Base'):
                q = z['base' if method == 'Base' else 'prediction_' + method]
                rows.append({'task_id': tid, 'data_id': m['data_id'], 'rho': m['rho'],
                    'context_target_rows': n, 'method': method, **metrics(y, q, 'classification', 1.)})
    detail = pd.DataFrame(rows)
    assert len(detail) == 72*5*len(BUDGETS)*5 and detail.task_id.nunique() == 72
    detail.to_csv(OUT / 'detail.csv', index=False)
    summaries = []
    for (budget, method), group in detail.groupby(['context_target_rows', 'method']):
        pert = group[group.rho > 0].groupby('data_id')[['accuracy', 'auc']].mean()
        clean = group[group.rho == 0][['accuracy', 'auc']].mean()
        summaries.append({'context_target_rows': budget, 'method': method, 'datasets': len(pert),
            'accuracy': pert.accuracy.mean(), 'auc': pert.auc.mean(),
            'clean_accuracy': clean.accuracy, 'clean_auc': clean.auc,
            'target_labels_used_by_existing_fusion_fit': 64, 'status': 'development_context_budget_ablation'})
    table = pd.DataFrame(summaries)
    table.to_csv(OUT / 'tables.csv', index=False)
    original = pd.read_csv(POLICIES / 'classification_tables.csv')
    for row in table[table.context_target_rows == 64].to_dict('records'):
        reference = original[(original.cohort == 'old_development') & (original.method == row['method'])].iloc[0]
        assert np.isclose(row['accuracy'], reference.accuracy, atol=1e-12)
        assert np.isclose(row['auc'], reference.auc, atol=1e-12)
    seal = json.loads((POLICIES / 'policy_seal.json').read_text())
    assert all(sha(POLICIES / p) == h for p, h in seal.items())
    assert (RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS', 'datasets': 72,
        'conditions': 72*5*len(BUDGETS), 'all_budgets_reported': True,
        '64_context_budget_matches_previous_predictions_and_metrics': True,
        'fusion_policy_files_unchanged': True, 'no_new_training': True,
        'query_labels_used_only_to_compute_metrics': True,
        'context_target_label_budget_not_total_label_access': True,
        'not_original_paper_replication': True, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Bridge the abrupt source-only supervision removal with a gradual inference-context budget sweep',
        'Setting': json.loads((OUT / 'plan.json').read_text()),
        'Results': summaries,
        'Analysis': 'Observed effects are inference-context budget sensitivity; calibration supervision remains 64 labels. No number is guaranteed or edited.',
        'Next Steps': 'Report every budget. Any highlighted budget is a post-hoc development selection, not a fair original ABCD result.',
    })
    print(table.to_string(index=False), flush=True)


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
