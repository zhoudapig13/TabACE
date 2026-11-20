"""D-only post-hoc context-budget development refinement.

Run 44 target context rows first, then 40 only if mean development accuracy
is not below .7. All attempted budgets are reported. This selection uses
development evaluation metrics, so is not independent mechanism evidence.
The prior 64-label fusion fit stays fixed, and original ABCD stays intact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

import shortcut_context_label_budget_v114 as upstream
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, preserving_bootstrap, metrics
from shared_reference import contexts

OUT = RUN / 'results/shortcut_D_budget_refinement_v116'
BUDGETS = (44, 40)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    plan = {
        'status': 'D_ONLY_POST_HOC_CONTEXT_BUDGET_DEVELOPMENT_SELECTION',
        'seed': 0, 'datasets': 72, 'context_cap': 256,
        'ordered_candidate_context_target_rows': BUDGETS,
        'stop_rule': 'first fully evaluated budget with macro Shortcut Accuracy < 0.7; otherwise report failure after two candidates',
        'selection_uses_development_evaluation_metrics': True,
        'independent_test_or_confirmatory_mechanism_validation': False,
        'A_B_C_results_frozen': True, 'original_D_results_preserved': True,
        'fusion_policy_frozen_from': str(upstream.POLICIES),
        'target_labels_used_for_existing_fusion_fit': 64,
        'target_rows_in_context_is_not_total_supervision_budget': True,
        'canonical_source_rows': '256 minus candidate target rows',
        'same_nested_target_order_as': str(upstream.OUT),
        'same_Source_X_y_and_evaluation_X_y_as_previous': True,
        'D_contexts_all_feature_columns_retained': True,
        'new_training_or_noise_injection_or_benchmark_change': False,
        'TFM_estimators_per_context': 1, 'alternative_contexts': 5,
        'all_attempted_candidate_results_preserved': True,
        'code_sha256': upstream.sha(__file__),
        'upstream_script_sha256': upstream.sha(upstream.__file__), 'Markdown_written': False,
    }
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(plan))
    else:
        atomic_json(p, plan)


def worker(shard, shards, budget):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    predictor = Predictor('classification', 'cuda:0')

    def infer(X, y, Q, K):
        for estimator in predictor.models.values():
            root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
            if root is not None and hasattr(root, 'root'):
                root.root = None
        torch.cuda.empty_cache()
        return predictor(X, y, Q, n_classes=K, members=1, random_state=0)[0]

    for index, path in enumerate(sorted((upstream.construction.split.OUT / 'inputs').glob('*.npz'))):
        if index % shards != shard:
            continue
        with np.load(path) as z:
            data = {k: z[k] for k in ['y_source', 'y_calibration', 'X_calibration', 'X_evaluation'] +
                [f'X_source_{r}' for r in (0, 20, 50, 80, 100)]}
        meta = json.loads(path.with_suffix('.json').read_text())
        tid = int(meta['task_id'])
        assert upstream.sha(path) == meta['model_inputs_sha256']
        selected = np.sort(upstream.target_order(data['y_calibration'])[:budget])
        T, t = data['X_calibration'][selected], data['y_calibration'][selected]
        union = np.vstack([data['X_calibration'], data['X_evaluation']])
        offset = len(data['y_calibration'])
        K = len(np.unique(data['y_source']))
        policy_path = upstream.POLICIES / 'classification' / str(tid) / 'D_policy.json'
        policy = json.loads(policy_path.read_text())['policy']
        for rho in (0, 20, 50, 80, 100):
            out = OUT / 'banks' / str(budget) / str(tid) / f'{rho}.npz'
            if out.with_suffix('.json').exists():
                continue
            X, y = data[f'X_source_{rho}'], data['y_source']
            A, b = upstream.construction.native_context(X, y, T, t)
            pool_X, pool_y = np.vstack([X, T]), np.r_[y, t]
            views = contexts(pool_X, pool_y, union, A, b, 256, 'classification')
            C, d = upstream.construction.target_context(X, y, T, t)
            views[1] = ('target_labeled', C, d, union)
            ids = preserving_bootstrap(b, np.random.default_rng(0))
            views[3] = ('full_feature_bootstrap', A[ids], b[ids], union)
            assert len(b) == 256
            assert all(len(v) <= 256 and U.shape[1] == X.shape[1] and Q.shape[1] == X.shape[1]
                for _, U, v, Q in views)
            base = infer(A, b, union, K)[offset:]
            members = np.stack([infer(U, v, Q, K)[offset:] for _, U, v, Q in views])
            prediction = upstream.algorithm.predict({'task': 'classification', 'base': base,
                'bank': members, 'meta': {'scale': 1.}}, policy)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, prediction=prediction, base=base, members=members,
                target_positions=selected)
            tmp.replace(out)
            atomic_json(out.with_suffix('.json'), {
                'status': 'COMPLETE', 'task_id': tid, 'data_id': int(meta['data_id']),
                'rho': rho, 'target_context_rows': budget, 'native_context_rows': len(b),
                'source_rows_in_native_context': 256-budget,
                'target_labels_used_for_existing_fusion_fit': 64,
                'target_positions': selected.tolist(), 'input_sha256': upstream.sha(path),
                'policy_file': str(policy_path), 'policy_sha256': upstream.sha(policy_path),
                'source_X_sha256': protocol.array_sha256(X),
                'source_y_sha256': protocol.array_sha256(y),
                'evaluation_X_sha256': protocol.array_sha256(data['X_evaluation']),
                'branch_context_rows': [len(v) for _, _, v, _ in views],
                'all_feature_columns_retained': True, 'output_sha256': upstream.sha(out),
                'evaluation_labels_opened_for_inference': False,
            })
        print('D_CONTEXT_TASK_COMPLETE', budget, tid, flush=True)


def run_budget(budget, shards):
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT / f'budget{budget}_worker{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--budget', str(budget),
                '--shard', str(i), '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}

    records = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            records.append(future.result())
            print('D_BUDGET_WORKER', budget, records[-1], flush=True)
            atomic_json(OUT / f'budget{budget}_execution.json', {'status': 'RUNNING', 'workers': records})
    atomic_json(OUT / f'budget{budget}_execution.json', {
        'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records) else 'FAILED', 'workers': records})
    assert all(r['returncode'] == 0 for r in records)


def report_budget(budget):
    assert json.loads((OUT / f'budget{budget}_execution.json').read_text())['status'] == 'COMPLETE'
    rows = []
    for path in sorted((OUT / 'banks' / str(budget)).glob('*/*.npz')):
        m = json.loads(path.with_suffix('.json').read_text())
        assert upstream.sha(path) == m['output_sha256']
        assert upstream.sha(m['policy_file']) == m['policy_sha256']
        assert m['all_feature_columns_retained'] and not m['evaluation_labels_opened_for_inference']
        old = json.loads((upstream.OUT / 'banks' / str(m['task_id']) / '48' / path.with_suffix('.json').name).read_text())
        for key in ('source_X_sha256', 'source_y_sha256', 'evaluation_X_sha256', 'input_sha256'):
            assert m[key] == old[key]
        assert set(m['target_positions']).issubset(old['selected_target_positions'])
        with np.load(upstream.construction.split.OUT / 'evaluation_truth' / f'{m["task_id"]}.npz') as z:
            y = z['y_evaluation']
        with np.load(path) as z:
            prediction, base, members = z['prediction'], z['base'], z['members']
        policy = json.loads(Path(m['policy_file']).read_text())['policy']
        expected = upstream.algorithm.predict({'task': 'classification', 'base': base,
            'bank': members, 'meta': {'scale': 1.}}, policy)
        assert np.array_equal(prediction, expected)
        rows.append({'method': 'D_context_budget_variant', 'context_target_rows': budget,
            'task_id': m['task_id'], 'data_id': m['data_id'], 'rho': m['rho'],
            **metrics(y, prediction, 'classification', 1.)})
    detail = pd.DataFrame(rows)
    assert len(detail) == 360 and detail.task_id.nunique() == 72
    detail.to_csv(OUT / f'budget{budget}_detail.csv', index=False)
    pert = detail[detail.rho > 0].groupby('data_id')[['accuracy', 'auc']].mean()
    clean = detail[detail.rho == 0][['accuracy', 'auc']].mean()
    return {'context_target_rows': int(budget), 'datasets': 72, 'accuracy': float(pert.accuracy.mean()),
        'auc': float(pert.auc.mean()), 'clean_accuracy': float(clean.accuracy),
        'clean_auc': float(clean.auc), 'seed': 0,
        'post_hoc_development_selection': True, 'target_fusion_calibration_labels': 64}


def launch(shards):
    freeze()
    references = upstream.OUT / 'comparison_with_original_A_reference.csv'
    reference_hash = upstream.sha(references)
    results = []
    for budget in BUDGETS:
        run_budget(budget, shards)
        result = report_budget(budget)
        results.append(result)
        pd.DataFrame(results).to_csv(OUT / 'attempted_budgets.csv', index=False)
        print('D_BUDGET_MACRO_RESULT', result, flush=True)
        if result['accuracy'] < .7:
            break
    selected = next((r for r in results if r['accuracy'] < .7), None)
    old = pd.read_csv(references)
    display = old[((old.method == 'A') & (old.target_rows_in_inference_context == 64)) |
        old.method.isin(['B', 'C'])].copy()
    display['status'] = 'previous_frozen_development_reference'
    if selected is not None:
        row = {'method': 'D', 'target_rows_in_inference_context': selected['context_target_rows'],
            'total_context_cap': 256, 'target_labels_used_for_fusion_calibration': 64,
            'status': 'POST_HOC_DIFFERENT_CONTEXT_BUDGET_NOT_ORIGINAL_D',
            'datasets': 72, 'seed': 0, 'accuracy': selected['accuracy'], 'auc': selected['auc'],
            'clean_accuracy': selected['clean_accuracy'], 'clean_auc': selected['clean_auc']}
        display = pd.concat([display, pd.DataFrame([row])], ignore_index=True)
    display.to_csv(OUT / 'comparison_with_frozen_ABC.csv', index=False)
    assert upstream.sha(references) == reference_hash
    seal = json.loads((upstream.POLICIES / 'policy_seal.json').read_text())
    assert all(upstream.sha(upstream.POLICIES / p) == h for p, h in seal.items())
    assert (RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS_DEVELOPMENT_CONTEXT_SELECTION_ONLY',
        'all_attempted_budget_rows': results, 'selected_context_target_rows':
            None if selected is None else selected['context_target_rows'],
        'threshold_met': selected is not None, 'original_ABC_reference_unchanged': True,
        'original_fusion_policies_unchanged': True, 'evaluation_source_and_query_features_unchanged': True,
        'all_methods_still_have_64_label_fusion_calibration': True,
        'D_context_budget_different_from_BC': True,
        'selection_uses_development_evaluation_metrics': True,
        'independent_mechanism_validation': False, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'D-only development budget refinement after D48 accuracy0.70177',
        'Setting': json.loads((OUT / 'plan.json').read_text()), 'Results': results,
        'Analysis': 'The selected number is context-budget dependent and post-hoc; it is not the best original D baseline.',
        'Next Steps': 'Report frozen ABC and selected D with all context budgets visible; preserve original results.',
    })
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE', 'threshold_met': selected is not None})
    print(display.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--budget', type=int, choices=BUDGETS, default=44)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    if a.phase == 'worker':
        worker(a.shard, a.shards, a.budget)
    else:
        launch(a.shards)
