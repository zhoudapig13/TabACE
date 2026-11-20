"""Declared prediction-only adaptation of v84; no constructor replacement.

The earlier diagnostic outcomes motivated this development round. Selection
uses only an inner dataset fold, obeying each seen/unseen mechanism boundary.
All original outcomes remain available, including nonsignificant C and D.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

import abcd_mechanism_v84 as study
import tabace_fusion_v28 as rule
from prepare_data import RUN, atomic_json
from run_tabicl import metrics, normalize

OUT = RUN / 'results/abcd_fusion_adaptation_v86'
CANDIDATES = ('base', 'paper_gate', 'disagreement_gate', 'disagreement_gain4', 'convex')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == study.sha(__file__)
        return
    atomic_json(p, {'status': 'FROZEN_BEFORE_ADAPTED_POLICY_FITTING', 'task': 'classification',
        'seed': 0, 'context_rows': 256, 'alternative_contexts': 5, 'estimators_per_context': 1,
        'candidates': CANDIDATES, 'regularization': .001,
        'primary_inner_selection': 'mean dataset AUC across allowed calibration episodes',
        'tie_break': 'first declared candidate', 'outer_dataset_folds': 5,
        'unseen_exclusions': 'all positive-strength five-source-perturbation families, including Mixed',
        'matching_constructor_removed_in_C_D': True,
        'new_matching_constructor_added_in_C_D': False, 'new_raw_feature_predictor': False,
        'backbone_finetuning': False, 'extra_per_dataset_clean_supervision': False,
        'new_benchmark': False, 'Base_modified': False,
        'adaptation_dagger': True, 'earlier_development_query_outcomes_motivated_candidate_design': True,
        'independent_confirmation_claim': False, 'all_candidates_and_outcomes_retained': True,
        'significance_is_acceptance_evidence_not_selection_objective': True,
        'statistics': 'same dataset bootstrap and Holm family as v84',
        'source_plan': str(study.OUT / 'plan.json'), 'source_plan_sha256': study.sha(study.OUT / 'plan.json'),
        'code_sha256': study.sha(__file__), 'Markdown_written': False})


def fit(cells, kind):
    if kind == 'base':
        return {'kind': 'base'}
    if kind != 'convex':
        rule.GATE_VARIANT = 'entropy' if kind == 'paper_gate' else 'entropy_or_disagreement'
        return rule.fit(cells, 'classification', 5, 4. if kind == 'disagreement_gain4' else 1., .001)
    counts = {}
    for c in cells:
        counts[c['meta']['data_id']] = counts.get(c['meta']['data_id'], 0) + 1
    truth_probabilities, weights = [], []
    for c in cells:
        y = c['y'].astype(int); index = np.arange(len(y))
        truth_probabilities.append(np.column_stack([c['base'][index, y], c['bank'][:, index, y].T]))
        weights.append(np.full(len(y), 1. / (len(counts) * counts[c['meta']['data_id']] * len(y))))
    p = torch.as_tensor(np.concatenate(truth_probabilities), dtype=torch.float64)
    w = torch.as_tensor(np.concatenate(weights), dtype=torch.float64)
    def objective(values):
        v = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        a = torch.softmax(v, 0)
        loss = -(w * torch.log((p @ a).clamp(1e-12, 1.))).sum() + .001 * (v - v.mean()).square().mean()
        loss.backward()
        return float(loss.detach()), v.grad.detach().numpy()
    result = minimize(objective, np.zeros(6), method='L-BFGS-B', jac=True,
                      bounds=[(-10, 10)] * 6, options={'maxiter': 250, 'ftol': 1e-10})
    assert np.isfinite(result.x).all()
    logits = result.x - result.x.max(); a = np.exp(logits); a /= a.sum()
    return {'kind': 'convex', 'weights': a.tolist(), 'optimization_success': bool(result.success)}


def predict(cell, model):
    if model['kind'] != 'convex':
        return rule.predict(cell, model)
    bank = np.concatenate([cell['base'][None], cell['bank']], axis=0)
    return normalize(np.einsum('r,rnc->nc', model['weights'], bank))


def score(cells, model):
    datasets = {}
    for c in cells:
        auc = metrics(c['y'], predict(c, model), 'classification', 1.)['auc']
        if auc is not None:
            datasets.setdefault(c['meta']['data_id'], []).append(auc)
    return float(np.mean([np.mean(v) for v in datasets.values()]))


def worker(fold):
    torch.set_num_threads(1); torch.manual_seed(0)
    rows = []
    folder = OUT / 'folds/classification' / str(fold)
    for method, seen, matched in [('A', True, True), ('B', False, True), ('C', True, False), ('D', False, False)]:
        cells = study.load('classification', matched)
        train = [c for c in cells if c['fold'] != fold and (seen or not study.exposed(c, 'classification'))]
        test = [c for c in cells if c['fold'] == fold and c['meta']['track'] == 'five_corrected' and
                (c['meta']['rho'] == 0 or c['meta']['family'] == 'shortcut')]
        train_ids = {c['meta']['data_id'] for c in train}; test_ids = {c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert seen or not any(study.exposed(c, 'classification') for c in train)
        inner = min(c['fold'] for c in train)
        calibration = [c for c in train if c['fold'] != inner]
        validation = [c for c in train if c['fold'] == inner]
        trials = []
        best_score, selected = -np.inf, None
        for kind in CANDIDATES:
            model = fit(calibration, kind)
            value = score(validation, model)
            trials.append({'candidate': kind, 'inner_dataset_auc': value, 'inner_model': model})
            if value > best_score + 1e-12:
                best_score, selected = value, kind
        model = fit(train, selected)
        atomic_json(folder / f'{method}_policy.json', {'model': model, 'selected': selected,
            'trials': trials, 'seen': seen, 'matched': matched, 'training_data_ids': sorted(train_ids),
            'excluded_data_ids': sorted(test_ids), 'inner_fold': inner, 'adaptation_dagger': True})
        for c in test:
            meta = c['meta']; predictions = {method: predict(c, model)}
            if method == 'A':
                predictions['Base'] = c['base']; predictions['Single-matched'] = c['bank'][3]
            for name, p in predictions.items():
                path = folder / 'predictions' / str(meta['task_id']) / f'{meta["rho"]}_{name}.npz'
                path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, prediction=p, y=c['y'])
                rows.append({'task': 'classification', 'method': name, 'data_id': meta['data_id'],
                    'task_id': meta['task_id'], 'dataset': meta['dataset'], 'rho': meta['rho'],
                    'fold': fold, **metrics(c['y'], p, 'classification', 1.)})
        print('ADAPTED_POLICY', fold, method, selected, best_score, flush=True)
    pd.DataFrame(rows).to_csv(folder / 'detail.csv', index=False)
    atomic_json(folder / 'completion.json', {'status': 'COMPLETE', 'rows': len(rows)})


def launch():
    freeze()
    def execute(fold):
        log = OUT / f'worker_{fold}.log'
        with log.open('a') as handle:
            process = subprocess.run([sys.executable, __file__, 'worker', '--fold', str(fold)], stdout=handle, stderr=subprocess.STDOUT)
        return {'fold': fold, 'returncode': process.returncode, 'log': str(log)}
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(5)]):
            r = future.result(); records.append(r); print(r, flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'workers': records})
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records) else 'FAILED', 'workers': records})
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['launch', 'worker']); p.add_argument('--fold', type=int)
    a = p.parse_args()
    if a.phase == 'launch':
        launch()
    else:
        worker(a.fold)
