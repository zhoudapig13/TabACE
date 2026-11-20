"""Frozen-test Base/Shared/Specific transfer on public clinical shifts.

Source labels alone calibrate policies and decision thresholds. Target labels
are opened only after policies and member predictions have been frozen.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/public_shift_v1'
OUT = ROOT / 'results/public_shift_v1'
CHECKPOINT = ROOT.parents[2] / 'TabSwift/artifacts/checkpoints/tabicl-classifier-v2-20260212.ckpt'
PLAN = json.loads((ROOT / 'configs/public_shift_protocol.json').read_text())


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def geometry(X, T):
    center = X.mean(0)
    scale = np.maximum(X.std(0), .5)
    A, B = np.clip((X-center)/scale, -10, 10), np.clip((T-center)/scale, -10, 10)
    if X.shape[1] > 32:
        matrix = np.random.default_rng(0).normal(size=(X.shape[1], 32))/np.sqrt(32)
        A, B = A @ matrix, B @ matrix
    return A.astype(np.float32), B.astype(np.float32), center, scale


def subset(y, score=None, weights=None, seed=0):
    """Class-stratified selection preserves source's natural label proportions."""
    rng = np.random.default_rng(seed)
    n = min(256, len(y))
    positive = int(np.clip(round(n * y.mean()), 1, n-1))
    result = []
    for label, count in [(0, n-positive), (1, positive)]:
        ids = np.flatnonzero(y == label)
        count = min(count, len(ids))
        if score is not None:
            chosen = ids[np.argsort(score[ids], kind='stable')[:count]]
        else:
            p = None if weights is None else np.maximum(weights[ids], 1e-12)
            if p is not None:
                p = p/p.sum()
            chosen = rng.choice(ids, count, replace=False, p=p)
        result.extend(chosen.tolist())
    rows = np.asarray(result, int)
    rng.shuffle(rows)
    return rows


def construct(X, y, target, align_columns):
    A, B, center, scale = geometry(X, target)
    base_ids = subset(y, seed=0)
    anchors = np.quantile(B, [.1, .3, .5, .7, .9], axis=0)
    anchor_distance = NearestNeighbors(n_neighbors=1).fit(anchors).kneighbors(A)[0][:, 0]
    target_distance = NearestNeighbors(n_neighbors=min(10, len(B))).fit(B).kneighbors(A)[0].mean(1)
    outlier = np.minimum(A*A, 100).mean(1)
    ids = NearestNeighbors(n_neighbors=min(11, len(A))).fit(A).kneighbors(A, return_distance=False)
    # Exclude SELF even for duplicated feature rows.
    neighbor_ids = np.stack([row[row != i][:10] for i, row in enumerate(ids)])
    vote = y[neighbor_ids].mean(1)
    disagreement = np.minimum(vote, 1-vote)
    scores = [('target_neighbor', anchor_distance),
              ('representative', target_distance + .35*outlier),
              ('locally_stable', disagreement + .1*target_distance),
              ('low_conflict', disagreement + .2*outlier)]
    views = []
    for name, score in scores:
        rows = subset(y, score=score)
        views.append(dict(name=name, rows=rows, columns=None, context=X[rows]))
    yc = y-y.mean()
    xc = X-X.mean(0)
    association = np.abs(xc.T @ yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()), 1e-8)
    columns = np.flatnonzero(np.arange(X.shape[1]) != association.argmax())
    views.insert(3, dict(name='feature_subspace', rows=base_ids, columns=columns,
                         context=X[base_ids][:, columns]))
    h = max(float(np.median(target_distance[target_distance > 0])), 1e-6)
    matching = np.exp(np.clip(-target_distance/h, -50, 0))
    rows = subset(y, weights=matching, seed=101)
    views.append(dict(name='specific_target_matched', rows=rows, columns=None, context=X[rows]))
    # Balanced-domain logistic fit estimates density ratios, without any clinical target labels.
    domain = LogisticRegression(C=.1, max_iter=300, random_state=0)
    n0, n1 = min(4096, len(A)), min(4096, len(B))
    s0 = np.random.default_rng(10).choice(len(A), n0, replace=False)
    s1 = np.random.default_rng(11).choice(len(B), n1, replace=False)
    domain.fit(np.r_[A[s0], B[s1]], np.r_[np.zeros(n0), np.ones(n1)])
    def density(query):
        C = np.clip((query-center)/scale, -10, 10)
        if X.shape[1] > 32:
            matrix = np.random.default_rng(0).normal(size=(X.shape[1], 32))/np.sqrt(32)
            C = C @ matrix
        p = np.clip(domain.predict_proba(C)[:, 1], 1e-6, 1-1e-6)
        return np.clip((p/(1-p))*(n0/n1), .05, 10)
    ratio = density(X)
    rows = subset(y, weights=np.sqrt(ratio), seed=102)
    views.append(dict(name='specific_density_ratio', rows=rows, columns=None, context=X[rows]))
    aligned = X[base_ids].copy()
    for col in align_columns:
        ms, mt = np.median(X[:, col]), np.median(target[:, col])
        ss = max(float(np.subtract(*np.quantile(X[:, col], [.75, .25]))), 1)
        st = max(float(np.subtract(*np.quantile(target[:, col], [.75, .25]))), 1)
        aligned[:, col] = .5*aligned[:, col] + .5*((aligned[:, col]-ms)*st/ss+mt)
    views.append(dict(name='specific_numeric_alignment', rows=base_ids, columns=None, context=aligned))
    assert len(views) == 8 and all(len(v['rows']) <= 256 for v in views)
    return base_ids, views, density


def member_bank(pack, gpu, cache):
    X, y = pack['X_train'], pack['y_train']
    base_ids, views, density = construct(X, y, pack['X_ood'], pack['align_columns'])
    cal_weights = np.sqrt(density(pack['X_cal']))
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}, cal_weights
    sizes = [len(pack[k]) for k in ['X_cal', 'X_id', 'X_ood']]
    queries = np.concatenate([pack[k] for k in ['X_cal', 'X_id', 'X_ood']])
    model = TabICLClassifier(n_estimators=1, device=f'cuda:{gpu}', model_path=str(CHECKPOINT),
                            allow_auto_download=False, n_jobs=1, batch_size=1, random_state=0)
    all_views = [dict(name='Base', rows=base_ids, columns=None, context=X[base_ids])] + views
    predictions = []
    for view in all_views:
        start = time.monotonic()
        root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
        if root is not None and hasattr(root, 'root'):
            root.root = None
        model.fit(view['context'], y[view['rows']])
        if not getattr(model, '_medical_weights_cached', False):
            model._load_model = lambda: None
            model._medical_weights_cached = True
        values = []
        for begin in range(0, len(queries), 1024):
            q = queries[begin:begin+1024]
            if view['columns'] is not None:
                q = q[:, view['columns']]
            values.append(model.predict_proba(q)[:, np.flatnonzero(model.classes_ == 1)[0]])
        predictions.append(np.concatenate(values))
        print('MEMBER_DONE', view['name'], 'rows', len(view['rows']),
              'seconds', round(time.monotonic()-start, 2), flush=True)
    predictions = np.asarray(predictions)
    result = {}
    for label, start, end in zip(['cal', 'id', 'ood'], np.cumsum([0]+sizes[:-1]), np.cumsum(sizes)):
        result['base_'+label] = predictions[0, start:end]
        result['bank_'+label] = predictions[1:, start:end]
    result['context_rows'] = np.asarray([len(v['rows']) for v in all_views])
    result['context_unique_source_rows'] = np.asarray([len(np.unique(v['rows'])) for v in all_views])
    result['context_roles'] = np.asarray([v['name'] for v in all_views])
    result['shared_unique_source_rows'] = np.asarray(len(np.unique(np.concatenate([v['rows'] for v in all_views[:6]]))))
    result['specific_unique_source_rows'] = np.asarray(len(np.unique(np.concatenate([v['rows'] for v in all_views]))))
    np.savez_compressed(cache, **result)
    return result, cal_weights


def features(base, bank, m):
    p = np.clip(bank[:m].T, 1e-7, 1-1e-7)
    quality = 1+(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2)
    agreement = 1-np.abs(p-p.mean(1, keepdims=True))
    b = np.clip(base, 1e-7, 1-1e-7)
    uncertainty = -(b*np.log(b)+(1-b)*np.log(1-b))/np.log(2)
    return p, quality, agreement, uncertainty


def probability(base, bank, policy):
    if policy['kind'] == 'base':
        return base
    m = policy['m']
    p, q, s, u = features(base, bank, m)
    c = np.asarray(policy['parameters'])
    logits = c[:m] + q*c[m:2*m] + s*c[2*m:3*m]
    w = np.exp(logits-logits.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    mix = (w*p).sum(1)
    alpha = np.clip(policy['gain']*c[-1]*u*(w*q*s).sum(1), 0, 1)
    return (1-alpha)*base+alpha*mix


def threshold(p, y, weight):
    grid = np.r_[.5, np.linspace(.1, .95, 18)]
    scores = [np.average((p >= t) == y, weights=weight) for t in grid]
    return float(grid[np.argmax(scores)])


def fit_gate(base, bank, y, weight, m, gain, regularization):
    p, q, s, u = features(base, bank, m)
    arrays = [p, q, s, u, base, y, weight/weight.sum()]
    P, Q, S, U, B, Y, Wt = [torch.as_tensor(a, dtype=torch.float64) for a in arrays]
    init = np.r_[np.zeros(m), np.ones(2*m), .5]
    def objective(x):
        c = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        w = torch.softmax(c[:m]+Q*c[m:2*m]+S*c[2*m:3*m], dim=1)
        mix = (w*P).sum(1)
        alpha = (gain*c[-1]*U*(w*Q*S).sum(1)).clamp(0, 1)
        pred = ((1-alpha)*B+alpha*mix).clamp(1e-8, 1-1e-8)
        loss = (Wt*(-Y*torch.log(pred)-(1-Y)*torch.log(1-pred))).sum()
        loss = loss+regularization*((c[:m]-c[:m].mean()).square().mean()+(c[m:3*m]-1).square().mean())
        loss.backward()
        return float(loss.detach()), c.grad.detach().numpy()
    result = minimize(objective, init, jac=True, method='L-BFGS-B',
                      bounds=[(-6, 6)]*m+[(0, 6)]*(2*m)+[(0, 1)],
                      options={'maxiter': 100, 'ftol': 1e-9})
    return dict(kind='gate', m=m, gain=gain, regularization=regularization,
                parameters=result.x.tolist(), optimizer_success=bool(result.success))


def select_policy(base, bank, y, weight, m, fallback, fallback_inner, fit_ids, val_ids):
    def score(policy):
        p = probability(base[val_ids], bank[:, val_ids], policy)
        return float(np.average((p >= policy['threshold']) == y[val_ids], weights=weight[val_ids]))
    best_value, selected = score(fallback_inner), None
    trials = [dict(kind='fallback', validation_accuracy=best_value)]
    for gain in [1., 2.]:
        for reg in [.01, .1]:
            policy = fit_gate(base[fit_ids], bank[:, fit_ids], y[fit_ids], weight[fit_ids], m, gain, reg)
            p = probability(base[fit_ids], bank[:, fit_ids], policy)
            policy['threshold'] = threshold(p, y[fit_ids], weight[fit_ids])
            value = score(policy)
            trials.append(dict(gain=gain, regularization=reg, threshold=policy['threshold'],
                               validation_accuracy=value))
            if value > best_value+1e-12:
                best_value, selected = value, policy
    if selected is None:
        return fallback, fallback_inner, trials
    final = fit_gate(base, bank, y, weight, m, selected['gain'], selected['regularization'])
    final['threshold'] = threshold(probability(base, bank, final), y, weight)
    return final, selected, trials


def metrics(y, p, decision_threshold):
    return dict(accuracy=float(accuracy_score(y, p >= decision_threshold)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= decision_threshold)),
                auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if y.sum() else None,
                brier=float(brier_score_loss(y, p)), threshold=decision_threshold)


def paired(y, previous, current):
    delta = (current == y).astype(int)-(previous == y).astype(int)
    gain, loss = int((delta == 1).sum()), int((delta == -1).sum())
    rng = np.random.default_rng(0)
    counts = rng.multinomial(len(y), [loss/len(y), (len(y)-loss-gain)/len(y), gain/len(y)], size=10000)
    boot = (counts[:, 2]-counts[:, 0])/len(y)
    return dict(accuracy_delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                improved_patients=gain, worsened_patients=loss,
                mcnemar_p=float(binomtest(gain, gain+loss, .5).pvalue) if gain+loss else 1.)


def run(task, gpu):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    source = DATA / task
    dest = OUT / task
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / 'metadata.json').read_text())
    with np.load(source / 'inputs.npz') as z:
        pack = {k: z[k] for k in z.files}
    assert {k: array_hash(v) for k, v in pack.items()} == meta['input_hashes']
    bank, weights = member_bank(pack, gpu, source / 'member_predictions.npz')
    y = pack['y_cal']
    fit_ids, val_ids = train_test_split(np.arange(len(y)), test_size=.4, random_state=0, stratify=y)
    base, members = bank['base_cal'], bank['bank_cal']
    base_policy = dict(kind='base', threshold=threshold(base, y, np.ones(len(y))))
    base_inner = dict(kind='base', threshold=threshold(base[fit_ids], y[fit_ids], np.ones(len(fit_ids))))
    shared, shared_inner, shared_trials = select_policy(base, members, y, np.ones(len(y)),
                                                       5, base_policy, base_inner, fit_ids, val_ids)
    specific, _, specific_trials = select_policy(base, members, y, weights, 8,
                                                shared, shared_inner, fit_ids, val_ids)
    policies = dict(Base=base_policy, Shared=shared, Specific=specific)
    write_json(dest / 'frozen_policies.json', dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        test_labels_opened=False, policies=policies,
        source_calibration_trials=dict(Shared=shared_trials, Specific=specific_trials),
        selection_metric='source holdout accuracy; Specific uses unlabelled density-ratio weights',
        target_labels_used_for_fitting=False, protocol=PLAN,
        input_hashes=meta['input_hashes'], specific_contains_shared=True))
    # This is the first access to final evaluation labels in this program.
    with np.load(source / 'test_truth.npz') as z:
        truth = {k: z[k] for k in z.files}
    summary = dict(task=task, seed=0, metadata=meta,
                   context_roles=bank['context_roles'].tolist(),
                   context_rows=bank['context_rows'].tolist(),
                   shared_unique_labelled_context_rows=int(bank['shared_unique_source_rows']),
                   specific_unique_labelled_context_rows=int(bank['specific_unique_source_rows']),
                   target_labels_used_for_fitting=False, method_status=PLAN['method_status'], domains={})
    for domain in ['id', 'ood']:
        target_y = truth['y_'+domain]
        p = {name: probability(bank['base_'+domain], bank['bank_'+domain], policy)
             for name, policy in policies.items()}
        scores = {name: metrics(target_y, value, policies[name]['threshold']) for name, value in p.items()}
        tests = {}
        for previous, current in [('Base', 'Shared'), ('Shared', 'Specific')]:
            tests[current+'_minus_'+previous] = paired(
                target_y, p[previous] >= policies[previous]['threshold'],
                p[current] >= policies[current]['threshold'])
        summary['domains'][domain] = dict(n=len(target_y), positives=int(target_y.sum()),
                                          always_negative_accuracy=float((target_y == 0).mean()),
                                          methods=scores, adjacent_comparisons=tests)
        print('RESULT', task, domain, json.dumps(summary['domains'][domain]), flush=True)
    write_json(dest / 'metrics.json', summary)
    print('TASK_COMPLETE', task, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--wait-for-inputs', action='store_true')
    args = parser.parse_args()
    if args.wait_for_inputs:
        deadline = time.monotonic()+1800
        while not (DATA/args.task/'metadata.json').exists():
            if time.monotonic() > deadline:
                raise SystemExit('Timed out waiting for prepared patient-disjoint inputs.')
            time.sleep(2)
    run(args.task, args.gpu)
