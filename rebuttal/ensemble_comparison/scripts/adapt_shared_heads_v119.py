"""Calibration-only fusion-head adaptations on the unchanged Shared bank.

No model here changes a Base prediction, context, benchmark, or TFM weight.
Heads are explicitly adapted alternatives, not the manuscript fusion rule.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp

from run_tabicl import metrics, normalize

REGS = (.001, .01, .1)


def components(c, cap=None):
    P = c['P']
    if c['task'] == 'classification':
        clipped = np.clip(P, 1e-12, 1.)
        K = P.shape[-1]
        Q = (1+(clipped*np.log(clipped)).sum(2)/np.log(max(2, K))).T
        S = (1-.5*np.abs(P-P.mean(0)[None]).sum(2)).T
        return P, Q, S
    Z = (P-c['mu'])/c['sigma']
    median = np.median(Z, axis=0)
    if cap is not None:
        Z = np.clip(Z, median[None]-cap, median[None]+cap)
    Q = -np.minimum(np.abs(Z-Z[0][None]), 10.).T
    S = -np.minimum(np.abs(Z-np.median(Z, axis=0)[None]), 10.).T
    return Z, Q, S


def predict(c, model):
    kind = model['kind']
    if kind == 'base':
        return c['P'][0]
    if kind in ('mean', 'median'):
        q = c['P'].mean(0) if kind == 'mean' else np.median(c['P'], axis=0)
        return normalize(q) if c['task'] == 'classification' else q
    if kind == 'ridge':
        Z = (c['P']-c['mu'])/c['sigma']
        return (np.asarray(model['beta'])@Z+model['intercept'])*c['sigma']+c['mu']
    if kind == 'loglinear':
        logits = np.einsum('r,rnc->nc', np.asarray(model['beta']), np.log(np.clip(c['P'], 1e-12, 1.)))
        return softmax(logits, axis=1)
    P, Q, S = components(c, model.get('cap'))
    p = np.asarray(model['parameters'])
    weights = softmax(p[:6]+Q*p[6:12]+S*p[12:18], axis=1)
    if c['task'] == 'classification':
        return normalize(np.einsum('nr,rnc->nc', weights, P))
    return (weights*P.T).sum(1)*c['sigma']+c['mu']


def score(cells, model):
    grouped = {}
    for c in cells:
        value = metrics(c['y'], predict(c, model), c['task'], max(float(c['y'].std()), 1e-12))
        if c['task'] == 'classification':
            primary = value['auc'] if value['auc'] is not None else value['accuracy']
            secondary = value['accuracy']
        else:
            primary, secondary = value['r2'], -value['nrmse']
        grouped.setdefault(c['data_id'], {}).setdefault(c['family'], []).append([primary, secondary])
    a = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        for families in grouped.values()])
    return {'primary': float(a[:, 0].mean()), 'secondary': float(a[:, 1].mean()), 'datasets': len(a)}


def weights(cells):
    counts = {}
    for c in cells:
        count = counts.setdefault(c['data_id'], {})
        count[c['family']] = count.get(c['family'], 0)+1
    return [np.full(len(c['y']), 1./(len(counts)*len(counts[c['data_id']])*
        counts[c['data_id']][c['family']]*len(c['y']))) for c in cells]


def candidates(task):
    fixed = [{'kind': k} for k in ('base', 'mean', 'median')]
    for reg in REGS:
        if task == 'classification':
            fixed += [{'kind': k, 'regularization': reg} for k in ('convex', 'adaptive', 'loglinear')]
        else:
            fixed += [{'kind': 'ridge', 'regularization': reg},
                {'kind': 'adaptive', 'regularization': reg, 'cap': None},
                {'kind': 'adaptive', 'regularization': reg, 'cap': 2.}]
    return fixed


def fit(cells, spec):
    task, kind = cells[0]['task'], spec['kind']
    if kind in ('base', 'mean', 'median'):
        return spec.copy()
    reg = spec['regularization']
    cell_weights = weights(cells)
    if kind == 'ridge':
        A, Y, W = [], [], []
        for c, w in zip(cells, cell_weights):
            A.append(np.c_[((c['P']-c['mu'])/c['sigma']).T, np.ones(len(c['y']))])
            Y.append((c['y']-c['mu'])/c['sigma'])
            W.append(w*c['sigma']**2/max(float(c['y'].var()), 1e-12))
        A, Y, W = np.concatenate(A), np.concatenate(Y), np.concatenate(W)
        prior = np.r_[1., np.zeros(6)]
        p = np.linalg.solve((A*W[:, None]).T@A+reg*np.eye(7), A.T@(W*Y)+reg*prior)
        return spec | {'beta': p[:6].tolist(), 'intercept': float(p[6]),
            'objective': float(np.sum(W*(A@p-Y)**2)+reg*np.sum((p-prior)**2)), 'optimizer_success': True}
    if kind == 'loglinear':
        groups = {}
        for c, w in zip(cells, cell_weights):
            K = c['P'].shape[-1]
            groups.setdefault(K, []).append((np.log(np.clip(c['P'].transpose(1, 2, 0), 1e-12, 1.)), c['y'].astype(int), w))
        groups = [tuple(np.concatenate([p[i] for p in values]) for i in range(3)) for values in groups.values()]
        prior = np.r_[1., np.zeros(5)]
        def objective(beta):
            value = reg*np.sum((beta-prior)**2)
            grad = 2*reg*(beta-prior)
            for X, y, w in groups:
                logits = X@beta
                normalizer = logsumexp(logits, axis=1)
                value += np.sum(w*(normalizer-logits[np.arange(len(y)), y]))
                residual = np.exp(logits-normalizer[:, None])
                residual[np.arange(len(y)), y] -= 1
                grad += np.einsum('nc,ncr,n->r', residual, X, w, optimize=True)
            return float(value), grad
        result = minimize(objective, prior, jac=True, method='L-BFGS-B',
            bounds=[(-5., 5.)]*6, options={'maxiter': 160, 'ftol': 1e-9})
        return spec | {'beta': result.x.tolist(), 'objective': float(result.fun),
            'optimizer_success': bool(result.success)}
    fields = {k: [] for k in ('Q', 'S', 'Y', 'P', 'W')}
    pairs = {k: [] for k in ('posQ', 'posS', 'posP', 'negQ', 'negS', 'negP', 'W')}
    rng = np.random.default_rng(0)
    for c, w in zip(cells, cell_weights):
        P, Q, S = components(c, spec.get('cap'))
        fields['Q'].append(Q)
        fields['S'].append(S)
        if task == 'classification':
            fields['P'].append(P[:, np.arange(len(c['y'])), c['y'].astype(int)].T)
            fields['Y'].append(np.zeros(len(c['y'])))
            fields['W'].append(w)
            labels = np.unique(c['y'])
            for label in labels:
                if len(labels) == 1:
                    continue
                pos = rng.choice(np.flatnonzero(c['y'] == label), 16, replace=True)
                neg = rng.choice(np.flatnonzero(c['y'] != label), 16, replace=True)
                for side, ids in (('pos', pos), ('neg', neg)):
                    pairs[side+'Q'].append(Q[ids])
                    pairs[side+'S'].append(S[ids])
                    pairs[side+'P'].append(P[:, ids, int(label)].T)
                pairs['W'].append(np.full(16, w.sum()/len(labels)/16))
        else:
            fields['P'].append(P.T)
            fields['Y'].append((c['y']-c['mu'])/c['sigma'])
            fields['W'].append(w*c['sigma']**2/max(float(c['y'].var()), 1e-12))
    t = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float64) for k, v in fields.items()}
    pair = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float64) for k, v in pairs.items()} if task == 'classification' else {}
    initial = np.r_[3., np.zeros(5), np.ones(12)]
    if kind == 'convex':
        initial[6:] = 0.
    prior = torch.as_tensor(initial, dtype=torch.float64)
    def probability(p, q, s, bank):
        return (torch.softmax(p[:6]+q*p[6:12]+s*p[12:18], dim=1)*bank).sum(1)
    def objective(values):
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        prediction = probability(p, t['Q'], t['S'], t['P'])
        if task == 'classification':
            loss = -.5*(prediction.clamp(1e-12, 1).log()*t['W']).sum()
            positive = probability(p, pair['posQ'], pair['posS'], pair['posP']).clamp(1e-7, 1-1e-7)
            negative = probability(p, pair['negQ'], pair['negS'], pair['negP']).clamp(1e-7, 1-1e-7)
            loss += .5*(torch.nn.functional.softplus(-(torch.logit(positive)-torch.logit(negative)))*pair['W']).sum()
        else:
            loss = ((prediction-t['Y']).square()*t['W']).sum()
        loss += reg*(p-prior).square().mean()
        loss.backward()
        return float(loss.detach()), p.grad.detach().numpy()
    bounds = [(-10., 10.)]*18
    if kind == 'convex':
        bounds[6:] = [(0., 0.)]*12
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 160, 'ftol': 1e-9, 'gtol': 1e-6})
    assert np.isfinite(result.x).all()
    return spec | {'parameters': result.x.tolist(), 'objective': float(result.fun),
        'optimizer_success': bool(result.success)}
