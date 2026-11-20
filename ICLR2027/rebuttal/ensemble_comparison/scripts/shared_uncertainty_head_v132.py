"""Regression-only uncertainty-aware fusion adaptation on fixed predictions."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingRegressor

import adapt_shared_heads_v119 as original
import shared_residual_head_v129 as residual

FAMILIES = residual.FAMILIES


def components(cell):
    assert cell['task'] == 'regression'
    Z = (cell['P']-cell['mu'])/cell['sigma']
    width = np.clip(np.asarray(cell['IQR'])/cell['sigma'], 0., 100.)
    Q = -np.log1p(width).T
    S = -np.minimum(np.abs(Z-np.median(Z, axis=0)), 10.).T
    return Z, Q, S


def features(cell):
    baseline = residual.features(cell)
    _, Q, _ = components(cell)
    width = np.asarray(cell['IQR'])/cell['sigma']
    relative = width/np.maximum(np.median(width, axis=0), .01)
    summary = np.broadcast_to(Q.mean(axis=0), Q.shape)
    return np.column_stack([baseline, Q, relative.T, summary]).clip(-30., 30.)


def predict(cell, model):
    kind = model['kind']
    if kind == 'iqr_adaptive':
        Z, Q, S = components(cell)
        p = np.asarray(model['parameters'])
        weights = softmax(p[:6]+Q*p[6:12]+S*p[12:18], axis=1)
        return (weights*Z.T).sum(axis=1)*cell['sigma']+cell['mu']
    if kind == 'residual_hgb':
        delta = residual._decode(model['serialized_estimator']).predict(features(cell))
        return cell['P'][0]+model['strength']*np.clip(delta, -1., 1.)*cell['sigma']
    return original.predict(cell, model)


def fit(cells, specification):
    kind = specification['kind']
    if kind not in ('iqr_adaptive', 'residual_hgb'):
        return original.fit(cells, specification)
    if kind == 'residual_hgb':
        X, Y, W = [], [], []
        for c, w in zip(cells, original.weights(cells)):
            n = len(c['y'])
            ids = np.linspace(0, n-1, min(n, 128), dtype=int)
            X.append(features(c)[ids])
            Y.append(((c['y']-c['P'][0])/c['sigma'])[ids])
            W.append(w[ids]*n/len(ids)*c['sigma']**2/max(float(c['y'].var()), 1e-12))
        X, Y, W = np.concatenate(X), np.concatenate(Y), np.concatenate(W)
        W = W*len(W)/W.sum()
        estimator = HistGradientBoostingRegressor(loss='squared_error', max_iter=80,
            learning_rate=.05, max_leaf_nodes=specification['max_leaf_nodes'],
            min_samples_leaf=64, l2_regularization=10., early_stopping=False, random_state=0)
        estimator.fit(X, Y, sample_weight=W)
        return specification | {'serialized_estimator': residual._encode(estimator),
            'training_rows': len(Y), 'training_features': X.shape[1],
            'query_labels_required_at_deployment': False}
    fields = {k: [] for k in ('P', 'Q', 'S', 'Y', 'W')}
    for c, w in zip(cells, original.weights(cells)):
        P, Q, S = components(c)
        fields['P'].append(P.T)
        fields['Q'].append(Q)
        fields['S'].append(S)
        fields['Y'].append((c['y']-c['mu'])/c['sigma'])
        fields['W'].append(w*c['sigma']**2/max(float(c['y'].var()), 1e-12))
    tensors = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float64) for k, v in fields.items()}
    initial = np.r_[3., np.zeros(5), np.ones(12)]
    prior = torch.as_tensor(initial, dtype=torch.float64)
    regularization = specification['regularization']
    def objective(values):
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        weights = torch.softmax(p[:6]+tensors['Q']*p[6:12]+tensors['S']*p[12:18], dim=1)
        prediction = (weights*tensors['P']).sum(dim=1)
        loss = ((prediction-tensors['Y']).square()*tensors['W']).sum()
        loss += regularization*(p-prior).square().mean()
        loss.backward()
        return float(loss.detach()), p.grad.detach().numpy()
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
        bounds=[(-10., 10.)]*6+[(0., 10.)]*12,
        options={'maxiter': 160, 'ftol': 1e-9, 'gtol': 1e-6})
    assert np.isfinite(result.x).all()
    return specification | {'parameters': result.x.tolist(), 'objective': float(result.fun),
        'optimizer_success': bool(result.success), 'interval_quality_coefficients_nonnegative': True}


def score(cells, model):
    copied = [c | {'P': np.stack([predict(c, model)])} for c in cells]
    return original.score(copied, {'kind': 'base'})


def candidates(task):
    assert task == 'regression'
    return residual.candidates(task)+[{'kind': 'iqr_adaptive', 'regularization': r}
        for r in (.001, .01, .1)]


def choose(source):
    folds = sorted({c['fold'] for c in source})
    families = sorted({c['family'] for c in source if c['family'] != 'clean'})
    assert len(folds) == 4
    trials, cache = [], {}
    for specification in candidates(source[0]['task']):
        inner, per_family = [], {f: [] for f in families}
        for fold in folds:
            training = [c for c in source if c['fold'] != fold]
            validation = [c for c in source if c['fold'] == fold]
            train_ids, val_ids = {c['data_id'] for c in training}, {c['data_id'] for c in validation}
            assert train_ids.isdisjoint(val_ids)
            if specification['kind'] == 'residual_hgb':
                key = (fold, specification['max_leaf_nodes'])
                if key not in cache:
                    cache[key] = fit(training, specification)
                model = cache[key] | {'strength': specification['strength']}
            else:
                model = fit(training, specification)
            family_scores = {}
            for family in families:
                values = [c for c in validation if c['family'] == family]
                a = score(values, model)
                b = original.score(values, {'kind': 'base'})
                gain = a['primary']-b['primary']
                per_family[family].append((a['datasets'], gain))
                family_scores[family] = {'candidate': a, 'Base': b, 'gain': gain}
            inner.append({'fold': fold,
                'model_summary': {k: v for k, v in model.items() if k != 'serialized_estimator'},
                'scores_by_family': family_scores, 'training_data_ids': sorted(train_ids),
                'validation_data_ids': sorted(val_ids)})
        gains = {f: sum(n*v for n, v in scores)/sum(n for n, v in scores)
            for f, scores in per_family.items()}
        trials.append({'specification': specification, 'inner': inner, 'family_gains': gains,
            'score': {'worst_family_gain': min(gains.values()),
                'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
