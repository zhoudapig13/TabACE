"""Explicit nonlinear residual-fusion adaptation; not the original Shared.

Only other-dataset calibration labels enter fitting. At deployment the model
accepts six branch predictions and source label scale, never query labels,
perturbation identity, severity, or a clean source counterfactual.
"""
import base64
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import adapt_shared_heads_v119 as original

FAMILIES = ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization')


def features(cell):
    P = np.asarray(cell['P'], dtype=np.float64)
    if cell['task'] == 'regression':
        Z = (P-cell['mu'])/cell['sigma']
        differences = Z[1:]-Z[0]
        median = np.median(Z, axis=0)
        summary = np.array([np.mean(z) for z in Z]+[np.std(z) for z in Z])
        return np.column_stack([Z.T, differences.T, np.abs(differences).T,
            median, Z.std(axis=0), Z[0]-median,
            np.broadcast_to(summary, (Z.shape[1], len(summary)))]).clip(-30., 30.)
    R, N, K = P.shape
    clipped = np.clip(P, 1e-9, 1.)
    probability = P.transpose(1, 2, 0)
    confidence = 1+(clipped*np.log(clipped)).sum(axis=2)/np.log(max(K, 2))
    gap = P.max(axis=2)[:, :, None]-P
    summaries = np.concatenate([
        np.broadcast_to(confidence.T[:, None, :], (N, K, R)),
        np.broadcast_to(P.std(axis=1).T[None], (N, K, R))], axis=2)
    X = np.concatenate([probability, np.log(clipped).transpose(1, 2, 0),
        probability-probability[:, :, :1], gap.transpose(1, 2, 0), summaries,
        np.full((N, K, 1), np.log(K))], axis=2)
    return X.reshape(N*K, -1).clip(-30., 30.)


def _encode(model):
    return base64.b64encode(pickle.dumps(model, protocol=5)).decode('ascii')


_MODEL_CACHE = {}


def _decode(encoded):
    # Only internally produced, sealed local models are passed here.
    if encoded not in _MODEL_CACHE:
        if len(_MODEL_CACHE) >= 64:
            _MODEL_CACHE.clear()
        _MODEL_CACHE[encoded] = pickle.loads(base64.b64decode(encoded))
    return _MODEL_CACHE[encoded]


def predict(cell, model):
    if model['kind'] != 'residual_hgb':
        return original.predict(cell, model)
    delta = _decode(model['serialized_estimator']).predict(features(cell))
    scale = model['strength']
    if cell['task'] == 'regression':
        # Do not let a cross-dataset correction become an unbounded regressor.
        return cell['P'][0]+scale*np.clip(delta, -1., 1.)*cell['sigma']
    K = cell['P'].shape[2]
    q = np.clip(cell['P'][0]+scale*delta.reshape(-1, K), 1e-9, 1.)
    return q/q.sum(axis=1, keepdims=True)


def fit(cells, specification):
    if specification['kind'] != 'residual_hgb':
        return original.fit(cells, specification)
    X, Y, W = [], [], []
    for c, w in zip(cells, original.weights(cells)):
        # Fixed deterministic subsampling controls compute, not outcomes.
        n = len(c['y'])
        cap = 32 if c['task'] == 'classification' else 128
        ids = np.linspace(0, n-1, min(n, cap), dtype=int)
        values = features(c)
        if c['task'] == 'regression':
            X.append(values[ids])
            Y.append(((c['y']-c['P'][0])/c['sigma'])[ids])
            W.append(w[ids]*n/len(ids)*c['sigma']**2/max(float(c['y'].var()), 1e-12))
        else:
            K = c['P'].shape[2]
            X.append(values.reshape(n, K, -1)[ids].reshape(len(ids)*K, -1))
            target = np.eye(K)[c['y'][ids].astype(int)]-c['P'][0, ids]
            Y.append(target.ravel())
            W.append(np.repeat(w[ids]*n/len(ids)/K, K))
    X, Y, W = np.concatenate(X), np.concatenate(Y), np.concatenate(W)
    # sklearn's regularization is tied to the sample-weight mass; normalize
    # it explicitly so dataset count changes do not change shrinkage silently.
    W = W*len(W)/W.sum()
    estimator = HistGradientBoostingRegressor(loss='squared_error',
        max_iter=80, learning_rate=.05, max_leaf_nodes=specification['max_leaf_nodes'],
        min_samples_leaf=64, l2_regularization=10., early_stopping=False,
        random_state=0)
    estimator.fit(X, Y, sample_weight=W)
    return specification | {'serialized_estimator': _encode(estimator),
        'training_rows': len(Y), 'training_features': X.shape[1],
        'query_labels_required_at_deployment': False}


def score(cells, model):
    # Reuse the established dataset -> family -> severity metric aggregation.
    temporary = []
    for c in cells:
        temporary.append(c | {'P': np.stack([predict(c, model)])})
    return original.score(temporary, {'kind': 'base'})


def candidates(task):
    values = [{'kind': 'base'}, {'kind': 'mean'},
        {'kind': 'adaptive', 'regularization': .01}]
    values.append({'kind': 'loglinear' if task == 'classification' else 'ridge',
        'regularization': .01})
    for leaves in (7, 15):
        for strength in (.5, 1.):
            values.append({'kind': 'residual_hgb', 'max_leaf_nodes': leaves,
                'strength': strength})
    return values


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
            # Persist each estimator once, not once for every blending strength.
            summary = {k: v for k, v in model.items() if k != 'serialized_estimator'}
            inner.append({'fold': fold, 'model_summary': summary,
                'scores_by_family': family_scores, 'training_data_ids': sorted(train_ids),
                'validation_data_ids': sorted(val_ids)})
        gains = {f: sum(n*v for n, v in scores)/sum(n for n, v in scores)
            for f, scores in per_family.items()}
        trials.append({'specification': specification, 'inner': inner,
            'family_gains': gains, 'score': {'worst_family_gain': min(gains.values()),
                'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
