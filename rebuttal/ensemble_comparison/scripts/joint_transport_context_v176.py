"""Mild source-feature transport with source labels and Base pseudo outcomes.

Inspired by joint-distribution OT, but not an official JDOT reproduction:
the Base predictor is fixed and only new context views are constructed.
No true query outcome, corruption identity, or clean source counterpart enters.
"""
import numpy as np
from scipy.special import logsumexp
from domain_overlap_context_v166 import encode


def coupling(cost, epsilon=.15, max_iterations=200):
    """Approximate balanced entropic transport, with recorded residual error."""
    cost = np.asarray(cost, dtype=np.float64)
    assert np.isfinite(cost).all() and cost.ndim == 2
    n, m = cost.shape
    kernel = -cost/epsilon
    u, v = np.zeros(n), np.zeros(m)
    loga, logb = -np.log(n), -np.log(m)
    for iteration in range(max_iterations):
        u = loga-logsumexp(kernel+v[None, :], axis=1)
        v = logb-logsumexp(kernel+u[:, None], axis=0)
        if iteration % 10 == 9:
            plan = np.exp(kernel+u[:, None]+v[None, :])
            if max(np.max(np.abs(plan.sum(axis=1)-1/n)),
                   np.max(np.abs(plan.sum(axis=0)-1/m))) < 1e-7:
                break
    plan = np.exp(kernel+u[:, None]+v[None, :])
    assert np.isfinite(plan).all() and np.all(plan >= 0)
    rows = plan.sum(axis=1, keepdims=True)
    assert np.all(rows > 0)
    return plan/rows, {'iterations': iteration+1,
        'marginal_tolerance': 1e-7,
        'converged': bool(max(np.max(np.abs(rows[:, 0]-1/n)),
            np.max(np.abs(plan.sum(axis=0)-1/m))) < 1e-7),
        'row_marginal_max_error': float(np.max(np.abs(rows[:, 0]-1/n))),
        'column_marginal_max_error': float(np.max(np.abs(plan.sum(axis=0)-1/m)))}


def construct(X_source, y_source, X_target, numeric_columns, task, base_prediction):
    X, y, Q = np.asarray(X_source), np.asarray(y_source), np.asarray(X_target)
    assert task in ('classification', 'regression') and len(base_prediction) == len(Q)
    features = encode(X, Q, numeric_columns).astype(np.float64)
    if features.shape[1] > 64:
        projection = np.random.default_rng(0).normal(size=(features.shape[1], 32))/np.sqrt(32.)
        features = features@projection
    A, B = features[:len(X)], features[len(X):]
    geometry = np.maximum((A*A).sum(axis=1)[:, None]+(B*B).sum(axis=1)[None]-2*A@B.T, 0.)
    positive = geometry[geometry > 1e-12]
    geometry /= max(float(np.median(positive)) if len(positive) else 1., 1e-12)
    if task == 'classification':
        p = np.asarray(base_prediction)
        semantic = -np.log(np.clip(p[:, y.astype(int)].T, 1e-6, 1.))
    else:
        sigma = max(float(y.std()), 1e-12)
        semantic = ((y[:, None]-np.asarray(base_prediction)[None])/sigma)**2
    semantic = np.clip(semantic, 0., 20.)
    positive = semantic[semantic > 1e-12]
    semantic /= max(float(np.median(positive)) if len(positive) else 1., .25)
    weights, info = coupling(np.clip(geometry+.5*semantic, 0., 20.))
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = np.setdiff1d(np.arange(X.shape[1]), numeric)
    mapped = weights@Q[:, numeric] if len(numeric) else np.empty((len(X), 0))
    result = []
    for gain in (.25, .5):
        repaired = X.astype(np.float32).copy()
        repaired[:, numeric] = (1-gain)*X[:, numeric]+gain*mapped
        if gain >= .5:
            for column in categorical:
                codes, inverse = np.unique(Q[:, column], return_inverse=True)
                probabilities = weights@np.eye(len(codes))[inverse]
                best = probabilities.argmax(axis=1)
                confident = probabilities.max(axis=1) >= .9
                repaired[confident, column] = codes[best[confident]]
        assert np.isfinite(repaired).all()
        result.append(repaired)
    return result, info | {'semantic_cost_weight': .5, 'entropy_epsilon': .15,
        'gains': [.25, .5], 'source_labels_modified': False,
        'query_outcomes_used': False, 'Base_pseudo_outcomes_used': True}
