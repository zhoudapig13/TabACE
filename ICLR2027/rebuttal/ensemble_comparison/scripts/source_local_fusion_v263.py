"""Three-expert fusion fitted on existing SOURCE out-of-fold labels only.

Fixed regularizations and minimum support; no query-label argument, no task
query metric selection. Predictions consume only already computed forecasts.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from run_tabicl import normalize


def fit(P, y_source, task, mu, sigma, kind):
    P, y = np.asarray(P, dtype=np.float64), np.asarray(y_source)
    assert P.shape[0] == 3 and P.shape[1] == len(y)
    assert task in ('classification', 'regression') and kind in ('convex', 'stack')
    if len(y) < 16:
        return {'kind': 'base', 'source_scored_rows': len(y), 'minimum_rows': 16}
    assert np.isfinite(P).all() and np.isfinite(y).all()
    prior = np.r_[1., 0., 0.]
    if kind == 'convex':
        regularization = .01
        if task == 'classification':
            data = P[:, np.arange(len(y)), y.astype(int)]
        else:
            data = (P-mu)/sigma
            target = (y-mu)/sigma
        def objective(logits):
            weights = softmax(logits)
            mixed = weights@data
            if task == 'classification':
                value = -np.log(np.clip(mixed, 1e-12, 1.)).mean()
                gradient = -(data/np.clip(mixed, 1e-12, None)).mean(1)
            else:
                residual = mixed-target
                value = np.mean(residual**2)
                gradient = 2*(data*residual).mean(1)
            value += regularization*np.sum((weights-prior)**2)
            gradient += 2*regularization*(weights-prior)
            return float(value), weights*(gradient-weights@gradient)
        result = minimize(objective, np.r_[3., 0., 0.], jac=True, method='L-BFGS-B',
            bounds=[(-8., 8.)]*3, options={'maxiter': 160, 'ftol': 1e-10})
        return {'kind': 'convex', 'weights': softmax(result.x).tolist(),
            'regularization': regularization, 'source_scored_rows': len(y),
            'optimizer_success': bool(result.success), 'objective': float(result.fun)}
    regularization = .1
    if task == 'regression':
        A = np.c_[((P-mu)/sigma).T, np.ones(len(y))]
        target = (y-mu)/sigma
        anchor = np.r_[prior, 0.]
        parameter = np.linalg.solve(A.T@A/len(y)+regularization*np.eye(4),
            A.T@target/len(y)+regularization*anchor)
        return {'kind': 'ridge', 'beta': parameter[:3].tolist(), 'intercept': float(parameter[3]),
            'regularization': regularization, 'source_scored_rows': len(y)}
    A = np.log(np.clip(P.transpose(1, 2, 0), 1e-12, 1.))
    def objective(beta):
        logits = A@beta
        value = np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(y)), y.astype(int)])
        residual = softmax(logits, axis=1)
        residual[np.arange(len(y)), y.astype(int)] -= 1
        gradient = np.einsum('nk,nkr->r', residual, A)/len(y)
        return float(value+regularization*np.sum((beta-prior)**2)), gradient+2*regularization*(beta-prior)
    result = minimize(objective, prior, jac=True, method='L-BFGS-B', bounds=[(-2., 2.)]*3,
        options={'maxiter': 160, 'ftol': 1e-10})
    return {'kind': 'loglinear', 'beta': result.x.tolist(), 'regularization': regularization,
        'source_scored_rows': len(y), 'optimizer_success': bool(result.success), 'objective': float(result.fun)}


def predict(P, model, task, mu, sigma):
    P = np.asarray(P, dtype=np.float64)
    assert P.shape[0] == 3
    kind = model['kind']
    if kind == 'base':
        return P[0].copy()
    if kind == 'convex':
        prediction = np.tensordot(np.asarray(model['weights']), P, axes=(0, 0))
        return normalize(prediction) if task == 'classification' else prediction
    if kind == 'ridge':
        return (np.asarray(model['beta'])@((P-mu)/sigma)+model['intercept'])*sigma+mu
    assert kind == 'loglinear' and task == 'classification'
    return softmax(np.einsum('r,rnk->nk', np.asarray(model['beta']), np.log(np.clip(P, 1e-12, 1.))), axis=1)
