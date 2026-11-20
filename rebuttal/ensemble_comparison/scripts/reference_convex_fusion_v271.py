"""Fixed regularized convex fusion using independent calibration labels only."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def fit(P, y_calibration, task, scale):
    P, y = np.asarray(P, dtype=np.float64), np.asarray(y_calibration)
    assert P.shape[0] == 4 and P.shape[1] == len(y)
    assert np.isfinite(P).all() and np.isfinite(y).all()
    prior = np.array([.5, .5, 0., 0.])
    regularization = .02
    if task == 'classification':
        A = P[:, np.arange(len(y)), y.astype(int)]
    else:
        A, target = P/max(scale, 1e-12), y/max(scale, 1e-12)
    def objective(logits):
        w = softmax(logits)
        p = w@A
        if task == 'classification':
            loss = -np.log(np.maximum(p, 1e-12)).mean()
            gradient = -(A/np.maximum(p, 1e-12)).mean(1)
        else:
            residual = p-target
            loss = np.mean(residual**2)
            gradient = 2*(A*residual).mean(1)
        loss += regularization*np.sum((w-prior)**2)
        gradient += 2*regularization*(w-prior)
        return float(loss), w*(gradient-w@gradient)
    result = minimize(objective, np.array([2., 2., 0., 0.]), jac=True,
        method='L-BFGS-B', bounds=[(-8., 8.)]*4,
        options={'maxiter': 200, 'ftol': 1e-11})
    assert np.isfinite(result.x).all()
    return {'weights': softmax(result.x).tolist(), 'regularization': regularization,
        'calibration_rows': len(y), 'optimizer_success': bool(result.success),
        'loss': float(result.fun)}


def predict(P, model):
    w = np.asarray(model['weights'])
    assert len(w)==4 and np.isclose(w.sum(), 1.) and np.all(w>=0)
    return np.tensordot(w, P, axes=(0, 0))
