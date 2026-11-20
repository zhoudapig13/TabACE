"""Unlabeled-target conditional feature repair, explicitly a new constructor.

Task labels are absent from the API. Categorical columns remain unchanged.
Conditional residual bounds are calibrated by target-feature cross-validation,
not by any downstream task outcome.
"""
import hashlib
import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage

_CACHE = {}


def fit_model(A):
    center = A.mean(axis=0)
    Z = A-center
    _, singular, basis = np.linalg.svd(Z, full_matrices=False)
    eigen = singular**2/max(len(A), 1)
    mean_variance = max(float(eigen.sum()/A.shape[1]), 1e-6)
    shrinkage = float(np.clip(ledoit_wolf_shrinkage(Z, assume_centered=True), .05, .95))
    ridge = max(shrinkage*mean_variance, 1e-6)
    signal = (1-shrinkage)*eigen
    fraction = signal/(ridge+signal)
    diagonal = np.maximum((1-(basis**2*fraction[:, None]).sum(axis=0))/ridge, 1e-8)
    return {'center': center, 'basis': basis, 'fraction': fraction,
        'ridge': ridge, 'diagonal': diagonal, 'shrinkage': shrinkage}


def conditional(A, model):
    Z = A-model['center']
    projected = ((Z@model['basis'].T)*model['fraction'])@model['basis']
    precision_product = (Z-projected)/model['ridge']
    return A-precision_product/model['diagonal']


def target_model(Q, numeric):
    raw = np.ascontiguousarray(Q[:, numeric], dtype=np.float64)
    key = (raw.shape, tuple(numeric), hashlib.sha256(raw.tobytes()).hexdigest())
    if key in _CACHE:
        return _CACHE[key]
    center = raw.mean(axis=0)
    sd = raw.std(axis=0)
    active = sd > 1e-8
    iqr = np.percentile(raw, 75, axis=0)-np.percentile(raw, 25, axis=0)
    scale = np.where(iqr > 1e-8, iqr, np.maximum(sd, 1e-6))
    if not active.any():
        result = {'active': active, 'center': center, 'scale': scale, 'model': None}
    else:
        standardized = ((raw-center)/scale)[:, active]
        order = np.random.default_rng(0).permutation(len(raw))
        residual = np.empty_like(standardized)
        if len(raw) >= 6:
            for valid in np.array_split(order, 3):
                train = np.setdiff1d(np.arange(len(raw)), valid)
                m = fit_model(standardized[train])
                residual[valid] = standardized[valid]-conditional(standardized[valid], m)
        else:
            residual[:] = standardized
        low, high = np.quantile(residual, [.0125, .9875], axis=0)
        width = np.maximum(high-low, .1)
        low, high = low-.25*width, high+.25*width
        result = {'active': active, 'center': center, 'scale': scale,
            'model': fit_model(standardized), 'residual_low': low, 'residual_high': high,
            'support_low': standardized.min(axis=0)-.25,
            'support_high': standardized.max(axis=0)+.25}
    if len(_CACHE) >= 4:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result


def repair(X_source, X_target, numeric_columns):
    X = np.asarray(X_source, dtype=np.float32)
    Q = np.asarray(X_target, dtype=np.float64)
    numeric = np.asarray(numeric_columns, dtype=int)
    output = X.copy()
    if not len(numeric):
        return output, {'numeric_features': 0, 'modified_cells': 0, 'task_labels_used': False}
    fitted = target_model(Q, numeric)
    active = fitted['active']
    if (~active).any():
        output[:, numeric[~active]] = fitted['center'][~active]
    if active.any():
        columns = numeric[active]
        raw = (np.asarray(X[:, columns], dtype=np.float64)-fitted['center'][active])/fitted['scale'][active]
        low, high = fitted['support_low'], fitted['support_high']
        bad = (raw < low) | (raw > high)
        bounded = np.clip(raw, low, high)
        current = bounded.copy()
        # Impute obvious marginal outliers without moving their initially
        # plausible neighbors. This avoids spreading one gross error.
        for _ in range(5):
            expected = conditional(current, fitted['model'])
            proposal = np.where(bad, expected, bounded)
            current = np.clip(.5*current+.5*proposal, low, high)
        anchor = current.copy()
        for _ in range(5):
            expected = conditional(current, fitted['model'])
            proposal = expected+np.clip(anchor-expected,
                fitted['residual_low'], fitted['residual_high'])
            current = np.clip(.5*current+.5*proposal, low, high)
        output[:, columns] = current*fitted['scale'][active]+fitted['center'][active]
    assert np.isfinite(output).all()
    categorical = sorted(set(range(X.shape[1]))-set(numeric))
    np.testing.assert_array_equal(output[:, categorical], X[:, categorical])
    return output, {'numeric_features': len(numeric), 'active_numeric_features': int(active.sum()),
        'modified_cells': int(np.sum(output != X)), 'task_labels_used': False,
        'target_features_used_transductively': True,
        'target_feature_cross_validation_folds': 3, 'outcome_labels_used_in_cross_validation': False,
        'residual_quantiles': [.0125, .9875], 'residual_interval_extra_margin': .25,
        'marginal_support_margin_standardized': .25, 'marginal_iterations': 5,
        'conditional_iterations': 5, 'iteration_damping': .5,
        'target_shrinkage': float(fitted['model']['shrinkage']) if fitted['model'] else None,
        'categorical_columns_unchanged': True}
