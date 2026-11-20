"""Raw excess-variance contraction before any marginal/CDF matching.

This tests a measurement-error hypothesis, not an assumed universal repair.
Source/query task labels, generator names, and severity are absent from the
API. Categorical coordinates remain unchanged. Context labels are preserved
by the caller. All transforms use observed source and unlabeled query X only.
"""
import numpy as np


def fit_target(X_target, numeric_columns):
    Q = np.asarray(X_target, dtype=np.float64)
    numeric = np.asarray(numeric_columns, dtype=int)
    raw = Q[:, numeric]
    center = raw.mean(axis=0)
    sd = raw.std(axis=0)
    iqr = np.percentile(raw, 75, axis=0)-np.percentile(raw, 25, axis=0)
    scale = np.where(iqr > 1e-8, iqr, np.maximum(sd, 1e-6))
    active = sd > 1e-8
    model = {'numeric': numeric, 'center': center, 'scale': scale, 'active': active}
    if active.any():
        Z = ((raw-center)/scale)[:, active]
        _, singular, basis = np.linalg.svd(Z, full_matrices=False)
        rank = min(32, int(active.sum()), max(len(Q)-1, 0))
        basis = basis[:rank]
        spectrum = singular[:rank]**2/max(len(Q)-1, 1)
        floor = max(.01*float(spectrum.mean()) if rank else 0., 1e-8)
        covariance = np.diag(spectrum+floor)
        residual = Z-(Z@basis.T)@basis
        low, high = np.quantile(residual, [.005, .995], axis=0)
        margin = .25*np.maximum(high-low, .1)
        model.update(basis=basis, covariance=covariance, residual_low=low-margin, residual_high=high+margin)
    else:
        model.update(basis=np.zeros((0, 0)), covariance=np.zeros((0, 0)))
    return model


def construct(X_source, target_model):
    X = np.asarray(X_source)
    numeric, active = target_model['numeric'], target_model['active']
    proposal = X.astype(np.float64).copy()
    if len(numeric):
        proposal[:, numeric[~active]] = target_model['center'][~active]
    rank = len(target_model['basis'])
    noise_trace, source_trace = 0., 0.
    if active.any() and rank:
        columns = numeric[active]
        scale = target_model['scale'][active]
        # Center the OBSERVED source separately, retaining raw variance rather
        # than first replacing its marginals by query quantiles.
        source_center = X[:, columns].mean(axis=0)
        Z = (np.asarray(X[:, columns], dtype=np.float64)-source_center)/scale
        basis, target_cov = target_model['basis'], target_model['covariance']
        projected = Z@basis.T
        covariance = projected.T@projected/max(len(X)-1, 1)
        difference = .5*((covariance-target_cov)+(covariance-target_cov).T)
        eigen, axes = np.linalg.eigh(difference)
        noise = (axes*np.maximum(eigen, 0.))@axes.T
        epsilon = max(1e-12, 1e-10*float(np.trace(target_cov))/rank)
        transform = np.linalg.solve(target_cov+noise+epsilon*np.eye(rank), target_cov)
        residual = Z-projected@basis
        residual = np.clip(residual, target_model['residual_low'], target_model['residual_high'])
        repaired = (projected@transform)@basis+residual
        proposal[:, columns] = target_model['center'][active]+repaired*scale
        noise_trace, source_trace = float(np.trace(noise)), float(np.trace(covariance))
    outputs = []
    for gain in (.5, 1.):
        result = np.asarray(X, dtype=np.float64).copy()
        result[:, numeric] = (1-gain)*X[:, numeric]+gain*proposal[:, numeric]
        result = result.astype(np.float32)
        categorical = np.setdiff1d(np.arange(X.shape[1]), numeric)
        np.testing.assert_array_equal(result[:, categorical], np.asarray(X[:, categorical], dtype=np.float32))
        assert np.isfinite(result).all()
        outputs.append({'X': result, 'info': {'rule': 'raw_excess_covariance_contraction', 'gain': gain,
            'numeric_features': len(numeric), 'target_latent_rank': rank,
            'raw_source_projected_variance_trace': source_trace, 'estimated_excess_variance_trace': noise_trace,
            'marginal_CDF_alignment_applied_first': False,
            'source_task_labels_used': False, 'query_task_labels_used': False,
            'categorical_values_unchanged': True, 'source_and_query_mean_alignment': True,
            'orthogonal_residual_bounds_query_quantiles': [.005, .995],
            'orthogonal_residual_extra_margin': .25}})
    return outputs
