"""Two generic, unlabeled context-alignment adaptations, not paper Shared.

The functions receive neither labels, generator names, severities, nor clean
counterfactual source measurements. Categorical coordinates stay untouched.
"""
import hashlib

import numpy as np
from scipy.stats import rankdata

_TARGET_CACHE = {}


def align(X_source, X_target, numeric_columns):
    X = np.asarray(X_source, dtype=np.float32)
    Q = np.asarray(X_target, dtype=np.float32)
    numeric = np.asarray(numeric_columns, dtype=int)
    first = X.copy()
    second = X.copy()
    if not len(numeric):
        return first, second, {'numeric_coordinates': 0, 'target_pca_rank': 0,
            'uses_labels_or_mechanism_identity': False}
    key = (hashlib.sha256(np.ascontiguousarray(Q).tobytes()).hexdigest(), tuple(numeric))
    if key not in _TARGET_CACHE:
        raw = np.asarray(Q[:, numeric], dtype=np.float64)
        sorted_Q = np.sort(raw, axis=0)
        keep = np.flatnonzero(raw.std(axis=0) > 1e-8)
        center = raw[:, keep].mean(axis=0)
        iqr = np.percentile(raw[:, keep], 75, axis=0)-np.percentile(raw[:, keep], 25, axis=0)
        sd = raw[:, keep].std(axis=0)
        scale = np.where(iqr > 1e-8, iqr, np.maximum(sd, 1e-8))
        if len(keep):
            z = (raw[:, keep]-center)/scale
            _, singular, Vt = np.linalg.svd(z, full_matrices=False)
            rank = min(32, len(keep), len(Q)-1)
            basis = Vt[:rank]
            spectrum = singular[:rank]**2/max(len(Q)-1, 1)
            floor = max(1e-8, .01*float(spectrum.mean()))
            covariance = np.diag(spectrum+floor)
        else:
            basis, covariance = np.zeros((0, 0)), np.zeros((0, 0))
        if len(_TARGET_CACHE) >= 8:
            _TARGET_CACHE.pop(next(iter(_TARGET_CACHE)))
        _TARGET_CACHE[key] = (sorted_Q, keep, center, scale, basis, covariance)
    sorted_Q, keep, center, scale, basis, covariance = _TARGET_CACHE[key]
    # End-point empirical ranks make identical source and target samples an
    # exact identity, including ties; ties are never broken with labels.
    p = (rankdata(X[:, numeric], method='average', axis=0)-1)/max(len(X)-1, 1)
    positions = p*(len(Q)-1)
    low = np.floor(positions).astype(int)
    high = np.minimum(low+1, len(Q)-1)
    fraction = positions-low
    mapped = (1-fraction)*np.take_along_axis(sorted_Q, low, axis=0)+fraction*np.take_along_axis(sorted_Q, high, axis=0)
    first[:, numeric] = mapped.astype(np.float32)
    second[:] = first
    if len(keep):
        z = (np.asarray(first[:, numeric[keep]], dtype=np.float64)-center)/scale
        projected = z@basis.T
        centered = projected-projected.mean(axis=0)
        source_cov = centered.T@centered/max(len(projected)-1, 1)
        discrepancy = (source_cov-covariance)
        discrepancy = .5*(discrepancy+discrepancy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(discrepancy)
        noise = (eigenvectors*np.maximum(eigenvalues, 0.))@eigenvectors.T
        epsilon = max(1e-12, 1e-10*float(np.trace(covariance))/len(covariance))
        transform = np.linalg.solve(covariance+noise+epsilon*np.eye(len(covariance)), covariance)
        # Keep the unmodeled orthogonal residual; only correct excess-variance
        # directions supported by the unlabeled target covariance estimate.
        repaired = z+(projected@transform-projected)@basis
        second[:, numeric[keep]] = (center+repaired*scale).astype(np.float32)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    return first, second, {'numeric_coordinates': int(len(numeric)),
        'target_pca_rank': int(len(basis)), 'excess_covariance_positive_part': True,
        'uses_labels_or_mechanism_identity': False, 'retains_orthogonal_residual': True}
