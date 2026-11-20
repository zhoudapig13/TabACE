"""Small, outcome-free source/query distribution descriptors for fusion.

This does not identify a registered corruption by name, receive severity, or
access task labels. It describes observable marginal and dependence changes.
"""
import numpy as np

NAMES = ('log_feature_count', 'numeric_fraction', 'log_matched_unique_count_ratio',
    'log_numeric_IQR_ratio', 'log_numeric_tail_ratio', 'numeric_location_gap',
    'numeric_absolute_correlation_gap', 'numeric_correlation_difference',
    'categorical_marginal_TV', 'constant_numeric_fraction_gap')


def descriptors(X_source, X_target, numeric_columns):
    X, Q = np.asarray(X_source, dtype=np.float64), np.asarray(X_target, dtype=np.float64)
    assert X.ndim == Q.ndim == 2 and X.shape[1] == Q.shape[1]
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = np.setdiff1d(np.arange(X.shape[1]), numeric)
    k = min(len(X), len(Q), 256)
    # Match row counts so discrete features do not look shifted merely because
    # the two observed domains have different sample sizes.
    rng = np.random.default_rng(0)
    S, T = X[rng.permutation(len(X))[:k]], Q[rng.permutation(len(Q))[:k]]
    values = np.zeros(len(NAMES), dtype=np.float64)
    values[0] = np.log1p(X.shape[1])/np.log(2049.)
    values[1] = len(numeric)/max(X.shape[1], 1)
    if len(numeric):
        cols = numeric[np.random.default_rng(0).permutation(len(numeric))[:64]]
        A, B = S[:, cols], T[:, cols]
        unique_s = np.array([len(np.unique(A[:, j])) for j in range(len(cols))])
        unique_t = np.array([len(np.unique(B[:, j])) for j in range(len(cols))])
        values[2] = np.median(np.log((unique_s+.5)/(unique_t+.5)))
        center_s, center_t = np.median(A, axis=0), np.median(B, axis=0)
        iqr_s, iqr_t = (np.percentile(v, 75, axis=0)-np.percentile(v, 25, axis=0) for v in (A, B))
        floor = np.maximum(B.std(axis=0)*.01, 1e-8)
        values[3] = np.median(np.log((iqr_s+floor)/(iqr_t+floor)))
        tail_s = np.percentile(np.abs(A-center_s), 95, axis=0)
        tail_t = np.percentile(np.abs(B-center_t), 95, axis=0)
        values[4] = np.median(np.log((tail_s+floor)/(tail_t+floor)))
        scale = np.maximum(iqr_t, np.maximum(B.std(axis=0), 1e-8))
        values[5] = np.median(np.abs(center_s-center_t)/scale)
        varying = (A.std(axis=0) > 1e-8) & (B.std(axis=0) > 1e-8)
        if varying.sum() >= 2:
            a = np.clip((A[:, varying]-center_t[varying])/scale[varying], -5., 5.)
            b = np.clip((B[:, varying]-center_t[varying])/scale[varying], -5., 5.)
            a = (a-a.mean(axis=0))/np.maximum(a.std(axis=0), 1e-8)
            b = (b-b.mean(axis=0))/np.maximum(b.std(axis=0), 1e-8)
            ca, cb = a.T@a/len(a), b.T@b/len(b)
            upper = np.triu_indices(len(ca), 1)
            values[6] = np.mean(np.abs(ca[upper])-np.abs(cb[upper]))
            values[7] = np.mean(np.abs(ca[upper]-cb[upper]))
        values[9] = np.mean(A.std(axis=0) < 1e-8)-np.mean(B.std(axis=0) < 1e-8)
    if len(categorical):
        tv = []
        for j in categorical[np.random.default_rng(0).permutation(len(categorical))[:64]]:
            codes = np.union1d(S[:, j], T[:, j])
            a, b = (np.array([(v[:, j] == code).mean() for code in codes]) for v in (S, T))
            tv.append(.5*np.abs(a-b).sum())
        values[8] = np.mean(tv)
    values = np.clip(values, -4., 4.)
    assert values.shape == (10,) and np.isfinite(values).all()
    return values
