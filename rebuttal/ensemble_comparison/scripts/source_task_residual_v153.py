"""Per-dataset source-only residual calibration; no query outcomes accepted.

Residual targets use cached out-of-fold predictions for the observed source
context. Source task labels are existing training labels, not extra labels.
"""
import numpy as np

SPECIFICATIONS = [{'kernel': kernel, 'regularization': value}
    for kernel in ('linear', 'rbf') for value in (.01, .1, 1.)]


def representation(X_source, X_target, numeric_columns):
    X, Q = np.asarray(X_source, dtype=np.float64), np.asarray(X_target, dtype=np.float64)
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = sorted(set(range(X.shape[1]))-set(numeric))
    a, b = [], []
    if len(numeric):
        V = X[:, numeric]
        center = np.median(V, axis=0)
        spread = np.percentile(V, 75, axis=0)-np.percentile(V, 25, axis=0)
        spread = np.where(spread > 1e-8, spread, np.maximum(V.std(axis=0), 1e-6))
        a.append(np.clip((V-center)/spread, -5., 5.))
        b.append(np.clip((Q[:, numeric]-center)/spread, -5., 5.))
    for column in categorical:
        categories = np.unique(X[:, column])
        a.append((X[:, column, None] == categories[None]).astype(float))
        b.append((Q[:, column, None] == categories[None]).astype(float))
    if not a:
        return np.ones((len(X), 1)), np.ones((len(Q), 1))
    return np.column_stack(a), np.column_stack(b)


def fit_predict(X_source, y_source, X_target, numeric_columns,
                source_cv_prediction, task):
    X, Q, y = np.asarray(X_source), np.asarray(X_target), np.asarray(y_source)
    cv = np.asarray(source_cv_prediction, dtype=np.float64)
    assert len(X) == len(y) == len(cv)
    valid = np.isfinite(cv).all(axis=1) if cv.ndim == 2 else np.isfinite(cv)
    if task == 'classification':
        K = cv.shape[1]
        residual = np.eye(K)[y[valid].astype(int)]-cv[valid]
        supported_classes = np.bincount(y[valid].astype(int), minlength=K) > 0
        shape = (len(Q), K)
    else:
        residual = (y[valid]-cv[valid])/max(float(y.std()), 1e-12)
        supported_classes = None
        shape = (len(Q),)
    if valid.sum() < 4:
        return np.zeros((len(SPECIFICATIONS),)+shape), {
            'source_rows': len(y), 'scored_source_rows': int(valid.sum()),
            'no_query_outcome_labels': True, 'too_few_scored_source_rows': True}
    A, B = representation(X, Q, numeric_columns)
    A = A[valid]
    dimensions = max(X.shape[1], 1)
    source_dot, query_dot = (A@A.T)/dimensions, (B@A.T)/dimensions
    source_distance = np.maximum(np.diag(source_dot)[:, None]+np.diag(source_dot)[None]-2*source_dot, 0.)
    query_distance = np.maximum((B*B).sum(axis=1)[:, None]/dimensions+
        np.diag(source_dot)[None]-2*query_dot, 0.)
    positive = source_distance[np.triu_indices(len(A), 1)]
    positive = positive[positive > 1e-12]
    bandwidth = max(float(np.median(positive)) if len(positive) else 1., 1e-8)
    kernels = {'linear': (source_dot, query_dot),
        'rbf': (np.exp(-source_distance/bandwidth), np.exp(-query_distance/bandwidth))}
    mean = residual.mean(axis=0)
    centered = residual-mean
    predictions = []
    for name in ('linear', 'rbf'):
        train, query = kernels[name]
        eigen, vectors = np.linalg.eigh((train+train.T)/2)
        eigen = np.maximum(eigen, 0.)
        projected = vectors.T@centered
        for regularization in (.01, .1, 1.):
            denominator = eigen+len(A)*regularization
            coefficients = vectors@(projected/denominator[:, None] if residual.ndim == 2 else projected/denominator)
            correction = query@coefficients+mean
            if supported_classes is not None:
                correction[:, ~supported_classes] = 0.
            assert np.isfinite(correction).all()
            predictions.append(correction)
    return np.stack(predictions), {'source_rows': len(y), 'scored_source_rows': int(valid.sum()),
        'source_feature_encoding_columns': A.shape[1], 'numeric_clipping': 5.,
        'source_only_median_IQR_scaling': True, 'categorical_encoding': 'source-fitted one-hot; unknown query categories ignored',
        'source_OOF_residual_target': True, 'task_residual_kernel_ridge_fitted': True,
        'mean_residual_intercept_fitted_from_source_only': True,
        'kernel_regularization_added_to_Gram': 'scored_source_rows times regularization',
        'rbf_bandwidth_source_median_squared_distance': bandwidth,
        'regression_label_scale': 'source std only', 'no_query_outcome_labels': True,
        'query_features_not_used_for_fitting_encoder_or_kernel_hyperparameters': True}
