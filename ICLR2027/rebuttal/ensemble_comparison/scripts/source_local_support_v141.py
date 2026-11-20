"""Query-local support from observed source labels and source-OOF residuals.

No query labels, mechanism names, or severity values enter this function.
These are explicit new fusion features, not original Shared signals.
"""
import numpy as np


def features(X_source, y_source, X_target, numeric_columns,
             source_cv_prediction, source_cv_loss, task):
    X = np.asarray(X_source, dtype=np.float64)
    Q = np.asarray(X_target, dtype=np.float64)
    y = np.asarray(y_source)
    cv = np.asarray(source_cv_prediction, dtype=np.float64)
    loss = np.asarray(source_cv_loss, dtype=np.float64)
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = sorted(set(range(X.shape[1]))-set(numeric))
    assert len(X) == len(y) == len(loss) == len(cv)
    assert np.isfinite(X).all() and np.isfinite(Q).all() and np.isfinite(loss).all()
    distance = np.zeros((len(Q), len(X)), dtype=np.float64)
    if len(numeric):
        target = Q[:, numeric]
        center = np.median(target, axis=0)
        spread = np.percentile(target, 75, axis=0)-np.percentile(target, 25, axis=0)
        spread = np.where(spread > 1e-8, spread, np.maximum(target.std(axis=0), 1e-6))
        A = np.clip((X[:, numeric]-center)/spread, -5., 5.)
        B = np.clip((target-center)/spread, -5., 5.)
        if len(numeric) > 64:
            projection = np.random.default_rng(0).normal(size=(len(numeric), 32))/np.sqrt(32.)
            A, B = A@projection, B@projection
        distance += np.maximum((B*B).sum(axis=1)[:, None]+(A*A).sum(axis=1)[None]-2*B@A.T, 0.)
    for column in categorical:
        distance += Q[:, column, None] != X[None, :, column]
    distance /= max(1, X.shape[1])
    order = np.argsort(distance, axis=1, kind='stable')
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    valid = (np.isfinite(cv).all(axis=1) if cv.ndim == 2 else np.isfinite(cv)) & (loss >= 0)
    quality = np.exp(-np.clip(np.maximum(loss, 0.), 0., 5.))
    if task == 'classification':
        K = cv.shape[1]
        labels = np.eye(K)[y.astype(int)]
        residual = np.where(valid[:, None], labels-np.nan_to_num(cv), 0.)
        columns = []
    else:
        normalized = (y-y.mean())/max(float(y.std()), 1e-12)
        residual = np.where(valid, (y-np.nan_to_num(cv))/max(float(y.std()), 1e-12), 0.)
        columns = []
    for requested in (8, 32, 128):
        k = min(requested, len(y))
        ids, d = order[:, :k], sorted_distance[:, :k]
        h = np.median(d, axis=1)
        positive = distance[distance > 1e-12]
        fallback = max(float(np.median(positive)) if len(positive) else 1., 1e-8)
        h = np.where(h > 1e-8, h, fallback)
        geometric = np.exp(-d/h[:, None])
        geometric /= geometric.sum(axis=1, keepdims=True)
        weighted = geometric*quality[ids]
        weighted /= weighted.sum(axis=1, keepdims=True)
        statistics = [d[:, 0], (d*geometric).sum(axis=1), h,
            1./(geometric**2).sum(axis=1), 1./(weighted**2).sum(axis=1),
            (np.maximum(loss[ids], 0.)*geometric).sum(axis=1),
            (np.maximum(loss[ids], 0.)*weighted).sum(axis=1),
            (valid[ids]*geometric).sum(axis=1)]
        local = []
        for weights in (geometric, weighted):
            scored = weights*valid[ids]
            denominator = np.maximum(scored.sum(axis=1), 1e-12)
            if task == 'classification':
                vote = np.einsum('nk,nkc->nc', weights, labels[ids])
                correction = np.einsum('nk,nkc->nc', scored, residual[ids])/denominator[:, None]
                local.extend([vote, correction])
            else:
                mean = (weights*normalized[ids]).sum(axis=1)
                sd = np.sqrt(np.maximum((weights*(normalized[ids]-mean[:, None])**2).sum(axis=1), 0.))
                correction = (scored*residual[ids]).sum(axis=1)/denominator
                correction_sd = np.sqrt(np.maximum((scored*(residual[ids]-correction[:, None])**2).sum(axis=1)/denominator, 0.))
                local.extend([mean, sd, correction, correction_sd])
        if task == 'classification':
            fields = np.stack(local, axis=2)
            shared = np.column_stack(statistics)
            columns.append(np.concatenate([fields,
                np.broadcast_to(shared[:, None], (len(Q), K, len(statistics)))], axis=2))
        else:
            columns.append(np.column_stack(local+statistics))
    dataset = [np.log1p(len(y)), np.log1p(X.shape[1]), len(numeric)/max(X.shape[1], 1), float(valid.mean())]
    if task == 'classification':
        prior = labels.mean(axis=0)
        result = np.concatenate(columns+[
            np.broadcast_to(prior[None, :, None], (len(Q), K, 1)),
            np.broadcast_to(np.asarray(dataset)[None, None], (len(Q), K, len(dataset)))], axis=2)
    else:
        result = np.column_stack(columns+[np.broadcast_to(dataset, (len(Q), len(dataset)))])
    assert np.isfinite(result).all()
    return result, {'neighbor_counts_requested': [8, 32, 128],
        'numeric_distance_clip': 5., 'numeric_geometry_projection_above64': 32,
        'observed_source_labels_used': True, 'source_crossfit_residuals_used': True,
        'unscored_rare_source_rows_never_get_residuals': True,
        'query_labels_or_generator_or_strength_used': False,
        'source_label_standardization': 'source mean/std only',
        'features_per_query_class_or_regression_row': int(result.shape[-1])}
