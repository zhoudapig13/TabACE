"""Source-label-guided, transductive context repair using Base pseudo-labels.

This is explicitly a new Shared adaptation, not an original constructor.
The target batch is unlabeled. Source losses must be source-cross-fitted.
"""
import numpy as np


def repair(X_source, y_source, X_target, base_prediction, numeric_columns,
           source_cv_loss, task):
    X, Q = np.asarray(X_source, dtype=np.float32), np.asarray(X_target, dtype=np.float32)
    y = np.asarray(y_source)
    P = np.asarray(base_prediction, dtype=np.float64)
    loss = np.asarray(source_cv_loss, dtype=np.float64)
    numeric = np.asarray(numeric_columns, dtype=int)
    categories = np.asarray(sorted(set(range(X.shape[1]))-set(numeric)), dtype=int)
    assert len(X) == len(y) == len(loss) and len(Q) > 0
    assert np.isfinite(loss).all() and np.isfinite(P).all()
    d = np.zeros((len(X), len(Q)), dtype=np.float64)
    if len(numeric):
        B = np.asarray(Q[:, numeric], dtype=np.float64)
        center = np.median(B, axis=0)
        scale = np.percentile(B, 75, axis=0)-np.percentile(B, 25, axis=0)
        scale = np.where(scale > 1e-8, scale, np.maximum(B.std(axis=0), 1e-6))
        A = np.clip((X[:, numeric]-center)/scale, -5., 5.)
        B = np.clip((B-center)/scale, -5., 5.)
        if len(numeric) > 64:
            projection = np.random.default_rng(0).normal(size=(len(numeric), 32))/np.sqrt(32.)
            A, B = A@projection, B@projection
        d += np.maximum((A*A).sum(axis=1)[:, None]+(B*B).sum(axis=1)[None]-2*A@B.T, 0.)
    for column in categories:
        d += X[:, column, None] != Q[None, :, column]
    d /= max(X.shape[1], 1)
    k = min(10, len(Q))
    nearest = np.partition(d, k-1, axis=1)[:, :k]
    bandwidth = np.median(nearest, axis=1)
    positive = d[d > 1e-12]
    fallback = max(float(np.median(positive)) if len(positive) else 1., 1e-8)
    bandwidth = np.where(bandwidth > 1e-8, bandwidth, fallback)
    if task == 'classification':
        assert P.ndim == 2 and P.shape[0] == len(Q)
        assert (P >= 0).all()
        np.testing.assert_allclose(P.sum(axis=1), 1., atol=1e-6)
        compatibility = P[:, y.astype(int)].T
        groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    else:
        assert P.shape == (len(Q),)
        label_bandwidth = max(.5*float(y.std()), 1e-8)
        compatibility = np.exp(np.clip(-.5*((P[None]-y[:, None])/label_bandwidth)**2, -700., 0.))
        groups = [g for g in np.array_split(np.argsort(y, kind='stable'), 8) if len(g)]
    log_weight = -d/bandwidth[:, None]+np.log(np.maximum(compatibility, 1e-12))
    ids = np.argsort(-log_weight, axis=1, kind='stable')[:, :k]
    weights = np.take_along_axis(log_weight, ids, axis=1)
    weights = np.exp(weights-weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    reliability = np.zeros(len(y), dtype=np.float64)
    for group in groups:
        values = loss[group]
        valid = values >= 0
        if valid.sum() < 3:
            continue
        low, high = np.quantile(values[valid], [.5, .9])
        if high > low+1e-12:
            reliability[group] = np.where(valid, np.clip((values-low)/(high-low), 0., 1.), 0.)
    compatibility = (np.take_along_axis(compatibility, ids, axis=1)*weights).sum(axis=1)
    if task == 'classification':
        chance = 1./P.shape[1]
        compatibility = np.clip((compatibility-chance)/max(1-chance, 1e-8), 0., 1.)
    reliability *= compatibility
    barycenter = X.copy()
    if len(numeric):
        # A weighted median, rather than interpolating source categorical codes.
        for i in range(len(y)):
            A = np.asarray(Q[ids[i]][:, numeric], dtype=np.float64)
            order = np.argsort(A, axis=0, kind='stable')
            values = np.take_along_axis(A, order, axis=0)
            w = weights[i][order]
            median_rows = (np.cumsum(w, axis=0) < .5).sum(axis=0)
            barycenter[i, numeric] = values[np.minimum(median_rows, k-1), np.arange(len(numeric))]
    category_modes = X.copy()
    category_confidence = np.zeros_like(X, dtype=np.float64)
    for i in range(len(y)):
        for column in categories:
            values = Q[ids[i], column]
            unique, inverse = np.unique(values, return_inverse=True)
            probabilities = np.bincount(inverse, weights=weights[i], minlength=len(unique))
            best = int(probabilities.argmax())
            category_modes[i, column] = unique[best]
            category_confidence[i, column] = probabilities[best]
    outputs, details = [], []
    for strength in (.5, .9):
        alpha = strength*reliability
        output = X.copy()
        if len(numeric):
            output[:, numeric] = X[:, numeric]+alpha[:, None]*(barycenter[:, numeric]-X[:, numeric])
        for column in categories:
            changed = (alpha >= .25) & (category_confidence[:, column] >= .9)
            output[changed, column] = category_modes[changed, column]
        assert np.isfinite(output).all()
        outputs.append(output)
        details.append({'maximum_row_blend': strength, 'changed_cells': int(np.sum(output != X)),
            'modified_source_rows': int(np.any(output != X, axis=1).sum()),
            'mean_row_blend': float(alpha.mean()), 'maximum_actual_row_blend': float(alpha.max())})
    return outputs, y.copy(), {'source_cv_loss_required': True,
        'uses_source_labels': True, 'uses_unlabeled_target_features': True,
        'uses_Base_predictions_as_target_pseudo_labels': True,
        'uses_true_target_labels_or_clean_source_features': False,
        'target_neighborhood_size': k, 'numeric_neighbor_summary': 'weighted median',
        'feature_distance_clip': 5., 'source_group_loss_quantiles': [.5, .9],
        'categorical_mode_probability_threshold': .9, 'categorical_row_blend_threshold': .25,
        'source_label_values_unchanged': True, 'branch_details': details}
