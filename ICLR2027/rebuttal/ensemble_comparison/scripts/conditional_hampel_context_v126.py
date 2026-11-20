"""Selective supervised source-feature clipping, not original Shared.

Uses observed source labels to define same-class/near-response groups, and
unlabeled query features for conservative numerical support bounds. Assumes
source labels are reasonably reliable. Never accepts query labels or clean
counterfactual source features. Ordinary in-range measurements remain intact.
"""
import numpy as np


def repair(X_source, y_source, X_target, numeric_columns, task):
    X = np.asarray(X_source, dtype=np.float32)
    Q = np.asarray(X_target, dtype=np.float32)
    y = np.asarray(y_source)
    numeric = np.asarray(numeric_columns, dtype=int)
    categories = np.asarray(sorted(set(range(X.shape[1]))-set(numeric)), dtype=int)
    output = X.copy()
    if len(numeric):
        T = np.asarray(Q[:, numeric], dtype=np.float64)
        iqr = np.percentile(T, 75, axis=0)-np.percentile(T, 25, axis=0)
        sd = T.std(axis=0)
        scale = np.where(iqr > 1e-8, iqr, np.maximum(sd, 1e-8))
        support_low, support_high = T.min(axis=0)-.5*scale, T.max(axis=0)+.5*scale
        output[:, numeric] = np.clip(output[:, numeric], support_low, support_high)
    if task == 'classification':
        groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    else:
        groups = [g for g in np.array_split(np.argsort(y, kind='stable'), 8) if len(g)]
    for group in groups:
        if len(group) < 8:
            continue
        if len(numeric):
            A = np.asarray(X[group][:, numeric], dtype=np.float64)
            center = np.median(A, axis=0)
            mad = np.median(np.abs(A-center), axis=0)
            spread = np.maximum(1.4826*mad, .05*scale)
            unique = np.array([len(np.unique(A[:, j])) for j in range(A.shape[1])])
            eligible = (mad > 1e-8) & (unique >= 4)
            # For zero-MAD or discrete columns, retain only the broad unlabeled
            # support clipping; do not collapse legitimate rare category-like values.
            if eligible.any():
                columns = numeric[eligible]
                low = np.maximum(center[eligible]-2.5*spread[eligible], support_low[eligible])
                high = np.minimum(center[eligible]+2.5*spread[eligible], support_high[eligible])
                crossed = low > high
                low[crossed], high[crossed] = support_low[eligible][crossed], support_high[eligible][crossed]
                output[np.ix_(group, columns)] = np.clip(X[np.ix_(group, columns)], low, high)
        for j in categories:
            values, counts = np.unique(X[group, j], return_counts=True)
            k = int(counts.argmax())
            if counts[k]/len(group) >= .9:
                output[group, j] = values[k]
    assert np.isfinite(output).all()
    return output, y.copy(), {'source_label_values_unchanged': True,
        'uses_source_labels': True, 'uses_unlabeled_target_feature_support': True,
        'uses_query_labels_or_clean_source_features': False,
        'minimum_conditional_group_size': 8, 'Hampel_scale_multiplier': 2.5,
        'MAD_normalization': 1.4826, 'target_support_margin': .5,
        'categorical_mode_threshold': .9,
        'changed_cells': int(np.sum(output != X)), 'total_cells': int(X.size),
        'conditional_groups': 'classes' if task == 'classification' else '8 response rank strata',
        'source_label_reliability_assumption': True}
