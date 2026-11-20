"""Observed-training-label conditional local feature smoothing.

This is an explicitly supervised constructor adaptation, not original Shared.
Only source labels form neighborhoods; no query labels, clean source features,
corruption identifiers or severities are accepted. Source labels stay intact.
It assumes reasonably reliable source labels and may fail under label noise.
"""
import numpy as np
from scipy.spatial.distance import cdist


def smooth(X_source, y_source, numeric_columns, task):
    X = np.asarray(X_source, dtype=np.float32)
    y = np.asarray(y_source)
    numeric = np.asarray(numeric_columns, dtype=int)
    categories = np.asarray(sorted(set(range(X.shape[1]))-set(numeric)), dtype=int)
    distances = np.zeros((len(X), len(X)), dtype=np.float64)
    if len(numeric):
        A = np.asarray(X[:, numeric], dtype=np.float64)
        center = np.median(A, axis=0)
        iqr = np.percentile(A, 75, axis=0)-np.percentile(A, 25, axis=0)
        sd = A.std(axis=0)
        scale = np.where(iqr > 1e-8, iqr, np.maximum(sd, 1e-8))
        Z = np.clip((A-center)/scale, -8., 8.)
        if Z.shape[1] > 64:
            projection = np.random.default_rng(0).normal(size=(Z.shape[1], 32))/np.sqrt(32.)
            Z = Z@projection
        distances += cdist(Z, Z, 'sqeuclidean')/max(len(numeric), 1)
    if len(categories):
        for j in categories:
            distances += (X[:, j, None] != X[None, :, j])/len(categories)
    if task == 'classification':
        groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    else:
        groups = [g for g in np.array_split(np.argsort(y, kind='stable'), 8) if len(g)]
    output = X.copy()
    donor_sets = []
    for group in groups:
        for i in group:
            other = group[group != i]
            nearest = other[np.argsort(distances[i, other], kind='stable')[:4]]
            donors = np.r_[int(i), nearest].astype(int)
            donor_sets.append((int(i), donors.tolist()))
            if len(numeric):
                median = np.median(X[donors][:, numeric], axis=0)
                output[i, numeric] = .5*X[i, numeric]+.5*median
            for j in categories:
                values, counts = np.unique(X[donors, j], return_counts=True)
                k = int(counts.argmax())
                if counts[k]/len(donors) >= .8:
                    output[i, j] = values[k]
    assert np.isfinite(output).all()
    return output, y.copy(), {'uses_observed_source_labels': True,
        'uses_query_labels_or_clean_source_features': False,
        'query_or_mechanism_identity_arguments': False, 'numeric_smoothing_fraction': .5,
        'maximum_conditional_neighbors_including_self': 5,
        'categorical_mode_threshold': .8, 'source_label_values_modified': False,
        'conditional_groups': 'same class' if task == 'classification' else '8 response rank strata',
        'source_label_reliability_assumption': True,
        'donor_sets': sorted(donor_sets)}
