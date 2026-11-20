"""Two source-supervised latent feature contexts, explicitly a Shared extension.

Only existing observed source labels are fitted. Query features may define an
unlabeled marginal-rank transformation; no query outcome or generator ID is
accepted. Rows and source labels remain unchanged; both source and query
features are projected into the same source-fitted latent coordinate system.
"""
import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression

from domain_overlap_context_v166 import encode


def design(X, Q, numeric_columns, rank_marginals):
    X, Q = np.asarray(X, dtype=np.float64).copy(), np.asarray(Q, dtype=np.float64).copy()
    if rank_marginals:
        for values in (X, Q):
            for j in numeric_columns:
                probabilities = (rankdata(values[:, j], method='average')-.5)/len(values)
                values[:, j] = ndtri(np.clip(probabilities, .01, .99))
    return encode(X, Q, numeric_columns).astype(np.float64)


def construct(X_source, y_source, X_target, numeric_columns, task):
    X, y, Q = np.asarray(X_source), np.asarray(y_source), np.asarray(X_target)
    assert task in ('classification', 'regression')
    assert len(X) == len(y) and len(X) >= 2 and len(Q) >= 1
    assert np.isfinite(X).all() and np.isfinite(Q).all() and np.isfinite(y).all()
    if task == 'classification':
        _, labels = np.unique(y, return_inverse=True)
        response = np.eye(int(labels.max())+1)[labels]
    else:
        response = y[:, None].astype(np.float64)
    response -= response.mean(axis=0)
    response /= np.maximum(response.std(axis=0), 1e-8)
    contexts = []
    for rank in (False, True):
        encoded = design(X, Q, numeric_columns, rank)
        active = encoded[:len(X)].std(axis=0) > 1e-8
        matrix = encoded[:, active]
        components = min(4, matrix.shape[1], len(X)-1)
        fallback = None
        messages = []
        if components == 0:
            transformed = np.zeros((len(encoded), 1), dtype=np.float64)
            components, fallback = 1, 'all_source_features_constant'
        elif np.max(np.abs(response)) < 1e-12:
            center = matrix[:len(X)].mean(axis=0)
            _, _, axes = np.linalg.svd(matrix[:len(X)]-center, full_matrices=False)
            transformed = (matrix-center)@axes[:components].T
            fallback = 'constant_source_response_source_PCA'
        else:
            model = PLSRegression(n_components=components, scale=False, max_iter=500, tol=1e-6)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                model.fit(matrix[:len(X)], response)
            messages = sorted({str(w.message) for w in recorded})
            transformed = model.transform(matrix)
        transformed = transformed.astype(np.float32)
        assert np.isfinite(transformed).all()
        contexts.append({'X': transformed[:len(X)], 'y': y.copy(), 'Q': transformed[len(X):],
            'info': {'rule': 'marginal_rank_source_PLS' if rank else 'robust_scaled_source_PLS',
                'latent_dimensions': transformed.shape[1], 'active_encoded_dimensions': int(active.sum()),
                'fallback': fallback, 'fit_warnings': messages,
                'source_labels_modified': False, 'query_labels_used': False,
                'source_supervision_used': fallback is None,
                'unlabeled_query_features_used': True,
                'independent_source_target_marginal_rank_alignment': rank}})
    return contexts
