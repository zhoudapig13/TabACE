"""Two query-cluster local contexts selected from observed source rows only.

Geometry is fitted to unlabeled query features only. Source labels ensure class
or response-range coverage; optional source-OOF loss supplies a fixed quality
penalty. Source features and labels are selected, never modified or synthesized.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


def encode(A, model):
    parts = []
    if len(model['numeric']):
        parts.append(np.clip((A[:, model['numeric']]-model['center'])/model['scale'], -5., 5.))
    if len(model['categorical']):
        parts.append(model['encoder'].transform(A[:, model['categorical']]))
    result = np.concatenate(parts, axis=1).astype(np.float64)
    assert np.isfinite(result).all()
    return result


def fit_target(X_target, numeric_columns):
    Q = np.asarray(X_target, dtype=np.float64)
    assert len(Q) and Q.shape[1] and np.isfinite(Q).all()
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = np.setdiff1d(np.arange(Q.shape[1]), numeric)
    center = np.median(Q[:, numeric], axis=0)
    scale = np.percentile(Q[:, numeric], 75, axis=0)-np.percentile(Q[:, numeric], 25, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(Q[:, numeric].std(0), 1e-6))
    model = {'numeric': numeric, 'categorical': categorical, 'center': center, 'scale': scale}
    if len(categorical):
        model['encoder'] = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64).fit(Q[:, categorical])
    design = encode(Q, model)
    pca = None
    if design.shape[1] > 32 and len(Q) > 16 and np.any(design.std(0) > 1e-8):
        pca = PCA(n_components=min(16, len(Q)-1, design.shape[1]-1),
            svd_solver='randomized', iterated_power=3, random_state=0)
        features = pca.fit_transform(design)
        residual = ((design-pca.inverse_transform(features))**2).mean(1)
        residual_reference = max(float(np.median(residual)), .1)
    else:
        features = design
        residual_reference = .1
    count = min(8, len(np.unique(features, axis=0)))
    if count == 1:
        assignment = np.zeros(len(Q), dtype=int)
        centers = features.mean(0)[None]
    else:
        clustering = KMeans(n_clusters=count, n_init=5, random_state=0).fit(features)
        assignment, centers = clustering.labels_, clustering.cluster_centers_
    model.update(pca=pca, query_assignment=assignment, cluster_centers=centers,
        residual_reference=residual_reference, geometry_scale=max(float(np.var(features, axis=0).mean()), 1e-6))
    return model


def select(y_source, score, requested, task):
    y = np.asarray(y_source)
    required = []
    if task == 'classification':
        for label in np.unique(y):
            group = np.flatnonzero(y == label)
            required.extend(group.tolist() if len(group) < 3 else [int(group[np.argmin(score[group])])])
    else:
        for group in np.array_split(np.argsort(y, kind='stable'), min(8, len(y))):
            required.append(int(group[np.argmin(score[group])]))
    required = np.unique(required)
    count = min(len(y), max(requested, len(required)))
    order = np.argsort(score, kind='stable')
    extra = order[~np.isin(order, required)][:count-len(required)]
    return np.sort(np.r_[required, extra])


def construct(X_source, y_source, source_cv_loss, task, target_model):
    X, y = np.asarray(X_source), np.asarray(y_source)
    loss = np.asarray(source_cv_loss)
    assert len(X) == len(y) == len(loss) and len(y) <= 256
    assert np.isfinite(loss).all() and task in ('classification', 'regression')
    encoded = encode(X, target_model)
    pca = target_model['pca']
    if pca is None:
        features, off_manifold = encoded, np.zeros(len(X))
    else:
        features = pca.transform(encoded)
        residual = ((encoded-pca.inverse_transform(features))**2).mean(1)
        reference = target_model['residual_reference']
        off_manifold = np.clip((residual-reference)/reference, 0., 10.)
    centers, assignment = target_model['cluster_centers'], target_model['query_assignment']
    result = []
    for requested, quality_penalty in ((64, 0.), (128, .25)):
        branch = []
        for cluster, center in enumerate(centers):
            query_rows = np.flatnonzero(assignment == cluster)
            if not len(query_rows):
                continue
            distance = ((features-center)**2).mean(1)/target_model['geometry_scale']
            score = distance+.25*off_manifold+quality_penalty*np.clip(loss, 0., 5.)
            source_rows = select(y, score, requested, task)
            assert len(source_rows) <= 256 and len(np.unique(source_rows)) == len(source_rows)
            if task == 'classification':
                assert set(np.unique(y[source_rows])) == set(np.unique(y))
            branch.append({'source_rows': source_rows, 'query_rows': query_rows,
                'requested_rows': requested, 'quality_penalty': quality_penalty, 'cluster': cluster})
        covered = np.concatenate([p['query_rows'] for p in branch])
        np.testing.assert_array_equal(np.sort(covered), np.arange(len(assignment)))
        result.append(branch)
    return result
