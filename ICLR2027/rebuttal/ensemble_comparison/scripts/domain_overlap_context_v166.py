"""Cross-fitted nonlinear domain overlap using features and domain IDs only.

Domain IDs distinguish observed source rows from unlabeled query features;
they are not task outcomes. This is a new transductive constructor adaptation.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def encode(X_source, X_target, numeric_columns):
    X = np.concatenate([X_source, X_target]).astype(np.float64)
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = np.setdiff1d(np.arange(X.shape[1]), numeric)
    parts = []
    if len(numeric):
        values = X[:, numeric]
        center = np.median(values, axis=0)
        scale = np.percentile(values, 75, axis=0)-np.percentile(values, 25, axis=0)
        scale = np.maximum(scale, np.maximum(values.std(axis=0)*.1, 1e-8))
        parts.append(np.clip((values-center)/scale, -10., 10.))
    if len(categorical):
        parts.append(OneHotEncoder(handle_unknown='ignore', sparse_output=False,
            dtype=np.float32).fit_transform(X[:, categorical]))
    result = np.concatenate(parts, axis=1).astype(np.float32)
    assert np.isfinite(result).all()
    return result


def quality(X_source, X_target, numeric_columns):
    """Return source-domain probabilities for held-out source and target rows.

    Lower source probability indicates greater overlap with target features.
    No source-task labels or query-task outcomes are accepted by this API.
    """
    assert len(X_source) >= 3 and len(X_target) >= 3
    design = encode(X_source, X_target, numeric_columns)
    domain = np.r_[np.ones(len(X_source), dtype=int), np.zeros(len(X_target), dtype=int)]
    probability = np.full(len(domain), np.nan)
    partitions = []
    for fold, (train, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=0).split(design, domain)):
        model = ExtraTreesClassifier(n_estimators=96, max_depth=8, min_samples_leaf=4,
            max_features=1., class_weight='balanced', n_jobs=1, random_state=fold)
        model.fit(design[train], domain[train])
        probability[valid] = model.predict_proba(design[valid])[:, 1]
        assert not set(train) & set(valid)
        partitions.append({'training_rows': train.tolist(), 'validation_rows': valid.tolist()})
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    return probability[:len(X_source)], probability[len(X_source):], partitions
