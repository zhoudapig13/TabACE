from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _clone_spec(spec, name_suffix: str, X):
    return type(spec)(
        name=f"{spec.name}__{name_suffix}",
        task=spec.task,
        X=np.asarray(X),
        y=spec.y,
    )


def identity(spec):
    """Return the dataset unchanged."""
    return spec


def robust_scale(spec):
    """Apply median/IQR scaling fitted on the full candidate dataset."""
    from sklearn.preprocessing import RobustScaler

    X = np.asarray(spec.X, dtype=float)
    X_scaled = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(X)
    return _clone_spec(spec, "robust_scale", X_scaled)


def quantile_normal(spec):
    """Map each feature to an approximately normal marginal distribution."""
    from sklearn.preprocessing import QuantileTransformer

    X = np.asarray(spec.X, dtype=float)
    n_quantiles = max(10, min(200, X.shape[0]))
    X_scaled = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=0,
    ).fit_transform(X)
    return _clone_spec(spec, "quantile_normal", X_scaled)


def winsorized_features(spec):
    """Clip feature tails using robust per-feature quantile bounds."""
    X = np.asarray(spec.X, dtype=float)
    lo = np.nanquantile(X, 0.01, axis=0)
    hi = np.nanquantile(X, 0.99, axis=0)
    X_clipped = np.clip(X, lo, hi)
    return _clone_spec(spec, "winsorized_features", X_clipped)


def row_stats_feature(spec):
    """Append compact row-level statistics as a lightweight baseline improvement hook."""
    X = np.asarray(spec.X, dtype=float)
    row_mean = np.nanmean(X, axis=1, keepdims=True)
    row_std = np.nanstd(X, axis=1, keepdims=True)
    row_abs_max = np.nanmax(np.abs(X), axis=1, keepdims=True)
    X_augmented = np.hstack([X, row_mean, row_std, row_abs_max])
    return _clone_spec(spec, "row_stats_feature", X_augmented)


def robust_row_stats(spec):
    """Combine robust scaling with row-level summary features."""
    return row_stats_feature(robust_scale(spec))


METHOD_REGISTRY: dict[str, Callable] = {
    "identity": identity,
    "robust_scale": robust_scale,
    "quantile_normal": quantile_normal,
    "winsorized_features": winsorized_features,
    "row_stats_feature": row_stats_feature,
    "robust_row_stats": robust_row_stats,
}


def apply_method(spec, method_name: str):
    try:
        method = METHOD_REGISTRY[method_name]
    except KeyError as exc:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}") from exc
    return method(spec)
