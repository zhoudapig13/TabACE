from __future__ import annotations

from collections.abc import Callable

import numpy as np


def identity(spec):
    """Return the dataset unchanged."""
    return spec


def row_norm_feature(spec):
    """Append a simple per-row L2 norm feature as a first innovation hook."""
    X = np.asarray(spec.X, dtype=float)
    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_augmented = np.hstack([X, row_norm])
    return type(spec)(name=f"{spec.name}__row_norm_feature", task=spec.task, X=X_augmented, y=spec.y)


METHOD_REGISTRY: dict[str, Callable] = {
    "identity": identity,
    "row_norm_feature": row_norm_feature,
}


def apply_method(spec, method_name: str):
    try:
        method = METHOD_REGISTRY[method_name]
    except KeyError as exc:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}") from exc
    return method(spec)
