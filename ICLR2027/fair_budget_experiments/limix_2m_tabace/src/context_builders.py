"""Label-safe per-query Context constructors for LimiX + TabACE.

Every constructor consumes labeled source data, unlabeled query features, and
optionally frozen Base predictions.  Evaluation labels are intentionally absent
from the API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


EPS = 1.0e-12


def standardized_features(
    X_source: np.ndarray, X_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    source = scaler.fit_transform(imputer.fit_transform(np.asarray(X_source)))
    query = scaler.transform(imputer.transform(np.asarray(X_query)))
    return np.asarray(source, dtype=np.float32), np.asarray(query, dtype=np.float32)


def _knn(source: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    count = min(max(2, int(k)), len(source))
    neighbors = NearestNeighbors(n_neighbors=count, metric="euclidean")
    neighbors.fit(source)
    return neighbors.kneighbors(query, return_distance=False).astype(np.int64)


def _unique_fill(primary: Sequence[int], fill: np.ndarray, k: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for raw in [*primary, *np.asarray(fill, dtype=int).tolist()]:
        index = int(raw)
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) >= k:
            break
    return np.asarray(selected, dtype=np.int64)


def _balanced_contexts(
    source: np.ndarray, y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    classes = np.unique(y)
    per_class = max(1, int(math.ceil(k / max(1, len(classes)))))
    rows = []
    fallback = _knn(source, query, k)
    for x, fill in zip(query, fallback):
        ranked: list[tuple[float, int]] = []
        for label in classes:
            indices = np.flatnonzero(y == label)
            distance = np.linalg.norm(source[indices] - x, axis=1)
            order = np.argsort(distance)[: min(per_class, len(indices))]
            ranked.extend((float(distance[pos]), int(indices[pos])) for pos in order)
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        rows.append(_unique_fill([index for _, index in ranked], fill, k))
    return np.stack(rows)


def _prototype_contexts(
    source: np.ndarray, y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    classes = np.unique(y)
    per_class = max(1, int(math.ceil(k / max(1, len(classes)))))
    candidates: dict[object, np.ndarray] = {}
    for label in classes:
        indices = np.flatnonzero(y == label)
        center = np.mean(source[indices], axis=0)
        distance = np.linalg.norm(source[indices] - center, axis=1)
        candidates[label] = indices[
            np.argsort(distance)[: min(max(per_class * 3, per_class), len(indices))]
        ]
    fallback = _knn(source, query, k)
    rows = []
    for x, fill in zip(query, fallback):
        ranked: list[tuple[float, int]] = []
        for label in classes:
            indices = candidates[label]
            distance = np.linalg.norm(source[indices] - x, axis=1)
            order = np.argsort(distance)[: min(per_class, len(indices))]
            ranked.extend((float(distance[pos]), int(indices[pos])) for pos in order)
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        rows.append(_unique_fill([index for _, index in ranked], fill, k))
    return np.stack(rows)


def _clean_contexts(
    source: np.ndarray, y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    train_k = min(max(2, 13), len(y))
    neighbors = NearestNeighbors(n_neighbors=train_k, metric="euclidean").fit(source)
    train_neighbors = neighbors.kneighbors(source, return_distance=False)
    agreement = np.zeros(len(y), dtype=np.float64)
    for index, row in enumerate(train_neighbors):
        row = row[row != index]
        agreement[index] = 1.0 if not len(row) else float(np.mean(y[row] == y[index]))
    clean = agreement >= 0.65
    clean_indices = np.flatnonzero(clean)
    fallback = _knn(source, query, k)
    if len(clean_indices) >= min(k, len(y)):
        count = min(k, len(clean_indices))
        clean_neighbors = NearestNeighbors(n_neighbors=count).fit(source[clean_indices])
        local = clean_neighbors.kneighbors(query, return_distance=False)
        return clean_indices[local].astype(np.int64)
    rows = []
    wide = _knn(source, query, min(len(source), max(5 * k, k)))
    for candidates, fill in zip(wide, fallback):
        rows.append(_unique_fill([i for i in candidates if clean[int(i)]], fill, k))
    return np.stack(rows)


def _boundary_contexts(
    source: np.ndarray,
    y: np.ndarray,
    query: np.ndarray,
    base_probability: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> np.ndarray:
    class_to_position = {label: index for index, label in enumerate(classes.tolist())}
    encoded = np.asarray([class_to_position[label] for label in y], dtype=int)
    fallback = _knn(source, query, k)
    rows = []
    per_side = max(1, int(math.ceil(k / 2)))
    for probability, x, fill in zip(base_probability, query, fallback):
        order = np.argsort(probability)[::-1]
        selected: list[int] = []
        for class_position in order[:2]:
            indices = np.flatnonzero(encoded == int(class_position))
            if not len(indices):
                continue
            distance = np.linalg.norm(source[indices] - x, axis=1)
            selected.extend(
                indices[np.argsort(distance)[: min(per_side, len(indices))]].tolist()
            )
        rows.append(_unique_fill(selected, fill, k))
    return np.stack(rows)


def _minority_contexts(
    source: np.ndarray,
    y: np.ndarray,
    query: np.ndarray,
    base_probability: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> np.ndarray:
    class_to_position = {label: index for index, label in enumerate(classes.tolist())}
    encoded = np.asarray([class_to_position[label] for label in y], dtype=int)
    positions, counts = np.unique(encoded, return_counts=True)
    minority = int(positions[np.argmin(counts)])
    fallback = _knn(source, query, k)
    rows = []
    for probability, x, fill in zip(base_probability, query, fallback):
        selected: list[int] = []
        for position, count in (
            (minority, max(2, k // 2)),
            (int(np.argmax(probability)), max(1, k // 4)),
        ):
            indices = np.flatnonzero(encoded == position)
            if not len(indices):
                continue
            distance = np.linalg.norm(source[indices] - x, axis=1)
            selected.extend(indices[np.argsort(distance)[: min(count, len(indices))]].tolist())
        rows.append(_unique_fill(selected, fill, k))
    return np.stack(rows)


def classification_contexts(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_query: np.ndarray,
    base_probability: np.ndarray,
    classes: np.ndarray,
    specs: Mapping[str, tuple[str, int]],
) -> dict[str, np.ndarray]:
    """Build a named classification Context bank from constructor/k specs."""

    source, query = standardized_features(X_source, X_query)
    y = np.asarray(y_source)
    output: dict[str, np.ndarray] = {}
    for name, (constructor, raw_k) in specs.items():
        k = min(int(raw_k), len(y))
        if constructor == "knn":
            value = _knn(source, query, k)
        elif constructor == "balanced":
            value = _balanced_contexts(source, y, query, k)
        elif constructor == "prototype":
            value = _prototype_contexts(source, y, query, k)
        elif constructor == "clean":
            value = _clean_contexts(source, y, query, k)
        elif constructor == "boundary":
            value = _boundary_contexts(
                source, y, query, base_probability, np.asarray(classes), k
            )
        elif constructor == "minority":
            value = _minority_contexts(
                source, y, query, base_probability, np.asarray(classes), k
            )
        else:
            raise ValueError(f"unknown classification Context constructor: {constructor}")
        output[str(name)] = np.asarray(value, dtype=np.int64)
    return output


def _regression_clean_contexts(
    source: np.ndarray, y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    fallback = _knn(source, query, k)
    wide = _knn(source, query, min(len(source), max(2 * k, k + 8)))
    global_scale = max(float(np.median(np.abs(y - np.median(y))) * 1.4826), EPS)
    rows = []
    for candidates, fill in zip(wide, fallback):
        local = y[candidates]
        median = float(np.median(local))
        scale = max(float(np.median(np.abs(local - median)) * 1.4826), global_scale)
        clean = candidates[np.abs(y[candidates] - median) <= 2.5 * scale]
        rows.append(_unique_fill(clean.tolist(), fill, k))
    return np.stack(rows)


def _regression_bracket_contexts(
    source: np.ndarray,
    y: np.ndarray,
    query: np.ndarray,
    base_prediction: np.ndarray,
    k: int,
) -> np.ndarray:
    fallback = _knn(source, query, k)
    rows = []
    for prediction, fill in zip(base_prediction, fallback):
        below = np.flatnonzero(y <= prediction)
        above = np.flatnonzero(y > prediction)
        half = max(1, k // 2)
        below = below[np.argsort(np.abs(y[below] - prediction))[:half]]
        above = above[np.argsort(np.abs(y[above] - prediction))[: k - len(below)]]
        rows.append(_unique_fill(np.concatenate([below, above]).tolist(), fill, k))
    return np.stack(rows)


def _regression_prototype_contexts(
    source: np.ndarray, y: np.ndarray, query: np.ndarray, k: int, bins: int = 8
) -> np.ndarray:
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return _knn(source, query, k)
    labels = np.clip(np.digitize(y, edges[1:-1], right=True), 0, len(edges) - 2)
    return _balanced_contexts(source, labels, query, k)


def _regression_trimmed_contexts(
    source: np.ndarray, y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    low, high = np.quantile(y, [0.02, 0.98])
    keep = np.flatnonzero((y >= low) & (y <= high))
    if len(keep) < min(k, len(y)):
        return _knn(source, query, k)
    count = min(k, len(keep))
    neighbors = NearestNeighbors(n_neighbors=count).fit(source[keep])
    return keep[neighbors.kneighbors(query, return_distance=False)].astype(np.int64)


def regression_contexts(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_query: np.ndarray,
    base_prediction: np.ndarray,
    specs: Mapping[str, tuple[str, int]],
) -> dict[str, np.ndarray]:
    """Build a named regression Context bank from constructor/k specs."""

    source, query = standardized_features(X_source, X_query)
    y = np.asarray(y_source, dtype=np.float64)
    output: dict[str, np.ndarray] = {}
    for name, (constructor, raw_k) in specs.items():
        k = min(int(raw_k), len(y))
        if constructor == "knn":
            value = _knn(source, query, k)
        elif constructor == "clean":
            value = _regression_clean_contexts(source, y, query, k)
        elif constructor == "bracket":
            value = _regression_bracket_contexts(
                source, y, query, np.asarray(base_prediction), k
            )
        elif constructor == "prototype":
            value = _regression_prototype_contexts(source, y, query, k)
        elif constructor == "trimmed":
            value = _regression_trimmed_contexts(source, y, query, k)
        else:
            raise ValueError(f"unknown regression Context constructor: {constructor}")
        output[str(name)] = np.asarray(value, dtype=np.int64)
    return output


def regression_global_contexts(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_query: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build batch-level regression Contexts for efficient LimiX inference.

    These are valid TabACE constructors: they select labeled reference rows from
    source statistics and, for the target-near views, the unlabeled target batch.
    They avoid repeating a long regression Context independently for every query.
    """

    source, query = standardized_features(X_source, X_query)
    y = np.asarray(y_source, dtype=np.float64)
    n = len(y)

    def take(order: np.ndarray, fraction: float) -> np.ndarray:
        count = min(n, max(32, int(round(fraction * n))))
        return np.sort(np.asarray(order[:count], dtype=np.int64))

    target_center = np.mean(query, axis=0)
    target_distance = np.linalg.norm(source - target_center, axis=1)
    feature_center = np.median(source, axis=0)
    feature_distance = np.linalg.norm(source - feature_center, axis=1)
    y_center = float(np.median(y))
    y_distance = np.abs(y - y_center)

    neighbor_count = min(max(3, int(round(math.sqrt(n)))), n)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(source)
    local_indices = neighbors.kneighbors(source, return_distance=False)
    local_median = np.median(y[local_indices], axis=1)
    local_residual = np.abs(y - local_median)

    quantile_count = min(16, max(4, int(round(math.sqrt(n) / 2))))
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, quantile_count + 1)))
    bin_ids = np.clip(np.digitize(y, edges[1:-1], right=True), 0, max(len(edges) - 2, 0))
    quota = max(2, int(math.ceil(0.75 * n / max(1, len(np.unique(bin_ids))))))
    balanced: list[int] = []
    for bin_id in np.unique(bin_ids):
        indices = np.flatnonzero(bin_ids == bin_id)
        order = indices[np.argsort(target_distance[indices], kind="stable")]
        balanced.extend(order[: min(quota, len(order))].tolist())
    balanced_array = np.asarray(sorted(set(balanced)), dtype=np.int64)
    if len(balanced_array) < min(32, n):
        balanced_array = take(np.argsort(target_distance, kind="stable"), 0.75)

    return {
        "target_near/q50": take(np.argsort(target_distance, kind="stable"), 0.50),
        "target_near/q75": take(np.argsort(target_distance, kind="stable"), 0.75),
        "feature_central/q75": take(np.argsort(feature_distance, kind="stable"), 0.75),
        "label_central/q75": take(np.argsort(y_distance, kind="stable"), 0.75),
        "local_clean/q75": take(np.argsort(local_residual, kind="stable"), 0.75),
        "quantile_balanced/q75": balanced_array,
    }
