"""Fixed OpenML loading and five-family source-Context perturbations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


CC18_SUITE_ID = 99
CTR23_SUITE_ID = 353
CC18_SEED = 0
CC18_ROWS = 384
CC18_SOURCE_ROWS = 256
CC18_TARGET_ROWS = 128
CTR23_CAP_SEED = 42
CTR23_SOURCE_CAP = 2048
CTR23_TARGET_CAP = 512
MANIPULATION_SEED = 1729
STRENGTHS = (0, 20, 50, 80, 100)
FAMILIES = ("shortcut", "boundary", "band", "local", "mixed")


@dataclass(frozen=True)
class DatasetSpec:
    task_id: int
    data_id: int
    name: str
    target_name: str


@dataclass(frozen=True)
class FixedSplit:
    spec: DatasetSpec
    X_source: np.ndarray
    X_target: np.ndarray
    y_source: np.ndarray
    y_target: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    categorical_features: tuple[int, ...]


@dataclass(frozen=True)
class ManipulationResult:
    X_context: np.ndarray
    y_context: np.ndarray
    affected_rows: np.ndarray
    top_feature: int
    metadata: dict[str, Any]


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _numeric_matrix(X: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    """Match the numeric conversion used by the paper's OpenML utilities."""

    X_object = np.asarray(X, dtype=object)
    numeric = np.empty(X_object.shape, dtype=np.float32)
    categorical: list[int] = []
    for column in range(X_object.shape[1]):
        values = X_object[:, column]
        try:
            numeric[:, column] = values.astype(np.float32)
        except (TypeError, ValueError):
            categorical.append(column)
    if categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        numeric[:, categorical] = encoder.fit_transform(
            X_object[:, categorical]
        ).astype(np.float32)
    numeric = SimpleImputer(strategy="median").fit_transform(numeric)
    return np.asarray(numeric, dtype=np.float32), tuple(categorical)


def suite_specs(
    suite_id: int, requested: set[str] | None = None
) -> list[DatasetSpec]:
    import openml

    suite = openml.study.get_suite(suite_id)
    task_ids = list(suite.tasks)
    # A numeric selector is usually a data ID (as shown in result tables), but
    # CC18 task IDs often equal data IDs. Resolve these selectors without
    # downloading metadata for the whole suite; full runs still enumerate all
    # registered tasks.
    if requested and all(token.isdigit() for token in requested):
        resolved: set[int] = set()
        suite_task_ids = set(int(value) for value in task_ids)
        for token in requested:
            value = int(token)
            if value in suite_task_ids:
                resolved.add(value)
                continue
            listed = openml.tasks.list_tasks(
                data_id=value,
                output_format="dataframe",
            )
            listed_ids = listed.index.tolist()
            resolved.update(
                int(task_id)
                for task_id in listed_ids
                if int(task_id) in suite_task_ids
            )
        if resolved:
            task_ids = [task_id for task_id in task_ids if int(task_id) in resolved]
    output: list[DatasetSpec] = []
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(
            task.dataset_id,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        output.append(
            DatasetSpec(
                task_id=int(task_id),
                data_id=int(task.dataset_id),
                name=str(dataset.name),
                target_name=str(task.target_name),
            )
        )
    return output


def filter_specs(
    specs: Iterable[DatasetSpec], requested: set[str] | None
) -> list[DatasetSpec]:
    values = list(specs)
    if not requested:
        return values
    normalized = {item.casefold() for item in requested}
    selected = [
        spec
        for spec in values
        if str(spec.data_id) in normalized
        or str(spec.task_id) in normalized
        or spec.name.casefold() in normalized
    ]
    found = {
        token
        for spec in selected
        for token in (str(spec.data_id), str(spec.task_id), spec.name.casefold())
    }
    missing = sorted(normalized - found)
    if missing:
        raise ValueError(f"unknown suite datasets/tasks: {missing}")
    return selected


def _task_data(spec: DatasetSpec) -> tuple[Any, np.ndarray, tuple[int, ...], Any]:
    import openml

    task = openml.tasks.get_task(spec.task_id, download_data=False)
    dataset = task.get_dataset()
    X, y, categorical, _names = dataset.get_data(
        dataset_format="dataframe", target=spec.target_name
    )
    target = np.asarray(y)
    valid = np.asarray([value is not None for value in target], dtype=bool)
    if hasattr(y, "isna"):
        valid &= ~np.asarray(y.isna())
    X = X.loc[valid] if hasattr(X, "loc") else np.asarray(X)[valid]
    target = target[valid]
    original_indices = np.flatnonzero(valid)
    categorical_indices = tuple(
        index for index, flag in enumerate(categorical) if bool(flag)
    )
    return X, target, categorical_indices, (task, original_indices)


def _balanced_cap_indices(y: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(y) <= maximum:
        return np.arange(len(y), dtype=int)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    selected: list[int] = []
    per_class = max(1, maximum // len(classes))
    for label in classes:
        indices = np.flatnonzero(y == label)
        selected.extend(
            rng.choice(indices, size=min(per_class, len(indices)), replace=False).tolist()
        )
    remaining = maximum - len(selected)
    if remaining > 0:
        available = np.ones(len(y), dtype=bool)
        available[selected] = False
        candidates = np.flatnonzero(available)
        selected.extend(
            rng.choice(candidates, size=min(remaining, len(candidates)), replace=False).tolist()
        )
    output = np.asarray(selected, dtype=int)
    rng.shuffle(output)
    return output


def load_cc18_split(spec: DatasetSpec) -> FixedSplit:
    X_raw, y_raw, categorical, (_task, original_indices) = _task_data(spec)
    X, _inferred_categorical = _numeric_matrix(X_raw)
    y = LabelEncoder().fit_transform(y_raw)
    selected = _balanced_cap_indices(y, CC18_ROWS, CC18_SEED)
    selected_X = X[selected]
    selected_y = y[selected]
    selected_original = original_indices[selected]
    train_position, test_position = train_test_split(
        np.arange(len(selected_y)),
        train_size=CC18_SOURCE_ROWS,
        test_size=CC18_TARGET_ROWS,
        stratify=selected_y,
        random_state=CC18_SEED,
    )
    return FixedSplit(
        spec=spec,
        X_source=np.asarray(selected_X[train_position], dtype=np.float32),
        X_target=np.asarray(selected_X[test_position], dtype=np.float32),
        y_source=np.asarray(selected_y[train_position], dtype=np.int64),
        y_target=np.asarray(selected_y[test_position], dtype=np.int64),
        source_indices=np.asarray(selected_original[train_position], dtype=np.int64),
        target_indices=np.asarray(selected_original[test_position], dtype=np.int64),
        categorical_features=categorical,
    )


def _cap_partition(
    indices: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= maximum:
        return indices.copy()
    return np.asarray(rng.choice(indices, size=maximum, replace=False), dtype=int)


def load_ctr23_split(spec: DatasetSpec) -> FixedSplit:
    X_raw, y_raw, categorical, (task, original_indices) = _task_data(spec)
    X, _inferred_categorical = _numeric_matrix(X_raw)
    y = np.asarray(y_raw, dtype=np.float64)
    source_original, target_original = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0
    )
    position = {int(original): offset for offset, original in enumerate(original_indices)}
    source = np.asarray(
        [position[int(index)] for index in source_original if int(index) in position],
        dtype=int,
    )
    target = np.asarray(
        [position[int(index)] for index in target_original if int(index) in position],
        dtype=int,
    )
    rng = np.random.default_rng(CTR23_CAP_SEED)
    source = _cap_partition(source, CTR23_SOURCE_CAP, rng)
    target = _cap_partition(target, CTR23_TARGET_CAP, rng)
    return FixedSplit(
        spec=spec,
        X_source=np.asarray(X[source], dtype=np.float32),
        X_target=np.asarray(X[target], dtype=np.float32),
        y_source=np.asarray(y[source], dtype=np.float64),
        y_target=np.asarray(y[target], dtype=np.float64),
        source_indices=np.asarray(original_indices[source], dtype=np.int64),
        target_indices=np.asarray(original_indices[target], dtype=np.int64),
        categorical_features=categorical,
    )


def _encoded_target(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, encoded = np.unique(np.asarray(y), return_inverse=True)
    return classes, encoded.astype(np.float64)


def _top_feature(X: np.ndarray, target: np.ndarray) -> int:
    values = np.asarray(X, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    y_centered = y - float(np.mean(y))
    X_centered = values - np.mean(values, axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(X_centered**2, axis=0)
        * max(float(np.sum(y_centered**2)), 1.0e-12)
    )
    score = np.abs(np.sum(X_centered * y_centered[:, None], axis=0)) / np.maximum(
        denominator, 1.0e-12
    )
    return int(np.nanargmax(np.nan_to_num(score, nan=0.0)))


def _take_nearest(score: np.ndarray, center: float, count: int) -> np.ndarray:
    return np.sort(
        np.argsort(np.abs(np.asarray(score) - center), kind="stable")[:count]
    )


def _select_rows(
    X: np.ndarray,
    y_numeric: np.ndarray,
    *,
    family: str,
    count: int,
    top: int,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=int)
    x = np.asarray(X, dtype=np.float64)[:, top]
    y = np.asarray(y_numeric, dtype=np.float64)
    if family == "shortcut":
        score = np.abs(y - np.median(y))
        return np.sort(np.argsort(score, kind="stable")[-count:])
    if family == "boundary":
        cuts = np.unique(np.quantile(y, [0.25, 0.5, 0.75]))
        distance = np.min(np.abs(y[:, None] - cuts[None, :]), axis=1)
        return np.sort(np.argsort(distance, kind="stable")[:count])
    if family == "band":
        return _take_nearest(x, float(np.median(x)), count)
    if family == "local":
        return _take_nearest(x, float(np.quantile(x, 0.75)), count)
    if family == "mixed":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(len(y), size=count, replace=False))
    raise ValueError(f"unknown source-Context manipulation: {family}")


def _classification_corruption(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    family: str,
    top: int,
    shift_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_modified = np.asarray(X, dtype=np.float64).copy()
    y_modified = np.asarray(y).copy()
    if not len(rows):
        return X_modified, y_modified
    classes, encoded = _encoded_target(y_modified)
    x_mean = np.mean(X_modified, axis=0)
    x_std = np.where(np.std(X_modified, axis=0) > 1.0e-12, np.std(X_modified, axis=0), 1.0)
    midpoint = float(np.median(encoded))
    if family == "shortcut":
        direction = np.where(encoded[rows] >= midpoint, -1.0, 1.0)
        X_modified[rows, top] = x_mean[top] + direction * shift_scale * x_std[top]
    elif family in {"boundary", "band"}:
        stride = 2 if family == "boundary" else 3
        label_rows = rows[::stride]
        y_modified[label_rows] = classes[
            (encoded[label_rows].astype(int) + 1) % len(classes)
        ]
        direction = np.where(encoded[rows] >= midpoint, -1.0, 1.0)
        multiplier = 0.35 if family == "boundary" else 0.22
        X_modified[rows, top] += direction * multiplier * shift_scale * x_std[top]
    elif family == "local":
        label_rows = rows[::2]
        y_modified[label_rows] = classes[
            (encoded[label_rows].astype(int) + 1) % len(classes)
        ]
        direction = np.where(encoded[rows] >= midpoint, -1.0, 1.0)
        X_modified[rows, top] = (
            x_mean[top] + direction * 0.7 * shift_scale * x_std[top]
        )
    else:
        raise ValueError(family)
    return X_modified, y_modified


def _regression_corruption(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    family: str,
    top: int,
    shift_scale: float,
    label_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_modified = np.asarray(X, dtype=np.float64).copy()
    y_modified = np.asarray(y, dtype=np.float64).copy()
    if not len(rows):
        return X_modified, y_modified
    x_mean = np.mean(X_modified, axis=0)
    x_std = np.where(np.std(X_modified, axis=0) > 1.0e-12, np.std(X_modified, axis=0), 1.0)
    y_mid = float(np.median(y_modified))
    y_low, y_high = np.quantile(y_modified, [0.05, 0.95])
    y_std = max(float(np.std(y_modified)), 1.0e-12)
    if family == "shortcut":
        direction = np.where(y_modified[rows] >= y_mid, -1.0, 1.0)
        X_modified[rows, top] = x_mean[top] + direction * shift_scale * x_std[top]
    elif family in {"boundary", "band", "local"}:
        multiplier = {"boundary": 0.5, "band": 0.4, "local": 0.7}[family]
        high_side = y_modified[rows] >= y_mid
        y_modified[rows] = np.where(
            high_side,
            y_low - multiplier * label_scale * y_std,
            y_high + multiplier * label_scale * y_std,
        )
        if family == "local":
            direction = np.where(high_side, -1.0, 1.0)
            X_modified[rows, top] = (
                x_mean[top] + direction * shift_scale * x_std[top]
            )
    else:
        raise ValueError(family)
    return X_modified, y_modified


def manipulate_source_context(
    task: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    *,
    family: str,
    rho: float,
    seed: int,
    shift_scale: float,
    label_scale: float = 1.0,
) -> ManipulationResult:
    if task not in {"classification", "regression"}:
        raise ValueError(task)
    if family not in (*FAMILIES, "original"):
        raise ValueError(family)
    X = np.asarray(X_source, dtype=np.float64)
    y = np.asarray(y_source)
    numeric = _encoded_target(y)[1] if task == "classification" else y.astype(np.float64)
    top = _top_feature(X, numeric)
    count = 0 if family == "original" or rho <= 0 else min(
        len(y), max(1, int(np.ceil(float(rho) * len(y))))
    )
    rows = (
        _select_rows(X, numeric, family=family, count=count, top=top, seed=seed)
        if count
        else np.empty(0, dtype=int)
    )
    if family == "mixed" and len(rows):
        X_modified, y_modified = X.copy(), y.copy()
        for kind, chunk in zip(
            ("shortcut", "boundary", "band", "local"), np.array_split(rows, 4)
        ):
            if task == "classification":
                X_modified, y_modified = _classification_corruption(
                    X_modified,
                    y_modified,
                    np.asarray(chunk),
                    family=kind,
                    top=top,
                    shift_scale=shift_scale,
                )
            else:
                X_modified, y_modified = _regression_corruption(
                    X_modified,
                    y_modified,
                    np.asarray(chunk),
                    family=kind,
                    top=top,
                    shift_scale=shift_scale,
                    label_scale=label_scale,
                )
    elif task == "classification" and count:
        X_modified, y_modified = _classification_corruption(
            X, y, rows, family=family, top=top, shift_scale=shift_scale
        )
    elif task == "regression" and count:
        X_modified, y_modified = _regression_corruption(
            X,
            y,
            rows,
            family=family,
            top=top,
            shift_scale=shift_scale,
            label_scale=label_scale,
        )
    else:
        X_modified, y_modified = X.copy(), y.copy()
    return ManipulationResult(
        X_context=np.asarray(X_modified, dtype=np.float32),
        y_context=np.asarray(y_modified),
        affected_rows=np.asarray(rows, dtype=int),
        top_feature=top,
        metadata={
            "task": task,
            "family": family,
            "rho": float(rho),
            "seed": int(seed),
            "affected_rows": int(len(rows)),
            "source_rows": int(len(y)),
            "query_argument_present": False,
        },
    )
