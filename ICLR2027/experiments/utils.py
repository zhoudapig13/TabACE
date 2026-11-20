from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import psutil
import torch
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl.__about__ import __version__ as tabicl_version


RESULTS_DIR = Path("results")
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    X: np.ndarray
    y: np.ndarray


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def environment_info(device: str) -> dict[str, str | bool]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tabicl": tabicl_version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }


def parse_scales(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_named_dataset(name: str, n_samples: int | None = None, random_state: int = DEFAULT_RANDOM_STATE) -> DatasetSpec:
    name = name.strip().lower()
    if name == "synthetic_classification":
        sample_count = n_samples or 300
        X, y = make_classification(
            n_samples=sample_count,
            n_features=20,
            n_informative=12,
            n_redundant=2,
            class_sep=1.0,
            random_state=random_state,
        )
        return DatasetSpec(name=name, task="classification", X=X, y=y)

    if name == "synthetic_regression":
        sample_count = n_samples or 300
        X, y = make_regression(
            n_samples=sample_count,
            n_features=20,
            n_informative=12,
            noise=10.0,
            random_state=random_state,
        )
        return DatasetSpec(name=name, task="regression", X=X, y=y)

    loaders = {
        "breast_cancer": (load_breast_cancer, "classification"),
        "wine": (load_wine, "classification"),
        "digits": (load_digits, "classification"),
        "diabetes": (load_diabetes, "regression"),
    }
    if name not in loaders:
        available = ", ".join(sorted([*loaders, "synthetic_classification", "synthetic_regression"]))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

    loader, task = loaders[name]
    data = loader()
    X = np.asarray(data.data)
    y = np.asarray(data.target)
    if n_samples is not None and n_samples < X.shape[0]:
        rng = np.random.default_rng(random_state)
        if task == "classification":
            # Keep a simple deterministic subset; stratification happens in train_test_split.
            indices = rng.permutation(X.shape[0])[:n_samples]
        else:
            indices = rng.permutation(X.shape[0])[:n_samples]
        X = X[indices]
        y = y[indices]
    return DatasetSpec(name=name, task=task, X=X, y=y)


def run_tabicl_experiment(
    spec: DatasetSpec,
    *,
    n_estimators: int,
    device: str,
    random_state: int,
    test_size: float = 0.25,
    kv_cache: bool = False,
    classifier_model_path: str | None = None,
    regressor_model_path: str | None = None,
) -> dict[str, object]:
    X = np.asarray(spec.X)
    y = np.asarray(spec.y)
    stratify = y if spec.task == "classification" and len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()

    if spec.task == "classification":
        model = TabICLClassifier(
            n_estimators=n_estimators,
            device=device,
            random_state=random_state,
            kv_cache=kv_cache,
            model_path=classifier_model_path,
            verbose=False,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric_name = "accuracy"
        metric_value = float(accuracy_score(y_test, pred))
        extra_metrics = {}
    elif spec.task == "regression":
        model = TabICLRegressor(
            n_estimators=n_estimators,
            device=device,
            random_state=random_state,
            kv_cache=kv_cache,
            model_path=regressor_model_path,
            verbose=False,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric_name = "r2"
        metric_value = float(r2_score(y_test, pred))
        extra_metrics = {"mae": float(mean_absolute_error(y_test, pred))}
    else:
        raise ValueError(f"Unsupported task type: {spec.task}")

    elapsed = time.perf_counter() - start
    rss_after = process.memory_info().rss

    result: dict[str, object] = {
        "dataset": spec.name,
        "task": spec.task,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "model": "TabICL",
        "model_path": classifier_model_path if spec.task == "classification" else regressor_model_path,
        "n_estimators": int(n_estimators),
        "device": device,
        "kv_cache": bool(kv_cache),
        "random_state": int(random_state),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "runtime_sec": elapsed,
        "rss_before_mb": rss_before / 1024**2,
        "rss_after_mb": rss_after / 1024**2,
        "rss_delta_mb": (rss_after - rss_before) / 1024**2,
    }
    result.update(extra_metrics)
    return result


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    ensure_results_dir()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    ensure_results_dir()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--kv-cache", action="store_true")
