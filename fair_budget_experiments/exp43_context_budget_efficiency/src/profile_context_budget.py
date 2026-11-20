#!/usr/bin/env python3
"""Measure synchronized inference cost for Context budgets K=1,...,5."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
SHIFTS = ("covariate", "label", "concept", "selection", "domain", "temporal", "support")
STRENGTH = 0.5
CLASS_DATASETS = (3, 31, 1049)
REG_DATASETS = ("abalone", "airfoil_self_noise", "california_housing")
K_VALUES = (0, 1, 2, 3, 4, 5)
CLASS_CONTEXTS = (
    "uniform",
    "class_balanced",
    "target_matched",
    "balanced_target_matched",
    "affine_transport",
)
REG_CONTEXTS = (
    "quantile_balanced",
    "target_matched",
    "mean_transport",
    "reference_heavy",
    "support_coverage",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sync_time(
    function: Callable[[], object], *, reset_peak: bool = True
) -> tuple[object, float, float]:
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
    return value, elapsed, peak_mb


class _CallableTimer:
    def __init__(self, instance: object):
        self.instance = instance
        self.cls = type(instance)
        self.original = self.cls.__call__
        self.seconds = 0.0
        self.calls = 0

    def __enter__(self):
        timer = self

        def wrapped(instance, *args, **kwargs):
            value, elapsed, _peak = _sync_time(
                lambda: timer.original(instance, *args, **kwargs), reset_peak=False
            )
            timer.seconds += elapsed
            timer.calls += 1
            return value

        self.cls.__call__ = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.cls.__call__ = self.original


class _FunctionTimer:
    def __init__(self, function: Callable[..., object]):
        self.original = function
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        value, elapsed, _peak = _sync_time(
            lambda: self.original(*args, **kwargs), reset_peak=False
        )
        self.seconds += elapsed
        self.calls += 1
        return value


def _order(cell_index: int, repeat: int) -> tuple[int, ...]:
    orders = (
        (0, 1, 2, 3, 4, 5),
        (5, 4, 3, 2, 1, 0),
        (2, 0, 4, 1, 5, 3),
    )
    return orders[(cell_index + repeat) % len(orders)]


def _row(
    task: str,
    dataset: object,
    shift: str,
    repeat: int,
    k: int,
    names: tuple[str, ...],
    total: float,
    backbone: float,
    calls: int,
    peak_mb: float,
) -> dict[str, object]:
    expected_calls = 1 if k == 0 else k
    if calls != expected_calls:
        raise RuntimeError(
            f"{task}/{dataset}/{shift}/K={k}: expected {expected_calls} "
            f"backbone calls, observed {calls}"
        )
    return {
        "task": task,
        "dataset": dataset,
        "shift": shift,
        "strength": STRENGTH,
        "repeat": repeat,
        "k": k,
        "label": "Base" if k == 0 else f"K={k}",
        "context_names": "|".join(names),
        "logical_calls": expected_calls,
        "observed_backbone_calls": calls,
        "total_seconds": total,
        "backbone_seconds": backbone,
        "context_and_fusion_seconds": max(0.0, total - backbone),
        "peak_cuda_allocated_mb": peak_mb,
    }


def _classification(repeats: int) -> list[dict[str, object]]:
    module = _load(
        "exp43_class_runner",
        ROOT / "fair_budget_experiments" / "exp01_classification" / "src" / "run_fair_budget.py",
    )
    _, warm = module._condition(CLASS_DATASETS[0], "covariate", STRENGTH)
    warm_seed = module._seed(CLASS_DATASETS[0], "covariate", 90_000)
    warm_X, warm_y = module.E14._base_context(warm, warm_seed)
    module._REUSABLE_CLASSIFIER(warm_X, warm_y, warm.X_target, device="cuda")

    rows: list[dict[str, object]] = []
    cell_index = 0
    for dataset in CLASS_DATASETS:
        for shift in SHIFTS:
            _, condition = module._condition(dataset, shift, STRENGTH)
            seed = module._seed(dataset, shift, 70_000)
            source_X = np.vstack([condition.X_source, condition.X_reference])
            source_y = np.concatenate([condition.y_source, condition.y_reference])

            def base_method():
                X, y = module.E14._base_context(condition, seed)
                return module._REUSABLE_CLASSIFIER(X, y, condition.X_target, device="cuda")

            def budget_method(k: int):
                predictions: list[np.ndarray] = []
                propensity: np.ndarray | None = None
                if k >= 1:
                    predictions.append(
                        module._uniform_prediction(
                            source_X,
                            source_y,
                            condition.X_target,
                            seed=seed + 101,
                            device="cuda",
                        )
                    )
                if k >= 2:
                    predictions.append(
                        module._balanced_prediction(
                            source_X,
                            source_y,
                            condition.X_target,
                            seed=seed + 211,
                            device="cuda",
                        )
                    )
                if k >= 3:
                    propensity, _ = module.E14.propensity_weights_cv(
                        source_X, condition.X_target, seed + 3
                    )
                    predictions.append(
                        module._uniform_prediction(
                            source_X,
                            source_y,
                            condition.X_target,
                            seed=seed + 307,
                            device="cuda",
                            probability=propensity,
                        )
                    )
                if k >= 4:
                    if propensity is None:
                        raise RuntimeError("propensity must be available at K>=4")
                    predictions.append(
                        module._balanced_prediction(
                            source_X,
                            source_y,
                            condition.X_target,
                            seed=seed + 401,
                            device="cuda",
                            probability=propensity,
                        )
                    )
                if k >= 5:
                    affine_X = module.E14.transport_source_to_target(
                        source_X, condition.X_target, scale=True
                    )
                    predictions.append(
                        module._uniform_prediction(
                            affine_X,
                            source_y,
                            condition.X_target,
                            seed=seed + 503,
                            device="cuda",
                        )
                    )
                return np.mean(predictions, axis=0)

            for repeat in range(repeats):
                for k in _order(cell_index, repeat):
                    function = base_method if k == 0 else lambda k=k: budget_method(k)
                    with _CallableTimer(module._REUSABLE_CLASSIFIER) as timer:
                        _prediction, total, peak_mb = _sync_time(function)
                    rows.append(
                        _row(
                            "classification",
                            dataset,
                            shift,
                            repeat,
                            k,
                            () if k == 0 else CLASS_CONTEXTS[:k],
                            total,
                            timer.seconds,
                            timer.calls,
                            peak_mb,
                        )
                    )
            print(
                f"classification cell {cell_index + 1}/21: dataset={dataset}, shift={shift}",
                flush=True,
            )
            cell_index += 1
    return rows


def _regression(repeats: int) -> list[dict[str, object]]:
    module = _load(
        "exp43_reg_runner",
        ROOT / "fair_budget_experiments" / "exp02_regression" / "src" / "run_fair_budget.py",
    )
    predictor = module.E28.Predictor(device="cuda", n_estimators=1, random_state=42)
    warm_split = module.E28.load_dataset_split(REG_DATASETS[0])
    warm_shifted = module.E28.construct_shift(warm_split, "covariate", STRENGTH)
    warm_context = module.E28._sample_context(
        warm_shifted.X, warm_shifted.y, seed=91_001
    )
    module._predict(predictor, warm_context, warm_split.X_target)

    rows: list[dict[str, object]] = []
    cell_index = 0
    for dataset in REG_DATASETS:
        split = module.E28.load_dataset_split(dataset)
        for shift in SHIFTS:
            shifted = module.E28.construct_shift(split, shift, STRENGTH)
            seed = module._seed(dataset, shift, 70_000)
            reservoir_X = np.vstack([shifted.X, split.X_history, split.X_reference])
            reservoir_y = np.concatenate([shifted.y, split.y_history, split.y_reference])

            def base_method():
                context = module.E28._sample_context(shifted.X, shifted.y, seed=seed)
                return module._predict(predictor, context, split.X_target)

            def budget_method(k: int):
                contexts: list[tuple[np.ndarray, np.ndarray]] = []
                shared_seed = module._seed(dataset, shift, 10_000)
                if k >= 1:
                    contexts.append(
                        module._balanced_context(reservoir_X, reservoir_y, shared_seed + 101)
                    )
                if k >= 2:
                    contexts.append(
                        module._target_context(
                            reservoir_X,
                            reservoir_y,
                            split.X_target,
                            shared_seed + 211,
                        )
                    )
                if k >= 3:
                    transported = module.E28._mean_transport(
                        reservoir_X, split.X_target, scale=True
                    )
                    contexts.append(
                        module.E28._sample_context(
                            transported, reservoir_y, seed=shared_seed + 307
                        )
                    )
                if k >= 4:
                    contexts.append(module._reference_heavy(split, shared_seed + 401))
                if k >= 5:
                    contexts.append(module.E28._coverage_context(reservoir_X, reservoir_y))
                predictions = [
                    module._predict(predictor, context, split.X_target)
                    for context in contexts
                ]
                return np.mean(predictions, axis=0)

            original = module._predict
            for repeat in range(repeats):
                for k in _order(cell_index, repeat):
                    function = base_method if k == 0 else lambda k=k: budget_method(k)
                    timer = _FunctionTimer(original)
                    module._predict = timer
                    try:
                        _prediction, total, peak_mb = _sync_time(function)
                    finally:
                        module._predict = original
                    rows.append(
                        _row(
                            "regression",
                            dataset,
                            shift,
                            repeat,
                            k,
                            () if k == 0 else REG_CONTEXTS[:k],
                            total,
                            timer.seconds,
                            timer.calls,
                            peak_mb,
                        )
                    )
            print(
                f"regression cell {cell_index + 1}/21: dataset={dataset}, shift={shift}",
                flush=True,
            )
            cell_index += 1
    return rows


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bases = {
        (row["task"], row["dataset"], row["shift"], row["repeat"]): float(
            row["total_seconds"]
        )
        for row in rows
        if int(row["k"]) == 0
    }
    output: list[dict[str, object]] = []
    for task in ("classification", "regression"):
        for k in K_VALUES:
            cell = [
                row for row in rows if row["task"] == task and int(row["k"]) == k
            ]
            total = np.asarray([float(row["total_seconds"]) for row in cell])
            backbone = np.asarray([float(row["backbone_seconds"]) for row in cell])
            residual = np.asarray(
                [float(row["context_and_fusion_seconds"]) for row in cell]
            )
            relative = np.asarray(
                [
                    float(row["total_seconds"])
                    / bases[(row["task"], row["dataset"], row["shift"], row["repeat"])]
                    for row in cell
                ]
            )
            peak = np.asarray([float(row["peak_cuda_allocated_mb"]) for row in cell])
            output.append(
                {
                    "task": task,
                    "k": k,
                    "label": "Base" if k == 0 else f"K={k}",
                    "logical_calls": 1 if k == 0 else k,
                    "n_cells": len(cell),
                    "median_total_seconds": statistics.median(total.tolist()),
                    "p25_total_seconds": float(np.percentile(total, 25)),
                    "p75_total_seconds": float(np.percentile(total, 75)),
                    "median_backbone_seconds": statistics.median(backbone.tolist()),
                    "median_context_and_fusion_seconds": statistics.median(
                        residual.tolist()
                    ),
                    "median_relative_to_paired_base": statistics.median(
                        relative.tolist()
                    ),
                    "p25_relative_to_paired_base": float(np.percentile(relative, 25)),
                    "p75_relative_to_paired_base": float(np.percentile(relative, 75)),
                    "median_peak_cuda_allocated_mb": statistics.median(peak.tolist()),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=EXP / "results")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for publication-comparable timing")

    rows = _classification(args.repeats) + _regression(args.repeats)
    summary = _summary(rows)
    _write_csv(args.output_dir / "timing_detail.csv", rows)
    _write_csv(args.output_dir / "timing_summary.csv", summary)
    metadata = {
        "experiment_id": "exp43_context_budget_efficiency",
        "source_protocol": "exp11_component_efficiency",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_synchronized": True,
        "checkpoint_warmed_once_per_task": True,
        "shift_construction_included": False,
        "context_construction_included": True,
        "fusion_included": True,
        "strength": STRENGTH,
        "classification_datasets": list(CLASS_DATASETS),
        "regression_datasets": list(REG_DATASETS),
        "shift_families": list(SHIFTS),
        "repeats_per_cell": args.repeats,
        "k_values": list(K_VALUES),
        "base_is_separately_measured": True,
        "k_counts_nonbase_context_predictions": True,
        "prefixes": {
            "classification": list(CLASS_CONTEXTS),
            "regression": list(REG_CONTEXTS),
        },
        "timing_order": "deterministic balanced rotation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
