#!/usr/bin/env python3
"""Run resumable LimiX-2M + OvR controlled-Context benchmarks."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import platform
import signal
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPOSITORY = EXPERIMENT.parents[1]
SCRIPTS = REPOSITORY / "scripts"
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from limix_compat import predict_proba_native_or_ovr  # noqa: E402
from protocol import (  # noqa: E402
    CC18_SEED,
    CC18_SOURCE_ROWS,
    CC18_SUITE_ID,
    CC18_TARGET_ROWS,
    CTR23_CAP_SEED,
    CTR23_SOURCE_CAP,
    CTR23_SUITE_ID,
    CTR23_TARGET_CAP,
    FAMILIES,
    MANIPULATION_SEED,
    STRENGTHS,
    FixedSplit,
    array_sha256,
    filter_specs,
    load_cc18_split,
    load_ctr23_split,
    manipulate_source_context,
    suite_specs,
)


MODEL_SHA256 = "16f385d5201dc17cd6a8eb18becb0b4f8b3603838423b9f06062106acc965513"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("classification", "regression", "both"), default="both"
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Comma-separated OpenML data IDs, task IDs, or exact dataset names.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--classification-pipelines", type=int, default=4)
    parser.add_argument("--regression-pipelines", type=int, default=8)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results" / "paper_run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument(
        "--high-dimensional-threshold",
        type=int,
        default=1024,
        help="Apply source-fit SVD above this feature count; use 0 to disable.",
    )
    parser.add_argument(
        "--high-dimensional-components",
        type=int,
        default=256,
        help="Maximum SVD components for the predeclared high-dimensional path.",
    )
    parser.add_argument(
        "--condition",
        default="",
        help=(
            "Optional comma-separated family:rho cells (for example original:0 "
            "or shortcut:20,mixed:100). Empty runs all 21 unique cells."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_config(root: Path, name: str, count: int) -> list[dict[str, Any]]:
    members = json.loads((root / "config" / name).read_text(encoding="utf-8"))
    if not 1 <= count <= len(members):
        raise ValueError(f"{name}: pipelines must be in [1, {len(members)}]")
    return members[:count]


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(name)


def classification_metrics(
    target: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-12, None)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = classes[np.argmax(probability, axis=1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = float(balanced_accuracy_score(target, predicted))
        try:
            if len(classes) == 2:
                positive = (np.asarray(target) == classes[1]).astype(int)
                auc = float(roc_auc_score(positive, probability[:, 1]))
            else:
                positions = {label: index for index, label in enumerate(classes.tolist())}
                encoded = np.asarray([positions[label] for label in target], dtype=int)
                classwise = []
                for index in range(len(classes)):
                    binary = (encoded == index).astype(int)
                    if np.any(binary == 1) and np.any(binary == 0):
                        classwise.append(float(roc_auc_score(binary, probability[:, index])))
                auc = float(np.mean(classwise)) if classwise else 0.5
        except ValueError:
            auc = 0.5
    return {
        "accuracy": float(np.mean(predicted == target)),
        "balanced_accuracy": balanced,
        "auc": auc,
    }


def regression_metrics(
    target: np.ndarray, prediction: np.ndarray, scale: float
) -> dict[str, float]:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    denominator = max(float(np.sum((target - np.mean(target)) ** 2)), 1.0e-12)
    return {
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "nmae": float(np.mean(np.abs(error))) / scale,
        "nrmse": float(np.sqrt(np.mean(error**2))) / scale,
    }


def conditions(requested: str = "") -> list[tuple[str, int]]:
    available = [("original", 0)] + [
        (family, rho) for family in FAMILIES for rho in STRENGTHS if rho > 0
    ]
    if not requested.strip():
        return available
    selected: list[tuple[str, int]] = []
    for token in requested.split(","):
        family, separator, raw_rho = token.strip().partition(":")
        if not separator:
            raise ValueError(f"invalid --condition token: {token!r}")
        item = (family, int(raw_rho))
        if item not in available:
            raise ValueError(f"unknown --condition cell: {item}")
        if item not in selected:
            selected.append(item)
    return selected


def cell_path(root: Path, task: str, data_id: int, family: str, rho: int) -> Path:
    return root / "cells" / task / f"dataset_{data_id}_{family}_r{rho}.json"


def split_metadata(split: FixedSplit) -> dict[str, Any]:
    return {
        "task_id": split.spec.task_id,
        "data_id": split.spec.data_id,
        "dataset": split.spec.name,
        "n_source": int(len(split.y_source)),
        "n_target": int(len(split.y_target)),
        "n_features": int(split.X_source.shape[1]),
        "n_classes": (
            int(len(np.unique(split.y_source)))
            if np.issubdtype(split.y_source.dtype, np.integer)
            else None
        ),
        "source_indices_sha256": array_sha256(split.source_indices),
        "target_indices_sha256": array_sha256(split.target_indices),
        "source_features_sha256": array_sha256(split.X_source),
        "target_features_sha256": array_sha256(split.X_target),
        "source_target_sha256": array_sha256(split.y_source),
        "evaluation_target_sha256": array_sha256(split.y_target),
    }


def model_features(
    split: FixedSplit,
    X_context: np.ndarray,
    *,
    threshold: int,
    components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Preserve the original-space intervention, then adapt only model input.

    The projection is fit on the unmodified labeled source features and is
    fixed across every condition of a dataset. Target labels never enter it.
    """

    if threshold <= 0 or split.X_source.shape[1] <= threshold:
        return (
            np.asarray(X_context, dtype=np.float32),
            np.asarray(split.X_target, dtype=np.float32),
            {
                "feature_adapter": "identity",
                "model_n_features": int(split.X_source.shape[1]),
                "feature_adapter_uses_target_labels": False,
            },
        )
    n_components = min(
        int(components),
        int(split.X_source.shape[0] - 1),
        int(split.X_source.shape[1] - 1),
    )
    if n_components < 1:
        raise ValueError("high-dimensional SVD requires at least one component")
    projector = TruncatedSVD(n_components=n_components, random_state=CC18_SEED)
    projector.fit(np.asarray(split.X_source, dtype=np.float64))
    return (
        np.asarray(projector.transform(X_context), dtype=np.float32),
        np.asarray(projector.transform(split.X_target), dtype=np.float32),
        {
            "feature_adapter": "truncated_svd_source_fit",
            "model_n_features": n_components,
            "svd_explained_variance_ratio_sum": float(
                np.sum(projector.explained_variance_ratio_)
            ),
            "feature_adapter_uses_target_labels": False,
        },
    )


def run_classification_cell(
    predictor: Any,
    split: FixedSplit,
    family: str,
    rho: int,
    *,
    high_dimensional_threshold: int,
    high_dimensional_components: int,
) -> dict[str, Any]:
    result = manipulate_source_context(
        "classification",
        split.X_source,
        split.y_source,
        family=family,
        rho=rho / 100.0,
        seed=MANIPULATION_SEED + rho,
        shift_scale=6.0,
    )
    X_context, X_target, feature_metadata = model_features(
        split,
        result.X_context,
        threshold=high_dimensional_threshold,
        components=high_dimensional_components,
    )
    started = time.perf_counter()
    prediction = predict_proba_native_or_ovr(
        predictor,
        X_context,
        result.y_context,
        X_target,
        max_native_classes=10,
    )
    metrics = classification_metrics(
        split.y_target, prediction.probabilities, prediction.classes
    )
    return {
        **split_metadata(split),
        "cell_schema_version": 2,
        "task": "classification",
        "family": family,
        "rho": rho,
        "strategy": prediction.strategy,
        "model_calls": prediction.model_calls,
        "affected_rows": int(len(result.affected_rows)),
        "top_feature": int(result.top_feature),
        **feature_metadata,
        "seconds": time.perf_counter() - started,
        **metrics,
    }


def run_regression_cell(
    predictor: Any,
    split: FixedSplit,
    family: str,
    rho: int,
    *,
    high_dimensional_threshold: int,
    high_dimensional_components: int,
) -> dict[str, Any]:
    result = manipulate_source_context(
        "regression",
        split.X_source,
        split.y_source,
        family=family,
        rho=rho / 100.0,
        seed=MANIPULATION_SEED + rho,
        shift_scale=10.0,
        label_scale=1.0,
    )
    target_mean = float(np.mean(result.y_context))
    target_std = max(float(np.std(result.y_context)), 1.0e-12)
    normalized = (np.asarray(result.y_context, dtype=np.float64) - target_mean) / target_std
    X_context, X_target, feature_metadata = model_features(
        split,
        result.X_context,
        threshold=high_dimensional_threshold,
        components=high_dimensional_components,
    )
    started = time.perf_counter()
    raw_prediction = predictor.predict(
        X_context,
        normalized,
        X_target,
        task_type="Regression",
    )
    if hasattr(raw_prediction, "detach"):
        raw_prediction = raw_prediction.detach()
    if hasattr(raw_prediction, "cpu"):
        raw_prediction = raw_prediction.cpu()
    normalized_prediction = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
    prediction = normalized_prediction * target_std + target_mean
    metric_scale = max(float(np.std(split.y_source)), 1.0e-12)
    metrics = regression_metrics(split.y_target, prediction, metric_scale)
    return {
        **split_metadata(split),
        "cell_schema_version": 2,
        "task": "regression",
        "family": family,
        "rho": rho,
        "strategy": "native",
        "model_calls": 1,
        "affected_rows": int(len(result.affected_rows)),
        "top_feature": int(result.top_feature),
        **feature_metadata,
        "source_scale": metric_scale,
        "context_target_mean": target_mean,
        "context_target_std": target_std,
        "seconds": time.perf_counter() - started,
        **metrics,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def lock_run_signature(output: Path, signature: dict[str, Any]) -> None:
    path = output / "run_signature.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError(
                "run signature differs from cached cells; choose a new --output "
                "directory instead of mixing model/configuration variants"
            )
        return
    write_json(path, signature)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in cells:
        if row["family"] == "original":
            for family in FAMILIES:
                expanded.append({**row, "family": "aucband" if family == "band" else family})
        else:
            expanded.append(
                {**row, "family": "aucband" if row["family"] == "band" else row["family"]}
            )
    metrics = {
        "classification": ("accuracy", "auc"),
        "regression": ("r2", "nmae", "nrmse"),
    }
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in expanded:
        for metric in metrics[row["task"]]:
            grouped[(row["task"], row["family"], int(row["rho"]), metric)].append(
                float(row[metric])
            )
    output = []
    for (task, family, rho, metric), values in sorted(grouped.items()):
        output.append(
            {
                "suite": "five_shift",
                "task": task,
                "model": "limix_2m_ovr",
                "display_name": "LimiX-2M + OvR",
                "family": family,
                "intensity": float(rho),
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "dataset_n": len(values),
            }
        )
    return output


def main() -> None:
    args = arguments()
    stack_signal = getattr(signal, "SIGUSR1", None)
    if stack_signal is not None:
        faulthandler.register(stack_signal, all_threads=True)
    root = args.limix_root.resolve()
    model = args.model_path.resolve()
    output = args.output.resolve()
    # Keep concurrent smoke/full runs from racing on OpenML's default global
    # cache. The benchmark output is also the natural provenance boundary.
    import openml

    openml.config.set_root_cache_directory(output / "openml_cache")
    if not model.is_file():
        raise FileNotFoundError(model)
    checkpoint_hash = file_sha256(model)
    if checkpoint_hash != MODEL_SHA256:
        raise RuntimeError(
            f"unexpected checkpoint SHA256: {checkpoint_hash}; expected {MODEL_SHA256}"
        )
    sys.path.insert(0, str(root))
    from inference.predictor import LimiXPredictor

    device = select_device(args.device)
    limix_revision = git_revision(root)
    run_signature = {
        "protocol": "limix_2m_ovr_five_v1",
        "checkpoint_sha256": checkpoint_hash,
        "limix_git_revision": limix_revision,
        "device_type": device.type,
        "classification_pipelines": args.classification_pipelines,
        "regression_pipelines": args.regression_pipelines,
        "high_dimensional_threshold": args.high_dimensional_threshold,
        "high_dimensional_components": args.high_dimensional_components,
    }
    lock_run_signature(output, run_signature)
    requested = {
        item.strip().casefold() for item in args.dataset.split(",") if item.strip()
    }
    selected_tasks = (
        ("classification", "regression") if args.task == "both" else (args.task,)
    )
    predictor_by_task: dict[str, Any] = {}
    configs = {
        "classification": load_config(
            root, "cls_default_noretrieval.json", args.classification_pipelines
        ),
        "regression": load_config(
            root, "reg_default_noretrieval.json", args.regression_pipelines
        ),
    }
    all_cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_name in selected_tasks:
        suite_id = CC18_SUITE_ID if task_name == "classification" else CTR23_SUITE_ID
        specs = filter_specs(suite_specs(suite_id, requested), requested)
        if args.max_datasets > 0:
            specs = specs[: args.max_datasets]
        predictor_by_task[task_name] = LimiXPredictor(
            device=device,
            model_path=str(model),
            inference_config=configs[task_name],
            mix_precision=device.type == "cuda",
            seed=CC18_SEED if task_name == "classification" else CTR23_CAP_SEED,
        )
        predictor = predictor_by_task[task_name]
        for position, spec in enumerate(specs, start=1):
            selected_conditions = conditions(args.condition)
            cached_paths = [
                cell_path(output, task_name, spec.data_id, family, rho)
                for family, rho in selected_conditions
            ]
            if not args.force and all(path.exists() for path in cached_paths):
                cached_rows = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in cached_paths
                ]
                if all("feature_adapter" in row for row in cached_rows):
                    all_cells.extend(cached_rows)
                    print(
                        f"[{task_name} {position}/{len(specs)}] cached dataset "
                        f"{spec.name} data_id={spec.data_id}",
                        flush=True,
                    )
                    continue
            print(
                f"[{task_name} {position}/{len(specs)}] loading "
                f"{spec.name} data_id={spec.data_id}",
                flush=True,
            )
            try:
                split = (
                    load_cc18_split(spec)
                    if task_name == "classification"
                    else load_ctr23_split(spec)
                )
            except Exception as error:
                failure = {
                    "task": task_name,
                    "data_id": spec.data_id,
                    "dataset": spec.name,
                    "stage": "load",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                write_json(
                    output / "failures" / f"{task_name}_{spec.data_id}_load.json", failure
                )
                if not args.continue_on_error:
                    raise
                continue
            for family, rho in selected_conditions:
                path = cell_path(output, task_name, spec.data_id, family, rho)
                if path.exists() and not args.force:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if "feature_adapter" not in row:
                        if (
                            args.high_dimensional_threshold > 0
                            and int(row["n_features"])
                            > args.high_dimensional_threshold
                        ):
                            raise RuntimeError(
                                f"stale high-dimensional cached cell requires a new "
                                f"output directory: {path}"
                            )
                        row.update(
                            {
                                "cell_schema_version": 2,
                                "feature_adapter": "identity",
                                "model_n_features": int(row["n_features"]),
                                "feature_adapter_uses_target_labels": False,
                            }
                        )
                        write_json(path, row)
                    all_cells.append(row)
                    print(f"  cached {family} rho={rho}", flush=True)
                    continue
                try:
                    row = (
                        run_classification_cell(
                            predictor,
                            split,
                            family,
                            rho,
                            high_dimensional_threshold=args.high_dimensional_threshold,
                            high_dimensional_components=args.high_dimensional_components,
                        )
                        if task_name == "classification"
                        else run_regression_cell(
                            predictor,
                            split,
                            family,
                            rho,
                            high_dimensional_threshold=args.high_dimensional_threshold,
                            high_dimensional_components=args.high_dimensional_components,
                        )
                    )
                    write_json(path, row)
                    all_cells.append(row)
                    print(
                        f"  completed {family} rho={rho} "
                        f"strategy={row['strategy']} seconds={row['seconds']:.2f}",
                        flush=True,
                    )
                except Exception as error:
                    failure = {
                        "task": task_name,
                        "data_id": spec.data_id,
                        "dataset": spec.name,
                        "stage": "inference",
                        "family": family,
                        "rho": rho,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    write_json(
                        output
                        / "failures"
                        / f"{task_name}_{spec.data_id}_{family}_r{rho}.json",
                        failure,
                    )
                    if not args.continue_on_error:
                        raise
    # Rebuild aggregate artifacts from every durable cell, including cells
    # created by earlier filtered/resumed invocations.
    all_cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "cells").rglob("*.json")
    ] if (output / "cells").exists() else []
    all_cells.sort(key=lambda row: (row["task"], int(row["data_id"]), row["family"], int(row["rho"])))
    write_csv(output / "detail.csv", all_cells)
    write_csv(output / "summary_long.csv", summarize(all_cells))
    audit = {
        "method": "LimiX-2M + OvR",
        "checkpoint": str(model),
        "checkpoint_sha256": checkpoint_hash,
        "limix_git_revision": limix_revision,
        "expected_limix_tag": "V1.1.0",
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "classification_pipelines": args.classification_pipelines,
        "regression_pipelines": args.regression_pipelines,
        "native_max_classes": 10,
        "multiclass_compatibility": "one-vs-rest binary LimiX with row normalization",
        "high_dimensional_adapter": {
            "threshold": args.high_dimensional_threshold,
            "maximum_components": args.high_dimensional_components,
            "method": "source-fit TruncatedSVD after original-space perturbation",
            "uses_target_labels": False,
            "adapted_cells": sum(
                row.get("feature_adapter") == "truncated_svd_source_fit"
                for row in all_cells
            ),
        },
        "cc18": {
            "suite_id": CC18_SUITE_ID,
            "seed": CC18_SEED,
            "source_rows": CC18_SOURCE_ROWS,
            "target_rows": CC18_TARGET_ROWS,
            "loader_provenance": "recovered call sites plus existing OpenML utility",
        },
        "ctr23": {
            "suite_id": CTR23_SUITE_ID,
            "official_split": {"repeat": 0, "fold": 0, "sample": 0},
            "cap_seed": CTR23_CAP_SEED,
            "source_cap": CTR23_SOURCE_CAP,
            "target_cap": CTR23_TARGET_CAP,
            "loader_provenance": "paper protocol; historical loader absent locally",
        },
        "controlled_perturbations": {
            "families": list(FAMILIES),
            "strengths": list(STRENGTHS),
            "manipulation_seed": MANIPULATION_SEED,
            "classification_shift_scale": 6.0,
            "regression_shift_scale": 10.0,
            "target_evaluation_rows_fixed": True,
            "target_labels_used_for_construction": False,
        },
        "seven_shift_status": "not_run_missing_historical_materializer",
        "completed_cells": len(all_cells),
        "failures": len(failures),
        "native_classification_cells": sum(
            row["task"] == "classification" and row["strategy"] == "native"
            for row in all_cells
        ),
        "ovr_classification_cells": sum(
            row["task"] == "classification" and row["strategy"] == "one_vs_rest"
            for row in all_cells
        ),
        "total_model_calls": int(sum(int(row["model_calls"]) for row in all_cells)),
        "total_inference_seconds": float(sum(float(row["seconds"]) for row in all_cells)),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
