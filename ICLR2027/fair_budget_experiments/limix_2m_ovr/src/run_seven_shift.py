#!/usr/bin/env python3
"""Run LimiX-2M on the paper's frozen seven-distribution-shift protocol."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
SCRIPTS = REPOSITORY / "scripts"
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from limix_compat import predict_proba_native_or_ovr  # noqa: E402
from run_benchmark import (  # noqa: E402
    MODEL_SHA256,
    classification_metrics,
    file_sha256,
    git_revision,
    load_config,
    lock_run_signature,
    regression_metrics,
    select_device,
    write_csv,
    write_json,
)


FAMILIES = (
    "covariate",
    "label",
    "concept",
    "selection",
    "domain",
    "temporal",
    "support",
)
STRENGTHS = (0.0, 0.2, 0.5, 0.8, 1.0)
EXPECTED_DATASETS = {"classification": 35, "regression": 33}
EXPECTED_PIPELINES = {"classification": 4, "regression": 8}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        required=True,
        help="Repository root containing the frozen baseline seven-shift sources.",
    )
    parser.add_argument(
        "--task", choices=("classification", "regression", "both"), default="both"
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional comma-separated data IDs or exact dataset names.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--classification-pipelines", type=int, default=4)
    parser.add_argument("--regression-pipelines", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=HERE.parent / "results" / "paper_run_seven"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--high-dimensional-threshold", type=int, default=1024)
    parser.add_argument("--high-dimensional-components", type=int, default=256)
    return parser.parse_args()


def load_frozen_protocol(root: Path) -> Any:
    paths = (
        root,
        root / "src",
        root / "experiment6" / "scripts",
        root / "experiment7" / "scripts",
        root / "experiment8" / "scripts",
        root / "experiment11" / "scripts",
        root / "experiment13",
        root / "experiment14",
        root / "experiment15",
        root / "experiment17",
        root / "experiment28",
        root / "fair_budget_experiments" / "baseline_shift_comparison" / "src",
    )
    required = paths[-1] / "run_direct_baselines.py"
    if not required.is_file():
        raise FileNotFoundError(
            f"frozen seven-shift materializer is missing: {required}"
        )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("run_direct_baselines")


def protocol_hashes(root: Path) -> dict[str, str]:
    relative = (
        "fair_budget_experiments/baseline_shift_comparison/src/run_direct_baselines.py",
        "experiment14/run_unified_context_bank.py",
        "experiment15/run_shift_specific_contexts.py",
        "experiment28/run_ctr23_regression_shifts.py",
        "fair_budget_experiments/unified_context_ensemble/src/repairable_seven_regression_shifts.py",
        "fair_budget_experiments/unified_context_ensemble/src/hierarchical_seven_regression_shifts_v4.py",
        "fair_budget_experiments/unified_context_ensemble/src/hierarchical_seven_regression_shifts_v7.py",
    )
    output = {}
    for name in relative:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required frozen protocol source is missing: {path}")
        output[name] = file_sha256(path)
    return output


def prepare_cc18_numeric_mirrors(
    protocol_root: Path, output: Path, data_ids: list[int]
) -> dict[str, Any]:
    """Reproduce the frozen loader's numeric conversion from local parquet.

    The historical seven-shift loader has two OpenML paths: its network path
    applies ``as_numeric_matrix`` while its parquet fast path assumes every
    feature is already numeric.  Official OpenML parquet retains nominal
    strings, so stage numeric mirrors inside this run and redirect only the
    loader's data root.  The frozen shift constructors themselves are not
    modified.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import run_apa_tabicl as apa

    source_root = output / "openml_cache" / "org" / "openml" / "www" / "datasets"
    stage_root = output / "protocol_input"
    mirror_root = stage_root / "experiment7" / "data" / "openml_parquet"
    mirror_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for data_id in data_ids:
        source = source_root / str(data_id) / f"dataset_{data_id}.pq"
        destination = mirror_root / f"dataset_{data_id}.pq"
        if not source.is_file():
            raise FileNotFoundError(
                f"missing cached OpenML parquet for CC18 data_id={data_id}: {source}"
            )
        if not destination.is_file():
            table = pq.read_table(source)
            column_names = list(table.column_names)
            target_name = "class" if "class" in column_names else column_names[-1]
            feature_names = [name for name in column_names if name != target_name]
            raw = np.column_stack(
                [
                    table[name].combine_chunks().to_numpy(zero_copy_only=False)
                    for name in feature_names
                ]
            )
            numeric = apa.as_numeric_matrix(raw)
            arrays = [
                pa.array(numeric[:, index], type=pa.float32())
                for index in range(numeric.shape[1])
            ]
            arrays.append(table[target_name].combine_chunks())
            staged_feature_names = (
                feature_names
                if numeric.shape[1] == len(feature_names)
                else [
                    f"__numeric_feature_{index:05d}"
                    for index in range(numeric.shape[1])
                ]
            )
            staged = pa.Table.from_arrays(
                arrays, names=[*staged_feature_names, target_name]
            )
            temporary = destination.with_suffix(".tmp.pq")
            pq.write_table(staged, temporary)
            temporary.replace(destination)
        rows.append(
            {
                "data_id": data_id,
                "source_sha256": file_sha256(source),
                "mirror": str(destination),
            }
        )
    apa.ROOT = stage_root
    manifest = {
        "adapter": "frozen_as_numeric_matrix_v1",
        "protocol_root_unchanged": str(protocol_root),
        "dataset_count": len(rows),
        "datasets": rows,
    }
    write_json(output / "classification_input_manifest.json", manifest)
    return manifest


def selected_datasets(protocol: Any, task: str, requested: str) -> list[str | int]:
    if task == "classification":
        available: list[str | int] = [int(value) for value in protocol.CC18_BINARY_DATASETS]
        names = {int(key): str(value) for key, value in protocol.CC18_DATASET_NAMES.items()}
    else:
        available = list(protocol._regression_names("all"))
        names = {str(value): str(value) for value in available}
    tokens = {value.strip().casefold() for value in requested.split(",") if value.strip()}
    if tokens:
        selected = [
            value
            for value in available
            if str(value).casefold() in tokens or names[value].casefold() in tokens
        ]
        missing = tokens - {
            token
            for value in selected
            for token in (str(value).casefold(), names[value].casefold())
        }
        if missing:
            raise ValueError(f"unknown {task} seven-shift datasets: {sorted(missing)}")
        available = selected
    return available


def cell_path(
    output: Path, task: str, data_id: int, family: str, strength: float
) -> Path:
    return (
        output
        / "cells"
        / task
        / f"dataset_{data_id}_{family}_s{int(round(100 * strength))}.json"
    )


def array_sha256(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fit_feature_adapter(
    clean_source: np.ndarray, *, threshold: int, components: int
) -> tuple[TruncatedSVD | None, dict[str, Any]]:
    clean_source = np.asarray(clean_source)
    if threshold <= 0 or clean_source.shape[1] <= threshold:
        return None, {
            "feature_adapter": "identity",
            "model_n_features": int(clean_source.shape[1]),
            "feature_adapter_uses_target_labels": False,
        }
    n_components = min(
        int(components),
        int(clean_source.shape[0] - 1),
        int(clean_source.shape[1] - 1),
    )
    if n_components < 1:
        raise ValueError("high-dimensional SVD requires at least one component")
    projector = TruncatedSVD(n_components=n_components, random_state=0)
    projector.fit(np.asarray(clean_source, dtype=np.float64))
    return projector, {
        "feature_adapter": "truncated_svd_source_fit",
        "model_n_features": n_components,
        "svd_explained_variance_ratio_sum": float(
            np.sum(projector.explained_variance_ratio_)
        ),
        "feature_adapter_uses_target_labels": False,
    }


def adapt_features(
    projector: TruncatedSVD | None,
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if projector is None:
        return (
            np.asarray(X_source, dtype=np.float32),
            np.asarray(X_target, dtype=np.float32),
        )
    return (
        np.asarray(projector.transform(X_source), dtype=np.float32),
        np.asarray(projector.transform(X_target), dtype=np.float32),
    )


def evaluate_condition(
    predictor: Any,
    condition: Any,
    *,
    projector: TruncatedSVD | None,
    feature_metadata: dict[str, Any],
) -> dict[str, Any]:
    X_source, X_target = adapt_features(
        projector, condition.X_train, condition.X_target
    )
    started = time.perf_counter()
    if condition.task == "classification":
        prediction = predict_proba_native_or_ovr(
            predictor,
            X_source,
            np.asarray(condition.y_train),
            X_target,
            max_native_classes=10,
        )
        metrics = classification_metrics(
            np.asarray(condition.y_target),
            prediction.probabilities,
            prediction.classes,
        )
        strategy = prediction.strategy
        model_calls = prediction.model_calls
    else:
        center = float(np.mean(condition.y_train))
        scale = max(float(np.std(condition.y_train)), 1.0e-12)
        normalized = (np.asarray(condition.y_train, dtype=np.float64) - center) / scale
        raw = predictor.predict(
            X_source, normalized, X_target, task_type="Regression"
        )
        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        prediction_values = np.asarray(raw, dtype=np.float64).reshape(-1) * scale + center
        metrics = regression_metrics(
            np.asarray(condition.y_target),
            prediction_values,
            float(condition.metric_scale),
        )
        strategy = "native"
        model_calls = 1
    return {
        "cell_schema_version": 1,
        "suite": "seven_shift",
        "task": condition.task,
        "dataset": condition.dataset,
        "data_id": int(condition.data_id),
        "family": condition.family,
        "strength": float(condition.strength),
        "n_source": int(len(condition.y_train)),
        "n_target": int(len(condition.y_target)),
        "n_features": int(condition.X_train.shape[1]),
        "n_classes": (
            int(len(np.unique(condition.y_train)))
            if condition.task == "classification"
            else None
        ),
        "strategy": strategy,
        "model_calls": int(model_calls),
        "source_features_sha256": array_sha256(condition.X_train),
        "source_target_sha256": array_sha256(condition.y_train),
        "target_features_sha256": array_sha256(condition.X_target),
        "evaluation_target_sha256": array_sha256(condition.y_target),
        "adapter_metadata": condition.metadata,
        **feature_metadata,
        "seconds": time.perf_counter() - started,
        **metrics,
    }


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = {
        "classification": ("accuracy", "auc"),
        "regression": ("r2", "nmae", "nrmse"),
    }
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in cells:
        for metric in metrics[row["task"]]:
            grouped[
                (row["task"], row["family"], float(row["strength"]), metric)
            ].append(float(row[metric]))
    output = []
    for (task, family, strength, metric), values in sorted(grouped.items()):
        output.append(
            {
                "suite": "seven_shift",
                "task": task,
                "model": "limix_2m_ovr",
                "display_name": "LimiX-2M + OvR",
                "family": family,
                "intensity": strength,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "dataset_n": len(values),
            }
        )
    return output


def completion_audit(
    cells: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    by_task_dataset: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        by_task_dataset[(row["task"], int(row["data_id"]))].append(row)
    dataset_counts = Counter(task for task, _data_id in by_task_dataset)
    summary_counts = Counter(row["task"] for row in summary)
    checks: dict[str, Any] = {
        "dataset_counts": dict(dataset_counts),
        "dataset_coverage": all(
            dataset_counts[task] == expected
            for task, expected in EXPECTED_DATASETS.items()
        ),
        "condition_coverage": all(
            len(rows) == len(FAMILIES) * len(STRENGTHS)
            and len({(row["family"], float(row["strength"])) for row in rows})
            == len(FAMILIES) * len(STRENGTHS)
            for rows in by_task_dataset.values()
        ),
        "classification_native_binary": all(
            row["strategy"] == "native"
            and int(row["model_calls"]) == 1
            and int(row["n_classes"]) == 2
            for row in cells
            if row["task"] == "classification"
        ),
        "regression_native": all(
            row["strategy"] == "native" and int(row["model_calls"]) == 1
            for row in cells
            if row["task"] == "regression"
        ),
        "summary_row_counts": dict(summary_counts),
        "summary_coverage": (
            summary_counts["classification"] == 70
            and summary_counts["regression"] == 105
            and all(
                int(row["dataset_n"]) == EXPECTED_DATASETS[row["task"]]
                for row in summary
            )
        ),
        "official_ensemble": all(
            int(audit[f"{task}_pipelines"]) == expected
            for task, expected in EXPECTED_PIPELINES.items()
        ),
        "no_failures": int(audit["failures"]) == 0,
        "target_label_leakage_declared_false": (
            audit["target_labels_used_for_construction_or_prediction"] is False
            and audit["high_dimensional_adapter"]["uses_target_labels"] is False
        ),
    }
    ignored = {"dataset_counts", "summary_row_counts"}
    passed = all(value is True for key, value in checks.items() if key not in ignored)
    return {
        "paper_complete": passed,
        "checks": checks,
        "detail_rows": len(cells),
        "expected_detail_rows": 35 * 35 + 33 * 35,
    }


def main() -> None:
    args = arguments()
    limix_root = args.limix_root.resolve()
    model = args.model_path.resolve()
    protocol_root = args.protocol_root.resolve()
    output = args.output.resolve()
    # Keep every OpenML lookup inside the run artifact so the frozen protocol
    # can be executed from a complete, pre-seeded cache without network access.
    import openml

    openml_cache = output / "openml_cache"
    openml_cache.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(openml_cache)
    if not model.is_file():
        raise FileNotFoundError(model)
    checkpoint_hash = file_sha256(model)
    if checkpoint_hash != MODEL_SHA256:
        raise RuntimeError(
            f"unexpected checkpoint SHA256: {checkpoint_hash}; expected {MODEL_SHA256}"
        )
    protocol = load_frozen_protocol(protocol_root)
    classification_input_manifest: dict[str, Any] | None = None
    if args.task in {"classification", "both"}:
        classification_input_manifest = prepare_cc18_numeric_mirrors(
            protocol_root,
            output,
            [int(value) for value in protocol.CC18_BINARY_DATASETS],
        )
    frozen_hashes = protocol_hashes(protocol_root)
    sys.path.insert(0, str(limix_root))
    from inference.predictor import LimiXPredictor

    device = select_device(args.device)
    signature = {
        "protocol": "limix_2m_seven_shift_frozen_v1",
        "checkpoint_sha256": checkpoint_hash,
        "limix_git_revision": git_revision(limix_root),
        "protocol_git_revision": git_revision(protocol_root),
        "protocol_source_sha256": frozen_hashes,
        "device_type": device.type,
        "classification_pipelines": args.classification_pipelines,
        "regression_pipelines": args.regression_pipelines,
        "high_dimensional_threshold": args.high_dimensional_threshold,
        "high_dimensional_components": args.high_dimensional_components,
    }
    lock_run_signature(output, signature)
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    configs = {
        "classification": load_config(
            limix_root, "cls_default_noretrieval.json", args.classification_pipelines
        ),
        "regression": load_config(
            limix_root, "reg_default_noretrieval.json", args.regression_pipelines
        ),
    }
    failures: list[dict[str, Any]] = []
    for task in tasks:
        datasets = selected_datasets(protocol, task, args.dataset)
        if args.max_datasets > 0:
            datasets = datasets[: args.max_datasets]
        predictor = LimiXPredictor(
            device=device,
            model_path=str(model),
            inference_config=configs[task],
            mix_precision=device.type == "cuda",
            seed=0 if task == "classification" else 2718,
        )
        for position, dataset in enumerate(datasets, start=1):
            print(f"[{task} {position}/{len(datasets)}] materializing {dataset}", flush=True)
            try:
                condition_rows = protocol._seven_conditions(task, dataset)
                if len(condition_rows) != len(FAMILIES) * len(STRENGTHS):
                    raise RuntimeError(
                        f"expected 35 conditions, found {len(condition_rows)}"
                    )
                clean = next(
                    row
                    for row in condition_rows
                    if row.family == FAMILIES[0] and float(row.strength) == 0.0
                )
                projector, feature_metadata = fit_feature_adapter(
                    clean.X_train,
                    threshold=args.high_dimensional_threshold,
                    components=args.high_dimensional_components,
                )
                stale_failure = (
                    output / "failures" / f"{task}_{dataset}_load.json"
                )
                if stale_failure.exists():
                    stale_failure.unlink()
            except Exception as error:
                failure = {
                    "task": task,
                    "dataset": str(dataset),
                    "stage": "materialize",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                write_json(output / "failures" / f"{task}_{dataset}_load.json", failure)
                print(
                    f"  failed materialization for {dataset}: {error!r}",
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
                continue
            for condition in condition_rows:
                path = cell_path(
                    output,
                    task,
                    int(condition.data_id),
                    str(condition.family),
                    float(condition.strength),
                )
                if path.exists() and not args.force:
                    print(
                        f"  cached {condition.family} strength={condition.strength:g}",
                        flush=True,
                    )
                    continue
                try:
                    row = evaluate_condition(
                        predictor,
                        condition,
                        projector=projector,
                        feature_metadata=feature_metadata,
                    )
                    write_json(path, row)
                    stale_failure = output / "failures" / (
                        f"{task}_{condition.data_id}_{condition.family}_"
                        f"s{int(round(100 * condition.strength))}.json"
                    )
                    if stale_failure.exists():
                        stale_failure.unlink()
                    print(
                        f"  completed {condition.family} strength={condition.strength:g} "
                        f"strategy={row['strategy']} seconds={row['seconds']:.2f}",
                        flush=True,
                    )
                except Exception as error:
                    failure = {
                        "task": task,
                        "dataset": str(dataset),
                        "data_id": int(condition.data_id),
                        "stage": "inference",
                        "family": str(condition.family),
                        "strength": float(condition.strength),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    write_json(
                        output
                        / "failures"
                        / (
                            f"{task}_{condition.data_id}_{condition.family}_"
                            f"s{int(round(100 * condition.strength))}.json"
                        ),
                        failure,
                    )
                    if not args.continue_on_error:
                        raise

    cells = (
        [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (output / "cells").rglob("*.json")
        ]
        if (output / "cells").exists()
        else []
    )
    cells.sort(
        key=lambda row: (
            row["task"],
            int(row["data_id"]),
            row["family"],
            float(row["strength"]),
        )
    )
    summary = summarize(cells)
    write_csv(output / "detail.csv", cells)
    write_csv(output / "summary_long.csv", summary)
    audit = {
        "method": "LimiX-2M + OvR",
        "checkpoint": str(model),
        "checkpoint_sha256": checkpoint_hash,
        "limix_git_revision": git_revision(limix_root),
        "expected_limix_tag": "V1.1.0",
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "classification_pipelines": args.classification_pipelines,
        "regression_pipelines": args.regression_pipelines,
        "protocol_git_revision": git_revision(protocol_root),
        "protocol_source_sha256": frozen_hashes,
        "classification_input_adapter": classification_input_manifest,
        "families": list(FAMILIES),
        "strengths": list(STRENGTHS),
        "expected_datasets": EXPECTED_DATASETS,
        "target_labels_used_for_construction_or_prediction": False,
        "target_labels_used_for_scoring": True,
        "high_dimensional_adapter": {
            "threshold": args.high_dimensional_threshold,
            "maximum_components": args.high_dimensional_components,
            "method": "clean-source-fit TruncatedSVD after original-space shift",
            "uses_target_labels": False,
            "adapted_cells": sum(
                row.get("feature_adapter") == "truncated_svd_source_fit"
                for row in cells
            ),
        },
        "completed_cells": len(cells),
        "failures": len(failures),
        "total_model_calls": int(sum(int(row["model_calls"]) for row in cells)),
        "total_inference_seconds": float(sum(float(row["seconds"]) for row in cells)),
    }
    write_json(output / "audit.json", audit)
    completion = completion_audit(cells, summary, audit)
    write_json(output / "completion_audit.json", completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
