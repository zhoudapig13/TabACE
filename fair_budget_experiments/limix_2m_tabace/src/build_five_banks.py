#!/usr/bin/env python3
"""Capture LimiX prediction banks for the canonical five-perturbation benchmark.

The saved schema is intentionally identical to the TabICL/TabPFN/TabDPT bank
schema consumed by ``unified_context_ensemble``.  Context construction uses
only source labels and unlabeled query features.  Target labels are stored only
so the later dataset-level OOF evaluator can score its held-out datasets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = HERE.parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabace-root", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("classification", "regression", "both"), default="both"
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--classification-pipelines", type=int, default=4)
    parser.add_argument("--regression-pipelines", type=int, default=8)
    parser.add_argument("--high-dimensional-threshold", type=int, default=1024)
    parser.add_argument("--high-dimensional-components", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_config(root: Path, name: str, count: int) -> list[dict[str, Any]]:
    value = json.loads((root / "config" / name).read_text(encoding="utf-8"))
    if not 1 <= count <= len(value):
        raise ValueError(f"pipeline count for {name} must be in [1, {len(value)}]")
    return value[:count]


def feature_adapter(
    X_source: np.ndarray,
    X_query: np.ndarray,
    *,
    threshold: int,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = np.asarray(X_source, dtype=np.float32)
    query = np.asarray(X_query, dtype=np.float32)
    if threshold <= 0 or source.shape[1] <= threshold:
        return source, query, {
            "feature_adapter": "identity",
            "model_n_features": int(source.shape[1]),
        }
    count = min(int(components), source.shape[0] - 1, source.shape[1] - 1)
    projector = TruncatedSVD(n_components=count, random_state=seed)
    source_model = projector.fit_transform(np.asarray(source, dtype=np.float64))
    query_model = projector.transform(np.asarray(query, dtype=np.float64))
    return (
        np.asarray(source_model, dtype=np.float32),
        np.asarray(query_model, dtype=np.float32),
        {
            "feature_adapter": "truncated_svd_source_fit",
            "model_n_features": int(count),
            "feature_adapter_uses_target_labels": False,
        },
    )


def source_numeric_features(
    X_source: np.ndarray, X_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a source-fitted ordinal encoding to non-numeric OpenML columns."""

    source = np.asarray(X_source)
    query = np.asarray(X_query)
    if source.ndim != 2 or query.ndim != 2 or source.shape[1] != query.shape[1]:
        raise ValueError("source/query feature matrices are not aligned")
    source_columns: list[np.ndarray] = []
    query_columns: list[np.ndarray] = []
    for position in range(source.shape[1]):
        source_column = source[:, position]
        query_column = query[:, position]
        try:
            source_value = np.asarray(source_column, dtype=np.float64)
            query_value = np.asarray(query_column, dtype=np.float64)
        except (TypeError, ValueError):
            normalized_source = np.asarray(
                ["<missing>" if value is None else str(value) for value in source_column]
            )
            normalized_query = np.asarray(
                ["<missing>" if value is None else str(value) for value in query_column]
            )
            vocabulary = {
                value: index
                for index, value in enumerate(sorted(np.unique(normalized_source).tolist()))
            }
            source_value = np.asarray(
                [vocabulary[value] for value in normalized_source], dtype=np.float64
            )
            query_value = np.asarray(
                [vocabulary.get(value, -1) for value in normalized_query], dtype=np.float64
            )
        finite = source_value[np.isfinite(source_value)]
        replacement = float(np.median(finite)) if len(finite) else 0.0
        source_columns.append(
            np.where(np.isfinite(source_value), source_value, replacement)
        )
        query_columns.append(
            np.where(np.isfinite(query_value), query_value, replacement)
        )
    return (
        np.column_stack(source_columns).astype(np.float32),
        np.column_stack(query_columns).astype(np.float32),
    )


def regression_prediction(
    predictor: Any,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    target = np.asarray(y_source, dtype=np.float64)
    mean = float(np.mean(target))
    std = max(float(np.std(target)), 1.0e-12)
    raw = predictor.predict(
        np.asarray(X_source, dtype=np.float32),
        (target - mean) / std,
        np.asarray(X_query, dtype=np.float32),
        task_type="Regression",
    )
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    return np.asarray(raw, dtype=np.float64).reshape(-1) * std + mean


def atomic_save(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def load_benchmark_protocol(repository: Path):
    path = (
        repository
        / "fair_budget_experiments"
        / "limix_2m_ovr"
        / "src"
        / "protocol.py"
    )
    spec = importlib.util.spec_from_file_location("limix_five_benchmark_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_classification(
    args: argparse.Namespace,
    predictor: Any,
    *,
    requested: set[str],
) -> None:
    from limix_compat import predict_proba_native_or_ovr
    from run_apa_tabicl import apply_public_shift_to_split
    from run_cc18_tabdpt_no_shift import reference_pseudo_labels
    from run_coce_no_shift_context_ablation import (
        CONTEXT_TYPES,
        _condition_beta,
        _conditions,
        _predict_context_types,
    )
    from tabicl_zero_oracle_common import CC18Protocol, selected_datasets

    output_root = args.output / "classification"
    protocol = CC18Protocol(seed=0, n_samples=384, train_size=256, test_size=128)
    benchmark = load_benchmark_protocol(args.tabace_root.resolve())
    benchmark_specs = {
        int(value.data_id): value
        for value in benchmark.suite_specs(benchmark.CC18_SUITE_ID)
    }
    datasets = selected_datasets("all")
    if requested:
        datasets = [
            value
            for value in datasets
            if value.name.casefold() in requested or str(value.data_id) in requested
        ]
    for position, dataset in enumerate(datasets, start=1):
        output = output_root / f"dataset_{dataset.data_id}.npz"
        if output.exists() and not args.force:
            print(
                f"[classification {position}/{len(datasets)}] cached {dataset.name}",
                flush=True,
            )
            continue
        started = time.perf_counter()
        try:
            split = benchmark.load_cc18_split(benchmark_specs[int(dataset.data_id)])
            X_source = np.asarray(split.X_source, dtype=np.float32)
            X_target = np.asarray(split.X_target, dtype=np.float32)
            y_source = np.asarray(split.y_source)
            y_target = np.asarray(split.y_target)
            pseudo_target = reference_pseudo_labels(
                X_source, y_source, X_target
            )
            conditions = _conditions(dataset.source)
            query_blocks: list[np.ndarray] = []
            for problem, _rho, condition in conditions:
                if problem == "original":
                    X_query = np.asarray(X_target, dtype=np.float32)
                else:
                    X_query, _ = apply_public_shift_to_split(
                        condition,
                        X_source,
                        y_source,
                        X_target,
                        pseudo_target,
                        protocol.seed,
                    )
                query_blocks.append(np.asarray(X_query, dtype=np.float32))
            lengths = [len(value) for value in query_blocks]
            bounds = np.cumsum([0, *lengths])
            X_model, Q_all, adapter = feature_adapter(
                X_source,
                np.vstack(query_blocks),
                threshold=args.high_dimensional_threshold,
                components=args.high_dimensional_components,
                seed=protocol.seed,
            )
            predicted_all = predict_proba_native_or_ovr(
                predictor,
                X_model,
                y_source,
                Q_all,
                max_native_classes=10,
            )
            classes = np.asarray(predicted_all.classes)
            probability_blocks = [
                predicted_all.probabilities[bounds[index] : bounds[index + 1]]
                for index in range(len(lengths))
            ]
            query_model_blocks = [
                Q_all[bounds[index] : bounds[index + 1]]
                for index in range(len(lengths))
            ]
            base_blocks: list[np.ndarray] = []
            member_blocks: list[np.ndarray] = []
            strategies: list[str] = []
            model_calls: list[int] = []
            adapters: list[str] = []
            for cell_position, ((problem, _rho, _condition), Q_model, probability) in enumerate(
                zip(conditions, query_model_blocks, probability_blocks), start=1
            ):
                members, failures = _predict_context_types(
                    X_model,
                    y_source,
                    Q_model,
                    probability,
                    classes,
                    device="cpu",
                    seed=protocol.seed + int(dataset.data_id),
                    subspace_policy=None,
                    subspace_prediction=None,
                )
                if any(failures.values()):
                    raise RuntimeError(f"Context construction failures: {failures}")
                base_blocks.append(np.asarray(probability, dtype=np.float32))
                member_blocks.append(
                    np.stack(
                        [np.asarray(members[name], dtype=np.float32) for name in CONTEXT_TYPES],
                        axis=1,
                    )
                )
                strategies.append(str(predicted_all.strategy))
                model_calls.append(int(predicted_all.model_calls))
                adapters.append(str(adapter["feature_adapter"]))
                print(
                    f"  {dataset.name} {cell_position}/{len(conditions)} "
                    f"{problem} rho={int(_rho)}",
                    flush=True,
                )
            calibration = {
                "weights": {name: 1.0 / len(CONTEXT_TYPES) for name in CONTEXT_TYPES},
                "alpha": 0.5,
                "selection_scope": "deferred_dataset_level_oof",
                "target_labels_used": False,
            }
            atomic_save(
                output,
                y_target=np.asarray(y_target),
                classes=classes,
                base=np.stack(base_blocks),
                members=np.stack(member_blocks),
                member_names=np.asarray(CONTEXT_TYPES),
                problem=np.asarray([value[0] for value in conditions]),
                rho=np.asarray([value[1] for value in conditions], dtype=np.int64),
                beta=np.asarray(
                    [_condition_beta(value[0], value[2]) for value in conditions],
                    dtype=np.float64,
                ),
                condition=np.asarray([value[2] for value in conditions]),
                gate_applied=np.ones(len(conditions), dtype=bool),
                fusion_calibration=np.asarray(json.dumps(calibration, sort_keys=True)),
                prediction_strategy=np.asarray(strategies),
                model_calls=np.asarray(model_calls, dtype=np.int64),
                feature_adapter=np.asarray(adapters),
                selection_uses_target_labels=np.asarray(False),
                construction_uses_target_labels=np.asarray(False),
            )
            print(
                f"[classification {position}/{len(datasets)}] built {dataset.name} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
        except Exception as error:
            failure = output.with_suffix(".failure.json")
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                json.dumps(
                    {
                        "dataset": dataset.name,
                        "data_id": int(dataset.data_id),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"FAILED {dataset.name}: {error!r}", flush=True)
            if not args.continue_on_error:
                raise


def build_regression(
    args: argparse.Namespace,
    predictor: Any,
    *,
    requested: set[str],
) -> None:
    from ctr23_datasets import CTR23_BY_NAME, CTR23_SPECS
    from ctr23_protocol import load_capped_ctr23_fold
    from run_coce_regression_context_ablation import CONTEXT_TYPES, context_predictions
    from run_ctr23_tabdpt import (
        LOCKED_CAP_SEED,
        LOCKED_MANIPULATION_SEED,
        LOCKED_SOURCE_CAP,
        LOCKED_TARGET_CAP,
    )
    from run_regression_train_manipulation_multidataset import manipulate_context

    output_root = args.output / "regression"
    names = [spec.name for spec in CTR23_SPECS]
    if requested:
        names = [
            name
            for name in names
            if name.casefold() in requested
            or str(CTR23_BY_NAME[name].data_id) in requested
        ]
    for position, name in enumerate(names, start=1):
        data_id = int(CTR23_BY_NAME[name].data_id)
        output = output_root / f"dataset_{data_id}.npz"
        if output.exists() and not args.force:
            print(f"[regression {position}/{len(names)}] cached {name}", flush=True)
            continue
        started = time.perf_counter()
        try:
            fold = load_capped_ctr23_fold(
                name,
                source_cap=LOCKED_SOURCE_CAP,
                target_cap=LOCKED_TARGET_CAP,
                seed=LOCKED_CAP_SEED,
            )
            scale = max(float(np.std(fold.y_source)), 1.0e-12)
            conditions = [("original", 0)] + [
                (problem, rho)
                for problem in ("shortcut", "boundary", "rankband", "local", "mixed")
                for rho in (20, 50, 80, 100)
            ]
            base_blocks: list[np.ndarray] = []
            member_blocks: list[np.ndarray] = []
            adapters: list[str] = []
            clean_prediction: np.ndarray | None = None
            for cell_position, (problem, rho) in enumerate(conditions, start=1):
                if problem == "original":
                    X_context = np.asarray(fold.X_source)
                    y_context = np.asarray(fold.y_source)
                else:
                    X_context, y_context, _affected, _top = manipulate_context(
                        fold.X_source,
                        fold.y_source,
                        problem=problem,
                        rho=rho / 100.0,
                        seed=LOCKED_MANIPULATION_SEED + rho,
                        shift_scale=10.0,
                        label_scale=1.0,
                    )
                X_model, Q_model, adapter = feature_adapter(
                    X_context,
                    fold.X_target,
                    threshold=args.high_dimensional_threshold,
                    components=args.high_dimensional_components,
                    seed=LOCKED_CAP_SEED,
                )
                base = regression_prediction(
                    predictor, X_model, y_context, Q_model
                )
                if clean_prediction is None:
                    clean_prediction = np.asarray(base, dtype=np.float64)
                members = context_predictions(
                    np.asarray(X_context),
                    np.asarray(y_context),
                    np.asarray(fold.X_target),
                    np.asarray(base),
                    historical_clean_prediction=clean_prediction,
                )
                base_blocks.append(np.asarray(base, dtype=np.float64))
                member_blocks.append(
                    np.stack(
                        [np.asarray(members[member], dtype=np.float64) for member in CONTEXT_TYPES],
                        axis=1,
                    )
                )
                adapters.append(str(adapter["feature_adapter"]))
                print(
                    f"  {name} {cell_position}/{len(conditions)} {problem} rho={rho}",
                    flush=True,
                )
            calibration = {
                "weights": {member: 1.0 / len(CONTEXT_TYPES) for member in CONTEXT_TYPES},
                "foundation_weight": 0.5,
                "selection_scope": "deferred_dataset_level_oof",
                "selection_uses_target_labels": False,
            }
            atomic_save(
                output,
                y_target=np.asarray(fold.y_target, dtype=np.float64),
                scale=np.asarray(scale, dtype=np.float64),
                base=np.stack(base_blocks),
                members=np.stack(member_blocks),
                member_names=np.asarray(CONTEXT_TYPES),
                problem=np.asarray([value[0] for value in conditions]),
                rho=np.asarray([value[1] for value in conditions], dtype=np.int64),
                source_calibration=np.asarray(json.dumps(calibration, sort_keys=True)),
                feature_adapter=np.asarray(adapters),
                selection_uses_target_labels=np.asarray(False),
                construction_uses_target_labels=np.asarray(False),
                clean_reference_is_extra_information=np.asarray(True),
            )
            print(
                f"[regression {position}/{len(names)}] built {name} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
        except Exception as error:
            failure = output.with_suffix(".failure.json")
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                json.dumps(
                    {
                        "dataset": name,
                        "data_id": data_id,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"FAILED {name}: {error!r}", flush=True)
            if not args.continue_on_error:
                raise


def main() -> None:
    args = arguments()
    repository = args.tabace_root.resolve()
    limix_root = args.limix_root.resolve()
    model_path = args.model_path.resolve()
    for path in (
        repository,
        repository / "src",
        repository / "scripts",
        repository / "experiment6" / "scripts",
        repository / "experiment7" / "scripts",
        repository / "experiment8" / "scripts",
        repository / "experiment10" / "scripts",
        repository / "experiment11" / "scripts",
        limix_root,
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import openml
    from inference.predictor import LimiXPredictor

    openml.config.set_root_cache_directory(
        Path("/root/autodl-tmp/limix_tabace/results/paper_run/openml_cache")
    )
    requested = {
        item.strip().casefold() for item in args.dataset.split(",") if item.strip()
    }
    configs = {
        "classification": load_config(
            limix_root, "cls_default_noretrieval.json", args.classification_pipelines
        ),
        "regression": load_config(
            limix_root, "reg_default_noretrieval.json", args.regression_pipelines
        ),
    }
    tasks = (
        ("classification", "regression") if args.task == "both" else (args.task,)
    )
    for task in tasks:
        predictor = LimiXPredictor(
            device=torch.device("cuda"),
            model_path=str(model_path),
            inference_config=configs[task],
            mix_precision=True,
            seed=0,
        )
        if task == "classification":
            build_classification(args, predictor, requested=requested)
        else:
            build_regression(args, predictor, requested=requested)


if __name__ == "__main__":
    main()
