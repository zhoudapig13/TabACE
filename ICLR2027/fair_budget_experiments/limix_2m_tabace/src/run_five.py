#!/usr/bin/env python3
"""Run source-tuned LimiX + TabACE Shared/Specific on the five perturbations.

Policies are selected independently for each dataset from an inner split of the
labeled source partition.  The fixed outer target labels are not passed to any
Context constructor or policy-selection function.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPOSITORY = EXPERIMENT.parents[1]
BASE_SRC = REPOSITORY / "fair_budget_experiments" / "limix_2m_ovr" / "src"
SCRIPTS = REPOSITORY / "scripts"
for path in (HERE, BASE_SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from context_builders import (  # noqa: E402
    classification_contexts,
    regression_global_contexts,
)
from contextual_predictor import (  # noqa: E402
    predict_context_bank_classification,
)
from limix_compat import predict_proba_native_or_ovr  # noqa: E402
from policy_selection import (  # noqa: E402
    ContextPolicy,
    PredictionRecord,
    fuse,
    select_policy,
)
from protocol import (  # noqa: E402
    CC18_SEED,
    CC18_SUITE_ID,
    CTR23_CAP_SEED,
    CTR23_SUITE_ID,
    FAMILIES,
    MANIPULATION_SEED,
    STRENGTHS,
    filter_specs,
    load_cc18_split,
    load_ctr23_split,
    manipulate_source_context,
    suite_specs,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classification-pipelines", type=int, default=4)
    parser.add_argument("--regression-pipelines", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--high-dimensional-threshold", type=int, default=1024)
    parser.add_argument("--high-dimensional-components", type=int, default=256)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(root: Path, name: str, count: int) -> list[dict[str, Any]]:
    value = json.loads((root / "config" / name).read_text(encoding="utf-8"))
    if not 1 <= count <= len(value):
        raise ValueError(f"pipeline count for {name} must be in [1, {len(value)}]")
    return value[:count]


def _count(n_source: int, fraction: float, minimum: int = 16) -> int:
    return min(n_source, max(minimum, int(round(n_source * fraction))))


def classification_specs(n_source: int) -> dict[str, tuple[str, int]]:
    return {
        "knn/q25": ("knn", _count(n_source, 0.25)),
        "knn/q50": ("knn", _count(n_source, 0.50)),
        "balanced/q50": ("balanced", _count(n_source, 0.50)),
        "prototype/q50": ("prototype", _count(n_source, 0.50)),
        "clean/q50": ("clean", _count(n_source, 0.50)),
        "boundary/q50": ("boundary", _count(n_source, 0.50)),
        "minority/q50": ("minority", _count(n_source, 0.50)),
        "balanced/q75": ("balanced", _count(n_source, 0.75)),
        "prototype/q75": ("prototype", _count(n_source, 0.75)),
        "clean/q75": ("clean", _count(n_source, 0.75)),
        "boundary/q75": ("boundary", _count(n_source, 0.75)),
        "minority/q75": ("minority", _count(n_source, 0.75)),
        "clean/q875": ("clean", _count(n_source, 0.875)),
    }


def regression_specs(n_source: int) -> dict[str, tuple[str, int]]:
    counts = sorted({min(n_source, value) for value in (32, 64, 128, 256)})
    output = {f"knn/k{k}": ("knn", k) for k in counts}
    largest = counts[-1]
    for constructor in ("clean", "prototype", "bracket", "trimmed"):
        output[f"{constructor}/k{largest}"] = (constructor, largest)
    return output


def adapt_features(
    X_fit: np.ndarray,
    X_context: np.ndarray,
    X_query: np.ndarray,
    *,
    threshold: int,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if threshold <= 0 or X_fit.shape[1] <= threshold:
        return (
            np.asarray(X_context, dtype=np.float32),
            np.asarray(X_query, dtype=np.float32),
            {"feature_adapter": "identity", "model_n_features": int(X_fit.shape[1])},
        )
    count = min(components, X_fit.shape[0] - 1, X_fit.shape[1] - 1)
    projector = TruncatedSVD(n_components=count, random_state=seed)
    projector.fit(np.asarray(X_fit, dtype=np.float64))
    return (
        np.asarray(projector.transform(X_context), dtype=np.float32),
        np.asarray(projector.transform(X_query), dtype=np.float32),
        {
            "feature_adapter": "truncated_svd_source_fit",
            "model_n_features": int(count),
            "feature_adapter_uses_target_labels": False,
        },
    )


def inner_indices(task: str, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if len(counts) and int(np.min(counts)) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            return next(splitter.split(indices, y))
    splitter = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    return next(splitter.split(indices))


def classification_base(predictor: Any, X: np.ndarray, y: np.ndarray, Q: np.ndarray):
    return predict_proba_native_or_ovr(
        predictor, X, y, Q, max_native_classes=10
    )


def regression_base(
    predictor: Any, X: np.ndarray, y: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(y))
    std = max(float(np.std(y)), 1.0e-12)
    normalized = (np.asarray(y, dtype=np.float64) - mean) / std
    raw = predictor.predict(X, normalized, Q, task_type="Regression")
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    value = np.asarray(raw, dtype=np.float64).reshape(-1) * std + mean
    return value, mean, std


def prediction_record(
    *,
    task: str,
    predictor: Any,
    X_fit: np.ndarray,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    family: str,
    strength: int,
    threshold: int,
    components: int,
    batch_size: int,
    seed: int,
) -> tuple[PredictionRecord, dict[str, Any]]:
    X_model, Q_model, adapter = adapt_features(
        X_fit,
        X_context,
        X_query,
        threshold=threshold,
        components=components,
        seed=seed,
    )
    if task == "classification":
        base = classification_base(predictor, X_model, y_context, Q_model)
        specs = classification_specs(len(y_context))
        contexts = classification_contexts(
            X_model,
            y_context,
            Q_model,
            base.probabilities,
            base.classes,
            specs,
        )
        members, classes, strategy, calls = predict_context_bank_classification(
            predictor,
            X_model,
            y_context,
            Q_model,
            contexts,
            max_native_classes=10,
            batch_size=batch_size,
        )
        if not np.array_equal(classes, base.classes):
            raise RuntimeError("Base and Context class order differ")
        return (
            PredictionRecord(
                family=family,
                strength=strength,
                target=np.asarray(y_query),
                base=base.probabilities,
                members=members,
                classes=classes,
            ),
            {**adapter, "strategy": strategy, "model_calls_per_context": calls},
        )
    base, mean, std = regression_base(predictor, X_model, y_context, Q_model)
    contexts = regression_global_contexts(X_model, y_context, Q_model)
    members: dict[str, np.ndarray] = {}
    for name, indices in contexts.items():
        context_y = np.asarray(y_context, dtype=np.float64)[indices]
        context_mean = float(np.mean(context_y))
        context_std = max(float(np.std(context_y)), 1.0e-12)
        normalized = (context_y - context_mean) / context_std
        raw = predictor.predict(
            X_model[indices],
            normalized,
            Q_model,
            task_type="Regression",
        )
        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        members[name] = (
            np.asarray(raw, dtype=np.float64).reshape(-1) * context_std + context_mean
        )
    return (
        PredictionRecord(
            family=family,
            strength=strength,
            target=np.asarray(y_query, dtype=np.float64),
            base=base,
            members=members,
        ),
        {**adapter, "strategy": "native", "model_calls_per_context": 1},
    )


def select_dataset_policies(
    *,
    task: str,
    predictor: Any,
    X_source: np.ndarray,
    y_source: np.ndarray,
    threshold: int,
    components: int,
    batch_size: int,
    seed: int,
) -> tuple[ContextPolicy, dict[str, ContextPolicy], dict[str, Any]]:
    train_indices, calibration_indices = inner_indices(task, y_source, seed)
    X_train = np.asarray(X_source)[train_indices]
    y_train = np.asarray(y_source)[train_indices]
    X_calibration = np.asarray(X_source)[calibration_indices]
    y_calibration = np.asarray(y_source)[calibration_indices]
    records: list[PredictionRecord] = []
    conditions = [("original", 0)] + [
        (family, int(strength))
        for family in FAMILIES
        for strength in STRENGTHS
        if strength > 0
    ]
    started = time.perf_counter()
    for position, (family, strength) in enumerate(conditions, start=1):
        manipulated = manipulate_source_context(
            task,
            X_train,
            y_train,
            family=family,
            rho=strength / 100.0,
            seed=MANIPULATION_SEED + strength,
            shift_scale=6.0 if task == "classification" else 10.0,
            label_scale=1.0,
        )
        record, _ = prediction_record(
            task=task,
            predictor=predictor,
            X_fit=X_train,
            X_context=manipulated.X_context,
            y_context=manipulated.y_context,
            X_query=X_calibration,
            y_query=y_calibration,
            family=family,
            strength=strength,
            threshold=threshold,
            components=components,
            batch_size=batch_size,
            seed=seed,
        )
        records.append(record)
        print(
            f"    validation {position}/{len(conditions)} {family} rho={strength}",
            flush=True,
        )
    member_names = tuple(records[0].members)
    perturbed = [record for record in records if record.strength > 0]
    shared = select_policy(
        perturbed,
        task=task,
        member_names=member_names,
        selection_scope="dataset_shared_nonzero_perturbations",
        max_members=3,
    )
    specific = {}
    for family in FAMILIES:
        family_records = [record for record in perturbed if record.family == family]
        specific[family] = select_policy(
            family_records,
            task=task,
            member_names=member_names,
            selection_scope=f"dataset_family_specific:{family}",
            reference=shared,
            max_members=3,
        )
    metadata = {
        "selection_schema_version": 1,
        "task": task,
        "inner_train_rows": int(len(train_indices)),
        "inner_calibration_rows": int(len(calibration_indices)),
        "inner_train_indices": train_indices.tolist(),
        "inner_calibration_indices": calibration_indices.tolist(),
        "selection_conditions": [list(item) for item in conditions],
        "candidate_members": list(member_names),
        "target_labels_used_for_selection": False,
        "source_calibration_labels_used_for_selection": True,
        "seconds": time.perf_counter() - started,
    }
    return shared, specific, metadata


def classification_metrics(
    target: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-12, None)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = classes[np.argmax(probability, axis=1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = float(balanced_accuracy_score(target, predicted))
        values = []
        for index, label in enumerate(classes):
            binary = (np.asarray(target) == label).astype(int)
            if np.any(binary == 1) and np.any(binary == 0):
                values.append(float(roc_auc_score(binary, probability[:, index])))
    return {
        "accuracy": float(np.mean(predicted == target)),
        "balanced_accuracy": balanced,
        "auc": float(np.mean(values)) if values else 0.5,
    }


def regression_metrics(
    target: np.ndarray, prediction: np.ndarray, source_scale: float
) -> dict[str, float]:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    denominator = max(float(np.sum((target - np.mean(target)) ** 2)), 1.0e-12)
    return {
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "nrmse": float(np.sqrt(np.mean(error**2))) / source_scale,
        "nmae": float(np.mean(np.abs(error))) / source_scale,
    }


def evaluate_outer_cell(
    *,
    task: str,
    predictor: Any,
    split: Any,
    family: str,
    strength: int,
    shared: ContextPolicy,
    specific: ContextPolicy,
    threshold: int,
    components: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    manipulated = manipulate_source_context(
        task,
        split.X_source,
        split.y_source,
        family=family,
        rho=strength / 100.0,
        seed=MANIPULATION_SEED + strength,
        shift_scale=6.0 if task == "classification" else 10.0,
        label_scale=1.0,
    )
    # Prediction generation has no access to split.y_target.  It is supplied to
    # the metric block only after both frozen policies have produced outputs.
    record, inference = prediction_record(
        task=task,
        predictor=predictor,
        X_fit=split.X_source,
        X_context=manipulated.X_context,
        y_context=manipulated.y_context,
        X_query=split.X_target,
        y_query=np.zeros(len(split.X_target)),
        family=family,
        strength=strength,
        threshold=threshold,
        components=components,
        batch_size=batch_size,
        seed=seed,
    )
    shared_prediction = fuse(record.base, record.members, shared)
    specific_prediction = fuse(record.base, record.members, specific)
    if task == "classification":
        metrics = {
            "base": classification_metrics(split.y_target, record.base, record.classes),
            "shared": classification_metrics(split.y_target, shared_prediction, record.classes),
            "specific": classification_metrics(split.y_target, specific_prediction, record.classes),
        }
    else:
        scale = max(float(np.std(split.y_source)), 1.0e-12)
        metrics = {
            "base": regression_metrics(split.y_target, record.base, scale),
            "shared": regression_metrics(split.y_target, shared_prediction, scale),
            "specific": regression_metrics(split.y_target, specific_prediction, scale),
        }
    return {
        "cell_schema_version": 1,
        "task": task,
        "task_id": int(split.spec.task_id),
        "data_id": int(split.spec.data_id),
        "dataset": split.spec.name,
        "family": family,
        "rho": int(strength),
        "n_source": int(len(split.y_source)),
        "n_target": int(len(split.y_target)),
        "n_features": int(split.X_source.shape[1]),
        "affected_rows": int(len(manipulated.affected_rows)),
        "metrics": metrics,
        "shared_policy": shared.to_dict(),
        "specific_policy": specific.to_dict(),
        "target_labels_used_for_construction": False,
        "target_labels_used_for_selection": False,
        **inference,
    }


def main() -> None:
    args = arguments()
    root = args.limix_root.resolve()
    model_path = args.model_path.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(root))
    from inference.predictor import LimiXPredictor

    import openml

    openml.config.set_root_cache_directory(
        Path("/root/autodl-tmp/limix_tabace/results/paper_run/openml_cache")
    )
    requested = {item.strip().casefold() for item in args.dataset.split(",") if item.strip()}
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    configs = {
        "classification": load_config(root, "cls_default_noretrieval.json", args.classification_pipelines),
        "regression": load_config(root, "reg_default_noretrieval.json", args.regression_pipelines),
    }
    for task in tasks:
        suite_id = CC18_SUITE_ID if task == "classification" else CTR23_SUITE_ID
        specs = filter_specs(suite_specs(suite_id, requested), requested)
        if args.max_datasets > 0:
            specs = specs[: args.max_datasets]
        seed = CC18_SEED if task == "classification" else CTR23_CAP_SEED
        predictor = LimiXPredictor(
            device=torch.device("cuda"),
            model_path=str(model_path),
            inference_config=configs[task],
            mix_precision=True,
            seed=seed,
        )
        for position, spec in enumerate(specs, start=1):
            print(f"[{task} {position}/{len(specs)}] {spec.name} data_id={spec.data_id}", flush=True)
            dataset_root = output / task / f"dataset_{spec.data_id}"
            policy_path = dataset_root / "selection.json"
            try:
                split = load_cc18_split(spec) if task == "classification" else load_ctr23_split(spec)
                if policy_path.exists():
                    saved = json.loads(policy_path.read_text(encoding="utf-8"))
                    shared = ContextPolicy.from_dict(saved["shared"])
                    specific = {
                        family: ContextPolicy.from_dict(value)
                        for family, value in saved["specific"].items()
                    }
                    print("  loaded frozen source-selected policies", flush=True)
                else:
                    shared, specific, metadata = select_dataset_policies(
                        task=task,
                        predictor=predictor,
                        X_source=split.X_source,
                        y_source=split.y_source,
                        threshold=args.high_dimensional_threshold,
                        components=args.high_dimensional_components,
                        batch_size=args.batch_size,
                        seed=seed,
                    )
                    write_json(
                        policy_path,
                        {
                            **metadata,
                            "data_id": int(spec.data_id),
                            "dataset": spec.name,
                            "shared": shared.to_dict(),
                            "specific": {family: policy.to_dict() for family, policy in specific.items()},
                        },
                    )
                    print(f"  froze Shared {shared.to_dict()}", flush=True)
                    for family, policy in specific.items():
                        print(f"  froze Specific[{family}] {policy.to_dict()}", flush=True)
                conditions = [("original", 0)] + [
                    (family, int(strength))
                    for family in FAMILIES
                    for strength in STRENGTHS
                    if strength > 0
                ]
                for family, strength in conditions:
                    path = dataset_root / "cells" / f"{family}_r{strength}.json"
                    if path.exists():
                        print(f"  cached outer {family} rho={strength}", flush=True)
                        continue
                    started = time.perf_counter()
                    row = evaluate_outer_cell(
                        task=task,
                        predictor=predictor,
                        split=split,
                        family=family,
                        strength=strength,
                        shared=shared,
                        specific=shared if family == "original" else specific[family],
                        threshold=args.high_dimensional_threshold,
                        components=args.high_dimensional_components,
                        batch_size=args.batch_size,
                        seed=seed,
                    )
                    row["seconds"] = time.perf_counter() - started
                    write_json(path, row)
                    print(
                        f"  completed outer {family} rho={strength} seconds={row['seconds']:.2f}",
                        flush=True,
                    )
            except Exception as error:
                failure = {
                    "task": task,
                    "data_id": int(spec.data_id),
                    "dataset": spec.name,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(dataset_root / "failure.json", failure)
                print(json.dumps(failure, indent=2), flush=True)
                if not args.continue_on_error:
                    raise


if __name__ == "__main__":
    main()
