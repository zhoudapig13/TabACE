#!/usr/bin/env python3
"""Capture label-safe LimiX + TabACE banks on the frozen seven-shift suite.

The official LimiX forward pass supplies the Base prediction.  Shared and
family-specific candidates are built only from the labeled source Context,
unlabeled query features, and the frozen Base prediction.  Target labels are
saved exclusively for the later dataset-level OOF evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPTS = ROOT / "scripts"
for path in (
    ROOT,
    ROOT / "src",
    ROOT / "experiment6" / "scripts",
    ROOT / "experiment7" / "scripts",
    ROOT / "experiment8" / "scripts",
    ROOT / "experiment11" / "scripts",
    ROOT / "experiment13",
    ROOT / "experiment14",
    ROOT / "experiment15",
    ROOT / "experiment17",
    ROOT / "experiment28",
    ROOT / "fair_budget_experiments" / "baseline_shift_comparison" / "src",
    HERE,
    SCRIPTS,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from context_builders import classification_contexts  # noqa: E402
from limix_compat import predict_proba_native_or_ovr  # noqa: E402
from run_coce_no_shift_context_ablation import (  # noqa: E402
    _kernel_context_prediction,
)
import run_direct_baselines as frozen  # noqa: E402


FAMILIES = frozen.SEVEN_SHIFTS
STRENGTHS = frozen.SEVEN_STRENGTHS
CLASS_SHARED = (
    "knn/k20",
    "balanced/k20",
    "prototype/k20",
    "clean/k20",
    "boundary/k20",
)
CLASS_SPECIFIC = {
    "covariate": ("knn/k8", "knn/k20", "knn/k48", "clean/k20", "prototype/k20"),
    "label": ("prior/em", "prior/pcc", "minority/k20", "balanced/k20", "boundary/k20"),
    "concept": ("clean/k8", "clean/k20", "clean/k48", "boundary/k20", "knn/k8"),
    "selection": ("balanced/k8", "balanced/k20", "knn/k8", "knn/k20", "prototype/k20"),
    "domain": ("knn/k8", "knn/k20", "knn/k48", "clean/k20", "prototype/k20"),
    "temporal": ("clean/k8", "clean/k20", "clean/k48", "knn/k8", "boundary/k20"),
    "support": ("knn/k8", "knn/k20", "balanced/k20", "clean/k20", "prototype/k20"),
}
REG_SHARED = ("knn/k32", "knn/k64", "knn/k128", "prototype", "clean/k32")
REG_SPECIFIC = {
    "covariate": ("knn/k16", "knn/k32", "knn/k64", "clean/k32", "moment/a25"),
    "label": ("moment/a10", "moment/a25", "moment/a50", "quantile/a25", "knn/k64"),
    "concept": ("recent/q25", "recent/q50", "knn/k16", "clean/k32", "moment/a25"),
    "selection": ("knn/k16", "knn/k32", "knn/k64", "prototype", "clean/k32"),
    "domain": ("knn/k16", "knn/k32", "knn/k64", "clean/k32", "moment/a25"),
    "temporal": ("recent/q25", "recent/q50", "knn/k16", "clean/k32", "moment/a25"),
    "support": ("knn/k16", "knn/k32", "knn/k64", "prototype", "clean/k32"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
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
    return value[:count]


def atomic_save(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _projector(clean_source: np.ndarray, threshold: int, components: int):
    if threshold <= 0 or clean_source.shape[1] <= threshold:
        return None
    count = min(components, clean_source.shape[0] - 1, clean_source.shape[1] - 1)
    model = TruncatedSVD(n_components=count, random_state=0)
    model.fit(np.asarray(clean_source, dtype=np.float64))
    return model


def _adapt(model: TruncatedSVD | None, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.asarray(X, dtype=np.float32)
    return np.asarray(model.transform(X), dtype=np.float32)


def _regression_base(predictor: Any, X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    center = float(np.mean(y))
    scale = max(float(np.std(y)), 1.0e-12)
    raw = predictor.predict(X, (np.asarray(y) - center) / scale, Q, task_type="Regression")
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    return np.asarray(raw, dtype=np.float64).reshape(-1) * scale + center


def _normalize_probability(value: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(value, dtype=np.float64), 1.0e-9, None)
    return output / output.sum(axis=1, keepdims=True)


def _prior_correct(probability: np.ndarray, source_prior: float, target_prior: float) -> np.ndarray:
    positive = np.clip(np.asarray(probability)[:, 1], 1.0e-6, 1.0 - 1.0e-6)
    source_prior = float(np.clip(source_prior, 0.02, 0.98))
    target_prior = float(np.clip(target_prior, 0.02, 0.98))
    odds = positive / (1.0 - positive)
    ratio = (target_prior / (1.0 - target_prior)) / (source_prior / (1.0 - source_prior))
    adjusted = odds * ratio / (1.0 + odds * ratio)
    return _normalize_probability(np.column_stack([1.0 - adjusted, adjusted]))


def _em_prior(probability: np.ndarray, source_prior: float) -> float:
    source_prior = float(np.clip(source_prior, 0.02, 0.98))
    positive = np.clip(np.asarray(probability)[:, 1], 1.0e-6, 1.0 - 1.0e-6)
    target_prior = source_prior
    for _ in range(100):
        numerator = positive * target_prior / source_prior
        denominator = numerator + (1.0 - positive) * (1.0 - target_prior) / (1.0 - source_prior)
        updated = float(np.clip(np.mean(numerator / denominator), 0.02, 0.98))
        if abs(updated - target_prior) < 1.0e-7:
            break
        target_prior = updated
    return target_prior


def _classification_members(
    X: np.ndarray,
    y: np.ndarray,
    Q: np.ndarray,
    base: np.ndarray,
    classes: np.ndarray,
) -> dict[str, np.ndarray]:
    specs = {
        "knn/k8": ("knn", 8), "knn/k20": ("knn", 20), "knn/k48": ("knn", 48),
        "balanced/k8": ("balanced", 8), "balanced/k20": ("balanced", 20),
        "prototype/k20": ("prototype", 20),
        "clean/k8": ("clean", 8), "clean/k20": ("clean", 20), "clean/k48": ("clean", 48),
        "boundary/k20": ("boundary", 20), "minority/k20": ("minority", 20),
    }
    contexts = classification_contexts(X, y, Q, base, classes, specs)
    output = {
        name: _kernel_context_prediction(X, y, Q, indices, classes, base)
        for name, indices in contexts.items()
    }
    source_prior = float(np.mean(np.asarray(y) == classes[1]))
    output["prior/em"] = _prior_correct(base, source_prior, _em_prior(base, source_prior))
    output["prior/pcc"] = _prior_correct(base, source_prior, float(np.mean(base[:, 1])))
    return {name: _normalize_probability(value) for name, value in output.items()}


def _standardized(X: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    source = scaler.fit_transform(imputer.fit_transform(np.asarray(X)))
    query = scaler.transform(imputer.transform(np.asarray(Q)))
    return source, query


def _knn_regression(X: np.ndarray, y: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    model = KNeighborsRegressor(
        n_neighbors=min(max(2, int(k)), len(y)), weights="distance", n_jobs=-1
    )
    model.fit(X, y)
    return np.asarray(model.predict(Q), dtype=np.float64)


def _prototype_regression(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, 17)))
    selected: list[int] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y >= low) & (y <= high if high == edges[-1] else y < high)
        indices = np.flatnonzero(mask)
        if len(indices):
            center = np.mean(X[indices], axis=0)
            selected.append(int(indices[np.argmin(np.sum((X[indices] - center) ** 2, axis=1))]))
    if len(selected) < 2:
        return np.full(len(Q), float(np.mean(y)))
    indices = np.asarray(selected, dtype=int)
    return _knn_regression(X[indices], y[indices], Q, min(8, len(indices)))


def _quantile_map(base: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(np.asarray(base), kind="stable"), kind="stable")
    quantiles = (order + 0.5) / len(order)
    return np.quantile(np.asarray(y, dtype=np.float64), quantiles)


def _regression_members(
    X: np.ndarray, y: np.ndarray, Q: np.ndarray, base: np.ndarray
) -> dict[str, np.ndarray]:
    source, query = _standardized(X, Q)
    y = np.asarray(y, dtype=np.float64)
    output = {
        f"knn/k{k}": _knn_regression(source, y, query, k)
        for k in (16, 32, 64, 128)
    }
    output["prototype"] = _prototype_regression(source, y, query)
    low, high = np.quantile(y, [0.02, 0.98])
    keep = np.flatnonzero((y >= low) & (y <= high))
    output["clean/k32"] = _knn_regression(source[keep], y[keep], query, 32)
    for fraction in (0.25, 0.50):
        count = min(len(y), max(32, int(round(fraction * len(y)))))
        output[f"recent/q{int(100 * fraction)}"] = _knn_regression(
            source[-count:], y[-count:], query, 32
        )
    base_mean = float(np.mean(base))
    base_std = max(float(np.std(base)), 1.0e-12)
    aligned = (np.asarray(base) - base_mean) / base_std * max(float(np.std(y)), 1.0e-12) + float(np.mean(y))
    for alpha in (0.10, 0.25, 0.50):
        output[f"moment/a{int(100 * alpha)}"] = (1.0 - alpha) * base + alpha * aligned
    mapped = _quantile_map(base, y)
    output["quantile/a25"] = 0.75 * base + 0.25 * mapped
    return output


def _datasets(task: str, requested: set[str]) -> list[str | int]:
    if task == "classification":
        values: list[str | int] = [int(value) for value in frozen.CC18_BINARY_DATASETS]
        names = {int(key): str(value) for key, value in frozen.CC18_DATASET_NAMES.items()}
    else:
        values = list(frozen._regression_names("all"))
        names = {str(value): str(value) for value in values}
    if not requested:
        return values
    return [value for value in values if str(value).casefold() in requested or names[value].casefold() in requested]


def build_task(args: argparse.Namespace, task: str, predictor: Any) -> None:
    requested = {value.strip().casefold() for value in args.dataset.split(",") if value.strip()}
    datasets = _datasets(task, requested)
    for position, dataset in enumerate(datasets, start=1):
        output = args.output / task / f"dataset_{dataset}.npz"
        if output.exists() and not args.force:
            print(f"[{task} {position}/{len(datasets)}] cached {dataset}", flush=True)
            continue
        started = time.perf_counter()
        try:
            conditions = frozen._seven_conditions(task, dataset)
            clean = next(row for row in conditions if row.family == FAMILIES[0] and float(row.strength) == 0.0)
            projector = _projector(
                np.asarray(clean.X_train),
                args.high_dimensional_threshold,
                args.high_dimensional_components,
            )
            families: list[str] = []
            strengths: list[float] = []
            targets: list[np.ndarray] = []
            scales: list[float] = []
            bases: list[np.ndarray] = []
            all_members: list[np.ndarray] = []
            member_names: tuple[str, ...] | None = None
            source_hashes: list[str] = []
            for cell_position, condition in enumerate(conditions, start=1):
                X = _adapt(projector, condition.X_train)
                Q = _adapt(projector, condition.X_target)
                y = np.asarray(condition.y_train)
                if task == "classification":
                    base_result = predict_proba_native_or_ovr(
                        predictor, X, y, Q, max_native_classes=10
                    )
                    classes = np.asarray(base_result.classes)
                    base = np.asarray(base_result.probabilities, dtype=np.float64)
                    members = _classification_members(X, y, Q, base, classes)
                else:
                    base = _regression_base(predictor, X, y, Q)
                    members = _regression_members(X, y, Q, base)
                current_names = tuple(members)
                if member_names is None:
                    member_names = current_names
                elif current_names != member_names:
                    raise RuntimeError("candidate schema changed across conditions")
                families.append(str(condition.family))
                strengths.append(float(condition.strength))
                targets.append(np.asarray(condition.y_target))
                scales.append(float(condition.metric_scale or np.std(condition.y_target)))
                bases.append(base)
                all_members.append(np.stack([members[name] for name in current_names], axis=-1))
                source_hashes.append(_hash(np.asarray(condition.X_train), np.asarray(condition.y_train)))
                print(
                    f"  {dataset} {cell_position}/{len(conditions)} {condition.family} s={condition.strength:g}",
                    flush=True,
                )
            assert member_names is not None
            atomic_save(
                output,
                dataset=np.asarray(str(clean.dataset)),
                data_id=np.asarray(int(clean.data_id)),
                family=np.asarray(families),
                strength=np.asarray(strengths, dtype=np.float64),
                y_target=np.stack(targets),
                metric_scale=np.asarray(scales, dtype=np.float64),
                base=np.stack(bases),
                members=np.stack(all_members),
                member_names=np.asarray(member_names),
                shared_names=np.asarray(CLASS_SHARED if task == "classification" else REG_SHARED),
                specific_registry=np.asarray(json.dumps(CLASS_SPECIFIC if task == "classification" else REG_SPECIFIC, sort_keys=True)),
                source_hash=np.asarray(source_hashes),
                seed=np.asarray(0 if task == "classification" else 2718),
                selection_uses_target_labels=np.asarray(False),
                construction_uses_target_labels=np.asarray(False),
            )
            print(
                f"[{task} {position}/{len(datasets)}] built {dataset} seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
        except Exception as error:
            failure = output.with_suffix(".failure.json")
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(json.dumps({"dataset": str(dataset), "task": task, "error": repr(error), "traceback": traceback.format_exc()}, indent=2) + "\n")
            print(f"FAILED {task}/{dataset}: {error!r}", flush=True)
            if not args.continue_on_error:
                raise


def main() -> None:
    args = arguments()
    args.output = args.output.resolve()
    args.base_run = args.base_run.resolve()
    import openml
    import run_apa_tabicl as apa

    openml.config.set_root_cache_directory(args.base_run / "openml_cache")
    apa.ROOT = args.base_run / "protocol_input"
    limix_root = args.limix_root.resolve()
    sys.path.insert(0, str(limix_root))
    from inference.predictor import LimiXPredictor

    configs = {
        "classification": load_config(limix_root, "cls_default_noretrieval.json", args.classification_pipelines),
        "regression": load_config(limix_root, "reg_default_noretrieval.json", args.regression_pipelines),
    }
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    for task in tasks:
        predictor = LimiXPredictor(
            device=torch.device("cuda"),
            model_path=str(args.model_path.resolve()),
            inference_config=configs[task],
            mix_precision=True,
            seed=0 if task == "classification" else 2718,
        )
        build_task(args, task, predictor)


if __name__ == "__main__":
    main()
