#!/usr/bin/env python3
"""Select clean-only LimiX TabACE policies with dataset-level OOF safety.

This script reads the already frozen five-perturbation candidate banks but only
accesses their original (rho=0) row.  For every outer fold, Shared is selected
from the common context pool using the remaining datasets.  Specific is then
selected from the full context pool against that fold's frozen Shared output.
The held-out dataset labels are used only after both policies are fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ALPHAS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
)


@dataclass(frozen=True)
class Dataset:
    data_id: int
    name: str
    fold: int
    task: str
    target: np.ndarray
    classes: np.ndarray | None
    scale: float | None
    base: np.ndarray
    pools: dict[str, np.ndarray]


@dataclass(frozen=True)
class Recipe:
    members: tuple[str, ...]
    alpha: float

    @property
    def key(self) -> str:
        return f"mean({','.join(self.members)});alpha={self.alpha:.6g}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("classification", "regression", "all"), default="all"
    )
    parser.add_argument("--output-name", default="oof_clean_safe")
    return parser.parse_args()


def normalise_proba(value: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(value, dtype=np.float64), 1.0e-12, None)
    return output / np.clip(output.sum(axis=1, keepdims=True), 1.0e-12, None)


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    """Compute tie-corrected binary AUC without sklearn's validation overhead."""

    positive = np.asarray(target, dtype=bool)
    positive_n = int(np.sum(positive))
    negative_n = len(positive) - positive_n
    if positive_n == 0 or negative_n == 0:
        return 0.5
    ranks = rankdata(np.asarray(score, dtype=np.float64), method="average")
    numerator = float(np.sum(ranks[positive])) - positive_n * (positive_n + 1) / 2
    return numerator / (positive_n * negative_n)


def metrics(data: Dataset, prediction: np.ndarray) -> dict[str, float]:
    if data.task == "classification":
        assert data.classes is not None
        probability = normalise_proba(prediction)
        predicted = data.classes[np.argmax(probability, axis=1)]
        recalls = [
            float(np.mean(predicted[data.target == label] == label))
            for label in data.classes
            if np.any(data.target == label)
        ]
        balanced = float(np.mean(recalls)) if recalls else 0.0
        if len(data.classes) == 2:
            auc = binary_auc(data.target == data.classes[1], probability[:, 1])
        else:
            classwise = [
                binary_auc(data.target == label, probability[:, class_pos])
                for class_pos, label in enumerate(data.classes)
                if np.any(data.target == label) and np.any(data.target != label)
            ]
            auc = float(np.mean(classwise)) if classwise else 0.5
        return {
            "accuracy": float(np.mean(predicted == data.target)),
            "balanced_accuracy": balanced,
            "auc": auc,
        }

    assert data.scale is not None
    error = np.asarray(data.target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    denominator = max(
        float(np.sum((data.target - np.mean(data.target)) ** 2)), 1.0e-12
    )
    return {
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "nmae": float(np.mean(np.abs(error))) / data.scale,
        "nrmse": float(np.sqrt(np.mean(error**2))) / data.scale,
    }


def quality(data: Dataset, prediction: np.ndarray) -> np.ndarray:
    values = metrics(data, prediction)
    if data.task == "classification":
        return np.asarray([values["accuracy"], values["auc"]], dtype=np.float64)
    return np.asarray([values["r2"], -values["nrmse"]], dtype=np.float64)


def fold_metadata(results_root: Path, task: str) -> dict[int, tuple[int, str]]:
    path = results_root / "oof" / task / "detail_oof.csv"
    rows = pd.read_csv(path, usecols=["data_id", "dataset", "outer_fold"])
    rows = rows.drop_duplicates("data_id")
    return {
        int(row.data_id): (int(row.outer_fold), str(row.dataset))
        for row in rows.itertuples(index=False)
    }


def load_datasets(results_root: Path, task: str) -> list[Dataset]:
    metadata = fold_metadata(results_root, task)
    bank_dir = results_root / "source_candidate" / "banks" / task
    output = []
    for path in sorted(bank_dir.glob("dataset_*.npz"), key=lambda p: int(p.stem[8:])):
        data_id = int(path.stem[8:])
        if data_id not in metadata:
            raise KeyError(f"missing fold metadata for dataset {data_id}")
        fold, name = metadata[data_id]
        with np.load(path, allow_pickle=False) as bank:
            problem = np.asarray(bank["problem"]).astype(str)
            rho = np.asarray(bank["rho"], dtype=int)
            clean_positions = np.flatnonzero((problem == "original") & (rho == 0))
            if clean_positions.tolist() != [0]:
                raise ValueError(f"{path}: expected one original row at index 0")
            if bool(np.asarray(bank["construction_uses_target_labels"]).item()):
                raise ValueError(f"{path}: candidate construction used target labels")
            pools: dict[str, np.ndarray] = {
                "candidate/shared": np.asarray(bank["shared"][0], dtype=np.float64)
            }
            pro_names = [str(value) for value in np.asarray(bank["pro_names"]).tolist()]
            pro = np.asarray(bank["pro"][0], dtype=np.float64)
            for index, member in enumerate(pro_names):
                if member == "shared_output":
                    continue
                pools[member] = pro[:, index] if task == "regression" else pro[:, index, :]
            output.append(
                Dataset(
                    data_id=data_id,
                    name=name,
                    fold=fold,
                    task=task,
                    target=np.asarray(bank["y_target"]),
                    classes=(
                        np.asarray(bank["classes"])
                        if task == "classification"
                        else None
                    ),
                    scale=(
                        float(np.asarray(bank["scale"]).item())
                        if task == "regression"
                        else None
                    ),
                    base=np.asarray(bank["base"][0], dtype=np.float64),
                    pools=pools,
                )
            )
    expected = 72 if task == "classification" else 35
    if len(output) != expected:
        raise ValueError(f"{task}: expected {expected} banks, found {len(output)}")
    schemas = {tuple(data.pools) for data in output}
    if len(schemas) != 1:
        raise ValueError(f"{task}: clean candidate schemas are inconsistent")
    return output


def common_names(data: Dataset) -> tuple[str, ...]:
    names = ["candidate/shared"]
    names.extend(name for name in data.pools if name.startswith("shared/"))
    return tuple(dict.fromkeys(names))


def full_names(data: Dataset) -> tuple[str, ...]:
    return tuple(data.pools)


def apply_recipe(data: Dataset, baseline: np.ndarray, recipe: Recipe) -> np.ndarray:
    center = np.mean([data.pools[name] for name in recipe.members], axis=0)
    prediction = (1.0 - recipe.alpha) * baseline + recipe.alpha * center
    return normalise_proba(prediction) if data.task == "classification" else prediction


def singleton_rank(
    datasets: list[Dataset],
    train_ids: set[int],
    baselines: dict[int, np.ndarray],
    names: tuple[str, ...],
) -> tuple[str, ...]:
    base_quality = {
        data.data_id: quality(data, baselines[data.data_id])
        for data in datasets
        if data.data_id in train_ids
    }
    scored = []
    for name in names:
        gains = []
        for data in datasets:
            if data.data_id not in train_ids:
                continue
            prediction = apply_recipe(data, baselines[data.data_id], Recipe((name,), 1.0))
            gains.append(quality(data, prediction) - base_quality[data.data_id])
        mean_gain = np.mean(gains, axis=0)
        scales = np.asarray([0.005, 0.002]) if datasets[0].task == "classification" else np.asarray([0.02, 0.01])
        scored.append((float(np.sum(mean_gain / scales)), name))
    return tuple(name for _score, name in sorted(scored, reverse=True))


def candidate_member_sets(ranking: tuple[str, ...]) -> list[tuple[str, ...]]:
    candidates: set[tuple[str, ...]] = {(name,) for name in ranking}
    top_pairs = ranking[: min(6, len(ranking))]
    candidates.update(combinations(top_pairs, 2))
    for size in range(2, min(6, len(ranking)) + 1):
        candidates.add(tuple(ranking[:size]))
    candidates.add(tuple(ranking))
    return sorted(candidates, key=lambda value: (len(value), value))


def select_recipe(
    datasets: list[Dataset],
    outer_fold: int,
    baselines: dict[int, np.ndarray],
    names: tuple[str, ...],
    stage: str,
) -> tuple[Recipe, dict[str, object]]:
    train = [data for data in datasets if data.fold != outer_fold]
    train_ids = {data.data_id for data in train}
    ranking = singleton_rank(datasets, train_ids, baselines, names)
    # Clean correction is intentionally low-capacity: a single context is
    # blended into the lower tier.  This avoids selecting an unnecessarily
    # complex ensemble from small cross-dataset differences at rho=0.
    member_sets = [(name,) for name in ranking]
    base_quality = {
        data.data_id: quality(data, baselines[data.data_id]) for data in train
    }
    scales = (
        np.asarray([0.005, 0.002], dtype=np.float64)
        if datasets[0].task == "classification"
        else np.asarray([0.02, 0.01], dtype=np.float64)
    )
    inner_fold_tolerance = (
        np.asarray([5.0e-4, 2.5e-4], dtype=np.float64)
        if datasets[0].task == "classification"
        else np.asarray([2.0e-3, 1.0e-3], dtype=np.float64)
    )
    alphas = (
        (0.001,)
        if stage == "shared"
        else (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15)
    )
    candidates = []
    for members in member_sets:
        for alpha in alphas:
            recipe = Recipe(tuple(members), alpha)
            by_fold: dict[int, list[np.ndarray]] = {}
            all_gains = []
            changed = []
            for data in train:
                prediction = apply_recipe(data, baselines[data.data_id], recipe)
                gain = quality(data, prediction) - base_quality[data.data_id]
                by_fold.setdefault(data.fold, []).append(gain)
                all_gains.append(gain)
                changed.append(
                    float(np.max(np.abs(prediction - baselines[data.data_id])))
                )
            group_gains = np.asarray(
                [np.mean(by_fold[fold], axis=0) for fold in sorted(by_fold)]
            )
            mean_gain = np.mean(all_gains, axis=0)
            # The policy must be non-inferior on both metrics in every available
            # inner dataset fold.  A positive metric change is also required so
            # selecting an exact copy of the lower tier is impossible.
            group_safe = (
                np.all(group_gains >= -inner_fold_tolerance)
                if stage == "shared"
                else np.all(group_gains >= -5.0 * inner_fold_tolerance)
            )
            eligible = bool(
                group_safe
                and np.all(mean_gain >= -1.0e-12)
                and np.max(mean_gain) >= 1.0e-6
                and np.mean(changed) >= 1.0e-8
            )
            normalized_groups = group_gains / scales
            if stage == "shared":
                score = float(
                    np.min(normalized_groups)
                    + 0.25 * np.sum(mean_gain / scales)
                    - 0.05 * np.mean(np.minimum(normalized_groups, 0.0) ** 2)
                )
            else:
                score = float(
                    np.sum(mean_gain / scales)
                    + 0.10 * np.min(mean_gain / scales)
                    - 0.02 * np.mean(np.minimum(normalized_groups, 0.0) ** 2)
                )
            candidates.append(
                {
                    "recipe": recipe,
                    "eligible": eligible,
                    "score": score,
                    "mean_gain": mean_gain,
                    "group_gains": group_gains,
                    "mean_prediction_change": float(np.mean(changed)),
                }
            )
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"{datasets[0].task} fold={outer_fold} stage={stage}: "
            "no distinct clean policy passed the training-fold noninferiority constraints"
        )
    selected = max(eligible, key=lambda item: (item["score"], -len(item["recipe"].members)))
    recipe = selected["recipe"]
    diagnostics = {
        "task": datasets[0].task,
        "outer_fold": outer_fold,
        "stage": stage,
        "outer_test_labels_used_for_selection": False,
        "perturbation_rows_used_for_selection": False,
        "family_or_strength_used_for_selection": False,
        "train_dataset_count": len(train),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "members": list(recipe.members),
        "alpha": recipe.alpha,
        "train_mean_quality_gain": selected["mean_gain"].tolist(),
        "inner_fold_quality_gains": selected["group_gains"].tolist(),
        "mean_prediction_change": selected["mean_prediction_change"],
        "ranking": list(ranking),
    }
    return recipe, diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regression_router_features(
    baseline: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Unlabeled batch diagnostics used by the clean regression router."""

    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    delta = candidate - baseline
    base_std = max(float(np.std(baseline)), 1.0e-8)
    base_range = max(float(np.ptp(baseline)), 1.0e-8)
    if np.std(candidate) <= 1.0e-12 or np.std(baseline) <= 1.0e-12:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(baseline, candidate)[0, 1])
    return np.asarray(
        [
            float(np.mean(delta)) / base_std,
            float(np.std(delta)) / base_std,
            float(np.sqrt(np.mean(delta**2))) / base_std,
            float(np.quantile(np.abs(delta), 0.9)) / base_std,
            float(np.std(candidate)) / base_std,
            correlation,
            float(np.mean(candidate) - np.mean(baseline)) / base_range,
            float(np.ptp(candidate)) / base_range,
        ],
        dtype=np.float64,
    )


def fit_ridge_gain(
    features: np.ndarray, gains: np.ndarray, regularization: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    design = np.column_stack(
        [np.ones(len(features)), (features - mean) / scale]
    )
    penalty = np.diag([0.0, *([regularization] * features.shape[1])])
    beta = np.linalg.solve(
        design.T @ design + penalty + 1.0e-10 * np.eye(design.shape[1]),
        design.T @ gains,
    )
    return beta, mean, scale


def predict_ridge_gain(
    feature: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
    beta, mean, scale = model
    return float(beta[0] + ((feature - mean) / scale) @ beta[1:])


def router_options(
    data: Dataset,
    baseline: np.ndarray,
    names: tuple[str, ...],
    alphas: tuple[float, ...],
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]:
    output = {}
    for name in names:
        for alpha in alphas:
            prediction = apply_recipe(data, baseline, Recipe((name,), alpha))
            if float(np.max(np.abs(prediction - baseline))) <= 1.0e-12:
                continue
            output[(name, alpha)] = (
                prediction,
                regression_router_features(baseline, prediction),
            )
    return output


def fit_regression_router(
    train: list[Dataset],
    baselines: dict[int, np.ndarray],
    names: tuple[str, ...],
    alphas: tuple[float, ...],
    regularization: float,
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    options = {
        data.data_id: router_options(
            data, baselines[data.data_id], names, alphas
        )
        for data in train
    }
    common = set.intersection(*(set(value) for value in options.values()))
    models = {}
    for candidate in sorted(common):
        feature_rows = []
        gain_rows = []
        for data in train:
            prediction, feature = options[data.data_id][candidate]
            feature_rows.append(feature)
            gain_rows.append(
                quality(data, prediction)[0]
                - quality(data, baselines[data.data_id])[0]
            )
        models[candidate] = fit_ridge_gain(
            np.stack(feature_rows), np.asarray(gain_rows), regularization
        )
    return models


def apply_regression_router(
    data: Dataset,
    baseline: np.ndarray,
    names: tuple[str, ...],
    alphas: tuple[float, ...],
    models: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    minimum_gain: float,
) -> tuple[np.ndarray, str, float]:
    options = router_options(data, baseline, names, alphas)
    predicted = [
        (predict_ridge_gain(options[candidate][1], model), candidate)
        for candidate, model in models.items()
        if candidate in options
    ]
    if not predicted:
        return baseline, "base", float("-inf")
    predicted_gain, candidate = max(predicted)
    if predicted_gain <= minimum_gain:
        return baseline, "base", predicted_gain
    name, alpha = candidate
    return options[candidate][0], f"{name}@{alpha:.6g}", predicted_gain


def select_regression_router(
    train: list[Dataset],
    baselines: dict[int, np.ndarray],
    names: tuple[str, ...],
    alphas: tuple[float, ...],
    stage: str,
) -> tuple[float, float, dict[str, object]]:
    regularizations = (0.1, 1.0, 10.0, 100.0)
    minimum_gains = (0.0, 0.001, 0.003, 0.005, 0.01, 0.02)
    candidates = []
    for regularization in regularizations:
        routed: dict[int, tuple[np.ndarray, str, float]] = {}
        # Leave-one-dataset-out predictions tune the router without evaluating
        # a model on the labels used to fit its gain regressions.
        for held_out in train:
            inner_train = [
                data for data in train if data.data_id != held_out.data_id
            ]
            models = fit_regression_router(
                inner_train, baselines, names, alphas, regularization
            )
            routed[held_out.data_id] = apply_regression_router(
                held_out,
                baselines[held_out.data_id],
                names,
                alphas,
                models,
                -float("inf"),
            )
        for minimum_gain in minimum_gains:
            gains = []
            changed = 0
            choices = []
            for data in train:
                prediction, choice, predicted_gain = routed[data.data_id]
                if predicted_gain <= minimum_gain:
                    prediction, choice = baselines[data.data_id], "base"
                else:
                    changed += 1
                gains.append(
                    quality(data, prediction)
                    - quality(data, baselines[data.data_id])
                )
                choices.append(choice)
            mean_gain = np.mean(gains, axis=0)
            eligible = bool(
                np.all(mean_gain >= -1.0e-12) and changed > 0
            )
            score = float(mean_gain[0] / 0.01 + mean_gain[1] / 0.005)
            candidates.append(
                {
                    "regularization": regularization,
                    "minimum_gain": minimum_gain,
                    "eligible": eligible,
                    "score": score,
                    "mean_gain": mean_gain,
                    "changed": changed,
                    "choices": choices,
                }
            )
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"regression stage={stage}: no nested-OOF router improved both clean metrics"
        )
    selected = max(eligible, key=lambda item: item["score"])
    diagnostics = {
        "stage": stage,
        "router_kind": "linear_ridge_gain",
        "router_inputs": "unlabeled_prediction_diagnostics",
        "inner_validation": "leave_one_dataset_out",
        "regularization": selected["regularization"],
        "minimum_predicted_gain": selected["minimum_gain"],
        "inner_oof_quality_gain": selected["mean_gain"].tolist(),
        "inner_oof_changed_datasets": selected["changed"],
        "candidate_configuration_count": len(candidates),
        "outer_test_labels_used_for_selection": False,
        "perturbation_rows_used_for_selection": False,
    }
    return (
        float(selected["regularization"]),
        float(selected["minimum_gain"]),
        diagnostics,
    )


def run_regression_router(
    results_root: Path, output_root: Path
) -> dict[str, object]:
    task = "regression"
    datasets = load_datasets(results_root, task)
    shared_names = tuple(
        name
        for name in common_names(datasets[0])
        if name not in {"candidate/shared", "shared/clean"}
    )
    specific_names = tuple(
        name
        for name in full_names(datasets[0])
        if name not in {"candidate/shared", "shared/clean", "source_clean"}
    )
    shared_alphas = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    specific_alphas = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    base_predictions = {data.data_id: data.base for data in datasets}
    final_predictions: dict[str, dict[int, np.ndarray]] = {
        "base": dict(base_predictions),
        "shared": {},
        "specific": {},
    }
    selection_rows = []
    route_rows = []
    for fold in range(5):
        train = [data for data in datasets if data.fold != fold]
        shared_reg, shared_threshold, shared_diagnostics = select_regression_router(
            train, base_predictions, shared_names, shared_alphas, "shared"
        )
        shared_models = fit_regression_router(
            train, base_predictions, shared_names, shared_alphas, shared_reg
        )
        shared_all = {}
        for data in datasets:
            prediction, choice, predicted_gain = apply_regression_router(
                data,
                data.base,
                shared_names,
                shared_alphas,
                shared_models,
                shared_threshold,
            )
            shared_all[data.data_id] = prediction
            if data.fold == fold:
                route_rows.append(
                    {
                        "task": task,
                        "outer_fold": fold,
                        "data_id": data.data_id,
                        "stage": "shared",
                        "choice": choice,
                        "predicted_gain": predicted_gain,
                    }
                )
        specific_reg, specific_threshold, specific_diagnostics = select_regression_router(
            train, shared_all, specific_names, specific_alphas, "specific"
        )
        specific_models = fit_regression_router(
            train, shared_all, specific_names, specific_alphas, specific_reg
        )
        for data in datasets:
            prediction, choice, predicted_gain = apply_regression_router(
                data,
                shared_all[data.data_id],
                specific_names,
                specific_alphas,
                specific_models,
                specific_threshold,
            )
            if data.fold == fold:
                final_predictions["shared"][data.data_id] = shared_all[data.data_id]
                final_predictions["specific"][data.data_id] = prediction
                route_rows.append(
                    {
                        "task": task,
                        "outer_fold": fold,
                        "data_id": data.data_id,
                        "stage": "specific",
                        "choice": choice,
                        "predicted_gain": predicted_gain,
                    }
                )
        selection_rows.extend(
            [
                {"outer_fold": fold, **shared_diagnostics},
                {"outer_fold": fold, **specific_diagnostics},
            ]
        )
        print(
            f"[regression-router] fold={fold} shared_reg={shared_reg:g} "
            f"shared_min_gain={shared_threshold:g} specific_reg={specific_reg:g} "
            f"specific_min_gain={specific_threshold:g}",
            flush=True,
        )

    detail_rows = []
    for data in datasets:
        row: dict[str, object] = {
            "task": task,
            "data_id": data.data_id,
            "dataset": data.name,
            "outer_fold": data.fold,
            "condition": "clean",
            "rho": 0,
            "outer_test_labels_used_for_selection": False,
            "perturbation_rows_used_for_selection": False,
            "family_or_strength_used_for_selection": False,
        }
        for method in ("base", "shared", "specific"):
            prediction = final_predictions[method][data.data_id]
            for metric, value in metrics(data, prediction).items():
                row[f"{method}_{metric}"] = value
            row[f"{method}_max_abs_prediction_change_from_base"] = float(
                np.max(np.abs(prediction - data.base))
            )
        detail_rows.append(row)

    task_output = output_root / task
    write_csv(task_output / "detail_clean_oof.csv", detail_rows)
    write_csv(task_output / "outer_clean_router_selections.csv", [
        {
            **row,
            "inner_oof_quality_gain": json.dumps(row["inner_oof_quality_gain"]),
        }
        for row in selection_rows
    ])
    write_csv(task_output / "outer_clean_routes.csv", route_rows)
    detail = pd.DataFrame(detail_rows)
    summary = {
        method: {
            metric: float(detail[f"{method}_{metric}"].mean())
            for metric in ("r2", "nmae", "nrmse")
        }
        for method in ("base", "shared", "specific")
    }
    paired = paired_audit(detail, task)
    write_csv(task_output / "paired_clean_tests.csv", paired)
    audit = {
        "task": task,
        "dataset_count": len(datasets),
        "outer_folds": 5,
        "condition": "clean_only",
        "single_seed": True,
        "router_inputs": "unlabeled_prediction_diagnostics",
        "inner_validation": "leave_one_dataset_out",
        "nonzero_perturbation_predictions_recomputed": False,
        "nonzero_perturbation_results_modified": False,
        "outer_test_labels_used_for_selection": False,
        "prediction_construction_uses_target_labels": False,
        "summary": summary,
        "paired_tests": paired,
    }
    (task_output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def source_calibrated_prediction(
    base: np.ndarray, source_target: np.ndarray, mode: str, alpha: float
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    source_target = np.asarray(source_target, dtype=np.float64)
    if mode == "mean":
        center = base + (float(np.mean(source_target)) - float(np.mean(base)))
    elif mode == "moment":
        center = (
            (base - float(np.mean(base)))
            * max(float(np.std(source_target)), 1.0e-12)
            / max(float(np.std(base)), 1.0e-12)
            + float(np.mean(source_target))
        )
    elif mode == "median_iqr":
        source_iqr = max(
            float(np.quantile(source_target, 0.75) - np.quantile(source_target, 0.25)),
            1.0e-12,
        )
        base_iqr = max(
            float(np.quantile(base, 0.75) - np.quantile(base, 0.25)),
            1.0e-12,
        )
        center = (
            (base - float(np.median(base))) * source_iqr / base_iqr
            + float(np.median(source_target))
        )
    else:
        raise ValueError(f"unknown source calibration mode: {mode}")
    return (1.0 - alpha) * base + alpha * center


def run_regression_source_calibration(
    results_root: Path, output_root: Path
) -> dict[str, object]:
    """Clean regression calibration using source y and unlabeled target predictions."""

    import openml
    from fair_budget_experiments.limix_2m_ovr.src import protocol

    task = "regression"
    openml.config.set_root_cache_directory(
        results_root.parent / "paper_run" / "openml_cache"
    )
    datasets = load_datasets(results_root, task)
    source_targets = {
        int(spec.data_id): protocol.load_ctr23_split(spec).y_source
        for spec in protocol.suite_specs(protocol.CTR23_SUITE_ID)
    }
    if set(source_targets) != {data.data_id for data in datasets}:
        raise ValueError("CTR23 source-target calibration set does not match clean banks")
    shared_recipes = [
        (mode, alpha)
        for mode in ("mean", "moment", "median_iqr")
        for alpha in (0.001, 0.002, 0.005, 0.01, 0.02)
    ]
    specific_recipes = [
        (mode, alpha)
        for mode in ("mean", "moment", "median_iqr")
        for alpha in (0.005, 0.01, 0.02, 0.05)
    ]
    base_predictions = {data.data_id: data.base for data in datasets}
    final_predictions: dict[str, dict[int, np.ndarray]] = {
        "base": dict(base_predictions),
        "shared": {},
        "specific": {},
    }
    selection_rows = []

    def select(
        train: list[Dataset],
        recipes: list[tuple[str, float]],
        references: dict[int, np.ndarray],
        stage: str,
    ) -> tuple[str, float, dict[str, object]]:
        candidates = []
        for mode, alpha in recipes:
            gains = []
            changed = []
            by_fold: dict[int, list[np.ndarray]] = {}
            for data in train:
                prediction = source_calibrated_prediction(
                    data.base, source_targets[data.data_id], mode, alpha
                )
                gain = quality(data, prediction) - quality(
                    data, references[data.data_id]
                )
                gains.append(gain)
                by_fold.setdefault(data.fold, []).append(gain)
                changed.append(float(np.max(np.abs(prediction - data.base))))
            mean_gain = np.mean(gains, axis=0)
            group_gains = np.asarray(
                [np.mean(by_fold[fold], axis=0) for fold in sorted(by_fold)]
            )
            eligible = bool(
                np.all(mean_gain >= -1.0e-12)
                and np.max(mean_gain) >= 1.0e-7
                and np.mean(changed) > 1.0e-8
            )
            score = float(
                mean_gain[0] / 0.001
                + mean_gain[1] / 0.0005
                - 0.01
                * np.mean(
                    np.minimum(
                        group_gains / np.asarray([0.002, 0.001]), 0.0
                    )
                    ** 2
                )
            )
            candidates.append(
                {
                    "mode": mode,
                    "alpha": alpha,
                    "eligible": eligible,
                    "score": score,
                    "mean_gain": mean_gain,
                    "group_gains": group_gains,
                    "mean_prediction_change": float(np.mean(changed)),
                }
            )
        eligible = [item for item in candidates if item["eligible"]]
        if not eligible:
            raise RuntimeError(
                f"regression stage={stage}: no source calibration improved both clean metrics"
            )
        selected = max(eligible, key=lambda item: item["score"])
        diagnostics = {
            "stage": stage,
            "mode": selected["mode"],
            "alpha": selected["alpha"],
            "train_mean_quality_gain": selected["mean_gain"].tolist(),
            "inner_fold_quality_gains": selected["group_gains"].tolist(),
            "mean_prediction_change": selected["mean_prediction_change"],
            "candidate_count": len(candidates),
            "source_labels_used": True,
            "target_labels_used_for_selection": False,
            "perturbation_rows_used_for_selection": False,
        }
        return str(selected["mode"]), float(selected["alpha"]), diagnostics

    for fold in range(5):
        train = [data for data in datasets if data.fold != fold]
        shared_mode, shared_alpha, shared_diagnostics = select(
            train, shared_recipes, base_predictions, "shared"
        )
        shared_all = {
            data.data_id: source_calibrated_prediction(
                data.base,
                source_targets[data.data_id],
                shared_mode,
                shared_alpha,
            )
            for data in datasets
        }
        specific_mode, specific_alpha, specific_diagnostics = select(
            train, specific_recipes, shared_all, "specific"
        )
        specific_all = {
            data.data_id: source_calibrated_prediction(
                data.base,
                source_targets[data.data_id],
                specific_mode,
                specific_alpha,
            )
            for data in datasets
        }
        for data in datasets:
            if data.fold == fold:
                final_predictions["shared"][data.data_id] = shared_all[data.data_id]
                final_predictions["specific"][data.data_id] = specific_all[data.data_id]
        selection_rows.extend(
            [
                {"outer_fold": fold, **shared_diagnostics},
                {"outer_fold": fold, **specific_diagnostics},
            ]
        )
        print(
            f"[regression-source-clean] fold={fold} "
            f"shared={shared_mode}@{shared_alpha:g} "
            f"specific={specific_mode}@{specific_alpha:g}",
            flush=True,
        )

    detail_rows = []
    for data in datasets:
        row: dict[str, object] = {
            "task": task,
            "data_id": data.data_id,
            "dataset": data.name,
            "outer_fold": data.fold,
            "condition": "clean",
            "rho": 0,
            "source_labels_used_for_calibration": True,
            "outer_test_labels_used_for_selection": False,
            "perturbation_rows_used_for_selection": False,
            "family_or_strength_used_for_selection": False,
        }
        for method in ("base", "shared", "specific"):
            prediction = final_predictions[method][data.data_id]
            for metric, value in metrics(data, prediction).items():
                row[f"{method}_{metric}"] = value
            row[f"{method}_max_abs_prediction_change_from_base"] = float(
                np.max(np.abs(prediction - data.base))
            )
        detail_rows.append(row)

    task_output = output_root / task
    write_csv(task_output / "detail_clean_oof.csv", detail_rows)
    write_csv(
        task_output / "outer_clean_selections.csv",
        [
            {
                **row,
                "train_mean_quality_gain": json.dumps(
                    row["train_mean_quality_gain"]
                ),
                "inner_fold_quality_gains": json.dumps(
                    row["inner_fold_quality_gains"]
                ),
            }
            for row in selection_rows
        ],
    )
    detail = pd.DataFrame(detail_rows)
    summary = {
        method: {
            metric: float(detail[f"{method}_{metric}"].mean())
            for metric in ("r2", "nmae", "nrmse")
        }
        for method in ("base", "shared", "specific")
    }
    paired = paired_audit(detail, task)
    write_csv(task_output / "paired_clean_tests.csv", paired)
    audit = {
        "task": task,
        "dataset_count": len(datasets),
        "outer_folds": 5,
        "condition": "clean_only",
        "single_seed": True,
        "calibration_inputs": "source_y_moments_and_unlabeled_target_predictions",
        "nonzero_perturbation_predictions_recomputed": False,
        "nonzero_perturbation_results_modified": False,
        "outer_test_labels_used_for_selection": False,
        "prediction_construction_uses_target_labels": False,
        "summary": summary,
        "paired_tests": paired,
    }
    (task_output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def paired_audit(detail: pd.DataFrame, task: str) -> list[dict[str, object]]:
    pairs = [("shared", "base"), ("specific", "shared")]
    metric_directions = (
        (("accuracy", 1.0), ("auc", 1.0))
        if task == "classification"
        else (("r2", 1.0), ("nrmse", -1.0))
    )
    rows = []
    for upper, lower in pairs:
        for metric, direction in metric_directions:
            difference = direction * (
                detail[f"{upper}_{metric}"] - detail[f"{lower}_{metric}"]
            )
            statistic, pvalue = wilcoxon(
                difference.to_numpy(), zero_method="zsplit", alternative="greater"
            )
            rows.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "metric": metric,
                    "higher_is_better_direction": direction,
                    "mean_directed_gain": float(difference.mean()),
                    "wins": int((difference > 0).sum()),
                    "ties": int((difference == 0).sum()),
                    "losses": int((difference < 0).sum()),
                    "wilcoxon_statistic": float(statistic),
                    "wilcoxon_one_sided_p": float(pvalue),
                }
            )
    return rows


def run_task(results_root: Path, output_root: Path, task: str) -> dict[str, object]:
    datasets = load_datasets(results_root, task)
    shared_names = common_names(datasets[0])
    specific_names = full_names(datasets[0])
    base_predictions = {data.data_id: data.base for data in datasets}
    final_predictions: dict[str, dict[int, np.ndarray]] = {
        "base": dict(base_predictions),
        "shared": {},
        "specific": {},
    }
    selections = []
    for fold in range(5):
        shared_recipe, shared_diagnostics = select_recipe(
            datasets, fold, base_predictions, shared_names, "shared"
        )
        shared_predictions = {
            data.data_id: apply_recipe(data, data.base, shared_recipe)
            for data in datasets
        }
        specific_recipe, specific_diagnostics = select_recipe(
            datasets, fold, shared_predictions, specific_names, "specific"
        )
        specific_predictions = {
            data.data_id: apply_recipe(
                data, shared_predictions[data.data_id], specific_recipe
            )
            for data in datasets
        }
        for data in datasets:
            if data.fold == fold:
                final_predictions["shared"][data.data_id] = shared_predictions[data.data_id]
                final_predictions["specific"][data.data_id] = specific_predictions[data.data_id]
        selections.extend([shared_diagnostics, specific_diagnostics])
        print(
            f"[{task}] fold={fold} shared={shared_recipe.key} "
            f"specific={specific_recipe.key}",
            flush=True,
        )

    detail_rows = []
    for data in datasets:
        row: dict[str, object] = {
            "task": task,
            "data_id": data.data_id,
            "dataset": data.name,
            "outer_fold": data.fold,
            "condition": "clean",
            "rho": 0,
            "outer_test_labels_used_for_selection": False,
            "perturbation_rows_used_for_selection": False,
            "family_or_strength_used_for_selection": False,
        }
        for method in ("base", "shared", "specific"):
            prediction = final_predictions[method][data.data_id]
            for metric, value in metrics(data, prediction).items():
                row[f"{method}_{metric}"] = value
            row[f"{method}_max_abs_prediction_change_from_base"] = float(
                np.max(np.abs(prediction - data.base))
            )
        detail_rows.append(row)

    task_output = output_root / task
    write_csv(task_output / "detail_clean_oof.csv", detail_rows)
    selection_rows = []
    for item in selections:
        selection_rows.append(
            {
                **item,
                "members": json.dumps(item["members"]),
                "train_mean_quality_gain": json.dumps(item["train_mean_quality_gain"]),
                "inner_fold_quality_gains": json.dumps(item["inner_fold_quality_gains"]),
                "ranking": json.dumps(item["ranking"]),
            }
        )
    write_csv(task_output / "outer_clean_selections.csv", selection_rows)
    detail = pd.DataFrame(detail_rows)
    summary = {
        method: {
            metric: float(detail[f"{method}_{metric}"].mean())
            for metric in (
                ("accuracy", "balanced_accuracy", "auc")
                if task == "classification"
                else ("r2", "nmae", "nrmse")
            )
        }
        for method in ("base", "shared", "specific")
    }
    paired = paired_audit(detail, task)
    write_csv(task_output / "paired_clean_tests.csv", paired)
    audit = {
        "task": task,
        "dataset_count": len(datasets),
        "outer_folds": 5,
        "condition": "clean_only",
        "source_candidate_banks": str(
            results_root / "source_candidate" / "banks" / task
        ),
        "nonzero_perturbation_predictions_recomputed": False,
        "nonzero_perturbation_results_modified": False,
        "outer_test_labels_used_for_selection": False,
        "prediction_construction_uses_target_labels": False,
        "shared_pool": list(shared_names),
        "specific_pool_size": len(specific_names),
        "summary": summary,
        "paired_tests": paired,
    }
    (task_output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    args = arguments()
    results_root = args.results_root.resolve()
    output_root = results_root / args.output_name
    tasks: Iterable[str] = (
        ("classification", "regression") if args.task == "all" else (args.task,)
    )
    audits = {
        task: (
            run_regression_source_calibration(results_root, output_root)
            if task == "regression"
            else run_task(results_root, output_root, task)
        )
        for task in tasks
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "combined_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
