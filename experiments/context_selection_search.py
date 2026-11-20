from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_classification
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from tabicl import TabICLClassifier

from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


@dataclass(frozen=True)
class ClassificationDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    source: str


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    r_train: np.ndarray
    r_test: np.ndarray
    p_train: np.ndarray
    p_test: np.ndarray
    causal_weights: np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str
    transform: Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]
    model_kwargs: dict[str, object]
    decision_rule: str = "argmax"
    tabicl_gate_repeats: int = 0
    tabicl_gate_threshold: float = 0.0


def reset_runtime_seed(random_state: int) -> None:
    np.random.seed(random_state)
    random.seed(random_state)
    try:
        import torch

        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    except Exception:
        pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def standardize_for_score(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(X, dtype=np.float32))


def load_datasets(synthetic_samples: int, random_state: int) -> list[ClassificationDataset]:
    datasets: list[ClassificationDataset] = []
    for name, loader in [
        ("breast_cancer", load_breast_cancer),
        ("wine", load_wine),
        ("digits", load_digits),
    ]:
        data = loader()
        datasets.append(
            ClassificationDataset(
                name=name,
                X=np.asarray(data.data, dtype=np.float32),
                y=np.asarray(data.target),
                source="sklearn",
            )
        )

    synth_specs = [
        ("synth_selection_linear", 16, 6, 1.0, 0.01),
        ("synth_selection_hard_overlap", 24, 8, 0.55, 0.03),
        ("synth_rare_boundary", 32, 6, 0.75, 0.02),
        ("synth_high_dim_sparse", 80, 8, 0.85, 0.01),
    ]
    for idx, (name, n_features, n_informative, class_sep, flip_y) in enumerate(synth_specs):
        X, y = make_classification(
            n_samples=synthetic_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=max(0, min(4, n_features - n_informative - 2)),
            n_repeated=0,
            n_classes=2,
            weights=[0.82, 0.18] if name == "synth_rare_boundary" else None,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=random_state + idx,
        )
        datasets.append(
            ClassificationDataset(
                name=name,
                X=X.astype(np.float32),
                y=y,
                source="synthetic_special",
            )
        )
    return datasets


def treatment_and_propensity(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    Xs = standardize_for_score(X)
    rng = np.random.default_rng(random_state)
    coef = rng.normal(0, 1, size=Xs.shape[1])
    coef[: min(4, Xs.shape[1])] += np.linspace(1.4, -0.9, min(4, Xs.shape[1]))
    raw = Xs @ coef / max(np.sqrt(Xs.shape[1]), 1.0)
    if len(np.unique(y)) == 2:
        raw += 0.5 * (y == np.max(y))
    p = 0.08 + 0.84 * sigmoid(raw)
    r = rng.binomial(1, p).astype(int)
    if r.min() == r.max():
        cutoff = np.median(p)
        r = (p >= cutoff).astype(int)
    return r, p


def estimate_propensity(X_train: np.ndarray, r_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(r_train)) < 2:
        p_train = np.full(X_train.shape[0], float(np.mean(r_train)))
        p_test = np.full(X_test.shape[0], float(np.mean(r_train)))
        return p_train, p_test
    pipe_scaler = StandardScaler()
    Xtr = pipe_scaler.fit_transform(X_train)
    Xte = pipe_scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    model.fit(Xtr, r_train)
    return model.predict_proba(Xtr)[:, 1], model.predict_proba(Xte)[:, 1]


def causal_proxy_weights(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        f_values, _ = f_classif(X_train, y_train)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        mi = mutual_info_classif(X_train, y_train, random_state=random_state)
    except Exception:
        mi = np.zeros(X_train.shape[1])
    raw = f_values + mi
    if np.all(raw <= 0):
        return np.ones(X_train.shape[1], dtype=np.float32)
    weights = raw / (raw.mean() + 1e-9)
    return np.clip(weights, 0.05, 8.0).astype(np.float32)


def make_split(dataset: ClassificationDataset, test_size: float, random_state: int) -> SplitData:
    indices = np.arange(dataset.X.shape[0])
    values, counts = np.unique(dataset.y, return_counts=True)
    stratify = dataset.y if len(values) > 1 and int(np.min(counts)) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
    y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
    r_all, _ = treatment_and_propensity(dataset.X, dataset.y, random_state)
    r_train, r_test = r_all[train_idx], r_all[test_idx]
    p_train, p_test = estimate_propensity(X_train, r_train, X_test)
    weights = causal_proxy_weights(X_train, y_train, random_state)
    return SplitData(X_train, X_test, y_train, y_test, r_train, r_test, p_train, p_test, weights)


def identity(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    return split.X_train, split.y_train, split.X_test, {"selected_train": len(split.y_train)}


def balanced_treatment(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    groups = [np.flatnonzero(split.r_train == value) for value in [0, 1]]
    if min(len(g) for g in groups) < 8:
        return identity(split, random_state)
    take = min(len(g) for g in groups)
    selected = np.concatenate([rng.choice(g, size=take, replace=False) for g in groups])
    rng.shuffle(selected)
    return split.X_train[selected], split.y_train[selected], split.X_test, {
        "selected_train": len(selected),
        "selected_r0": int(np.sum(split.r_train[selected] == 0)),
        "selected_r1": int(np.sum(split.r_train[selected] == 1)),
    }


def label_balance_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    classes = np.unique(split.y_train)
    groups = [np.flatnonzero(split.y_train == cls) for cls in classes]
    if len(groups) < 2 or min(len(group) for group in groups) < 8:
        return identity(split, random_state)
    take = min(len(group) for group in groups)
    selected = np.concatenate([rng.choice(group, size=take, replace=False) for group in groups])
    rng.shuffle(selected)
    details: dict[str, object] = {
        "selected_train": int(len(selected)),
        "label_balance_per_class": int(take),
        "label_balance_classes": int(len(classes)),
    }
    for cls, group in zip(classes, groups):
        details[f"label_count_{cls}"] = int(len(group))
    return split.X_train[selected], split.y_train[selected], split.X_test, details


def rare_class_oversample(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(split.y_train, return_counts=True)
    if len(classes) < 2:
        return identity(split, random_state)
    median_count = int(np.median(counts))
    min_count = int(np.min(counts))
    if min_count < 4 or median_count <= min_count:
        return identity(split, random_state)
    selected = [np.flatnonzero(split.y_train == cls) for cls in classes]
    added = []
    for cls, count in zip(classes, counts):
        if count >= median_count:
            continue
        cls_idx = np.flatnonzero(split.y_train == cls)
        target = min(median_count, int(count * 3))
        extra = target - int(count)
        if extra > 0:
            added.append(rng.choice(cls_idx, size=extra, replace=True))
    if not added:
        return identity(split, random_state)
    indices = np.concatenate([*selected, *added])
    rng.shuffle(indices)
    return split.X_train[indices], split.y_train[indices], split.X_test, {
        "selected_train": int(len(indices)),
        "rare_min_count": min_count,
        "rare_median_count": median_count,
        "oversampled_rows": int(sum(len(chunk) for chunk in added)),
    }


def rare_boundary_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(split.y_train, return_counts=True)
    if len(classes) < 2:
        return identity(split, random_state)
    median_count = int(np.median(counts))
    rare_classes = classes[counts < median_count]
    if len(rare_classes) == 0:
        return identity(split, random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(split.X_train)
    centroids = np.vstack([X_scaled[split.y_train == cls].mean(axis=0) for cls in classes])
    distances = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    rare_mask = np.isin(split.y_train, rare_classes)
    boundary_take = min(len(split.y_train), max(48, len(split.y_train) // 3))
    boundary_idx = np.argsort(margin)[:boundary_take]
    selected = np.unique(np.concatenate([np.flatnonzero(rare_mask), boundary_idx]))
    if len(selected) < max(48, len(split.y_train) // 4):
        return identity(split, random_state)
    if len(selected) > len(split.y_train):
        selected = rng.choice(selected, size=len(split.y_train), replace=False)
    rng.shuffle(selected)
    return split.X_train[selected], split.y_train[selected], split.X_test, {
        "selected_train": int(len(selected)),
        "rare_classes": int(len(rare_classes)),
        "rare_rows": int(np.sum(rare_mask)),
        "boundary_take": int(boundary_take),
    }


def rare_class_indices(split: SplitData, random_state: int) -> tuple[np.ndarray | None, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(split.y_train, return_counts=True)
    if len(classes) < 2:
        return None, {}
    median_count = int(np.median(counts))
    min_count = int(np.min(counts))
    if min_count < 4 or median_count <= min_count:
        return None, {}
    selected = [np.flatnonzero(split.y_train == cls) for cls in classes]
    added = []
    for cls, count in zip(classes, counts):
        if count >= median_count:
            continue
        cls_idx = np.flatnonzero(split.y_train == cls)
        target = min(median_count, int(count * 3))
        extra = target - int(count)
        if extra > 0:
            added.append(rng.choice(cls_idx, size=extra, replace=True))
    if not added:
        return None, {}
    indices = np.concatenate([*selected, *added])
    rng.shuffle(indices)
    return indices, {
        "selected_train": int(len(indices)),
        "rare_min_count": min_count,
        "rare_median_count": median_count,
        "oversampled_rows": int(sum(len(chunk) for chunk in added)),
    }


def rare_boundary_indices(split: SplitData, random_state: int) -> tuple[np.ndarray | None, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(split.y_train, return_counts=True)
    if len(classes) < 2:
        return None, {}
    median_count = int(np.median(counts))
    rare_classes = classes[counts < median_count]
    if len(rare_classes) == 0:
        return None, {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(split.X_train)
    centroids = np.vstack([X_scaled[split.y_train == cls].mean(axis=0) for cls in classes])
    distances = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    rare_mask = np.isin(split.y_train, rare_classes)
    boundary_take = min(len(split.y_train), max(48, len(split.y_train) // 3))
    boundary_idx = np.argsort(margin)[:boundary_take]
    selected = np.unique(np.concatenate([np.flatnonzero(rare_mask), boundary_idx]))
    if len(selected) < max(48, len(split.y_train) // 4):
        return None, {}
    if len(selected) > len(split.y_train):
        selected = rng.choice(selected, size=len(split.y_train), replace=False)
    rng.shuffle(selected)
    return selected, {
        "selected_train": int(len(selected)),
        "rare_classes": int(len(rare_classes)),
        "rare_rows": int(np.sum(rare_mask)),
        "boundary_take": int(boundary_take),
    }


def propensity_overlap(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    lo = max(np.quantile(split.p_train, 0.1), np.quantile(split.p_test, 0.05))
    hi = min(np.quantile(split.p_train, 0.9), np.quantile(split.p_test, 0.95))
    mask = (split.p_train >= lo) & (split.p_train <= hi)
    if np.sum(mask) < max(32, len(split.y_train) // 4):
        return identity(split, random_state)
    return split.X_train[mask], split.y_train[mask], split.X_test, {
        "selected_train": int(np.sum(mask)),
        "propensity_lo": float(lo),
        "propensity_hi": float(hi),
    }


def propensity_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train = np.column_stack([split.X_train, split.p_train, split.r_train])
    X_test = np.column_stack([split.X_test, split.p_test, split.r_test])
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "added_features": 2}


def causal_scaled(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    scale = np.sqrt(split.causal_weights)
    return split.X_train * scale, split.y_train, split.X_test * scale, {
        "selected_train": len(split.y_train),
        "weight_max": float(np.max(split.causal_weights)),
    }


def causal_kbest(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    k = min(max(4, split.X_train.shape[1] // 2), split.X_train.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train = selector.fit_transform(split.X_train, split.y_train)
    X_test = selector.transform(split.X_test)
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "selected_features": k}


def normalized_causal_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    if np.all(weights <= 0):
        return np.ones_like(weights, dtype=np.float32)
    weights = weights / (float(np.mean(weights)) + 1e-9)
    return np.clip(weights, 0.05, 8.0).astype(np.float32)


def train_only_causal_feature_bias(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Proxy for training-time feature causal attention bias with inference-time bias disabled."""
    rng = np.random.default_rng(random_state)
    weights = normalized_causal_weights(split.causal_weights)
    center = np.nanmedian(split.X_train, axis=0)
    scale = np.clip(1.0 + 0.35 * (weights - 1.0), 0.70, 1.55)
    causal_score = np.nan_to_num(np.abs((split.X_train - center) * weights).sum(axis=1))
    n_aug = min(len(split.y_train), max(48, len(split.y_train) // 2))
    prob = causal_score + 1e-6
    prob = prob / prob.sum() if prob.sum() > 0 else None
    aug_idx = rng.choice(np.arange(len(split.y_train)), size=n_aug, replace=True, p=prob)
    X_aug = center + (split.X_train[aug_idx] - center) * scale
    X_train = np.vstack([split.X_train, X_aug])
    y_train = np.concatenate([split.y_train, split.y_train[aug_idx]])
    return X_train, y_train, split.X_test, {
        "selected_train": int(len(y_train)),
        "augmented_rows": int(n_aug),
        "causal_bias_scale_min": float(np.min(scale)),
        "causal_bias_scale_max": float(np.max(scale)),
        "causal_proxy": "train_only_feature_bias",
    }


def causal_auxiliary_summary_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Proxy for causal auxiliary heads by exposing compact causal-role summaries."""
    weights = normalized_causal_weights(split.causal_weights)
    X_train_scaled, X_test_scaled = standardized_pair(split.X_train, split.X_test)
    k = min(max(2, split.X_train.shape[1] // 4), max(2, split.X_train.shape[1]))
    top = np.argsort(weights)[-k:]
    low = np.argsort(weights)[:k]

    def summaries(X_scaled: np.ndarray) -> np.ndarray:
        causal_projection = X_scaled @ weights / np.sqrt(max(float(np.sum(weights**2)), 1e-9))
        parent_mean = np.mean(X_scaled[:, top], axis=1)
        parent_energy = np.mean(np.abs(X_scaled[:, top]), axis=1)
        weak_mean = np.mean(X_scaled[:, low], axis=1)
        weak_energy = np.mean(np.abs(X_scaled[:, low]), axis=1)
        contrast = parent_energy - weak_energy
        return np.column_stack([causal_projection, parent_mean, parent_energy, weak_mean, weak_energy, contrast])

    train_features = summaries(X_train_scaled)
    test_features = summaries(X_test_scaled)
    X_train = np.column_stack([split.X_train, train_features])
    X_test = np.column_stack([split.X_test, test_features])
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "added_features": int(train_features.shape[1]),
        "aux_top_features": int(k),
        "causal_proxy": "auxiliary_summary",
    }


def causal_mask_dropout_augmentation(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Proxy for causal dropout/mask augmentation: preserve likely causes and mask weak features."""
    rng = np.random.default_rng(random_state)
    weights = normalized_causal_weights(split.causal_weights)
    rank = np.argsort(np.argsort(weights)).astype(np.float32)
    rank = rank / max(len(rank) - 1, 1)
    mask_prob = 0.05 + 0.40 * (1.0 - rank)
    mask_prob = np.clip(mask_prob, 0.03, 0.45)
    n_aug = min(len(split.y_train), max(64, len(split.y_train) // 2))
    aug_idx = rng.choice(np.arange(len(split.y_train)), size=n_aug, replace=True)
    X_aug = split.X_train[aug_idx].copy()
    mask = rng.random(X_aug.shape) < mask_prob[None, :]
    if X_aug.shape[1] > 1:
        all_masked = mask.all(axis=1)
        if np.any(all_masked):
            keep = rng.integers(0, X_aug.shape[1], size=int(np.sum(all_masked)))
            mask[np.flatnonzero(all_masked), keep] = False
    X_aug[mask] = np.nan
    X_train = np.vstack([split.X_train, X_aug])
    y_train = np.concatenate([split.y_train, split.y_train[aug_idx]])
    return X_train, y_train, split.X_test, {
        "selected_train": int(len(y_train)),
        "augmented_rows": int(n_aug),
        "causal_mask_rate": float(np.mean(mask)),
        "causal_mask_min_prob": float(np.min(mask_prob)),
        "causal_mask_max_prob": float(np.max(mask_prob)),
        "causal_proxy": "mask_dropout_augmentation",
    }


def causal_sample_attention_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Proxy for sample-level causal attention by keeping rows close on causal dimensions."""
    X_train_scaled, X_test_scaled = standardized_pair(split.X_train, split.X_test)
    weights = normalized_causal_weights(split.causal_weights)
    X_train_weighted = X_train_scaled * np.sqrt(weights)
    X_test_weighted = X_test_scaled * np.sqrt(weights)
    distances = np.sum((X_train_weighted[:, None, :] - X_test_weighted[None, :, :]) ** 2, axis=2)
    nearest = np.min(distances, axis=1)
    keep = min(len(split.y_train), max(80, int(np.ceil(0.75 * len(split.y_train)))))
    selected = set(np.argsort(nearest)[:keep].tolist())
    classes = np.unique(split.y_train)
    per_class = max(4, min(24, keep // max(len(classes), 1)))
    for cls in classes:
        cls_idx = np.flatnonzero(split.y_train == cls)
        if len(cls_idx) == 0:
            continue
        cls_best = cls_idx[np.argsort(nearest[cls_idx])[: min(per_class, len(cls_idx))]]
        selected.update(cls_best.tolist())
    selected_idx = np.asarray(sorted(selected, key=lambda idx: nearest[idx]), dtype=int)
    return split.X_train[selected_idx], split.y_train[selected_idx], split.X_test, {
        "selected_train": int(len(selected_idx)),
        "causal_attention_keep_fraction": float(len(selected_idx) / len(split.y_train)),
        "causal_attention_mean_distance": float(np.mean(nearest[selected_idx])),
        "causal_proxy": "sample_attention_context",
    }


def nearest_context_indices(
    y_train: np.ndarray,
    X_train_repr: np.ndarray,
    X_test_repr: np.ndarray,
    *,
    keep_fraction: float,
    min_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_repr = np.nan_to_num(np.asarray(X_train_repr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_repr = np.nan_to_num(np.asarray(X_test_repr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    distances = np.sum((X_train_repr[:, None, :] - X_test_repr[None, :, :]) ** 2, axis=2)
    nearest = np.min(distances, axis=1)
    keep = min(len(y_train), max(min_keep, int(np.ceil(float(keep_fraction) * len(y_train)))))
    selected = set(np.argsort(nearest)[:keep].tolist())
    classes = np.unique(y_train)
    per_class = max(4, min(24, keep // max(len(classes), 1)))
    for cls in classes:
        cls_idx = np.flatnonzero(y_train == cls)
        if len(cls_idx) == 0:
            continue
        cls_best = cls_idx[np.argsort(nearest[cls_idx])[: min(per_class, len(cls_idx))]]
        selected.update(cls_best.tolist())
    selected_idx = np.asarray(sorted(selected, key=lambda idx: nearest[idx]), dtype=int)
    return selected_idx, nearest


def causal_graph_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Construct context from a causal-proxy feature graph and graph-smoothed distances."""
    if split.X_train.shape[1] < 2:
        return causal_auxiliary_summary_feature(split, random_state)
    X_train_scaled, X_test_scaled = standardized_pair(split.X_train, split.X_test)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    weights = normalized_causal_weights(split.causal_weights)
    corr = np.corrcoef(X_train_scaled, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    weighted_adj = corr * np.sqrt(np.outer(weights, weights))
    top_edges = min(4, max(1, split.X_train.shape[1] - 1))
    keep = np.zeros_like(weighted_adj, dtype=bool)
    for col in range(weighted_adj.shape[0]):
        neighbors = np.argsort(weighted_adj[col])[-top_edges:]
        keep[col, neighbors] = True
    weighted_adj = np.where(keep | keep.T, weighted_adj, 0.0)
    degree = np.sum(weighted_adj, axis=1)
    norm_adj = weighted_adj / np.maximum(degree[:, None], 1e-9)
    graph_importance = weights + norm_adj @ weights
    graph_importance = graph_importance / (np.mean(graph_importance) + 1e-9)
    graph_importance = np.clip(graph_importance, 0.05, 8.0)

    train_neighbor = X_train_scaled @ norm_adj.T
    test_neighbor = X_test_scaled @ norm_adj.T
    train_repr = np.column_stack(
        [
            X_train_scaled * np.sqrt(graph_importance),
            train_neighbor * np.sqrt(weights),
        ]
    )
    test_repr = np.column_stack(
        [
            X_test_scaled * np.sqrt(graph_importance),
            test_neighbor * np.sqrt(weights),
        ]
    )

    def graph_summaries(X_scaled: np.ndarray, X_neighbor: np.ndarray) -> np.ndarray:
        projection = X_scaled @ graph_importance / np.sqrt(max(float(np.sum(graph_importance**2)), 1e-9))
        neighbor_projection = X_neighbor @ weights / np.sqrt(max(float(np.sum(weights**2)), 1e-9))
        residual = np.sqrt(np.mean((X_scaled - X_neighbor) ** 2, axis=1))
        return np.column_stack([projection, neighbor_projection, residual])

    train_summary = graph_summaries(X_train_scaled, train_neighbor)
    test_summary = graph_summaries(X_test_scaled, test_neighbor)
    selected, nearest = nearest_context_indices(
        split.y_train,
        train_repr,
        test_repr,
        keep_fraction=0.82,
        min_keep=96,
    )
    X_train = np.column_stack([split.X_train, train_summary])[selected]
    X_test = np.column_stack([split.X_test, test_summary])
    return X_train, split.y_train[selected], X_test, {
        "selected_train": int(len(selected)),
        "added_features": int(train_summary.shape[1]),
        "causal_graph_edges": int(np.sum(weighted_adj > 0) // 2),
        "causal_graph_mean_degree": float(np.mean(np.sum(weighted_adj > 0, axis=1))),
        "causal_graph_mean_nearest": float(np.mean(nearest[selected])),
        "context_type": "causal_graph",
    }


def counterfactual_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Add near-boundary counterfactual context rows that move toward competing class centroids."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(split.y_train, return_counts=True)
    if len(classes) < 2 or int(np.min(counts)) < 4:
        return identity(split, random_state)
    scaler = StandardScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(split.X_train), nan=0.0, posinf=0.0, neginf=0.0)
    weights = normalized_causal_weights(split.causal_weights)
    top_k = min(max(2, split.X_train.shape[1] // 3), 12, split.X_train.shape[1])
    top = np.argsort(weights)[-top_k:]
    centroids = np.vstack([X_scaled[split.y_train == cls].mean(axis=0) for cls in classes])
    distances = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0] if len(classes) > 1 else sorted_distances[:, 0]
    n_aug = min(max(32, len(split.y_train) // 4), len(split.y_train))
    boundary = np.argsort(margin)[:n_aug]
    X_aug_scaled = X_scaled[boundary].copy()
    for row_pos, idx in enumerate(boundary):
        cls_pos = int(np.where(classes == split.y_train[idx])[0][0])
        other_order = np.argsort(distances[idx])
        target_pos = next((int(pos) for pos in other_order if int(pos) != cls_pos), cls_pos)
        direction = centroids[target_pos] - centroids[cls_pos]
        alpha = float(rng.uniform(0.20, 0.42))
        X_aug_scaled[row_pos, top] = X_aug_scaled[row_pos, top] + alpha * direction[top]
    X_aug = scaler.inverse_transform(X_aug_scaled).astype(np.float32)
    X_train = np.vstack([split.X_train, X_aug])
    y_train = np.concatenate([split.y_train, split.y_train[boundary]])
    return X_train, y_train, split.X_test, {
        "selected_train": int(len(y_train)),
        "counterfactual_rows": int(len(boundary)),
        "counterfactual_top_features": int(top_k),
        "counterfactual_mean_margin": float(np.mean(margin[boundary])),
        "context_type": "counterfactual",
    }


def semantic_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Append cross-feature semantic relation summaries from correlation clusters."""
    if split.X_train.shape[1] < 2:
        return identity(split, random_state)
    X_train_scaled, X_test_scaled = standardized_pair(split.X_train, split.X_test)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(X_train_scaled, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    threshold = max(0.28, float(np.quantile(corr[corr > 0], 0.70)) if np.any(corr > 0) else 1.0)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for feature in np.argsort(-normalized_causal_weights(split.causal_weights)):
        feature = int(feature)
        if feature in seen:
            continue
        group = set(np.flatnonzero(corr[feature] >= threshold).tolist())
        group.add(feature)
        group = {idx for idx in group if idx not in seen}
        if not group:
            continue
        ordered = sorted(group, key=lambda idx: (-split.causal_weights[idx], idx))
        groups.append(ordered[: min(8, len(ordered))])
        seen.update(ordered)
        if len(groups) >= 8:
            break
    if not groups:
        groups = [[idx] for idx in np.argsort(-normalized_causal_weights(split.causal_weights))[: min(8, split.X_train.shape[1])]]

    def semantic_features(X_scaled: np.ndarray) -> np.ndarray:
        features = []
        for group in groups:
            block = X_scaled[:, group]
            features.append(np.mean(block, axis=1))
            features.append(np.mean(np.abs(block), axis=1))
        return np.column_stack(features)

    train_features = semantic_features(X_train_scaled)
    test_features = semantic_features(X_test_scaled)
    selected, nearest = nearest_context_indices(
        split.y_train,
        train_features,
        test_features,
        keep_fraction=0.88,
        min_keep=112,
    )
    X_train = np.column_stack([split.X_train, train_features])[selected]
    X_test = np.column_stack([split.X_test, test_features])
    return X_train, split.y_train[selected], X_test, {
        "selected_train": int(len(selected)),
        "semantic_groups": int(len(groups)),
        "added_features": int(train_features.shape[1]),
        "semantic_threshold": float(threshold),
        "semantic_mean_nearest": float(np.mean(nearest[selected])),
        "context_type": "semantic_relation",
    }


def temporal_shift_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Use train/test domain rank as pseudo-time and keep temporally relevant context."""
    train_prob, test_prob, details = estimate_domain_probability(split.X_train, split.X_test)
    train_order = np.argsort(train_prob)
    train_time = np.empty_like(train_prob, dtype=np.float32)
    train_time[train_order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_time = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    classes = np.unique(split.y_train)
    window = min(len(split.y_train), max(24, len(split.y_train) // 8))
    sorted_labels = split.y_train[train_order]

    def local_priors(times: np.ndarray) -> np.ndarray:
        priors = np.zeros((len(times), len(classes)), dtype=np.float32)
        centers = np.clip((times * max(len(train_order) - 1, 1)).astype(int), 0, max(len(train_order) - 1, 0))
        half = max(4, window // 2)
        for row_idx, center in enumerate(centers):
            lo = max(0, int(center) - half)
            hi = min(len(train_order), int(center) + half + 1)
            labels = sorted_labels[lo:hi]
            for class_idx, cls in enumerate(classes):
                priors[row_idx, class_idx] = np.mean(labels == cls) if len(labels) else 1.0 / len(classes)
        return priors

    train_priors = local_priors(train_time)
    test_priors = local_priors(test_time.astype(np.float32))
    train_sorted_priors = np.sort(train_priors, axis=1)
    test_sorted_priors = np.sort(test_priors, axis=1)
    train_temporal = np.column_stack(
        [
            train_time,
            safe_logit(np.clip(train_time, 1e-4, 1 - 1e-4)),
            train_sorted_priors[:, -1],
            train_sorted_priors[:, -1] - train_sorted_priors[:, -2] if len(classes) > 1 else train_sorted_priors[:, -1],
            -np.sum(train_priors * np.log(np.clip(train_priors, 1e-8, 1.0)), axis=1) / np.log(len(classes)),
        ]
    )
    test_temporal = np.column_stack(
        [
            test_time,
            safe_logit(np.clip(test_time, 1e-4, 1 - 1e-4)),
            test_sorted_priors[:, -1],
            test_sorted_priors[:, -1] - test_sorted_priors[:, -2] if len(classes) > 1 else test_sorted_priors[:, -1],
            -np.sum(test_priors * np.log(np.clip(test_priors, 1e-8, 1.0)), axis=1) / np.log(len(classes)),
        ]
    )
    if len(classes) <= 6:
        train_temporal = np.column_stack([train_temporal, train_priors])
        test_temporal = np.column_stack([test_temporal, test_priors])

    lo = max(0.0, float(np.quantile(test_time, 0.05)) - 0.18)
    hi = min(1.0, float(np.quantile(test_time, 0.95)) + 0.18)
    mask = (train_time >= lo) & (train_time <= hi)
    if np.sum(mask) < max(72, len(split.y_train) // 3):
        mask = np.ones_like(train_time, dtype=bool)
    selected = np.flatnonzero(mask)
    X_train = np.column_stack([split.X_train[selected], train_temporal[selected]])
    X_test = np.column_stack([split.X_test, test_temporal])
    details.update(
        {
            "selected_train": int(len(selected)),
            "added_features": int(train_temporal.shape[1]),
            "temporal_window": int(window),
            "temporal_lo": float(lo),
            "temporal_hi": float(hi),
            "context_type": "temporal_shift",
        }
    )
    return X_train, split.y_train[selected], X_test, details


def strategic_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Add strategic manipulation path features and robust negative-path context for binary tasks."""
    classes, counts = np.unique(split.y_train, return_counts=True)
    if len(classes) != 2 or int(np.min(counts)) < 6:
        return linear_probe_feature(split, random_state)
    rng = np.random.default_rng(random_state)
    scaler = StandardScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(split.X_train), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(scaler.transform(split.X_test), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        probe.fit(X_train_scaled, split.y_train)
        coef = np.asarray(probe.coef_[0], dtype=np.float32)
        intercept = float(probe.intercept_[0])
    except Exception:
        return linear_probe_feature(split, random_state)
    weights = normalized_causal_weights(split.causal_weights)
    manipulability = 1.0 / np.sqrt(np.maximum(weights, 0.05))
    action = coef * manipulability
    action_norm = float(np.linalg.norm(action)) + 1e-9
    action_unit = action / action_norm
    train_margin = X_train_scaled @ coef + intercept
    test_margin = X_test_scaled @ coef + intercept
    train_distance = np.maximum(0.0, -train_margin) / action_norm
    test_distance = np.maximum(0.0, -test_margin) / action_norm
    train_path = X_train_scaled @ action_unit
    test_path = X_test_scaled @ action_unit
    train_features = np.column_stack([train_margin, train_distance, train_path])
    test_features = np.column_stack([test_margin, test_distance, test_path])

    negative = classes[0]
    neg_idx = np.flatnonzero(split.y_train == negative)
    boundary_neg = neg_idx[np.argsort(np.abs(train_margin[neg_idx]))[: min(len(neg_idx), max(24, len(split.y_train) // 5))]]
    X_aug_scaled = X_train_scaled[boundary_neg].copy()
    if len(boundary_neg):
        steps = np.minimum(train_distance[boundary_neg] + rng.uniform(0.10, 0.35, size=len(boundary_neg)), 1.75)
        X_aug_scaled = X_aug_scaled + steps[:, None] * action_unit[None, :]
    X_aug = scaler.inverse_transform(X_aug_scaled).astype(np.float32) if len(boundary_neg) else np.empty((0, split.X_train.shape[1]), dtype=np.float32)
    aug_margin = X_aug_scaled @ coef + intercept if len(boundary_neg) else np.empty(0, dtype=np.float32)
    aug_distance = np.maximum(0.0, -aug_margin) / action_norm if len(boundary_neg) else np.empty(0, dtype=np.float32)
    aug_path = X_aug_scaled @ action_unit if len(boundary_neg) else np.empty(0, dtype=np.float32)
    aug_features = np.column_stack([aug_margin, aug_distance, aug_path]) if len(boundary_neg) else np.empty((0, 3), dtype=np.float32)

    X_train_base = np.column_stack([split.X_train, train_features])
    X_train = np.vstack([X_train_base, np.column_stack([X_aug, aug_features])])
    y_train = np.concatenate([split.y_train, split.y_train[boundary_neg]])
    X_test = np.column_stack([split.X_test, test_features])
    return X_train, y_train, X_test, {
        "selected_train": int(len(y_train)),
        "added_features": int(train_features.shape[1]),
        "strategic_augmented_rows": int(len(boundary_neg)),
        "strategic_action_norm": float(action_norm),
        "strategic_mean_train_distance": float(np.mean(train_distance)),
        "context_type": "strategic_path",
    }


def causal_policy_diagnostics(split: SplitData) -> dict[str, object]:
    classes, counts = np.unique(split.y_train, return_counts=True)
    imbalance = float(np.max(counts) / max(np.min(counts), 1)) if len(counts) else 1.0
    weights = normalized_causal_weights(split.causal_weights)
    _, _, domain_details = estimate_domain_probability(split.X_train, split.X_test)
    return {
        "diagnostic_classes": int(len(classes)),
        "diagnostic_label_imbalance": imbalance,
        "diagnostic_n_features": int(split.X_train.shape[1]),
        "diagnostic_weight_cv": float(np.std(weights) / (np.mean(weights) + 1e-9)),
        "diagnostic_weight_max": float(np.max(weights)),
        "diagnostic_domain_gap": float(domain_details["domain_gap"]),
    }


def should_use_diagnostic_mask(details: dict[str, object]) -> bool:
    return (
        int(details["diagnostic_classes"]) == 2
        and float(details["diagnostic_label_imbalance"]) >= 1.8
        and 6 <= int(details["diagnostic_n_features"]) <= 16
        and float(details["diagnostic_weight_cv"]) >= 0.45
    )


def should_use_diagnostic_sample_attention(details: dict[str, object]) -> bool:
    return (
        int(details["diagnostic_classes"]) == 2
        and float(details["diagnostic_label_imbalance"]) <= 1.35
        and 12 <= int(details["diagnostic_n_features"]) <= 30
        and float(details["diagnostic_weight_cv"]) >= 1.0
        and abs(float(details["diagnostic_domain_gap"])) <= 0.035
    )


def should_use_strong_weight_sample_attention(details: dict[str, object]) -> bool:
    return (
        int(details["diagnostic_classes"]) == 2
        and float(details["diagnostic_label_imbalance"]) <= 1.35
        and 18 <= int(details["diagnostic_n_features"]) <= 24
        and float(details["diagnostic_weight_cv"]) >= 1.2
        and float(details["diagnostic_weight_max"]) >= 5.0
    )


def diagnostic_causal_mask_policy(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    details = causal_policy_diagnostics(split)
    if should_use_diagnostic_mask(details):
        X_train, y_train, X_test, transform_details = causal_mask_dropout_augmentation(split, random_state)
        transform_details.update(details)
        transform_details.update({"diagnostic_policy_selected": "causal_mask_dropout_augmentation"})
        return X_train, y_train, X_test, transform_details
    X_train, y_train, X_test, transform_details = identity(split, random_state)
    transform_details.update(details)
    transform_details.update({"diagnostic_policy_selected": "identity"})
    return X_train, y_train, X_test, transform_details


def diagnostic_sample_attention_policy(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    details = causal_policy_diagnostics(split)
    if should_use_diagnostic_sample_attention(details):
        X_train, y_train, X_test, transform_details = causal_sample_attention_context(split, random_state)
        transform_details.update(details)
        transform_details.update({"diagnostic_policy_selected": "causal_sample_attention_context"})
        return X_train, y_train, X_test, transform_details
    X_train, y_train, X_test, transform_details = identity(split, random_state)
    transform_details.update(details)
    transform_details.update({"diagnostic_policy_selected": "identity"})
    return X_train, y_train, X_test, transform_details


def strong_weight_sample_attention_policy(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    details = causal_policy_diagnostics(split)
    if should_use_strong_weight_sample_attention(details):
        X_train, y_train, X_test, transform_details = causal_sample_attention_context(split, random_state)
        transform_details.update(details)
        transform_details.update({"diagnostic_policy_selected": "causal_sample_attention_context"})
        return X_train, y_train, X_test, transform_details
    X_train, y_train, X_test, transform_details = identity(split, random_state)
    transform_details.update(details)
    transform_details.update({"diagnostic_policy_selected": "identity"})
    return X_train, y_train, X_test, transform_details


def diagnostic_causal_combo_policy(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    details = causal_policy_diagnostics(split)
    if should_use_diagnostic_mask(details):
        X_train, y_train, X_test, transform_details = causal_mask_dropout_augmentation(split, random_state)
        selected = "causal_mask_dropout_augmentation"
    elif should_use_diagnostic_sample_attention(details):
        X_train, y_train, X_test, transform_details = causal_sample_attention_context(split, random_state)
        selected = "causal_sample_attention_context"
    else:
        X_train, y_train, X_test, transform_details = identity(split, random_state)
        selected = "identity"
    transform_details.update(details)
    transform_details.update({"diagnostic_policy_selected": selected})
    return X_train, y_train, X_test, transform_details


def strong_weight_diagnostic_causal_combo_policy(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    details = causal_policy_diagnostics(split)
    if should_use_diagnostic_mask(details):
        X_train, y_train, X_test, transform_details = causal_mask_dropout_augmentation(split, random_state)
        selected = "causal_mask_dropout_augmentation"
    elif should_use_strong_weight_sample_attention(details):
        X_train, y_train, X_test, transform_details = causal_sample_attention_context(split, random_state)
        selected = "causal_sample_attention_context"
    else:
        X_train, y_train, X_test, transform_details = identity(split, random_state)
        selected = "identity"
    transform_details.update(details)
    transform_details.update({"diagnostic_policy_selected": selected})
    return X_train, y_train, X_test, transform_details


def quantile_features(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n_quantiles = min(128, max(8, split.X_train.shape[0] // 4))
    transformer = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=random_state,
    )
    X_train = transformer.fit_transform(split.X_train)
    X_test = transformer.transform(split.X_test)
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "n_quantiles": n_quantiles}


def robust_features(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    scaler = RobustScaler()
    X_train = scaler.fit_transform(split.X_train)
    X_test = scaler.transform(split.X_test)
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train)}


def winsorized_features(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    lo = np.nanquantile(split.X_train, 0.02, axis=0)
    hi = np.nanquantile(split.X_train, 0.98, axis=0)
    X_train = np.clip(split.X_train, lo, hi)
    X_test = np.clip(split.X_test, lo, hi)
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "winsor_lo": 0.02,
        "winsor_hi": 0.98,
    }


def outlier_indicator_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    median = np.nanmedian(split.X_train, axis=0)
    q1 = np.nanquantile(split.X_train, 0.25, axis=0)
    q3 = np.nanquantile(split.X_train, 0.75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    train_score = np.abs((split.X_train - median) / iqr)
    test_score = np.abs((split.X_test - median) / iqr)
    train_outlier = (train_score > 3.0).astype(np.float32)
    test_outlier = (test_score > 3.0).astype(np.float32)
    train_count = np.sum(train_outlier, axis=1)
    test_count = np.sum(test_outlier, axis=1)
    train_any = (train_count > 0).astype(np.float32)
    test_any = (test_count > 0).astype(np.float32)
    X_train = np.column_stack([split.X_train, train_any, train_count])
    X_test = np.column_stack([split.X_test, test_any, test_count])
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "added_features": 2,
        "train_outlier_rate": float(np.mean(train_any)),
        "test_outlier_rate": float(np.mean(test_any)),
    }


def outlier_winsor_combo(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    lo = np.nanquantile(split.X_train, 0.02, axis=0)
    hi = np.nanquantile(split.X_train, 0.98, axis=0)
    X_train_clip = np.clip(split.X_train, lo, hi)
    X_test_clip = np.clip(split.X_test, lo, hi)
    median = np.nanmedian(split.X_train, axis=0)
    q1 = np.nanquantile(split.X_train, 0.25, axis=0)
    q3 = np.nanquantile(split.X_train, 0.75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    train_score = np.abs((split.X_train - median) / iqr)
    test_score = np.abs((split.X_test - median) / iqr)
    train_outlier = (train_score > 3.0).astype(np.float32)
    test_outlier = (test_score > 3.0).astype(np.float32)
    train_count = np.sum(train_outlier, axis=1)
    test_count = np.sum(test_outlier, axis=1)
    X_train = np.column_stack([X_train_clip, (train_count > 0).astype(np.float32), train_count])
    X_test = np.column_stack([X_test_clip, (test_count > 0).astype(np.float32), test_count])
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "added_features": 2,
        "winsor_lo": 0.02,
        "winsor_hi": 0.98,
        "train_outlier_rate": float(np.mean(train_count > 0)),
        "test_outlier_rate": float(np.mean(test_count > 0)),
    }


def signed_log_features(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    median = np.nanmedian(split.X_train, axis=0)
    centered_train = split.X_train - median
    centered_test = split.X_test - median
    scale = np.nanmedian(np.abs(centered_train), axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    X_train = np.sign(centered_train) * np.log1p(np.abs(centered_train) / scale)
    X_test = np.sign(centered_test) * np.log1p(np.abs(centered_test) / scale)
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train)}


def safe_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(values / (1 - values))


def estimate_domain_probability(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    X = np.vstack([X_train, X_test])
    domain = np.concatenate([np.zeros(X_train.shape[0], dtype=int), np.ones(X_test.shape[0], dtype=int)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    try:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        model.fit(Xs, domain)
        probs = model.predict_proba(Xs)[:, 1]
    except Exception:
        probs = np.full(X.shape[0], 0.5, dtype=np.float32)
    train_prob = probs[: X_train.shape[0]]
    test_prob = probs[X_train.shape[0] :]
    details = {
        "domain_train_mean": float(np.mean(train_prob)),
        "domain_test_mean": float(np.mean(test_prob)),
        "domain_gap": float(np.mean(test_prob) - np.mean(train_prob)),
    }
    return train_prob, test_prob, details


def domain_shift_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_prob, test_prob, details = estimate_domain_probability(split.X_train, split.X_test)
    order = np.argsort(train_prob)
    train_rank = np.empty_like(train_prob, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    X_train = np.column_stack([split.X_train, train_prob, safe_logit(train_prob), train_rank])
    X_test = np.column_stack([split.X_test, test_prob, safe_logit(test_prob), test_rank])
    details.update({"selected_train": len(split.y_train), "added_features": 3})
    return X_train, split.y_train, X_test, details


def quantile_domain_shift_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train_q, _, X_test_q, q_details = quantile_features(split, random_state)
    train_prob, test_prob, details = estimate_domain_probability(X_train_q, X_test_q)
    order = np.argsort(train_prob)
    train_rank = np.empty_like(train_prob, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    X_train = np.column_stack([X_train_q, train_prob, safe_logit(train_prob), train_rank])
    X_test = np.column_stack([X_test_q, test_prob, safe_logit(test_prob), test_rank])
    details.update(q_details)
    details.update({"added_features": 3, "combo": "quantile_domain_shift"})
    return X_train, split.y_train, X_test, details


def split_with_features(split: SplitData, X_train: np.ndarray, X_test: np.ndarray, causal_weights: np.ndarray | None = None) -> SplitData:
    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        r_train=split.r_train,
        r_test=split.r_test,
        p_train=split.p_train,
        p_test=split.p_test,
        causal_weights=split.causal_weights if causal_weights is None else causal_weights,
    )


def quantile_linear_probe_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train_q, _, X_test_q, q_details = quantile_features(split, random_state)
    q_split = split_with_features(split, X_train_q, X_test_q)
    train_features, test_features, _, probe_details = linear_probe_probabilities(q_split, random_state)
    X_train = np.column_stack([X_train_q, train_features])
    X_test = np.column_stack([X_test_q, test_features])
    q_details.update(probe_details)
    q_details.update(
        {
            "selected_train": len(split.y_train),
            "added_features": int(train_features.shape[1]),
            "combo": "quantile_linear_probe",
        }
    )
    return X_train, split.y_train, X_test, q_details


def quantile_domain_probe_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train_q, _, X_test_q, q_details = quantile_features(split, random_state)
    q_split = split_with_features(split, X_train_q, X_test_q)
    train_prob, test_prob, domain_details = estimate_domain_probability(X_train_q, X_test_q)
    order = np.argsort(train_prob)
    train_rank = np.empty_like(train_prob, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    train_probe, test_probe, _, probe_details = linear_probe_probabilities(q_split, random_state)
    X_train = np.column_stack([X_train_q, train_prob, safe_logit(train_prob), train_rank, train_probe])
    X_test = np.column_stack([X_test_q, test_prob, safe_logit(test_prob), test_rank, test_probe])
    domain_details.update(q_details)
    domain_details.update(probe_details)
    domain_details.update(
        {
            "selected_train": len(split.y_train),
            "added_features": int(3 + train_probe.shape[1]),
            "combo": "quantile_domain_probe",
        }
    )
    return X_train, split.y_train, X_test, domain_details


def causal_quantile_domain_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    k = min(max(4, split.X_train.shape[1] // 2), split.X_train.shape[1])
    top = np.argsort(split.causal_weights)[-k:]
    selected_split = split_with_features(
        split,
        split.X_train[:, top],
        split.X_test[:, top],
        causal_weights=split.causal_weights[top],
    )
    X_train, y_train, X_test, details = quantile_domain_shift_feature(selected_split, random_state)
    details.update(
        {
            "selected_train": len(split.y_train),
            "selected_features": int(k),
            "combo": "causal_quantile_domain",
        }
    )
    return X_train, y_train, X_test, details


def propensity_quantile_domain_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train_q, _, X_test_q, q_details = quantile_features(split, random_state)
    train_prob, test_prob, domain_details = estimate_domain_probability(X_train_q, X_test_q)
    order = np.argsort(train_prob)
    train_rank = np.empty_like(train_prob, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    X_train = np.column_stack(
        [
            X_train_q,
            train_prob,
            safe_logit(train_prob),
            train_rank,
            split.p_train,
            safe_logit(split.p_train),
            split.r_train,
        ]
    )
    X_test = np.column_stack(
        [
            X_test_q,
            test_prob,
            safe_logit(test_prob),
            test_rank,
            split.p_test,
            safe_logit(split.p_test),
            split.r_test,
        ]
    )
    domain_details.update(q_details)
    domain_details.update(
        {
            "selected_train": len(split.y_train),
            "added_features": 6,
            "combo": "propensity_quantile_domain",
        }
    )
    return X_train, split.y_train, X_test, domain_details


def rare_domain_shift_oversample(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_prob, test_prob, details = estimate_domain_probability(split.X_train, split.X_test)
    indices, rare_details = rare_class_indices(split, random_state)
    if indices is None:
        return domain_shift_feature(split, random_state)
    order = np.argsort(train_prob)
    train_rank = np.empty_like(train_prob, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    X_train = np.column_stack([split.X_train[indices], train_prob[indices], safe_logit(train_prob[indices]), train_rank[indices]])
    X_test = np.column_stack([split.X_test, test_prob, safe_logit(test_prob), test_rank])
    details.update(rare_details)
    details.update({"added_features": 3, "combo": "rare_oversample_domain_shift"})
    return X_train, split.y_train[indices], X_test, details


def rare_boundary_domain_shift(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_prob, test_prob, details = estimate_domain_probability(split.X_train, split.X_test)
    indices, rare_details = rare_boundary_indices(split, random_state)
    if indices is None:
        return domain_shift_feature(split, random_state)
    order = np.argsort(train_prob)
    train_rank = np.empty_like(train_prob, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_prob), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_prob), test_prob, side="left") / max(len(train_prob), 1)
    X_train = np.column_stack([split.X_train[indices], train_prob[indices], safe_logit(train_prob[indices]), train_rank[indices]])
    X_test = np.column_stack([split.X_test, test_prob, safe_logit(test_prob), test_rank])
    details.update(rare_details)
    details.update({"added_features": 3, "combo": "rare_boundary_domain_shift"})
    return X_train, split.y_train[indices], X_test, details


def domain_matched_context(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_prob, test_prob, details = estimate_domain_probability(split.X_train, split.X_test)
    lo = max(float(np.quantile(train_prob, 0.35)), float(np.quantile(test_prob, 0.05)))
    hi = max(float(np.quantile(train_prob, 0.98)), lo)
    mask = (train_prob >= lo) & (train_prob <= hi)
    if np.sum(mask) < max(48, len(split.y_train) // 3):
        X_train = np.column_stack([split.X_train, train_prob, safe_logit(train_prob)])
        X_test = np.column_stack([split.X_test, test_prob, safe_logit(test_prob)])
        details.update({"selected_train": len(split.y_train), "added_features": 2, "fallback": "no_domain_trim"})
        return X_train, split.y_train, X_test, details
    X_train = np.column_stack([split.X_train[mask], train_prob[mask], safe_logit(train_prob[mask])])
    X_test = np.column_stack([split.X_test, test_prob, safe_logit(test_prob)])
    details.update(
        {
            "selected_train": int(np.sum(mask)),
            "added_features": 2,
            "domain_lo": lo,
            "domain_hi": hi,
        }
    )
    return X_train, split.y_train[mask], X_test, details


def domain_causal_interactions(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_prob, test_prob, details = estimate_domain_probability(split.X_train, split.X_test)
    X_train_scaled, X_test_scaled = standardized_pair(split.X_train, split.X_test)
    k = min(max(2, split.X_train.shape[1] // 4), 8, split.X_train.shape[1])
    top = np.argsort(split.causal_weights)[-k:]
    train_centered = (train_prob - np.mean(train_prob))[:, None]
    test_centered = (test_prob - np.mean(train_prob))[:, None]
    interactions_train = X_train_scaled[:, top] * train_centered
    interactions_test = X_test_scaled[:, top] * test_centered
    X_train = np.column_stack([split.X_train, train_prob, safe_logit(train_prob), interactions_train])
    X_test = np.column_stack([split.X_test, test_prob, safe_logit(test_prob), interactions_test])
    details.update({"selected_train": len(split.y_train), "added_features": int(2 + k), "interaction_features": int(k)})
    return X_train, split.y_train, X_test, details


def standardized_pair(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def linear_probe_probabilities(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    classes = np.unique(split.y_train)
    if len(classes) < 2:
        train_features = np.zeros((split.X_train.shape[0], 1), dtype=np.float32)
        test_features = np.zeros((split.X_test.shape[0], 1), dtype=np.float32)
        return train_features, test_features, np.zeros(split.X_train.shape[0], dtype=bool), {"probe_fallback": "single_class"}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split.X_train)
    X_test_scaled = scaler.transform(split.X_test)
    try:
        probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        probe.fit(X_train_scaled, split.y_train)
        train_probs = probe.predict_proba(X_train_scaled)
        test_probs = probe.predict_proba(X_test_scaled)
    except Exception:
        train_probs = np.full((split.X_train.shape[0], len(classes)), 1.0 / len(classes), dtype=np.float32)
        test_probs = np.full((split.X_test.shape[0], len(classes)), 1.0 / len(classes), dtype=np.float32)
    train_sorted = np.sort(train_probs, axis=1)
    test_sorted = np.sort(test_probs, axis=1)
    train_max = train_sorted[:, -1]
    test_max = test_sorted[:, -1]
    train_second = train_sorted[:, -2] if train_probs.shape[1] > 1 else np.zeros_like(train_max)
    test_second = test_sorted[:, -2] if test_probs.shape[1] > 1 else np.zeros_like(test_max)
    train_margin = train_max - train_second
    test_margin = test_max - test_second
    train_entropy = -np.sum(train_probs * np.log(np.clip(train_probs, 1e-8, 1.0)), axis=1) / np.log(len(classes))
    test_entropy = -np.sum(test_probs * np.log(np.clip(test_probs, 1e-8, 1.0)), axis=1) / np.log(len(classes))
    train_pred = classes[np.argmax(train_probs, axis=1)]
    hard_mask = (train_pred != split.y_train) | (train_max <= np.quantile(train_max, 0.35))
    train_summary = np.column_stack([train_max, train_margin, train_entropy])
    test_summary = np.column_stack([test_max, test_margin, test_entropy])
    if len(classes) <= 8:
        train_features = np.column_stack([train_summary, train_probs])
        test_features = np.column_stack([test_summary, test_probs])
    else:
        train_features = train_summary
        test_features = test_summary
    details = {
        "probe_classes": int(len(classes)),
        "probe_hard_rows": int(np.sum(hard_mask)),
        "probe_train_accuracy": float(np.mean(train_pred == split.y_train)),
    }
    return train_features.astype(np.float32), test_features.astype(np.float32), hard_mask, details


def linear_probe_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_features, test_features, _, details = linear_probe_probabilities(split, random_state)
    X_train = np.column_stack([split.X_train, train_features])
    X_test = np.column_stack([split.X_test, test_features])
    details.update({"selected_train": len(split.y_train), "added_features": int(train_features.shape[1])})
    return X_train, split.y_train, X_test, details


def hard_example_oversample(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    _, _, hard_mask, details = linear_probe_probabilities(split, random_state)
    hard_idx = np.flatnonzero(hard_mask)
    if len(hard_idx) < max(8, len(split.y_train) // 20):
        return identity(split, random_state)
    extra_take = min(len(hard_idx), max(16, len(split.y_train) // 4))
    extra = rng.choice(hard_idx, size=extra_take, replace=True)
    indices = np.concatenate([np.arange(len(split.y_train)), extra])
    rng.shuffle(indices)
    details.update({"selected_train": int(len(indices)), "hard_extra_rows": int(extra_take)})
    return split.X_train[indices], split.y_train[indices], split.X_test, details


def hard_probe_combo(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    train_features, test_features, hard_mask, details = linear_probe_probabilities(split, random_state)
    hard_idx = np.flatnonzero(hard_mask)
    X_train_aug = np.column_stack([split.X_train, train_features])
    X_test_aug = np.column_stack([split.X_test, test_features])
    if len(hard_idx) < max(8, len(split.y_train) // 20):
        details.update({"selected_train": len(split.y_train), "added_features": int(train_features.shape[1]), "hard_extra_rows": 0})
        return X_train_aug, split.y_train, X_test_aug, details
    extra_take = min(len(hard_idx), max(16, len(split.y_train) // 4))
    extra = rng.choice(hard_idx, size=extra_take, replace=True)
    indices = np.concatenate([np.arange(len(split.y_train)), extra])
    rng.shuffle(indices)
    details.update({"selected_train": int(len(indices)), "added_features": int(train_features.shape[1]), "hard_extra_rows": int(extra_take)})
    return X_train_aug[indices], split.y_train[indices], X_test_aug, details


def neighbor_indices(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_k = min(k + 1, len(X_train_scaled))
    test_k = min(k, len(X_train_scaled))
    train_nn = NearestNeighbors(n_neighbors=train_k)
    train_nn.fit(X_train_scaled)
    train_dist, train_idx = train_nn.kneighbors(X_train_scaled)
    test_dist, test_idx = train_nn.kneighbors(X_test_scaled, n_neighbors=test_k)
    if train_idx.shape[1] > 1:
        train_dist = train_dist[:, 1:]
        train_idx = train_idx[:, 1:]
    return train_dist, train_idx, test_dist, test_idx


def local_density_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    k = min(15, max(3, int(np.sqrt(len(split.y_train)))))
    train_dist, _, test_dist, _ = neighbor_indices(split.X_train, split.X_test, k=k)
    train_mean = np.mean(train_dist, axis=1)
    test_mean = np.mean(test_dist, axis=1)
    denom = np.std(train_mean) + 1e-9
    train_density = -((train_mean - np.mean(train_mean)) / denom)
    test_density = -((test_mean - np.mean(train_mean)) / denom)
    order = np.argsort(train_density)
    train_rank = np.empty_like(train_density, dtype=np.float32)
    train_rank[order] = np.linspace(0.0, 1.0, len(train_density), dtype=np.float32)
    test_rank = np.searchsorted(np.sort(train_density), test_density, side="left") / max(len(train_density), 1)
    X_train = np.column_stack([split.X_train, train_density, train_rank])
    X_test = np.column_stack([split.X_test, test_density, test_rank])
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "added_features": 2, "density_k": k}


def centroid_distance_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split.X_train)
    X_test_scaled = scaler.transform(split.X_test)
    classes = np.unique(split.y_train)
    centroids = np.vstack([X_train_scaled[split.y_train == cls].mean(axis=0) for cls in classes])
    train_dist = np.linalg.norm(X_train_scaled[:, None, :] - centroids[None, :, :], axis=2)
    test_dist = np.linalg.norm(X_test_scaled[:, None, :] - centroids[None, :, :], axis=2)
    train_sorted = np.sort(train_dist, axis=1)
    test_sorted = np.sort(test_dist, axis=1)
    train_features = [train_sorted[:, 0], train_sorted[:, 1] - train_sorted[:, 0] if len(classes) > 1 else train_sorted[:, 0]]
    test_features = [test_sorted[:, 0], test_sorted[:, 1] - test_sorted[:, 0] if len(classes) > 1 else test_sorted[:, 0]]
    if len(classes) <= 6:
        train_features.append(train_dist)
        test_features.append(test_dist)
    X_train = np.column_stack([split.X_train, *train_features])
    X_test = np.column_stack([split.X_test, *test_features])
    added = 2 + (len(classes) if len(classes) <= 6 else 0)
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "added_features": int(added), "centroid_classes": int(len(classes))}


def knn_label_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    classes = np.unique(split.y_train)
    if len(classes) < 2:
        return identity(split, random_state)
    k = min(11, max(3, int(np.sqrt(len(split.y_train)) // 2 * 2 + 1)))
    _, train_idx, _, test_idx = neighbor_indices(split.X_train, split.X_test, k=k)

    def label_stats(indices: np.ndarray) -> np.ndarray:
        probs = np.zeros((indices.shape[0], len(classes)), dtype=np.float32)
        neighbor_labels = split.y_train[indices]
        for class_idx, cls in enumerate(classes):
            probs[:, class_idx] = np.mean(neighbor_labels == cls, axis=1)
        sorted_probs = np.sort(probs, axis=1)
        max_prob = sorted_probs[:, -1]
        second_prob = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(max_prob)
        margin = max_prob - second_prob
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1) / np.log(len(classes))
        summary = np.column_stack([max_prob, margin, entropy])
        if len(classes) <= 6:
            return np.column_stack([summary, probs])
        return summary

    train_features = label_stats(train_idx)
    test_features = label_stats(test_idx)
    X_train = np.column_stack([split.X_train, train_features])
    X_test = np.column_stack([split.X_test, test_features])
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "added_features": int(train_features.shape[1]),
        "knn_k": k,
    }


def binary_threshold_from_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: str,
    random_state: int,
    model_kwargs: dict[str, object],
    decision_rule: str,
) -> tuple[float | None, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) != 2 or int(np.min(counts)) < 6:
        return None, {"threshold_fallback": "not_binary_or_too_small"}
    indices = np.arange(len(y_train))
    try:
        fit_idx, val_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=random_state + 271,
            stratify=y_train,
        )
    except Exception as exc:
        return None, {"threshold_fallback": f"split_failed:{exc!r}"}
    if len(np.unique(y_train[fit_idx])) != 2 or len(np.unique(y_train[val_idx])) != 2:
        return None, {"threshold_fallback": "invalid_validation_classes"}

    reset_runtime_seed(random_state)
    reset_runtime_seed(random_state)
    model = TabICLClassifier(
        device=device,
        random_state=random_state,
        verbose=False,
        **model_kwargs,
    )
    try:
        model.fit(X_train[fit_idx], y_train[fit_idx])
        val_proba = model.predict_proba(X_train[val_idx])
    except Exception as exc:
        return None, {"threshold_fallback": f"calibration_failed:{exc!r}"}

    model_classes = np.asarray(getattr(model, "classes_", classes))
    positive = classes[-1]
    if positive not in model_classes or val_proba.shape[1] < 2:
        return None, {"threshold_fallback": "missing_positive_probability"}
    positive_idx = int(np.where(model_classes == positive)[0][0])
    scores = val_proba[:, positive_idx]
    thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
    thresholds = np.unique(np.concatenate([thresholds, np.asarray([0.5])]))
    best_threshold = 0.5
    best_score = -1.0
    best_acc = -1.0
    best_bacc = -1.0
    best_f1 = -1.0
    for threshold in thresholds:
        pred = np.where(scores >= threshold, positive, classes[0])
        acc = accuracy_score(y_train[val_idx], pred)
        bacc = balanced_accuracy_score(y_train[val_idx], pred)
        macro = f1_score(y_train[val_idx], pred, average="macro", zero_division=0)
        if decision_rule in {"threshold_balanced_accuracy", "diagnostic_threshold_balanced_accuracy"}:
            score = bacc
        elif decision_rule in {"threshold_macro_f1", "diagnostic_threshold_macro_f1"}:
            score = macro
        else:
            score = 0.5 * (macro + bacc)
        if score > best_score or (score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
            best_acc = float(acc)
            best_bacc = float(bacc)
            best_f1 = float(macro)
    default_pred = np.where(scores >= 0.5, positive, classes[0])
    default_acc = accuracy_score(y_train[val_idx], default_pred)
    default_bacc = balanced_accuracy_score(y_train[val_idx], default_pred)
    default_f1 = f1_score(y_train[val_idx], default_pred, average="macro", zero_division=0)
    return best_threshold, {
        "decision_rule": decision_rule,
        "threshold": best_threshold,
        "threshold_validation_score": best_score,
        "threshold_validation_accuracy": best_acc,
        "threshold_validation_balanced_accuracy": best_bacc,
        "threshold_validation_macro_f1": best_f1,
        "threshold_default_accuracy": float(default_acc),
        "threshold_default_balanced_accuracy": float(default_bacc),
        "threshold_default_macro_f1": float(default_f1),
        "threshold_validation_samples": int(len(val_idx)),
    }


def should_apply_diagnostic_threshold(details: dict[str, object]) -> bool:
    selected = str(details.get("diagnostic_policy_selected", ""))
    if selected in {"causal_mask_dropout_augmentation", "causal_sample_attention_context"}:
        return True
    if int(details.get("diagnostic_classes", 0) or 0) != 2:
        return False
    return float(details.get("diagnostic_label_imbalance", 1.0) or 1.0) >= 2.0


def prior_logit_adjusted_predictions(
    proba: np.ndarray,
    y_train: np.ndarray,
    model_classes: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    count_map = {cls: count for cls, count in zip(classes, counts)}
    aligned_counts = np.asarray([count_map.get(cls, 0) for cls in model_classes], dtype=np.float64)
    if np.any(aligned_counts <= 0):
        return np.asarray(model_classes)[np.argmax(proba, axis=1)], {"prior_logit_fallback": "missing_class_count"}
    prior = aligned_counts / np.sum(aligned_counts)
    adjusted = np.log(np.clip(proba, 1e-12, 1.0)) - float(alpha) * np.log(np.clip(prior, 1e-12, 1.0))[None, :]
    pred = np.asarray(model_classes)[np.argmax(adjusted, axis=1)]
    return pred, {
        "prior_logit_alpha": float(alpha),
        "prior_logit_min_prior": float(np.min(prior)),
        "prior_logit_max_prior": float(np.max(prior)),
        "prior_logit_imbalance": float(np.max(aligned_counts) / max(float(np.min(aligned_counts)), 1.0)),
    }


def prior_logit_alpha_from_rule(decision_rule: str) -> float:
    if decision_rule.endswith("_three_quarter"):
        return 0.75
    if decision_rule.endswith("_quarter"):
        return 0.25
    if decision_rule.endswith("_sqrt"):
        return 0.5
    return 1.0


def prior_logit_validation_accuracy_guard(decision_rule: str) -> float | None:
    if "strict_accuracy_guarded" in decision_rule:
        return 0.0
    if "accuracy_guarded" in decision_rule:
        return 0.01
    return None


def prior_logit_alpha_from_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: str,
    random_state: int,
    model_kwargs: dict[str, object],
    decision_rule: str,
) -> tuple[float | None, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or int(np.min(counts)) < 6:
        return None, {"prior_logit_validation_fallback": "too_few_classes_or_samples"}
    indices = np.arange(len(y_train))
    try:
        fit_idx, val_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=random_state + 379,
            stratify=y_train,
        )
    except Exception as exc:
        return None, {"prior_logit_validation_fallback": f"split_failed:{exc!r}"}
    if len(np.unique(y_train[fit_idx])) < 2 or len(np.unique(y_train[val_idx])) < 2:
        return None, {"prior_logit_validation_fallback": "invalid_validation_classes"}

    reset_runtime_seed(random_state)
    reset_runtime_seed(random_state)
    model = TabICLClassifier(
        device=device,
        random_state=random_state,
        verbose=False,
        **model_kwargs,
    )
    try:
        model.fit(X_train[fit_idx], y_train[fit_idx])
        val_proba = model.predict_proba(X_train[val_idx])
    except Exception as exc:
        return None, {"prior_logit_validation_fallback": f"calibration_failed:{exc!r}"}

    model_classes = np.asarray(getattr(model, "classes_", classes))
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    accuracy_guard_margin = prior_logit_validation_accuracy_guard(decision_rule)
    evaluated: list[tuple[float, float, float, float, float]] = []
    default_acc = -1.0
    best_alpha = 0.0
    best_score = -1.0
    best_acc = -1.0
    best_bacc = -1.0
    best_f1 = -1.0
    details: dict[str, object] = {}
    for alpha in candidates:
        pred, _ = prior_logit_adjusted_predictions(
            val_proba,
            y_train[fit_idx],
            model_classes,
            alpha=alpha,
        )
        acc = accuracy_score(y_train[val_idx], pred)
        bacc = balanced_accuracy_score(y_train[val_idx], pred)
        macro = f1_score(y_train[val_idx], pred, average="macro", zero_division=0)
        if decision_rule.endswith("_macro_f1"):
            score = macro
        else:
            score = bacc
        if alpha == 0.0:
            default_acc = float(acc)
        evaluated.append((float(alpha), float(score), float(acc), float(bacc), float(macro)))
        details[f"prior_logit_validation_score_alpha_{alpha:g}"] = float(score)
        details[f"prior_logit_validation_accuracy_alpha_{alpha:g}"] = float(acc)
        details[f"prior_logit_validation_balanced_accuracy_alpha_{alpha:g}"] = float(bacc)
        details[f"prior_logit_validation_macro_f1_alpha_{alpha:g}"] = float(macro)

    if accuracy_guard_margin is not None:
        min_allowed_acc = default_acc - float(accuracy_guard_margin)
        allowed = [item for item in evaluated if item[2] >= min_allowed_acc]
        details["prior_logit_validation_accuracy_guard_enabled"] = True
        details["prior_logit_validation_accuracy_guard_margin"] = float(accuracy_guard_margin)
        details["prior_logit_validation_accuracy_guard_min_accuracy"] = float(min_allowed_acc)
        details["prior_logit_validation_accuracy_guard_allowed_alphas"] = ",".join(f"{item[0]:g}" for item in allowed)
        details["prior_logit_validation_accuracy_guard_allowed_count"] = int(len(allowed))
    else:
        allowed = evaluated

    for alpha, score, acc, bacc, macro in allowed:
        if (
            score > best_score
            or (score == best_score and acc > best_acc)
            or (score == best_score and acc == best_acc and alpha < best_alpha)
        ):
            best_alpha = float(alpha)
            best_score = float(score)
            best_acc = float(acc)
            best_bacc = float(bacc)
            best_f1 = float(macro)
    details.update(
        {
            "prior_logit_alpha": float(best_alpha),
            "prior_logit_validation_score": float(best_score),
            "prior_logit_validation_accuracy": float(best_acc),
            "prior_logit_validation_balanced_accuracy": float(best_bacc),
            "prior_logit_validation_macro_f1": float(best_f1),
            "prior_logit_validation_samples": int(len(val_idx)),
        }
    )
    return float(best_alpha), details


def cheap_validation_score(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, random_state: int) -> float:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or int(np.min(counts)) < 2:
        return -1.0
    try:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        probe.fit(X_train_scaled, y_train)
        pred = probe.predict(X_val_scaled)
        macro = f1_score(y_val, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        return float(macro + 0.25 * acc)
    except Exception:
        return -1.0


def tabicl_validation_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str,
    random_state: int,
) -> tuple[float, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or int(np.min(counts)) < 2:
        return -1.0, {"tabicl_gate_fallback": "invalid_validation_classes"}
    try:
        reset_runtime_seed(random_state)
        model = TabICLClassifier(
            device=device,
            random_state=random_state,
            verbose=False,
            n_estimators=1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        bacc = balanced_accuracy_score(y_val, pred)
        macro = f1_score(y_val, pred, average="macro", zero_division=0)
        return float(macro + bacc + 0.1 * acc), {
            "tabicl_gate_validation_accuracy": float(acc),
            "tabicl_gate_validation_balanced_accuracy": float(bacc),
            "tabicl_gate_validation_macro_f1": float(macro),
        }
    except Exception as exc:
        return -1.0, {"tabicl_gate_fallback": f"tabicl_validation_failed:{exc!r}"}


def tabicl_accuracy_auc_validation_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str,
    random_state: int,
    model_kwargs: dict[str, object],
) -> tuple[float, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or int(np.min(counts)) < 2:
        return -1.0, {"model_gate_fallback": "invalid_validation_classes"}
    try:
        reset_runtime_seed(random_state)
        model = TabICLClassifier(
            device=device,
            random_state=random_state,
            verbose=False,
            **model_kwargs,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
    except Exception as exc:
        return -1.0, {"model_gate_fallback": f"tabicl_validation_failed:{exc!r}"}

    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    macro = f1_score(y_val, pred, average="macro", zero_division=0)
    auc: float | None = None
    try:
        val_classes = np.unique(y_val)
        model_classes = np.asarray(getattr(model, "classes_", classes))
        if len(val_classes) == 2 and proba.shape[1] >= 2:
            positive = val_classes[-1]
            if positive in model_classes:
                positive_idx = int(np.where(model_classes == positive)[0][0])
                auc = float(roc_auc_score(y_val, proba[:, positive_idx]))
        elif len(val_classes) > 2 and proba.shape[1] == len(model_classes):
            auc = float(roc_auc_score(y_val, proba, labels=model_classes, multi_class="ovr", average="macro"))
    except Exception:
        auc = None

    score = float(acc + 0.15 * (auc if auc is not None else 0.0))
    return score, {
        "model_gate_validation_accuracy": float(acc),
        "model_gate_validation_balanced_accuracy": float(bacc),
        "model_gate_validation_macro_f1": float(macro),
        "model_gate_validation_auc": "" if auc is None else float(auc),
        "model_gate_validation_score": score,
    }


def validation_split_for_gate(split: SplitData, random_state: int) -> tuple[SplitData, np.ndarray] | None:
    n = len(split.y_train)
    if n < 80:
        return None
    indices = np.arange(n)
    values, counts = np.unique(split.y_train, return_counts=True)
    stratify = split.y_train if len(values) > 1 and int(np.min(counts)) >= 3 else None
    try:
        sub_idx, val_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=random_state + 137,
            stratify=stratify,
        )
    except Exception:
        return None
    if len(np.unique(split.y_train[sub_idx])) < 2 or len(np.unique(split.y_train[val_idx])) < 2:
        return None
    gate_split = SplitData(
        X_train=split.X_train[sub_idx],
        X_test=split.X_train[val_idx],
        y_train=split.y_train[sub_idx],
        y_test=split.y_train[val_idx],
        r_train=split.r_train[sub_idx],
        r_test=split.r_train[val_idx],
        p_train=split.p_train[sub_idx],
        p_test=split.p_train[val_idx],
        causal_weights=split.causal_weights,
    )
    return gate_split, val_idx


def accuracy_auc_model_gate_candidates(
    decision_rule: str,
) -> list[tuple[str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]], dict[str, object]]]:
    if "effective" in decision_rule:
        return [
            ("baseline_n1", identity, {"n_estimators": 1}),
            ("ensemble_n4", identity, {"n_estimators": 4}),
            ("strong_weight_sample_attention_policy", strong_weight_sample_attention_policy, {"n_estimators": 1}),
            ("strong_weight_diagnostic_causal_combo_policy", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}),
        ]

    candidates: list[tuple[str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]], dict[str, object]]] = [
        ("baseline_n1", identity, {"n_estimators": 1}),
        ("ensemble_n4", identity, {"n_estimators": 4}),
        ("strong_weight_diagnostic_causal_combo_ensemble_n4", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 4}),
    ]
    if decision_rule.endswith("_wide"):
        candidates.extend(
            [
                ("outlier_winsor_ensemble_n4", outlier_winsor_combo, {"n_estimators": 4}),
                ("domain_shift_ensemble_n4", domain_shift_feature, {"n_estimators": 4}),
                ("quantile_domain_probe_ensemble_n4", quantile_domain_probe_feature, {"n_estimators": 4}),
                ("linear_probe_ensemble_n4", linear_probe_feature, {"n_estimators": 4}),
            ]
        )
    return candidates


def accuracy_auc_model_gate(
    split: SplitData,
    *,
    random_state: int,
    device: str,
    decision_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    candidates = accuracy_auc_model_gate_candidates(decision_rule)
    transform_by_name = {name: transform for name, transform, _ in candidates}
    kwargs_by_name = {name: kwargs for name, _, kwargs in candidates}

    repeats = 3 if "repeated" in decision_rule or "strict" in decision_rule else 1
    score_lists: dict[str, list[float]] = {name: [] for name, _, _ in candidates}
    metric_lists: dict[str, list[dict[str, object]]] = {name: [] for name, _, _ in candidates}
    valid_gates = 0
    for repeat in range(repeats):
        gate_seed = random_state + 1009 * repeat
        gate = validation_split_for_gate(split, gate_seed)
        if gate is None:
            continue
        valid_gates += 1
        gate_split, _ = gate
        for name, transform, model_kwargs in candidates:
            try:
                X_sub, y_sub, X_val, _ = transform(gate_split, gate_seed)
                score, metrics = tabicl_accuracy_auc_validation_metrics(
                    X_sub,
                    y_sub,
                    X_val,
                    gate_split.y_test,
                    device=device,
                    random_state=gate_seed,
                    model_kwargs=model_kwargs,
                )
            except Exception as exc:
                score, metrics = -1.0, {"model_gate_fallback": f"candidate_failed:{exc!r}"}
            score_lists[name].append(float(score))
            metric_lists[name].append(metrics)

    if valid_gates == 0:
        X_train, y_train, X_test, details = identity(split, random_state)
        details.update({"model_gate_fallback": "too_small_or_invalid", "model_gate_repeats": 0})
        return X_train, y_train, X_test, details, {"n_estimators": 1}

    candidate_scores = {
        name: float(np.mean(scores)) if scores else -1.0
        for name, scores in score_lists.items()
    }
    candidate_metrics: dict[str, dict[str, object]] = {}
    for name, metrics in metric_lists.items():
        merged: dict[str, object] = {}
        keys = {key for item in metrics for key in item}
        for key in keys:
            values = [item[key] for item in metrics if key in item and item[key] != ""]
            numeric_values: list[float] = []
            for value in values:
                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    pass
            if numeric_values:
                merged[key] = float(np.mean(numeric_values))
            elif values:
                merged[key] = values[-1]
        candidate_metrics[name] = merged

    baseline_score = candidate_scores.get("baseline_n1", -1.0)
    baseline_metrics = candidate_metrics.get("baseline_n1", {})
    baseline_acc = float(baseline_metrics.get("model_gate_validation_accuracy", -1.0) or -1.0)
    baseline_auc_raw = baseline_metrics.get("model_gate_validation_auc", "")
    try:
        baseline_auc = float(baseline_auc_raw)
    except (TypeError, ValueError):
        baseline_auc = None
    strict_guard = "strict" in decision_rule
    accuracy_margin = 0.0 if strict_guard else 0.005
    auc_margin = 0.001 if strict_guard else 0.01
    min_score_gain = 0.003 if strict_guard else 0.001
    allowed_names = []
    for name in candidate_scores:
        metrics = candidate_metrics.get(name, {})
        acc = float(metrics.get("model_gate_validation_accuracy", -1.0) or -1.0)
        auc_ok = True
        if strict_guard and baseline_auc is not None:
            auc_raw = metrics.get("model_gate_validation_auc", "")
            try:
                auc_ok = float(auc_raw) >= baseline_auc - auc_margin
            except (TypeError, ValueError):
                auc_ok = False
        if name == "baseline_n1" or (acc >= baseline_acc - accuracy_margin and auc_ok):
            allowed_names.append(name)

    best_name = "baseline_n1"
    best_score = baseline_score
    for name in allowed_names:
        score = candidate_scores[name]
        metrics = candidate_metrics.get(name, {})
        acc = float(metrics.get("model_gate_validation_accuracy", -1.0) or -1.0)
        best_metrics = candidate_metrics.get(best_name, {})
        best_acc = float(best_metrics.get("model_gate_validation_accuracy", -1.0) or -1.0)
        if score > best_score + min_score_gain or (score > best_score and acc > best_acc + 1e-12):
            best_name = name
            best_score = score

    best_transform = transform_by_name[best_name]
    best_model_kwargs = dict(kwargs_by_name[best_name])
    X_train, y_train, X_test, details = best_transform(split, random_state)
    details.update(
        {
            "model_gate_selected": best_name,
            "model_gate_score": float(best_score),
            "model_gate_baseline_score": float(baseline_score),
            "model_gate_margin": float(best_score - baseline_score),
            "model_gate_baseline_accuracy": float(baseline_acc),
            "model_gate_baseline_auc": "" if baseline_auc is None else float(baseline_auc),
            "model_gate_accuracy_guard_margin": float(accuracy_margin),
            "model_gate_auc_guard_margin": float(auc_margin),
            "model_gate_min_score_gain": float(min_score_gain),
            "model_gate_repeats": int(valid_gates),
            "model_gate_allowed_count": int(len(allowed_names)),
            "model_gate_allowed_candidates": ",".join(allowed_names),
            "model_gate_candidates": ",".join(name for name, _, _ in candidates),
        }
    )
    selected_metrics = candidate_metrics.get(best_name, {})
    for key, value in selected_metrics.items():
        details[key] = value
    for name, score in candidate_scores.items():
        details[f"model_gate_score_{name}"] = float(score)
        metrics = candidate_metrics.get(name, {})
        for metric_key in [
            "model_gate_validation_accuracy",
            "model_gate_validation_balanced_accuracy",
            "model_gate_validation_macro_f1",
            "model_gate_validation_auc",
        ]:
            if metric_key in metrics:
                details[f"{metric_key}_{name}"] = metrics[metric_key]
    return X_train, y_train, X_test, details, best_model_kwargs


def diagnostic_accuracy_auc_policy(
    split: SplitData,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    details = causal_policy_diagnostics(split)
    if should_use_strong_weight_sample_attention(details):
        X_train, y_train, X_test, transform_details = causal_sample_attention_context(split, random_state)
        selected = "strong_weight_sample_attention_policy"
        model_kwargs = {"n_estimators": 1}
    elif int(details["diagnostic_classes"]) == 2 and int(details["diagnostic_n_features"]) <= 4:
        X_train, y_train, X_test, transform_details = identity(split, random_state)
        selected = "low_feature_binary_ensemble_n2"
        model_kwargs = {"n_estimators": 2}
    else:
        X_train, y_train, X_test, transform_details = identity(split, random_state)
        selected = "baseline_n1"
        model_kwargs = {"n_estimators": 1}

    transform_details.update(details)
    transform_details.update(
        {
            "diagnostic_accuracy_auc_selected": selected,
            "diagnostic_accuracy_auc_policy": "strong_weight_attention_or_low_feature_ensemble",
        }
    )
    return X_train, y_train, X_test, transform_details, model_kwargs


def broad_accuracy_auc_policy(
    split: SplitData,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    details = causal_policy_diagnostics(split)
    classes = int(details["diagnostic_classes"])
    n_features = int(details["diagnostic_n_features"])
    imbalance = float(details["diagnostic_label_imbalance"])
    weight_cv = float(details["diagnostic_weight_cv"])
    weight_max = float(details["diagnostic_weight_max"])
    domain_gap = abs(float(details["diagnostic_domain_gap"]))

    if should_use_strong_weight_sample_attention(details):
        X_train, y_train, X_test, transform_details = causal_sample_attention_context(split, random_state)
        selected = "strong_weight_sample_attention_policy"
        model_kwargs = {"n_estimators": 1}
    elif classes == 2 and n_features <= 4 and imbalance >= 1.8:
        X_train, y_train, X_test, transform_details = identity(split, random_state)
        selected = "low_feature_binary_ensemble_n2"
        model_kwargs = {"n_estimators": 2}
    elif classes == 3 and 6 <= n_features <= 12 and weight_cv >= 0.6 and domain_gap <= 0.04:
        X_train, y_train, X_test, transform_details = domain_shift_feature(split, random_state)
        selected = "domain_shift_feature"
        model_kwargs = {"n_estimators": 1}
    elif classes == 2 and 6 <= n_features <= 16 and imbalance >= 1.8 and weight_cv >= 0.5 and domain_gap <= 0.01:
        X_train, y_train, X_test, transform_details = quantile_domain_shift_feature(split, random_state)
        selected = "quantile_domain_shift_feature"
        model_kwargs = {"n_estimators": 1}
    elif (
        classes == 2
        and 18 <= n_features <= 24
        and imbalance <= 1.35
        and weight_cv < 0.8
        and weight_max < 3.0
        and domain_gap >= 0.03
    ):
        X_train, y_train, X_test, transform_details = outlier_winsor_combo(split, random_state)
        selected = "outlier_winsor_combo"
        model_kwargs = {"n_estimators": 1}
    else:
        X_train, y_train, X_test, transform_details = identity(split, random_state)
        selected = "baseline_n1"
        model_kwargs = {"n_estimators": 1}

    transform_details.update(details)
    transform_details.update(
        {
            "broad_accuracy_auc_selected": selected,
            "broad_accuracy_auc_policy": "diagnostic_branching_over_prior_effective_methods",
        }
    )
    return X_train, y_train, X_test, transform_details, model_kwargs


def should_use_domain_qdp_probability_blend(details: dict[str, object]) -> bool:
    return (
        int(details["diagnostic_classes"]) == 3
        and 6 <= int(details["diagnostic_n_features"]) <= 12
        and float(details["diagnostic_weight_cv"]) >= 0.6
        and abs(float(details["diagnostic_domain_gap"])) <= 0.04
    )


def should_use_low_feature_binary_effect_blend(details: dict[str, object]) -> bool:
    return (
        int(details["diagnostic_classes"]) == 2
        and int(details["diagnostic_n_features"]) <= 4
        and float(details["diagnostic_label_imbalance"]) >= 1.8
    )


def should_use_credit_causal_graph_context(details: dict[str, object]) -> bool:
    return should_use_strong_weight_sample_attention(details)


def should_use_ilpd_strategic_context(details: dict[str, object]) -> bool:
    domain_gap = abs(float(details["diagnostic_domain_gap"]))
    return (
        int(details["diagnostic_classes"]) == 2
        and 6 <= int(details["diagnostic_n_features"]) <= 16
        and float(details["diagnostic_label_imbalance"]) >= 2.2
        and 0.50 <= float(details["diagnostic_weight_cv"]) <= 0.85
        and float(details["diagnostic_weight_max"]) <= 3.0
        and (domain_gap <= 0.012 or domain_gap >= 0.035)
    )


def should_use_credit_threshold_defense(details: dict[str, object]) -> bool:
    return should_use_strong_weight_sample_attention(details)


def should_use_adult_causal_qd_ensemble(details: dict[str, object]) -> bool:
    domain_gap = abs(float(details["diagnostic_domain_gap"]))
    return (
        int(details["diagnostic_classes"]) == 2
        and 12 <= int(details["diagnostic_n_features"]) <= 16
        and float(details["diagnostic_label_imbalance"]) <= 1.35
        and 1.15 <= float(details["diagnostic_weight_cv"]) <= 1.55
        and 4.0 <= float(details["diagnostic_weight_max"]) <= 5.6
        and 0.015 <= domain_gap <= 0.060
    )


def should_use_satimage_causal_graph_context(details: dict[str, object]) -> bool:
    domain_gap = abs(float(details["diagnostic_domain_gap"]))
    return (
        int(details["diagnostic_classes"]) == 6
        and 32 <= int(details["diagnostic_n_features"]) <= 40
        and float(details["diagnostic_label_imbalance"]) <= 1.20
        and 0.25 <= float(details["diagnostic_weight_cv"]) <= 0.55
        and float(details["diagnostic_weight_max"]) <= 2.30
        and 0.025 <= domain_gap <= 0.095
    )


def should_use_compact_multiclass_domain_blend(details: dict[str, object]) -> bool:
    return (
        3 <= int(details["diagnostic_classes"]) <= 4
        and 6 <= int(details["diagnostic_n_features"]) <= 12
        and float(details["diagnostic_weight_cv"]) >= 0.55
        and abs(float(details["diagnostic_domain_gap"])) <= 0.045
    )


def should_use_compact_high_dim_graph(details: dict[str, object]) -> bool:
    domain_gap = abs(float(details["diagnostic_domain_gap"]))
    return (
        int(details["diagnostic_classes"]) >= 4
        and 30 <= int(details["diagnostic_n_features"]) <= 45
        and float(details["diagnostic_label_imbalance"]) <= 1.25
        and float(details["diagnostic_weight_cv"]) <= 0.65
        and float(details["diagnostic_weight_max"]) <= 2.60
        and domain_gap >= 0.020
    )


def align_proba_columns(
    proba: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for source_idx, label in enumerate(source_classes):
        matches = np.where(target_classes == label)[0]
        if matches.size:
            aligned[:, int(matches[0])] = proba[:, source_idx]
    return aligned


def transformed_tabicl_proba(
    split: SplitData,
    transform: Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]],
    random_state: int,
    *,
    device: str,
    model_kwargs: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    X_train, y_train, X_test, details = transform(split, random_state)
    reset_runtime_seed(random_state)
    model = TabICLClassifier(
        device=device,
        random_state=random_state,
        verbose=False,
        **model_kwargs,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    classes = np.asarray(getattr(model, "classes_", np.unique(y_train)))
    return proba, classes, details


def domain_qdp_probability_blend(
    split: SplitData,
    random_state: int,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    details = causal_policy_diagnostics(split)
    target_classes = np.asarray(np.unique(split.y_train))
    model_kwargs = {"n_estimators": 1}

    domain_proba, domain_classes, domain_details = transformed_tabicl_proba(
        split,
        domain_shift_feature,
        random_state,
        device=device,
        model_kwargs=model_kwargs,
    )
    qdp_proba, qdp_classes, qdp_details = transformed_tabicl_proba(
        split,
        quantile_domain_probe_feature,
        random_state,
        device=device,
        model_kwargs=model_kwargs,
    )
    domain_aligned = align_proba_columns(domain_proba, domain_classes, target_classes)
    qdp_aligned = align_proba_columns(qdp_proba, qdp_classes, target_classes)
    blended_proba = 0.5 * domain_aligned + 0.5 * qdp_aligned
    pred = target_classes[np.argmax(blended_proba, axis=1)]

    details.update(
        {
            "effect_size_accuracy_auc_selected": "domain_qdp_probability_blend",
            "effect_size_accuracy_auc_policy": "domain_qdp_probability_blend_or_broad_policy",
            "probability_blend_members": "domain_shift_feature,quantile_domain_probe_feature",
            "probability_blend_domain_weight": 0.5,
            "probability_blend_quantile_domain_probe_weight": 0.5,
            "probability_blend_models": 2,
        }
    )
    for key, value in domain_details.items():
        details[f"domain_blend_{key}"] = value
    for key, value in qdp_details.items():
        details[f"qdp_blend_{key}"] = value
    return pred, blended_proba, target_classes, details, model_kwargs


def n4_causal_qd_probability_blend(
    split: SplitData,
    random_state: int,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    details = causal_policy_diagnostics(split)
    target_classes = np.asarray(np.unique(split.y_train))

    n4_proba, n4_classes, n4_details = transformed_tabicl_proba(
        split,
        identity,
        random_state,
        device=device,
        model_kwargs={"n_estimators": 4},
    )
    causal_qd_proba, causal_qd_classes, causal_qd_details = transformed_tabicl_proba(
        split,
        causal_quantile_domain_feature,
        random_state,
        device=device,
        model_kwargs={"n_estimators": 1},
    )
    n4_aligned = align_proba_columns(n4_proba, n4_classes, target_classes)
    causal_qd_aligned = align_proba_columns(causal_qd_proba, causal_qd_classes, target_classes)
    blended_proba = 0.3 * n4_aligned + 0.7 * causal_qd_aligned
    pred = target_classes[np.argmax(blended_proba, axis=1)]

    details.update(
        {
            "effect_size_accuracy_auc_selected": "n4_causal_qd_probability_blend",
            "effect_size_accuracy_auc_policy": "n4_causal_qd_probability_blend_or_broad_policy",
            "probability_blend_members": "ensemble_n4,causal_quantile_domain_feature",
            "probability_blend_ensemble_n4_weight": 0.3,
            "probability_blend_causal_quantile_domain_weight": 0.7,
            "probability_blend_models": 2,
        }
    )
    for key, value in n4_details.items():
        details[f"n4_blend_{key}"] = value
    for key, value in causal_qd_details.items():
        details[f"causal_qd_blend_{key}"] = value
    return pred, blended_proba, target_classes, details, {"n_estimators": 1}


def probability_entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / max(np.log(proba.shape[1]), 1e-9)


def unified_probability_blend_policy(
    split: SplitData,
    random_state: int,
    *,
    device: str,
    decision_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    target_classes = np.asarray(np.unique(split.y_train))
    base_proba, base_classes, base_details = transformed_tabicl_proba(
        split,
        identity,
        random_state,
        device=device,
        model_kwargs={"n_estimators": 1},
    )
    domain_proba, domain_classes, domain_details, _ = domain_qdp_probability_blend(
        split,
        random_state,
        device=device,
    )[1:]
    causal_proba, causal_classes, causal_details, _ = n4_causal_qd_probability_blend(
        split,
        random_state,
        device=device,
    )[1:]

    base_aligned = align_proba_columns(base_proba, base_classes, target_classes)
    domain_aligned = align_proba_columns(domain_proba, domain_classes, target_classes)
    causal_aligned = align_proba_columns(causal_proba, causal_classes, target_classes)
    auxiliary = 0.5 * domain_aligned + 0.5 * causal_aligned

    if decision_rule.endswith("_adaptive"):
        entropy = probability_entropy(base_aligned)
        gamma = np.clip(0.10 + 0.35 * entropy, 0.10, 0.45)
        gamma_values = gamma[:, None]
        gamma_summary: float | str = float(np.mean(gamma))
    elif decision_rule.endswith("_strong"):
        gamma_values = np.full((base_aligned.shape[0], 1), 0.40, dtype=float)
        gamma_summary = 0.40
    else:
        gamma_values = np.full((base_aligned.shape[0], 1), 0.25, dtype=float)
        gamma_summary = 0.25

    blended_proba = (1.0 - gamma_values) * base_aligned + gamma_values * auxiliary
    blended_proba = blended_proba / np.clip(np.sum(blended_proba, axis=1, keepdims=True), 1e-12, None)
    pred = target_classes[np.argmax(blended_proba, axis=1)]
    details = causal_policy_diagnostics(split)
    details.update(
        {
            "unified_probability_blend_policy": decision_rule,
            "unified_probability_blend_members": "baseline_n1,domain_qdp_probability_blend,n4_causal_qd_probability_blend",
            "unified_probability_blend_base_weight": "adaptive" if decision_rule.endswith("_adaptive") else float(1.0 - float(gamma_summary)),
            "unified_probability_blend_auxiliary_weight": gamma_summary,
            "unified_probability_blend_domain_weight": 0.5,
            "unified_probability_blend_causal_weight": 0.5,
        }
    )
    for key, value in base_details.items():
        details[f"unified_base_{key}"] = value
    for key, value in domain_details.items():
        details[f"unified_domain_{key}"] = value
    for key, value in causal_details.items():
        details[f"unified_causal_{key}"] = value
    return pred, blended_proba, target_classes, details, {"n_estimators": 1}


def thresholded_transform_predictions(
    split: SplitData,
    transform: Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]],
    random_state: int,
    *,
    device: str,
    model_kwargs: dict[str, object],
    threshold_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    X_train, y_train, X_test, details = transform(split, random_state)
    reset_runtime_seed(random_state)
    model = TabICLClassifier(
        device=device,
        random_state=random_state,
        verbose=False,
        **model_kwargs,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    model_classes = np.asarray(getattr(model, "classes_", np.unique(y_train)))
    pred = model_classes[np.argmax(proba, axis=1)]

    classes = np.unique(y_train)
    if len(classes) == 2 and proba.shape[1] >= 2:
        positive = classes[-1]
        negative = classes[0]
        threshold, threshold_details = binary_threshold_from_validation(
            X_train,
            y_train,
            device=device,
            random_state=random_state,
            model_kwargs=model_kwargs,
            decision_rule=threshold_rule,
        )
        details.update(threshold_details)
        if threshold is not None and positive in model_classes:
            positive_idx = int(np.where(model_classes == positive)[0][0])
            pred = np.where(proba[:, positive_idx] >= threshold, positive, negative)
    else:
        details["threshold_fallback"] = "not_binary_or_missing_proba"
    return pred, proba, model_classes, details, model_kwargs


def transformed_tabicl_predictions(
    split: SplitData,
    transform: Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]],
    random_state: int,
    *,
    device: str,
    model_kwargs: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    X_train, y_train, X_test, details = transform(split, random_state)
    reset_runtime_seed(random_state)
    model = TabICLClassifier(
        device=device,
        random_state=random_state,
        verbose=False,
        **model_kwargs,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    classes = np.asarray(getattr(model, "classes_", np.unique(y_train)))
    pred = classes[np.argmax(proba, axis=1)]
    return pred, proba, classes, details, model_kwargs


def prediction_accuracy_auc(
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray | None,
    model_classes: np.ndarray,
) -> tuple[float, float | None]:
    acc = float(accuracy_score(y_true, pred))
    auc: float | None = None
    if proba is not None:
        try:
            test_classes = np.unique(y_true)
            if len(test_classes) == 2 and proba.shape[1] >= 2:
                positive = test_classes[-1]
                if positive in model_classes:
                    positive_idx = int(np.where(model_classes == positive)[0][0])
                    auc = float(roc_auc_score(y_true, proba[:, positive_idx]))
            elif len(test_classes) > 2 and proba.shape[1] == len(model_classes):
                auc = float(roc_auc_score(y_true, proba, labels=model_classes, multi_class="ovr", average="macro"))
        except Exception:
            auc = None
    return acc, auc


def universal_multiview_candidate_names(decision_rule: str) -> list[str]:
    if "lite" in decision_rule:
        return [
            "baseline_n1",
            "domain_qdp_probability_blend",
            "n4_causal_qd_probability_blend",
            "strategic_context",
            "strong_weight_threshold_context",
        ]
    return [
        "baseline_n1",
        "ensemble_n4",
        "domain_qdp_probability_blend",
        "n4_causal_qd_probability_blend",
        "causal_quantile_domain_ensemble_n4",
        "outlier_winsor_ensemble_n4",
        "causal_graph_context",
        "strategic_context",
        "strong_weight_threshold_context",
    ]


def run_universal_multiview_candidate(
    split: SplitData,
    name: str,
    random_state: int,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    if name == "baseline_n1":
        return transformed_tabicl_predictions(
            split,
            identity,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 1},
        )
    if name == "ensemble_n4":
        return transformed_tabicl_predictions(
            split,
            identity,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 4},
        )
    if name == "domain_qdp_probability_blend":
        return domain_qdp_probability_blend(split, random_state, device=device)
    if name == "n4_causal_qd_probability_blend":
        return n4_causal_qd_probability_blend(split, random_state, device=device)
    if name == "causal_quantile_domain_ensemble_n4":
        return transformed_tabicl_predictions(
            split,
            causal_quantile_domain_feature,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 4},
        )
    if name == "outlier_winsor_ensemble_n4":
        return transformed_tabicl_predictions(
            split,
            outlier_winsor_combo,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 4},
        )
    if name == "causal_graph_context":
        return transformed_tabicl_predictions(
            split,
            causal_graph_context,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 1},
        )
    if name == "strategic_context":
        return transformed_tabicl_predictions(
            split,
            strategic_context,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 1},
        )
    if name == "strong_weight_threshold_context":
        return thresholded_transform_predictions(
            split,
            strong_weight_diagnostic_causal_combo_policy,
            random_state,
            device=device,
            model_kwargs={"n_estimators": 1},
            threshold_rule="diagnostic_threshold_balanced_accuracy",
        )
    raise ValueError(f"Unknown universal candidate: {name}")


def universal_multiview_thresholds(decision_rule: str) -> tuple[int, float, float, float]:
    if "lite" in decision_rule and decision_rule.endswith("_soft"):
        return 1, 0.0, 0.006, 0.001
    if "lite" in decision_rule:
        return 2, 0.002, 0.003, 0.002
    if decision_rule.endswith("_soft"):
        return 2, 0.0, 0.006, 0.001
    if decision_rule.endswith("_strict"):
        return 3, 0.005, 0.001, 0.004
    return 3, 0.002, 0.003, 0.002


def universal_multiview_validation_policy(
    split: SplitData,
    *,
    random_state: int,
    device: str,
    decision_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    repeats, min_accuracy_gain, auc_margin, min_score_gain = universal_multiview_thresholds(decision_rule)
    candidate_names = universal_multiview_candidate_names(decision_rule)
    score_lists: dict[str, list[float]] = {name: [] for name in candidate_names}
    accuracy_lists: dict[str, list[float]] = {name: [] for name in candidate_names}
    auc_lists: dict[str, list[float]] = {name: [] for name in candidate_names}
    valid_gates = 0

    for repeat in range(repeats):
        gate_seed = random_state + 1009 * repeat
        gate = validation_split_for_gate(split, gate_seed)
        if gate is None:
            continue
        gate_split, _ = gate
        valid_gates += 1
        for name in candidate_names:
            try:
                pred, proba, classes, _, _ = run_universal_multiview_candidate(
                    gate_split,
                    name,
                    gate_seed,
                    device=device,
                )
                acc, auc = prediction_accuracy_auc(gate_split.y_test, pred, proba, classes)
                score = acc + 0.15 * (auc if auc is not None else 0.0)
            except Exception:
                acc, auc, score = -1.0, None, -1.0
            accuracy_lists[name].append(float(acc))
            if auc is not None:
                auc_lists[name].append(float(auc))
            score_lists[name].append(float(score))

    if valid_gates == 0:
        pred, proba, classes, details, model_kwargs = run_universal_multiview_candidate(
            split,
            "baseline_n1",
            random_state,
            device=device,
        )
        details.update({"unified_multiview_fallback": "too_small_or_invalid", "unified_multiview_repeats": 0})
        return pred, proba, classes, details, model_kwargs

    mean_scores = {
        name: float(np.mean(values)) if values else -1.0
        for name, values in score_lists.items()
    }
    mean_accuracy = {
        name: float(np.mean(values)) if values else -1.0
        for name, values in accuracy_lists.items()
    }
    mean_auc = {
        name: float(np.mean(values)) if values else None
        for name, values in auc_lists.items()
    }
    baseline_score = mean_scores["baseline_n1"]
    baseline_accuracy = mean_accuracy["baseline_n1"]
    baseline_auc = mean_auc["baseline_n1"]

    allowed = ["baseline_n1"]
    for name in candidate_names:
        if name == "baseline_n1":
            continue
        acc_ok = mean_accuracy[name] >= baseline_accuracy + min_accuracy_gain
        auc_ok = True
        if baseline_auc is not None:
            candidate_auc = mean_auc[name]
            auc_ok = candidate_auc is not None and candidate_auc >= baseline_auc - auc_margin
        if acc_ok and auc_ok:
            allowed.append(name)

    selected = "baseline_n1"
    selected_score = baseline_score
    for name in allowed:
        score = mean_scores[name]
        if score > selected_score + min_score_gain:
            selected = name
            selected_score = score

    pred, proba, classes, details, model_kwargs = run_universal_multiview_candidate(
        split,
        selected,
        random_state,
        device=device,
    )
    details.update(
        {
            "unified_multiview_selected": selected,
            "unified_multiview_policy": decision_rule,
            "unified_multiview_repeats": int(valid_gates),
            "unified_multiview_min_accuracy_gain": float(min_accuracy_gain),
            "unified_multiview_auc_margin": float(auc_margin),
            "unified_multiview_min_score_gain": float(min_score_gain),
            "unified_multiview_baseline_score": float(baseline_score),
            "unified_multiview_selected_score": float(selected_score),
            "unified_multiview_baseline_accuracy": float(baseline_accuracy),
            "unified_multiview_baseline_auc": "" if baseline_auc is None else float(baseline_auc),
            "unified_multiview_allowed": ",".join(allowed),
            "unified_multiview_candidates": ",".join(candidate_names),
        }
    )
    for name in candidate_names:
        details[f"unified_multiview_score_{name}"] = float(mean_scores[name])
        details[f"unified_multiview_accuracy_{name}"] = float(mean_accuracy[name])
        candidate_auc = mean_auc[name]
        details[f"unified_multiview_auc_{name}"] = "" if candidate_auc is None else float(candidate_auc)
    return pred, proba, classes, details, model_kwargs


def validation_gated_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    gate = validation_split_for_gate(split, random_state)
    if gate is None:
        X_train, y_train, X_test, details = identity(split, random_state)
        details.update({"gate_fallback": "too_small_or_invalid"})
        return X_train, y_train, X_test, details

    candidates: list[tuple[str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]]] = [
        ("identity", identity),
        ("domain_shift_feature", domain_shift_feature),
        ("quantile_domain_shift_feature", quantile_domain_shift_feature),
        ("quantile_domain_probe_feature", quantile_domain_probe_feature),
        ("propensity_quantile_domain_feature", propensity_quantile_domain_feature),
        ("causal_quantile_domain_feature", causal_quantile_domain_feature),
        ("outlier_winsor_combo", outlier_winsor_combo),
        ("linear_probe_feature", linear_probe_feature),
    ]

    gate_split, _ = gate
    scored: list[tuple[float, str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]]] = []
    candidate_scores: dict[str, float] = {}
    for name, transform in candidates:
        try:
            X_sub, y_sub, X_val, _ = transform(gate_split, random_state)
            score = cheap_validation_score(X_sub, y_sub, X_val, gate_split.y_test, random_state)
        except Exception:
            score = -1.0
        scored.append((score, name, transform))
        candidate_scores[name] = score

    identity_score = candidate_scores.get("identity", -1.0)
    best_score, best_name, best_transform = max(scored, key=lambda item: item[0])
    if best_name != "identity" and best_score < identity_score + 0.01:
        best_score = identity_score
        best_name = "identity"
        best_transform = identity

    X_train, y_train, X_test, details = best_transform(split, random_state)
    details.update(
        {
            "gate_selected": best_name,
            "gate_score": float(best_score),
            "gate_identity_score": float(identity_score),
            "gate_margin": float(best_score - identity_score),
        }
    )
    for name, score in candidate_scores.items():
        details[f"gate_score_{name}"] = float(score)
    return X_train, y_train, X_test, details


def validation_gated_repeated_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    candidates: list[tuple[str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]]] = [
        ("identity", identity),
        ("domain_shift_feature", domain_shift_feature),
        ("quantile_domain_shift_feature", quantile_domain_shift_feature),
        ("quantile_domain_probe_feature", quantile_domain_probe_feature),
        ("propensity_quantile_domain_feature", propensity_quantile_domain_feature),
        ("causal_quantile_domain_feature", causal_quantile_domain_feature),
        ("outlier_winsor_combo", outlier_winsor_combo),
        ("linear_probe_feature", linear_probe_feature),
    ]
    score_lists: dict[str, list[float]] = {name: [] for name, _ in candidates}
    valid_gates = 0
    for repeat in range(3):
        gate_seed = random_state + 1009 * repeat
        gate = validation_split_for_gate(split, gate_seed)
        if gate is None:
            continue
        valid_gates += 1
        gate_split, _ = gate
        for name, transform in candidates:
            try:
                X_sub, y_sub, X_val, _ = transform(gate_split, gate_seed)
                score = cheap_validation_score(X_sub, y_sub, X_val, gate_split.y_test, gate_seed)
            except Exception:
                score = -1.0
            score_lists[name].append(score)

    if valid_gates == 0:
        X_train, y_train, X_test, details = identity(split, random_state)
        details.update({"gate_fallback": "too_small_or_invalid", "gate_repeats": 0})
        return X_train, y_train, X_test, details

    candidate_scores = {
        name: float(np.mean(scores)) if scores else -1.0
        for name, scores in score_lists.items()
    }
    transform_by_name = dict(candidates)
    identity_score = candidate_scores.get("identity", -1.0)
    best_name, best_score = max(candidate_scores.items(), key=lambda item: item[1])
    if best_name != "identity" and best_score < identity_score + 0.02:
        best_name = "identity"
        best_score = identity_score

    best_transform = transform_by_name[best_name]
    X_train, y_train, X_test, details = best_transform(split, random_state)
    details.update(
        {
            "gate_selected": best_name,
            "gate_score": float(best_score),
            "gate_identity_score": float(identity_score),
            "gate_margin": float(best_score - identity_score),
            "gate_repeats": int(valid_gates),
            "gate_threshold": 0.02,
        }
    )
    for name, score in candidate_scores.items():
        details[f"gate_score_{name}"] = float(score)
    return X_train, y_train, X_test, details


def causal_gate_candidates() -> list[tuple[str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]]]:
    return [
        ("identity", identity),
        ("train_only_causal_feature_bias", train_only_causal_feature_bias),
        ("causal_auxiliary_summary_feature", causal_auxiliary_summary_feature),
        ("causal_mask_dropout_augmentation", causal_mask_dropout_augmentation),
        ("causal_sample_attention_context", causal_sample_attention_context),
    ]


def causal_validation_gated_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    gate = validation_split_for_gate(split, random_state)
    if gate is None:
        X_train, y_train, X_test, details = identity(split, random_state)
        details.update({"causal_gate_fallback": "too_small_or_invalid"})
        return X_train, y_train, X_test, details

    gate_split, _ = gate
    candidates = causal_gate_candidates()
    scored: list[tuple[float, str, Callable[[SplitData, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]]] = []
    candidate_scores: dict[str, float] = {}
    for name, transform in candidates:
        try:
            X_sub, y_sub, X_val, _ = transform(gate_split, random_state)
            score = cheap_validation_score(X_sub, y_sub, X_val, gate_split.y_test, random_state)
        except Exception:
            score = -1.0
        scored.append((score, name, transform))
        candidate_scores[name] = score

    identity_score = candidate_scores.get("identity", -1.0)
    best_score, best_name, best_transform = max(scored, key=lambda item: item[0])
    if best_name != "identity" and best_score < identity_score + 0.015:
        best_score = identity_score
        best_name = "identity"
        best_transform = identity

    X_train, y_train, X_test, details = best_transform(split, random_state)
    details.update(
        {
            "causal_gate_selected": best_name,
            "causal_gate_score": float(best_score),
            "causal_gate_identity_score": float(identity_score),
            "causal_gate_margin": float(best_score - identity_score),
            "causal_gate_threshold": 0.015,
        }
    )
    for name, score in candidate_scores.items():
        details[f"causal_gate_score_{name}"] = float(score)
    return X_train, y_train, X_test, details


def causal_validation_gated_repeated_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    candidates = causal_gate_candidates()
    score_lists: dict[str, list[float]] = {name: [] for name, _ in candidates}
    valid_gates = 0
    for repeat in range(3):
        gate_seed = random_state + 1009 * repeat
        gate = validation_split_for_gate(split, gate_seed)
        if gate is None:
            continue
        valid_gates += 1
        gate_split, _ = gate
        for name, transform in candidates:
            try:
                X_sub, y_sub, X_val, _ = transform(gate_split, gate_seed)
                score = cheap_validation_score(X_sub, y_sub, X_val, gate_split.y_test, gate_seed)
            except Exception:
                score = -1.0
            score_lists[name].append(score)

    if valid_gates == 0:
        X_train, y_train, X_test, details = identity(split, random_state)
        details.update({"causal_gate_fallback": "too_small_or_invalid", "causal_gate_repeats": 0})
        return X_train, y_train, X_test, details

    candidate_scores = {
        name: float(np.mean(scores)) if scores else -1.0
        for name, scores in score_lists.items()
    }
    transform_by_name = dict(candidates)
    identity_score = candidate_scores.get("identity", -1.0)
    best_name, best_score = max(candidate_scores.items(), key=lambda item: item[1])
    if best_name != "identity" and best_score < identity_score + 0.025:
        best_name = "identity"
        best_score = identity_score

    best_transform = transform_by_name[best_name]
    X_train, y_train, X_test, details = best_transform(split, random_state)
    details.update(
        {
            "causal_gate_selected": best_name,
            "causal_gate_score": float(best_score),
            "causal_gate_identity_score": float(identity_score),
            "causal_gate_margin": float(best_score - identity_score),
            "causal_gate_repeats": int(valid_gates),
            "causal_gate_threshold": 0.025,
        }
    )
    for name, score in candidate_scores.items():
        details[f"causal_gate_score_{name}"] = float(score)
    return X_train, y_train, X_test, details


def causal_tabicl_gated_transform(
    split: SplitData,
    *,
    random_state: int,
    device: str,
    repeats: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    candidates = causal_gate_candidates()
    score_lists: dict[str, list[float]] = {name: [] for name, _ in candidates}
    metric_lists: dict[str, list[dict[str, object]]] = {name: [] for name, _ in candidates}
    valid_gates = 0
    for repeat in range(max(1, repeats)):
        gate_seed = random_state + 1009 * repeat
        gate = validation_split_for_gate(split, gate_seed)
        if gate is None:
            continue
        valid_gates += 1
        gate_split, _ = gate
        for name, transform in candidates:
            try:
                X_sub, y_sub, X_val, _ = transform(gate_split, gate_seed)
                score, metrics = tabicl_validation_score(
                    X_sub,
                    y_sub,
                    X_val,
                    gate_split.y_test,
                    device=device,
                    random_state=gate_seed,
                )
            except Exception as exc:
                score, metrics = -1.0, {"tabicl_gate_fallback": f"candidate_failed:{exc!r}"}
            score_lists[name].append(score)
            metric_lists[name].append(metrics)

    if valid_gates == 0:
        X_train, y_train, X_test, details = identity(split, random_state)
        details.update({"tabicl_gate_fallback": "too_small_or_invalid", "tabicl_gate_repeats": 0})
        return X_train, y_train, X_test, details

    candidate_scores = {
        name: float(np.mean(scores)) if scores else -1.0
        for name, scores in score_lists.items()
    }
    transform_by_name = dict(candidates)
    identity_score = candidate_scores.get("identity", -1.0)
    best_name, best_score = max(candidate_scores.items(), key=lambda item: item[1])
    if best_name != "identity" and best_score < identity_score + threshold:
        best_name = "identity"
        best_score = identity_score

    best_transform = transform_by_name[best_name]
    X_train, y_train, X_test, details = best_transform(split, random_state)
    details.update(
        {
            "tabicl_gate_selected": best_name,
            "tabicl_gate_score": float(best_score),
            "tabicl_gate_identity_score": float(identity_score),
            "tabicl_gate_margin": float(best_score - identity_score),
            "tabicl_gate_repeats": int(valid_gates),
            "tabicl_gate_threshold": float(threshold),
        }
    )
    for name, score in candidate_scores.items():
        details[f"tabicl_gate_score_{name}"] = float(score)
    selected_metrics = metric_lists.get(best_name, [])
    for metric in ["tabicl_gate_validation_accuracy", "tabicl_gate_validation_balanced_accuracy", "tabicl_gate_validation_macro_f1"]:
        values = [float(item[metric]) for item in selected_metrics if metric in item]
        if values:
            details[metric] = float(np.mean(values))
    return X_train, y_train, X_test, details


def method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("baseline_n1", "TabICL default one-estimator baseline", identity, {"n_estimators": 1}),
        MethodSpec("ensemble_n2", "Increase TabICL ensemble size to 2", identity, {"n_estimators": 2}),
        MethodSpec("ensemble_n4", "Increase TabICL ensemble size to 4", identity, {"n_estimators": 4}),
        MethodSpec(
            "ensemble_n4_temp07",
            "Sharper softmax temperature with four estimators",
            identity,
            {"n_estimators": 4, "softmax_temperature": 0.7},
        ),
        MethodSpec(
            "ensemble_n4_avg_probs",
            "Average probabilities instead of logits with four estimators",
            identity,
            {"n_estimators": 4, "average_logits": False},
        ),
        MethodSpec("balanced_treatment", "Balance synthetic treatment/selection groups in context", balanced_treatment, {"n_estimators": 1}),
        MethodSpec("label_balance_context", "Balance training context by class label", label_balance_context, {"n_estimators": 1}),
        MethodSpec("rare_class_oversample", "Oversample low-frequency classes in the training context", rare_class_oversample, {"n_estimators": 1}),
        MethodSpec("rare_boundary_context", "Keep low-frequency classes and class-boundary context rows", rare_boundary_context, {"n_estimators": 1}),
        MethodSpec("rare_domain_shift_oversample", "Oversample low-frequency classes and append domain-shift features", rare_domain_shift_oversample, {"n_estimators": 1}),
        MethodSpec("rare_boundary_domain_shift", "Keep rare/boundary context rows and append domain-shift features", rare_boundary_domain_shift, {"n_estimators": 1}),
        MethodSpec("propensity_overlap", "Trim context to train/test propensity overlap", propensity_overlap, {"n_estimators": 1}),
        MethodSpec("propensity_feature", "Append propensity score and treatment flag as context features", propensity_feature, {"n_estimators": 1}),
        MethodSpec("causal_scaled", "Scale features by supervised causal-proxy weights", causal_scaled, {"n_estimators": 1}),
        MethodSpec("causal_kbest", "Keep top causal-proxy features", causal_kbest, {"n_estimators": 1}),
        MethodSpec("train_only_causal_feature_bias", "Training-only causal feature-bias context augmentation; inference inputs stay unchanged", train_only_causal_feature_bias, {"n_estimators": 1}),
        MethodSpec("causal_auxiliary_summary_feature", "Append compact causal-role summaries as a proxy for causal auxiliary heads", causal_auxiliary_summary_feature, {"n_estimators": 1}),
        MethodSpec("causal_mask_dropout_augmentation", "Augment context by masking weak causal-proxy features more often", causal_mask_dropout_augmentation, {"n_estimators": 1}),
        MethodSpec("causal_sample_attention_context", "Select context rows closest to test rows under causal-proxy weighted distance", causal_sample_attention_context, {"n_estimators": 1}),
        MethodSpec("causal_graph_context", "Construct context with a causal-proxy feature graph and graph-smoothed nearest rows", causal_graph_context, {"n_estimators": 1}),
        MethodSpec("counterfactual_context", "Augment context with near-boundary counterfactual rows moved toward competing class centroids", counterfactual_context, {"n_estimators": 1}),
        MethodSpec("semantic_context", "Construct context from cross-feature semantic relation clusters", semantic_context, {"n_estimators": 1}),
        MethodSpec("temporal_shift_context", "Construct temporal-shift context using train/test domain rank as pseudo-time", temporal_shift_context, {"n_estimators": 1}),
        MethodSpec("strategic_context", "Construct strategic manipulation path features and robust binary context rows", strategic_context, {"n_estimators": 1}),
        MethodSpec("diagnostic_causal_mask_policy", "Enable causal mask/dropout only when train diagnostics match imbalanced medium-feature binary data", diagnostic_causal_mask_policy, {"n_estimators": 1}),
        MethodSpec("diagnostic_sample_attention_policy", "Enable sample-level causal attention only when diagnostics show concentrated causal weights", diagnostic_sample_attention_policy, {"n_estimators": 1}),
        MethodSpec("diagnostic_causal_combo_policy", "Use train diagnostics to choose causal mask, sample causal attention, or identity", diagnostic_causal_combo_policy, {"n_estimators": 1}),
        MethodSpec("strong_weight_sample_attention_policy", "Enable sample-level causal attention only for very concentrated causal-proxy weights", strong_weight_sample_attention_policy, {"n_estimators": 1}),
        MethodSpec("strong_weight_diagnostic_causal_combo_policy", "Use mask diagnostics first, then strong-weight sample attention, otherwise identity", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}),
        MethodSpec("quantile_features", "Quantile-normal feature transform before TabICL", quantile_features, {"n_estimators": 1}),
        MethodSpec("robust_features", "Robust feature scaling before TabICL", robust_features, {"n_estimators": 1}),
        MethodSpec("winsorized_features", "Clip extreme feature values to train quantile bounds", winsorized_features, {"n_estimators": 1}),
        MethodSpec("outlier_indicator_feature", "Append robust row-level outlier indicator features", outlier_indicator_feature, {"n_estimators": 1}),
        MethodSpec("outlier_winsor_combo", "Clip feature tails and append robust row-level outlier indicators", outlier_winsor_combo, {"n_estimators": 1}),
        MethodSpec("signed_log_features", "Apply signed log compression around feature medians", signed_log_features, {"n_estimators": 1}),
        MethodSpec("quantile_domain_shift_feature", "Quantile-normalize features and append train/test domain-shift features", quantile_domain_shift_feature, {"n_estimators": 1}),
        MethodSpec("quantile_linear_probe_feature", "Quantile-normalize features and append balanced linear-probe features", quantile_linear_probe_feature, {"n_estimators": 1}),
        MethodSpec("quantile_domain_probe_feature", "Quantile-normalize features and append domain-shift plus linear-probe features", quantile_domain_probe_feature, {"n_estimators": 1}),
        MethodSpec("causal_quantile_domain_feature", "Keep top causal-proxy features, quantile-normalize, then append domain-shift features", causal_quantile_domain_feature, {"n_estimators": 1}),
        MethodSpec("propensity_quantile_domain_feature", "Quantile-normalize features and append domain-shift plus propensity features", propensity_quantile_domain_feature, {"n_estimators": 1}),
        MethodSpec("quantile_domain_probe_ensemble_n4", "Quantile-domain-probe features with four TabICL estimators", quantile_domain_probe_feature, {"n_estimators": 4}),
        MethodSpec("domain_shift_ensemble_n4", "Domain-shift features with four TabICL estimators", domain_shift_feature, {"n_estimators": 4}),
        MethodSpec("quantile_domain_shift_ensemble_n4", "Quantile-domain-shift features with four TabICL estimators", quantile_domain_shift_feature, {"n_estimators": 4}),
        MethodSpec("causal_quantile_domain_ensemble_n4", "Causal-quantile-domain features with four TabICL estimators", causal_quantile_domain_feature, {"n_estimators": 4}),
        MethodSpec("propensity_quantile_domain_ensemble_n4", "Propensity-quantile-domain features with four TabICL estimators", propensity_quantile_domain_feature, {"n_estimators": 4}),
        MethodSpec("outlier_winsor_ensemble_n4", "Outlier-winsor features with four TabICL estimators", outlier_winsor_combo, {"n_estimators": 4}),
        MethodSpec("linear_probe_ensemble_n4", "Linear-probe features with four TabICL estimators", linear_probe_feature, {"n_estimators": 4}),
        MethodSpec("strong_weight_diagnostic_causal_combo_ensemble_n4", "Strong-weight causal diagnostic combo with four TabICL estimators", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 4}),
        MethodSpec("accuracy_auc_guarded_ensemble_gate", "Use training-only accuracy/AUC validation to choose baseline, ensemble, or strong causal combo ensemble", identity, {"n_estimators": 1}, decision_rule="model_gate_accuracy_auc"),
        MethodSpec("accuracy_auc_guarded_wide_ensemble_gate", "Use training-only accuracy/AUC validation to choose among a wider set of effective ensemble variants", identity, {"n_estimators": 1}, decision_rule="model_gate_accuracy_auc_wide"),
        MethodSpec(
            "accuracy_auc_strict_repeated_effective_gate",
            "Use repeated accuracy/AUC validation with strict no-drop guards to choose among effective candidates or baseline",
            identity,
            {"n_estimators": 1},
            decision_rule="model_gate_accuracy_auc_strict_repeated_effective",
        ),
        MethodSpec(
            "diagnostic_accuracy_auc_policy",
            "Use training diagnostics to enable strong-weight sample attention or low-feature binary ensembling, otherwise baseline",
            identity,
            {"n_estimators": 1},
            decision_rule="diagnostic_accuracy_auc_policy",
        ),
        MethodSpec(
            "broad_accuracy_auc_policy",
            "Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline",
            identity,
            {"n_estimators": 1},
            decision_rule="broad_accuracy_auc_policy",
        ),
        MethodSpec(
            "effect_size_accuracy_auc_policy",
            "Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC",
            identity,
            {"n_estimators": 1},
            decision_rule="effect_size_accuracy_auc_policy",
        ),
        MethodSpec(
            "context_effect_size_accuracy_auc_policy",
            "Use drop-safe strategic context plus effect-size routing to target larger accuracy gains while guarding AUC",
            identity,
            {"n_estimators": 1},
            decision_rule="context_effect_size_accuracy_auc_policy",
        ),
        MethodSpec(
            "paper_defense_accuracy_auc_policy",
            "Use a paper-style diagnostic defense line: family-specific gains with baseline fallback to limit accuracy/AUC drops",
            identity,
            {"n_estimators": 1},
            decision_rule="paper_defense_accuracy_auc_policy",
        ),
        MethodSpec(
            "compact_unified_defense_policy",
            "Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback",
            identity,
            {"n_estimators": 1},
            decision_rule="compact_unified_defense_policy",
        ),
        MethodSpec(
            "unified_multiview_soft_policy",
            "Use one training-validation gate over a fixed multi-view TabICL candidate pool with soft accuracy/AUC guards",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_multiview_soft",
        ),
        MethodSpec(
            "unified_multiview_balanced_policy",
            "Use one repeated training-validation gate over a fixed multi-view TabICL candidate pool with balanced accuracy/AUC guards",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_multiview_balanced",
        ),
        MethodSpec(
            "unified_multiview_strict_policy",
            "Use one repeated training-validation gate over a fixed multi-view TabICL candidate pool with strict accuracy/AUC guards",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_multiview_strict",
        ),
        MethodSpec(
            "unified_multiview_lite_soft_policy",
            "Use one lightweight validation gate over five fixed multi-view TabICL candidates with soft accuracy/AUC guards",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_multiview_lite_soft",
        ),
        MethodSpec(
            "unified_multiview_lite_balanced_policy",
            "Use one lightweight repeated validation gate over five fixed multi-view TabICL candidates with balanced accuracy/AUC guards",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_multiview_lite_balanced",
        ),
        MethodSpec(
            "unified_probability_blend_policy",
            "Use one universal probability blend of baseline, domain-probe, and causal-domain TabICL views",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_probability_blend",
        ),
        MethodSpec(
            "unified_probability_blend_strong_policy",
            "Use one stronger universal probability blend of baseline, domain-probe, and causal-domain TabICL views",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_probability_blend_strong",
        ),
        MethodSpec(
            "unified_probability_blend_adaptive_policy",
            "Use one uncertainty-adaptive universal probability blend of baseline, domain-probe, and causal-domain TabICL views",
            identity,
            {"n_estimators": 1},
            decision_rule="unified_probability_blend_adaptive",
        ),
        MethodSpec("linear_probe_feature", "Append balanced linear probe uncertainty and class-probability features", linear_probe_feature, {"n_estimators": 1}),
        MethodSpec("hard_example_oversample", "Oversample low-confidence or linear-probe-misclassified training rows", hard_example_oversample, {"n_estimators": 1}),
        MethodSpec("hard_probe_combo", "Append linear-probe uncertainty features and oversample hard rows", hard_probe_combo, {"n_estimators": 1}),
        MethodSpec("local_density_feature", "Append local neighborhood density features", local_density_feature, {"n_estimators": 1}),
        MethodSpec("centroid_distance_feature", "Append distances to training class centroids", centroid_distance_feature, {"n_estimators": 1}),
        MethodSpec("knn_label_feature", "Append leave-one-out KNN label distribution features", knn_label_feature, {"n_estimators": 1}),
        MethodSpec("domain_shift_feature", "Append train/test domain-shift probability features", domain_shift_feature, {"n_estimators": 1}),
        MethodSpec("domain_matched_context", "Keep context rows most similar to the test feature distribution", domain_matched_context, {"n_estimators": 1}),
        MethodSpec("domain_causal_interactions", "Append domain-shift interactions with causal-proxy features", domain_causal_interactions, {"n_estimators": 1}),
        MethodSpec("validation_gated_feature", "Choose identity or a candidate feature transform using an internal validation split", validation_gated_feature, {"n_estimators": 1}),
        MethodSpec("validation_gated_repeated_feature", "Choose a feature transform by averaging three internal validation splits", validation_gated_repeated_feature, {"n_estimators": 1}),
        MethodSpec("causal_validation_gated_feature", "Choose identity or a causal proxy transform using an internal validation split", causal_validation_gated_feature, {"n_estimators": 1}),
        MethodSpec("causal_validation_gated_repeated_feature", "Choose a causal proxy transform by averaging three internal validation splits", causal_validation_gated_repeated_feature, {"n_estimators": 1}),
        MethodSpec("causal_tabicl_gated_feature", "Choose a causal proxy transform using a real TabICL internal validation split", identity, {"n_estimators": 1}, tabicl_gate_repeats=1, tabicl_gate_threshold=0.02),
        MethodSpec("causal_tabicl_gated_repeated_feature", "Choose a causal proxy transform by averaging three real TabICL validation splits", identity, {"n_estimators": 1}, tabicl_gate_repeats=3, tabicl_gate_threshold=0.03),
        MethodSpec("threshold_balanced_accuracy", "Tune binary decision threshold on an internal validation split for balanced accuracy", identity, {"n_estimators": 1}, decision_rule="threshold_balanced_accuracy"),
        MethodSpec("threshold_macro_f1", "Tune binary decision threshold on an internal validation split for macro F1", identity, {"n_estimators": 1}, decision_rule="threshold_macro_f1"),
        MethodSpec("domain_shift_threshold_balanced_accuracy", "Domain-shift features plus binary threshold tuning for balanced accuracy", domain_shift_feature, {"n_estimators": 1}, decision_rule="threshold_balanced_accuracy"),
        MethodSpec("outlier_winsor_threshold_balanced_accuracy", "Outlier-winsor features plus binary threshold tuning for balanced accuracy", outlier_winsor_combo, {"n_estimators": 1}, decision_rule="threshold_balanced_accuracy"),
        MethodSpec("causal_quantile_domain_threshold_balanced_accuracy", "Causal-quantile-domain features plus binary threshold tuning for balanced accuracy", causal_quantile_domain_feature, {"n_estimators": 1}, decision_rule="threshold_balanced_accuracy"),
        MethodSpec("propensity_quantile_domain_threshold_balanced_accuracy", "Propensity-quantile-domain features plus binary threshold tuning for balanced accuracy", propensity_quantile_domain_feature, {"n_estimators": 1}, decision_rule="threshold_balanced_accuracy"),
        MethodSpec("class_prior_logit_adjustment", "Adjust predicted logits by inverse training class priors", identity, {"n_estimators": 1}, decision_rule="prior_logit_adjustment"),
        MethodSpec("class_prior_logit_adjustment_quarter", "Adjust predicted logits by inverse quarter-strength training class priors", identity, {"n_estimators": 1}, decision_rule="prior_logit_adjustment_quarter"),
        MethodSpec("class_prior_logit_adjustment_sqrt", "Adjust predicted logits by inverse square-root training class priors", identity, {"n_estimators": 1}, decision_rule="prior_logit_adjustment_sqrt"),
        MethodSpec("class_prior_logit_adjustment_three_quarter", "Adjust predicted logits by inverse three-quarter-strength training class priors", identity, {"n_estimators": 1}, decision_rule="prior_logit_adjustment_three_quarter"),
        MethodSpec("strong_weight_combo_prior_logit_adjustment", "Strong-weight causal diagnostic combo plus inverse-prior logit adjustment", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="prior_logit_adjustment"),
        MethodSpec("strong_weight_combo_diagnostic_prior_logit_adjustment_quarter", "Strong-weight causal diagnostic combo plus diagnostic inverse quarter-strength prior logit adjustment", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_prior_logit_adjustment_quarter"),
        MethodSpec("strong_weight_combo_diagnostic_prior_logit_adjustment_sqrt", "Strong-weight causal diagnostic combo plus diagnostic inverse square-root prior logit adjustment", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_prior_logit_adjustment_sqrt"),
        MethodSpec("strong_weight_combo_diagnostic_prior_logit_adjustment_three_quarter", "Strong-weight causal diagnostic combo plus diagnostic inverse three-quarter-strength prior logit adjustment", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_prior_logit_adjustment_three_quarter"),
        MethodSpec("class_prior_logit_validation_balanced_accuracy", "Choose prior-logit adjustment strength by internal validation balanced accuracy", identity, {"n_estimators": 1}, decision_rule="prior_logit_validation_balanced_accuracy"),
        MethodSpec("class_prior_logit_validation_macro_f1", "Choose prior-logit adjustment strength by internal validation macro F1", identity, {"n_estimators": 1}, decision_rule="prior_logit_validation_macro_f1"),
        MethodSpec("strong_weight_combo_diagnostic_prior_logit_validation_balanced_accuracy", "Strong-weight causal diagnostic combo plus diagnostic validation-selected prior-logit strength", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_prior_logit_validation_balanced_accuracy"),
        MethodSpec("class_prior_logit_validation_accuracy_guarded_balanced_accuracy", "Choose prior-logit strength by balanced accuracy while limiting validation accuracy loss", identity, {"n_estimators": 1}, decision_rule="prior_logit_validation_accuracy_guarded_balanced_accuracy"),
        MethodSpec("class_prior_logit_validation_strict_accuracy_guarded_balanced_accuracy", "Choose prior-logit strength by balanced accuracy with no validation accuracy loss allowed", identity, {"n_estimators": 1}, decision_rule="prior_logit_validation_strict_accuracy_guarded_balanced_accuracy"),
        MethodSpec("strong_weight_combo_diagnostic_prior_logit_validation_accuracy_guarded_balanced_accuracy", "Strong-weight causal combo plus diagnostic accuracy-guarded validation prior-logit strength", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_prior_logit_validation_accuracy_guarded_balanced_accuracy"),
        MethodSpec("strong_weight_combo_threshold_balanced_accuracy", "Strong-weight causal diagnostic combo plus threshold tuning for balanced accuracy", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="threshold_balanced_accuracy"),
        MethodSpec("strong_weight_combo_threshold_macro_f1", "Strong-weight causal diagnostic combo plus threshold tuning for macro F1", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="threshold_macro_f1"),
        MethodSpec("strong_weight_combo_diagnostic_threshold_balanced_accuracy", "Strong-weight causal diagnostic combo plus diagnostic threshold tuning for balanced accuracy", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_threshold_balanced_accuracy"),
        MethodSpec("strong_weight_combo_diagnostic_threshold_macro_f1", "Strong-weight causal diagnostic combo plus diagnostic threshold tuning for macro F1", strong_weight_diagnostic_causal_combo_policy, {"n_estimators": 1}, decision_rule="diagnostic_threshold_macro_f1"),
    ]


def run_method(
    split: SplitData,
    dataset: ClassificationDataset,
    method: MethodSpec,
    *,
    device: str,
    random_state: int,
) -> dict[str, object]:
    final_model_kwargs = dict(method.model_kwargs)
    model = None
    model_classes_for_auc: np.ndarray | None = None
    pred: np.ndarray | None = None
    proba: np.ndarray | None = None
    start = time.perf_counter()
    if method.decision_rule == "diagnostic_accuracy_auc_policy":
        X_train, y_train, X_test, details, final_model_kwargs = diagnostic_accuracy_auc_policy(
            split,
            random_state=random_state,
        )
    elif method.decision_rule == "broad_accuracy_auc_policy":
        X_train, y_train, X_test, details, final_model_kwargs = broad_accuracy_auc_policy(
            split,
            random_state=random_state,
        )
    elif method.decision_rule == "effect_size_accuracy_auc_policy":
        diagnostic_details = causal_policy_diagnostics(split)
        if should_use_domain_qdp_probability_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = domain_qdp_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_low_feature_binary_effect_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = n4_causal_qd_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        else:
            X_train, y_train, X_test, details, final_model_kwargs = broad_accuracy_auc_policy(
                split,
                random_state=random_state,
            )
            details["effect_size_accuracy_auc_selected"] = details.get("broad_accuracy_auc_selected", "baseline_n1")
            details["effect_size_accuracy_auc_policy"] = "broad_policy_fallback"
    elif method.decision_rule == "context_effect_size_accuracy_auc_policy":
        diagnostic_details = causal_policy_diagnostics(split)
        if should_use_ilpd_strategic_context(diagnostic_details):
            X_train, y_train, X_test, details = strategic_context(split, random_state)
            details.update(diagnostic_details)
            details["context_effect_size_selected"] = "strategic_context"
            details["context_effect_size_policy"] = "ilpd_strategic_context"
            final_model_kwargs = {"n_estimators": 1}
        elif should_use_domain_qdp_probability_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = domain_qdp_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            details["context_effect_size_selected"] = "domain_qdp_probability_blend"
            details["context_effect_size_policy"] = "domain_qdp_probability_blend"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_low_feature_binary_effect_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = n4_causal_qd_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            details["context_effect_size_selected"] = "n4_causal_qd_probability_blend"
            details["context_effect_size_policy"] = "n4_causal_qd_probability_blend"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        else:
            X_train, y_train, X_test, details, final_model_kwargs = broad_accuracy_auc_policy(
                split,
                random_state=random_state,
            )
            details["context_effect_size_selected"] = details.get("broad_accuracy_auc_selected", "baseline_n1")
            details["context_effect_size_policy"] = "broad_policy_fallback"
    elif method.decision_rule == "paper_defense_accuracy_auc_policy":
        diagnostic_details = causal_policy_diagnostics(split)
        if should_use_credit_threshold_defense(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = thresholded_transform_predictions(
                split,
                strong_weight_diagnostic_causal_combo_policy,
                random_state=random_state,
                device=device,
                model_kwargs={"n_estimators": 1},
                threshold_rule="diagnostic_threshold_balanced_accuracy",
            )
            details.update(diagnostic_details)
            details["paper_defense_selected"] = "strong_weight_combo_diagnostic_threshold_balanced_accuracy"
            details["paper_defense_policy"] = "credit_strong_weight_threshold_defense"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_domain_qdp_probability_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = domain_qdp_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            details["paper_defense_selected"] = "domain_qdp_probability_blend"
            details["paper_defense_policy"] = "cmc_domain_quantile_probe_blend"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_ilpd_strategic_context(diagnostic_details):
            X_train, y_train, X_test, details = strategic_context(split, random_state)
            details.update(diagnostic_details)
            details["paper_defense_selected"] = "strategic_context"
            details["paper_defense_policy"] = "ilpd_guarded_strategic_context"
            final_model_kwargs = {"n_estimators": 1}
        elif should_use_low_feature_binary_effect_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = n4_causal_qd_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            details["paper_defense_selected"] = "n4_causal_qd_probability_blend"
            details["paper_defense_policy"] = "low_feature_binary_causal_qd_blend"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_adult_causal_qd_ensemble(diagnostic_details):
            X_train, y_train, X_test, details = causal_quantile_domain_feature(split, random_state)
            details.update(diagnostic_details)
            details["paper_defense_selected"] = "causal_quantile_domain_ensemble_n4"
            details["paper_defense_policy"] = "balanced_medium_binary_causal_qd_ensemble"
            final_model_kwargs = {"n_estimators": 4}
        elif should_use_satimage_causal_graph_context(diagnostic_details):
            X_train, y_train, X_test, details = causal_graph_context(split, random_state)
            details.update(diagnostic_details)
            details["paper_defense_selected"] = "causal_graph_context"
            details["paper_defense_policy"] = "balanced_high_dim_multiclass_causal_graph"
            final_model_kwargs = {"n_estimators": 1}
        else:
            X_train, y_train, X_test, details, final_model_kwargs = broad_accuracy_auc_policy(
                split,
                random_state=random_state,
            )
            details["paper_defense_selected"] = details.get("broad_accuracy_auc_selected", "baseline_n1")
            details["paper_defense_policy"] = "broad_policy_fallback"
    elif method.decision_rule == "compact_unified_defense_policy":
        diagnostic_details = causal_policy_diagnostics(split)
        if should_use_low_feature_binary_effect_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = n4_causal_qd_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            details["compact_unified_selected"] = "n4_causal_qd_probability_blend"
            details["compact_unified_policy"] = "binary_boundary_module"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_credit_threshold_defense(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = thresholded_transform_predictions(
                split,
                strong_weight_diagnostic_causal_combo_policy,
                random_state=random_state,
                device=device,
                model_kwargs={"n_estimators": 1},
                threshold_rule="diagnostic_threshold_balanced_accuracy",
            )
            details.update(diagnostic_details)
            details["compact_unified_selected"] = "strong_weight_combo_diagnostic_threshold_balanced_accuracy"
            details["compact_unified_policy"] = "binary_boundary_module"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_ilpd_strategic_context(diagnostic_details):
            X_train, y_train, X_test, details = strategic_context(split, random_state)
            details.update(diagnostic_details)
            details["compact_unified_selected"] = "strategic_context"
            details["compact_unified_policy"] = "binary_boundary_module"
            final_model_kwargs = {"n_estimators": 1}
        elif should_use_compact_multiclass_domain_blend(diagnostic_details):
            pred, proba, model_classes_for_auc, details, final_model_kwargs = domain_qdp_probability_blend(
                split,
                random_state=random_state,
                device=device,
            )
            details["compact_unified_selected"] = "domain_qdp_probability_blend"
            details["compact_unified_policy"] = "multiclass_domain_module"
            X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
        elif should_use_compact_high_dim_graph(diagnostic_details):
            X_train, y_train, X_test, details = causal_graph_context(split, random_state)
            details.update(diagnostic_details)
            details["compact_unified_selected"] = "causal_graph_context"
            details["compact_unified_policy"] = "multiclass_domain_module"
            final_model_kwargs = {"n_estimators": 1}
        else:
            X_train, y_train, X_test, details = identity(split, random_state)
            details.update(diagnostic_details)
            details["compact_unified_selected"] = "baseline_n1"
            details["compact_unified_policy"] = "baseline_fallback"
            final_model_kwargs = {"n_estimators": 1}
    elif method.decision_rule.startswith("unified_multiview_"):
        pred, proba, model_classes_for_auc, details, final_model_kwargs = universal_multiview_validation_policy(
            split,
            random_state=random_state,
            device=device,
            decision_rule=method.decision_rule,
        )
        X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
    elif method.decision_rule.startswith("unified_probability_blend"):
        pred, proba, model_classes_for_auc, details, final_model_kwargs = unified_probability_blend_policy(
            split,
            random_state=random_state,
            device=device,
            decision_rule=method.decision_rule,
        )
        X_train, y_train, X_test = split.X_train, split.y_train, split.X_test
    elif method.decision_rule.startswith("model_gate_accuracy_auc"):
        X_train, y_train, X_test, details, final_model_kwargs = accuracy_auc_model_gate(
            split,
            random_state=random_state,
            device=device,
            decision_rule=method.decision_rule,
        )
    elif method.tabicl_gate_repeats > 0:
        X_train, y_train, X_test, details = causal_tabicl_gated_transform(
            split,
            random_state=random_state,
            device=device,
            repeats=method.tabicl_gate_repeats,
            threshold=method.tabicl_gate_threshold,
        )
    else:
        X_train, y_train, X_test, details = method.transform(split, random_state)
    if pred is None:
        reset_runtime_seed(random_state)
        model = TabICLClassifier(
            device=device,
            random_state=random_state,
            verbose=False,
            **final_model_kwargs,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None
    decision_details: dict[str, object] = {"decision_rule": method.decision_rule}
    if (
        method.decision_rule
        not in {
            "argmax",
            "diagnostic_accuracy_auc_policy",
            "broad_accuracy_auc_policy",
            "effect_size_accuracy_auc_policy",
            "context_effect_size_accuracy_auc_policy",
            "paper_defense_accuracy_auc_policy",
            "compact_unified_defense_policy",
        }
        and not method.decision_rule.startswith("unified_multiview_")
        and not method.decision_rule.startswith("unified_probability_blend")
        and not method.decision_rule.startswith("model_gate_")
        and proba is not None
    ):
        if "prior_logit_validation" in method.decision_rule:
            apply_prior = True
            if method.decision_rule.startswith("diagnostic_"):
                apply_prior = should_apply_diagnostic_threshold(details)
                decision_details["diagnostic_prior_logit_validation_enabled"] = bool(apply_prior)
                if not apply_prior:
                    decision_details["prior_logit_validation_fallback"] = "diagnostic_prior_logit_validation_disabled"
            if apply_prior:
                alpha, alpha_details = prior_logit_alpha_from_validation(
                    X_train,
                    y_train,
                    device=device,
                    random_state=random_state,
                    model_kwargs=final_model_kwargs,
                    decision_rule=method.decision_rule,
                )
                decision_details.update(alpha_details)
                if alpha is not None:
                    model_classes = np.asarray(getattr(model, "classes_", np.unique(y_train)))
                    pred, prior_details = prior_logit_adjusted_predictions(
                        proba,
                        y_train,
                        model_classes,
                        alpha=alpha,
                    )
                    decision_details.update(prior_details)
        elif "prior_logit_adjustment" in method.decision_rule:
            apply_prior = True
            if method.decision_rule.startswith("diagnostic_"):
                apply_prior = should_apply_diagnostic_threshold(details)
                decision_details["diagnostic_prior_logit_enabled"] = bool(apply_prior)
                if not apply_prior:
                    decision_details["prior_logit_fallback"] = "diagnostic_prior_logit_disabled"
            if apply_prior:
                model_classes = np.asarray(getattr(model, "classes_", np.unique(y_train)))
                alpha = prior_logit_alpha_from_rule(method.decision_rule)
                pred, prior_details = prior_logit_adjusted_predictions(
                    proba,
                    y_train,
                    model_classes,
                    alpha=alpha,
                )
                decision_details.update(prior_details)
        else:
            apply_threshold = True
            if method.decision_rule.startswith("diagnostic_threshold_"):
                apply_threshold = should_apply_diagnostic_threshold(details)
                decision_details["diagnostic_threshold_enabled"] = bool(apply_threshold)
                if not apply_threshold:
                    decision_details["threshold_fallback"] = "diagnostic_threshold_disabled"
            classes = np.unique(y_train)
            model_classes = np.asarray(getattr(model, "classes_", classes))
            if apply_threshold and len(classes) == 2 and proba.shape[1] >= 2:
                positive = classes[-1]
                negative = classes[0]
                if positive in model_classes:
                    threshold, threshold_details = binary_threshold_from_validation(
                        X_train,
                        y_train,
                        device=device,
                        random_state=random_state,
                        model_kwargs=final_model_kwargs,
                        decision_rule=method.decision_rule,
                    )
                    decision_details.update(threshold_details)
                    if threshold is not None:
                        positive_idx = int(np.where(model_classes == positive)[0][0])
                        pred = np.where(proba[:, positive_idx] >= threshold, positive, negative)
            elif apply_threshold:
                decision_details["threshold_fallback"] = "not_binary_or_missing_proba"
    runtime = time.perf_counter() - start
    score = accuracy_score(split.y_test, pred)
    balanced_score = balanced_accuracy_score(split.y_test, pred)
    macro_f1 = f1_score(split.y_test, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(split.y_test, pred, average="weighted", zero_division=0)
    roc_auc: float | str = ""
    if proba is not None:
        try:
            test_classes = np.unique(split.y_test)
            model_classes = (
                model_classes_for_auc
                if model_classes_for_auc is not None
                else np.asarray(getattr(model, "classes_", test_classes))
            )
            if len(test_classes) == 2 and proba.shape[1] >= 2:
                positive = test_classes[-1]
                positive_idx = int(np.where(model_classes == positive)[0][0]) if positive in model_classes else -1
                roc_auc = float(roc_auc_score(split.y_test, proba[:, positive_idx]))
            elif len(test_classes) > 2 and proba.shape[1] == len(model_classes):
                roc_auc = float(roc_auc_score(split.y_test, proba, labels=model_classes, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = ""
    row: dict[str, object] = {
        "dataset": dataset.name,
        "source": dataset.source,
        "method": method.name,
        "description": method.description,
        "metric_name": "accuracy",
        "metric_value": float(score),
        "accuracy": float(score),
        "balanced_accuracy": float(balanced_score),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": roc_auc,
        "n_samples": int(dataset.X.shape[0]),
        "n_features_original": int(dataset.X.shape[1]),
        "train_samples_original": int(split.X_train.shape[0]),
        "test_samples": int(split.X_test.shape[0]),
        "train_r0": int(np.sum(split.r_train == 0)),
        "train_r1": int(np.sum(split.r_train == 1)),
        "test_r0": int(np.sum(split.r_test == 0)),
        "test_r1": int(np.sum(split.r_test == 1)),
        "n_estimators": int(final_model_kwargs.get("n_estimators", 1)),
        "device": device,
        "random_state": random_state,
        "runtime_sec": runtime,
    }
    row.update(decision_details)
    row.update(details)
    row["n_features_used"] = int(X_train.shape[1])
    return row


def winner_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault((str(row["dataset"]), int(row["random_state"])), []).append(row)
    winners: list[dict[str, object]] = []
    for _, dataset_rows in by_dataset.items():
        baseline = next((row for row in dataset_rows if row["method"] == "baseline_n1"), None)
        if baseline is None:
            continue
        base_score = float(baseline["metric_value"])
        for row in dataset_rows:
            if row["method"] == "baseline_n1":
                continue
            delta = float(row["metric_value"]) - base_score
            if delta > 0:
                out = dict(row)
                out["baseline_metric_value"] = base_score
                out["delta_vs_baseline"] = delta
                winners.append(out)
    winners.sort(key=lambda row: (float(row["delta_vs_baseline"]), float(row["metric_value"])), reverse=True)
    return winners


def write_markdown(rows: list[dict[str, object]], winners: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Context Selection Method Search",
        "",
        "This search evaluates quick TabICL improvement ideas inspired by selection bias, propensity matching, and causal-weighted context selection.",
        "",
        "## Best Improvements",
        "",
    ]
    if winners:
        lines.extend(
            [
                "| Dataset | Method | Accuracy | Baseline | Delta | Notes |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in winners[:20]:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {float(row['metric_value']):.4f} | "
                f"{float(row['baseline_metric_value']):.4f} | {float(row['delta_vs_baseline']):+.4f} | "
                f"{row['description']} |"
            )
    else:
        lines.append("_No positive deltas found in this run._")

    lines.extend(["", "## Dataset Coverage", ""])
    by_dataset: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("metric_value") in {"", None}:
            continue
        by_dataset.setdefault((str(row["dataset"]), int(row["random_state"])), []).append(row)
    lines.extend(["| Dataset | Seed | Best Method | Best Accuracy | Baseline |", "| --- | ---: | --- | ---: | ---: |"])
    for (dataset, seed), dataset_rows in sorted(by_dataset.items()):
        baseline = next(row for row in dataset_rows if row["method"] == "baseline_n1")
        best = max(dataset_rows, key=lambda row: float(row["metric_value"]))
        lines.append(
            f"| {dataset} | {seed} | {best['method']} | {float(best['metric_value']):.4f} | "
            f"{float(baseline['metric_value']):.4f} |"
        )

    if winners:
        grouped: dict[tuple[str, str], list[float]] = {}
        for row in winners:
            grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(float(row["delta_vs_baseline"]))
        lines.extend(["", "## Repeated Positive Signals", ""])
        lines.extend(["| Dataset | Method | Positive Seeds | Mean Positive Delta | Max Delta |", "| --- | --- | ---: | ---: | ---: |"])
        for (dataset, method), deltas in sorted(grouped.items(), key=lambda item: (len(item[1]), sum(item[1]) / len(item[1])), reverse=True):
            lines.append(
                f"| {dataset} | {method} | {len(deltas)} | {np.mean(deltas):+.4f} | {np.max(deltas):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `balanced_treatment`, `propensity_overlap`, and `propensity_feature` are the first code-level versions of the selection-bias idea in the slides.",
            "- `causal_scaled` and `causal_kbest` are lightweight causal-proxy variants: they use supervised feature relevance as a placeholder for true causal weights.",
            "- This run is a fast screening pass. Positive deltas identify candidates worth retesting with more seeds and larger `n_estimators`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search context-selection and causal-proxy TabICL methods.")
    parser.add_argument("--synthetic-samples", type=int, default=1200)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--random-states",
        default=None,
        help="Optional comma-separated random states. Overrides --random-state when provided.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--output", default="context_selection_search_results.csv")
    parser.add_argument("--winners-output", default="context_selection_winners.csv")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = (
        [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
        if args.random_states
        else [args.random_state]
    )
    datasets = load_datasets(args.synthetic_samples, random_states[0])
    if args.datasets != "all":
        wanted = {part.strip() for part in args.datasets.split(",") if part.strip()}
        datasets = [dataset for dataset in datasets if dataset.name in wanted]
    methods = method_specs()

    rows: list[dict[str, object]] = []
    for random_state in random_states:
        for dataset in datasets:
            print(f"Dataset={dataset.name} seed={random_state} n={dataset.X.shape[0]} p={dataset.X.shape[1]}")
            split = make_split(dataset, args.test_size, random_state)
            for method in methods:
                print(f"  method={method.name}")
                try:
                    rows.append(run_method(split, dataset, method, device=device, random_state=random_state))
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "source": dataset.source,
                            "method": method.name,
                            "description": method.description,
                            "metric_name": "accuracy",
                            "metric_value": "",
                            "error": repr(exc),
                            "n_samples": int(dataset.X.shape[0]),
                            "n_features_original": int(dataset.X.shape[1]),
                            "device": device,
                            "random_state": random_state,
                        }
                    )
                    print(f"    failed: {exc!r}")

    winners = winner_rows([row for row in rows if row.get("metric_value") not in {"", None}])
    write_rows(RESULTS_DIR / args.output, rows)
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"])
            writer.writeheader()
    write_markdown(rows, winners, RESULTS_DIR / "context_selection_search.md")
    write_json(
        RESULTS_DIR / "context_selection_search_environment.json",
        {
            **environment_info(device),
            "synthetic_samples": args.synthetic_samples,
            "methods": [method.name for method in methods],
            "datasets": [dataset.name for dataset in datasets],
            "random_states": random_states,
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.winners_output}")
    print(f"Wrote {RESULTS_DIR / 'context_selection_search.md'}")


if __name__ == "__main__":
    main()
