from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine, make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apa_tabicl import (
    APAConfig,
    APAPrediction,
    CounterfactualContextConfig,
    NoShiftFeatureConfig,
    apa_v7_config_for_proba,
    apa_predict_proba,
    baseline_predict_proba,
    classification_metrics,
    context_prior,
    counterfactual_context_predict_proba,
    no_shift_feature_context_predict_proba,
    correct_existing_proba_with_context_prior,
    js_divergence,
)
from tabicl import TabICLClassifier


RESULTS_DIR = ROOT / "experiment6" / "results"
METRIC_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "minority_recall", "roc_auc", "ece"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray
    split_kind: str = "standard"


@dataclass(frozen=True)
class OpenMLSpec:
    data_id: int
    alias: str
    note: str


OPENML_SPECS = {
    "openml_cmc": OpenMLSpec(23, "openml_cmc", "Contraceptive method choice; weak multiclass"),
    "openml_ilpd": OpenMLSpec(1480, "openml_ilpd", "Indian liver patient; imbalanced binary"),
    "openml_credit_g": OpenMLSpec(31, "openml_credit_g", "Credit risk; mixed binary"),
    "openml_blood_transfusion": OpenMLSpec(1464, "openml_blood_transfusion", "Blood donation; imbalanced binary"),
    "openml_kc1": OpenMLSpec(1067, "openml_kc1", "Software defect; noisy binary"),
    "openml_dmft": OpenMLSpec(469, "openml_dmft", "Dental caries DMFT; multiclass"),
    "openml_diabetes_pima": OpenMLSpec(37, "openml_diabetes_pima", "Pima diabetes; binary"),
    "openml_pc1": OpenMLSpec(1068, "openml_pc1", "PC1 defects; binary"),
}


def openml_name_to_spec(name: str) -> OpenMLSpec | None:
    if name in OPENML_SPECS:
        return OPENML_SPECS[name]
    match = re.match(r"^openml_(\d+)(?:_.+)?$", name)
    if not match:
        return None
    return OpenMLSpec(int(match.group(1)), name, "Dynamic OpenML dataset id.")


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, signed: bool = False, digits: int = 4) -> str:
    if value in {"", None}:
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def as_numeric_matrix(X: np.ndarray) -> np.ndarray:
    X_obj = np.asarray(X, dtype=object)
    numeric = np.empty(X_obj.shape, dtype=np.float32)
    categorical_cols: list[int] = []
    for col in range(X_obj.shape[1]):
        values = X_obj[:, col]
        try:
            numeric[:, col] = values.astype(np.float32)
        except (TypeError, ValueError):
            categorical_cols.append(col)
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        numeric[:, categorical_cols] = encoder.fit_transform(X_obj[:, categorical_cols]).astype(np.float32)
    return SimpleImputer(strategy="median").fit_transform(numeric).astype(np.float32)


def load_openml_dataset(spec: OpenMLSpec, max_samples: int, seed: int) -> DatasetSpec:
    try:
        data = fetch_openml(data_id=spec.data_id, as_frame=False, parser="liac-arff")
    except ValueError as exc:
        if "STRING attributes are not supported" not in str(exc):
            raise
        data = fetch_openml(data_id=spec.data_id, as_frame=True, parser="liac-arff")
    X = as_numeric_matrix(data.data)
    target = np.asarray(data.target)
    mask = target != None  # noqa: E711
    X = X[mask]
    target = target[mask]
    y = LabelEncoder().fit_transform(target)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{spec.alias} has fewer than two classes")
    X, y = subsample(X, y, max_samples, seed)
    return DatasetSpec(spec.alias, X.astype(np.float32), y)


def make_dataset(name: str, n_samples: int, seed: int) -> DatasetSpec:
    openml_spec = openml_name_to_spec(name)
    if openml_spec is not None:
        return load_openml_dataset(openml_spec, n_samples, seed)
    if name == "breast_cancer":
        data = load_breast_cancer()
        return DatasetSpec(name, data.data.astype(np.float32), data.target)
    if name == "wine":
        data = load_wine()
        return DatasetSpec(name, data.data.astype(np.float32), data.target)
    if name == "iris":
        data = load_iris()
        return DatasetSpec(name, data.data.astype(np.float32), data.target)
    if name == "synthetic_balanced":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=16,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.5, 0.5],
            class_sep=1.1,
            flip_y=0.02,
            random_state=seed,
        )
        return DatasetSpec(name, X.astype(np.float32), y)
    if name == "synthetic_imbalanced":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=18,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.88, 0.12],
            class_sep=0.9,
            flip_y=0.03,
            random_state=seed,
        )
        return DatasetSpec(name, X.astype(np.float32), y)
    if name == "synthetic_label_shift":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=18,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.5, 0.5],
            class_sep=0.95,
            flip_y=0.02,
            random_state=seed,
        )
        return DatasetSpec(name, X.astype(np.float32), y, split_kind="label_shift")
    if name == "synthetic_label_shift_hard":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=22,
            n_informative=8,
            n_redundant=4,
            n_classes=2,
            weights=[0.5, 0.5],
            class_sep=0.62,
            flip_y=0.05,
            random_state=seed,
        )
        return DatasetSpec(name, X.astype(np.float32), y, split_kind="label_shift")
    if name == "synthetic_multiclass_imbalance":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=1,
            weights=[0.58, 0.22, 0.14, 0.06],
            class_sep=0.85,
            flip_y=0.03,
            random_state=seed,
        )
        return DatasetSpec(name, X.astype(np.float32), y)
    raise ValueError(f"Unknown dataset {name!r}")


def subsample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    selected: list[int] = []
    per_class = max(1, max_samples // len(classes))
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        selected.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    remaining = max_samples - len(selected)
    if remaining > 0:
        mask = np.ones(len(y), dtype=bool)
        mask[selected] = False
        rest = np.flatnonzero(mask)
        selected.extend(rng.choice(rest, size=min(remaining, len(rest)), replace=False).tolist())
    selected_arr = np.asarray(selected, dtype=int)
    rng.shuffle(selected_arr)
    return X[selected_arr], y[selected_arr]


def _take_binary_shift_indices(
    y: np.ndarray,
    seed: int,
    train_size: int,
    test_size: int,
    shift_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    shift_strength = float(np.clip(shift_strength, 0.0, 0.95))
    train_class0_ratio = 0.5 + 0.5 * shift_strength
    test_class1_ratio = 0.5 + 0.5 * shift_strength

    n_train0 = min(max(1, len(idx0) - 1), int(round(train_size * train_class0_ratio)))
    n_train1 = min(max(1, len(idx1) - 1), train_size - n_train0)
    while n_train0 + n_train1 < min(train_size, len(y) - 2):
        can0 = n_train0 < len(idx0) - 1
        can1 = n_train1 < len(idx1) - 1
        if can0 and (not can1 or n_train0 / max(1, n_train0 + n_train1) < train_class0_ratio):
            n_train0 += 1
        elif can1:
            n_train1 += 1
        else:
            break

    train_idx = np.concatenate([idx0[:n_train0], idx1[:n_train1]])
    used = np.zeros(len(y), dtype=bool)
    used[train_idx] = True
    rem0 = np.flatnonzero((y == 0) & ~used)
    rem1 = np.flatnonzero((y == 1) & ~used)

    n_test1 = min(max(1, len(rem1)), int(round(test_size * test_class1_ratio)))
    n_test0 = min(max(1, len(rem0)), test_size - n_test1)
    while n_test0 + n_test1 < min(test_size, len(rem0) + len(rem1)):
        can0 = n_test0 < len(rem0)
        can1 = n_test1 < len(rem1)
        if can1 and (not can0 or n_test1 / max(1, n_test0 + n_test1) < test_class1_ratio):
            n_test1 += 1
        elif can0:
            n_test0 += 1
        else:
            break

    test_idx = np.concatenate([rem1[:n_test1], rem0[:n_test0]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def split_dataset(
    spec: DatasetSpec,
    seed: int,
    train_size: int,
    test_size: int,
    label_shift_strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spec.split_kind == "ordered":
        X = spec.X
        y = spec.y
        total = min(max(train_size + test_size, 1), len(y))
        train_n = min(train_size, max(1, total - 1))
        test_n = min(test_size, max(1, total - train_n))
        total = train_n + test_n
        max_start = max(0, len(y) - total)
        rng = np.random.default_rng(seed)
        starts = np.arange(max_start + 1, dtype=int)
        if len(starts) > 1:
            rng.shuffle(starts)
        best_start = int(starts[0]) if len(starts) else 0
        found_valid_window = False
        for start in starts:
            start = int(start)
            train_idx = np.arange(start, start + train_n)
            test_idx = np.arange(start + train_n, start + total)
            if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[test_idx])) >= 2:
                best_start = start
                found_valid_window = True
                break
        if not found_valid_window:
            raise ValueError(
                f"{spec.name} ordered split has no contiguous window with at least two classes "
                f"in both train and test for train_size={train_n}, test_size={test_n}"
            )
        train_idx = np.arange(best_start, best_start + train_n)
        test_idx = np.arange(best_start + train_n, best_start + total)
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    X, y = subsample(spec.X, spec.y, max(train_size + test_size, 1), seed)
    if spec.split_kind == "label_shift" and len(np.unique(y)) == 2:
        train_idx, test_idx = _take_binary_shift_indices(y, seed, train_size, test_size, label_shift_strength)
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=min(train_size, len(y) - 2),
        test_size=min(test_size, len(y) - min(train_size, len(y) - 2)),
        stratify=y,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test


@dataclass(frozen=True)
class MethodRun:
    config: APAConfig | CounterfactualContextConfig | None
    mode: str
    version: str
    description: str


def validation_candidate_configs(k: int) -> dict[str, APAConfig]:
    return {
        "tiny_batch": APAConfig(
            context_strategy="prior_aware_knn",
            correction="adaptive",
            k=k,
            lambda_balance=0.10,
            alpha=2.0,
            query_mode="batch",
            eta=0.0,
            gamma_min=0.0,
            gamma_max=0.25,
        ),
        "soft_batch": APAConfig(
            context_strategy="prior_aware_knn",
            correction="adaptive",
            k=k,
            lambda_balance=0.15,
            alpha=2.0,
            query_mode="batch",
            eta=0.0,
            gamma_min=0.0,
            gamma_max=0.7,
            blend_weight=0.7,
        ),
        "hard_shift_gate": APAConfig(
            context_strategy="prior_aware_knn",
            correction="adaptive",
            k=k,
            lambda_balance=0.20,
            alpha=1.0,
            query_mode="batch",
            eta=0.0,
            gamma_min=0.0,
            gamma_max=1.0,
            min_batch_prior_js=0.005,
            max_baseline_margin=0.28,
            max_baseline_confidence=0.72,
        ),
        "hybrid_local": APAConfig(
            context_strategy="prior_aware_knn",
            correction="adaptive",
            k=k,
            lambda_balance=0.25,
            alpha=1.0,
            query_mode="hybrid",
            eta=0.5,
            gamma_min=0.0,
            gamma_max=0.75,
            max_baseline_margin=0.45,
            max_baseline_confidence=0.85,
            blend_weight=0.8,
        ),
    }



def _entropy_rows(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _top2_margin_rows(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape[1] <= 1:
        return np.ones(proba.shape[0], dtype=np.float64)
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _train_label_positions(y_train: np.ndarray, classes: np.ndarray) -> np.ndarray:
    positions = np.full(len(y_train), -1, dtype=int)
    for pos, label in enumerate(y_train):
        matches = np.flatnonzero(classes == label)
        if len(matches):
            positions[pos] = int(matches[0])
    return positions


def _local_neighbor_evidence(
    X_train: np.ndarray,
    y_train_pos: np.ndarray,
    X_test: np.ndarray,
    candidate_pred: np.ndarray,
    baseline_pred: np.ndarray,
    n_classes: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    if Xtr.ndim != 2 or Xte.ndim != 2 or len(Xtr) == 0:
        zeros = np.zeros(len(Xte), dtype=np.float64)
        return zeros, zeros, zeros
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    scale = np.nanstd(Xtr, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    Xtr = (Xtr - med) / scale
    Xte = (Xte - med) / scale
    valid = y_train_pos >= 0
    if not np.any(valid):
        zeros = np.zeros(len(Xte), dtype=np.float64)
        return zeros, zeros, zeros
    Xtr_valid = Xtr[valid]
    y_valid = y_train_pos[valid]
    local_k = max(1, min(int(k), len(y_valid)))
    candidate_support = np.zeros(len(Xte), dtype=np.float64)
    baseline_support = np.zeros(len(Xte), dtype=np.float64)
    local_purity = np.zeros(len(Xte), dtype=np.float64)
    for row_pos, x in enumerate(Xte):
        dist = np.sum((Xtr_valid - x) ** 2, axis=1)
        idx = np.argpartition(dist, local_k - 1)[:local_k]
        hist = np.bincount(y_valid[idx], minlength=n_classes).astype(np.float64) / float(local_k)
        cand = int(candidate_pred[row_pos])
        base = int(baseline_pred[row_pos])
        if 0 <= cand < n_classes:
            candidate_support[row_pos] = float(hist[cand])
        if 0 <= base < n_classes:
            baseline_support[row_pos] = float(hist[base])
        local_purity[row_pos] = float(np.max(hist)) if len(hist) else 0.0
    return candidate_support, baseline_support, local_purity


def _binomial_upper_tail(n: int, p: float, observed: int) -> float:
    n = max(0, int(n))
    observed = min(max(0, int(observed)), n)
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    total = 0.0
    for r in range(observed, n + 1):
        total += math.comb(n, r) * (p**r) * ((1.0 - p) ** (n - r))
    return float(np.clip(total, 0.0, 1.0))


def _local_support_p_values(
    local_candidate_support: np.ndarray,
    candidate_pred: np.ndarray,
    train_prior: np.ndarray,
    k: int,
) -> np.ndarray:
    p_values = np.ones(len(candidate_pred), dtype=np.float64)
    local_k = max(1, int(k))
    for row_pos, cand in enumerate(candidate_pred):
        cand = int(cand)
        if 0 <= cand < len(train_prior):
            observed = int(round(float(local_candidate_support[row_pos]) * local_k))
            p_values[row_pos] = _binomial_upper_tail(local_k, float(train_prior[cand]), observed)
    return p_values


def _align_sklearn_proba_to_classes(estimator, proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(proba), len(classes)), dtype=np.float64)
    estimator_classes = np.asarray(getattr(estimator, "classes_", []))
    for src_pos, cls in enumerate(estimator_classes):
        dst = np.flatnonzero(classes == cls)
        if len(dst):
            aligned[:, int(dst[0])] = np.asarray(proba, dtype=np.float64)[:, src_pos]
    empty = np.sum(aligned, axis=1) <= 1e-12
    if np.any(empty) and len(classes):
        aligned[empty] = 1.0 / float(len(classes))
    aligned /= np.maximum(np.sum(aligned, axis=1, keepdims=True), 1e-12)
    return aligned


def _temperature_scale_proba(proba: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    logits = np.log(clipped) / max(float(temperature), 1e-6)
    logits -= np.max(logits, axis=1, keepdims=True)
    scaled = np.exp(np.clip(logits, -60.0, 60.0))
    scaled /= np.maximum(np.sum(scaled, axis=1, keepdims=True), 1e-12)
    return scaled


def family_agreement_risk_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
) -> tuple[APAPrediction, dict[str, object]]:
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    n_rows, n_classes = baseline_proba.shape
    baseline_pred = np.argmax(baseline_proba, axis=1)
    confidence = np.max(baseline_proba, axis=1)
    sorted_raw = np.sort(baseline_proba, axis=1)
    baseline_margin = sorted_raw[:, -1] - sorted_raw[:, -2] if n_classes > 1 else np.ones(n_rows)

    family_configs = [
        (
            "boundary",
            replace(
                config,
                families=("boundary_contrast",),
                tau_clean=0.65,
                max_baseline_margin=0.24,
                max_baseline_confidence=0.84,
                max_context_disagreement=0.030,
                min_context_vote_fraction=0.99,
                local_evidence_margin=0.050,
                blend_weight=0.22,
                max_changed_fraction=1.0,
            ),
        ),
        (
            "diverse",
            replace(
                config,
                families=("diverse_knn",),
                lambda_div=0.40,
                max_baseline_margin=0.38,
                max_baseline_confidence=0.92,
                max_context_disagreement=0.070,
                min_context_vote_fraction=0.99,
                local_evidence_margin=-0.010,
                blend_weight=0.30,
                max_changed_fraction=1.0,
            ),
        ),
        (
            "prototype",
            replace(
                config,
                families=("prototype",),
                max_baseline_margin=0.36,
                max_baseline_confidence=0.92,
                max_context_disagreement=0.060,
                min_context_vote_fraction=0.99,
                local_evidence_margin=0.000,
                blend_weight=0.28,
                max_changed_fraction=1.0,
            ),
        ),
        (
            "clean",
            replace(
                config,
                families=("clean_knn",),
                tau_clean=0.70,
                max_baseline_margin=0.30,
                max_baseline_confidence=0.88,
                max_context_disagreement=0.045,
                min_context_vote_fraction=0.99,
                local_evidence_margin=0.025,
                blend_weight=0.24,
                max_changed_fraction=1.0,
            ),
        ),
        (
            "minority_density",
            replace(
                config,
                families=("minority_contrast",),
                lambda_div=0.30,
                max_batch_prior_js=0.020,
                max_baseline_margin=0.90,
                max_baseline_confidence=0.99,
                max_context_disagreement=0.080,
                min_context_vote_fraction=0.99,
                local_evidence_margin=0.000,
                blend_weight=0.30,
                max_changed_fraction=1.0,
                require_candidate_minority=True,
                min_minority_distance_advantage=0.020,
            ),
        ),
    ]

    family_names: list[str] = []
    family_bundles: list[APAPrediction] = []
    for family_name, family_config in family_configs:
        family_names.append(family_name)
        family_bundles.append(
            no_shift_feature_context_predict_proba(
                X_train,
                y_train,
                X_test,
                estimator_factory,
                family_config,
                baseline_proba=baseline_proba,
                baseline_classes=classes,
            )
        )

    num_families = len(family_bundles)
    family_probas = np.stack([np.asarray(bundle.proba_adj, dtype=np.float64) for bundle in family_bundles], axis=1)
    family_pred = np.argmax(family_probas, axis=2)
    family_changed = family_pred != baseline_pred[:, None]
    vote_counts = np.zeros((n_rows, n_classes), dtype=int)
    for family_pos in range(num_families):
        changed_idx = np.flatnonzero(family_changed[:, family_pos])
        if len(changed_idx):
            vote_counts[changed_idx, family_pred[changed_idx, family_pos]] += 1
    max_votes = np.max(vote_counts, axis=1) if n_classes else np.zeros(n_rows, dtype=int)
    candidate_pred = np.argmax(vote_counts, axis=1) if n_classes else baseline_pred.copy()
    has_candidate = (max_votes > 0) & (candidate_pred != baseline_pred)
    agreement_ratio = max_votes.astype(np.float64) / max(1, num_families)

    family_mean = np.mean(family_probas, axis=1, keepdims=True)
    family_disagreement = np.mean(np.sum((family_probas - family_mean) ** 2, axis=2), axis=1)
    disagreement_norm = np.clip(family_disagreement / 0.080, 0.0, 1.0)

    candidate_proba = baseline_proba.copy()
    support_matrix = np.zeros((n_rows, num_families), dtype=bool)
    for row_pos in range(n_rows):
        cand = int(candidate_pred[row_pos])
        support = family_changed[row_pos] & (family_pred[row_pos] == cand)
        support_matrix[row_pos] = support
        if np.any(support):
            candidate_proba[row_pos] = np.mean(family_probas[row_pos, support, :], axis=0)
            candidate_proba[row_pos] /= max(float(candidate_proba[row_pos].sum()), 1e-12)

    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=n_classes).astype(np.float64)
    train_prior = train_counts / max(1.0, float(train_counts.sum()))
    minority_mask = train_prior <= 0.35
    if n_classes > 1 and np.any(train_counts > 0):
        minority_mask[int(np.argmin(np.where(train_counts > 0, train_counts, np.inf)))] = True
    train_minority_prior = float(np.min(train_prior[train_counts > 0])) if np.any(train_counts > 0) else 0.0

    local_candidate_support, local_baseline_support, local_purity = _local_neighbor_evidence(
        X_train,
        y_train_pos,
        X_test,
        candidate_pred,
        baseline_pred,
        n_classes,
        k=max(config.k, 16),
    )
    baseline_is_minority = minority_mask[baseline_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
    candidate_is_majority = ~minority_mask[candidate_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
    minority_risk = has_candidate & baseline_is_minority & candidate_is_majority

    high_risk = (
        (train_minority_prior <= 0.18)
        | baseline_is_minority
        | (local_purity < 0.60)
        | (family_disagreement > 0.060)
    )
    required_votes = np.where(high_risk, 3, 2)
    agreement_pass = max_votes >= required_votes

    candidate_confidence = candidate_proba[np.arange(n_rows), candidate_pred]
    baseline_confidence = baseline_proba[np.arange(n_rows), baseline_pred]
    confidence_gain = candidate_confidence - baseline_confidence
    entropy_gain = _entropy_rows(baseline_proba) - _entropy_rows(candidate_proba)
    no_entropy_gain = (entropy_gain <= 0.0) | (confidence_gain <= 0.0)
    local_evidence_pass = (local_candidate_support + 0.02 >= local_baseline_support) | (max_votes >= 3)

    family_support = {name: support_matrix[:, pos] for pos, name in enumerate(family_names)}
    boundary_and_prototype = family_support.get("boundary", np.zeros(n_rows, dtype=bool)) & family_support.get(
        "prototype", np.zeros(n_rows, dtype=bool)
    )
    minority_prior_window = (train_minority_prior >= 0.18) & (train_minority_prior <= 0.34)
    minority_strict_pass = (
        minority_prior_window
        & (max_votes >= 3)
        & (local_candidate_support + 0.02 >= local_baseline_support)
        & (confidence_gain >= -0.020)
        & (family_disagreement <= 0.055)
    )
    minority_alt_pass = (
        minority_prior_window
        & boundary_and_prototype
        & (max_votes >= 2)
        & (confidence_gain >= -0.020)
        & (family_disagreement <= 0.040)
    )
    minority_guard_pass = ~minority_risk

    risk_score = (
        0.30 * (1.0 - agreement_ratio)
        + 0.25 * disagreement_norm
        + 0.25 * minority_risk.astype(np.float64)
        + 0.10 * no_entropy_gain.astype(np.float64)
        + 0.10 * (1.0 - local_purity)
    )
    tau_risk = 0.40
    eligible = (
        has_candidate
        & (baseline_margin <= config.max_baseline_margin)
        & (confidence <= config.max_baseline_confidence)
        & agreement_pass
        & local_evidence_pass
        & minority_guard_pass
        & (risk_score <= tau_risk)
    )
    pre_cap_eligible = eligible.copy()
    if config.max_changed_fraction < 1.0 and np.any(eligible):
        max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
        rank_score = max_votes + confidence_gain + entropy_gain + local_candidate_support - risk_score - baseline_margin
        keep = np.zeros_like(eligible, dtype=bool)
        idx = np.flatnonzero(eligible)
        keep[idx[np.argsort(rank_score[idx])[-max_changes:]]] = True
        eligible = keep

    proba_adj = np.where(eligible[:, None], candidate_proba, baseline_proba)
    pred = classes[np.argmax(proba_adj, axis=1)]
    changed = np.argmax(proba_adj, axis=1) != baseline_pred
    batch_js_values = [
        np.asarray(getattr(bundle, "diagnostics", {}).get("no_shift_batch_prior_js", np.zeros(n_rows)), dtype=float)
        for bundle in family_bundles
    ]
    context_failure_values = [
        np.asarray(getattr(bundle, "diagnostics", {}).get("no_shift_context_failure_fraction", np.zeros(n_rows)), dtype=float)
        for bundle in family_bundles
    ]
    diagnostics = {
        "no_shift_eligible_final": eligible.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_batch_prior_js": batch_js_values[0] if batch_js_values else np.zeros(n_rows),
        "no_shift_vote_fraction": agreement_ratio,
        "no_shift_context_disagreement": family_disagreement,
        "no_shift_local_support_margin": local_candidate_support - local_baseline_support,
        "no_shift_context_failure_fraction": np.maximum.reduce(context_failure_values) if context_failure_values else np.zeros(n_rows),
        "no_shift_family_agreement_votes": max_votes.astype(float),
        "no_shift_family_agreement_ratio": agreement_ratio,
        "no_shift_risk_score": risk_score,
        "no_shift_candidate_fraction": has_candidate.astype(float),
        "no_shift_rejected_candidate_fraction": (has_candidate & ~eligible).astype(float),
        "no_shift_pre_cap_eligible_fraction": pre_cap_eligible.astype(float),
        "no_shift_minority_risk": minority_risk.astype(float),
        "no_shift_minority_risk_rejected": (minority_risk & ~eligible).astype(float),
        "no_shift_local_purity": local_purity,
        "no_shift_confidence_gain": confidence_gain,
        "no_shift_entropy_gain": entropy_gain,
    }
    family_hist = {vote: int(np.sum(max_votes == vote)) for vote in range(num_families + 1)}
    extra = {
        "no_shift_batch_prior_js": float(np.mean(diagnostics["no_shift_batch_prior_js"])),
        "no_shift_eligible_fraction": float(np.mean(diagnostics["no_shift_eligible_final"])),
        "no_shift_changed_fraction": float(np.mean(diagnostics["no_shift_changed_fraction"])),
        "no_shift_candidate_fraction": float(np.mean(diagnostics["no_shift_candidate_fraction"])),
        "no_shift_rejected_candidate_fraction": float(np.mean(diagnostics["no_shift_rejected_candidate_fraction"])),
        "no_shift_pre_cap_eligible_fraction": float(np.mean(diagnostics["no_shift_pre_cap_eligible_fraction"])),
        "no_shift_mean_family_agreement_votes": float(np.mean(max_votes)),
        "no_shift_family_agreement_ge2_fraction": float(np.mean(max_votes >= 2)),
        "no_shift_family_agreement_ge3_fraction": float(np.mean(max_votes >= 3)),
        "no_shift_family_agreement_hist": ";".join(f"{vote}:{family_hist[vote]}" for vote in range(num_families + 1)),
        "no_shift_mean_context_disagreement": float(np.mean(family_disagreement)),
        "no_shift_context_disagreement_p50": float(np.quantile(family_disagreement, 0.50)),
        "no_shift_context_disagreement_p90": float(np.quantile(family_disagreement, 0.90)),
        "no_shift_mean_risk_score": float(np.mean(risk_score)),
        "no_shift_risk_score_p90": float(np.quantile(risk_score, 0.90)),
        "no_shift_minority_risk_fraction": float(np.mean(minority_risk)),
        "no_shift_minority_risk_rejected_count": int(np.sum(minority_risk & ~eligible)),
        "no_shift_mean_local_purity": float(np.mean(local_purity)),
        "no_shift_mean_confidence_gain": float(np.mean(confidence_gain[has_candidate])) if np.any(has_candidate) else 0.0,
        "no_shift_mean_entropy_gain": float(np.mean(entropy_gain[has_candidate])) if np.any(has_candidate) else 0.0,
        "no_shift_context_failure_fraction": float(np.mean(diagnostics["no_shift_context_failure_fraction"])),
    }
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=pred,
        classes=classes,
        context_indices=family_bundles[0].context_indices if family_bundles else baseline_bundle.context_indices,
        effective_prior=family_bundles[0].effective_prior if family_bundles else baseline_bundle.effective_prior,
        gamma=agreement_ratio,
        delta=risk_score,
        runtime_sec=sum(float(bundle.runtime_sec) for bundle in family_bundles),
        diagnostics=diagnostics,
    )
    return bundle, extra


def guarded_boundary_union_stability_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
) -> tuple[APAPrediction, dict[str, object]]:
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    n_rows = len(baseline_pred)

    strict_config = config
    soft_config = replace(
        config,
        lambda_div=0.35,
        max_baseline_margin=0.38,
        max_baseline_confidence=0.92,
        max_context_disagreement=0.065,
        min_context_vote_fraction=0.67,
        local_evidence_margin=-0.010,
        blend_weight=0.22,
        max_changed_fraction=config.max_changed_fraction,
    )
    boundary_config = replace(
        config,
        tau_clean=0.65,
        max_baseline_margin=0.24,
        max_baseline_confidence=0.84,
        max_context_disagreement=0.030,
        min_context_vote_fraction=0.80,
        local_evidence_margin=0.050,
        blend_weight=0.22,
        max_changed_fraction=0.015625,
        families=("boundary_contrast", "clean_knn", "diverse_knn"),
    )

    def run_cfg(cfg: NoShiftFeatureConfig) -> APAPrediction:
        return no_shift_feature_context_predict_proba(
            X_train,
            y_train,
            X_test,
            estimator_factory,
            cfg,
            baseline_proba=baseline_proba,
            baseline_classes=classes,
        )

    strict_bundle = run_cfg(strict_config)
    soft_bundle = run_cfg(soft_config)
    boundary_bundle = run_cfg(boundary_config)
    candidate_bundles = [strict_bundle, soft_bundle, boundary_bundle]
    candidate_diags = [getattr(bundle, "diagnostics", {}) for bundle in candidate_bundles]

    def candidate_score(diag: dict[str, object]) -> np.ndarray:
        return (
            np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float)
            + np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float)
            - np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float)
        )

    candidate_scores = [candidate_score(diag) for diag in candidate_diags]
    candidate_changed = [np.argmax(bundle.proba_adj, axis=1) != baseline_pred for bundle in candidate_bundles]

    _, train_counts_guard = np.unique(y_train, return_counts=True)
    train_minority_prior_guard = float(np.min(train_counts_guard) / max(1, np.sum(train_counts_guard)))
    combined_vote_guard = np.maximum.reduce([
        np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float) for diag in candidate_diags
    ])
    combined_disagreement_guard = np.minimum.reduce([
        np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float) for diag in candidate_diags
    ])
    combined_support_guard = np.maximum.reduce([
        np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float) for diag in candidate_diags
    ])
    boundary_prior_window = (train_minority_prior_guard >= 0.18) & (train_minority_prior_guard <= 0.34)
    strong_support = combined_support_guard >= 0.030
    strong_consensus = (combined_vote_guard >= 0.940) & (combined_disagreement_guard <= 0.035)
    boundary_guard_pass = boundary_prior_window & (strong_support | strong_consensus)
    candidate_changed[2] = candidate_changed[2] & boundary_guard_pass

    pre_stability_proba = baseline_proba.copy()
    chosen = np.zeros(n_rows, dtype=bool)
    chosen_score = np.full(n_rows, -np.inf)
    chosen_source = np.full(n_rows, -1, dtype=int)
    for source_pos, (bundle, score, changed) in enumerate(zip(candidate_bundles, candidate_scores, candidate_changed)):
        take = changed & (score > chosen_score)
        if np.any(take):
            pre_stability_proba[take] = bundle.proba_adj[take]
            chosen[take] = True
            chosen_score[take] = score[take]
            chosen_source[take] = source_pos

    candidate_pred = np.argmax(pre_stability_proba, axis=1)
    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=len(classes)).astype(np.float64)
    train_prior = train_counts / max(1.0, float(train_counts.sum()))
    minority_mask = train_prior <= 0.35
    if len(classes) > 1 and np.any(train_counts > 0):
        minority_mask[int(np.argmin(np.where(train_counts > 0, train_counts, np.inf)))] = True
    baseline_is_minority = minority_mask[baseline_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
    candidate_is_majority = ~minority_mask[candidate_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
    minority_risk = chosen & baseline_is_minority & candidate_is_majority

    local_candidate_support, local_baseline_support, local_purity = _local_neighbor_evidence(
        X_train,
        y_train_pos,
        X_test,
        candidate_pred,
        baseline_pred,
        len(classes),
        k=max(config.k, 16),
    )
    high_risk = chosen & (
        (chosen_source == 2)
        | minority_risk
        | (local_purity < 0.60)
        | (combined_disagreement_guard > 0.055)
    )

    perturb_configs = [
        replace(
            config,
            candidate_multiplier=4,
            lambda_div=0.30,
            max_baseline_margin=0.34,
            max_context_disagreement=0.055,
            local_evidence_margin=0.000,
            max_changed_fraction=1.0,
            random_state=config.random_state + 11,
        ),
        replace(
            config,
            candidate_multiplier=6,
            lambda_div=0.50,
            max_baseline_margin=0.36,
            max_context_disagreement=0.060,
            local_evidence_margin=0.000,
            max_changed_fraction=1.0,
            random_state=config.random_state + 23,
        ),
        replace(
            config,
            families=("prototype", "clean_knn", "diverse_knn"),
            tau_clean=0.70,
            max_baseline_margin=0.34,
            max_context_disagreement=0.050,
            local_evidence_margin=0.010,
            max_changed_fraction=1.0,
            random_state=config.random_state + 37,
        ),
        replace(
            boundary_config,
            tau_clean=0.70,
            families=("boundary_contrast", "clean_knn"),
            max_context_disagreement=0.035,
            max_changed_fraction=1.0,
            random_state=config.random_state + 41,
        ),
    ]
    perturb_bundles = [run_cfg(cfg) for cfg in perturb_configs]
    stability_votes = np.zeros(n_rows, dtype=np.float64)
    for bundle in perturb_bundles:
        pred = np.argmax(bundle.proba_adj, axis=1)
        stability_votes += ((pred == candidate_pred) & (pred != baseline_pred)).astype(float)
    stability_rate = stability_votes / max(1, len(perturb_bundles))

    stability_pass = stability_rate >= 0.75
    chosen_after_stability = chosen & stability_pass & ~minority_risk
    stability_rejected = chosen & ~chosen_after_stability

    if config.max_changed_fraction < 1.0 and np.any(chosen_after_stability):
        max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
        final_score = chosen_score + stability_rate + local_candidate_support - local_baseline_support
        keep = np.zeros_like(chosen_after_stability, dtype=bool)
        idx = np.flatnonzero(chosen_after_stability)
        keep[idx[np.argsort(final_score[idx])[-max_changes:]]] = True
        chosen_after_stability = keep

    proba_adj = np.where(chosen_after_stability[:, None], pre_stability_proba, baseline_proba)
    changed = np.argmax(proba_adj, axis=1) != baseline_pred
    all_bundles = candidate_bundles + perturb_bundles
    runtime_sec = sum(float(bundle.runtime_sec) for bundle in all_bundles)
    diagnostics = {
        "no_shift_eligible_final": chosen_after_stability.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_vote_fraction": combined_vote_guard,
        "no_shift_context_disagreement": combined_disagreement_guard,
        "no_shift_local_support_margin": combined_support_guard,
        "no_shift_batch_prior_js": np.asarray(candidate_diags[0].get("no_shift_batch_prior_js", np.zeros(n_rows)), dtype=float),
        "no_shift_context_failure_fraction": np.maximum.reduce([
            np.asarray(getattr(bundle, "diagnostics", {}).get("no_shift_context_failure_fraction", np.zeros(n_rows)), dtype=float)
            for bundle in all_bundles
        ]),
        "no_shift_stability_rate": stability_rate,
        "no_shift_high_risk_candidate": high_risk.astype(float),
        "no_shift_stability_rejected": stability_rejected.astype(float),
        "no_shift_minority_risk": minority_risk.astype(float),
        "no_shift_local_purity": local_purity,
        "no_shift_boundary_guard_pass_fraction": np.full(n_rows, float(np.mean(boundary_guard_pass))),
    }
    extra = {
        "no_shift_batch_prior_js": float(np.mean(diagnostics["no_shift_batch_prior_js"])),
        "no_shift_eligible_fraction": float(np.mean(diagnostics["no_shift_eligible_final"])),
        "no_shift_mean_vote_fraction": float(np.mean(combined_vote_guard)),
        "no_shift_mean_context_disagreement": float(np.mean(combined_disagreement_guard)),
        "no_shift_mean_local_support_margin": float(np.mean(combined_support_guard)),
        "no_shift_changed_fraction": float(np.mean(changed)),
        "no_shift_context_failure_fraction": float(np.mean(diagnostics["no_shift_context_failure_fraction"])),
        "no_shift_mean_stability_rate": float(np.mean(stability_rate[chosen])) if np.any(chosen) else 0.0,
        "no_shift_stability_rejected_fraction": float(np.mean(stability_rejected)),
        "no_shift_high_risk_candidate_fraction": float(np.mean(high_risk)),
        "no_shift_minority_risk_fraction": float(np.mean(minority_risk)),
        "no_shift_mean_local_purity": float(np.mean(local_purity)),
    }
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=strict_bundle.context_indices,
        effective_prior=strict_bundle.effective_prior,
        gamma=stability_rate,
        delta=combined_disagreement_guard,
        runtime_sec=runtime_sec,
        diagnostics=diagnostics,
    )
    return bundle, extra


def guarded_boundary_union_v27_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
) -> tuple[APAPrediction, dict[str, object]]:
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    n_rows = len(baseline_pred)
    strict_config = config
    soft_config = NoShiftFeatureConfig(
        k=config.k,
        candidate_multiplier=config.candidate_multiplier,
        lambda_div=0.35,
        tau_clean=config.tau_clean,
        clean_neighbor_k=config.clean_neighbor_k,
        n_bootstrap=config.n_bootstrap,
        n_subspaces=config.n_subspaces,
        subspace_fraction=config.subspace_fraction,
        max_batch_prior_js=config.max_batch_prior_js,
        max_baseline_margin=0.38,
        max_baseline_confidence=0.92,
        max_context_disagreement=0.065,
        min_context_vote_fraction=0.67,
        local_evidence_margin=-0.010,
        blend_weight=0.22,
        max_changed_fraction=config.max_changed_fraction,
        families=config.families,
        random_state=config.random_state,
    )
    boundary_config = NoShiftFeatureConfig(
        k=config.k,
        candidate_multiplier=config.candidate_multiplier,
        lambda_div=config.lambda_div,
        tau_clean=0.65,
        clean_neighbor_k=config.clean_neighbor_k,
        n_bootstrap=config.n_bootstrap,
        n_subspaces=config.n_subspaces,
        subspace_fraction=config.subspace_fraction,
        max_batch_prior_js=config.max_batch_prior_js,
        max_baseline_margin=0.24,
        max_baseline_confidence=0.84,
        max_context_disagreement=0.030,
        min_context_vote_fraction=0.80,
        local_evidence_margin=0.050,
        blend_weight=0.22,
        max_changed_fraction=0.015625,
        families=("boundary_contrast", "clean_knn", "diverse_knn"),
        random_state=config.random_state,
    )

    def run_cfg(cfg: NoShiftFeatureConfig) -> APAPrediction:
        return no_shift_feature_context_predict_proba(
            X_train,
            y_train,
            X_test,
            estimator_factory,
            cfg,
            baseline_proba=baseline_proba,
            baseline_classes=baseline_bundle.classes,
        )

    strict_bundle = run_cfg(strict_config)
    soft_bundle = run_cfg(soft_config)
    boundary_bundle = run_cfg(boundary_config)
    candidate_bundles = [strict_bundle, soft_bundle, boundary_bundle]
    candidate_diags = [getattr(bundle, "diagnostics", {}) for bundle in candidate_bundles]

    def candidate_score(diag: dict[str, object]) -> np.ndarray:
        return (
            np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float)
            + np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float)
            - np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float)
        )

    candidate_scores = [candidate_score(diag) for diag in candidate_diags]
    candidate_changed = [np.argmax(bundle.proba_adj, axis=1) != baseline_pred for bundle in candidate_bundles]
    _, train_counts_guard = np.unique(y_train, return_counts=True)
    train_minority_prior_guard = float(np.min(train_counts_guard) / max(1, np.sum(train_counts_guard)))
    combined_vote_guard = np.maximum.reduce([
        np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float) for diag in candidate_diags
    ])
    combined_disagreement_guard = np.minimum.reduce([
        np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float) for diag in candidate_diags
    ])
    combined_support_guard = np.maximum.reduce([
        np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float) for diag in candidate_diags
    ])
    boundary_prior_window = (train_minority_prior_guard >= 0.18) & (train_minority_prior_guard <= 0.34)
    strong_support = combined_support_guard >= 0.030
    strong_consensus = (combined_vote_guard >= 0.940) & (combined_disagreement_guard <= 0.035)
    boundary_guard_pass = boundary_prior_window & (strong_support | strong_consensus)
    candidate_changed[2] = candidate_changed[2] & boundary_guard_pass

    proba_adj = baseline_proba.copy()
    chosen = np.zeros(n_rows, dtype=bool)
    chosen_score = np.full(n_rows, -np.inf)
    for bundle, score, changed in zip(candidate_bundles, candidate_scores, candidate_changed):
        take = changed & (score > chosen_score)
        if np.any(take):
            proba_adj[take] = bundle.proba_adj[take]
            chosen[take] = True
            chosen_score[take] = score[take]
    if config.max_changed_fraction < 1.0 and np.any(chosen):
        max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
        keep = np.zeros_like(chosen, dtype=bool)
        idx = np.flatnonzero(chosen)
        keep[idx[np.argsort(chosen_score[idx])[-max_changes:]]] = True
        proba_adj = np.where(keep[:, None], proba_adj, baseline_proba)
        chosen = keep
    diagnostics = {
        "no_shift_eligible_final": chosen.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(chosen))),
        "no_shift_vote_fraction": combined_vote_guard,
        "no_shift_context_disagreement": combined_disagreement_guard,
        "no_shift_local_support_margin": combined_support_guard,
        "no_shift_batch_prior_js": np.asarray(candidate_diags[0].get("no_shift_batch_prior_js", np.zeros(n_rows)), dtype=float),
        "no_shift_context_failure_fraction": np.maximum.reduce([
            np.asarray(diag.get("no_shift_context_failure_fraction", np.zeros(n_rows)), dtype=float)
            for diag in candidate_diags
        ]),
        "no_shift_boundary_guard_pass_fraction": np.full(n_rows, float(np.mean(boundary_guard_pass))),
    }
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=baseline_bundle.classes[np.argmax(proba_adj, axis=1)],
        classes=baseline_bundle.classes,
        context_indices=strict_bundle.context_indices,
        effective_prior=strict_bundle.effective_prior,
        gamma=np.maximum.reduce([b.gamma for b in candidate_bundles]),
        delta=np.minimum.reduce([b.delta for b in candidate_bundles]),
        runtime_sec=sum(float(b.runtime_sec) for b in candidate_bundles),
        diagnostics=diagnostics,
    )
    extra = {
        "no_shift_batch_prior_js": float(np.mean(diagnostics["no_shift_batch_prior_js"])),
        "no_shift_eligible_fraction": float(np.mean(diagnostics["no_shift_eligible_final"])),
        "no_shift_mean_vote_fraction": float(np.mean(diagnostics["no_shift_vote_fraction"])),
        "no_shift_mean_context_disagreement": float(np.mean(diagnostics["no_shift_context_disagreement"])),
        "no_shift_mean_local_support_margin": float(np.mean(diagnostics["no_shift_local_support_margin"])),
        "no_shift_changed_fraction": float(np.mean(diagnostics["no_shift_changed_fraction"])),
        "no_shift_context_failure_fraction": float(np.mean(diagnostics["no_shift_context_failure_fraction"])),
    }
    return bundle, extra


def guarded_boundary_union_validation_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    large_scale = len(y_train) >= 192 or len(X_test) >= 96
    if not large_scale:
        bundle, details = guarded_boundary_union_v27_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            config,
        )
        details.update(
            {
                "no_shift_validation_selected": "direct_small_scale_v27",
                "no_shift_validation_large_scale": large_scale,
                "no_shift_validation_applied": True,
            }
        )
        return bundle, details

    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or len(y_train) < 64 or np.min(counts) < 6:
        return baseline_bundle, {
            "no_shift_validation_selected": "baseline_insufficient_train",
            "no_shift_validation_large_scale": large_scale,
            "no_shift_validation_applied": False,
        }

    val_size = min(
        max(len(y_train) // 4, len(classes) * 8),
        max(len(y_train) - len(classes) * 4, len(classes) * 8),
    )
    try:
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=seed + 3901,
        )
    except ValueError:
        return baseline_bundle, {
            "no_shift_validation_selected": "baseline_split_failed",
            "no_shift_validation_large_scale": large_scale,
            "no_shift_validation_applied": False,
        }

    val_baseline = baseline_predict_proba(X_sub, y_sub, X_val, estimator_factory)
    base_metrics = classification_metrics(y_val, val_baseline.proba_adj, val_baseline.classes)
    val_candidate, val_details = guarded_boundary_union_v27_predict(
        X_sub,
        y_sub,
        X_val,
        val_baseline,
        estimator_factory,
        config,
    )
    cand_metrics = classification_metrics(y_val, val_candidate.proba_adj, val_candidate.classes)

    def metric_delta(name: str) -> float | None:
        base = base_metrics.get(name)
        cand = cand_metrics.get(name)
        if base in {"", None} or cand in {"", None}:
            return None
        return float(cand) - float(base)

    acc_delta = metric_delta("accuracy")
    bacc_delta = metric_delta("balanced_accuracy")
    f1_delta = metric_delta("macro_f1")
    auc_delta = metric_delta("roc_auc")
    changed_fraction = float(
        np.mean(
            np.argmax(val_candidate.proba_adj, axis=1)
            != np.argmax(val_baseline.proba_adj, axis=1)
        )
    )
    validation_applied = (
        changed_fraction > 0.0
        and acc_delta is not None
        and acc_delta > 1e-12
        and (bacc_delta is None or bacc_delta >= -1e-12)
        and (f1_delta is None or f1_delta >= -1e-12)
        and (auc_delta is None or auc_delta >= -0.002)
    )
    details: dict[str, object] = {
        "no_shift_validation_selected": "v27_validation_candidate" if validation_applied else "baseline_no_validation_gain",
        "no_shift_validation_large_scale": large_scale,
        "no_shift_validation_applied": validation_applied,
        "no_shift_validation_size": len(y_val),
        "no_shift_validation_changed_fraction": changed_fraction,
        "no_shift_validation_accuracy_delta": acc_delta if acc_delta is not None else "",
        "no_shift_validation_balanced_accuracy_delta": bacc_delta if bacc_delta is not None else "",
        "no_shift_validation_macro_f1_delta": f1_delta if f1_delta is not None else "",
        "no_shift_validation_roc_auc_delta": auc_delta if auc_delta is not None else "",
        "no_shift_validation_batch_prior_js": val_details.get("no_shift_batch_prior_js", ""),
        "no_shift_validation_mean_context_disagreement": val_details.get("no_shift_mean_context_disagreement", ""),
        "no_shift_validation_mean_local_support_margin": val_details.get("no_shift_mean_local_support_margin", ""),
    }
    validation_runtime = float(val_baseline.runtime_sec) + float(val_candidate.runtime_sec)
    if not validation_applied:
        return replace(
            baseline_bundle,
            runtime_sec=float(baseline_bundle.runtime_sec) + validation_runtime,
        ), details

    final_bundle, final_details = guarded_boundary_union_v27_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
    )
    details.update(final_details)
    details["no_shift_validation_selected"] = "v27_validation_candidate"
    return replace(
        final_bundle,
        runtime_sec=float(final_bundle.runtime_sec) + validation_runtime,
    ), details


def instancewise_context_ensemble_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
) -> tuple[APAPrediction, dict[str, object]]:
    eps = 1e-12
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    n_rows = len(baseline_pred)
    if n_rows == 0:
        return baseline_bundle, {"instance_ensemble_selected": "baseline_empty_test"}

    js_value = batch_prior_js(baseline_proba, y_train, classes, alpha=1.0)
    if js_value > config.max_batch_prior_js:
        return baseline_bundle, {
            "instance_ensemble_selected": "baseline_batch_prior_js_guard",
            "instance_ensemble_batch_prior_js": js_value,
            "instance_ensemble_changed_fraction": 0.0,
        }

    member_configs = [
        ("strict_diverse", config),
        (
            "soft_diverse",
            replace(
                config,
                lambda_div=0.35,
                max_baseline_margin=0.38,
                max_baseline_confidence=0.92,
                max_context_disagreement=0.065,
                min_context_vote_fraction=0.67,
                local_evidence_margin=-0.010,
                blend_weight=0.20,
            ),
        ),
        (
            "boundary_contrast",
            replace(
                config,
                tau_clean=0.65,
                max_baseline_margin=0.24,
                max_baseline_confidence=0.84,
                max_context_disagreement=0.030,
                min_context_vote_fraction=0.80,
                local_evidence_margin=0.050,
                blend_weight=0.20,
                max_changed_fraction=0.015625,
                families=("boundary_contrast", "clean_knn", "diverse_knn"),
            ),
        ),
        (
            "clean_consensus",
            replace(
                config,
                tau_clean=0.72,
                lambda_div=0.25,
                max_baseline_margin=0.34,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.045,
                min_context_vote_fraction=0.75,
                local_evidence_margin=0.010,
                blend_weight=0.18,
                families=("clean_knn", "prototype", "diverse_knn"),
            ),
        ),
    ]

    sorted_proba = np.sort(baseline_proba, axis=1)
    baseline_margin = (
        sorted_proba[:, -1] - sorted_proba[:, -2]
        if baseline_proba.shape[1] > 1
        else np.ones(n_rows, dtype=np.float64)
    )
    baseline_conf = np.max(baseline_proba, axis=1)
    baseline_uncertainty = np.clip(1.0 - baseline_margin, 0.0, 1.0)

    member_names: list[str] = []
    member_probas: list[np.ndarray] = []
    member_weights: list[np.ndarray] = []
    member_changed: list[np.ndarray] = []
    member_runtimes: list[float] = []
    vote_arrays: list[np.ndarray] = []
    disagreement_arrays: list[np.ndarray] = []
    support_arrays: list[np.ndarray] = []
    eligible_arrays: list[np.ndarray] = []

    for name, member_config in member_configs:
        member = no_shift_feature_context_predict_proba(
            X_train,
            y_train,
            X_test,
            estimator_factory,
            member_config,
            baseline_proba=baseline_proba,
            baseline_classes=classes,
        )
        diag = getattr(member, "diagnostics", {})
        vote = np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=np.float64)
        disagreement = np.asarray(diag.get("no_shift_context_disagreement", np.ones(n_rows)), dtype=np.float64)
        support = np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=np.float64)
        eligible = np.asarray(diag.get("no_shift_eligible_final", np.zeros(n_rows)), dtype=np.float64) > 0.0
        member_proba = np.asarray(member.proba_adj, dtype=np.float64)
        member_pred = np.argmax(member_proba, axis=1)
        changed = member_pred != baseline_pred
        member_conf = np.max(member_proba, axis=1)
        reliability_logit = (
            7.0 * (vote - 0.78)
            + 18.0 * support
            - 16.0 * disagreement
            + 2.0 * (member_conf - baseline_conf)
        )
        reliability = 1.0 / (1.0 + np.exp(-np.clip(reliability_logit, -30.0, 30.0)))
        context_ok = eligible | (
            (vote >= 0.90)
            & (disagreement <= 0.045)
            & (support >= -0.005)
            & (baseline_margin <= 0.40)
        )
        weight = reliability * context_ok.astype(np.float64)
        member_names.append(name)
        member_probas.append(member_proba)
        member_weights.append(weight)
        member_changed.append(changed)
        member_runtimes.append(float(member.runtime_sec))
        vote_arrays.append(vote)
        disagreement_arrays.append(disagreement)
        support_arrays.append(support)
        eligible_arrays.append(eligible.astype(np.float64))

    weight_matrix = np.vstack(member_weights)
    weight_sum = np.sum(weight_matrix, axis=0)
    normalized_weights = np.divide(
        weight_matrix,
        np.maximum(weight_sum, eps)[None, :],
        out=np.zeros_like(weight_matrix),
        where=np.maximum(weight_sum, eps)[None, :] > 0,
    )
    context_proba = np.zeros_like(baseline_proba)
    for member_weight, member_proba in zip(normalized_weights, member_probas):
        context_proba += member_weight[:, None] * member_proba
    context_proba[weight_sum <= eps] = baseline_proba[weight_sum <= eps]

    max_vote = np.maximum.reduce(vote_arrays)
    min_disagreement = np.minimum.reduce(disagreement_arrays)
    max_support = np.maximum.reduce(support_arrays)
    member_change_count = np.sum(np.vstack(member_changed), axis=0)
    alpha_signal = (
        0.55 * baseline_uncertainty
        + 0.25 * np.clip(max_vote - 0.67, 0.0, 0.33) / 0.33
        + 0.20 * np.clip(max_support + 0.02, 0.0, 0.08) / 0.08
    )
    consensus_signal = np.clip(1.0 - min_disagreement / 0.08, 0.0, 1.0)
    alpha = 0.30 * alpha_signal * consensus_signal * (weight_sum > 0.0)
    alpha = np.clip(alpha, 0.0, 0.30)

    proba_adj = (1.0 - alpha[:, None]) * baseline_proba + alpha[:, None] * context_proba
    proba_adj = proba_adj / np.maximum(np.sum(proba_adj, axis=1, keepdims=True), eps)
    changed = np.argmax(proba_adj, axis=1) != baseline_pred
    if config.max_changed_fraction < 1.0 and np.any(changed):
        max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
        change_score = (
            alpha
            + 0.10 * member_change_count
            + max_support
            - min_disagreement
            - 0.20 * baseline_margin
        )
        keep = np.zeros_like(changed, dtype=bool)
        idx = np.flatnonzero(changed)
        keep[idx[np.argsort(change_score[idx])[-max_changes:]]] = True
        proba_adj = np.where(keep[:, None], proba_adj, baseline_proba)
        changed = keep

    diagnostics = {
        "instance_ensemble_alpha": alpha,
        "instance_ensemble_member_weight_sum": weight_sum,
        "instance_ensemble_member_change_count": member_change_count.astype(np.float64),
        "instance_ensemble_vote_fraction": max_vote,
        "instance_ensemble_context_disagreement": min_disagreement,
        "instance_ensemble_local_support_margin": max_support,
        "instance_ensemble_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "instance_ensemble_eligible_fraction": np.mean(np.vstack(eligible_arrays), axis=0),
        "no_shift_batch_prior_js": np.full(n_rows, js_value),
    }
    extra = {
        "instance_ensemble_selected": "instancewise_context_gating",
        "instance_ensemble_members": ",".join(member_names),
        "instance_ensemble_batch_prior_js": js_value,
        "instance_ensemble_mean_alpha": float(np.mean(alpha)),
        "instance_ensemble_max_alpha": float(np.max(alpha)),
        "instance_ensemble_mean_member_weight_sum": float(np.mean(weight_sum)),
        "instance_ensemble_changed_fraction": float(np.mean(changed)),
        "instance_ensemble_mean_member_change_count": float(np.mean(member_change_count)),
        "instance_ensemble_mean_vote_fraction": float(np.mean(max_vote)),
        "instance_ensemble_mean_context_disagreement": float(np.mean(min_disagreement)),
        "instance_ensemble_mean_local_support_margin": float(np.mean(max_support)),
        "instance_ensemble_mean_eligible_fraction": float(np.mean(diagnostics["instance_ensemble_eligible_fraction"])),
    }
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=baseline_bundle.context_indices,
        effective_prior=baseline_bundle.effective_prior,
        gamma=alpha,
        delta=min_disagreement,
        runtime_sec=sum(member_runtimes),
        diagnostics=diagnostics,
    )
    return bundle, extra


def diversity_greedy_instance_router_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    pvalue_rescue: bool = False,
) -> tuple[APAPrediction, dict[str, object]]:
    """Per-sample greedy selection over structured context members.

    This is the V46 candidate: generate a modest member pool, score each
    member by local evidence, add a small diversity reward, and accept only
    low-risk sample-level changes.
    """

    eps = 1e-12
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    n_rows, n_classes = baseline_proba.shape
    baseline_pred = np.argmax(baseline_proba, axis=1)
    sorted_base = np.sort(baseline_proba, axis=1)
    baseline_margin = sorted_base[:, -1] - sorted_base[:, -2] if n_classes > 1 else np.ones(n_rows)
    baseline_conf = np.max(baseline_proba, axis=1)
    baseline_entropy = _entropy_rows(baseline_proba)

    if n_rows == 0:
        return baseline_bundle, {"diversity_greedy_selected": "baseline_empty_test"}

    js_value = batch_prior_js(baseline_proba, y_train, classes, alpha=1.0)
    if js_value > config.max_batch_prior_js:
        return baseline_bundle, {
            "diversity_greedy_selected": "baseline_batch_prior_js_guard",
            "diversity_greedy_batch_prior_js": js_value,
            "diversity_greedy_changed_fraction": 0.0,
        }

    member_configs = [
        (
            "diverse",
            replace(
                config,
                families=("diverse_knn",),
                lambda_div=0.38,
                max_baseline_margin=0.40,
                max_baseline_confidence=0.93,
                max_context_disagreement=0.070,
                min_context_vote_fraction=0.67,
                local_evidence_margin=-0.010,
                blend_weight=0.26,
                max_changed_fraction=1.0,
            ),
            1.00,
        ),
        (
            "prototype",
            replace(
                config,
                families=("prototype",),
                max_baseline_margin=0.40,
                max_baseline_confidence=0.93,
                max_context_disagreement=0.065,
                min_context_vote_fraction=0.67,
                local_evidence_margin=-0.005,
                blend_weight=0.24,
                max_changed_fraction=1.0,
            ),
            0.95,
        ),
        (
            "clean",
            replace(
                config,
                families=("clean_knn",),
                tau_clean=0.70,
                max_baseline_margin=0.34,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.72,
                local_evidence_margin=0.010,
                blend_weight=0.22,
                max_changed_fraction=1.0,
            ),
            1.00,
        ),
        (
            "boundary",
            replace(
                config,
                families=("boundary_contrast", "clean_knn"),
                tau_clean=0.66,
                max_baseline_margin=0.28,
                max_baseline_confidence=0.86,
                max_context_disagreement=0.040,
                min_context_vote_fraction=0.78,
                local_evidence_margin=0.030,
                blend_weight=0.22,
                max_changed_fraction=1.0,
            ),
            1.15,
        ),
        (
            "feature_subspace",
            replace(
                config,
                families=("feature_subspace_knn",),
                n_subspaces=max(2, config.n_subspaces),
                subspace_fraction=0.60,
                max_baseline_margin=0.42,
                max_baseline_confidence=0.94,
                max_context_disagreement=0.075,
                min_context_vote_fraction=0.67,
                local_evidence_margin=-0.010,
                blend_weight=0.24,
                max_changed_fraction=1.0,
            ),
            1.25,
        ),
        (
            "minority_density",
            replace(
                config,
                families=("minority_contrast",),
                max_batch_prior_js=max(config.max_batch_prior_js, 0.020),
                max_baseline_margin=0.90,
                max_baseline_confidence=0.99,
                max_context_disagreement=0.080,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.000,
                blend_weight=0.24,
                max_changed_fraction=1.0,
                require_candidate_minority=True,
                min_minority_distance_advantage=0.020,
            ),
            1.10,
        ),
    ]

    member_names: list[str] = []
    member_probas: list[np.ndarray] = []
    member_preds: list[np.ndarray] = []
    member_scores: list[np.ndarray] = []
    member_changed: list[np.ndarray] = []
    member_votes: list[np.ndarray] = []
    member_disagreements: list[np.ndarray] = []
    member_supports: list[np.ndarray] = []
    member_costs: list[float] = []
    member_runtimes: list[float] = []

    for name, member_config, cost in member_configs:
        member = no_shift_feature_context_predict_proba(
            X_train,
            y_train,
            X_test,
            estimator_factory,
            member_config,
            baseline_proba=baseline_proba,
            baseline_classes=classes,
        )
        diag = getattr(member, "diagnostics", {})
        member_proba = np.asarray(member.proba_adj, dtype=np.float64)
        member_pred = np.argmax(member_proba, axis=1)
        changed = member_pred != baseline_pred
        vote = np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=np.float64)
        disagreement = np.asarray(diag.get("no_shift_context_disagreement", np.ones(n_rows)), dtype=np.float64)
        support = np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=np.float64)
        member_conf = member_proba[np.arange(n_rows), member_pred]
        confidence_gain = member_conf - baseline_proba[np.arange(n_rows), baseline_pred]
        entropy_gain = baseline_entropy - _entropy_rows(member_proba)
        score = (
            0.85 * vote
            + 1.60 * support
            + 0.35 * confidence_gain
            + 0.25 * entropy_gain
            - 1.60 * disagreement
            - 0.20 * baseline_margin
            - 0.08 * cost
        )
        score = np.where(changed, score, -np.inf)
        member_names.append(name)
        member_probas.append(member_proba)
        member_preds.append(member_pred)
        member_scores.append(score)
        member_changed.append(changed)
        member_votes.append(vote)
        member_disagreements.append(disagreement)
        member_supports.append(support)
        member_costs.append(float(cost))
        member_runtimes.append(float(member.runtime_sec))

    score_matrix = np.vstack(member_scores)
    proba_stack = np.stack(member_probas, axis=1)
    pred_matrix = np.vstack(member_preds)
    changed_matrix = np.vstack(member_changed)
    vote_matrix = np.vstack(member_votes)
    disagreement_matrix = np.vstack(member_disagreements)
    support_matrix = np.vstack(member_supports)
    cost_vector = np.asarray(member_costs, dtype=np.float64)
    n_members = len(member_names)
    max_selected = min(3, n_members)

    selected_mask = np.zeros((n_rows, n_members), dtype=bool)
    selected_count = np.zeros(n_rows, dtype=np.float64)
    selected_diversity = np.zeros(n_rows, dtype=np.float64)
    selected_mean_score = np.full(n_rows, -np.inf, dtype=np.float64)

    for row_pos in range(n_rows):
        row_scores = score_matrix[:, row_pos].copy()
        if not np.any(np.isfinite(row_scores)):
            continue
        selected: list[int] = []
        for _ in range(max_selected):
            best_pos = -1
            best_score = -math.inf
            for member_pos in range(n_members):
                if member_pos in selected or not np.isfinite(row_scores[member_pos]):
                    continue
                diversity_bonus = 0.0
                if selected:
                    cand = proba_stack[row_pos, member_pos : member_pos + 1, :]
                    prev = proba_stack[row_pos, selected, :]
                    repeated = np.repeat(cand, len(selected), axis=0)
                    diversity_bonus = float(np.mean(js_divergence(repeated, prev) / math.log(2.0)))
                greedy_score = row_scores[member_pos] + 0.18 * diversity_bonus - 0.04 * cost_vector[member_pos]
                if greedy_score > best_score:
                    best_score = greedy_score
                    best_pos = member_pos
            if best_pos < 0 or best_score < 0.50:
                break
            selected.append(best_pos)
        if not selected:
            continue
        selected_mask[row_pos, selected] = True
        selected_count[row_pos] = float(len(selected))
        selected_mean_score[row_pos] = float(np.mean(row_scores[selected]))
        if len(selected) > 1:
            pair_values: list[float] = []
            for a_pos, member_a in enumerate(selected):
                for member_b in selected[a_pos + 1 :]:
                    pair_values.append(
                        float(
                            js_divergence(
                                proba_stack[row_pos, member_a].reshape(1, -1),
                                proba_stack[row_pos, member_b].reshape(1, -1),
                            )[0]
                            / math.log(2.0)
                        )
                    )
            selected_diversity[row_pos] = float(np.mean(pair_values)) if pair_values else 0.0

    selected_any = selected_count > 0
    selected_weights = np.zeros_like(score_matrix.T)
    for row_pos in np.flatnonzero(selected_any):
        idx = np.flatnonzero(selected_mask[row_pos])
        logits = score_matrix[idx, row_pos] + 0.10 * selected_diversity[row_pos]
        logits = logits - np.max(logits)
        weights = np.exp(np.clip(logits, -30.0, 30.0))
        weights /= max(float(np.sum(weights)), eps)
        selected_weights[row_pos, idx] = weights

    candidate_proba = np.einsum("rm,rmc->rc", selected_weights, proba_stack)
    candidate_proba[~selected_any] = baseline_proba[~selected_any]
    candidate_proba /= np.maximum(np.sum(candidate_proba, axis=1, keepdims=True), eps)
    candidate_pred = np.argmax(candidate_proba, axis=1)
    candidate_changed = selected_any & (candidate_pred != baseline_pred)

    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=n_classes).astype(np.float64)
    train_prior = train_counts / max(1.0, float(np.sum(train_counts)))
    minority_mask = train_prior <= 0.35
    if n_classes > 1 and np.any(train_counts > 0):
        minority_mask[int(np.argmin(np.where(train_counts > 0, train_counts, np.inf)))] = True

    local_candidate_support, local_baseline_support, local_purity = _local_neighbor_evidence(
        X_train,
        y_train_pos,
        X_test,
        candidate_pred,
        baseline_pred,
        n_classes,
        k=max(config.k, 16),
    )
    local_support_advantage = local_candidate_support - local_baseline_support
    local_p_values = _local_support_p_values(
        local_candidate_support,
        candidate_pred,
        train_prior,
        k=max(config.k, 16),
    )
    baseline_is_minority = minority_mask[baseline_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
    candidate_is_majority = ~minority_mask[candidate_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
    minority_risk = candidate_changed & baseline_is_minority & candidate_is_majority

    target_vote_fraction = np.zeros(n_rows, dtype=np.float64)
    for row_pos in np.flatnonzero(selected_any):
        idx = np.flatnonzero(selected_mask[row_pos])
        if len(idx) == 0:
            continue
        target_vote_fraction[row_pos] = float(np.mean(pred_matrix[idx, row_pos] == candidate_pred[row_pos]))

    selected_vote = np.divide(
        np.sum(vote_matrix.T * selected_mask, axis=1),
        np.maximum(selected_count, 1.0),
        out=np.zeros(n_rows, dtype=np.float64),
        where=np.maximum(selected_count, 1.0) > 0,
    )
    selected_disagreement = np.divide(
        np.sum(disagreement_matrix.T * selected_mask, axis=1),
        np.maximum(selected_count, 1.0),
        out=np.ones(n_rows, dtype=np.float64),
        where=np.maximum(selected_count, 1.0) > 0,
    )
    selected_support = np.divide(
        np.sum(support_matrix.T * selected_mask, axis=1),
        np.maximum(selected_count, 1.0),
        out=np.zeros(n_rows, dtype=np.float64),
        where=np.maximum(selected_count, 1.0) > 0,
    )

    risk_score = (
        0.28 * (1.0 - target_vote_fraction)
        + 1.80 * np.clip(selected_disagreement, 0.0, 0.12)
        + 0.20 * minority_risk.astype(np.float64)
        + 0.12 * (local_purity < 0.55).astype(np.float64)
        + 0.12 * (local_support_advantage < -0.010).astype(np.float64)
        + 0.08 * np.clip(baseline_margin / 0.45, 0.0, 1.0)
    )
    minority_rescue = (
        minority_risk
        & (selected_count >= 3)
        & (target_vote_fraction >= 0.67)
        & (local_support_advantage >= 0.030)
        & (local_purity >= 0.65)
    )
    if pvalue_rescue:
        minority_rescue = minority_rescue | (
            minority_risk
            & (selected_count >= 1)
            & (target_vote_fraction >= 0.99)
            & (selected_mean_score >= 0.50)
            & (selected_disagreement <= 0.075)
            & (local_support_advantage >= -0.020)
            & (local_p_values <= 0.75)
            & (local_purity >= 0.50)
        )
    support_floor = -0.025 if pvalue_rescue else -0.015
    risk_ceiling = 0.46 if pvalue_rescue else 0.42
    disagreement_ceiling = 0.075 if pvalue_rescue else 0.065
    eligible = (
        candidate_changed
        & (baseline_margin <= config.max_baseline_margin + 0.08)
        & (baseline_conf <= config.max_baseline_confidence + 0.03)
        & (selected_mean_score >= (0.50 if pvalue_rescue else 0.55))
        & (target_vote_fraction >= 0.50)
        & (selected_vote >= 0.67)
        & (selected_disagreement <= disagreement_ceiling)
        & (local_support_advantage >= support_floor)
        & ((~minority_risk) | minority_rescue)
        & (risk_score <= risk_ceiling)
    )
    pre_cap_eligible = eligible.copy()
    if config.max_changed_fraction < 1.0 and np.any(eligible):
        max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
        rank_score = (
            selected_mean_score
            + 0.25 * target_vote_fraction
            + local_support_advantage
            - risk_score
            - 0.15 * baseline_margin
        )
        keep = np.zeros_like(eligible, dtype=bool)
        idx = np.flatnonzero(eligible)
        keep[idx[np.argsort(rank_score[idx])[-max_changes:]]] = True
        eligible = keep

    alpha = np.where(
        eligible,
        np.clip(0.18 + 0.08 * selected_count + 0.10 * np.clip(selected_mean_score - 0.55, 0.0, 0.45), 0.0, 0.34),
        0.0,
    )
    proba_adj = (1.0 - alpha[:, None]) * baseline_proba + alpha[:, None] * candidate_proba
    proba_adj = proba_adj / np.maximum(np.sum(proba_adj, axis=1, keepdims=True), eps)
    proba_adj[~eligible] = baseline_proba[~eligible]
    changed = np.argmax(proba_adj, axis=1) != baseline_pred

    diagnostics = {
        "diversity_greedy_selected_count": selected_count,
        "diversity_greedy_selected_diversity": selected_diversity,
        "diversity_greedy_selected_score": np.where(np.isfinite(selected_mean_score), selected_mean_score, 0.0),
        "diversity_greedy_target_vote_fraction": target_vote_fraction,
        "diversity_greedy_risk_score": risk_score,
        "diversity_greedy_alpha": alpha,
        "diversity_greedy_pvalue_rescue": np.full(n_rows, float(pvalue_rescue)),
        "diversity_greedy_local_p_value": local_p_values,
        "no_shift_batch_prior_js": np.full(n_rows, js_value),
        "no_shift_eligible_final": eligible.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_vote_fraction": selected_vote,
        "no_shift_context_disagreement": selected_disagreement,
        "no_shift_local_support_margin": local_support_advantage,
        "no_shift_context_failure_fraction": np.zeros(n_rows, dtype=np.float64),
        "no_shift_local_purity": local_purity,
        "no_shift_minority_risk": minority_risk.astype(float),
        "no_shift_pre_cap_eligible_fraction": pre_cap_eligible.astype(float),
    }
    extra = {
        "diversity_greedy_selected": "per_sample_greedy_diverse_context_members",
        "diversity_greedy_pvalue_rescue": pvalue_rescue,
        "diversity_greedy_members": ",".join(member_names),
        "diversity_greedy_batch_prior_js": js_value,
        "diversity_greedy_mean_selected_count": float(np.mean(selected_count)),
        "diversity_greedy_changed_fraction": float(np.mean(changed)),
        "diversity_greedy_pre_cap_eligible_fraction": float(np.mean(pre_cap_eligible)),
        "diversity_greedy_mean_selected_diversity": float(np.mean(selected_diversity[selected_any])) if np.any(selected_any) else 0.0,
        "diversity_greedy_mean_target_vote_fraction": float(np.mean(target_vote_fraction[selected_any])) if np.any(selected_any) else 0.0,
        "diversity_greedy_mean_risk_score": float(np.mean(risk_score[selected_any])) if np.any(selected_any) else 0.0,
        "diversity_greedy_mean_alpha": float(np.mean(alpha)),
        "no_shift_batch_prior_js": js_value,
        "no_shift_eligible_fraction": float(np.mean(eligible)),
        "no_shift_changed_fraction": float(np.mean(changed)),
        "no_shift_mean_vote_fraction": float(np.mean(selected_vote)),
        "no_shift_mean_context_disagreement": float(np.mean(selected_disagreement[selected_any])) if np.any(selected_any) else 0.0,
        "no_shift_mean_local_support_margin": float(np.mean(local_support_advantage[selected_any])) if np.any(selected_any) else 0.0,
        "no_shift_mean_local_purity": float(np.mean(local_purity)),
        "no_shift_mean_local_support_p_value": float(np.mean(local_p_values[selected_any])) if np.any(selected_any) else 1.0,
        "no_shift_minority_risk_fraction": float(np.mean(minority_risk)),
        "no_shift_context_failure_fraction": 0.0,
    }
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=baseline_bundle.context_indices,
        effective_prior=baseline_bundle.effective_prior,
        gamma=alpha,
        delta=risk_score,
        runtime_sec=sum(member_runtimes),
        diagnostics=diagnostics,
    )
    return bundle, extra


def scale_aware_validation_instance_combo_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    large_scale = len(y_train) >= 192 or len(X_test) >= 96
    v27_config = replace(config, blend_weight=0.30)
    if not large_scale:
        bundle, details = guarded_boundary_union_v27_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            v27_config,
        )
        details.update(
            {
                "scale_combo_branch": "small_scale_v27",
                "scale_combo_large_scale": large_scale,
                "scale_combo_v39_accepted": False,
                "scale_combo_v40_accepted": False,
            }
        )
        return bundle, details

    v39_bundle, v39_details = guarded_boundary_union_validation_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        v27_config,
        seed,
    )
    if bool(v39_details.get("no_shift_validation_applied", False)):
        details = dict(v39_details)
        details.update(
            {
                "scale_combo_branch": "large_scale_validation_v27",
                "scale_combo_large_scale": large_scale,
                "scale_combo_v39_accepted": True,
                "scale_combo_v40_accepted": False,
            }
        )
        return v39_bundle, details

    v40_bundle, v40_details = instancewise_context_ensemble_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
    )
    changed_fraction = float(v40_details.get("instance_ensemble_changed_fraction") or 0.0)
    member_change = float(v40_details.get("instance_ensemble_mean_member_change_count") or 0.0)
    support_margin = float(v40_details.get("instance_ensemble_mean_local_support_margin") or 0.0)
    context_disagreement = float(v40_details.get("instance_ensemble_mean_context_disagreement") or 1.0)
    js_value = float(v40_details.get("instance_ensemble_batch_prior_js") or 0.0)
    stable_instance_signal = (
        changed_fraction > 0.0
        and member_change <= 0.035
        and support_margin >= 0.0
        and context_disagreement <= 0.035
        and js_value <= 0.004
    )
    details = dict(v39_details)
    details.update(v40_details)
    details.update(
        {
            "scale_combo_branch": (
                "large_scale_stable_instance_ensemble"
                if stable_instance_signal
                else "large_scale_baseline_guard"
            ),
            "scale_combo_large_scale": large_scale,
            "scale_combo_v39_accepted": False,
            "scale_combo_v40_accepted": stable_instance_signal,
            "scale_combo_v40_member_change_threshold": 0.035,
            "scale_combo_v40_context_threshold": 0.035,
            "scale_combo_v40_js_threshold": 0.004,
        }
    )
    if stable_instance_signal:
        return replace(
            v40_bundle,
            runtime_sec=float(v39_bundle.runtime_sec) + float(v40_bundle.runtime_sec),
        ), details
    return replace(
        baseline_bundle,
        runtime_sec=float(v39_bundle.runtime_sec) + float(v40_bundle.runtime_sec),
    ), details


def scale_aware_boundary_bypass_combo_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    large_scale = len(y_train) >= 192 or len(X_test) >= 96
    v27_config = replace(config, blend_weight=0.30)
    if not large_scale:
        bundle, details = guarded_boundary_union_v27_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            v27_config,
        )
        details.update(
            {
                "scale_combo_branch": "small_scale_v27",
                "scale_combo_large_scale": large_scale,
                "scale_combo_v39_accepted": False,
                "scale_combo_v27_bypass_accepted": False,
                "scale_combo_v40_accepted": False,
            }
        )
        return bundle, details

    v39_bundle, v39_details = guarded_boundary_union_validation_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        v27_config,
        seed,
    )
    if bool(v39_details.get("no_shift_validation_applied", False)):
        details = dict(v39_details)
        details.update(
            {
                "scale_combo_branch": "large_scale_validation_v27",
                "scale_combo_large_scale": large_scale,
                "scale_combo_v39_accepted": True,
                "scale_combo_v27_bypass_accepted": False,
                "scale_combo_v40_accepted": False,
            }
        )
        return v39_bundle, details

    v27_bundle, v27_details = guarded_boundary_union_v27_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        v27_config,
    )
    v27_changed = float(v27_details.get("no_shift_changed_fraction") or 0.0)
    v27_js = float(v27_details.get("no_shift_batch_prior_js") or 0.0)
    v27_context = float(v27_details.get("no_shift_mean_context_disagreement") or 0.0)
    v27_support = float(v27_details.get("no_shift_mean_local_support_margin") or 0.0)
    boundary_bypass = (
        v27_changed > 0.0
        and v27_js <= 0.001
        and v27_context >= 0.036
        and v27_support <= 0.020
    )
    if boundary_bypass:
        details = dict(v39_details)
        details.update(v27_details)
        details.update(
            {
                "scale_combo_branch": "large_scale_v27_boundary_bypass",
                "scale_combo_large_scale": large_scale,
                "scale_combo_v39_accepted": False,
                "scale_combo_v27_bypass_accepted": True,
                "scale_combo_v40_accepted": False,
                "scale_combo_v27_bypass_js_threshold": 0.001,
                "scale_combo_v27_bypass_context_threshold": 0.036,
                "scale_combo_v27_bypass_support_ceiling": 0.020,
            }
        )
        return replace(
            v27_bundle,
            runtime_sec=float(v39_bundle.runtime_sec) + float(v27_bundle.runtime_sec),
        ), details

    v40_bundle, v40_details = instancewise_context_ensemble_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
    )
    changed_fraction = float(v40_details.get("instance_ensemble_changed_fraction") or 0.0)
    member_change = float(v40_details.get("instance_ensemble_mean_member_change_count") or 0.0)
    support_margin = float(v40_details.get("instance_ensemble_mean_local_support_margin") or 0.0)
    context_disagreement = float(v40_details.get("instance_ensemble_mean_context_disagreement") or 1.0)
    js_value = float(v40_details.get("instance_ensemble_batch_prior_js") or 0.0)
    stable_instance_signal = (
        changed_fraction > 0.0
        and member_change <= 0.035
        and support_margin >= 0.0
        and context_disagreement <= 0.035
        and js_value <= 0.004
    )
    details = dict(v39_details)
    details.update(v27_details)
    details.update(v40_details)
    details.update(
        {
            "scale_combo_branch": (
                "large_scale_stable_instance_ensemble"
                if stable_instance_signal
                else "large_scale_baseline_guard"
            ),
            "scale_combo_large_scale": large_scale,
            "scale_combo_v39_accepted": False,
            "scale_combo_v27_bypass_accepted": False,
            "scale_combo_v40_accepted": stable_instance_signal,
            "scale_combo_v40_member_change_threshold": 0.035,
            "scale_combo_v40_context_threshold": 0.035,
            "scale_combo_v40_js_threshold": 0.004,
        }
    )
    if stable_instance_signal:
        return replace(
            v40_bundle,
            runtime_sec=(
                float(v39_bundle.runtime_sec)
                + float(v27_bundle.runtime_sec)
                + float(v40_bundle.runtime_sec)
            ),
        ), details
    return replace(
        baseline_bundle,
        runtime_sec=(
            float(v39_bundle.runtime_sec)
            + float(v27_bundle.runtime_sec)
            + float(v40_bundle.runtime_sec)
        ),
    ), details


def boundary_diversity_member_selector_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
    permissive_rescue: bool = False,
) -> tuple[APAPrediction, dict[str, object]]:
    """Sample-wise selector over V42 boundary and diversity rescue members."""

    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    n_rows, n_classes = baseline_proba.shape
    baseline_pred = np.argmax(baseline_proba, axis=1)
    sorted_base = np.sort(baseline_proba, axis=1)
    baseline_margin = sorted_base[:, -1] - sorted_base[:, -2] if n_classes > 1 else np.ones(n_rows)

    v42_config = replace(
        config,
        max_baseline_margin=0.32,
        max_baseline_confidence=0.90,
        max_context_disagreement=0.050,
        min_context_vote_fraction=0.67,
        local_evidence_margin=0.015,
        blend_weight=0.24,
        max_changed_fraction=0.03125,
        families=("diverse_knn", "prototype", "feature_subspace_knn"),
    )
    v42_bundle, v42_details = scale_aware_boundary_bypass_combo_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        v42_config,
        seed,
    )
    rescue_config = config
    if permissive_rescue:
        rescue_config = replace(
            config,
            max_baseline_margin=0.34,
            max_baseline_confidence=0.91,
            max_context_disagreement=0.055,
            min_context_vote_fraction=0.67,
            local_evidence_margin=0.000,
            blend_weight=0.24,
            max_changed_fraction=0.03125,
            families=("diverse_knn", "prototype", "clean_knn", "boundary_contrast", "feature_subspace_knn"),
        )

    v47_bundle, v47_details = diversity_greedy_instance_router_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        rescue_config,
        pvalue_rescue=True,
    )

    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=n_classes).astype(np.float64)
    train_prior = train_counts / max(1.0, float(np.sum(train_counts)))
    minority_mask = train_prior <= 0.35
    if n_classes > 1 and np.any(train_counts > 0):
        minority_mask[int(np.argmin(np.where(train_counts > 0, train_counts, np.inf)))] = True

    member_bundles = [v42_bundle, v47_bundle]
    member_names = ["v42_boundary_bypass", "v47_diversity_pvalue"]
    member_probas = [np.asarray(bundle.proba_adj, dtype=np.float64) for bundle in member_bundles]
    member_preds = [np.argmax(proba, axis=1) for proba in member_probas]
    member_changed = [pred != baseline_pred for pred in member_preds]

    candidate_proba = baseline_proba.copy()
    chosen = np.zeros(n_rows, dtype=bool)
    chosen_member = np.full(n_rows, -1, dtype=int)
    chosen_score = np.full(n_rows, -np.inf, dtype=np.float64)
    local_supports: list[np.ndarray] = []
    local_p_values: list[np.ndarray] = []
    local_purities: list[np.ndarray] = []
    minority_risks: list[np.ndarray] = []

    for member_pos, (proba, pred, changed) in enumerate(zip(member_probas, member_preds, member_changed)):
        local_candidate_support, local_baseline_support, local_purity = _local_neighbor_evidence(
            X_train,
            y_train_pos,
            X_test,
            pred,
            baseline_pred,
            n_classes,
            k=max(config.k, 16),
        )
        support_advantage = local_candidate_support - local_baseline_support
        p_values = _local_support_p_values(
            local_candidate_support,
            pred,
            train_prior,
            k=max(config.k, 16),
        )
        baseline_is_minority = minority_mask[baseline_pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
        candidate_is_majority = ~minority_mask[pred] if len(minority_mask) else np.zeros(n_rows, dtype=bool)
        minority_risk = changed & baseline_is_minority & candidate_is_majority
        confidence_gain = proba[np.arange(n_rows), pred] - baseline_proba[np.arange(n_rows), baseline_pred]
        entropy_gain = _entropy_rows(baseline_proba) - _entropy_rows(proba)
        if member_pos == 0:
            eligible = changed
            base_bonus = 0.35
        else:
            minority_rescue = (
                minority_risk
                & (support_advantage >= -0.010)
                & (p_values <= 0.72)
                & (local_purity >= 0.50)
            )
            eligible = (
                changed
                & (~member_changed[0] | (pred == member_preds[0]))
                & (baseline_margin <= config.max_baseline_margin + 0.08)
                & (support_advantage >= -0.020)
                & (p_values <= 0.78)
                & (local_purity >= 0.50)
                & ((~minority_risk) | minority_rescue)
            )
            base_bonus = 0.0
        score = (
            base_bonus
            + 0.60 * confidence_gain
            + 0.30 * entropy_gain
            + 1.50 * support_advantage
            - 0.30 * p_values
            - 0.15 * minority_risk.astype(np.float64)
            - 0.10 * baseline_margin
        )
        take = eligible & (score > chosen_score)
        if np.any(take):
            candidate_proba[take] = proba[take]
            chosen[take] = True
            chosen_member[take] = member_pos
            chosen_score[take] = score[take]
        local_supports.append(support_advantage)
        local_p_values.append(p_values)
        local_purities.append(local_purity)
        minority_risks.append(minority_risk)

    cap_fraction = min(0.046875, max(config.max_changed_fraction, 1.5 * config.max_changed_fraction))
    pre_cap_chosen = chosen.copy()
    if cap_fraction < 1.0 and np.any(chosen):
        max_changes = max(1, int(np.ceil(float(cap_fraction) * n_rows)))
        keep = np.zeros_like(chosen, dtype=bool)
        idx = np.flatnonzero(chosen)
        keep[idx[np.argsort(chosen_score[idx])[-max_changes:]]] = True
        candidate_proba = np.where(keep[:, None], candidate_proba, baseline_proba)
        chosen = keep
        chosen_member[~keep] = -1
        chosen_score[~keep] = -np.inf

    proba_adj = candidate_proba / np.maximum(candidate_proba.sum(axis=1, keepdims=True), 1e-12)
    changed = np.argmax(proba_adj, axis=1) != baseline_pred
    support_matrix = np.vstack(local_supports)
    pvalue_matrix = np.vstack(local_p_values)
    purity_matrix = np.vstack(local_purities)
    minority_matrix = np.vstack(minority_risks)
    chosen_support = np.zeros(n_rows, dtype=np.float64)
    chosen_pvalue = np.ones(n_rows, dtype=np.float64)
    chosen_purity = np.zeros(n_rows, dtype=np.float64)
    chosen_minority_risk = np.zeros(n_rows, dtype=np.float64)
    for row_pos, member_pos in enumerate(chosen_member):
        if member_pos >= 0:
            chosen_support[row_pos] = support_matrix[member_pos, row_pos]
            chosen_pvalue[row_pos] = pvalue_matrix[member_pos, row_pos]
            chosen_purity[row_pos] = purity_matrix[member_pos, row_pos]
            chosen_minority_risk[row_pos] = float(minority_matrix[member_pos, row_pos])

    diagnostics = {
        "member_selector_chosen": chosen.astype(float),
        "member_selector_pre_cap_chosen": pre_cap_chosen.astype(float),
        "member_selector_chosen_member": chosen_member.astype(float),
        "member_selector_chosen_score": np.where(np.isfinite(chosen_score), chosen_score, 0.0),
        "member_selector_local_support_margin": chosen_support,
        "member_selector_local_p_value": chosen_pvalue,
        "member_selector_local_purity": chosen_purity,
        "member_selector_minority_risk": chosen_minority_risk,
        "member_selector_v42_changed": member_changed[0].astype(float),
        "member_selector_v47_changed": member_changed[1].astype(float),
        "no_shift_batch_prior_js": np.asarray(
            getattr(v42_bundle, "diagnostics", {}).get(
                "no_shift_batch_prior_js",
                getattr(v47_bundle, "diagnostics", {}).get("no_shift_batch_prior_js", np.zeros(n_rows)),
            ),
            dtype=np.float64,
        ),
        "no_shift_eligible_final": chosen.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_vote_fraction": chosen.astype(float),
        "no_shift_context_disagreement": np.zeros(n_rows, dtype=np.float64),
        "no_shift_local_support_margin": chosen_support,
        "no_shift_context_failure_fraction": np.zeros(n_rows, dtype=np.float64),
    }
    extra = {
        "member_selector_members": ",".join(member_names),
        "member_selector_changed_fraction": float(np.mean(changed)),
        "member_selector_pre_cap_eligible_fraction": float(np.mean(pre_cap_chosen)),
        "member_selector_v42_changed_fraction": float(np.mean(member_changed[0])),
        "member_selector_v47_changed_fraction": float(np.mean(member_changed[1])),
        "member_selector_v42_selected_fraction": float(np.mean(chosen_member == 0)),
        "member_selector_v47_selected_fraction": float(np.mean(chosen_member == 1)),
        "member_selector_mean_local_support_margin": float(np.mean(chosen_support[chosen])) if np.any(chosen) else 0.0,
        "member_selector_mean_local_p_value": float(np.mean(chosen_pvalue[chosen])) if np.any(chosen) else 1.0,
        "member_selector_mean_local_purity": float(np.mean(chosen_purity[chosen])) if np.any(chosen) else 0.0,
        "member_selector_minority_risk_fraction": float(np.mean(chosen_minority_risk[chosen])) if np.any(chosen) else 0.0,
        "member_selector_v42_branch": v42_details.get("scale_combo_branch", ""),
        "member_selector_v47_selected": v47_details.get("diversity_greedy_selected", ""),
        "member_selector_permissive_rescue": permissive_rescue,
        "no_shift_batch_prior_js": float(np.mean(diagnostics["no_shift_batch_prior_js"])),
        "no_shift_eligible_fraction": float(np.mean(chosen)),
        "no_shift_changed_fraction": float(np.mean(changed)),
        "no_shift_mean_local_support_margin": float(np.mean(chosen_support[chosen])) if np.any(chosen) else 0.0,
        "no_shift_mean_local_support_p_value": float(np.mean(chosen_pvalue[chosen])) if np.any(chosen) else 1.0,
        "no_shift_context_failure_fraction": 0.0,
    }
    details = dict(v42_details)
    details.update({f"v47_{key}": value for key, value in v47_details.items()})
    details.update(extra)
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=v42_bundle.context_indices,
        effective_prior=v42_bundle.effective_prior,
        gamma=chosen.astype(float),
        delta=np.where(np.isfinite(chosen_score), chosen_score, 0.0),
        runtime_sec=float(v42_bundle.runtime_sec) + float(v47_bundle.runtime_sec),
        diagnostics=diagnostics,
    )
    return bundle, details


def cross_model_tree_consensus_rescue_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
    calibrated_tree_rescue: bool = False,
    primary_member: str = "v42",
    prp_tree_rescue: bool = False,
) -> tuple[APAPrediction, dict[str, object]]:
    """V42 primary member plus a guarded ExtraTrees/RandomForest rescue member."""

    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    n_rows, n_classes = baseline_proba.shape
    if n_rows == 0 or n_classes < 2:
        return baseline_bundle, {"cross_model_selected": "baseline_empty_or_single_class"}

    v42_config = replace(
        config,
        max_baseline_margin=0.32,
        max_baseline_confidence=0.90,
        max_context_disagreement=0.050,
        min_context_vote_fraction=0.67,
        local_evidence_margin=0.015,
        blend_weight=0.24,
        max_changed_fraction=0.03125,
        families=("diverse_knn", "prototype", "feature_subspace_knn"),
    )
    if primary_member == "v27_fast":
        primary_bundle, primary_details = guarded_boundary_union_v27_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            replace(v42_config, blend_weight=0.30),
        )
        primary_name = "v27_guarded_boundary"
        primary_branch = "v27_fast_primary"
    else:
        primary_bundle, primary_details = scale_aware_boundary_bypass_combo_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            v42_config,
            seed,
        )
        primary_name = "v42_boundary_bypass"
        primary_branch = str(primary_details.get("scale_combo_branch", ""))

    primary_proba = np.asarray(primary_bundle.proba_adj, dtype=np.float64)
    primary_pred = np.argmax(primary_proba, axis=1)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    primary_changed = primary_pred != baseline_pred
    primary_margin = _top2_margin_rows(primary_proba)
    primary_conf = np.max(primary_proba, axis=1)

    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=n_classes).astype(np.float64)
    train_prior = train_counts / max(1.0, float(np.sum(train_counts)))
    primary_js = batch_prior_js(primary_proba, y_train, classes, alpha=1.0)
    if primary_js > config.max_batch_prior_js:
        details = dict(primary_details)
        details.update(
            {
                "cross_model_selected": f"{primary_name}_only_primary_js_guard",
                "cross_model_primary_member": primary_name,
                "cross_model_primary_branch": primary_branch,
                "cross_model_primary_batch_prior_js": primary_js,
                "cross_model_changed_fraction": float(np.mean(primary_changed)),
                "cross_model_tree_selected_fraction": 0.0,
            }
        )
        return primary_bundle, details

    tree_rescue_js_threshold = min(0.0015, config.max_batch_prior_js)
    if primary_js > tree_rescue_js_threshold:
        details = dict(primary_details)
        details.update(
            {
                "cross_model_selected": f"{primary_name}_only_tree_rescue_js_guard",
                "cross_model_primary_member": primary_name,
                "cross_model_primary_branch": primary_branch,
                "cross_model_primary_batch_prior_js": primary_js,
                "cross_model_tree_rescue_js_threshold": tree_rescue_js_threshold,
                "cross_model_changed_fraction": float(np.mean(primary_changed)),
                "cross_model_tree_selected_fraction": 0.0,
                "cross_model_primary_changed_fraction": float(np.mean(primary_changed)),
            }
        )
        return primary_bundle, details

    tree_start = time.time()
    leaf_size = max(2, min(8, int(math.ceil(0.015 * max(1, len(y_train))))))
    tree_models = [
        ExtraTreesClassifier(
            n_estimators=128,
            max_features="sqrt",
            min_samples_leaf=leaf_size,
            class_weight="balanced",
            random_state=seed + 1701,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=128,
            max_features="sqrt",
            min_samples_leaf=leaf_size,
            class_weight="balanced",
            random_state=seed + 2609,
            n_jobs=1,
        ),
    ]
    tree_probas: list[np.ndarray] = []
    tree_preds: list[np.ndarray] = []
    try:
        for model in tree_models:
            model.fit(X_train, y_train)
            aligned = _align_sklearn_proba_to_classes(model, model.predict_proba(X_test), classes)
            tree_probas.append(aligned)
            tree_preds.append(np.argmax(aligned, axis=1))
    except Exception as exc:
        details = dict(primary_details)
        details.update(
            {
                "cross_model_selected": f"{primary_name}_only_tree_fit_failed",
                "cross_model_primary_member": primary_name,
                "cross_model_primary_branch": primary_branch,
                "cross_model_error": str(exc),
                "cross_model_changed_fraction": float(np.mean(primary_changed)),
                "cross_model_tree_selected_fraction": 0.0,
            }
        )
        return replace(primary_bundle, runtime_sec=float(primary_bundle.runtime_sec) + time.time() - tree_start), details

    tree_proba = np.mean(np.stack(tree_probas, axis=0), axis=0)
    tree_proba /= np.maximum(np.sum(tree_proba, axis=1, keepdims=True), 1e-12)
    tree_pred = np.argmax(tree_proba, axis=1)
    tree_margin = _top2_margin_rows(tree_proba)
    tree_conf = np.max(tree_proba, axis=1)
    model_agreement = np.logical_and.reduce([pred == tree_preds[0] for pred in tree_preds])
    tree_js = batch_prior_js(tree_proba, y_train, classes, alpha=1.0)

    local_candidate_support, local_primary_support, local_purity = _local_neighbor_evidence(
        X_train,
        y_train_pos,
        X_test,
        tree_pred,
        primary_pred,
        n_classes,
        k=max(config.k, 16),
    )
    support_advantage = local_candidate_support - local_primary_support
    p_values = _local_support_p_values(
        local_candidate_support,
        tree_pred,
        train_prior,
        k=max(config.k, 16),
    )
    tree_pred_prior = np.bincount(tree_pred, minlength=n_classes).astype(np.float64) / max(1.0, float(n_rows))
    candidate_prior_delta = np.abs(tree_pred_prior[tree_pred] - train_prior[tree_pred])
    minority_mask = train_prior <= 0.35
    if np.any(train_counts > 0):
        minority_mask[int(np.argmin(np.where(train_counts > 0, train_counts, np.inf)))] = True
    primary_is_minority = minority_mask[primary_pred]
    tree_is_majority = ~minority_mask[tree_pred]
    minority_to_majority_risk = primary_is_minority & tree_is_majority

    candidate_changed = tree_pred != primary_pred
    minority_rescue = (
        minority_to_majority_risk
        & (support_advantage >= 0.16)
        & (p_values <= 0.18)
        & (local_purity >= 0.58)
    )
    eligible = (
        candidate_changed
        & ~primary_changed
        & model_agreement
        & (primary_js <= tree_rescue_js_threshold)
        & (tree_js <= 0.085)
        & (primary_margin <= config.max_baseline_margin + 0.04)
        & (tree_margin >= 0.08)
        & (tree_conf >= 0.54)
        & (tree_conf >= primary_conf - 0.04)
        & (support_advantage >= 0.08)
        & (p_values <= 0.35)
        & (local_purity >= 0.52)
        & (candidate_prior_delta <= 0.28)
        & ((~minority_to_majority_risk) | minority_rescue)
    )

    prp_details: dict[str, object] = {}
    prp_risk_score = np.ones(n_rows, dtype=np.float64)
    prp_context_disagreement = np.ones(n_rows, dtype=np.float64)
    prp_agreement_ratio = np.zeros(n_rows, dtype=np.float64)
    prp_local_purity = np.zeros(n_rows, dtype=np.float64)
    prp_candidate_fraction = np.zeros(n_rows, dtype=np.float64)
    prp_agrees_with_tree = np.zeros(n_rows, dtype=bool)
    prp_extra_eligible = np.zeros(n_rows, dtype=bool)
    prp_rejected_v50 = np.zeros(n_rows, dtype=bool)
    prp_evaluated = False
    prp_loose_candidate = (
        candidate_changed
        & ~primary_changed
        & model_agreement
        & (primary_js <= tree_rescue_js_threshold)
        & (tree_js <= 0.045)
        & (primary_margin <= config.max_baseline_margin + 0.08)
        & (tree_margin >= 0.07)
        & (tree_conf >= 0.55)
        & (tree_conf >= primary_conf - 0.03)
        & (support_advantage >= 0.055)
        & (p_values <= 0.30)
        & (local_purity >= 0.52)
        & (candidate_prior_delta <= 0.24)
        & ((~minority_to_majority_risk) | minority_rescue)
    )
    if prp_tree_rescue and np.any(eligible | prp_loose_candidate):
        prp_evaluated = True
        prp_bundle, prp_details = family_agreement_risk_predict(
            X_train,
            y_train,
            X_test,
            primary_bundle,
            estimator_factory,
            replace(config, max_changed_fraction=1.0),
        )
        prp_diag = getattr(prp_bundle, "diagnostics", {})
        prp_risk_score = np.asarray(prp_diag.get("no_shift_risk_score", prp_risk_score), dtype=np.float64)
        prp_context_disagreement = np.asarray(
            prp_diag.get("no_shift_context_disagreement", prp_context_disagreement),
            dtype=np.float64,
        )
        prp_agreement_ratio = np.asarray(
            prp_diag.get("no_shift_family_agreement_ratio", prp_agreement_ratio),
            dtype=np.float64,
        )
        prp_local_purity = np.asarray(prp_diag.get("no_shift_local_purity", prp_local_purity), dtype=np.float64)
        prp_candidate_fraction = np.asarray(
            prp_diag.get("no_shift_candidate_fraction", prp_candidate_fraction),
            dtype=np.float64,
        )
        prp_pred = np.argmax(np.asarray(prp_bundle.proba_adj, dtype=np.float64), axis=1)
        prp_agrees_with_tree = (prp_pred == tree_pred) & (tree_pred != primary_pred)
        prp_reliability_pass = (
            (
                (prp_risk_score <= 0.58)
                & (prp_agreement_ratio >= 0.20)
                & (prp_context_disagreement <= 0.065)
                & (prp_local_purity >= 0.45)
            )
            | (prp_agrees_with_tree & (prp_risk_score <= 0.70))
        )
        strong_tree_evidence = (
            (support_advantage >= 0.15)
            & (p_values <= 0.20)
            & (tree_margin >= 0.18)
            & (local_purity >= 0.55)
        )
        prp_rejected_v50 = eligible & ~(prp_reliability_pass | strong_tree_evidence)
        eligible = eligible & (prp_reliability_pass | strong_tree_evidence)
        prp_extra_eligible = prp_loose_candidate & prp_reliability_pass
        eligible = eligible | prp_extra_eligible
    pre_cap_eligible = eligible.copy()
    score = (
        1.60 * support_advantage
        + 0.70 * tree_margin
        + 0.35 * (tree_conf - primary_conf)
        - 0.35 * p_values
        - 0.20 * primary_margin
        - 0.15 * candidate_prior_delta
        - 0.20 * minority_to_majority_risk.astype(np.float64)
    )
    if prp_tree_rescue:
        score = (
            score
            + 0.24 * prp_agreement_ratio
            + 0.12 * prp_agrees_with_tree.astype(np.float64)
            - 0.18 * prp_risk_score
            - 0.10 * prp_context_disagreement
        )
    if config.max_changed_fraction < 1.0 and np.any(eligible):
        max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
        keep = np.zeros_like(eligible, dtype=bool)
        idx = np.flatnonzero(eligible)
        keep[idx[np.argsort(score[idx])[-max_changes:]]] = True
        eligible = keep

    proba_adj = primary_proba.copy()
    calibration_boost = np.zeros(n_rows, dtype=np.float64)
    if np.any(eligible):
        alpha = 0.46
        blended = (1.0 - alpha) * primary_proba[eligible] + alpha * tree_proba[eligible]
        blended /= np.maximum(np.sum(blended, axis=1, keepdims=True), 1e-12)
        proba_adj[eligible] = blended
        stalled = eligible & (np.argmax(proba_adj, axis=1) != tree_pred)
        for row_pos in np.flatnonzero(stalled):
            old_pos = int(primary_pred[row_pos])
            new_pos = int(tree_pred[row_pos])
            needed = max(0.0, proba_adj[row_pos, old_pos] - proba_adj[row_pos, new_pos] + 1e-6)
            move = min(0.18, needed + 0.012)
            if move > 0.0:
                proba_adj[row_pos, old_pos] = max(1e-9, proba_adj[row_pos, old_pos] - move)
                proba_adj[row_pos, new_pos] += move
        proba_adj /= np.maximum(np.sum(proba_adj, axis=1, keepdims=True), 1e-12)
        if calibrated_tree_rescue:
            for row_pos in np.flatnonzero(eligible):
                new_pos = int(tree_pred[row_pos])
                current_conf = float(proba_adj[row_pos, new_pos])
                target_conf = (
                    0.60
                    + 0.50 * float(np.clip(support_advantage[row_pos], 0.0, 0.40))
                    + 0.22 * float(np.clip(tree_margin[row_pos], 0.0, 0.50))
                    - 0.08 * float(np.clip(p_values[row_pos], 0.0, 1.0))
                )
                target_conf = float(np.clip(max(current_conf, target_conf), current_conf, 0.86))
                if target_conf <= current_conf + 1e-9:
                    continue
                other_sum = max(1e-12, 1.0 - current_conf)
                scale = (1.0 - target_conf) / other_sum
                proba_adj[row_pos] *= scale
                proba_adj[row_pos, new_pos] = target_conf
                proba_adj[row_pos] /= max(float(np.sum(proba_adj[row_pos])), 1e-12)
                calibration_boost[row_pos] = target_conf - current_conf

    changed = np.argmax(proba_adj, axis=1) != baseline_pred
    tree_selected = eligible.astype(float)
    diagnostics = {
        "cross_model_tree_selected": tree_selected,
        "cross_model_tree_pre_cap_eligible": pre_cap_eligible.astype(float),
        "cross_model_tree_model_agreement": model_agreement.astype(float),
        "cross_model_tree_support_advantage": support_advantage,
        "cross_model_tree_local_p_value": p_values,
        "cross_model_tree_local_purity": local_purity,
        "cross_model_tree_margin": tree_margin,
        "cross_model_tree_confidence": tree_conf,
        "cross_model_tree_calibration_boost": calibration_boost,
        "cross_model_prp_risk_score": prp_risk_score,
        "cross_model_prp_context_disagreement": prp_context_disagreement,
        "cross_model_prp_agreement_ratio": prp_agreement_ratio,
        "cross_model_prp_local_purity": prp_local_purity,
        "cross_model_prp_candidate_fraction": prp_candidate_fraction,
        "cross_model_prp_agrees_with_tree": prp_agrees_with_tree.astype(float),
        "cross_model_prp_extra_eligible": prp_extra_eligible.astype(float),
        "cross_model_prp_rejected_v50": prp_rejected_v50.astype(float),
        "cross_model_primary_margin": primary_margin,
        "cross_model_candidate_prior_delta": candidate_prior_delta,
        "cross_model_minority_to_majority_risk": minority_to_majority_risk.astype(float),
        "no_shift_batch_prior_js": np.full(n_rows, primary_js, dtype=np.float64),
        "no_shift_eligible_final": tree_selected,
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_vote_fraction": model_agreement.astype(float),
        "no_shift_context_disagreement": np.full(n_rows, tree_js, dtype=np.float64),
        "no_shift_local_support_margin": support_advantage,
        "no_shift_context_failure_fraction": np.zeros(n_rows, dtype=np.float64),
    }
    extra = {
        "cross_model_selected": (
            f"{primary_name}_plus_calibrated_tree_consensus_rescue"
            if calibrated_tree_rescue
            else (
                f"{primary_name}_plus_prp_tree_consensus_rescue"
                if prp_tree_rescue
                else f"{primary_name}_plus_tree_consensus_rescue"
            )
        ),
        "cross_model_members": f"{primary_name},extratrees,randomforest",
        "cross_model_primary_member": primary_name,
        "cross_model_primary_branch": primary_branch,
        "cross_model_calibrated_tree_rescue": calibrated_tree_rescue,
        "cross_model_prp_tree_rescue": prp_tree_rescue,
        "cross_model_prp_evaluated": prp_evaluated,
        "cross_model_leaf_size": leaf_size,
        "cross_model_primary_batch_prior_js": primary_js,
        "cross_model_tree_rescue_js_threshold": tree_rescue_js_threshold,
        "cross_model_tree_batch_prior_js": tree_js,
        "cross_model_tree_pre_cap_eligible_fraction": float(np.mean(pre_cap_eligible)),
        "cross_model_tree_selected_fraction": float(np.mean(eligible)),
        "cross_model_prp_loose_candidate_fraction": float(np.mean(prp_loose_candidate)),
        "cross_model_prp_extra_eligible_fraction": float(np.mean(prp_extra_eligible)),
        "cross_model_prp_rejected_v50_fraction": float(np.mean(prp_rejected_v50)),
        "cross_model_prp_mean_risk_selected": float(np.mean(prp_risk_score[eligible])) if np.any(eligible) else 0.0,
        "cross_model_prp_mean_agreement_selected": (
            float(np.mean(prp_agreement_ratio[eligible])) if np.any(eligible) else 0.0
        ),
        "cross_model_prp_tree_agreement_selected_fraction": (
            float(np.mean(prp_agrees_with_tree[eligible])) if np.any(eligible) else 0.0
        ),
        "cross_model_tree_model_agreement_fraction": float(np.mean(model_agreement)),
        "cross_model_mean_support_advantage_selected": (
            float(np.mean(support_advantage[eligible])) if np.any(eligible) else 0.0
        ),
        "cross_model_mean_p_value_selected": float(np.mean(p_values[eligible])) if np.any(eligible) else 1.0,
        "cross_model_mean_tree_margin_selected": float(np.mean(tree_margin[eligible])) if np.any(eligible) else 0.0,
        "cross_model_mean_calibration_boost_selected": (
            float(np.mean(calibration_boost[eligible])) if np.any(eligible) else 0.0
        ),
        "cross_model_minority_to_majority_selected_fraction": (
            float(np.mean(minority_to_majority_risk[eligible])) if np.any(eligible) else 0.0
        ),
        "cross_model_primary_changed_fraction": float(np.mean(primary_changed)),
        "cross_model_changed_fraction": float(np.mean(changed)),
        "cross_model_v42_branch": primary_branch if primary_name == "v42_boundary_bypass" else "",
        "no_shift_batch_prior_js": primary_js,
        "no_shift_eligible_fraction": float(np.mean(eligible)),
        "no_shift_changed_fraction": float(np.mean(changed)),
        "no_shift_mean_local_support_margin": (
            float(np.mean(support_advantage[eligible])) if np.any(eligible) else 0.0
        ),
        "no_shift_context_failure_fraction": 0.0,
    }
    details = dict(primary_details)
    if prp_tree_rescue:
        details.update({f"cross_model_prp_diag_{key}": value for key, value in prp_details.items()})
    details.update(extra)
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=primary_bundle.context_indices,
        effective_prior=primary_bundle.effective_prior,
        gamma=tree_selected,
        delta=score,
        runtime_sec=float(primary_bundle.runtime_sec) + time.time() - tree_start,
        diagnostics=diagnostics,
    )
    return bundle, details


def context_consensus_calibrated_v42_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    v42_bundle, details = scale_aware_boundary_bypass_combo_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
        seed,
    )
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    v42_proba = np.asarray(v42_bundle.proba_adj, dtype=np.float64)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    v42_pred = np.argmax(v42_proba, axis=1)
    changed = v42_pred != baseline_pred
    if not np.any(changed):
        details = dict(details)
        details.update(
            {
                "calibration_v44_applied": False,
                "calibration_v44_changed_fraction": 0.0,
                "calibration_v44_mean_shrink": 0.0,
                "calibration_v44_argmax_guard_fraction": 0.0,
                "calibration_v44_mean_confidence_before": float(np.mean(np.max(v42_proba, axis=1))),
                "calibration_v44_mean_confidence_after": float(np.mean(np.max(v42_proba, axis=1))),
            }
        )
        return v42_bundle, details

    n_classes = max(1, v42_proba.shape[1])
    baseline_entropy = _entropy_rows(baseline_proba) / max(math.log(n_classes), 1e-12)
    v42_margin = _top2_margin_rows(v42_proba)
    diagnostics = getattr(v42_bundle, "diagnostics", {})
    context_disagreement = np.asarray(
        diagnostics.get(
            "no_shift_context_disagreement",
            diagnostics.get("instance_ensemble_context_disagreement", np.zeros(len(X_test))),
        ),
        dtype=np.float64,
    )
    if context_disagreement.shape[0] != len(X_test):
        context_disagreement = np.zeros(len(X_test), dtype=np.float64)

    disagreement_risk = np.clip(context_disagreement / 0.060, 0.0, 1.0)
    low_margin_risk = 1.0 - np.clip(v42_margin / 0.25, 0.0, 1.0)
    uncertainty_risk = np.clip(baseline_entropy, 0.0, 1.0)
    shrink = 0.06 + 0.10 * disagreement_risk + 0.08 * low_margin_risk + 0.06 * uncertainty_risk
    shrink = np.where(changed, np.clip(shrink, 0.0, 0.26), 0.0)

    calibrated = (1.0 - shrink[:, None]) * v42_proba + shrink[:, None] * baseline_proba
    calibrated = calibrated / np.clip(calibrated.sum(axis=1, keepdims=True), 1e-12, None)
    argmax_guard = np.argmax(calibrated, axis=1) != v42_pred
    calibrated[argmax_guard] = v42_proba[argmax_guard]
    effective_shrink = np.where(argmax_guard, 0.0, shrink)
    pred = np.argmax(calibrated, axis=1)
    details = dict(details)
    details.update(
        {
            "calibration_v44_applied": True,
            "calibration_v44_changed_fraction": float(np.mean(changed)),
            "calibration_v44_mean_shrink": float(np.mean(effective_shrink[changed])),
            "calibration_v44_max_shrink": float(np.max(effective_shrink[changed])),
            "calibration_v44_argmax_guard_fraction": float(np.mean(argmax_guard[changed])),
            "calibration_v44_mean_context_disagreement": float(np.mean(context_disagreement[changed])),
            "calibration_v44_mean_baseline_entropy": float(np.mean(baseline_entropy[changed])),
            "calibration_v44_mean_margin": float(np.mean(v42_margin[changed])),
            "calibration_v44_mean_confidence_before": float(np.mean(np.max(v42_proba, axis=1))),
            "calibration_v44_mean_confidence_after": float(np.mean(np.max(calibrated, axis=1))),
        }
    )
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "calibration_v44_shrink": effective_shrink,
            "calibration_v44_argmax_guard": argmax_guard.astype(np.float64),
        }
    )
    return replace(
        v42_bundle,
        proba_adj=calibrated,
        pred=v42_bundle.classes[pred],
        diagnostics=diagnostics,
    ), details


def pseudo_label_context_augmentation_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    """Use high-confidence test-batch pseudo labels as extra context."""

    start = time.time()
    eps = 1e-12
    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(baseline_bundle.classes)
    n_rows, n_classes = baseline_proba.shape
    if n_rows == 0 or n_classes < 2:
        return baseline_bundle, {"pseudo_label_selected": "baseline_empty_or_single_class"}

    baseline_pred = np.argmax(baseline_proba, axis=1)
    baseline_margin = _top2_margin_rows(baseline_proba)
    baseline_conf = np.max(baseline_proba, axis=1)
    js_value = batch_prior_js(baseline_proba, y_train, classes, alpha=1.0)
    if js_value > config.max_batch_prior_js:
        return baseline_bundle, {
            "pseudo_label_selected": "baseline_batch_prior_js_guard",
            "pseudo_label_batch_prior_js": js_value,
            "pseudo_label_changed_fraction": 0.0,
        }

    pseudo_mask = (baseline_conf >= 0.90) & (baseline_margin >= 0.48)
    min_pool = max(3, min(8, n_rows // 8))
    if np.sum(pseudo_mask) < min_pool:
        return baseline_bundle, {
            "pseudo_label_selected": "baseline_insufficient_pseudo_pool",
            "pseudo_label_batch_prior_js": js_value,
            "pseudo_label_pool_fraction": float(np.mean(pseudo_mask)),
            "pseudo_label_changed_fraction": 0.0,
        }

    candidate_mask = (baseline_margin <= config.max_baseline_margin + 0.04) & (baseline_conf <= 0.92)
    candidate_mask &= ~pseudo_mask
    if not np.any(candidate_mask):
        return baseline_bundle, {
            "pseudo_label_selected": "baseline_no_low_margin_targets",
            "pseudo_label_batch_prior_js": js_value,
            "pseudo_label_pool_fraction": float(np.mean(pseudo_mask)),
            "pseudo_label_changed_fraction": 0.0,
        }

    max_targets = max(1, min(8, int(math.ceil(0.125 * n_rows))))
    target_idx = np.flatnonzero(candidate_mask)
    target_score = (1.0 - baseline_conf[target_idx]) + (0.40 - baseline_margin[target_idx])
    target_idx = target_idx[np.argsort(target_score)[-max_targets:]]

    pseudo_idx_all = np.flatnonzero(pseudo_mask)
    pseudo_idx_all = pseudo_idx_all[np.argsort(baseline_conf[pseudo_idx_all])[-min(32, len(pseudo_idx_all)):]]

    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=n_classes).astype(np.float64)
    train_prior = train_counts / max(1.0, float(np.sum(train_counts)))
    proba_candidates = baseline_proba.copy()
    attempted = np.zeros(n_rows, dtype=bool)
    runtime_extra = 0.0
    pseudo_pool_size = np.zeros(n_rows, dtype=np.float64)

    for row_pos in target_idx:
        row_pos = int(row_pos)
        pseudo_idx = pseudo_idx_all[pseudo_idx_all != row_pos]
        if len(pseudo_idx) < min_pool:
            continue
        attempted[row_pos] = True
        pseudo_pool_size[row_pos] = float(len(pseudo_idx))
        X_aug = np.concatenate([X_train, X_test[pseudo_idx]], axis=0)
        y_aug = np.concatenate([y_train, classes[baseline_pred[pseudo_idx]]], axis=0)
        fit_start = time.time()
        try:
            estimator = estimator_factory()
            estimator.fit(X_aug, y_aug)
            raw = estimator.predict_proba(np.asarray(X_test[row_pos], dtype=np.float32).reshape(1, -1))
            pred_proba = _align_sklearn_proba_to_classes(estimator, raw, classes)[0]
        except Exception:
            runtime_extra += time.time() - fit_start
            continue
        runtime_extra += time.time() - fit_start
        pred_proba = np.asarray(pred_proba, dtype=np.float64)
        pred_proba /= max(float(np.sum(pred_proba)), eps)
        proba_candidates[row_pos] = pred_proba

    candidate_pred = np.argmax(proba_candidates, axis=1)
    local_candidate_support, local_baseline_support, local_purity = _local_neighbor_evidence(
        X_train,
        y_train_pos,
        X_test,
        candidate_pred,
        baseline_pred,
        n_classes,
        k=max(config.k, 16),
    )
    support_advantage = local_candidate_support - local_baseline_support
    p_values = _local_support_p_values(local_candidate_support, candidate_pred, train_prior, k=max(config.k, 16))
    candidate_conf = np.max(proba_candidates, axis=1)
    candidate_changed = candidate_pred != baseline_pred
    eligible = (
        attempted
        & candidate_changed
        & (candidate_conf >= baseline_conf - 0.02)
        & (support_advantage >= 0.02)
        & (p_values <= 0.45)
        & (local_purity >= 0.50)
    )
    pre_cap_eligible = eligible.copy()
    if np.any(eligible):
        max_changes = max(1, int(math.ceil(float(config.max_changed_fraction) * n_rows)))
        score = (
            1.50 * support_advantage
            + 0.50 * (candidate_conf - baseline_conf)
            - 0.35 * p_values
            - 0.15 * baseline_margin
        )
        keep = np.zeros_like(eligible, dtype=bool)
        idx = np.flatnonzero(eligible)
        keep[idx[np.argsort(score[idx])[-max_changes:]]] = True
        eligible = keep

    proba_adj = np.where(eligible[:, None], proba_candidates, baseline_proba)
    proba_adj /= np.maximum(np.sum(proba_adj, axis=1, keepdims=True), eps)
    changed = np.argmax(proba_adj, axis=1) != baseline_pred
    diagnostics = {
        "pseudo_label_attempted": attempted.astype(float),
        "pseudo_label_eligible": eligible.astype(float),
        "pseudo_label_pre_cap_eligible": pre_cap_eligible.astype(float),
        "pseudo_label_pool_size": pseudo_pool_size,
        "pseudo_label_support_advantage": support_advantage,
        "pseudo_label_local_p_value": p_values,
        "pseudo_label_local_purity": local_purity,
        "pseudo_label_candidate_confidence": candidate_conf,
        "no_shift_batch_prior_js": np.full(n_rows, js_value, dtype=np.float64),
        "no_shift_eligible_final": eligible.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_vote_fraction": eligible.astype(float),
        "no_shift_context_disagreement": np.zeros(n_rows, dtype=np.float64),
        "no_shift_local_support_margin": support_advantage,
        "no_shift_context_failure_fraction": np.zeros(n_rows, dtype=np.float64),
    }
    extra = {
        "pseudo_label_selected": "pseudo_label_context_augmentation",
        "pseudo_label_batch_prior_js": js_value,
        "pseudo_label_pool_fraction": float(np.mean(pseudo_mask)),
        "pseudo_label_attempted_fraction": float(np.mean(attempted)),
        "pseudo_label_pre_cap_eligible_fraction": float(np.mean(pre_cap_eligible)),
        "pseudo_label_changed_fraction": float(np.mean(changed)),
        "pseudo_label_mean_pool_size": float(np.mean(pseudo_pool_size[attempted])) if np.any(attempted) else 0.0,
        "pseudo_label_mean_support_advantage": float(np.mean(support_advantage[eligible])) if np.any(eligible) else 0.0,
        "pseudo_label_mean_p_value": float(np.mean(p_values[eligible])) if np.any(eligible) else 1.0,
        "pseudo_label_mean_candidate_confidence": (
            float(np.mean(candidate_conf[eligible])) if np.any(eligible) else 0.0
        ),
        "no_shift_batch_prior_js": js_value,
        "no_shift_eligible_fraction": float(np.mean(eligible)),
        "no_shift_changed_fraction": float(np.mean(changed)),
        "no_shift_mean_local_support_margin": float(np.mean(support_advantage[eligible])) if np.any(eligible) else 0.0,
        "no_shift_context_failure_fraction": 0.0,
    }
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=baseline_bundle.context_indices,
        effective_prior=baseline_bundle.effective_prior,
        gamma=eligible.astype(float),
        delta=support_advantage,
        runtime_sec=float(baseline_bundle.runtime_sec) + runtime_extra + (time.time() - start),
        diagnostics=diagnostics,
    )
    return bundle, extra


def prp_uncertainty_guarded_v42_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    """Filter V42 changes with multi-context PRP uncertainty diagnostics."""

    baseline_proba = np.asarray(baseline_bundle.proba_adj, dtype=np.float64)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    classes = np.asarray(baseline_bundle.classes)
    n_rows, n_classes = baseline_proba.shape
    if n_rows == 0 or n_classes < 2:
        return baseline_bundle, {"prp_uncertainty_selected": "baseline_empty_or_single_class"}

    v42_bundle, v42_details = scale_aware_boundary_bypass_combo_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
        seed,
    )
    v42_proba = np.asarray(v42_bundle.proba_adj, dtype=np.float64)
    v42_pred = np.argmax(v42_proba, axis=1)
    v42_changed = v42_pred != baseline_pred

    risk_bundle, risk_details = family_agreement_risk_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        replace(config, max_changed_fraction=1.0),
    )
    risk_diag = getattr(risk_bundle, "diagnostics", {})
    risk_score = np.asarray(risk_diag.get("no_shift_risk_score", np.ones(n_rows)), dtype=np.float64)
    context_disagreement = np.asarray(
        risk_diag.get("no_shift_context_disagreement", np.ones(n_rows)), dtype=np.float64
    )
    agreement_ratio = np.asarray(risk_diag.get("no_shift_family_agreement_ratio", np.zeros(n_rows)), dtype=np.float64)
    local_support = np.asarray(risk_diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=np.float64)
    local_purity = np.asarray(risk_diag.get("no_shift_local_purity", np.zeros(n_rows)), dtype=np.float64)
    risk_pred = np.argmax(np.asarray(risk_bundle.proba_adj, dtype=np.float64), axis=1)
    risk_changed = risk_pred != baseline_pred
    risk_agrees_with_v42 = risk_changed & (risk_pred == v42_pred)

    y_train_pos = _train_label_positions(y_train, classes)
    train_counts = np.bincount(y_train_pos[y_train_pos >= 0], minlength=n_classes).astype(np.float64)
    train_prior = train_counts / max(1.0, float(np.sum(train_counts)))
    local_candidate_support, local_baseline_support, local_purity_v42 = _local_neighbor_evidence(
        X_train,
        y_train_pos,
        X_test,
        v42_pred,
        baseline_pred,
        n_classes,
        k=max(config.k, 16),
    )
    v42_support_advantage = local_candidate_support - local_baseline_support
    p_values = _local_support_p_values(local_candidate_support, v42_pred, train_prior, k=max(config.k, 16))

    clear_low_risk = (
        (risk_score <= 0.48)
        & (context_disagreement <= 0.050)
        & (agreement_ratio >= 0.20)
        & (v42_support_advantage >= -0.020)
        & (p_values <= 0.72)
        & (local_purity_v42 >= 0.45)
    )
    diagnostic_rescue = (
        risk_agrees_with_v42
        & (risk_score <= 0.62)
        & (v42_support_advantage >= 0.020)
        & (p_values <= 0.55)
    )
    keep_v42_change = v42_changed & (clear_low_risk | diagnostic_rescue)
    rejected_v42_change = v42_changed & ~keep_v42_change
    proba_adj = np.where(keep_v42_change[:, None], v42_proba, baseline_proba)
    changed = np.argmax(proba_adj, axis=1) != baseline_pred

    diagnostics = {
        "prp_uncertainty_keep_v42_change": keep_v42_change.astype(float),
        "prp_uncertainty_rejected_v42_change": rejected_v42_change.astype(float),
        "prp_uncertainty_risk_score": risk_score,
        "prp_uncertainty_context_disagreement": context_disagreement,
        "prp_uncertainty_agreement_ratio": agreement_ratio,
        "prp_uncertainty_v42_support_advantage": v42_support_advantage,
        "prp_uncertainty_v42_local_p_value": p_values,
        "prp_uncertainty_v42_local_purity": local_purity_v42,
        "prp_uncertainty_risk_local_purity": local_purity,
        "no_shift_batch_prior_js": np.asarray(
            getattr(v42_bundle, "diagnostics", {}).get(
                "no_shift_batch_prior_js",
                risk_diag.get("no_shift_batch_prior_js", np.zeros(n_rows)),
            ),
            dtype=np.float64,
        ),
        "no_shift_eligible_final": keep_v42_change.astype(float),
        "no_shift_changed_fraction": np.full(n_rows, float(np.mean(changed))),
        "no_shift_vote_fraction": agreement_ratio,
        "no_shift_context_disagreement": context_disagreement,
        "no_shift_local_support_margin": v42_support_advantage,
        "no_shift_context_failure_fraction": np.asarray(
            risk_diag.get("no_shift_context_failure_fraction", np.zeros(n_rows)), dtype=np.float64
        ),
    }
    details = dict(v42_details)
    details.update({f"prp_diag_{key}": value for key, value in risk_details.items()})
    details.update(
        {
            "prp_uncertainty_selected": "v42_prp_uncertainty_guard",
            "prp_uncertainty_v42_changed_fraction": float(np.mean(v42_changed)),
            "prp_uncertainty_kept_fraction": float(np.mean(keep_v42_change)),
            "prp_uncertainty_rejected_fraction": float(np.mean(rejected_v42_change)),
            "prp_uncertainty_mean_risk_score_changed": (
                float(np.mean(risk_score[v42_changed])) if np.any(v42_changed) else 0.0
            ),
            "prp_uncertainty_mean_context_disagreement_changed": (
                float(np.mean(context_disagreement[v42_changed])) if np.any(v42_changed) else 0.0
            ),
            "prp_uncertainty_mean_support_advantage_kept": (
                float(np.mean(v42_support_advantage[keep_v42_change])) if np.any(keep_v42_change) else 0.0
            ),
            "no_shift_batch_prior_js": float(np.mean(diagnostics["no_shift_batch_prior_js"])),
            "no_shift_eligible_fraction": float(np.mean(keep_v42_change)),
            "no_shift_changed_fraction": float(np.mean(changed)),
            "no_shift_mean_context_disagreement": (
                float(np.mean(context_disagreement[v42_changed])) if np.any(v42_changed) else 0.0
            ),
            "no_shift_mean_local_support_margin": (
                float(np.mean(v42_support_advantage[keep_v42_change])) if np.any(keep_v42_change) else 0.0
            ),
            "no_shift_context_failure_fraction": float(np.mean(diagnostics["no_shift_context_failure_fraction"])),
        }
    )
    bundle = APAPrediction(
        proba_raw=baseline_proba,
        proba_adj=proba_adj,
        pred=classes[np.argmax(proba_adj, axis=1)],
        classes=classes,
        context_indices=v42_bundle.context_indices,
        effective_prior=v42_bundle.effective_prior,
        gamma=keep_v42_change.astype(float),
        delta=risk_score,
        runtime_sec=float(v42_bundle.runtime_sec) + float(risk_bundle.runtime_sec),
        diagnostics=diagnostics,
    )
    return bundle, details


def prior_matched_temperature_v42_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    """Route-level temperature calibration chosen by train-prior consistency."""

    v42_bundle, details = scale_aware_boundary_bypass_combo_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
        seed,
    )
    proba = np.asarray(v42_bundle.proba_adj, dtype=np.float64)
    classes = np.asarray(v42_bundle.classes)
    n_rows, n_classes = proba.shape
    if n_rows == 0 or n_classes < 2:
        details = dict(details)
        details.update({"temperature_v55_applied": False, "temperature_v55_selected": 1.0})
        return v42_bundle, details

    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    y_pos = np.asarray([class_to_pos[label] for label in y_train if label in class_to_pos], dtype=int)
    if len(y_pos) == 0:
        details = dict(details)
        details.update({"temperature_v55_applied": False, "temperature_v55_selected": 1.0})
        return v42_bundle, details
    train_prior = context_prior(y_pos, n_classes, alpha=1.0).reshape(1, -1)

    def prior_js(candidate: np.ndarray) -> float:
        batch_prior = np.mean(candidate, axis=0, keepdims=True)
        return float(js_divergence(batch_prior, train_prior)[0] / math.log(2.0))

    base_js = prior_js(proba)
    grid = [0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.80, 2.10]
    candidates: list[tuple[float, float, np.ndarray]] = []
    for temperature in grid:
        scaled = _temperature_scale_proba(proba, temperature)
        candidates.append((prior_js(scaled), temperature, scaled))
    best_js, best_temperature, best_proba = min(candidates, key=lambda item: (item[0], abs(item[1] - 1.0)))
    mean_conf_before = float(np.mean(np.max(proba, axis=1)))
    mean_conf_after = float(np.mean(np.max(best_proba, axis=1)))
    argmax_changed = np.argmax(best_proba, axis=1) != np.argmax(proba, axis=1)
    confidence_shift = abs(mean_conf_after - mean_conf_before)
    apply_temperature = (
        best_temperature != 1.0
        and best_js <= base_js - 1e-5
        and confidence_shift <= 0.12
        and not np.any(argmax_changed)
    )
    calibrated = best_proba if apply_temperature else proba
    diagnostics = dict(getattr(v42_bundle, "diagnostics", {}))
    diagnostics.update(
        {
            "temperature_v55_selected": np.full(n_rows, best_temperature if apply_temperature else 1.0),
            "temperature_v55_applied": np.full(n_rows, float(apply_temperature)),
        }
    )
    details = dict(details)
    details.update(
        {
            "temperature_v55_selected": best_temperature if apply_temperature else 1.0,
            "temperature_v55_candidate_best": best_temperature,
            "temperature_v55_applied": apply_temperature,
            "temperature_v55_prior_js_before": base_js,
            "temperature_v55_prior_js_after": prior_js(calibrated),
            "temperature_v55_best_prior_js": best_js,
            "temperature_v55_mean_confidence_before": mean_conf_before,
            "temperature_v55_mean_confidence_after": float(np.mean(np.max(calibrated, axis=1))),
            "temperature_v55_confidence_shift_candidate": confidence_shift,
            "temperature_v55_argmax_guard_fraction": float(np.mean(argmax_changed)),
        }
    )
    return replace(
        v42_bundle,
        proba_adj=calibrated,
        pred=classes[np.argmax(calibrated, axis=1)],
        diagnostics=diagnostics,
    ), details


def validation_member_selector_v42_fallback_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
) -> tuple[APAPrediction, dict[str, object]]:
    large_scale = len(y_train) >= 192 or len(X_test) >= 96
    if not large_scale:
        bundle, details = scale_aware_boundary_bypass_combo_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            config,
            seed,
        )
        details.update(
            {
                "validation_member_selected": "v42_small_scale_fallback",
                "validation_member_large_scale": large_scale,
                "validation_member_applied": False,
            }
        )
        return bundle, details

    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or len(y_train) < 80 or np.min(counts) < 8:
        bundle, details = scale_aware_boundary_bypass_combo_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            config,
            seed,
        )
        details.update(
            {
                "validation_member_selected": "v42_insufficient_train_fallback",
                "validation_member_large_scale": large_scale,
                "validation_member_applied": False,
            }
        )
        return bundle, details

    v27_config = replace(config, blend_weight=0.30)
    candidate_stats: dict[str, dict[str, list[float]]] = {
        "v27_boundary_union": {
            "accuracy": [],
            "balanced_accuracy": [],
            "macro_f1": [],
            "roc_auc": [],
            "ece": [],
            "changed_fraction": [],
        },
        "v40_instance_ensemble": {
            "accuracy": [],
            "balanced_accuracy": [],
            "macro_f1": [],
            "roc_auc": [],
            "ece": [],
            "changed_fraction": [],
        },
    }
    validation_runtime = 0.0
    split_count = 0
    split_failures = 0

    val_size = min(
        max(len(y_train) // 4, len(classes) * 8),
        max(len(y_train) - len(classes) * 4, len(classes) * 8),
    )
    for split_offset in (4501, 4502):
        try:
            X_sub, X_val, y_sub, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                stratify=y_train,
                random_state=seed + split_offset,
            )
        except ValueError:
            split_failures += 1
            continue

        val_baseline = baseline_predict_proba(X_sub, y_sub, X_val, estimator_factory)
        base_metrics = classification_metrics(y_val, val_baseline.proba_adj, val_baseline.classes)
        validation_runtime += float(val_baseline.runtime_sec)
        candidates = {
            "v27_boundary_union": guarded_boundary_union_v27_predict(
                X_sub,
                y_sub,
                X_val,
                val_baseline,
                estimator_factory,
                v27_config,
            ),
            "v40_instance_ensemble": instancewise_context_ensemble_predict(
                X_sub,
                y_sub,
                X_val,
                val_baseline,
                estimator_factory,
                config,
            ),
        }
        split_count += 1
        for name, (candidate_bundle, _candidate_details) in candidates.items():
            validation_runtime += float(candidate_bundle.runtime_sec)
            cand_metrics = classification_metrics(y_val, candidate_bundle.proba_adj, candidate_bundle.classes)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "ece"):
                base = base_metrics.get(metric)
                cand = cand_metrics.get(metric)
                if base in {"", None} or cand in {"", None}:
                    delta = 0.0
                else:
                    delta = float(cand) - float(base)
                candidate_stats[name][metric].append(delta)
            changed_fraction = float(
                np.mean(
                    np.argmax(candidate_bundle.proba_adj, axis=1)
                    != np.argmax(val_baseline.proba_adj, axis=1)
                )
            )
            candidate_stats[name]["changed_fraction"].append(changed_fraction)

    def _mean_stat(name: str, metric: str) -> float:
        values = candidate_stats[name][metric]
        return float(np.mean(values)) if values else 0.0

    def _min_stat(name: str, metric: str) -> float:
        values = candidate_stats[name][metric]
        return float(np.min(values)) if values else 0.0

    best_name = "v42_fallback_no_validation_gain"
    best_score = -math.inf
    for name in candidate_stats:
        mean_changed = _mean_stat(name, "changed_fraction")
        mean_acc = _mean_stat(name, "accuracy")
        mean_bacc = _mean_stat(name, "balanced_accuracy")
        mean_f1 = _mean_stat(name, "macro_f1")
        mean_auc = _mean_stat(name, "roc_auc")
        mean_ece = _mean_stat(name, "ece")
        valid = (
            split_count >= 2
            and mean_changed > 0.0
            and mean_acc > 1e-12
            and _min_stat(name, "accuracy") >= -1e-12
            and _min_stat(name, "balanced_accuracy") >= -0.004
            and _min_stat(name, "macro_f1") >= -0.004
            and _min_stat(name, "roc_auc") >= -0.002
        )
        if not valid:
            continue
        score = mean_acc + 0.35 * mean_bacc + 0.20 * mean_f1 + 0.10 * mean_auc - 0.05 * max(mean_ece, 0.0)
        if score > best_score:
            best_name = name
            best_score = score

    details: dict[str, object] = {
        "validation_member_selected": best_name,
        "validation_member_large_scale": large_scale,
        "validation_member_split_count": split_count,
        "validation_member_split_failures": split_failures,
        "validation_member_runtime_sec": validation_runtime,
        "validation_member_score": "" if best_score == -math.inf else best_score,
    }
    for name in candidate_stats:
        prefix = "validation_member_v27" if name == "v27_boundary_union" else "validation_member_v40"
        details.update(
            {
                f"{prefix}_accuracy_delta": _mean_stat(name, "accuracy"),
                f"{prefix}_min_accuracy_delta": _min_stat(name, "accuracy"),
                f"{prefix}_balanced_accuracy_delta": _mean_stat(name, "balanced_accuracy"),
                f"{prefix}_min_balanced_accuracy_delta": _min_stat(name, "balanced_accuracy"),
                f"{prefix}_macro_f1_delta": _mean_stat(name, "macro_f1"),
                f"{prefix}_min_macro_f1_delta": _min_stat(name, "macro_f1"),
                f"{prefix}_roc_auc_delta": _mean_stat(name, "roc_auc"),
                f"{prefix}_min_roc_auc_delta": _min_stat(name, "roc_auc"),
                f"{prefix}_ece_delta": _mean_stat(name, "ece"),
                f"{prefix}_changed_fraction": _mean_stat(name, "changed_fraction"),
            }
        )

    if best_name == "v27_boundary_union":
        final_bundle, final_details = guarded_boundary_union_v27_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            v27_config,
        )
        details.update(final_details)
        details.update(
            {
                "validation_member_applied": True,
                "scale_combo_branch": "validation_member_v27_boundary_union",
            }
        )
        return replace(final_bundle, runtime_sec=float(final_bundle.runtime_sec) + validation_runtime), details

    if best_name == "v40_instance_ensemble":
        final_bundle, final_details = instancewise_context_ensemble_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            config,
        )
        details.update(final_details)
        details.update(
            {
                "validation_member_applied": True,
                "scale_combo_branch": "validation_member_v40_instance_ensemble",
            }
        )
        return replace(final_bundle, runtime_sec=float(final_bundle.runtime_sec) + validation_runtime), details

    fallback_bundle, fallback_details = scale_aware_boundary_bypass_combo_predict(
        X_train,
        y_train,
        X_test,
        baseline_bundle,
        estimator_factory,
        config,
        seed,
    )
    details.update(fallback_details)
    details.update(
        {
            "validation_member_applied": False,
            "scale_combo_branch": f"validation_member_{fallback_details.get('scale_combo_branch', 'v42_fallback')}",
        }
    )
    return replace(fallback_bundle, runtime_sec=float(fallback_bundle.runtime_sec) + validation_runtime), details


def v14_priority_label_shift_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    k: int,
    min_context_disagreement: float = 0.012,
) -> tuple[APAPrediction, dict[str, object]]:
    cf_config = CounterfactualContextConfig(
        k=max(8, min(k, 24)),
        lambda_balance=0.35,
        alpha=0.75,
        prior_floor=0.08,
        evidence_temperature=1.10,
        blend_weight=0.60,
        sensitivity_gate=0.12,
        max_abs_sensitivity=0.90,
        max_abs_sensitivity_low_batch_margin_only=True,
        disagreement_gate=0.004,
        max_disagreement=0.040,
        max_disagreement_low_batch_margin_only=True,
        max_baseline_margin=0.55,
        max_baseline_confidence=0.92,
        min_batch_prior_js=0.008,
        min_batch_pred_margin=0.05,
        batch_sensitivity_rescue_gate=0.0,
        min_train_prior_imbalance=0.10,
        directional_shift_gate=True,
        max_changed_fraction=0.10,
        off_majority_max_baseline_margin=0.15,
        aggregation="stability_debias",
    )
    cf_bundle = counterfactual_context_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory,
        cf_config,
        baseline_proba=baseline_bundle.proba_adj,
        baseline_classes=baseline_bundle.classes,
    )
    cf_changed_fraction = float(
        np.mean(
            np.argmax(cf_bundle.proba_adj, axis=1)
            != np.argmax(baseline_bundle.proba_adj, axis=1)
        )
    )
    cf_mean_context_disagreement = float(np.mean(cf_bundle.delta))
    rejected_low_disagreement = (
        cf_changed_fraction > 0.0
        and cf_mean_context_disagreement < min_context_disagreement
    )
    accepted = cf_changed_fraction > 0.0 and not rejected_low_disagreement
    details = {
        "cf_batch_prior_js": batch_prior_js(
            baseline_bundle.proba_adj,
            y_train,
            baseline_bundle.classes,
            alpha=cf_config.alpha,
        ),
        "cf_mean_prior_sensitivity": float(np.mean(cf_bundle.gamma)),
        "cf_mean_abs_prior_sensitivity": float(np.mean(np.abs(cf_bundle.gamma))),
        "cf_mean_context_disagreement": cf_mean_context_disagreement,
        "cf_changed_fraction": cf_changed_fraction,
        "dual_mode_label_shift_context_guard_threshold": min_context_disagreement,
        "dual_mode_label_shift_context_guard_rejected": rejected_low_disagreement,
        "mixed_v43_shift_accepted": accepted,
    }
    if accepted:
        return cf_bundle, details
    return replace(
        baseline_bundle,
        runtime_sec=float(baseline_bundle.runtime_sec) + float(cf_bundle.runtime_sec),
    ), details


def mixed_v38_v42_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle: APAPrediction,
    estimator_factory,
    config: NoShiftFeatureConfig,
    seed: int,
    k: int,
    no_shift_member: str = "v42",
) -> tuple[APAPrediction, dict[str, object]]:
    js_value = batch_prior_js(
        baseline_bundle.proba_adj,
        y_train,
        baseline_bundle.classes,
        alpha=1.0,
    )
    shift_js_threshold = min(config.max_batch_prior_js, 0.004)
    shift_route = js_value > shift_js_threshold
    large_scale = len(y_train) >= 192 or len(X_test) >= 96
    no_shift_route = (not shift_route) and large_scale
    if not no_shift_route:
        bundle, details = v14_priority_label_shift_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            k=k,
            min_context_disagreement=0.012,
        )
        details.update(
            {
                "mixed_v43_branch": (
                    "label_shift_v38_priority"
                    if bool(details.get("mixed_v43_shift_accepted", False))
                    else "baseline_shift_guard"
                ),
                "mixed_router_branch": (
                    "label_shift_v38_priority"
                    if bool(details.get("mixed_v43_shift_accepted", False))
                    else "baseline_shift_guard"
                ),
                "mixed_router_no_shift_member": no_shift_member,
                "mixed_v43_batch_prior_js": js_value,
                "mixed_v43_shift_route": shift_route,
                "mixed_v43_no_shift_route": no_shift_route,
                "mixed_v43_large_scale_gate": large_scale,
                "mixed_v43_shift_js_threshold": shift_js_threshold,
            }
        )
        return bundle, details

    if no_shift_member == "v50":
        bundle, details = cross_model_tree_consensus_rescue_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            config,
            seed,
        )
        branch_name = "no_shift_v50_cross_model_tree_rescue"
    else:
        bundle, details = scale_aware_boundary_bypass_combo_predict(
            X_train,
            y_train,
            X_test,
            baseline_bundle,
            estimator_factory,
            config,
            seed,
        )
        branch_name = "no_shift_v42_boundary_bypass_combo"
    details.update(
        {
            "mixed_v43_branch": branch_name,
            "mixed_router_branch": branch_name,
            "mixed_router_no_shift_member": no_shift_member,
            "mixed_v43_batch_prior_js": js_value,
            "mixed_v43_shift_route": shift_route,
            "mixed_v43_no_shift_route": no_shift_route,
            "mixed_v43_large_scale_gate": large_scale,
            "mixed_v43_shift_js_threshold": shift_js_threshold,
        }
    )
    return bundle, details


def method_configs(k: int) -> dict[str, MethodRun]:
    return {
        "tabicl_baseline": MethodRun(None, "baseline", "baseline", "Original TabICL full-context inference."),
        "apa_knn_context": MethodRun(
            APAConfig(context_strategy="knn", correction="none", k=k),
            "rerun_context",
            "apa_v0",
            "Rerun TabICL per query with KNN context.",
        ),
        "apa_balanced_knn": MethodRun(
            APAConfig(context_strategy="balanced_knn", correction="none", k=k),
            "rerun_context",
            "apa_v0",
            "Rerun TabICL per query with class-balanced KNN context.",
        ),
        "apa_prior_aware_knn": MethodRun(
            APAConfig(context_strategy="prior_aware_knn", correction="none", k=k, lambda_balance=0.25),
            "rerun_context",
            "apa_v0",
            "Rerun TabICL per query with prior-aware KNN context.",
        ),
        "apa_local_correction": MethodRun(
            APAConfig(context_strategy="knn", correction="local", k=k, alpha=1.0),
            "rerun_context",
            "apa_v0",
            "Rerun KNN-context TabICL plus local DistPFN-style correction.",
        ),
        "apa_prior_aware_adaptive": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.25,
                alpha=1.0,
                query_mode="hybrid",
                eta=0.5,
                gamma_min=0.0,
                gamma_max=1.0,
            ),
            "rerun_context",
            "apa_v0",
            "Rerun prior-aware context plus adaptive posterior correction.",
        ),
        "apa_prior_aware_adaptive_soft": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.15,
                alpha=1.0,
                query_mode="hybrid",
                eta=0.35,
                gamma_min=0.0,
                gamma_max=0.6,
            ),
            "rerun_context",
            "apa_v0",
            "Softer rerun prior-aware adaptive correction.",
        ),
        "apa_full_prior_knn_adaptive": MethodRun(
            APAConfig(
                context_strategy="knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.0,
                alpha=1.0,
                query_mode="hybrid",
                eta=0.5,
                gamma_min=0.0,
                gamma_max=0.5,
            ),
            "posthoc_full",
            "apa_v1",
            "Keep original full-context TabICL probabilities; correct with KNN local prior.",
        ),
        "apa_full_prior_aware_adaptive": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.20,
                alpha=1.0,
                query_mode="hybrid",
                eta=0.45,
                gamma_min=0.0,
                gamma_max=0.5,
            ),
            "posthoc_full",
            "apa_v1",
            "Keep original full-context TabICL probabilities; correct with prior-aware local prior.",
        ),
        "apa_full_prior_aware_tiny": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.10,
                alpha=2.0,
                query_mode="batch",
                eta=0.0,
                gamma_min=0.0,
                gamma_max=0.25,
            ),
            "posthoc_full",
            "apa_v1",
            "Very conservative full-context correction using batch predicted distribution.",
        ),
        "apa_shift_margin_gate": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.20,
                alpha=1.0,
                query_mode="batch",
                eta=0.0,
                gamma_min=0.0,
                gamma_max=1.0,
                min_batch_prior_js=0.005,
                max_baseline_margin=0.28,
                max_baseline_confidence=0.72,
                blend_weight=1.0,
            ),
            "posthoc_full",
            "apa_v2",
            "Shift-aware margin gate: correct only when batch prior shift is visible and the baseline row is uncertain.",
        ),
        "apa_shift_margin_gate_soft": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.15,
                alpha=2.0,
                query_mode="batch",
                eta=0.0,
                gamma_min=0.0,
                gamma_max=0.7,
                min_batch_prior_js=0.003,
                max_baseline_margin=0.35,
                max_baseline_confidence=0.80,
                blend_weight=0.7,
            ),
            "posthoc_full",
            "apa_v2",
            "Softer shift-aware margin gate with smoothed local prior and blended correction.",
        ),
        "apa_validation_selector": MethodRun(
            None,
            "validation_selector",
            "apa_v3",
            "Train-internal validation selector over conservative APA post-hoc candidates.",
        ),
        "apa_aggressive_margin_push": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.30,
                alpha=0.5,
                query_mode="hybrid",
                eta=0.65,
                gamma_min=0.05,
                gamma_max=1.4,
                min_batch_prior_js=0.001,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.90,
                blend_weight=1.0,
            ),
            "posthoc_full",
            "apa_v4",
            "Aggressive but margin-gated prior-aware posterior push for hard shift cases.",
        ),
        "apa_aggressive_batch_push": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.25,
                alpha=0.75,
                query_mode="batch",
                eta=0.0,
                gamma_min=0.05,
                gamma_max=1.2,
                min_batch_prior_js=0.001,
                max_baseline_margin=0.60,
                max_baseline_confidence=0.92,
                blend_weight=1.0,
            ),
            "posthoc_full",
            "apa_v4",
            "Aggressive batch-prior correction gated to low-margin rows.",
        ),
        "apa_hard_shift_router": MethodRun(
            APAConfig(
                context_strategy="prior_aware_knn",
                correction="adaptive",
                k=k,
                lambda_balance=0.30,
                alpha=0.5,
                query_mode="hybrid",
                eta=0.65,
                gamma_min=0.05,
                gamma_max=1.4,
                min_batch_prior_js=0.004,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.90,
                blend_weight=1.0,
                binary_only=True,
            ),
            "posthoc_full",
            "apa_v5",
            "Hard-shift router: aggressive margin push only when batch prior shift exceeds no-shift diagnostics.",
        ),
        "apa_dual_shift_router": MethodRun(
            None,
            "dual_shift_router",
            "apa_v6",
            "Dual shift router: high batch-prior JS uses conservative v2; lower hard-shift JS uses binary v5.",
        ),
        "apa_dual_shift_router_v7": MethodRun(
            None,
            "dual_shift_router_v7",
            "apa_v7",
            "Safer dual shift router: raises hard-branch batch-prior JS threshold after OpenML stress.",
        ),
        "apa_cf_entropy_ensemble": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.30,
                alpha=0.75,
                blend_weight=0.70,
                sensitivity_gate=0.0,
                disagreement_gate=0.0,
                max_baseline_confidence=1.0,
                aggregation="entropy_ensemble",
            ),
            "counterfactual_context",
            "apa_v8",
            "Counterfactual multi-context ensemble weighted by context prediction entropy.",
        ),
        "apa_cf_prior_sensitivity": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.15,
                blend_weight=0.75,
                sensitivity_gate=0.0,
                disagreement_gate=0.0,
                max_baseline_confidence=1.0,
                aggregation="prior_sensitivity",
            ),
            "counterfactual_context",
            "apa_v8",
            "Counterfactual prior intervention: regress out context-prior sensitivity from multi-context outputs.",
        ),
        "apa_cf_stability_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                disagreement_gate=0.004,
                max_baseline_confidence=0.92,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v8",
            "Counterfactual context-prior intervention routed only for prior-sensitive unstable rows.",
        ),
        "apa_cf_directional_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                disagreement_gate=0.004,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.15,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v8",
            "Directional counterfactual router: prior-sensitive intervention may only flip toward batch-supported prior deficits.",
        ),
        "apa_cf_strict_margin_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                disagreement_gate=0.004,
                max_baseline_margin=0.15,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.03125,
                off_majority_max_baseline_margin=0.0,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v9",
            "Strict counterfactual router: only one very low-margin prior-sensitive flip per batch.",
        ),
        "apa_cf_batch_margin_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                disagreement_gate=0.004,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_batch_pred_margin=0.05,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.0,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v10",
            "Batch-margin counterfactual router: require a clear predicted batch majority before intervention.",
        ),
        "apa_cf_majority_guard_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                disagreement_gate=0.004,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.0,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v11",
            "Majority-guard counterfactual router: recover opportunity by dropping the hard batch-margin gate while still requiring batch-majority flips.",
        ),
        "apa_cf_sensitivity_rescue_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                disagreement_gate=0.004,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_batch_pred_margin=0.05,
                batch_sensitivity_rescue_gate=0.70,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.0,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v12",
            "Sensitivity-rescue counterfactual router: keep the batch-margin guard unless the batch is highly context-prior sensitive.",
        ),
        "apa_cf_tempered_rescue_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                max_abs_sensitivity=0.90,
                disagreement_gate=0.004,
                max_disagreement=0.040,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.15,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v13",
            "Tempered-rescue counterfactual router: recover low-margin off-majority flips while rejecting over-sensitive context-prior flips.",
        ),
        "apa_cf_hybrid_tempered_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                max_abs_sensitivity=0.90,
                max_abs_sensitivity_low_batch_margin_only=True,
                disagreement_gate=0.004,
                max_disagreement=0.040,
                max_disagreement_low_batch_margin_only=True,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_batch_pred_margin=0.05,
                batch_sensitivity_rescue_gate=0.0,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.15,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v14",
            "Hybrid tempered router: use the batch-margin route when batch evidence is clear, otherwise use row-level tempered rescue.",
        ),
        "apa_cf_hybrid_calibrated_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.50,
                sensitivity_gate=0.12,
                max_abs_sensitivity=0.90,
                max_abs_sensitivity_low_batch_margin_only=True,
                disagreement_gate=0.004,
                max_disagreement=0.040,
                max_disagreement_low_batch_margin_only=True,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_batch_pred_margin=0.05,
                batch_sensitivity_rescue_gate=0.0,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.15,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v15",
            "Calibrated hybrid router: same hybrid context intervention as v14 with a softer final blend for accuracy gains without ECE drift.",
        ),
        "apa_cf_hybrid_stability_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.60,
                sensitivity_gate=0.12,
                max_abs_sensitivity=0.90,
                max_abs_sensitivity_low_batch_margin_only=True,
                disagreement_gate=0.004,
                max_disagreement=0.060,
                max_disagreement_low_batch_margin_only=False,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.92,
                min_batch_prior_js=0.008,
                min_batch_pred_margin=0.05,
                batch_sensitivity_rescue_gate=0.0,
                min_train_prior_imbalance=0.10,
                directional_shift_gate=True,
                max_changed_fraction=0.10,
                off_majority_max_baseline_margin=0.15,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v16",
            "Hybrid stability router: v14 plus a global context-disagreement cap to avoid unstable clear-batch flips.",
        ),
        "apa_cf_natural_guard_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.45,
                sensitivity_gate=0.14,
                max_abs_sensitivity=0.85,
                disagreement_gate=0.004,
                max_disagreement=0.030,
                max_baseline_margin=0.30,
                max_baseline_confidence=0.88,
                min_batch_prior_js=0.012,
                min_batch_pred_margin=0.02,
                directional_shift_gate=True,
                max_changed_fraction=0.03125,
                off_majority_max_baseline_margin=0.04,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v17",
            "Natural-data guard router: a conservative v8 variant for ordered public-data splits with stricter margin, sensitivity, disagreement, and one-flip-per-batch caps.",
        ),
        "apa_cf_local_shift_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.50,
                sensitivity_gate=0.10,
                max_abs_sensitivity=1.10,
                disagreement_gate=0.003,
                max_disagreement=0.055,
                max_baseline_margin=0.30,
                max_baseline_confidence=0.88,
                min_local_prior_js=0.030,
                local_directional_shift_gate=True,
                max_changed_fraction=0.0625,
                off_majority_max_baseline_margin=0.04,
                aggregation="stability_debias",
            ),
            "counterfactual_context",
            "apa_v18",
            "Local-shift router: use each test point's KNN-context prior shift as the natural-data intervention signal instead of requiring an artificial or model-level batch shift.",
        ),
        "apa_cf_local_shift_val_router": MethodRun(
            CounterfactualContextConfig(
                k=max(8, min(k, 24)),
                lambda_balance=0.35,
                alpha=0.75,
                prior_floor=0.08,
                evidence_temperature=1.10,
                blend_weight=0.50,
                sensitivity_gate=0.08,
                max_abs_sensitivity=1.35,
                disagreement_gate=0.002,
                max_disagreement=0.080,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.95,
                min_local_prior_js=0.015,
                local_directional_shift_gate=True,
                max_changed_fraction=0.078125,
                off_majority_max_baseline_margin=0.08,
                aggregation="stability_debias",
            ),
            "cf_validation_router",
            "apa_v19",
            "Validation-gated local-shift router: first tests the local-prior counterfactual intervention on a train-only validation tail, and applies it to test only if validation accuracy improves without metric harm.",
        ),
        "apa_context_policy_val_selector": MethodRun(
            None,
            "context_policy_selector",
            "apa_v20",
            "Train-only context-policy selector: choose full, KNN, balanced KNN, or prior-aware KNN context using an internal validation tail, then rerun TabICL with the selected context policy on test.",
        ),
        "apa_cf_no_shift_diverse_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_feature_context",
            "apa_v21",
            "No-shift diverse feature-evidence router: diverse KNN, prototypes, and feature-subspace neighbors with sample-level gates.",
        ),
        "apa_cf_no_shift_clean_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                tau_clean=0.70,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.24,
                max_baseline_confidence=0.84,
                max_context_disagreement=0.030,
                min_context_vote_fraction=0.80,
                local_evidence_margin=0.050,
                blend_weight=0.22,
                max_changed_fraction=0.015625,
                families=("clean_knn", "prototype", "diverse_knn"),
            ),
            "no_shift_feature_context",
            "apa_v21",
            "No-shift clean feature-evidence router: filters locally inconsistent neighbors before small gated prediction blends.",
        ),
        "apa_cf_no_shift_boundary_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.24,
                max_baseline_confidence=0.84,
                max_context_disagreement=0.030,
                min_context_vote_fraction=0.80,
                local_evidence_margin=0.050,
                blend_weight=0.22,
                max_changed_fraction=0.015625,
                families=("boundary_contrast", "clean_knn", "diverse_knn"),
            ),
            "no_shift_feature_context",
            "apa_v21",
            "No-shift boundary feature-evidence router: contrasts top-1 and top-2 predicted classes for low-margin rows.",
        ),
        "apa_cf_no_shift_soft_diverse_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.35,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.38,
                max_baseline_confidence=0.92,
                max_context_disagreement=0.065,
                min_context_vote_fraction=0.67,
                local_evidence_margin=-0.010,
                blend_weight=0.22,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_feature_context",
            "apa_v22",
            "Soft no-shift diverse feature-evidence router: a lower-blend diverse/prototype/subspace variant to improve trigger rate while keeping sample-level gates.",
        ),
        "apa_cf_no_shift_union_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_union_context",
            "apa_v23",
            "Union no-shift feature-evidence router: combine strict diverse and soft-diverse candidates, keeping only gated sample-level changes.",
        ),
        "apa_cf_no_shift_boundary_union_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_boundary_union_context",
            "apa_v24",
            "Boundary-union no-shift feature-evidence router: combine strict diverse, soft diverse, and stricter boundary-contrast candidates.",
        ),
        "apa_cf_no_shift_guarded_boundary_union_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_boundary_union_context",
            "apa_v27",
            "Guarded boundary-union router: v24 plus imbalance and multi-evidence guards before accepting boundary-contrast changes.",
        ),
        "apa_cf_no_shift_minority_safe_boundary_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_boundary_union_context",
            "apa_v32",
            "Minority-safe guarded boundary-union router: V27 plus a hard guard against baseline-minority to candidate-majority flips.",
        ),
        "apa_cf_no_shift_pvalue_guarded_boundary_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_boundary_union_context",
            "apa_v33",
            "P-value guarded boundary-union router: V27 plus a local binomial evidence guard and softer minority-risk handling for candidate flips.",
        ),
        "apa_cf_no_shift_validation_boundary_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_validation_boundary_context",
            "apa_v39",
            "Validation-gated no-shift boundary router: use V27 directly on small batches, but require train-internal validation gains before applying V27 on larger no-shift batches.",
        ),
        "apa_cf_no_shift_instance_ensemble_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_instance_ensemble_context",
            "apa_v40",
            "Instance-wise context ensemble: dynamically weight strict, soft, boundary, and clean no-shift context members per test sample using vote, support, disagreement, and uncertainty signals.",
        ),
        "apa_cf_no_shift_diversity_greedy_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.34,
                max_baseline_confidence=0.91,
                max_context_disagreement=0.055,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.000,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "clean_knn", "boundary_contrast", "feature_subspace_knn"),
            ),
            "no_shift_diversity_greedy_context",
            "apa_v46",
            "Diversity-greedy instance router: per sample, greedily select a small complementary subset of structured context members and blend only low-risk consensus changes.",
        ),
        "apa_cf_no_shift_diversity_pvalue_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.34,
                max_baseline_confidence=0.91,
                max_context_disagreement=0.055,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.000,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "clean_knn", "boundary_contrast", "feature_subspace_knn"),
            ),
            "no_shift_diversity_greedy_pvalue_context",
            "apa_v47",
            "Diversity-greedy p-value router: V46 plus a local-neighborhood p-value rescue for otherwise rejected minority-risk boundary changes.",
        ),
        "apa_cf_no_shift_scale_combo_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_scale_combo_context",
            "apa_v41",
            "Scale-aware no-shift combo: small batches use V27; large batches use V39 validation gating, then a stable V40 instance-wise ensemble fallback when member disagreement is low.",
        ),
        "apa_cf_no_shift_boundary_bypass_combo_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_boundary_bypass_combo_context",
            "apa_v42",
            "Scale-aware boundary-bypass combo: V41 plus a strict low-JS/high-context V27 bypass for large no-shift boundary cases missed by validation.",
        ),
        "apa_cf_no_shift_boundary_diversity_selector_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_boundary_diversity_selector_context",
            "apa_v48",
            "Precision-first boundary-diversity selector: keep the high-precision V42 boundary member, and use the V47 diversity/p-value member only when V42 abstains.",
        ),
        "apa_cf_no_shift_precision_rescue_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_precision_rescue_context",
            "apa_v49",
            "Precision-first p-value rescue router: preserve V42 high-precision edits, then use a permissive diversity/p-value member only for V42 abstain samples.",
        ),
        "apa_cf_no_shift_cross_model_tree_rescue_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_cross_model_tree_rescue_context",
            "apa_v50",
            "Cross-model tree rescue router: keep V42 as the primary TabICL member, then let ExtraTrees/RandomForest consensus rescue only V42-abstain samples with local p-value support.",
        ),
        "apa_cf_no_shift_calibrated_tree_rescue_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_calibrated_tree_rescue_context",
            "apa_v51",
            "Calibration-aware tree rescue router: V50 plus confidence calibration on accepted tree-rescue samples, preserving the V42 primary member.",
        ),
        "apa_cf_no_shift_prp_tree_rescue_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_prp_tree_rescue_context",
            "apa_v56",
            "PRP-aware tree rescue router: keep V42 as primary, use ExtraTrees/RandomForest consensus as candidate rescue, and gate/expand rescues with multi-context PRP reliability diagnostics.",
        ),
        "apa_cf_no_shift_fast_tree_rescue_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_fast_tree_rescue_context",
            "apa_v52",
            "Fast-primary tree rescue router: use the cheaper V27 guarded-boundary member as primary, then apply the same ExtraTrees/RandomForest consensus rescue used by V50.",
        ),
        "apa_cf_no_shift_pseudo_label_context_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_pseudo_label_context",
            "apa_v53",
            "Pseudo-label context augmentation: add high-confidence test-batch predictions as label-side context for low-margin rows, excluding each row's own pseudo-label and guarding changes with local p-values.",
        ),
        "apa_cf_no_shift_prp_uncertainty_guard_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_prp_uncertainty_guard_context",
            "apa_v54",
            "PRP uncertainty guarded V42: compute multi-context prediction-risk diagnostics and keep V42 edits only when uncertainty, disagreement, local p-value, and support gates agree.",
        ),
        "apa_cf_no_shift_prior_temperature_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_prior_temperature_context",
            "apa_v55",
            "Prior-matched temperature calibration for V42: choose a route-level temperature by matching batch predicted prior to train prior, preserving hard predictions.",
        ),
        "apa_cf_no_shift_context_consensus_calibrated_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_context_consensus_calibrated_context",
            "apa_v44",
            "Context-consensus calibrated V42: preserve V42 hard predictions, but shrink changed-sample probabilities toward baseline according to context disagreement, uncertainty, and margin.",
        ),
        "apa_cf_no_shift_validation_member_selector_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_validation_member_selector_context",
            "apa_v45",
            "Multi-split validation member selector: choose V27 boundary or V40 instance ensemble only when train-internal validation is consistently positive; otherwise fall back to V42.",
        ),
        "apa_cf_mixed_v38_v42_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "mixed_v38_v42_context",
            "apa_v43",
            "Mixed V43 router: route low batch-prior JS batches to V42 no-shift boundary-bypass combo; ambiguous/high-JS batches use V38 V14-priority label-shift correction or baseline.",
        ),
        "apa_cf_mixed_v38_v50_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.24,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "mixed_v38_v50_context",
            "apa_v57",
            "Mixed V57 router: route large low-JS no-shift batches to V50 sparse cross-model tree rescue; ambiguous/high-JS batches use V38 V14-priority label-shift correction or baseline.",
        ),
        "apa_cf_no_shift_family_agreement_risk_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.38,
                max_baseline_confidence=0.92,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("boundary_contrast", "diverse_knn", "prototype", "clean_knn", "minority_contrast"),
            ),
            "no_shift_family_agreement_risk_context",
            "apa_v28",
            "Family-agreement no-shift router: require multiple feature-evidence context families plus minority-risk and selective risk guards before changing a baseline decision.",
        ),
        "apa_cf_no_shift_negative_control_stability_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "no_shift_negative_control_stability_context",
            "apa_v31",
            "Negative-control stability router: start from V27 guarded boundary-union candidates and keep high-risk flips only if perturbed contexts repeatedly support the same class.",
        ),
        "apa_cf_no_shift_minority_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.30,
                tau_clean=0.65,
                max_batch_prior_js=0.020,
                max_baseline_margin=0.75,
                max_baseline_confidence=0.98,
                max_context_disagreement=0.075,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.020,
                blend_weight=0.26,
                max_changed_fraction=0.015625,
                max_minority_pred_train_ratio=0.70,
                require_candidate_minority=True,
                families=("minority_contrast", "prototype", "clean_knn"),
            ),
            "no_shift_feature_context",
            "apa_v25",
            "Minority-aware no-shift feature-evidence router: use nearest minority anchors plus baseline-class contrast to rescue collapsed minority predictions.",
        ),
        "apa_cf_no_shift_minority_density_router": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.30,
                tau_clean=0.65,
                max_batch_prior_js=0.020,
                max_baseline_margin=0.90,
                max_baseline_confidence=0.99,
                max_context_disagreement=0.080,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.000,
                blend_weight=0.30,
                max_changed_fraction=0.015625,
                require_candidate_minority=True,
                min_minority_distance_advantage=0.020,
                minority_distance_k=5,
                families=("minority_contrast",),
            ),
            "no_shift_feature_context",
            "apa_v26",
            "Sample-level minority-density router: use minority contrast only when minority anchors are closer than the baseline class in feature space.",
        ),
        "apa_cf_dual_mode_router_v17": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                tau_clean=0.70,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "dual_mode_no_shift_cf",
            "apa_v21",
            "Dual-mode router: feature-evidence contexts for weak-shift batches, otherwise fall back to the v16 counterfactual prior-intervention router.",
        ),
        "apa_cf_dual_mode_v34": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "dual_mode_v27_v14",
            "apa_v34",
            "Dual-mode V34: use V27 guarded boundary-union for no-shift batches and V14 hybrid-tempered counterfactual intervention for visible label-shift batches.",
        ),
        "apa_cf_dual_mode_v35": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "dual_mode_v27_v14",
            "apa_v35",
            "Dual-mode V35: V34 plus a conservative small-batch no-shift evidence guard that reverts thin-support V27 edits to baseline.",
        ),
        "apa_cf_dual_mode_v36": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "dual_mode_v27_v14",
            "apa_v36",
            "Dual-mode V36: V35 plus a context-disagreement guard for label-shift edits, reverting low-disagreement prior interventions to baseline.",
        ),
        "apa_cf_dual_mode_v37": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "dual_mode_v14_priority_v27",
            "apa_v37",
            "Dual-mode V37: prioritize the V14 label-shift router, then use V27 only as a very strict low-JS fallback.",
        ),
        "apa_cf_dual_mode_v38": MethodRun(
            NoShiftFeatureConfig(
                k=max(8, min(k, 20)),
                candidate_multiplier=5,
                lambda_div=0.40,
                tau_clean=0.65,
                max_batch_prior_js=0.012,
                max_baseline_margin=0.32,
                max_baseline_confidence=0.90,
                max_context_disagreement=0.050,
                min_context_vote_fraction=0.67,
                local_evidence_margin=0.015,
                blend_weight=0.30,
                max_changed_fraction=0.03125,
                families=("diverse_knn", "prototype", "feature_subspace_knn"),
            ),
            "dual_mode_v14_priority_v27",
            "apa_v38",
            "Dual-mode V38: V37 with a relaxed 0.012 label-shift disagreement guard to recover low-disagreement V14 gains.",
        ),
    }


def score_for_selection(metrics: dict[str, object]) -> float:
    return (
        float(metrics["balanced_accuracy"])
        + 0.20 * float(metrics["macro_f1"])
        + 0.10 * float(metrics["accuracy"])
        - 0.03 * float(metrics["ece"])
    )


def validation_select_and_correct(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle,
    estimator_factory,
    k: int,
    seed: int,
) -> tuple[object, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 3:
        return baseline_bundle, {"validation_selected": "baseline_insufficient_classes"}

    val_size = min(max(len(y_train) // 4, len(classes) * 2), max(len(y_train) - len(classes), len(classes) * 2))
    try:
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=seed + 1009,
        )
    except ValueError:
        return baseline_bundle, {"validation_selected": "baseline_split_failed"}

    val_baseline = baseline_predict_proba(X_sub, y_sub, X_val, estimator_factory)
    base_metrics = classification_metrics(y_val, val_baseline.proba_adj, val_baseline.classes)
    best_name = "baseline"
    best_score = score_for_selection(base_metrics)
    best_metrics = base_metrics
    candidate_configs = validation_candidate_configs(k)

    for name, candidate in candidate_configs.items():
        corrected = correct_existing_proba_with_context_prior(
            val_baseline.proba_adj,
            X_sub,
            y_sub,
            X_val,
            val_baseline.classes,
            candidate,
        )
        metrics = classification_metrics(y_val, corrected.proba_adj, corrected.classes)
        score = score_for_selection(metrics)
        no_harm = (
            float(metrics["accuracy"]) >= float(base_metrics["accuracy"]) - 0.015
            and float(metrics["balanced_accuracy"]) >= float(base_metrics["balanced_accuracy"]) - 0.010
            and float(metrics["macro_f1"]) >= float(base_metrics["macro_f1"]) - 0.010
        )
        if no_harm and score > best_score:
            best_name = name
            best_score = score
            best_metrics = metrics

    min_gain = 0.001
    base_score = score_for_selection(base_metrics)
    details: dict[str, object] = {
        "validation_selected": best_name,
        "validation_base_score": base_score,
        "validation_selected_score": best_score,
        "validation_score_gain": best_score - base_score,
        "validation_base_balanced_accuracy": base_metrics["balanced_accuracy"],
        "validation_selected_balanced_accuracy": best_metrics["balanced_accuracy"],
    }
    if best_name == "baseline" or best_score < base_score + min_gain:
        details["validation_selected"] = "baseline_no_gain"
        return baseline_bundle, details

    final_config = candidate_configs[best_name]
    final_bundle = correct_existing_proba_with_context_prior(
        baseline_bundle.proba_adj,
        X_train,
        y_train,
        X_test,
        baseline_bundle.classes,
        final_config,
    )
    return final_bundle, details


def cf_validation_select_and_correct(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle,
    estimator_factory,
    config: CounterfactualContextConfig,
    seed: int,
) -> tuple[object, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or len(y_train) < 32 or np.min(counts) < 3:
        return baseline_bundle, {"cf_validation_selected": "baseline_insufficient_train"}

    val_size = min(max(len(y_train) // 4, len(classes) * 4), max(len(y_train) - len(classes) * 2, len(classes) * 4))
    train_n = len(y_train) - val_size
    use_ordered_tail = train_n >= len(classes) * 2
    if use_ordered_tail:
        X_sub, X_val = X_train[:train_n], X_train[train_n:]
        y_sub, y_val = y_train[:train_n], y_train[train_n:]
        use_ordered_tail = len(np.unique(y_sub)) >= 2 and len(np.unique(y_val)) >= 2
    if not use_ordered_tail:
        try:
            X_sub, X_val, y_sub, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                stratify=y_train,
                random_state=seed + 2027,
            )
        except ValueError:
            return baseline_bundle, {"cf_validation_selected": "baseline_validation_split_failed"}

    val_baseline = baseline_predict_proba(X_sub, y_sub, X_val, estimator_factory)
    base_metrics = classification_metrics(y_val, val_baseline.proba_adj, val_baseline.classes)
    val_candidate = counterfactual_context_predict_proba(
        X_sub,
        y_sub,
        X_val,
        estimator_factory,
        config,
        baseline_proba=val_baseline.proba_adj,
        baseline_classes=val_baseline.classes,
    )
    cand_metrics = classification_metrics(y_val, val_candidate.proba_adj, val_candidate.classes)
    changed_fraction = float(
        np.mean(np.argmax(val_candidate.proba_adj, axis=1) != np.argmax(val_baseline.proba_adj, axis=1))
    )
    acc_delta = float(cand_metrics["accuracy"]) - float(base_metrics["accuracy"])
    bacc_delta = float(cand_metrics["balanced_accuracy"]) - float(base_metrics["balanced_accuracy"])
    f1_delta = float(cand_metrics["macro_f1"]) - float(base_metrics["macro_f1"])
    details: dict[str, object] = {
        "cf_validation_selected": "candidate",
        "cf_validation_size": len(y_val),
        "cf_validation_ordered_tail": use_ordered_tail,
        "cf_validation_accuracy_delta": acc_delta,
        "cf_validation_balanced_accuracy_delta": bacc_delta,
        "cf_validation_macro_f1_delta": f1_delta,
        "cf_validation_changed_fraction": changed_fraction,
    }
    diagnostics = getattr(val_candidate, "diagnostics", {})
    if diagnostics:
        if "eligible_final" in diagnostics:
            details["cf_validation_eligible_fraction"] = float(np.mean(diagnostics["eligible_final"]))
        if "local_prior_js" in diagnostics:
            local_js = np.asarray(diagnostics["local_prior_js"], dtype=float)
            details["cf_validation_mean_local_prior_js"] = float(np.mean(local_js))
            details["cf_validation_max_local_prior_js"] = float(np.max(local_js))

    should_apply = (
        changed_fraction > 0.0
        and acc_delta > 1e-12
        and bacc_delta >= -0.010
        and f1_delta >= -0.010
    )
    if not should_apply:
        details["cf_validation_selected"] = "baseline_no_validation_gain"
        return baseline_bundle, details

    final_bundle = counterfactual_context_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory,
        config,
        baseline_proba=baseline_bundle.proba_adj,
        baseline_classes=baseline_bundle.classes,
    )
    return final_bundle, details


def context_policy_validation_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle,
    estimator_factory,
    k: int,
    seed: int,
) -> tuple[object, dict[str, object]]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or len(y_train) < 32 or np.min(counts) < 3:
        return baseline_bundle, {"context_policy_selected": "baseline_insufficient_train"}

    val_size = min(max(len(y_train) // 4, len(classes) * 4), max(len(y_train) - len(classes) * 2, len(classes) * 4))
    train_n = len(y_train) - val_size
    use_ordered_tail = train_n >= len(classes) * 2
    if use_ordered_tail:
        X_sub, X_val = X_train[:train_n], X_train[train_n:]
        y_sub, y_val = y_train[:train_n], y_train[train_n:]
        use_ordered_tail = len(np.unique(y_sub)) >= 2 and len(np.unique(y_val)) >= 2
    if not use_ordered_tail:
        try:
            X_sub, X_val, y_sub, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                stratify=y_train,
                random_state=seed + 3037,
            )
        except ValueError:
            return baseline_bundle, {"context_policy_selected": "baseline_validation_split_failed"}

    val_baseline = baseline_predict_proba(X_sub, y_sub, X_val, estimator_factory)
    base_metrics = classification_metrics(y_val, val_baseline.proba_adj, val_baseline.classes)
    base_score = score_for_selection(base_metrics)
    candidate_configs = {
        "knn_context": APAConfig(context_strategy="knn", correction="none", k=max(4, min(k, 24))),
        "balanced_knn_context": APAConfig(context_strategy="balanced_knn", correction="none", k=max(4, min(k, 24))),
        "prior_aware_context": APAConfig(
            context_strategy="prior_aware_knn",
            correction="none",
            k=max(4, min(k, 24)),
            lambda_balance=0.25,
        ),
        "prior_aware_strong_context": APAConfig(
            context_strategy="prior_aware_knn",
            correction="none",
            k=max(4, min(k, 24)),
            lambda_balance=0.45,
        ),
    }
    best_name = "baseline"
    best_config: APAConfig | None = None
    best_metrics = base_metrics
    best_score = base_score
    best_runtime = 0.0

    for name, candidate_config in candidate_configs.items():
        candidate_bundle = apa_predict_proba(X_sub, y_sub, X_val, estimator_factory, candidate_config)
        metrics = classification_metrics(y_val, candidate_bundle.proba_adj, candidate_bundle.classes)
        score = score_for_selection(metrics)
        acc_delta = float(metrics["accuracy"]) - float(base_metrics["accuracy"])
        bacc_delta = float(metrics["balanced_accuracy"]) - float(base_metrics["balanced_accuracy"])
        f1_delta = float(metrics["macro_f1"]) - float(base_metrics["macro_f1"])
        no_harm = acc_delta > 1e-12 and bacc_delta >= -0.020 and f1_delta >= -0.020
        if no_harm and score > best_score:
            best_name = name
            best_config = candidate_config
            best_metrics = metrics
            best_score = score
            best_runtime = candidate_bundle.runtime_sec

    details: dict[str, object] = {
        "context_policy_selected": best_name,
        "context_policy_validation_size": len(y_val),
        "context_policy_ordered_tail": use_ordered_tail,
        "context_policy_base_score": base_score,
        "context_policy_selected_score": best_score,
        "context_policy_validation_accuracy_delta": float(best_metrics["accuracy"]) - float(base_metrics["accuracy"]),
        "context_policy_validation_balanced_accuracy_delta": float(best_metrics["balanced_accuracy"]) - float(base_metrics["balanced_accuracy"]),
        "context_policy_validation_macro_f1_delta": float(best_metrics["macro_f1"]) - float(base_metrics["macro_f1"]),
        "context_policy_validation_runtime_sec": best_runtime,
    }
    if best_config is None:
        details["context_policy_selected"] = "baseline_no_validation_gain"
        return baseline_bundle, details

    final_bundle = apa_predict_proba(X_train, y_train, X_test, estimator_factory, best_config)
    return final_bundle, details


def posthoc_diagnostics(proba: np.ndarray, y_train: np.ndarray, classes: np.ndarray, config: APAConfig) -> dict[str, object]:
    if config.binary_only and len(classes) != 2:
        return {
            "diagnostic_batch_prior_js": "",
            "diagnostic_mean_margin": "",
            "diagnostic_mean_confidence": "",
            "diagnostic_eligible_fraction": 0.0,
        }
    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    y_pos = np.asarray([class_to_pos[label] for label in y_train], dtype=int)
    train_prior = context_prior(y_pos, len(classes), alpha=config.alpha).reshape(1, -1)
    batch_prior = np.mean(np.asarray(proba, dtype=np.float64), axis=0, keepdims=True)
    batch_prior_js = float(js_divergence(batch_prior, train_prior, eps=config.eps)[0] / np.log(2.0))
    sorted_proba = np.sort(np.asarray(proba, dtype=np.float64), axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else np.ones(proba.shape[0])
    confidence = np.max(np.asarray(proba, dtype=np.float64), axis=1)
    eligible = (
        (batch_prior_js >= config.min_batch_prior_js)
        & (margin <= config.max_baseline_margin)
        & (confidence <= config.max_baseline_confidence)
    )
    return {
        "diagnostic_batch_prior_js": batch_prior_js,
        "diagnostic_mean_margin": float(np.mean(margin)),
        "diagnostic_mean_confidence": float(np.mean(confidence)),
        "diagnostic_eligible_fraction": float(np.mean(eligible)),
    }


def batch_prior_js(proba: np.ndarray, y_train: np.ndarray, classes: np.ndarray, alpha: float = 1.0, eps: float = 1e-8) -> float:
    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    y_pos = np.asarray([class_to_pos[label] for label in y_train], dtype=int)
    train_prior = context_prior(y_pos, len(classes), alpha=alpha).reshape(1, -1)
    batch_prior = np.mean(np.asarray(proba, dtype=np.float64), axis=0, keepdims=True)
    return float(js_divergence(batch_prior, train_prior, eps=eps)[0] / np.log(2.0))


def dual_shift_router_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    baseline_bundle,
    k: int,
    hard_js_threshold: float = 0.004,
) -> tuple[object, dict[str, object]]:
    if hard_js_threshold == 0.008:
        config, branch, js_value = apa_v7_config_for_proba(
            baseline_bundle.proba_adj,
            y_train,
            baseline_bundle.classes,
            k=k,
        )
        if config is None:
            return baseline_bundle, {
                "dual_router_branch": branch,
                "diagnostic_batch_prior_js": js_value if np.isfinite(js_value) else "",
                "diagnostic_eligible_fraction": 0.0,
            }
        details = posthoc_diagnostics(baseline_bundle.proba_adj, y_train, baseline_bundle.classes, config)
        details["dual_router_branch"] = branch
        bundle = correct_existing_proba_with_context_prior(
            baseline_bundle.proba_adj,
            X_train,
            y_train,
            X_test,
            baseline_bundle.classes,
            config,
        )
        return bundle, details

    if len(baseline_bundle.classes) != 2:
        return baseline_bundle, {
            "dual_router_branch": "baseline_nonbinary",
            "diagnostic_batch_prior_js": "",
            "diagnostic_eligible_fraction": 0.0,
        }

    js_value = batch_prior_js(baseline_bundle.proba_adj, y_train, baseline_bundle.classes, alpha=1.0)
    if js_value >= 0.05:
        config = APAConfig(
            context_strategy="prior_aware_knn",
            correction="adaptive",
            k=k,
            lambda_balance=0.20,
            alpha=1.0,
            query_mode="batch",
            eta=0.0,
            gamma_min=0.0,
            gamma_max=1.0,
            min_batch_prior_js=0.005,
            max_baseline_margin=0.28,
            max_baseline_confidence=0.72,
            blend_weight=1.0,
            binary_only=True,
        )
        branch = "high_js_v2"
    elif js_value >= hard_js_threshold:
        config = APAConfig(
            context_strategy="prior_aware_knn",
            correction="adaptive",
            k=k,
            lambda_balance=0.30,
            alpha=0.5,
            query_mode="hybrid",
            eta=0.65,
            gamma_min=0.05,
            gamma_max=1.4,
            min_batch_prior_js=hard_js_threshold,
            max_baseline_margin=0.55,
            max_baseline_confidence=0.90,
            blend_weight=1.0,
            binary_only=True,
        )
        branch = "hard_js_v5"
    else:
        return baseline_bundle, {
            "dual_router_branch": "baseline_low_js",
            "diagnostic_batch_prior_js": js_value,
            "diagnostic_eligible_fraction": 0.0,
        }

    details = posthoc_diagnostics(baseline_bundle.proba_adj, y_train, baseline_bundle.classes, config)
    details["dual_router_branch"] = branch
    bundle = correct_existing_proba_with_context_prior(
        baseline_bundle.proba_adj,
        X_train,
        y_train,
        X_test,
        baseline_bundle.classes,
        config,
    )
    return bundle, details


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["method"]), []).append(row)
    summary: list[dict[str, object]] = []
    for method, group in sorted(groups.items()):
        out: dict[str, object] = {"method": method, "n": len(group)}
        for key in METRIC_KEYS:
            vals = [float(row[key]) for row in group if row.get(key) not in {"", None}]
            if vals:
                out[f"mean_{key}"] = float(np.mean(vals))
        for key in [f"{metric}_delta" for metric in METRIC_KEYS]:
            vals = [float(row[key]) for row in group if row.get(key) not in {"", None}]
            if vals:
                out[f"mean_{key}"] = float(np.mean(vals))
        summary.append(out)
    return summary


def summarize_shift_curve(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("split_kind") != "label_shift":
            continue
        key = (str(row["dataset"]), str(row["label_shift_strength"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    curve_rows: list[dict[str, object]] = []
    for (dataset, strength, method), group in sorted(groups.items(), key=lambda item: (item[0][0], float(item[0][1]), item[0][2])):
        out: dict[str, object] = {
            "dataset": dataset,
            "label_shift_strength": strength,
            "method": method,
            "n": len(group),
        }
        for key in METRIC_KEYS:
            vals = [float(row[key]) for row in group if row.get(key) not in {"", None}]
            if vals:
                out[f"mean_{key}"] = float(np.mean(vals))
        for key in [f"{metric}_delta" for metric in METRIC_KEYS]:
            vals = [float(row[key]) for row in group if row.get(key) not in {"", None}]
            if vals:
                out[f"mean_{key}"] = float(np.mean(vals))
        curve_rows.append(out)
    return curve_rows


def write_report(path: Path, detail_rows: list[dict[str, object]], summary_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# APA-TabICL Experiment 6 Report",
        "",
        "Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.",
        "",
        "This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.",
        "",
        "## Command",
        "",
        "```text",
        " ".join(sys.argv),
        "```",
        "",
        "## Method Summary",
        "",
        "| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['n']} | "
            f"{fmt(row.get('mean_accuracy_delta'), signed=True)} | "
            f"{fmt(row.get('mean_balanced_accuracy_delta'), signed=True)} | "
            f"{fmt(row.get('mean_macro_f1_delta'), signed=True)} | "
            f"{fmt(row.get('mean_minority_recall_delta'), signed=True)} | "
            f"{fmt(row.get('mean_roc_auc_delta'), signed=True)} | "
            f"{fmt(row.get('mean_ece_delta'), signed=True)} |"
        )
    lines.extend(["", "## Dataset Detail", ""])
    for row in detail_rows:
        if row["method"] == "tabicl_baseline":
            continue
        lines.append(
            f"- `{row['dataset']}` seed {row['seed']} `{row['method']}` ({row.get('algorithm_version', '')}): "
            f"shift {fmt(row.get('label_shift_strength'))}, "
            f"acc {fmt(row['accuracy_delta'], signed=True)}, "
            f"bacc {fmt(row['balanced_accuracy_delta'], signed=True)}, "
            f"macro-F1 {fmt(row['macro_f1_delta'], signed=True)}, "
            f"minority recall {fmt(row['minority_recall_delta'], signed=True)}, "
            f"ROC-AUC {fmt(row.get('roc_auc_delta'), signed=True)}, "
            f"ECE {fmt(row['ece_delta'], signed=True)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_prior_details(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    classes = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    train_counts = np.asarray([np.sum(y_train == cls) for cls in classes], dtype=float)
    test_counts = np.asarray([np.sum(y_test == cls) for cls in classes], dtype=float)
    train_prior = train_counts / max(1.0, float(train_counts.sum()))
    test_prior = test_counts / max(1.0, float(test_counts.sum()))
    return {
        "split_classes": "|".join(str(cls) for cls in classes),
        "train_class_counts": "|".join(str(int(v)) for v in train_counts),
        "test_class_counts": "|".join(str(int(v)) for v in test_counts),
        "train_class_prior": "|".join(f"{v:.6f}" for v in train_prior),
        "test_class_prior": "|".join(f"{v:.6f}" for v in test_prior),
        "train_test_prior_l1": float(np.sum(np.abs(train_prior - test_prior))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APA-TabICL inference-stage ablations.")
    parser.add_argument("--datasets", default="synthetic_imbalanced,synthetic_label_shift,wine")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-samples", type=int, default=360)
    parser.add_argument("--train-size", type=int, default=160)
    parser.add_argument("--test-size", type=int, default=80)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--label-shift-strengths", default="0.5")
    parser.add_argument("--force-label-shift", action="store_true")
    parser.add_argument("--split-mode", choices=["standard", "ordered"], default="standard")
    parser.add_argument("--output-prefix", default="apa_v0_smoke")
    args = parser.parse_args()

    methods = method_configs(args.k)
    selected_methods = list(methods) if args.methods == "all" else parse_csv(args.methods)
    unknown = [method for method in selected_methods if method not in methods]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    def estimator_factory() -> TabICLClassifier:
        return TabICLClassifier(n_estimators=args.n_estimators, device=args.device, random_state=0)

    detail_rows: list[dict[str, object]] = []
    for dataset_name in parse_csv(args.datasets):
        shift_strengths = (
            parse_floats(args.label_shift_strengths)
            if args.force_label_shift or dataset_name in {"synthetic_label_shift", "synthetic_label_shift_hard"}
            else [0.0]
        )
        for shift_strength in shift_strengths:
            for seed in [int(s) for s in parse_csv(args.seeds)]:
                reset_seed(seed)
                spec = make_dataset(dataset_name, args.n_samples, seed)
                if args.force_label_shift and len(np.unique(spec.y)) == 2:
                    spec = DatasetSpec(spec.name, spec.X, spec.y, split_kind="label_shift")
                elif args.split_mode == "ordered":
                    spec = DatasetSpec(spec.name, spec.X, spec.y, split_kind="ordered")
                X_train, X_test, y_train, y_test = split_dataset(
                    spec,
                    seed,
                    args.train_size,
                    args.test_size,
                    label_shift_strength=shift_strength,
                )

                baseline_bundle = baseline_predict_proba(X_train, y_train, X_test, estimator_factory)
                baseline_metrics = classification_metrics(y_test, baseline_bundle.proba_adj, baseline_bundle.classes)
                prior_details = split_prior_details(y_train, y_test)

                for method in selected_methods:
                    method_run = methods[method]
                    config = method_run.config
                    extra_details: dict[str, object] = {}
                    if method_run.mode == "baseline":
                        bundle = baseline_bundle
                    elif method_run.mode == "posthoc_full":
                        assert config is not None
                        extra_details.update(posthoc_diagnostics(baseline_bundle.proba_adj, y_train, baseline_bundle.classes, config))
                        bundle = correct_existing_proba_with_context_prior(
                            baseline_bundle.proba_adj,
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle.classes,
                            config,
                        )
                    elif method_run.mode == "validation_selector":
                        bundle, extra_details = validation_select_and_correct(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            args.k,
                            seed,
                        )
                    elif method_run.mode == "dual_shift_router":
                        bundle, extra_details = dual_shift_router_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            args.k,
                        )
                    elif method_run.mode == "dual_shift_router_v7":
                        bundle, extra_details = dual_shift_router_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            args.k,
                            hard_js_threshold=0.008,
                        )
                    elif method_run.mode == "context_policy_selector":
                        bundle, extra_details = context_policy_validation_select(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            args.k,
                            seed,
                        )
                    elif method_run.mode == "no_shift_feature_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle = no_shift_feature_context_predict_proba(
                            X_train,
                            y_train,
                            X_test,
                            estimator_factory,
                            config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        diagnostics = getattr(bundle, "diagnostics", {})
                        extra_details.update(
                            {
                                "no_shift_batch_prior_js": float(np.mean(diagnostics.get("no_shift_batch_prior_js", [0.0]))),
                                "no_shift_eligible_fraction": float(np.mean(diagnostics.get("no_shift_eligible_final", [0.0]))),
                                "no_shift_mean_vote_fraction": float(np.mean(diagnostics.get("no_shift_vote_fraction", [0.0]))),
                                "no_shift_mean_context_disagreement": float(np.mean(diagnostics.get("no_shift_context_disagreement", [0.0]))),
                                "no_shift_mean_local_support_margin": float(np.mean(diagnostics.get("no_shift_local_support_margin", [0.0]))),
                                "no_shift_changed_fraction": float(np.mean(diagnostics.get("no_shift_changed_fraction", [0.0]))),
                                "no_shift_context_failure_fraction": float(np.mean(diagnostics.get("no_shift_context_failure_fraction", [0.0]))),
                                "no_shift_minority_safe_rejected_fraction": float(np.mean(diagnostics.get("no_shift_minority_safe_rejected_fraction", [0.0]))),
                            }
                        )
                    elif method_run.mode == "no_shift_union_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        strict_config = config
                        soft_config = NoShiftFeatureConfig(
                            k=config.k,
                            candidate_multiplier=config.candidate_multiplier,
                            lambda_div=0.35,
                            tau_clean=config.tau_clean,
                            clean_neighbor_k=config.clean_neighbor_k,
                            n_bootstrap=config.n_bootstrap,
                            n_subspaces=config.n_subspaces,
                            subspace_fraction=config.subspace_fraction,
                            max_batch_prior_js=config.max_batch_prior_js,
                            max_baseline_margin=0.38,
                            max_baseline_confidence=0.92,
                            max_context_disagreement=0.065,
                            min_context_vote_fraction=0.67,
                            local_evidence_margin=-0.010,
                            blend_weight=0.22,
                            max_changed_fraction=config.max_changed_fraction,
                            families=config.families,
                            random_state=config.random_state,
                        )
                        strict_bundle = no_shift_feature_context_predict_proba(
                            X_train, y_train, X_test, estimator_factory, strict_config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        soft_bundle = no_shift_feature_context_predict_proba(
                            X_train, y_train, X_test, estimator_factory, soft_config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        baseline_pred = np.argmax(baseline_bundle.proba_adj, axis=1)
                        strict_diag = getattr(strict_bundle, "diagnostics", {})
                        soft_diag = getattr(soft_bundle, "diagnostics", {})
                        strict_changed = np.argmax(strict_bundle.proba_adj, axis=1) != baseline_pred
                        soft_changed = np.argmax(soft_bundle.proba_adj, axis=1) != baseline_pred
                        proba_adj = baseline_bundle.proba_adj.copy()
                        chosen = np.zeros(len(baseline_pred), dtype=bool)
                        strict_score = (
                            np.asarray(strict_diag.get("no_shift_vote_fraction", np.zeros(len(baseline_pred))))
                            + np.asarray(strict_diag.get("no_shift_local_support_margin", np.zeros(len(baseline_pred))))
                            - np.asarray(strict_diag.get("no_shift_context_disagreement", np.zeros(len(baseline_pred))))
                        )
                        soft_score = (
                            np.asarray(soft_diag.get("no_shift_vote_fraction", np.zeros(len(baseline_pred))))
                            + np.asarray(soft_diag.get("no_shift_local_support_margin", np.zeros(len(baseline_pred))))
                            - np.asarray(soft_diag.get("no_shift_context_disagreement", np.zeros(len(baseline_pred))))
                        )
                        for i in range(len(baseline_pred)):
                            if strict_changed[i] and (not soft_changed[i] or strict_score[i] >= soft_score[i]):
                                proba_adj[i] = strict_bundle.proba_adj[i]
                                chosen[i] = True
                            elif soft_changed[i]:
                                proba_adj[i] = soft_bundle.proba_adj[i]
                                chosen[i] = True
                        if config.max_changed_fraction < 1.0 and np.any(chosen):
                            max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * len(chosen))))
                            score = np.maximum(strict_score, soft_score)
                            keep = np.zeros_like(chosen, dtype=bool)
                            idx = np.flatnonzero(chosen)
                            keep[idx[np.argsort(score[idx])[-max_changes:]]] = True
                            proba_adj = np.where(keep[:, None], proba_adj, baseline_bundle.proba_adj)
                            chosen = keep
                        bundle = APAPrediction(
                            proba_raw=baseline_bundle.proba_adj,
                            proba_adj=proba_adj,
                            pred=baseline_bundle.classes[np.argmax(proba_adj, axis=1)],
                            classes=baseline_bundle.classes,
                            context_indices=strict_bundle.context_indices,
                            effective_prior=strict_bundle.effective_prior,
                            gamma=np.maximum(strict_bundle.gamma, soft_bundle.gamma),
                            delta=np.minimum(strict_bundle.delta, soft_bundle.delta),
                            runtime_sec=strict_bundle.runtime_sec + soft_bundle.runtime_sec,
                            diagnostics={
                                "no_shift_eligible_final": chosen.astype(float),
                                "no_shift_changed_fraction": np.full(len(chosen), float(np.mean(chosen))),
                                "no_shift_vote_fraction": np.maximum(
                                    np.asarray(strict_diag.get("no_shift_vote_fraction", np.zeros(len(chosen)))),
                                    np.asarray(soft_diag.get("no_shift_vote_fraction", np.zeros(len(chosen)))),
                                ),
                                "no_shift_context_disagreement": np.minimum(
                                    np.asarray(strict_diag.get("no_shift_context_disagreement", np.zeros(len(chosen)))),
                                    np.asarray(soft_diag.get("no_shift_context_disagreement", np.zeros(len(chosen)))),
                                ),
                                "no_shift_local_support_margin": np.maximum(
                                    np.asarray(strict_diag.get("no_shift_local_support_margin", np.zeros(len(chosen)))),
                                    np.asarray(soft_diag.get("no_shift_local_support_margin", np.zeros(len(chosen)))),
                                ),
                                "no_shift_batch_prior_js": np.asarray(strict_diag.get("no_shift_batch_prior_js", np.zeros(len(chosen)))),
                                "no_shift_context_failure_fraction": np.maximum(
                                    np.asarray(strict_diag.get("no_shift_context_failure_fraction", np.zeros(len(chosen)))),
                                    np.asarray(soft_diag.get("no_shift_context_failure_fraction", np.zeros(len(chosen)))),
                                ),
                            },
                        )
                        diagnostics = getattr(bundle, "diagnostics", {})
                        extra_details.update(
                            {
                                "no_shift_batch_prior_js": float(np.mean(diagnostics.get("no_shift_batch_prior_js", [0.0]))),
                                "no_shift_eligible_fraction": float(np.mean(diagnostics.get("no_shift_eligible_final", [0.0]))),
                                "no_shift_mean_vote_fraction": float(np.mean(diagnostics.get("no_shift_vote_fraction", [0.0]))),
                                "no_shift_mean_context_disagreement": float(np.mean(diagnostics.get("no_shift_context_disagreement", [0.0]))),
                                "no_shift_mean_local_support_margin": float(np.mean(diagnostics.get("no_shift_local_support_margin", [0.0]))),
                                "no_shift_changed_fraction": float(np.mean(diagnostics.get("no_shift_changed_fraction", [0.0]))),
                                "no_shift_context_failure_fraction": float(np.mean(diagnostics.get("no_shift_context_failure_fraction", [0.0]))),
                            }
                        )
                    elif method_run.mode == "no_shift_boundary_union_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        strict_config = config
                        soft_config = NoShiftFeatureConfig(
                            k=config.k,
                            candidate_multiplier=config.candidate_multiplier,
                            lambda_div=0.35,
                            tau_clean=config.tau_clean,
                            clean_neighbor_k=config.clean_neighbor_k,
                            n_bootstrap=config.n_bootstrap,
                            n_subspaces=config.n_subspaces,
                            subspace_fraction=config.subspace_fraction,
                            max_batch_prior_js=config.max_batch_prior_js,
                            max_baseline_margin=0.38,
                            max_baseline_confidence=0.92,
                            max_context_disagreement=0.065,
                            min_context_vote_fraction=0.67,
                            local_evidence_margin=-0.010,
                            blend_weight=0.22,
                            max_changed_fraction=config.max_changed_fraction,
                            families=config.families,
                            random_state=config.random_state,
                        )
                        boundary_config = NoShiftFeatureConfig(
                            k=config.k,
                            candidate_multiplier=config.candidate_multiplier,
                            lambda_div=config.lambda_div,
                            tau_clean=0.65,
                            clean_neighbor_k=config.clean_neighbor_k,
                            n_bootstrap=config.n_bootstrap,
                            n_subspaces=config.n_subspaces,
                            subspace_fraction=config.subspace_fraction,
                            max_batch_prior_js=config.max_batch_prior_js,
                            max_baseline_margin=0.24,
                            max_baseline_confidence=0.84,
                            max_context_disagreement=0.030,
                            min_context_vote_fraction=0.80,
                            local_evidence_margin=0.050,
                            blend_weight=0.22,
                            max_changed_fraction=0.015625,
                            families=("boundary_contrast", "clean_knn", "diverse_knn"),
                            random_state=config.random_state,
                        )
                        strict_bundle = no_shift_feature_context_predict_proba(
                            X_train, y_train, X_test, estimator_factory, strict_config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        soft_bundle = no_shift_feature_context_predict_proba(
                            X_train, y_train, X_test, estimator_factory, soft_config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        boundary_bundle = no_shift_feature_context_predict_proba(
                            X_train, y_train, X_test, estimator_factory, boundary_config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        baseline_pred = np.argmax(baseline_bundle.proba_adj, axis=1)
                        n_rows = len(baseline_pred)
                        candidate_bundles = [strict_bundle, soft_bundle, boundary_bundle]
                        candidate_diags = [getattr(bundle, "diagnostics", {}) for bundle in candidate_bundles]

                        def candidate_score(diag):
                            return (
                                np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float)
                                + np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float)
                                - np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float)
                            )

                        candidate_scores = [candidate_score(diag) for diag in candidate_diags]
                        candidate_changed = [
                            np.argmax(bundle.proba_adj, axis=1) != baseline_pred for bundle in candidate_bundles
                        ]
                        boundary_guard_pass = np.ones(n_rows, dtype=bool)
                        if method_run.version in {"apa_v27", "apa_v32", "apa_v33"}:
                            _, train_counts_guard = np.unique(y_train, return_counts=True)
                            train_minority_prior_guard = float(np.min(train_counts_guard) / max(1, np.sum(train_counts_guard)))
                            combined_vote_guard = np.maximum.reduce([
                                np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float)
                                for diag in candidate_diags
                            ])
                            combined_disagreement_guard = np.minimum.reduce([
                                np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float)
                                for diag in candidate_diags
                            ])
                            combined_support_guard = np.maximum.reduce([
                                np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float)
                                for diag in candidate_diags
                            ])
                            boundary_prior_window = (train_minority_prior_guard >= 0.18) & (train_minority_prior_guard <= 0.34)
                            strong_support = combined_support_guard >= 0.030
                            strong_consensus = (combined_vote_guard >= 0.940) & (combined_disagreement_guard <= 0.035)
                            boundary_guard_pass = boundary_prior_window & (strong_support | strong_consensus)
                            candidate_changed[2] = candidate_changed[2] & boundary_guard_pass
                        proba_adj = baseline_bundle.proba_adj.copy()
                        chosen = np.zeros(n_rows, dtype=bool)
                        chosen_score = np.full(n_rows, -np.inf)
                        for bundle, score, changed in zip(candidate_bundles, candidate_scores, candidate_changed):
                            take = changed & (score > chosen_score)
                            if np.any(take):
                                proba_adj[take] = bundle.proba_adj[take]
                                chosen[take] = True
                                chosen_score[take] = score[take]
                        minority_safe_reject = np.zeros(n_rows, dtype=bool)
                        pvalue_guard_reject = np.zeros(n_rows, dtype=bool)
                        local_p_values = np.ones(n_rows, dtype=np.float64)
                        local_candidate_support_guard = np.zeros(n_rows, dtype=np.float64)
                        local_baseline_support_guard = np.zeros(n_rows, dtype=np.float64)
                        local_purity_guard = np.zeros(n_rows, dtype=np.float64)
                        if method_run.version == "apa_v32" and np.any(chosen):
                            y_train_pos_guard = _train_label_positions(y_train, baseline_bundle.classes)
                            train_counts_safe = np.bincount(
                                y_train_pos_guard[y_train_pos_guard >= 0],
                                minlength=len(baseline_bundle.classes),
                            ).astype(float)
                            train_prior_safe = train_counts_safe / max(1.0, float(np.sum(train_counts_safe)))
                            minority_mask_safe = train_prior_safe <= 0.35
                            if len(baseline_bundle.classes) > 1 and np.any(train_counts_safe > 0):
                                minority_mask_safe[int(np.argmin(np.where(train_counts_safe > 0, train_counts_safe, np.inf)))] = True
                            candidate_pred_safe = np.argmax(proba_adj, axis=1)
                            minority_safe_reject = (
                                chosen
                                & minority_mask_safe[baseline_pred]
                                & ~minority_mask_safe[candidate_pred_safe]
                            )
                            if np.any(minority_safe_reject):
                                proba_adj[minority_safe_reject] = baseline_bundle.proba_adj[minority_safe_reject]
                                chosen[minority_safe_reject] = False
                                chosen_score[minority_safe_reject] = -np.inf
                        if method_run.version == "apa_v33" and np.any(chosen):
                            y_train_pos_guard = _train_label_positions(y_train, baseline_bundle.classes)
                            train_counts_safe = np.bincount(
                                y_train_pos_guard[y_train_pos_guard >= 0],
                                minlength=len(baseline_bundle.classes),
                            ).astype(float)
                            train_prior_safe = train_counts_safe / max(1.0, float(np.sum(train_counts_safe)))
                            minority_mask_safe = train_prior_safe <= 0.35
                            if len(baseline_bundle.classes) > 1 and np.any(train_counts_safe > 0):
                                minority_mask_safe[int(np.argmin(np.where(train_counts_safe > 0, train_counts_safe, np.inf)))] = True
                            candidate_pred_safe = np.argmax(proba_adj, axis=1)
                            local_k = max(config.k, 16)
                            local_candidate_support_guard, local_baseline_support_guard, local_purity_guard = _local_neighbor_evidence(
                                X_train,
                                y_train_pos_guard,
                                X_test,
                                candidate_pred_safe,
                                baseline_pred,
                                len(baseline_bundle.classes),
                                k=local_k,
                            )
                            local_p_values = _local_support_p_values(
                                local_candidate_support_guard,
                                candidate_pred_safe,
                                train_prior_safe,
                                k=local_k,
                            )
                            baseline_is_minority = minority_mask_safe[baseline_pred] if len(minority_mask_safe) else np.zeros(n_rows, dtype=bool)
                            candidate_is_majority = ~minority_mask_safe[candidate_pred_safe] if len(minority_mask_safe) else np.zeros(n_rows, dtype=bool)
                            minority_risk = chosen & baseline_is_minority & candidate_is_majority
                            support_advantage = local_candidate_support_guard - local_baseline_support_guard
                            weak_local_evidence = (support_advantage < -0.020) | (
                                (local_p_values > 0.80) & (combined_vote_guard < 0.94)
                            )
                            risky_without_significance = minority_risk & (
                                (local_p_values > 0.35)
                                | (support_advantage < -0.010)
                                | (local_purity_guard < 0.55)
                            )
                            pvalue_guard_reject = chosen & (weak_local_evidence | risky_without_significance)
                            if np.any(pvalue_guard_reject):
                                proba_adj[pvalue_guard_reject] = baseline_bundle.proba_adj[pvalue_guard_reject]
                                chosen[pvalue_guard_reject] = False
                                chosen_score[pvalue_guard_reject] = -np.inf
                        if config.max_changed_fraction < 1.0 and np.any(chosen):
                            max_changes = max(1, int(np.ceil(float(config.max_changed_fraction) * n_rows)))
                            keep = np.zeros_like(chosen, dtype=bool)
                            idx = np.flatnonzero(chosen)
                            keep[idx[np.argsort(chosen_score[idx])[-max_changes:]]] = True
                            proba_adj = np.where(keep[:, None], proba_adj, baseline_bundle.proba_adj)
                            chosen = keep
                        bundle = APAPrediction(
                            proba_raw=baseline_bundle.proba_adj,
                            proba_adj=proba_adj,
                            pred=baseline_bundle.classes[np.argmax(proba_adj, axis=1)],
                            classes=baseline_bundle.classes,
                            context_indices=strict_bundle.context_indices,
                            effective_prior=strict_bundle.effective_prior,
                            gamma=np.maximum.reduce([b.gamma for b in candidate_bundles]),
                            delta=np.minimum.reduce([b.delta for b in candidate_bundles]),
                            runtime_sec=sum(float(b.runtime_sec) for b in candidate_bundles),
                            diagnostics={
                                "no_shift_eligible_final": chosen.astype(float),
                                "no_shift_changed_fraction": np.full(n_rows, float(np.mean(chosen))),
                                "no_shift_vote_fraction": np.maximum.reduce([
                                    np.asarray(diag.get("no_shift_vote_fraction", np.zeros(n_rows)), dtype=float)
                                    for diag in candidate_diags
                                ]),
                                "no_shift_context_disagreement": np.minimum.reduce([
                                    np.asarray(diag.get("no_shift_context_disagreement", np.zeros(n_rows)), dtype=float)
                                    for diag in candidate_diags
                                ]),
                                "no_shift_local_support_margin": np.maximum.reduce([
                                    np.asarray(diag.get("no_shift_local_support_margin", np.zeros(n_rows)), dtype=float)
                                    for diag in candidate_diags
                                ]),
                                "no_shift_batch_prior_js": np.asarray(
                                    candidate_diags[0].get("no_shift_batch_prior_js", np.zeros(n_rows)), dtype=float
                                ),
                                "no_shift_context_failure_fraction": np.maximum.reduce([
                                    np.asarray(diag.get("no_shift_context_failure_fraction", np.zeros(n_rows)), dtype=float)
                                    for diag in candidate_diags
                                ]),
                                "no_shift_boundary_guard_pass_fraction": np.full(n_rows, float(np.mean(boundary_guard_pass))),
                                "no_shift_minority_safe_rejected_fraction": np.full(n_rows, float(np.mean(minority_safe_reject))),
                                "no_shift_pvalue_guard_rejected_fraction": np.full(n_rows, float(np.mean(pvalue_guard_reject))),
                                "no_shift_local_support_p_value": local_p_values,
                                "no_shift_local_candidate_support": local_candidate_support_guard,
                                "no_shift_local_baseline_support": local_baseline_support_guard,
                                "no_shift_local_purity": local_purity_guard,
                            },
                        )
                        diagnostics = getattr(bundle, "diagnostics", {})
                        extra_details.update(
                            {
                                "no_shift_batch_prior_js": float(np.mean(diagnostics.get("no_shift_batch_prior_js", [0.0]))),
                                "no_shift_eligible_fraction": float(np.mean(diagnostics.get("no_shift_eligible_final", [0.0]))),
                                "no_shift_mean_vote_fraction": float(np.mean(diagnostics.get("no_shift_vote_fraction", [0.0]))),
                                "no_shift_mean_context_disagreement": float(np.mean(diagnostics.get("no_shift_context_disagreement", [0.0]))),
                                "no_shift_mean_local_support_margin": float(np.mean(diagnostics.get("no_shift_local_support_margin", [0.0]))),
                                "no_shift_changed_fraction": float(np.mean(diagnostics.get("no_shift_changed_fraction", [0.0]))),
                                "no_shift_context_failure_fraction": float(np.mean(diagnostics.get("no_shift_context_failure_fraction", [0.0]))),
                                "no_shift_pvalue_guard_rejected_fraction": float(np.mean(diagnostics.get("no_shift_pvalue_guard_rejected_fraction", [0.0]))),
                                "no_shift_mean_local_support_p_value": float(np.mean(diagnostics.get("no_shift_local_support_p_value", [1.0]))),
                                "no_shift_mean_local_purity": float(np.mean(diagnostics.get("no_shift_local_purity", [0.0]))),
                            }
                        )
                    elif method_run.mode == "no_shift_family_agreement_risk_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = family_agreement_risk_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                        )
                    elif method_run.mode == "no_shift_validation_boundary_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = guarded_boundary_union_validation_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_instance_ensemble_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = instancewise_context_ensemble_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                        )
                    elif method_run.mode == "no_shift_diversity_greedy_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = diversity_greedy_instance_router_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                        )
                    elif method_run.mode == "no_shift_diversity_greedy_pvalue_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = diversity_greedy_instance_router_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            pvalue_rescue=True,
                        )
                    elif method_run.mode == "no_shift_scale_combo_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = scale_aware_validation_instance_combo_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_boundary_bypass_combo_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = scale_aware_boundary_bypass_combo_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_boundary_diversity_selector_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = boundary_diversity_member_selector_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_precision_rescue_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = boundary_diversity_member_selector_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                            permissive_rescue=True,
                        )
                    elif method_run.mode == "no_shift_cross_model_tree_rescue_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = cross_model_tree_consensus_rescue_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_calibrated_tree_rescue_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = cross_model_tree_consensus_rescue_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                            calibrated_tree_rescue=True,
                        )
                    elif method_run.mode == "no_shift_prp_tree_rescue_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = cross_model_tree_consensus_rescue_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                            prp_tree_rescue=True,
                        )
                    elif method_run.mode == "no_shift_fast_tree_rescue_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = cross_model_tree_consensus_rescue_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                            primary_member="v27_fast",
                        )
                    elif method_run.mode == "no_shift_pseudo_label_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = pseudo_label_context_augmentation_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_prp_uncertainty_guard_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = prp_uncertainty_guarded_v42_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_prior_temperature_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = prior_matched_temperature_v42_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_context_consensus_calibrated_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = context_consensus_calibrated_v42_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "no_shift_validation_member_selector_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = validation_member_selector_v42_fallback_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "mixed_v38_v42_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = mixed_v38_v42_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                            args.k,
                        )
                    elif method_run.mode == "mixed_v38_v50_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = mixed_v38_v42_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                            args.k,
                            no_shift_member="v50",
                        )
                    elif method_run.mode == "no_shift_negative_control_stability_context":
                        assert isinstance(config, NoShiftFeatureConfig)
                        bundle, extra_details = guarded_boundary_union_stability_predict(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                        )
                    elif method_run.mode == "dual_mode_no_shift_cf":
                        assert isinstance(config, NoShiftFeatureConfig)
                        js_value = batch_prior_js(
                            baseline_bundle.proba_adj,
                            y_train,
                            baseline_bundle.classes,
                            alpha=1.0,
                        )
                        if js_value <= config.max_batch_prior_js:
                            bundle = no_shift_feature_context_predict_proba(
                                X_train,
                                y_train,
                                X_test,
                                estimator_factory,
                                config,
                                baseline_proba=baseline_bundle.proba_adj,
                                baseline_classes=baseline_bundle.classes,
                            )
                            branch_name = "no_shift_feature"
                            diagnostics = getattr(bundle, "diagnostics", {})
                            extra_details.update(
                                {
                                    "dual_mode_branch": branch_name,
                                    "no_shift_batch_prior_js": js_value,
                                    "no_shift_eligible_fraction": float(np.mean(diagnostics.get("no_shift_eligible_final", [0.0]))),
                                    "no_shift_changed_fraction": float(np.mean(diagnostics.get("no_shift_changed_fraction", [0.0]))),
                                }
                            )
                        else:
                            cf_config = CounterfactualContextConfig(
                                k=max(8, min(args.k, 24)),
                                lambda_balance=0.35,
                                alpha=0.75,
                                prior_floor=0.08,
                                evidence_temperature=1.10,
                                blend_weight=0.60,
                                sensitivity_gate=0.12,
                                max_abs_sensitivity=0.90,
                                max_abs_sensitivity_low_batch_margin_only=True,
                                disagreement_gate=0.004,
                                max_disagreement=0.060,
                                max_disagreement_low_batch_margin_only=False,
                                max_baseline_margin=0.55,
                                max_baseline_confidence=0.92,
                                min_batch_prior_js=0.008,
                                min_batch_pred_margin=0.05,
                                batch_sensitivity_rescue_gate=0.0,
                                min_train_prior_imbalance=0.10,
                                directional_shift_gate=True,
                                max_changed_fraction=0.10,
                                off_majority_max_baseline_margin=0.15,
                                aggregation="stability_debias",
                            )
                            bundle = counterfactual_context_predict_proba(
                                X_train,
                                y_train,
                                X_test,
                                estimator_factory,
                                cf_config,
                                baseline_proba=baseline_bundle.proba_adj,
                                baseline_classes=baseline_bundle.classes,
                            )
                            extra_details.update({"dual_mode_branch": "prior_intervention_v16", "no_shift_batch_prior_js": js_value})
                    elif method_run.mode == "dual_mode_v27_v14":
                        assert isinstance(config, NoShiftFeatureConfig)
                        js_value = batch_prior_js(
                            baseline_bundle.proba_adj,
                            y_train,
                            baseline_bundle.classes,
                            alpha=1.0,
                        )
                        if js_value <= config.max_batch_prior_js:
                            bundle, extra_details = guarded_boundary_union_v27_predict(
                                X_train,
                                y_train,
                                X_test,
                                baseline_bundle,
                                estimator_factory,
                                config,
                            )
                            extra_details["dual_mode_branch"] = "no_shift_v27"
                            extra_details["dual_mode_batch_prior_js"] = js_value
                            if method_run.version in {"apa_v35", "apa_v36"}:
                                support_margin = float(extra_details.get("no_shift_mean_local_support_margin") or 0.0)
                                changed_fraction = float(extra_details.get("no_shift_changed_fraction") or 0.0)
                                small_batch = len(y_train) < 128 or len(X_test) < 64
                                guard_rejected = (
                                    small_batch
                                    and changed_fraction > 0.0
                                    and support_margin < config.local_evidence_margin
                                )
                                extra_details.update(
                                    {
                                        "dual_mode_small_batch_guard": small_batch,
                                        "dual_mode_small_batch_guard_threshold": config.local_evidence_margin,
                                        "dual_mode_small_batch_guard_rejected": guard_rejected,
                                    }
                                )
                                if guard_rejected:
                                    bundle = replace(
                                        bundle,
                                        proba_adj=np.asarray(baseline_bundle.proba_adj, dtype=np.float64),
                                        pred=np.asarray(baseline_bundle.pred),
                                    )
                        else:
                            cf_config = CounterfactualContextConfig(
                                k=max(8, min(args.k, 24)),
                                lambda_balance=0.35,
                                alpha=0.75,
                                prior_floor=0.08,
                                evidence_temperature=1.10,
                                blend_weight=0.60,
                                sensitivity_gate=0.12,
                                max_abs_sensitivity=0.90,
                                max_abs_sensitivity_low_batch_margin_only=True,
                                disagreement_gate=0.004,
                                max_disagreement=0.040,
                                max_disagreement_low_batch_margin_only=True,
                                max_baseline_margin=0.55,
                                max_baseline_confidence=0.92,
                                min_batch_prior_js=0.008,
                                min_batch_pred_margin=0.05,
                                batch_sensitivity_rescue_gate=0.0,
                                min_train_prior_imbalance=0.10,
                                directional_shift_gate=True,
                                max_changed_fraction=0.10,
                                off_majority_max_baseline_margin=0.15,
                                aggregation="stability_debias",
                            )
                            bundle = counterfactual_context_predict_proba(
                                X_train,
                                y_train,
                                X_test,
                                estimator_factory,
                                cf_config,
                                baseline_proba=baseline_bundle.proba_adj,
                                baseline_classes=baseline_bundle.classes,
                            )
                            extra_details.update(
                                {
                                    "dual_mode_branch": "label_shift_v14",
                                    "dual_mode_batch_prior_js": js_value,
                                    "cf_batch_prior_js": batch_prior_js(
                                        baseline_bundle.proba_adj,
                                        y_train,
                                        baseline_bundle.classes,
                                        alpha=cf_config.alpha,
                                    ),
                                    "cf_mean_prior_sensitivity": float(np.mean(bundle.gamma)),
                                    "cf_mean_abs_prior_sensitivity": float(np.mean(np.abs(bundle.gamma))),
                                    "cf_mean_context_disagreement": float(np.mean(bundle.delta)),
                                    "cf_changed_fraction": float(
                                        np.mean(
                                            np.argmax(bundle.proba_adj, axis=1)
                                            != np.argmax(baseline_bundle.proba_adj, axis=1)
                                        )
                                    ),
                                }
                            )
                            if method_run.version == "apa_v36":
                                label_shift_min_context_disagreement = 0.014
                                cf_changed_fraction = float(extra_details.get("cf_changed_fraction") or 0.0)
                                cf_mean_context_disagreement = float(
                                    extra_details.get("cf_mean_context_disagreement") or 0.0
                                )
                                guard_rejected = (
                                    cf_changed_fraction > 0.0
                                    and cf_mean_context_disagreement < label_shift_min_context_disagreement
                                )
                                extra_details.update(
                                    {
                                        "dual_mode_label_shift_context_guard_threshold": (
                                            label_shift_min_context_disagreement
                                        ),
                                        "dual_mode_label_shift_context_guard_rejected": guard_rejected,
                                    }
                                )
                                if guard_rejected:
                                    bundle = replace(
                                        bundle,
                                        proba_adj=np.asarray(baseline_bundle.proba_adj, dtype=np.float64),
                                        pred=np.asarray(baseline_bundle.pred),
                                    )
                    elif method_run.mode == "dual_mode_v14_priority_v27":
                        assert isinstance(config, NoShiftFeatureConfig)
                        js_value = batch_prior_js(
                            baseline_bundle.proba_adj,
                            y_train,
                            baseline_bundle.classes,
                            alpha=1.0,
                        )
                        cf_config = CounterfactualContextConfig(
                            k=max(8, min(args.k, 24)),
                            lambda_balance=0.35,
                            alpha=0.75,
                            prior_floor=0.08,
                            evidence_temperature=1.10,
                            blend_weight=0.60,
                            sensitivity_gate=0.12,
                            max_abs_sensitivity=0.90,
                            max_abs_sensitivity_low_batch_margin_only=True,
                            disagreement_gate=0.004,
                            max_disagreement=0.040,
                            max_disagreement_low_batch_margin_only=True,
                            max_baseline_margin=0.55,
                            max_baseline_confidence=0.92,
                            min_batch_prior_js=0.008,
                            min_batch_pred_margin=0.05,
                            batch_sensitivity_rescue_gate=0.0,
                            min_train_prior_imbalance=0.10,
                            directional_shift_gate=True,
                            max_changed_fraction=0.10,
                            off_majority_max_baseline_margin=0.15,
                            aggregation="stability_debias",
                        )
                        cf_bundle = counterfactual_context_predict_proba(
                            X_train,
                            y_train,
                            X_test,
                            estimator_factory,
                            cf_config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        cf_changed_fraction = float(
                            np.mean(
                                np.argmax(cf_bundle.proba_adj, axis=1)
                                != np.argmax(baseline_bundle.proba_adj, axis=1)
                            )
                        )
                        cf_mean_context_disagreement = float(np.mean(cf_bundle.delta))
                        label_shift_min_context_disagreement = (
                            0.012 if method_run.version == "apa_v38" else 0.014
                        )
                        cf_rejected_low_disagreement = (
                            cf_changed_fraction > 0.0
                            and cf_mean_context_disagreement < label_shift_min_context_disagreement
                        )
                        extra_details.update(
                            {
                                "dual_mode_batch_prior_js": js_value,
                                "cf_batch_prior_js": batch_prior_js(
                                    baseline_bundle.proba_adj,
                                    y_train,
                                    baseline_bundle.classes,
                                    alpha=cf_config.alpha,
                                ),
                                "cf_mean_prior_sensitivity": float(np.mean(cf_bundle.gamma)),
                                "cf_mean_abs_prior_sensitivity": float(np.mean(np.abs(cf_bundle.gamma))),
                                "cf_mean_context_disagreement": cf_mean_context_disagreement,
                                "cf_changed_fraction": cf_changed_fraction,
                                "dual_mode_label_shift_context_guard_threshold": (
                                    label_shift_min_context_disagreement
                                ),
                                "dual_mode_label_shift_context_guard_rejected": (
                                    cf_rejected_low_disagreement
                                ),
                                "dual_mode_v37_cf_accepted": (
                                    cf_changed_fraction > 0.0 and not cf_rejected_low_disagreement
                                ),
                            }
                        )
                        if cf_changed_fraction > 0.0 and not cf_rejected_low_disagreement:
                            bundle = cf_bundle
                            extra_details["dual_mode_branch"] = "label_shift_v14_priority"
                            extra_details["dual_mode_v37_v27_fallback_accepted"] = False
                        else:
                            v27_bundle, v27_details = guarded_boundary_union_v27_predict(
                                X_train,
                                y_train,
                                X_test,
                                baseline_bundle,
                                estimator_factory,
                                config,
                            )
                            support_margin = float(v27_details.get("no_shift_mean_local_support_margin") or 0.0)
                            changed_fraction = float(v27_details.get("no_shift_changed_fraction") or 0.0)
                            context_disagreement = float(
                                v27_details.get("no_shift_mean_context_disagreement") or 0.0
                            )
                            small_batch = len(y_train) < 128 or len(X_test) < 64
                            strong_support = support_margin >= 0.020
                            strong_context_with_nonnegative_support = (
                                context_disagreement >= 0.030 and support_margin >= -0.005
                            )
                            v27_fallback_accepted = (
                                changed_fraction > 0.0
                                and js_value <= 0.001
                                and (strong_support or strong_context_with_nonnegative_support)
                                and not (small_batch and support_margin < config.local_evidence_margin)
                            )
                            extra_details.update(v27_details)
                            extra_details.update(
                                {
                                    "dual_mode_branch": (
                                        "no_shift_v27_strict_fallback"
                                        if v27_fallback_accepted
                                        else "baseline_v37_reject"
                                    ),
                                    "dual_mode_small_batch_guard": small_batch,
                                    "dual_mode_small_batch_guard_threshold": config.local_evidence_margin,
                                    "dual_mode_v37_v27_js_threshold": 0.001,
                                    "dual_mode_v37_v27_support_threshold": 0.020,
                                    "dual_mode_v37_v27_context_threshold": 0.030,
                                    "dual_mode_v37_v27_support_floor": -0.005,
                                    "dual_mode_v37_v27_fallback_accepted": v27_fallback_accepted,
                                }
                            )
                            if v27_fallback_accepted:
                                bundle = v27_bundle
                            else:
                                bundle = replace(
                                    baseline_bundle,
                                    runtime_sec=(
                                        baseline_bundle.runtime_sec
                                        + cf_bundle.runtime_sec
                                        + v27_bundle.runtime_sec
                                    ),
                                )
                    elif method_run.mode == "cf_validation_router":
                        assert isinstance(config, CounterfactualContextConfig)
                        bundle, extra_details = cf_validation_select_and_correct(
                            X_train,
                            y_train,
                            X_test,
                            baseline_bundle,
                            estimator_factory,
                            config,
                            seed,
                        )
                    elif method_run.mode == "counterfactual_context":
                        assert isinstance(config, CounterfactualContextConfig)
                        bundle = counterfactual_context_predict_proba(
                            X_train,
                            y_train,
                            X_test,
                            estimator_factory,
                            config,
                            baseline_proba=baseline_bundle.proba_adj,
                            baseline_classes=baseline_bundle.classes,
                        )
                        extra_details.update(
                            {
                                "cf_batch_prior_js": batch_prior_js(
                                    baseline_bundle.proba_adj,
                                    y_train,
                                    baseline_bundle.classes,
                                    alpha=config.alpha,
                                ),
                                "cf_mean_prior_sensitivity": float(np.mean(bundle.gamma)),
                                "cf_mean_abs_prior_sensitivity": float(np.mean(np.abs(bundle.gamma))),
                                "cf_mean_context_disagreement": float(np.mean(bundle.delta)),
                                "cf_changed_fraction": float(
                                    np.mean(
                                        np.argmax(bundle.proba_adj, axis=1)
                                        != np.argmax(baseline_bundle.proba_adj, axis=1)
                                    )
                                ),
                            }
                        )
                        diagnostics = getattr(bundle, "diagnostics", {})
                        if diagnostics:
                            if "eligible_final" in diagnostics:
                                extra_details["cf_eligible_fraction"] = float(np.mean(diagnostics["eligible_final"]))
                            if "eligible_before_batch" in diagnostics:
                                extra_details["cf_pre_gate_eligible_fraction"] = float(
                                    np.mean(diagnostics["eligible_before_batch"])
                                )
                            if "local_prior_js" in diagnostics:
                                local_js = np.asarray(diagnostics["local_prior_js"], dtype=float)
                                extra_details["cf_mean_local_prior_js"] = float(np.mean(local_js))
                                extra_details["cf_max_local_prior_js"] = float(np.max(local_js))
                            if "context_failure_fraction" in diagnostics:
                                extra_details["cf_context_failure_fraction"] = float(
                                    np.mean(diagnostics["context_failure_fraction"])
                                )
                    else:
                        assert config is not None
                        assert isinstance(config, APAConfig)
                        bundle = apa_predict_proba(X_train, y_train, X_test, estimator_factory, config)
                    metrics = classification_metrics(y_test, bundle.proba_adj, bundle.classes)
                    row: dict[str, object] = {
                        "dataset": dataset_name,
                        "split_kind": spec.split_kind,
                        "split_mode": args.split_mode,
                        "label_shift_strength": shift_strength if spec.split_kind == "label_shift" else "",
                        "seed": seed,
                        "method": method,
                        "run_mode": method_run.mode,
                        "algorithm_version": method_run.version,
                        "description": method_run.description,
                        "train_size": len(y_train),
                        "test_size": len(y_test),
                        "n_classes": len(np.unique(y_train)),
                        "runtime_sec": bundle.runtime_sec,
                        "mean_gamma": float(np.mean(bundle.gamma)),
                        "mean_delta": float(np.mean(bundle.delta)),
                    }
                    row.update(prior_details)
                    row.update(extra_details)
                    row.update(metrics)
                    for key in METRIC_KEYS:
                        if metrics.get(key) in {"", None} or baseline_metrics.get(key) in {"", None}:
                            row[f"{key}_delta"] = ""
                        else:
                            row[f"{key}_delta"] = float(metrics[key]) - float(baseline_metrics[key])
                    detail_rows.append(row)
                    print(
                        f"{dataset_name} shift={shift_strength:.2f} seed={seed} {method}: "
                        f"acc_delta={row['accuracy_delta']:+.4f} "
                        f"bacc_delta={row['balanced_accuracy_delta']:+.4f} "
                        f"f1_delta={row['macro_f1_delta']:+.4f}"
                    )

    summary_rows = summarize(detail_rows)
    shift_curve_rows = summarize_shift_curve(detail_rows)
    prefix = RESULTS_DIR / args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_detail.csv"), detail_rows)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    if shift_curve_rows:
        write_csv(prefix.with_name(prefix.name + "_shift_curve.csv"), shift_curve_rows)
    (prefix.with_name(prefix.name + "_config.json")).write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    write_report(prefix.with_name(prefix.name + "_report.md"), detail_rows, summary_rows, args)
    print(f"Wrote {prefix.name}_detail.csv, {prefix.name}_summary.csv, and report under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
