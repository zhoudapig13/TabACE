from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


EPS = 1e-12


@dataclass(frozen=True)
class APAConfig:
    context_strategy: str = "prior_aware_knn"
    correction: str = "adaptive"
    k: int = 64
    candidate_multiplier: int = 5
    lambda_balance: float = 0.25
    alpha: float = 1.0
    query_mode: str = "hybrid"
    eta: float = 0.5
    gamma_min: float = 0.0
    gamma_max: float = 1.0
    eps: float = 1e-8
    use_attention: bool = False
    attention_rho: float = 0.5
    min_batch_prior_js: float = 0.0
    max_baseline_margin: float = 1.0
    max_baseline_confidence: float = 1.0
    blend_weight: float = 1.0
    binary_only: bool = False
    estimator_params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class APAPrediction:
    proba_raw: np.ndarray
    proba_adj: np.ndarray
    pred: np.ndarray
    classes: np.ndarray
    context_indices: list[np.ndarray]
    effective_prior: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    runtime_sec: float
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class CounterfactualContextConfig:
    k: int = 32
    candidate_multiplier: int = 5
    lambda_balance: float = 0.35
    alpha: float = 0.75
    prior_floor: float = 0.08
    evidence_temperature: float = 1.0
    sensitivity_gate: float = 0.15
    max_abs_sensitivity: float = math.inf
    max_abs_sensitivity_low_batch_margin_only: bool = False
    disagreement_gate: float = 0.02
    max_disagreement: float = math.inf
    max_disagreement_low_batch_margin_only: bool = False
    max_baseline_margin: float = 1.0
    max_baseline_confidence: float = 0.92
    blend_weight: float = 0.65
    min_batch_prior_js: float = 0.0
    min_batch_pred_margin: float = 0.0
    min_local_prior_js: float = 0.0
    batch_sensitivity_rescue_gate: float = math.inf
    min_train_prior_imbalance: float = 0.0
    directional_shift_gate: bool = False
    local_directional_shift_gate: bool = False
    max_changed_fraction: float = 1.0
    off_majority_max_baseline_margin: float = 1.0
    include_knn: bool = True
    include_balanced: bool = True
    include_train_prior: bool = True
    one_vs_rest_strength: float = 0.70
    binary_only: bool = False
    aggregation: str = "stability_debias"


@dataclass(frozen=True)
class NoShiftFeatureConfig:
    k: int = 24
    candidate_multiplier: int = 5
    lambda_div: float = 0.30
    tau_clean: float = 0.65
    clean_neighbor_k: int = 12
    n_bootstrap: int = 3
    n_subspaces: int = 3
    subspace_fraction: float = 0.65
    max_batch_prior_js: float = 0.012
    max_baseline_margin: float = 0.35
    max_baseline_confidence: float = 0.90
    max_context_disagreement: float = 0.055
    min_context_vote_fraction: float = 0.67
    local_evidence_margin: float = 0.02
    blend_weight: float = 0.30
    max_changed_fraction: float = 0.03125
    max_minority_pred_train_ratio: float = math.inf
    require_candidate_minority: bool = False
    min_minority_distance_advantage: float = -math.inf
    minority_distance_k: int = 5
    families: tuple[str, ...] = (
        "diverse_knn",
        "boundary_contrast",
        "prototype",
        "clean_knn",
        "bootstrap_knn",
        "feature_subspace_knn",
    )
    random_state: int = 0


def _as_float_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    return X.astype(np.float32, copy=False)


def prepare_distance_matrix(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_train = _as_float_matrix(X_train)
    X_test = _as_float_matrix(X_test)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_f = scaler.transform(imputer.transform(X_test))
    return X_train_f.astype(np.float32), X_test_f.astype(np.float32)


def _fit_neighbors(X_train_scaled: np.ndarray, n_neighbors: int) -> NearestNeighbors:
    n_neighbors = min(max(1, n_neighbors), X_train_scaled.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train_scaled)
    return nn


def select_knn_contexts(X_train: np.ndarray, X_test: np.ndarray, k: int) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    nn = _fit_neighbors(X_train_s, k)
    return [row.astype(int) for row in nn.kneighbors(X_test_s, return_distance=False)]


def select_balanced_knn_contexts(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    per_class = max(1, int(math.ceil(k / max(1, len(classes)))))
    class_indices = {cls: np.flatnonzero(y_train == cls) for cls in classes}
    contexts: list[np.ndarray] = []

    for x in X_test_s:
        chosen: list[int] = []
        dists: list[float] = []
        for cls in classes:
            idx = class_indices[cls]
            if len(idx) == 0:
                continue
            cls_dist = np.linalg.norm(X_train_s[idx] - x, axis=1)
            take_order = np.argsort(cls_dist)[: min(per_class, len(idx))]
            chosen.extend(idx[take_order].tolist())
            dists.extend(cls_dist[take_order].tolist())
        order = np.argsort(np.asarray(dists, dtype=np.float64))[:k]
        contexts.append(np.asarray(chosen, dtype=int)[order])
    return contexts


def select_prior_aware_contexts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    candidate_multiplier: int = 5,
    lambda_balance: float = 0.25,
    target_prior: np.ndarray | None = None,
) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    y_train = np.asarray(y_train, dtype=int)
    classes = np.unique(y_train)
    n_classes = len(classes)
    if target_prior is None:
        target_prior = np.full(n_classes, 1.0 / max(1, n_classes), dtype=np.float64)
    target_prior = np.asarray(target_prior, dtype=np.float64)
    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    y_pos = np.asarray([class_to_pos[label] for label in y_train], dtype=int)

    m = min(X_train_s.shape[0], max(k, int(candidate_multiplier * k)))
    nn = _fit_neighbors(X_train_s, m)
    distances, indices = nn.kneighbors(X_test_s, return_distance=True)
    contexts: list[np.ndarray] = []

    for cand_dist, cand_idx in zip(distances, indices):
        cand_dist = np.asarray(cand_dist, dtype=np.float64)
        if cand_dist.max() > cand_dist.min():
            norm_dist = (cand_dist - cand_dist.min()) / (cand_dist.max() - cand_dist.min())
        else:
            norm_dist = np.zeros_like(cand_dist)

        selected: list[int] = []
        used = np.zeros(len(cand_idx), dtype=bool)
        counts = np.zeros(n_classes, dtype=np.float64)
        for _ in range(min(k, len(cand_idx))):
            denom = max(1, len(selected))
            current_prior = counts / denom if selected else np.zeros(n_classes, dtype=np.float64)
            rarity = target_prior - current_prior
            labels = y_pos[cand_idx]
            score = -norm_dist + lambda_balance * rarity[labels]
            score[used] = -np.inf
            best_local = int(np.argmax(score))
            used[best_local] = True
            selected_idx = int(cand_idx[best_local])
            selected.append(selected_idx)
            counts[y_pos[selected_idx]] += 1.0
        contexts.append(np.asarray(selected, dtype=int))
    return contexts



def _unique_fill_context(primary: list[int], fill: np.ndarray, k: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for idx in primary:
        idx = int(idx)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= k:
            break
    for idx in np.asarray(fill, dtype=int).tolist():
        idx = int(idx)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
        if len(selected) >= k:
            break
    return np.asarray(selected, dtype=int)


def select_diverse_knn_contexts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    candidate_multiplier: int = 5,
    lambda_div: float = 0.30,
) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    m = min(X_train_s.shape[0], max(k, int(candidate_multiplier * k)))
    nn = _fit_neighbors(X_train_s, m)
    distances, indices = nn.kneighbors(X_test_s, return_distance=True)
    contexts: list[np.ndarray] = []
    for cand_dist, cand_idx in zip(distances, indices):
        selected_local: list[int] = [0]
        used = np.zeros(len(cand_idx), dtype=bool)
        used[0] = True
        if cand_dist.max() > cand_dist.min():
            query_score = (cand_dist - cand_dist.min()) / (cand_dist.max() - cand_dist.min())
        else:
            query_score = np.zeros_like(cand_dist)
        while len(selected_local) < min(k, len(cand_idx)):
            selected_train_idx = cand_idx[np.asarray(selected_local, dtype=int)]
            pair_dist = np.linalg.norm(
                X_train_s[cand_idx][:, None, :] - X_train_s[selected_train_idx][None, :, :],
                axis=2,
            )
            min_selected_dist = pair_dist.min(axis=1)
            if min_selected_dist.max() > min_selected_dist.min():
                div_score = (min_selected_dist - min_selected_dist.min()) / (
                    min_selected_dist.max() - min_selected_dist.min()
                )
            else:
                div_score = np.zeros_like(min_selected_dist)
            score = -query_score + lambda_div * div_score
            score[used] = -np.inf
            best = int(np.argmax(score))
            used[best] = True
            selected_local.append(best)
        contexts.append(cand_idx[np.asarray(selected_local, dtype=int)].astype(int))
    return contexts


def select_boundary_contrast_contexts(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    baseline_proba: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    y_train_encoded = np.asarray(y_train_encoded, dtype=int)
    classes = np.unique(y_train_encoded)
    plain_knn = select_knn_contexts(X_train, X_test, k)
    contexts: list[np.ndarray] = []
    for row, x, fill in zip(np.asarray(baseline_proba, dtype=np.float64), X_test_s, plain_knn):
        order = np.argsort(row)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        primary: list[int] = []
        per_side = max(1, k // 4)
        for cls_pos in [top1, top2]:
            if cls_pos not in classes:
                continue
            idx = np.flatnonzero(y_train_encoded == cls_pos)
            if len(idx) == 0:
                continue
            dist = np.linalg.norm(X_train_s[idx] - x, axis=1)
            primary.extend(idx[np.argsort(dist)[: min(per_side, len(idx))]].tolist())
        contexts.append(_unique_fill_context(primary, fill, k))
    return contexts


def select_minority_contrast_contexts(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    baseline_proba: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    y_train_encoded = np.asarray(y_train_encoded, dtype=int)
    classes, counts = np.unique(y_train_encoded, return_counts=True)
    if len(classes) <= 1:
        return select_knn_contexts(X_train, X_test, k)
    minority_cls = int(classes[int(np.argmin(counts))])
    plain_knn = select_knn_contexts(X_train, X_test, k)
    contexts: list[np.ndarray] = []
    minority_take = max(2, k // 2)
    baseline_take = max(1, k // 4)
    for row, x, fill in zip(np.asarray(baseline_proba, dtype=np.float64), X_test_s, plain_knn):
        baseline_cls = int(np.argmax(row))
        primary: list[int] = []
        for cls_pos, take_count in [(minority_cls, minority_take), (baseline_cls, baseline_take)]:
            idx = np.flatnonzero(y_train_encoded == cls_pos)
            if len(idx) == 0:
                continue
            dist = np.linalg.norm(X_train_s[idx] - x, axis=1)
            primary.extend(idx[np.argsort(dist)[: min(take_count, len(idx))]].tolist())
        contexts.append(_unique_fill_context(primary, fill, k))
    return contexts


def select_prototype_contexts(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    y_train_encoded = np.asarray(y_train_encoded, dtype=int)
    classes = np.unique(y_train_encoded)
    per_class = max(1, int(math.ceil(k / max(1, len(classes)))))
    prototype_candidates: dict[int, np.ndarray] = {}
    for cls in classes:
        idx = np.flatnonzero(y_train_encoded == cls)
        if len(idx) == 0:
            continue
        center = np.mean(X_train_s[idx], axis=0)
        dist_to_center = np.linalg.norm(X_train_s[idx] - center, axis=1)
        take = np.argsort(dist_to_center)[: min(max(per_class * 3, per_class), len(idx))]
        prototype_candidates[int(cls)] = idx[take]
    plain_knn = select_knn_contexts(X_train, X_test, k)
    contexts: list[np.ndarray] = []
    for x, fill in zip(X_test_s, plain_knn):
        primary: list[int] = []
        for cls in classes:
            candidates = prototype_candidates.get(int(cls))
            if candidates is None or len(candidates) == 0:
                continue
            dist = np.linalg.norm(X_train_s[candidates] - x, axis=1)
            primary.extend(candidates[np.argsort(dist)[: min(per_class, len(candidates))]].tolist())
        contexts.append(_unique_fill_context(primary, fill, k))
    return contexts


def select_clean_knn_contexts(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    k: int,
    tau_clean: float = 0.65,
    clean_neighbor_k: int = 12,
) -> list[np.ndarray]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    y_train_encoded = np.asarray(y_train_encoded, dtype=int)
    train_k = min(max(2, clean_neighbor_k + 1), len(y_train_encoded))
    train_nn = _fit_neighbors(X_train_s, train_k)
    train_neighbors = train_nn.kneighbors(X_train_s, return_distance=False)
    agreement = np.zeros(len(y_train_encoded), dtype=np.float64)
    for i, row in enumerate(train_neighbors):
        row = row[row != i]
        if len(row) == 0:
            agreement[i] = 1.0
        else:
            agreement[i] = float(np.mean(y_train_encoded[row] == y_train_encoded[i]))
    clean = agreement >= tau_clean
    m = min(X_train_s.shape[0], max(k, int(5 * k)))
    nn = _fit_neighbors(X_train_s, m)
    indices = nn.kneighbors(X_test_s, return_distance=False)
    contexts: list[np.ndarray] = []
    for cand_idx in indices:
        primary = [int(idx) for idx in cand_idx if clean[int(idx)]]
        contexts.append(_unique_fill_context(primary, cand_idx, k))
    return contexts


def select_bootstrap_knn_context_groups(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    candidate_multiplier: int = 5,
    n_bootstrap: int = 3,
    random_state: int = 0,
) -> list[tuple[str, list[np.ndarray], np.ndarray | None]]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    m = min(X_train_s.shape[0], max(k, int(candidate_multiplier * k)))
    nn = _fit_neighbors(X_train_s, m)
    indices = nn.kneighbors(X_test_s, return_distance=False)
    groups: list[tuple[str, list[np.ndarray], np.ndarray | None]] = []
    for boot_id in range(n_bootstrap):
        rng = np.random.default_rng(random_state + 7919 * (boot_id + 1))
        contexts: list[np.ndarray] = []
        for cand_idx in indices:
            if len(cand_idx) <= k:
                contexts.append(cand_idx.astype(int))
                continue
            always = cand_idx[: max(1, k // 4)].tolist()
            rest = np.asarray([idx for idx in cand_idx if idx not in set(always)], dtype=int)
            take = min(k - len(always), len(rest))
            sampled = rng.choice(rest, size=take, replace=False).tolist() if take > 0 else []
            contexts.append(_unique_fill_context(always + sampled, cand_idx, k))
        groups.append((f"bootstrap_knn_{boot_id}", contexts, None))
    return groups


def select_feature_subspace_knn_context_groups(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    n_subspaces: int = 3,
    subspace_fraction: float = 0.65,
    random_state: int = 0,
) -> list[tuple[str, list[np.ndarray], np.ndarray | None]]:
    X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
    n_features = X_train_s.shape[1]
    if n_features <= 1:
        return [("feature_subspace_all", select_knn_contexts(X_train, X_test, k), None)]
    width = min(n_features, max(1, int(math.ceil(subspace_fraction * n_features))))
    variances = np.var(X_train_s, axis=0)
    masks: list[np.ndarray] = [np.arange(n_features, dtype=int)]
    masks.append(np.argsort(variances)[::-1][:width].astype(int))
    rng = np.random.default_rng(random_state + 104729)
    while len(masks) < max(1, n_subspaces):
        masks.append(np.sort(rng.choice(np.arange(n_features), size=width, replace=False)).astype(int))
    groups: list[tuple[str, list[np.ndarray], np.ndarray | None]] = []
    for pos, mask in enumerate(masks[:n_subspaces]):
        nn = _fit_neighbors(X_train_s[:, mask], k)
        contexts = [row.astype(int) for row in nn.kneighbors(X_test_s[:, mask], return_distance=False)]
        groups.append((f"feature_subspace_knn_{pos}", contexts, None))
    return groups


def select_feature_evidence_context_groups(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    baseline_proba: np.ndarray,
    config: NoShiftFeatureConfig,
) -> list[tuple[str, list[np.ndarray], np.ndarray | None]]:
    groups: list[tuple[str, list[np.ndarray], np.ndarray | None]] = []
    families = set(config.families)
    if "diverse_knn" in families:
        groups.append((
            "diverse_knn",
            select_diverse_knn_contexts(
                X_train,
                X_test,
                config.k,
                candidate_multiplier=config.candidate_multiplier,
                lambda_div=config.lambda_div,
            ),
            None,
        ))
    if "boundary_contrast" in families:
        groups.append((
            "boundary_contrast",
            select_boundary_contrast_contexts(X_train, y_train_encoded, X_test, baseline_proba, config.k),
            None,
        ))
    if "minority_contrast" in families:
        groups.append((
            "minority_contrast",
            select_minority_contrast_contexts(X_train, y_train_encoded, X_test, baseline_proba, config.k),
            None,
        ))
    if "prototype" in families:
        groups.append(("prototype", select_prototype_contexts(X_train, y_train_encoded, X_test, config.k), None))
    if "clean_knn" in families:
        groups.append((
            "clean_knn",
            select_clean_knn_contexts(
                X_train,
                y_train_encoded,
                X_test,
                config.k,
                tau_clean=config.tau_clean,
                clean_neighbor_k=config.clean_neighbor_k,
            ),
            None,
        ))
    if "bootstrap_knn" in families:
        groups.extend(
            select_bootstrap_knn_context_groups(
                X_train,
                X_test,
                config.k,
                candidate_multiplier=config.candidate_multiplier,
                n_bootstrap=config.n_bootstrap,
                random_state=config.random_state,
            )
        )
    if "feature_subspace_knn" in families:
        groups.extend(
            select_feature_subspace_knn_context_groups(
                X_train,
                X_test,
                config.k,
                n_subspaces=config.n_subspaces,
                subspace_fraction=config.subspace_fraction,
                random_state=config.random_state,
            )
        )
    if not groups:
        groups.append(("knn", select_knn_contexts(X_train, X_test, config.k), None))
    return groups


def _normalize_prior(prior: np.ndarray, floor: float = 0.0) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    prior = np.clip(prior, floor, None)
    total = prior.sum()
    if total <= 0:
        return np.full_like(prior, 1.0 / max(1, len(prior)), dtype=np.float64)
    return prior / total


def counterfactual_target_priors(
    y_train_encoded: np.ndarray,
    n_classes: int,
    alpha: float = 0.75,
    floor: float = 0.08,
    include_train_prior: bool = True,
    one_vs_rest_strength: float = 0.70,
) -> list[tuple[str, np.ndarray]]:
    """Build intervention priors used to probe context-prior sensitivity."""

    targets: list[tuple[str, np.ndarray]] = []
    uniform = np.full(n_classes, 1.0 / max(1, n_classes), dtype=np.float64)
    targets.append(("uniform", uniform))
    if include_train_prior:
        targets.append(("train_prior", context_prior(y_train_encoded, n_classes, alpha=alpha)))

    if n_classes == 2:
        targets.append(("tilt_class_0", _normalize_prior(np.array([1.0 - floor, floor]), floor=floor)))
        targets.append(("tilt_class_1", _normalize_prior(np.array([floor, 1.0 - floor]), floor=floor)))
    else:
        rest = (1.0 - one_vs_rest_strength) / max(1, n_classes - 1)
        for cls_pos in range(n_classes):
            prior = np.full(n_classes, rest, dtype=np.float64)
            prior[cls_pos] = one_vs_rest_strength
            targets.append((f"tilt_class_{cls_pos}", _normalize_prior(prior, floor=floor)))

    deduped: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, prior in targets:
        normalized = _normalize_prior(prior, floor=floor)
        key = tuple(np.round(normalized, 6).tolist())
        if key not in seen:
            seen.add(key)
            deduped.append((name, normalized))
    return deduped


def select_counterfactual_context_groups(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    config: CounterfactualContextConfig,
    n_classes: int,
) -> list[tuple[str, list[np.ndarray], np.ndarray | None]]:
    """Construct multiple contexts whose label priors act as interventions."""

    groups: list[tuple[str, list[np.ndarray], np.ndarray | None]] = []
    if config.include_knn:
        groups.append(("knn", select_knn_contexts(X_train, X_test, config.k), None))
    if config.include_balanced:
        groups.append(("balanced", select_balanced_knn_contexts(X_train, y_train_encoded, X_test, config.k), None))

    for name, target in counterfactual_target_priors(
        y_train_encoded,
        n_classes,
        alpha=config.alpha,
        floor=config.prior_floor,
        include_train_prior=config.include_train_prior,
        one_vs_rest_strength=config.one_vs_rest_strength,
    ):
        contexts = select_prior_aware_contexts(
            X_train,
            y_train_encoded,
            X_test,
            k=config.k,
            candidate_multiplier=config.candidate_multiplier,
            lambda_balance=config.lambda_balance,
            target_prior=target,
        )
        groups.append((f"target_{name}", contexts, target))
    return groups


def select_contexts(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test: np.ndarray,
    config: APAConfig,
) -> list[np.ndarray]:
    if config.context_strategy == "full":
        return [np.arange(len(y_train_encoded), dtype=int) for _ in range(len(X_test))]
    if config.context_strategy == "knn":
        return select_knn_contexts(X_train, X_test, config.k)
    if config.context_strategy == "balanced_knn":
        return select_balanced_knn_contexts(X_train, y_train_encoded, X_test, config.k)
    if config.context_strategy == "prior_aware_knn":
        return select_prior_aware_contexts(
            X_train,
            y_train_encoded,
            X_test,
            k=config.k,
            candidate_multiplier=config.candidate_multiplier,
            lambda_balance=config.lambda_balance,
        )
    raise ValueError(f"Unknown context_strategy={config.context_strategy!r}")


def context_prior(y_context: np.ndarray, n_classes: int, alpha: float = 1.0) -> np.ndarray:
    counts = np.bincount(np.asarray(y_context, dtype=int), minlength=n_classes).astype(np.float64)
    return (counts + alpha) / (counts.sum() + n_classes * alpha)


def effective_priors(
    y_train_encoded: np.ndarray,
    contexts: Iterable[np.ndarray],
    n_classes: int,
    alpha: float,
    attention_weights: np.ndarray | None = None,
    rho: float = 0.5,
) -> np.ndarray:
    priors: list[np.ndarray] = []
    y_train_encoded = np.asarray(y_train_encoded, dtype=int)
    for i, idx in enumerate(contexts):
        p_context = context_prior(y_train_encoded[idx], n_classes, alpha=alpha)
        if attention_weights is None:
            priors.append(p_context)
            continue
        weights = np.asarray(attention_weights[i], dtype=np.float64)
        if weights.shape[0] != len(idx):
            raise ValueError("attention weight length must match context length")
        counts = np.zeros(n_classes, dtype=np.float64)
        np.add.at(counts, y_train_encoded[idx], weights)
        p_att = (counts + alpha) / (weights.sum() + n_classes * alpha)
        priors.append(rho * p_att + (1.0 - rho) * p_context)
    return np.vstack(priors)


def query_distribution(proba: np.ndarray, mode: str = "hybrid", eta: float = 0.5) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if mode == "single":
        return proba
    batch = np.mean(proba, axis=0, keepdims=True)
    if mode == "batch":
        return np.repeat(batch, proba.shape[0], axis=0)
    if mode == "hybrid":
        return eta * proba + (1.0 - eta) * batch
    raise ValueError(f"Unknown query_mode={mode!r}")


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def adaptive_posterior_correction(
    proba: np.ndarray,
    p_eff: np.ndarray,
    query_mode: str = "hybrid",
    eta: float = 0.5,
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
    eps: float = 1e-8,
    fixed_gamma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proba = np.asarray(proba, dtype=np.float64)
    p_eff = np.asarray(p_eff, dtype=np.float64)
    q = query_distribution(proba, mode=query_mode, eta=eta)
    delta = js_divergence(q, p_eff, eps=eps) / np.log(2.0)
    if fixed_gamma is None:
        gamma = gamma_min + (gamma_max - gamma_min) * delta
        gamma = np.clip(gamma, gamma_min, gamma_max)
    else:
        gamma = np.full(proba.shape[0], float(fixed_gamma), dtype=np.float64)
    ratio = (q + eps) / (p_eff + eps)
    adjusted = proba * np.power(ratio, gamma[:, None])
    adjusted = np.clip(adjusted, eps, None)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, q, delta, gamma


def correct_existing_proba_with_context_prior(
    proba: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    config: APAConfig,
) -> APAPrediction:
    start = time.time()
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)
    y_encoded = label_encoder.transform(y_train)
    contexts = select_contexts(X_train, y_encoded, X_test, config)
    p_eff = effective_priors(y_encoded, contexts, len(classes), alpha=config.alpha)

    if config.binary_only and len(classes) != 2:
        proba_adj = np.asarray(proba, dtype=np.float64).copy()
        zeros = np.zeros(proba_adj.shape[0], dtype=np.float64)
        pred = classes[np.argmax(proba_adj, axis=1)]
        return APAPrediction(
            proba_raw=np.asarray(proba, dtype=np.float64),
            proba_adj=proba_adj,
            pred=pred,
            classes=np.asarray(classes),
            context_indices=contexts,
            effective_prior=p_eff,
            gamma=zeros,
            delta=zeros,
            runtime_sec=time.time() - start,
        )

    if config.correction == "local":
        proba_adj, _, delta, gamma = adaptive_posterior_correction(
            proba,
            p_eff,
            query_mode="single",
            eta=1.0,
            gamma_min=1.0,
            gamma_max=1.0,
            eps=config.eps,
            fixed_gamma=1.0,
        )
    elif config.correction == "adaptive":
        proba_adj, _, delta, gamma = adaptive_posterior_correction(
            proba,
            p_eff,
            query_mode=config.query_mode,
            eta=config.eta,
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
            eps=config.eps,
        )
    elif config.correction == "none":
        proba_adj = np.asarray(proba, dtype=np.float64).copy()
        delta = np.zeros(proba_adj.shape[0], dtype=np.float64)
        gamma = np.zeros(proba_adj.shape[0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown correction={config.correction!r}")

    if config.blend_weight < 1.0:
        w = float(np.clip(config.blend_weight, 0.0, 1.0))
        proba_adj = (1.0 - w) * np.asarray(proba, dtype=np.float64) + w * proba_adj
        proba_adj /= proba_adj.sum(axis=1, keepdims=True)

    if (
        config.min_batch_prior_js > 0.0
        or config.max_baseline_margin < 1.0
        or config.max_baseline_confidence < 1.0
    ):
        train_prior = context_prior(y_encoded, len(classes), alpha=config.alpha).reshape(1, -1)
        batch_prior = np.mean(np.asarray(proba, dtype=np.float64), axis=0, keepdims=True)
        batch_shift = float(js_divergence(batch_prior, train_prior, eps=config.eps)[0] / np.log(2.0))
        sorted_proba = np.sort(np.asarray(proba, dtype=np.float64), axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else np.ones(proba.shape[0])
        confidence = np.max(np.asarray(proba, dtype=np.float64), axis=1)
        eligible = (
            (batch_shift >= config.min_batch_prior_js)
            & (margin <= config.max_baseline_margin)
            & (confidence <= config.max_baseline_confidence)
        )
        proba_adj = np.where(eligible[:, None], proba_adj, np.asarray(proba, dtype=np.float64))

    pred = classes[np.argmax(proba_adj, axis=1)]
    return APAPrediction(
        proba_raw=np.asarray(proba, dtype=np.float64),
        proba_adj=proba_adj,
        pred=pred,
        classes=np.asarray(classes),
        context_indices=contexts,
        effective_prior=p_eff,
        gamma=gamma,
        delta=delta,
        runtime_sec=time.time() - start,
    )


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    pred_idx = np.argmax(proba, axis=1)
    pred = classes[pred_idx]
    confidence = np.max(proba, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if lo == 0.0:
            mask = (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(pred[mask] == y_true[mask])
        conf = np.mean(confidence[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict[str, object]:
    pred = classes[np.argmax(proba, axis=1)]
    recalls = recall_score(y_true, pred, labels=classes, average=None, zero_division=0)
    class_counts = np.asarray([np.sum(y_true == cls) for cls in classes], dtype=int)
    minority_recall = float(recalls[np.argmin(class_counts)]) if len(recalls) else 0.0
    roc_auc: float | str = ""
    try:
        class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
        y_pos = np.asarray([class_to_pos[label] for label in y_true], dtype=int)
        if len(np.unique(y_pos)) >= 2:
            if len(classes) == 2:
                roc_auc = float(roc_auc_score(y_pos, np.asarray(proba, dtype=np.float64)[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_pos,
                        np.asarray(proba, dtype=np.float64),
                        labels=np.arange(len(classes)),
                        multi_class="ovr",
                        average="macro",
                    )
                )
    except Exception:
        roc_auc = ""
    row: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "minority_recall": minority_recall,
        "roc_auc": roc_auc,
        "ece": expected_calibration_error(y_true, proba, classes),
    }
    for cls, recall in zip(classes, recalls):
        row[f"recall_class_{cls}"] = float(recall)
    return row


def make_estimator(factory: Callable[[], object], params: dict[str, object]) -> object:
    estimator = factory()
    if params:
        estimator.set_params(**params)
    return estimator


def batch_prior_js_score(
    proba: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> float:
    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    y_pos = np.asarray([class_to_pos[label] for label in y_train], dtype=int)
    train_prior = context_prior(y_pos, len(classes), alpha=alpha).reshape(1, -1)
    batch_prior = np.mean(np.asarray(proba, dtype=np.float64), axis=0, keepdims=True)
    return float(js_divergence(batch_prior, train_prior, eps=eps)[0] / np.log(2.0))


def apa_v7_config_for_proba(
    proba: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    k: int = 30,
) -> tuple[APAConfig | None, str, float]:
    if len(classes) != 2:
        return None, "baseline_nonbinary", float("nan")
    js_value = batch_prior_js_score(proba, y_train, classes, alpha=1.0)
    if js_value >= 0.05:
        return (
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
                binary_only=True,
            ),
            "high_js_v2",
            js_value,
        )
    if js_value >= 0.008:
        return (
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
                min_batch_prior_js=0.008,
                max_baseline_margin=0.55,
                max_baseline_confidence=0.90,
                blend_weight=1.0,
                binary_only=True,
            ),
            "hard_js_v7",
            js_value,
        )
    return None, "baseline_low_js", js_value


def apa_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    estimator_factory: Callable[[], object],
    config: APAConfig,
) -> APAPrediction:
    start = time.time()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    classes = label_encoder.classes_
    contexts = select_contexts(X_train, y_encoded, X_test, config)

    raw_rows: list[np.ndarray] = []
    for idx, x in zip(contexts, X_test):
        estimator = make_estimator(estimator_factory, config.estimator_params)
        estimator.fit(X_train[idx], y_train[idx])
        raw = estimator.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
        aligned = np.zeros(len(classes), dtype=np.float64)
        est_classes = np.asarray(estimator.classes_)
        for pos, cls in enumerate(est_classes):
            global_pos = int(np.flatnonzero(classes == cls)[0])
            aligned[global_pos] = raw[pos]
        if aligned.sum() <= 0:
            aligned[:] = 1.0 / len(classes)
        else:
            aligned /= aligned.sum()
        raw_rows.append(aligned)

    proba_raw = np.vstack(raw_rows)
    p_eff = effective_priors(y_encoded, contexts, len(classes), alpha=config.alpha)
    if config.correction == "none":
        proba_adj = proba_raw.copy()
        delta = np.zeros(proba_raw.shape[0], dtype=np.float64)
        gamma = np.zeros(proba_raw.shape[0], dtype=np.float64)
    elif config.correction == "local":
        proba_adj, _, delta, gamma = adaptive_posterior_correction(
            proba_raw,
            p_eff,
            query_mode="single",
            eta=1.0,
            gamma_min=1.0,
            gamma_max=1.0,
            eps=config.eps,
            fixed_gamma=1.0,
        )
    elif config.correction == "adaptive":
        proba_adj, _, delta, gamma = adaptive_posterior_correction(
            proba_raw,
            p_eff,
            query_mode=config.query_mode,
            eta=config.eta,
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
            eps=config.eps,
        )
    else:
        raise ValueError(f"Unknown correction={config.correction!r}")

    pred = classes[np.argmax(proba_adj, axis=1)]
    return APAPrediction(
        proba_raw=proba_raw,
        proba_adj=proba_adj,
        pred=pred,
        classes=classes,
        context_indices=contexts,
        effective_prior=p_eff,
        gamma=gamma,
        delta=delta,
        runtime_sec=time.time() - start,
    )


def baseline_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    estimator_factory: Callable[[], object],
    estimator_params: dict[str, object] | None = None,
) -> APAPrediction:
    start = time.time()
    estimator = make_estimator(estimator_factory, estimator_params or {})
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)
    classes = np.asarray(estimator.classes_)
    pred = classes[np.argmax(proba, axis=1)]
    contexts = [np.arange(len(y_train), dtype=int) for _ in range(len(X_test))]
    le = LabelEncoder().fit(y_train)
    y_encoded = le.transform(y_train)
    p_eff = effective_priors(y_encoded, contexts, len(classes), alpha=1.0)
    zeros = np.zeros(len(X_test), dtype=np.float64)
    return APAPrediction(
        proba_raw=proba,
        proba_adj=proba,
        pred=pred,
        classes=classes,
        context_indices=contexts,
        effective_prior=p_eff,
        gamma=zeros,
        delta=zeros,
        runtime_sec=time.time() - start,
    )


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64) / max(float(temperature), EPS)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _entropy(proba: np.ndarray, eps: float = EPS) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0)
    return -np.sum(proba * np.log(proba), axis=1)


def prior_sensitivity_debias(
    context_probas: np.ndarray,
    context_priors: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate prior-invariant evidence from counterfactual context outputs.

    For each query, fit a pooled scalar sensitivity beta in:

        log p_m(c) ~= evidence_c + beta * log pi_m(c)

    The returned prediction is softmax(evidence). This treats context label
    priors as intervention variables instead of applying a fixed prior-ratio
    correction to one posterior.
    """

    probas = np.clip(np.asarray(context_probas, dtype=np.float64), eps, 1.0)
    priors = np.clip(np.asarray(context_priors, dtype=np.float64), eps, 1.0)
    log_p = np.log(probas)
    log_prior = np.log(priors)
    n_test, _, n_classes = log_p.shape
    evidence = np.zeros((n_test, n_classes), dtype=np.float64)
    sensitivity = np.zeros(n_test, dtype=np.float64)

    for i in range(n_test):
        x = log_prior[i].reshape(-1)
        z = log_p[i].reshape(-1)
        x_centered = x - x.mean()
        z_centered = z - z.mean()
        denom = float(np.dot(x_centered, x_centered))
        beta = float(np.dot(x_centered, z_centered) / denom) if denom > eps else 0.0
        beta = float(np.clip(beta, -2.0, 2.0))
        sensitivity[i] = beta
        evidence[i] = np.mean(log_p[i] - beta * log_prior[i], axis=0)

    return _softmax(evidence, temperature=temperature), sensitivity


def entropy_weighted_context_ensemble(context_probas: np.ndarray) -> np.ndarray:
    probas = np.asarray(context_probas, dtype=np.float64)
    ent = _entropy(probas.reshape(-1, probas.shape[-1])).reshape(probas.shape[:2])
    weights = np.exp(-ent)
    weights /= weights.sum(axis=1, keepdims=True)
    ensembled = np.sum(weights[:, :, None] * probas, axis=1)
    ensembled /= ensembled.sum(axis=1, keepdims=True)
    return ensembled


def counterfactual_context_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    estimator_factory: Callable[[], object],
    config: CounterfactualContextConfig,
    baseline_proba: np.ndarray | None = None,
    baseline_classes: np.ndarray | None = None,
) -> APAPrediction:
    start = time.time()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    classes = label_encoder.classes_ if baseline_classes is None else np.asarray(baseline_classes)
    if config.binary_only and len(classes) != 2 and baseline_proba is not None:
        proba = np.asarray(baseline_proba, dtype=np.float64)
        zeros = np.zeros(proba.shape[0], dtype=np.float64)
        return APAPrediction(
            proba_raw=proba,
            proba_adj=proba,
            pred=classes[np.argmax(proba, axis=1)],
            classes=classes,
            context_indices=[np.arange(len(y_train), dtype=int) for _ in range(len(X_test))],
            effective_prior=np.repeat(context_prior(y_encoded, len(classes), config.alpha).reshape(1, -1), len(X_test), axis=0),
            gamma=zeros,
            delta=zeros,
            runtime_sec=time.time() - start,
        )

    groups = select_counterfactual_context_groups(X_train, y_encoded, X_test, config, len(classes))
    all_probas: list[np.ndarray] = []
    all_priors: list[np.ndarray] = []
    reusable_estimator: object | None = None
    failed_context_predictions = 0
    total_context_predictions = 0

    for _, contexts, _ in groups:
        rows: list[np.ndarray] = []
        priors = effective_priors(y_encoded, contexts, len(classes), alpha=config.alpha)
        for row_pos, (idx, x) in enumerate(zip(contexts, X_test)):
            total_context_predictions += 1
            try:
                if reusable_estimator is None:
                    reusable_estimator = make_estimator(estimator_factory, {})
                estimator = reusable_estimator
                estimator.fit(X_train[idx], y_train[idx])
                if hasattr(estimator, "model_") and hasattr(estimator, "_load_model"):
                    estimator._load_model = lambda: None  # type: ignore[attr-defined]
                raw = estimator.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
                aligned = np.zeros(len(classes), dtype=np.float64)
                est_classes = np.asarray(estimator.classes_)
                for pos, cls in enumerate(est_classes):
                    matches = np.flatnonzero(classes == cls)
                    if len(matches):
                        aligned[int(matches[0])] = raw[pos]
                if aligned.sum() <= 0:
                    aligned[:] = 1.0 / len(classes)
                else:
                    aligned /= aligned.sum()
            except Exception:
                failed_context_predictions += 1
                if baseline_proba is not None:
                    aligned = np.asarray(baseline_proba[row_pos], dtype=np.float64).copy()
                    aligned /= max(float(aligned.sum()), EPS)
                else:
                    aligned = np.full(len(classes), 1.0 / max(1, len(classes)), dtype=np.float64)
            rows.append(aligned)
        all_probas.append(np.vstack(rows))
        all_priors.append(priors)

    context_probas = np.stack(all_probas, axis=1)
    context_priors = np.stack(all_priors, axis=1)
    debiased, sensitivity = prior_sensitivity_debias(
        context_probas,
        context_priors,
        temperature=config.evidence_temperature,
    )
    ensemble = entropy_weighted_context_ensemble(context_probas)
    disagreement = np.mean(np.sum((context_probas - context_probas.mean(axis=1, keepdims=True)) ** 2, axis=2), axis=1)

    if config.aggregation == "entropy_ensemble":
        candidate = ensemble
    elif config.aggregation == "prior_sensitivity":
        candidate = debiased
    elif config.aggregation == "stability_debias":
        candidate = 0.5 * debiased + 0.5 * ensemble
        candidate /= candidate.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown counterfactual aggregation={config.aggregation!r}")

    if baseline_proba is None:
        proba_raw = ensemble
        proba_adj = candidate
        diagnostics: dict[str, np.ndarray] = {}
    else:
        proba_raw = np.asarray(baseline_proba, dtype=np.float64)
        confidence = np.max(proba_raw, axis=1)
        sorted_raw = np.sort(proba_raw, axis=1)
        baseline_margin = sorted_raw[:, -1] - sorted_raw[:, -2] if proba_raw.shape[1] > 1 else np.ones(proba_raw.shape[0])
        baseline_pred = np.argmax(proba_raw, axis=1)
        candidate_pred = np.argmax(candidate, axis=1)
        eligible = (
            (np.abs(sensitivity) >= config.sensitivity_gate)
            & (disagreement >= config.disagreement_gate)
            & (baseline_margin <= config.max_baseline_margin)
            & (confidence <= config.max_baseline_confidence)
        )
        eligible_before_batch = eligible.copy()
        train_prior = context_prior(y_encoded, len(classes), alpha=config.alpha)
        local_reference_prior = context_priors[:, 0, :] if context_priors.shape[1] else np.repeat(
            train_prior.reshape(1, -1), len(proba_raw), axis=0
        )
        local_prior_js_rows = js_divergence(
            local_reference_prior,
            np.repeat(train_prior.reshape(1, -1), len(proba_raw), axis=0),
        ) / np.log(2.0)
        batch_pred_margin = 0.0
        apply_sensitivity_cap = not config.max_abs_sensitivity_low_batch_margin_only
        apply_disagreement_cap = not config.max_disagreement_low_batch_margin_only
        if config.min_batch_prior_js > 0.0 or config.directional_shift_gate:
            batch_prior = np.mean(proba_raw, axis=0)
            batch_js = float(js_divergence(batch_prior.reshape(1, -1), train_prior.reshape(1, -1))[0] / np.log(2.0))
            if batch_js < config.min_batch_prior_js:
                eligible &= False
            if len(classes) > 1:
                batch_sorted = np.sort(batch_prior)
                batch_pred_margin = float(batch_sorted[-1] - batch_sorted[-2])
            low_batch_margin = batch_pred_margin < config.min_batch_pred_margin
            if config.max_abs_sensitivity_low_batch_margin_only:
                apply_sensitivity_cap = low_batch_margin
            if config.max_disagreement_low_batch_margin_only:
                apply_disagreement_cap = low_batch_margin
            if config.min_batch_pred_margin > 0.0 and len(classes) > 1:
                mean_abs_sensitivity = float(np.mean(np.abs(sensitivity)))
                if low_batch_margin and mean_abs_sensitivity < config.batch_sensitivity_rescue_gate:
                    eligible &= False
            if config.min_train_prior_imbalance > 0.0:
                uniform = np.full(len(classes), 1.0 / max(1, len(classes)), dtype=np.float64)
                train_imbalance = float(np.sum(np.abs(train_prior - uniform)))
                if train_imbalance < config.min_train_prior_imbalance:
                    eligible &= False
            if config.directional_shift_gate:
                shift_direction = batch_prior - train_prior
                moves_toward_shift = shift_direction[candidate_pred] > 0.0
                changes_decision = candidate_pred != baseline_pred
                target_is_batch_majority = batch_prior[candidate_pred] >= np.max(batch_prior)
                low_margin_off_majority = baseline_margin <= config.off_majority_max_baseline_margin
                eligible &= moves_toward_shift & changes_decision
                eligible &= target_is_batch_majority | low_margin_off_majority
        if config.min_local_prior_js > 0.0:
            eligible &= local_prior_js_rows >= config.min_local_prior_js
        if config.local_directional_shift_gate:
            local_direction = local_reference_prior - train_prior.reshape(1, -1)
            moves_toward_local = local_direction[np.arange(len(candidate_pred)), candidate_pred] > 0.0
            changes_decision = candidate_pred != baseline_pred
            local_majority = np.argmax(local_reference_prior, axis=1)
            target_is_local_majority = candidate_pred == local_majority
            low_margin_off_majority = baseline_margin <= config.off_majority_max_baseline_margin
            eligible &= moves_toward_local & changes_decision
            eligible &= target_is_local_majority | low_margin_off_majority
        if apply_sensitivity_cap:
            eligible &= np.abs(sensitivity) <= config.max_abs_sensitivity
        if apply_disagreement_cap:
            eligible &= disagreement <= config.max_disagreement
        if config.max_changed_fraction < 1.0 and np.any(eligible):
            max_changes = max(1, int(math.ceil(float(config.max_changed_fraction) * len(eligible))))
            score = np.abs(sensitivity) * disagreement
            keep = np.zeros_like(eligible, dtype=bool)
            eligible_idx = np.flatnonzero(eligible)
            chosen = eligible_idx[np.argsort(score[eligible_idx])[-max_changes:]]
            keep[chosen] = True
            eligible = keep
        blended = (1.0 - config.blend_weight) * proba_raw + config.blend_weight * candidate
        blended /= blended.sum(axis=1, keepdims=True)
        proba_adj = np.where(eligible[:, None], blended, proba_raw)
        diagnostics = {
            "baseline_margin": baseline_margin,
            "baseline_confidence": confidence,
            "baseline_pred": baseline_pred,
            "candidate_pred": candidate_pred,
            "eligible_before_batch": eligible_before_batch.astype(float),
            "eligible_final": eligible.astype(float),
            "batch_pred_margin": np.full(len(proba_raw), batch_pred_margin, dtype=np.float64),
            "local_prior_js": local_prior_js_rows,
            "context_failure_fraction": np.full(
                len(proba_raw),
                failed_context_predictions / max(1, total_context_predictions),
                dtype=np.float64,
            ),
        }

    pred = classes[np.argmax(proba_adj, axis=1)]
    mean_prior = np.mean(context_priors, axis=1)
    first_contexts = groups[0][1] if groups else [np.arange(len(y_train), dtype=int) for _ in range(len(X_test))]
    return APAPrediction(
        proba_raw=proba_raw,
        proba_adj=proba_adj,
        pred=pred,
        classes=classes,
        context_indices=first_contexts,
        effective_prior=mean_prior,
        gamma=sensitivity,
        delta=disagreement,
        runtime_sec=time.time() - start,
        diagnostics=diagnostics,
    )


def no_shift_feature_context_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    estimator_factory: Callable[[], object],
    config: NoShiftFeatureConfig,
    baseline_proba: np.ndarray,
    baseline_classes: np.ndarray,
) -> APAPrediction:
    start = time.time()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    classes = np.asarray(baseline_classes)
    proba_raw = np.asarray(baseline_proba, dtype=np.float64)
    n_test, n_classes = proba_raw.shape
    train_prior = context_prior(y_encoded, n_classes, alpha=1.0)
    batch_prior = np.mean(proba_raw, axis=0)
    batch_prior_js = float(js_divergence(batch_prior.reshape(1, -1), train_prior.reshape(1, -1))[0] / np.log(2.0))
    zeros = np.zeros(n_test, dtype=np.float64)

    if batch_prior_js > config.max_batch_prior_js:
        pred = classes[np.argmax(proba_raw, axis=1)]
        return APAPrediction(
            proba_raw=proba_raw,
            proba_adj=proba_raw,
            pred=pred,
            classes=classes,
            context_indices=[np.arange(len(y_train), dtype=int) for _ in range(n_test)],
            effective_prior=np.repeat(train_prior.reshape(1, -1), n_test, axis=0),
            gamma=zeros,
            delta=zeros,
            runtime_sec=time.time() - start,
            diagnostics={
                "no_shift_batch_prior_js": np.full(n_test, batch_prior_js, dtype=np.float64),
                "no_shift_eligible_final": zeros,
                "no_shift_vote_fraction": zeros,
                "no_shift_context_disagreement": zeros,
                "no_shift_changed_fraction": zeros,
            },
        )

    groups = select_feature_evidence_context_groups(X_train, y_encoded, X_test, proba_raw, config)
    all_probas: list[np.ndarray] = []
    all_priors: list[np.ndarray] = []
    reusable_estimator: object | None = None
    failed_context_predictions = 0
    total_context_predictions = 0
    for _, contexts, _ in groups:
        rows: list[np.ndarray] = []
        priors = effective_priors(y_encoded, contexts, n_classes, alpha=0.75)
        for row_pos, (idx, x) in enumerate(zip(contexts, X_test)):
            total_context_predictions += 1
            try:
                if reusable_estimator is None:
                    reusable_estimator = make_estimator(estimator_factory, {})
                estimator = reusable_estimator
                estimator.fit(X_train[idx], y_train[idx])
                if hasattr(estimator, "model_") and hasattr(estimator, "_load_model"):
                    estimator._load_model = lambda: None  # type: ignore[attr-defined]
                raw = estimator.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
                aligned = np.zeros(n_classes, dtype=np.float64)
                est_classes = np.asarray(estimator.classes_)
                for pos, cls in enumerate(est_classes):
                    matches = np.flatnonzero(classes == cls)
                    if len(matches):
                        aligned[int(matches[0])] = raw[pos]
                if aligned.sum() <= 0:
                    aligned[:] = 1.0 / n_classes
                else:
                    aligned /= aligned.sum()
            except Exception:
                failed_context_predictions += 1
                aligned = proba_raw[row_pos].copy()
                aligned /= max(float(aligned.sum()), EPS)
            rows.append(aligned)
        all_probas.append(np.vstack(rows))
        all_priors.append(priors)

    context_probas = np.stack(all_probas, axis=1)
    context_priors = np.stack(all_priors, axis=1)
    candidate = entropy_weighted_context_ensemble(context_probas)
    context_pred = np.argmax(context_probas, axis=2)
    candidate_pred = np.argmax(candidate, axis=1)
    baseline_pred = np.argmax(proba_raw, axis=1)
    vote_fraction = np.zeros(n_test, dtype=np.float64)
    for i in range(n_test):
        vote_fraction[i] = float(np.mean(context_pred[i] == candidate_pred[i]))
    disagreement = np.mean(np.sum((context_probas - context_probas.mean(axis=1, keepdims=True)) ** 2, axis=2), axis=1)
    confidence = np.max(proba_raw, axis=1)
    sorted_raw = np.sort(proba_raw, axis=1)
    baseline_margin = sorted_raw[:, -1] - sorted_raw[:, -2] if n_classes > 1 else np.ones(n_test)
    local_prior = np.mean(context_priors, axis=1)
    local_support_margin = local_prior[np.arange(n_test), candidate_pred] - local_prior[np.arange(n_test), baseline_pred]
    changes_decision = candidate_pred != baseline_pred
    minority_pred_train_ratio = math.inf
    minority_collapse_gate = True
    minority_pos = int(np.argmin(train_prior)) if n_classes > 1 else 0
    if np.isfinite(config.max_minority_pred_train_ratio) and n_classes > 1:
        baseline_pred_prior = np.bincount(baseline_pred, minlength=n_classes).astype(np.float64) / max(1, n_test)
        minority_pred_train_ratio = float(
            baseline_pred_prior[minority_pos] / max(float(train_prior[minority_pos]), EPS)
        )
        minority_collapse_gate = minority_pred_train_ratio <= config.max_minority_pred_train_ratio
    minority_distance_advantage = np.full(n_test, -math.inf, dtype=np.float64)
    if np.isfinite(config.min_minority_distance_advantage) and n_classes > 1:
        X_train_s, X_test_s = prepare_distance_matrix(X_train, X_test)
        minority_idx = np.flatnonzero(y_encoded == minority_pos)
        k_dist = max(1, int(config.minority_distance_k))
        for row_pos, x in enumerate(X_test_s):
            baseline_cls = int(baseline_pred[row_pos])
            baseline_idx = np.flatnonzero(y_encoded == baseline_cls)
            if len(minority_idx) == 0 or len(baseline_idx) == 0:
                continue
            min_dist = np.linalg.norm(X_train_s[minority_idx] - x, axis=1)
            base_dist = np.linalg.norm(X_train_s[baseline_idx] - x, axis=1)
            min_mean = float(np.mean(np.sort(min_dist)[: min(k_dist, len(min_dist))]))
            base_mean = float(np.mean(np.sort(base_dist)[: min(k_dist, len(base_dist))]))
            minority_distance_advantage[row_pos] = (base_mean - min_mean) / max(base_mean, EPS)
    eligible = (
        changes_decision
        & (baseline_margin <= config.max_baseline_margin)
        & (confidence <= config.max_baseline_confidence)
        & (vote_fraction >= config.min_context_vote_fraction)
        & (disagreement <= config.max_context_disagreement)
        & (local_support_margin >= config.local_evidence_margin)
    )
    if config.require_candidate_minority and n_classes > 1:
        eligible &= candidate_pred == minority_pos
        eligible &= baseline_pred != minority_pos
    if np.isfinite(config.max_minority_pred_train_ratio):
        eligible &= minority_collapse_gate
    if np.isfinite(config.min_minority_distance_advantage):
        eligible &= minority_distance_advantage >= config.min_minority_distance_advantage
    if config.max_changed_fraction < 1.0 and np.any(eligible):
        max_changes = max(1, int(math.ceil(float(config.max_changed_fraction) * n_test)))
        score = vote_fraction + local_support_margin - disagreement - baseline_margin
        keep = np.zeros_like(eligible, dtype=bool)
        eligible_idx = np.flatnonzero(eligible)
        chosen = eligible_idx[np.argsort(score[eligible_idx])[-max_changes:]]
        keep[chosen] = True
        eligible = keep

    blended = (1.0 - config.blend_weight) * proba_raw + config.blend_weight * candidate
    blended /= blended.sum(axis=1, keepdims=True)
    proba_adj = np.where(eligible[:, None], blended, proba_raw)
    pred = classes[np.argmax(proba_adj, axis=1)]
    changed_fraction = float(np.mean(np.argmax(proba_adj, axis=1) != baseline_pred))
    first_contexts = groups[0][1] if groups else [np.arange(len(y_train), dtype=int) for _ in range(n_test)]
    return APAPrediction(
        proba_raw=proba_raw,
        proba_adj=proba_adj,
        pred=pred,
        classes=classes,
        context_indices=first_contexts,
        effective_prior=local_prior,
        gamma=vote_fraction,
        delta=disagreement,
        runtime_sec=time.time() - start,
        diagnostics={
            "no_shift_batch_prior_js": np.full(n_test, batch_prior_js, dtype=np.float64),
            "no_shift_eligible_final": eligible.astype(float),
            "no_shift_vote_fraction": vote_fraction,
            "no_shift_context_disagreement": disagreement,
            "no_shift_local_support_margin": local_support_margin,
            "no_shift_changed_fraction": np.full(n_test, changed_fraction, dtype=np.float64),
            "no_shift_minority_pred_train_ratio": np.full(n_test, minority_pred_train_ratio, dtype=np.float64),
            "no_shift_minority_collapse_gate": np.full(n_test, float(minority_collapse_gate), dtype=np.float64),
            "no_shift_minority_distance_advantage": minority_distance_advantage,
            "no_shift_context_failure_fraction": np.full(
                n_test,
                failed_context_predictions / max(1, total_context_predictions),
                dtype=np.float64,
            ),
        },
    )


class APATabICLClassifier:
    """Inference-only APA wrapper around a sklearn-style classifier.

    The default `method="apa_v7"` keeps the base estimator's full-context
    probabilities, then applies the current APA v7 post-hoc router. Pass
    `estimator_factory=lambda: TabICLClassifier(...)` to use TabICL. Tests may
    pass any classifier with `fit`, `predict_proba`, and `classes_`.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], object],
        method: str = "apa_v7",
        k: int = 30,
        estimator_params: dict[str, object] | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.method = method
        self.k = k
        self.estimator_params = estimator_params or {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "APATabICLClassifier":
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y)
        self.base_estimator_ = make_estimator(self.estimator_factory, self.estimator_params)
        self.base_estimator_.fit(self.X_train_, self.y_train_)
        self.classes_ = np.asarray(self.base_estimator_.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "base_estimator_"):
            raise RuntimeError("APATabICLClassifier must be fitted before predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        proba = np.asarray(self.base_estimator_.predict_proba(X), dtype=np.float64)
        if self.method in {"baseline", "tabicl_baseline"}:
            return proba / proba.sum(axis=1, keepdims=True)
        if self.method != "apa_v7":
            raise ValueError(f"Unknown APA wrapper method {self.method!r}")
        config, branch, js_value = apa_v7_config_for_proba(proba, self.y_train_, self.classes_, k=self.k)
        self.last_branch_ = branch
        self.last_batch_prior_js_ = js_value
        if config is None:
            return proba / proba.sum(axis=1, keepdims=True)
        result = correct_existing_proba_with_context_prior(proba, self.X_train_, self.y_train_, X, self.classes_, config)
        self.last_prediction_ = result
        return result.proba_adj

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
