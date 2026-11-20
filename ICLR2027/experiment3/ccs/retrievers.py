from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re

import numpy as np

from .types import CausalSplit, ContextBatch
from .utils import standardize


class ContextRetriever(ABC):
    name: str

    @abstractmethod
    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        raise NotImplementedError

    def _batch(self, split: CausalSplit, indices: np.ndarray) -> ContextBatch:
        indices = np.asarray(indices, dtype=int)
        train = split.train
        return ContextBatch(
            indices=indices,
            X=train.X[indices],
            treatment=train.treatment[indices],
            outcome=train.outcome[indices],
            propensity=train.propensity[indices],
        )


def _pad_or_sample(indices: np.ndarray, context_size: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return indices
    if indices.size >= context_size:
        return rng.choice(indices, size=context_size, replace=False)
    extra = rng.choice(indices, size=context_size - indices.size, replace=True)
    return np.concatenate([indices, extra])


def _nearest_by_distance(distances: np.ndarray, context_size: int) -> np.ndarray:
    k = min(context_size, distances.size)
    if k <= 0:
        return np.asarray([], dtype=int)
    return np.argpartition(distances, kth=k - 1)[:k]


def _query_distances(split: CausalSplit, query_index: int) -> tuple[np.ndarray, np.ndarray]:
    X_train, X_test = standardize(split.train.X, split.test.X)
    feature_distance = np.sqrt(np.sum((X_train - X_test[query_index]) ** 2, axis=1))
    propensity_distance = np.abs(split.train.propensity - split.test.propensity[query_index])
    return feature_distance, propensity_distance


def _common_support_indices(split: CausalSplit, lower_q: float = 0.05, upper_q: float = 0.95) -> np.ndarray:
    p_train = split.train.propensity
    treated = split.train.treatment > 0.5
    if not np.any(treated) or not np.any(~treated):
        return np.arange(split.train.n_samples)
    lo = max(float(np.quantile(p_train[treated], lower_q)), float(np.quantile(p_train[~treated], lower_q)))
    hi = min(float(np.quantile(p_train[treated], upper_q)), float(np.quantile(p_train[~treated], upper_q)))
    support = np.flatnonzero((p_train >= lo) & (p_train <= hi))
    return support if support.size else np.arange(split.train.n_samples)


def _balanced_select(
    split: CausalSplit,
    score: np.ndarray,
    context_size: int,
    rng: np.random.Generator,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    if candidate_mask is None:
        candidate_mask = np.ones(split.train.n_samples, dtype=bool)
    treated = np.flatnonzero((split.train.treatment > 0.5) & candidate_mask)
    control = np.flatnonzero((split.train.treatment <= 0.5) & candidate_mask)
    pair_count = max(1, context_size // 2)
    selected_parts = []
    for pool in [treated, control]:
        if pool.size:
            selected_parts.append(pool[np.argsort(score[pool])[:pair_count]])
    selected = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=int)
    selected = selected[:context_size]
    if selected.size < context_size:
        fallback_pool = np.flatnonzero(candidate_mask)
        if fallback_pool.size == 0:
            fallback_pool = np.arange(split.train.n_samples)
        fallback = fallback_pool[np.argsort(score[fallback_pool])[:context_size]]
        selected = np.unique(np.concatenate([selected, fallback]))[:context_size]
    if selected.size < context_size:
        selected = np.concatenate([selected, rng.choice(split.train.n_samples, size=context_size - selected.size, replace=True)])
    return selected


def _greedy_diverse_select(
    split: CausalSplit,
    score: np.ndarray,
    X_train: np.ndarray,
    context_size: int,
    rng: np.random.Generator,
    candidate_mask: np.ndarray | None = None,
    diversity_weight: float = 0.25,
) -> np.ndarray:
    if candidate_mask is None:
        candidate_mask = np.ones(split.train.n_samples, dtype=bool)
    selected = []
    pair_count = max(1, context_size // 2)
    for target_treated in [1, 0]:
        pool = np.flatnonzero(((split.train.treatment > 0.5) == bool(target_treated)) & candidate_mask)
        if pool.size == 0:
            continue
        pool = pool[np.argsort(score[pool])[: max(pair_count * 6, pair_count)]]
        group_selected = []
        for _ in range(min(pair_count, pool.size)):
            if not group_selected:
                chosen = int(pool[np.argmin(score[pool])])
            else:
                diff = X_train[pool, None, :] - X_train[np.asarray(group_selected)][None, :, :]
                min_dist = np.min(np.sqrt(np.sum(diff * diff, axis=2)), axis=1)
                min_dist = min_dist / (np.median(min_dist) + 1e-8)
                adjusted = score[pool] - diversity_weight * min_dist
                chosen = int(pool[np.argmin(adjusted)])
            group_selected.append(chosen)
            pool = pool[pool != chosen]
            if pool.size == 0:
                break
        selected.extend(group_selected)
    selected = np.asarray(selected[:context_size], dtype=int)
    if selected.size < context_size:
        selected = np.unique(np.concatenate([selected, _balanced_select(split, score, context_size, rng, candidate_mask)]))[:context_size]
    if selected.size < context_size:
        selected = np.concatenate([selected, rng.choice(split.train.n_samples, size=context_size - selected.size, replace=True)])
    return selected


@dataclass(frozen=True)
class RandomRetriever(ContextRetriever):
    name: str = "random"

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        indices = rng.choice(split.train.n_samples, size=context_size, replace=split.train.n_samples < context_size)
        return self._batch(split, indices)


@dataclass(frozen=True)
class KNNRetriever(ContextRetriever):
    name: str = "knn"

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        X_train, X_test = standardize(split.train.X, split.test.X)
        distances = np.sum((X_train - X_test[query_index]) ** 2, axis=1)
        return self._batch(split, _nearest_by_distance(distances, context_size))


@dataclass(frozen=True)
class TreatmentBalancedRandomRetriever(ContextRetriever):
    name: str = "treatment_balanced_random"

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        treated = np.flatnonzero(split.train.treatment > 0.5)
        control = np.flatnonzero(split.train.treatment <= 0.5)
        n_treated = context_size // 2
        n_control = context_size - n_treated
        selected = np.concatenate(
            [
                _pad_or_sample(treated, n_treated, rng),
                _pad_or_sample(control, n_control, rng),
            ]
        )
        if selected.size < context_size:
            fallback = rng.choice(split.train.n_samples, size=context_size - selected.size, replace=True)
            selected = np.concatenate([selected, fallback])
        rng.shuffle(selected)
        return self._batch(split, selected)


@dataclass(frozen=True)
class PropensityNearestRetriever(ContextRetriever):
    name: str = "propensity_nearest"
    feature_weight: float = 0.15

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        p_query = split.test.propensity[query_index]
        X_train, X_test = standardize(split.train.X, split.test.X)
        p_distance = np.abs(split.train.propensity - p_query)
        x_distance = np.sqrt(np.sum((X_train - X_test[query_index]) ** 2, axis=1))
        distances = p_distance + self.feature_weight * x_distance / max(1.0, np.median(x_distance) + 1e-8)
        return self._batch(split, _nearest_by_distance(distances, context_size))


@dataclass(frozen=True)
class MatchedPairRetriever(ContextRetriever):
    name: str = "matched_pair"
    feature_weight: float = 0.2

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        X_train, X_test = standardize(split.train.X, split.test.X)
        base_distance = np.sqrt(np.sum((X_train - X_test[query_index]) ** 2, axis=1))
        p_distance = np.abs(split.train.propensity - split.test.propensity[query_index])
        score = p_distance + self.feature_weight * base_distance / max(1.0, np.median(base_distance) + 1e-8)
        treated = np.flatnonzero(split.train.treatment > 0.5)
        control = np.flatnonzero(split.train.treatment <= 0.5)
        pair_count = max(1, context_size // 2)
        selected_parts = []
        for pool in [treated, control]:
            if pool.size:
                order = pool[np.argsort(score[pool])[:pair_count]]
                selected_parts.append(order)
        selected = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=int)
        selected = selected[:context_size]
        if selected.size < context_size:
            fallback = _nearest_by_distance(score, context_size)
            selected = np.unique(np.concatenate([selected, fallback]))[:context_size]
        if selected.size < context_size:
            selected = np.concatenate([selected, rng.choice(split.train.n_samples, size=context_size - selected.size, replace=True)])
        return self._batch(split, selected)


@dataclass(frozen=True)
class MatchedPairScoreRetriever(ContextRetriever):
    name: str = "matched_pair_hybrid"
    propensity_weight: float = 1.0
    feature_weight: float = 0.2
    caliper: float | None = None
    overlap_filter: bool = False

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        feature_distance, propensity_distance = _query_distances(split, query_index)
        feature_scale = max(1.0, float(np.median(feature_distance)) + 1e-8)
        score = self.propensity_weight * propensity_distance + self.feature_weight * feature_distance / feature_scale
        candidate_mask = np.ones(split.train.n_samples, dtype=bool)
        if self.caliper is not None:
            candidate_mask &= propensity_distance <= self.caliper
        if self.overlap_filter:
            support = _common_support_indices(split)
            support_mask = np.zeros(split.train.n_samples, dtype=bool)
            support_mask[support] = True
            candidate_mask &= support_mask
        if np.sum(candidate_mask) < max(2, context_size // 2):
            candidate_mask = np.ones(split.train.n_samples, dtype=bool)
        return self._batch(split, _balanced_select(split, score, context_size, rng, candidate_mask))


@dataclass(frozen=True)
class MatchedPairDiverseRetriever(ContextRetriever):
    name: str = "matched_pair_diverse"
    propensity_weight: float = 1.0
    feature_weight: float = 0.2
    diversity_weight: float = 0.25

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        feature_distance, propensity_distance = _query_distances(split, query_index)
        feature_scale = max(1.0, float(np.median(feature_distance)) + 1e-8)
        score = self.propensity_weight * propensity_distance + self.feature_weight * feature_distance / feature_scale
        X_train, _ = standardize(split.train.X, split.test.X)
        selected = _greedy_diverse_select(
            split,
            score,
            X_train,
            context_size,
            rng,
            diversity_weight=self.diversity_weight,
        )
        return self._batch(split, selected)


@dataclass(frozen=True)
class OverlapAwareRetriever(ContextRetriever):
    name: str = "overlap_aware"
    min_propensity: float = 0.05
    max_propensity: float = 0.95

    def select(self, split: CausalSplit, query_index: int, context_size: int, rng: np.random.Generator) -> ContextBatch:
        p_train = split.train.propensity
        p_query = split.test.propensity[query_index]
        treated = split.train.treatment > 0.5
        if np.any(treated) and np.any(~treated):
            lo = max(self.min_propensity, float(np.quantile(p_train[treated], 0.05)), float(np.quantile(p_train[~treated], 0.05)))
            hi = min(self.max_propensity, float(np.quantile(p_train[treated], 0.95)), float(np.quantile(p_train[~treated], 0.95)))
        else:
            lo, hi = self.min_propensity, self.max_propensity
        overlap = np.flatnonzero((p_train >= lo) & (p_train <= hi))
        if overlap.size < max(2, context_size // 2):
            overlap = np.flatnonzero((p_train >= self.min_propensity) & (p_train <= self.max_propensity))
        if overlap.size == 0:
            overlap = np.arange(split.train.n_samples)
        distances = np.abs(p_train[overlap] - np.clip(p_query, lo, hi))
        nearest = overlap[_nearest_by_distance(distances, min(context_size, overlap.size))]
        selected = _pad_or_sample(nearest, context_size, rng)
        return self._batch(split, selected)


RETRIEVER_REGISTRY = {
    "random": RandomRetriever,
    "knn": KNNRetriever,
    "treatment_balanced_random": TreatmentBalancedRandomRetriever,
    "propensity_nearest": PropensityNearestRetriever,
    "matched_pair": MatchedPairRetriever,
    "matched_pair_propensity": lambda: MatchedPairScoreRetriever(
        name="matched_pair_propensity",
        propensity_weight=1.0,
        feature_weight=0.0,
    ),
    "matched_pair_feature": lambda: MatchedPairScoreRetriever(
        name="matched_pair_feature",
        propensity_weight=0.0,
        feature_weight=1.0,
    ),
    "matched_pair_hybrid": lambda: MatchedPairScoreRetriever(name="matched_pair_hybrid"),
    "matched_pair_caliper": lambda: MatchedPairScoreRetriever(name="matched_pair_caliper", caliper=0.10),
    "matched_pair_diverse": MatchedPairDiverseRetriever,
    "overlap_filtered_matched_pair": lambda: MatchedPairScoreRetriever(
        name="overlap_filtered_matched_pair",
        overlap_filter=True,
    ),
    "overlap_aware": OverlapAwareRetriever,
}


def make_retriever(name: str) -> ContextRetriever:
    hybrid_match = re.fullmatch(r"matched_pair_hybrid_w([0-9.]+)", name)
    if hybrid_match:
        weight = float(hybrid_match.group(1))
        return MatchedPairScoreRetriever(name=name, feature_weight=weight)
    caliper_match = re.fullmatch(r"matched_pair_caliper_c([0-9.]+)", name)
    if caliper_match:
        caliper = float(caliper_match.group(1))
        return MatchedPairScoreRetriever(name=name, caliper=caliper)
    diverse_match = re.fullmatch(r"matched_pair_diverse_d([0-9.]+)", name)
    if diverse_match:
        diversity = float(diverse_match.group(1))
        return MatchedPairDiverseRetriever(name=name, diversity_weight=diversity)
    try:
        return RETRIEVER_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(sorted(RETRIEVER_REGISTRY))
        raise ValueError(f"Unknown retriever '{name}'. Available: {available}") from exc
