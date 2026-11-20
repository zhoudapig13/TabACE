from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retrievers import ContextRetriever
from .types import CausalSplit, ContextBatch, EstimateResult
from .utils import standardize


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    X_aug = _add_intercept(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)
    reg = alpha * np.eye(X_aug.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.pinv(X_aug.T @ X_aug + reg) @ X_aug.T @ y


def _ridge_predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return _add_intercept(np.asarray(X, dtype=float)) @ coef


def _fit_group_model(X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, group: int, alpha: float) -> np.ndarray:
    mask = treatment > 0.5 if group == 1 else treatment <= 0.5
    if int(np.sum(mask)) < 2:
        mask = np.ones_like(treatment, dtype=bool)
    return _ridge_fit(X[mask], outcome[mask], alpha=alpha)


def _mean_abs_smd(X: np.ndarray, treatment: np.ndarray) -> float:
    treated = treatment > 0.5
    control = ~treated
    if int(np.sum(treated)) < 2 or int(np.sum(control)) < 2:
        return float("nan")
    x_t = X[treated]
    x_c = X[control]
    mean_diff = np.mean(x_t, axis=0) - np.mean(x_c, axis=0)
    pooled = np.sqrt((np.var(x_t, axis=0) + np.var(x_c, axis=0)) / 2.0)
    pooled = np.where(pooled < 1e-8, 1.0, pooled)
    return float(np.mean(np.abs(mean_diff / pooled)))


def _context_diversity(X_context_std: np.ndarray) -> float:
    if X_context_std.shape[0] < 2:
        return 0.0
    diff = X_context_std[:, None, :] - X_context_std[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _duplicate_rate(indices: np.ndarray) -> float:
    if indices.size == 0:
        return float("nan")
    return float(1.0 - (np.unique(indices).size / indices.size))


class CATEEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalContextTLearner(CATEEstimator):
    name: str = "local_context_t_learner"
    ridge_alpha: float = 1e-2

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        if retriever is None:
            raise ValueError("LocalContextTLearner requires a context retriever.")
        start = time.perf_counter()
        rng = np.random.default_rng(random_state)
        tau_hat = np.zeros(split.test.n_samples)
        mu0_hat = np.zeros(split.test.n_samples)
        mu1_hat = np.zeros(split.test.n_samples)
        treated_counts = []
        overlap_scores = []
        context_smd = []
        context_p_distance = []
        context_feature_distance = []
        context_diversity = []
        context_duplicate_rate = []
        full_train_smd = _mean_abs_smd(split.train.X, split.train.treatment)
        X_train_std, X_test_std = standardize(split.train.X, split.test.X)
        for query_index in range(split.test.n_samples):
            context = retriever.select(split, query_index, context_size, rng)
            mu0, mu1 = self._estimate_one(split, context, query_index)
            mu0_hat[query_index] = mu0
            mu1_hat[query_index] = mu1
            tau_hat[query_index] = mu1 - mu0
            treated_counts.append(float(np.mean(context.treatment)))
            overlap_scores.append(float(np.mean(np.minimum(context.propensity, 1.0 - context.propensity))))
            context_smd.append(_mean_abs_smd(context.X, context.treatment))
            context_p_distance.append(float(np.mean(np.abs(context.propensity - split.test.propensity[query_index]))))
            context_feature_distance.append(
                float(np.mean(np.sqrt(np.sum((X_train_std[context.indices] - X_test_std[query_index]) ** 2, axis=1))))
            )
            context_diversity.append(_context_diversity(X_train_std[context.indices]))
            context_duplicate_rate.append(_duplicate_rate(context.indices))
        context_smd_arr = np.asarray(context_smd, dtype=float)
        mean_context_smd = float(np.nanmean(context_smd_arr)) if np.any(~np.isnan(context_smd_arr)) else float("nan")
        return EstimateResult(
            tau_hat=tau_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            ate_hat=float(np.mean(tau_hat)),
            runtime_sec=time.perf_counter() - start,
            metadata={
                "mean_context_treated_rate": float(np.mean(treated_counts)),
                "mean_context_overlap": float(np.mean(overlap_scores)),
                "full_train_abs_smd": full_train_smd,
                "mean_context_abs_smd": mean_context_smd,
                "context_abs_smd_delta_vs_full_train": mean_context_smd - full_train_smd,
                "mean_context_propensity_distance": float(np.mean(context_p_distance)),
                "mean_context_feature_distance": float(np.mean(context_feature_distance)),
                "mean_context_diversity": float(np.mean(context_diversity)),
                "mean_context_duplicate_rate": float(np.mean(context_duplicate_rate)),
            },
        )

    def _estimate_one(self, split: CausalSplit, context: ContextBatch, query_index: int) -> tuple[float, float]:
        X_train, X_query = standardize(context.X, split.test.X[query_index : query_index + 1])
        coef0 = _fit_group_model(X_train, context.treatment, context.outcome, group=0, alpha=self.ridge_alpha)
        coef1 = _fit_group_model(X_train, context.treatment, context.outcome, group=1, alpha=self.ridge_alpha)
        return float(_ridge_predict(X_query, coef0)[0]), float(_ridge_predict(X_query, coef1)[0])


@dataclass(frozen=True)
class GlobalTLearner(CATEEstimator):
    name: str = "global_t_learner"
    ridge_alpha: float = 1e-2

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        start = time.perf_counter()
        X_train, X_test = standardize(split.train.X, split.test.X)
        coef0 = _fit_group_model(X_train, split.train.treatment, split.train.outcome, group=0, alpha=self.ridge_alpha)
        coef1 = _fit_group_model(X_train, split.train.treatment, split.train.outcome, group=1, alpha=self.ridge_alpha)
        mu0_hat = _ridge_predict(X_test, coef0)
        mu1_hat = _ridge_predict(X_test, coef1)
        tau_hat = mu1_hat - mu0_hat
        return EstimateResult(
            tau_hat=tau_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            ate_hat=float(np.mean(tau_hat)),
            runtime_sec=time.perf_counter() - start,
            metadata={},
        )


@dataclass(frozen=True)
class GlobalSLearner(CATEEstimator):
    name: str = "global_s_learner"
    ridge_alpha: float = 1e-2

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        start = time.perf_counter()
        X_train, X_test = standardize(split.train.X, split.test.X)
        train_aug = np.hstack([X_train, split.train.treatment[:, None], X_train * split.train.treatment[:, None]])
        test0 = np.hstack([X_test, np.zeros((X_test.shape[0], 1)), np.zeros_like(X_test)])
        test1 = np.hstack([X_test, np.ones((X_test.shape[0], 1)), X_test])
        coef = _ridge_fit(train_aug, split.train.outcome, alpha=self.ridge_alpha)
        mu0_hat = _ridge_predict(test0, coef)
        mu1_hat = _ridge_predict(test1, coef)
        tau_hat = mu1_hat - mu0_hat
        return EstimateResult(
            tau_hat=tau_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            ate_hat=float(np.mean(tau_hat)),
            runtime_sec=time.perf_counter() - start,
            metadata={},
        )


@dataclass(frozen=True)
class GlobalXLearner(CATEEstimator):
    name: str = "global_x_learner"
    ridge_alpha: float = 1e-2

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        start = time.perf_counter()
        X_train, X_test = standardize(split.train.X, split.test.X)
        coef0 = _fit_group_model(X_train, split.train.treatment, split.train.outcome, group=0, alpha=self.ridge_alpha)
        coef1 = _fit_group_model(X_train, split.train.treatment, split.train.outcome, group=1, alpha=self.ridge_alpha)
        mu0_train = _ridge_predict(X_train, coef0)
        mu1_train = _ridge_predict(X_train, coef1)
        treated = split.train.treatment > 0.5
        d_treated = split.train.outcome[treated] - mu0_train[treated]
        d_control = mu1_train[~treated] - split.train.outcome[~treated]
        tau1_coef = _ridge_fit(X_train[treated], d_treated, alpha=self.ridge_alpha) if np.sum(treated) >= 2 else coef1 - coef0
        tau0_coef = _ridge_fit(X_train[~treated], d_control, alpha=self.ridge_alpha) if np.sum(~treated) >= 2 else coef1 - coef0
        tau1 = _ridge_predict(X_test, tau1_coef)
        tau0 = _ridge_predict(X_test, tau0_coef)
        p = np.clip(split.test.propensity, 0.05, 0.95)
        tau_hat = (1.0 - p) * tau1 + p * tau0
        mu0_hat = _ridge_predict(X_test, coef0)
        mu1_hat = mu0_hat + tau_hat
        return EstimateResult(
            tau_hat=tau_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            ate_hat=float(np.mean(tau_hat)),
            runtime_sec=time.perf_counter() - start,
            metadata={"uses_true_propensity": True},
        )


@dataclass(frozen=True)
class GlobalDRLearner(CATEEstimator):
    name: str = "global_dr_learner"
    ridge_alpha: float = 1e-2

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        start = time.perf_counter()
        X_train, X_test = standardize(split.train.X, split.test.X)
        coef0 = _fit_group_model(X_train, split.train.treatment, split.train.outcome, group=0, alpha=self.ridge_alpha)
        coef1 = _fit_group_model(X_train, split.train.treatment, split.train.outcome, group=1, alpha=self.ridge_alpha)
        mu0_train = _ridge_predict(X_train, coef0)
        mu1_train = _ridge_predict(X_train, coef1)
        p = np.clip(split.train.propensity, 0.05, 0.95)
        w = split.train.treatment
        pseudo_tau = (
            mu1_train
            - mu0_train
            + w * (split.train.outcome - mu1_train) / p
            - (1.0 - w) * (split.train.outcome - mu0_train) / (1.0 - p)
        )
        tau_coef = _ridge_fit(X_train, pseudo_tau, alpha=self.ridge_alpha)
        tau_hat = _ridge_predict(X_test, tau_coef)
        mu0_hat = _ridge_predict(X_test, coef0)
        mu1_hat = mu0_hat + tau_hat
        return EstimateResult(
            tau_hat=tau_hat,
            mu0_hat=mu0_hat,
            mu1_hat=mu1_hat,
            ate_hat=float(np.mean(tau_hat)),
            runtime_sec=time.perf_counter() - start,
            metadata={"uses_true_propensity": True},
        )


@dataclass(frozen=True)
class TabICLWrapper(LocalContextTLearner):
    name: str = "tabicl_or_fallback"

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        if self._tabicl_available():
            # The research prototype keeps the context-selection API stable.
            # Outcome-model adaptation for TabICL can replace this fallback later.
            pass
        result = super().estimate(split, retriever, context_size, random_state)
        metadata = dict(result.metadata)
        metadata["backend"] = "ridge_fallback"
        metadata["wrapper"] = "TabICL"
        return EstimateResult(result.tau_hat, result.mu0_hat, result.mu1_hat, result.ate_hat, result.runtime_sec, metadata)

    @staticmethod
    def _tabicl_available() -> bool:
        root = Path(__file__).resolve().parents[2]
        src = root / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            __import__("tabicl")
            return True
        except Exception:
            return False


@dataclass(frozen=True)
class TabPFNWrapper(LocalContextTLearner):
    name: str = "tabpfn_or_fallback"

    def estimate(self, split: CausalSplit, retriever: ContextRetriever | None, context_size: int, random_state: int) -> EstimateResult:
        try:
            __import__("tabpfn")
            backend = "tabpfn_available_ridge_fallback_for_cate"
        except Exception:
            backend = "ridge_fallback"
        result = super().estimate(split, retriever, context_size, random_state)
        metadata = dict(result.metadata)
        metadata["backend"] = backend
        metadata["wrapper"] = "TabPFN"
        return EstimateResult(result.tau_hat, result.mu0_hat, result.mu1_hat, result.ate_hat, result.runtime_sec, metadata)


ESTIMATOR_REGISTRY = {
    "local_context_t_learner": LocalContextTLearner,
    "tabicl_or_fallback": TabICLWrapper,
    "tabpfn_or_fallback": TabPFNWrapper,
    "global_t_learner": GlobalTLearner,
    "global_s_learner": GlobalSLearner,
    "global_x_learner": GlobalXLearner,
    "global_dr_learner": GlobalDRLearner,
}


def make_estimator(name: str) -> CATEEstimator:
    try:
        return ESTIMATOR_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(sorted(ESTIMATOR_REGISTRY))
        raise ValueError(f"Unknown estimator '{name}'. Available: {available}") from exc
