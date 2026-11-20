from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalDataset:
    name: str
    X: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    tau: np.ndarray
    propensity: np.ndarray

    @property
    def ate(self) -> float:
        return float(np.mean(self.tau))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


@dataclass(frozen=True)
class CausalSplit:
    train: CausalDataset
    test: CausalDataset
    random_state: int


@dataclass(frozen=True)
class ContextBatch:
    indices: np.ndarray
    X: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    propensity: np.ndarray


@dataclass(frozen=True)
class EstimateResult:
    tau_hat: np.ndarray
    mu0_hat: np.ndarray | None
    mu1_hat: np.ndarray | None
    ate_hat: float
    runtime_sec: float
    metadata: dict[str, object]
