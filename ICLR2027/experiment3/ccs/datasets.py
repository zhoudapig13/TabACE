from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import CausalDataset, CausalSplit


@dataclass(frozen=True)
class SyntheticSCMConfig:
    n_samples: int = 1000
    n_features: int = 10
    treatment_imbalance: float = 0.0
    confounding_strength: float = 1.0
    overlap_violation: float = 0.0
    heterogeneity: float = 1.0
    effect_nonlinearity: float = 1.0
    noise_std: float = 1.0
    random_state: int = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def generate_synthetic_scm(config: SyntheticSCMConfig) -> CausalDataset:
    rng = np.random.default_rng(config.random_state)
    X = rng.normal(size=(config.n_samples, config.n_features))

    main = X[:, 0]
    aux = X[:, 1] if config.n_features > 1 else 0.0
    het = X[:, 2] if config.n_features > 2 else main
    nonlinear = np.sin(X[:, 3]) if config.n_features > 3 else np.sin(main)

    logit = (
        config.treatment_imbalance
        + config.confounding_strength * (0.9 * main - 0.6 * aux + 0.35 * nonlinear)
    )
    if config.overlap_violation > 0:
        logit = logit + config.overlap_violation * np.sign(main) * (1.0 + np.abs(main))
    propensity = np.clip(sigmoid(logit), 0.01, 0.99)
    treatment = rng.binomial(1, propensity).astype(float)

    baseline = (
        1.0
        + 1.2 * main
        - 0.8 * aux
        + 0.5 * nonlinear
        + 0.25 * np.sum(X[:, 4:], axis=1) / max(1, config.n_features - 4)
    )
    tau = 1.0 + config.heterogeneity * (
        0.8 * sigmoid(het)
        + 0.35 * main * aux
        + config.effect_nonlinearity * 0.25 * np.sin(main * aux)
    )
    mu0 = baseline
    mu1 = baseline + tau
    outcome = mu0 + treatment * tau + rng.normal(scale=config.noise_std, size=config.n_samples)

    return CausalDataset(
        name="synthetic_scm",
        X=X.astype(float),
        treatment=treatment.astype(float),
        outcome=outcome.astype(float),
        mu0=mu0.astype(float),
        mu1=mu1.astype(float),
        tau=tau.astype(float),
        propensity=propensity.astype(float),
    )


def split_dataset(dataset: CausalDataset, test_size: float, random_state: int) -> CausalSplit:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(dataset.n_samples)
    n_test = max(1, int(round(dataset.n_samples * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return CausalSplit(
        train=_subset(dataset, train_idx),
        test=_subset(dataset, test_idx),
        random_state=random_state,
    )


def _subset(dataset: CausalDataset, indices: np.ndarray) -> CausalDataset:
    return CausalDataset(
        name=dataset.name,
        X=dataset.X[indices],
        treatment=dataset.treatment[indices],
        outcome=dataset.outcome[indices],
        mu0=dataset.mu0[indices],
        mu1=dataset.mu1[indices],
        tau=dataset.tau[indices],
        propensity=dataset.propensity[indices],
    )
