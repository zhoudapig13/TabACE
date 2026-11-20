from __future__ import annotations

import numpy as np


def pehe(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))


def ate_error(tau_true: np.ndarray, ate_hat: float) -> float:
    return float(abs(float(np.mean(tau_true)) - float(ate_hat)))


def policy_value(mu0: np.ndarray, mu1: np.ndarray, tau_hat: np.ndarray) -> float:
    treat = np.asarray(tau_hat) > 0
    potential = np.where(treat, mu1, mu0)
    return float(np.mean(potential))


def calibration_ece(tau_true: np.ndarray, tau_hat: np.ndarray, n_bins: int = 10) -> float:
    tau_true = np.asarray(tau_true)
    tau_hat = np.asarray(tau_hat)
    order = np.argsort(tau_hat)
    bins = np.array_split(order, n_bins)
    errors = []
    weights = []
    for bin_idx in bins:
        if bin_idx.size == 0:
            continue
        errors.append(abs(float(np.mean(tau_hat[bin_idx]) - np.mean(tau_true[bin_idx]))))
        weights.append(bin_idx.size / tau_hat.size)
    return float(np.sum(np.asarray(errors) * np.asarray(weights))) if errors else float("nan")


def uplift_at_fraction(mu0: np.ndarray, mu1: np.ndarray, tau_hat: np.ndarray, fraction: float = 0.3) -> float:
    n = max(1, int(round(len(tau_hat) * fraction)))
    chosen = np.argsort(-np.asarray(tau_hat))[:n]
    return float(np.mean(np.asarray(mu1)[chosen] - np.asarray(mu0)[chosen]))


def auuc_score(outcome: np.ndarray, treatment: np.ndarray, score: np.ndarray) -> float:
    """Approximate AUUC from observed uplift data sorted by predicted score."""
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float))
    y = outcome[order]
    w = treatment[order] > 0.5
    treated_count = np.cumsum(w)
    control_count = np.cumsum(~w)
    treated_gain = np.cumsum(y * w) / np.maximum(treated_count, 1)
    control_gain = np.cumsum(y * (~w)) / np.maximum(control_count, 1)
    uplift = treated_gain - control_gain
    return float(np.trapezoid(uplift, dx=1.0 / max(1, uplift.size - 1)))


def qini_score(outcome: np.ndarray, treatment: np.ndarray, score: np.ndarray) -> float:
    """Approximate Qini area relative to a random targeting baseline."""
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float))
    y = outcome[order]
    w = treatment[order] > 0.5
    n_treated = max(1, int(np.sum(w)))
    n_control = max(1, int(np.sum(~w)))
    gains = np.cumsum(y * w) - np.cumsum(y * (~w)) * (n_treated / n_control)
    random_line = np.linspace(0.0, float(gains[-1]), gains.size)
    return float(np.trapezoid(gains - random_line, dx=1.0 / max(1, gains.size - 1)))
