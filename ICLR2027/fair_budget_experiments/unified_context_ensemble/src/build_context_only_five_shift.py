#!/usr/bin/env python3
"""Build five-manipulation curves under the context-only information protocol.

Shared receives one policy pooled over all manipulation families and strengths.
Specific and Pro receive the family identifier, but each policy is pooled over
origin and every strength.  No selector reads the strength value.  Regression
excludes the historical unmanipulated clean Context and every prediction that
depends on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_five_shift_candidate as source,
)
from fair_budget_experiments.unified_context_ensemble.src.core import (  # noqa: E402
    AdaptiveGate,
    BatchMemberRouter,
    CLASSIFICATION_BATCH_FEATURE_NAMES,
    DecisionGuard,
    FusionPolicy,
    PredictionBank,
    REGRESSION_BATCH_FEATURE_NAMES,
    aggregate,
    batch_member_features,
    clear_batch_member_feature_cache,
    predict,
    prediction_diagnostics,
)
from fair_budget_experiments.unified_context_ensemble.src.protocol import (  # noqa: E402
    DEFAULT_PROTOCOL as CONTEXT_PROTOCOL,
)
from fair_budget_experiments.unified_context_ensemble.src.inference import (  # noqa: E402
    infer_context_view,
)
HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = HERE / "results" / "context_only_v14" / "five_shift"
RHOS = (0, 20, 50, 80, 100)
METHODS = ("base", "shared", "specific", "pro")
METRICS = source.TASK_METRICS
RISK_PENALTY = CONTEXT_PROTOCOL.tuning.risk_penalty
DOWNSIDE_PENALTY = CONTEXT_PROTOCOL.tuning.base_downside_penalty
TAIL_RISK_FRACTION = CONTEXT_PROTOCOL.tuning.tail_risk_fraction
TAIL_RISK_WEIGHT = CONTEXT_PROTOCOL.tuning.tail_risk_weight
REGULARIZATIONS = CONTEXT_PROTOCOL.tuning.regularization_grid
ALPHAS = CONTEXT_PROTOCOL.tuning.alpha_grid
GATED_FINALIST_COUNT = int(
    CONTEXT_PROTOCOL.raw["adaptive_tuning"].get("gated_finalist_count", 2)
)
REGRESSION_GATE_ORIGIN_WEIGHT = float(
    CONTEXT_PROTOCOL.raw["adaptive_tuning"].get(
        "regression_gate_origin_training_weight", 1.0
    )
)
REGRESSION_ROUTER_ALPHAS = tuple(
    float(value)
    for value in CONTEXT_PROTOCOL.raw["adaptive_tuning"].get(
        "regression_router_alpha_grid", [1.0]
    )
)
_ROUTER_ALPHA_GRID_RAW = os.environ.get("TABACE_ROUTER_ALPHA_GRID")
ROUTER_ALPHAS_OVERRIDE = (
    tuple(float(value) for value in _ROUTER_ALPHA_GRID_RAW.split(","))
    if _ROUTER_ALPHA_GRID_RAW
    else None
)
# Five-manipulation classification retains the compact prediction-only router
# used by the audited v20 snapshot.  The seven-shift adapter enables the four
# additional unlabeled class-prior summaries for label-shift detection.
CLASSIFICATION_BATCH_PRIOR_ROUTER = False
FIVE_TUNING = CONTEXT_PROTOCOL.raw["benchmarks"]["five_context_manipulations"].get(
    "adaptive_tuning_overrides", {}
)
CLASS_ORIGIN_NONINFERIORITY = bool(
    FIVE_TUNING.get("classification_origin_noninferiority_gate", False)
)
CLASS_ORIGIN_TOLERANCE = float(
    FIVE_TUNING.get("classification_origin_noninferiority_tolerance", 0.0)
)
_GROUP_NI_RAW = os.environ.get("TABACE_GROUP_NONINFERIORITY_MARGIN")
GROUP_NONINFERIORITY_ENABLED = _GROUP_NI_RAW is not None
GROUP_NONINFERIORITY_MARGIN = (
    float(_GROUP_NI_RAW) if _GROUP_NI_RAW is not None else 0.0
)
SHARED_GROUP_NONINFERIORITY_MARGIN = float(
    os.environ.get(
        "TABACE_SHARED_GROUP_NONINFERIORITY_MARGIN",
        GROUP_NONINFERIORITY_MARGIN,
    )
)
SPECIFIC_GROUP_NONINFERIORITY_MARGIN = float(
    os.environ.get(
        "TABACE_SPECIFIC_GROUP_NONINFERIORITY_MARGIN",
        GROUP_NONINFERIORITY_MARGIN,
    )
)
PRO_GROUP_NONINFERIORITY_MARGIN = float(
    os.environ.get(
        "TABACE_PRO_GROUP_NONINFERIORITY_MARGIN",
        GROUP_NONINFERIORITY_MARGIN,
    )
)
GROUP_NONINFERIORITY_BY_FOLD = os.environ.get(
    "TABACE_GROUP_NONINFERIORITY_BY_FOLD", "0"
).lower() in {"1", "true", "yes"}
GROUP_NONINFERIORITY_INCLUDE_ORIGIN = os.environ.get(
    "TABACE_GROUP_NONINFERIORITY_INCLUDE_ORIGIN", "0"
).lower() in {"1", "true", "yes"}
GROUP_NONINFERIORITY_ALLOW_EXACT = os.environ.get(
    "TABACE_GROUP_NONINFERIORITY_ALLOW_EXACT", "1"
).lower() in {"1", "true", "yes"}
PRO_COMPARE_TO_SHARED = os.environ.get(
    "TABACE_PRO_COMPARE_TO_SHARED", "0"
).lower() in {"1", "true", "yes"}
FOCUS_ENABLE_LOCAL_ROUTER = os.environ.get(
    "TABACE_FOCUS_ENABLE_LOCAL_ROUTER", "0"
).lower() in {"1", "true", "yes"}
_ROUTER_GAIN_GRID_RAW = os.environ.get("TABACE_ROUTER_MIN_GAIN_GRID")
ROUTER_MIN_GAIN_GRID = (
    tuple(float(value) for value in _ROUTER_GAIN_GRID_RAW.split(","))
    if _ROUTER_GAIN_GRID_RAW
    else (0.0, 0.001, 0.003, 0.005)
)
COMPACT_SAFETY_SEARCH = os.environ.get(
    "TABACE_COMPACT_SAFETY_SEARCH", "0"
).lower() in {"1", "true", "yes"}
ULTRA_COMPACT_SAFETY_SEARCH = os.environ.get(
    "TABACE_ULTRA_COMPACT_SAFETY_SEARCH", "0"
).lower() in {"1", "true", "yes"}
METRIC_RESIDUAL_SEARCH = os.environ.get(
    "TABACE_METRIC_RESIDUAL_SEARCH", "0"
).lower() in {"1", "true", "yes"}
METRIC_RESIDUAL_ONLY = os.environ.get(
    "TABACE_METRIC_RESIDUAL_ONLY", "0"
).lower() in {"1", "true", "yes"}
METRIC_RESIDUAL_SKIP_GATE = os.environ.get(
    "TABACE_METRIC_RESIDUAL_SKIP_GATE", "0"
).lower() in {"1", "true", "yes"}
FAST_NONFOCUS_SEARCH = os.environ.get(
    "TABACE_FAST_NONFOCUS_SEARCH", "0"
).lower() in {"1", "true", "yes"}
_METRIC_RESIDUAL_REG_RAW = os.environ.get(
    "TABACE_METRIC_RESIDUAL_REG_GRID", "0.0001,0.001,0.01,0.1,1,10,100"
)
METRIC_RESIDUAL_REGULARIZATIONS = tuple(
    float(value) for value in _METRIC_RESIDUAL_REG_RAW.split(",")
)
_METRIC_RESIDUAL_SIZE_RAW = os.environ.get(
    "TABACE_METRIC_RESIDUAL_MEMBER_SIZES", ""
).strip()
METRIC_RESIDUAL_MEMBER_SIZES = (
    tuple(int(value) for value in _METRIC_RESIDUAL_SIZE_RAW.split(","))
    if _METRIC_RESIDUAL_SIZE_RAW
    else ()
)


def _parse_focus_groups(name: str) -> frozenset[tuple[str, int]]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    groups = set()
    for item in raw.split(","):
        family, rho = item.split(":", 1)
        groups.add((family.strip(), int(rho)))
    return frozenset(groups)


def _parse_float_grid(name: str) -> tuple[float, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return tuple(float(value) for value in raw.split(","))


def _parse_member_subset(name: str) -> tuple[str, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return tuple(value.strip() for value in raw.split(",") if value.strip())


SHARED_FOCUS_GROUPS = _parse_focus_groups("TABACE_SHARED_FOCUS_GROUPS")
SPECIFIC_FOCUS_GROUPS = _parse_focus_groups("TABACE_SPECIFIC_FOCUS_GROUPS")
PRO_FOCUS_GROUPS = _parse_focus_groups("TABACE_PRO_FOCUS_GROUPS")
SHARED_FIT_GROUPS = _parse_focus_groups("TABACE_SHARED_FIT_GROUPS")
SPECIFIC_FIT_GROUPS = _parse_focus_groups("TABACE_SPECIFIC_FIT_GROUPS")
PRO_FIT_GROUPS = _parse_focus_groups("TABACE_PRO_FIT_GROUPS")
SHARED_METRIC_RESIDUAL_ALPHAS = _parse_float_grid(
    "TABACE_SHARED_METRIC_RESIDUAL_ALPHA_GRID"
)
SPECIFIC_METRIC_RESIDUAL_ALPHAS = _parse_float_grid(
    "TABACE_SPECIFIC_METRIC_RESIDUAL_ALPHA_GRID"
)
PRO_METRIC_RESIDUAL_ALPHAS = _parse_float_grid(
    "TABACE_PRO_METRIC_RESIDUAL_ALPHA_GRID"
)
SHARED_METRIC_RESIDUAL_MEMBERS = _parse_member_subset(
    "TABACE_SHARED_METRIC_RESIDUAL_MEMBERS"
)

CLASS_SHARED = (
    "shared/knn",
    "shared/prototype",
    "shared/boundary",
    "shared/subspace",
    "shared/clean",
)
CLASS_SPECIFIC = {
    "shortcut": (
        "sub/drop_top1/k20",
        "sub/drop_top2/k20",
        "sub/low_corr_half/k20",
        "sub/low_corr_three_quarters/k20",
        "shared/subspace",
    ),
    "boundary": (
        "boundary/k08",
        "boundary/k20",
        "boundary/k32",
        "sub/low_corr_three_quarters/k20",
        "shared/subspace",
    ),
    "aucband": (
        "minority/k08",
        "minority/k20",
        "minority/k32",
        "prototype/k20",
        "shared/subspace",
    ),
    "local": (
        "knn/all/k08",
        "knn/all/k16",
        "knn/all/k32",
        "clean/k16",
        "shared/subspace",
    ),
    "mixed": (
        "sub/low_corr_three_quarters/k20",
        "boundary/k20",
        "minority/k20",
        "prototype/k20",
        "shared/subspace",
    ),
}
REG_SHARED = (
    "shared/knn",
    "shared/prototype",
    "shared/bracket",
    "shared/subspace",
    "source_clean",
)
REG_SPECIFIC = {
    family: tuple("source_clean" if name == "shared/clean" else name for name in names)
    for family, names in source.REG_SPECIFIC.items()
}
_BANK_CACHE: dict[int, tuple[View, PredictionBank]] = {}
_VIEW_REFS: list[View] = []


@dataclass(frozen=True)
class View:
    task: str
    data_id: int
    dataset: str
    fold: int
    family: str
    rho: int
    target: np.ndarray
    classes: np.ndarray | None
    scale: float | None
    base: np.ndarray
    members: dict[str, np.ndarray]
    diagnostics: Mapping[str, float] = field(default_factory=dict)


def _with_member(view: View, name: str, prediction: np.ndarray) -> View:
    """Add one legal lower-tier aggregate as a Pro ensemble candidate."""

    extended = replace(
        view,
        members={**view.members, name: np.asarray(prediction, dtype=np.float64)},
    )
    # `_bank` caches by object identity; retaining extended views prevents
    # CPython from reusing an id for a different dataset-condition.
    _VIEW_REFS.append(extended)
    return extended


def _with_base(
    view: View,
    prediction: np.ndarray,
    *,
    exclude_members: Iterable[str] = (),
) -> View:
    """Use a frozen lower-tier prediction as the anchor of another tier."""

    excluded = set(exclude_members)
    extended = replace(
        view,
        base=np.asarray(prediction, dtype=np.float64),
        members={
            name: value for name, value in view.members.items()
            if name not in excluded
        },
    )
    _VIEW_REFS.append(extended)
    return extended


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _legal_names(task: str, family: str, method: str, available: Iterable[str]) -> tuple[str, ...]:
    available = tuple(available)
    if task == "classification":
        if method == "shared":
            names = CLASS_SHARED
        elif method == "specific":
            names = CLASS_SPECIFIC[family]
        else:
            names = tuple(name for name in available if name != "shared_output")
    else:
        if method == "shared":
            names = REG_SHARED
        elif method == "specific":
            names = REG_SPECIFIC[family]
        else:
            names = tuple(
                name for name in available
                if name not in {"shared/clean", "shared_output"}
            )
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"{task}/{family}/{method} missing Contexts: {missing}")
    return tuple(names)


def _view(
    record: source.Record,
    family: str,
    method: str,
) -> View:
    names = _legal_names(record.task, family, method, record.pro)
    return View(
        task=record.task,
        data_id=record.data_id,
        dataset=record.dataset,
        fold=record.fold,
        family=family,
        rho=record.rho,
        target=record.target,
        classes=record.classes,
        scale=record.scale,
        base=record.base,
        members={name: record.pro[name] for name in names},
    )


def _bank(view: View, names: tuple[str, ...] | None = None) -> PredictionBank:
    # Policies may select a subset from a bank without rebuilding it.  The
    # selector evaluates many policies on the same dataset-condition; caching
    # avoids repeatedly normalising every dense member prediction.
    key = id(view)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = (
            view,
            PredictionBank(
                view.task,
                view.base,
                view.members,
                scale=view.scale,
                diagnostics=view.diagnostics,
            ),
        )
    cached_view, bank = _BANK_CACHE[key]
    if cached_view is not view:
        # Defensive guard for runtimes that could reuse an identity token
        # despite a caller mutating the cache contract.
        raise RuntimeError("prediction-bank identity cache collision")
    if names is not None and any(name not in bank.members for name in names):
        raise KeyError("policy requests a Context outside the cached bank")
    return bank


def _primary(view: View, prediction: np.ndarray) -> float:
    if view.task == "classification":
        classes = np.asarray(view.classes)
        return float(np.mean(classes[np.argmax(prediction, axis=1)] == view.target))
    error = np.asarray(view.target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    scale = max(float(view.scale), 1.0e-12)
    return -float(np.sqrt(np.mean(error**2)) / scale)


def _display_metrics(view: View, prediction: np.ndarray) -> dict[str, float]:
    """Metrics exposed by the paper tables for safety-constrained selection."""

    if view.task == "classification":
        values = source._classification_metrics(
            np.asarray(view.target),
            np.asarray(prediction),
            np.asarray(view.classes),
        )
        return {key: float(values[key]) for key in ("accuracy", "auc")}
    values = source._regression_metrics(
        np.asarray(view.target, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
        float(view.scale),
    )
    return {key: float(values[key]) for key in ("r2", "nmae")}


def _group_metric_gains(
    views: list[View],
    policy: FusionPolicy,
    reference_member: str | None = None,
) -> dict[str, float]:
    """Return Base-relative macro gains for every training condition/metric.

    The selector may use family and strength to audit one shared policy during
    outer-fold training, but the selected policy and its inference gate receive
    neither signal.  This is a robustness constraint, not a condition router.
    """

    grouped: dict[tuple[str, int] | tuple[str, int, int], list[View]] = {}
    for view in views:
        # Origin can be audited by the same outer-training safety constraint
        # when repairing clean-condition failures.  The selected inference
        # policy still receives neither the family nor the strength label.
        if int(view.rho) == 0 and not GROUP_NONINFERIORITY_INCLUDE_ORIGIN:
            continue
        key = (
            (view.family, int(view.rho), int(view.fold))
            if GROUP_NONINFERIORITY_BY_FOLD
            else (view.family, int(view.rho))
        )
        grouped.setdefault(key, []).append(view)
    gains: dict[str, float] = {}
    for group_key, group in grouped.items():
        family, rho = group_key[:2]
        fold_suffix = (
            f"/fold{group_key[2]}" if len(group_key) == 3 else ""
        )
        base_rows = [
            _display_metrics(
                view,
                view.base if reference_member is None else view.members[reference_member],
            )
            for view in group
        ]
        method_rows = [
            _display_metrics(
                view,
                predict(_bank(view, policy.members), policy),
            )
            for view in group
        ]
        for metric in base_rows[0]:
            base = float(np.mean([row[metric] for row in base_rows]))
            method = float(np.mean([row[metric] for row in method_rows]))
            gain = method - base if metric != "nmae" else base - method
            gains[f"{family}/{rho}{fold_suffix}/{metric}"] = gain
    return gains


def _policy_score(
    views: list[View],
    policy: FusionPolicy,
    *,
    downside_penalty: float = DOWNSIDE_PENALTY,
    tail_risk_weight: float = TAIL_RISK_WEIGHT,
) -> tuple[float, float, float]:
    values = []
    gains = []
    for view in views:
        value = _primary(view, predict(_bank(view, policy.members), policy))
        base = _primary(view, view.base)
        values.append(value)
        gains.append(value - base)
    negative_gain = np.minimum(np.asarray(gains, dtype=np.float64), 0.0)
    downside = float(np.sqrt(np.mean(negative_gain**2)))
    value_array = np.asarray(values, dtype=np.float64)
    tail_size = max(1, int(math.ceil(TAIL_RISK_FRACTION * len(value_array))))
    tail_value = float(np.mean(np.sort(value_array)[:tail_size]))
    return (
        float(
            np.mean(values)
            + float(tail_risk_weight) * tail_value
            - RISK_PENALTY * np.std(gains)
            - float(downside_penalty) * downside
        ),
        float(np.mean(values)),
        float(np.min(gains)),
    )


def _weighted_policy(
    names: tuple[str, ...], coefficients: np.ndarray, metadata: dict[str, Any]
) -> FusionPolicy:
    coefficient = np.clip(np.asarray(coefficients, dtype=np.float64), 0.0, None)
    if float(coefficient.sum()) <= 1.0e-12:
        return FusionPolicy((), "mean", 0.0, metadata=metadata)
    coefficient /= float(coefficient.sum())
    alpha = 1.0 - float(coefficient[0])
    if alpha <= 1.0e-10:
        return FusionPolicy((), "mean", 0.0, metadata=metadata)
    context = coefficient[1:] / alpha
    active = np.flatnonzero(context > 1.0e-10)
    return FusionPolicy(
        tuple(names[index] for index in active),
        "weighted_mean",
        alpha,
        weights=tuple(float(context[index]) for index in active),
        metadata=metadata,
    )


def _fit_convex(
    views: list[View], names: tuple[str, ...], regularization: float
) -> FusionPolicy:
    design: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for view in views:
        columns = [view.base] + [view.members[name] for name in names]
        if view.task == "classification":
            matrix = np.stack(columns, axis=-1).reshape(-1, len(names) + 1)
            classes = np.asarray(view.classes)
            positions = {label: pos for pos, label in enumerate(classes.tolist())}
            encoded = np.asarray([positions[label] for label in view.target], dtype=int)
            target = np.eye(len(classes), dtype=np.float64)[encoded].reshape(-1)
        else:
            scale = max(float(view.scale), 1.0e-12)
            matrix = np.stack(columns, axis=1) / scale
            target = np.asarray(view.target, dtype=np.float64) / scale
        factor = 1.0 / math.sqrt(max(1, len(target)))
        design.append(np.asarray(matrix, dtype=np.float64) * factor)
        targets.append(np.asarray(target, dtype=np.float64) * factor)
    penalty = np.zeros((len(names), len(names) + 1), dtype=np.float64)
    penalty[:, 1:] = math.sqrt(float(regularization)) * np.eye(len(names))
    matrix = np.vstack([*design, penalty])
    target = np.concatenate([*targets, np.zeros(len(names), dtype=np.float64)])
    coefficients, _ = nnls(matrix, target, maxiter=40 * (len(names) + 1))
    return _weighted_policy(
        names,
        coefficients,
        {"candidate": "convex", "regularization": regularization},
    )


def _fit_regression_residual_stacker(
    views: list[View], names: tuple[str, ...], regularization: float
) -> FusionPolicy:
    """Fit a prediction-scale-normalized residual stacker on outer training."""

    design_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for view in views:
        scale = max(float(np.std(view.base)), 1.0e-12)
        residuals = np.column_stack([
            (np.asarray(view.members[name]) - np.asarray(view.base)) / scale
            for name in names
        ])
        design = np.column_stack([np.ones(len(view.base)), residuals])
        target = (np.asarray(view.target, dtype=np.float64) - view.base) / scale
        # Give every dataset-condition equal mass regardless of row count.
        factor = 1.0 / math.sqrt(max(1, len(target)))
        design_rows.append(design * factor)
        target_rows.append(target * factor)
    matrix = np.vstack(design_rows)
    target = np.concatenate(target_rows)
    # Solve the ridge objective as an augmented least-squares system.  Context
    # members can be exactly collinear (especially nested lower-tier outputs),
    # for which normal equations are unnecessarily ill-conditioned.  The
    # intercept remains unpenalized and the fitted objective is unchanged.
    ridge = np.diag(
        [0.0, *([math.sqrt(float(regularization))] * len(names))]
    )
    augmented_matrix = np.vstack([matrix, ridge])
    augmented_target = np.concatenate(
        [target, np.zeros(matrix.shape[1], dtype=np.float64)]
    )
    beta, *_ = np.linalg.lstsq(
        augmented_matrix, augmented_target, rcond=None
    )
    return FusionPolicy(
        names,
        "linear_residual",
        1.0,
        weights=tuple(float(value) for value in beta[1:]),
        bias=float(beta[0]),
        metadata={
            "candidate": "regression_residual_stacker",
            "regularization": float(regularization),
            "inference_normalization": "base_prediction_std",
            "training_targets": "outer_training_only",
        },
    )


def _fit_metric_scaled_residual_stacker(
    views: list[View], names: tuple[str, ...], regularization: float
) -> FusionPolicy:
    """Fit a Base-anchored residual ensemble on the reported metric scale.

    Unlike the generic residual stacker, this repair candidate has no free
    intercept: every correction must be supported by disagreement between a
    legal Context prediction and Base.  Normalising each outer-training view
    by its predeclared regression scale prevents high-variance datasets from
    dominating the fit while preserving a raw-scale residual update at
    inference.
    """

    design_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for view in views:
        scale = max(float(view.scale), 1.0e-12)
        residuals = np.column_stack([
            (np.asarray(view.members[name]) - np.asarray(view.base)) / scale
            for name in names
        ])
        target = (np.asarray(view.target, dtype=np.float64) - view.base) / scale
        # Give every dataset-condition equal mass regardless of row count.
        factor = 1.0 / math.sqrt(max(1, len(target)))
        design_rows.append(residuals * factor)
        target_rows.append(target * factor)
    matrix = np.vstack(design_rows)
    target = np.concatenate(target_rows)
    ridge = math.sqrt(float(regularization)) * np.eye(len(names))
    weights, *_ = np.linalg.lstsq(
        np.vstack([matrix, ridge]),
        np.concatenate([target, np.zeros(len(names), dtype=np.float64)]),
        rcond=None,
    )
    return FusionPolicy(
        names,
        "linear_residual",
        1.0,
        weights=tuple(float(value) for value in weights),
        bias=0.0,
        metadata={
            "candidate": "metric_scaled_residual_stacker",
            "regularization": float(regularization),
            "inference_normalization": "cancels_to_raw_context_residual",
            "training_normalization": "predeclared_regression_metric_scale",
            "intercept": "fixed_zero",
            "training_targets": "outer_training_only",
        },
    )


def _fit_binary_logit_residual_stacker(
    views: list[View], names: tuple[str, ...], regularization: float
) -> FusionPolicy:
    """Fit a binary logit residual stacker on outer-training labels only."""

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    epsilon = 1.0e-8
    for view in views:
        base = np.asarray(view.base, dtype=np.float64)
        if view.task != "classification" or base.ndim != 2 or base.shape[1] != 2:
            raise ValueError(
                "binary logit residual stacker requires two-class probabilities"
            )
        base_logit = np.log(
            np.clip(base[:, 1], epsilon, 1.0)
            / np.clip(base[:, 0], epsilon, 1.0)
        )
        residuals = []
        for name in names:
            member = np.asarray(view.members[name], dtype=np.float64)
            member_logit = np.log(
                np.clip(member[:, 1], epsilon, 1.0)
                / np.clip(member[:, 0], epsilon, 1.0)
            )
            residuals.append(member_logit - base_logit)
        feature_rows.append(
            np.column_stack([base_logit, *residuals])
        )
        classes = np.asarray(view.classes)
        if len(classes) != 2:
            raise ValueError("binary stacker requires exactly two classes")
        target_rows.append((np.asarray(view.target) == classes[1]).astype(np.float64))
        weight_rows.append(
            np.full(len(base), 1.0 / (len(views) * len(base)), dtype=np.float64)
        )
    feature = np.vstack(feature_rows)
    target = np.concatenate(target_rows)
    sample_weight = np.concatenate(weight_rows)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = feature[:, 0] + theta[0] + feature[:, 1:] @ theta[1:]
        probability = expit(eta)
        loss = np.logaddexp(0.0, eta) - target * eta
        penalty = float(regularization) * np.dot(theta[1:], theta[1:])
        value = float(np.dot(sample_weight, loss) + penalty)
        error = sample_weight * (probability - target)
        gradient = np.concatenate(
            [
                [float(np.sum(error))],
                feature[:, 1:].T @ error
                + 2.0 * float(regularization) * theta[1:],
            ]
        )
        return value, np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective,
        np.zeros(len(names) + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-8.0, 8.0)] * (len(names) + 1),
        options={"maxiter": 200, "ftol": 1.0e-12},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    return FusionPolicy(
        names,
        "binary_logit_residual",
        1.0,
        weights=tuple(float(value) for value in theta[1:]),
        bias=float(theta[0]),
        metadata={
            "candidate": "binary_logit_residual_stacker",
            "regularization": float(regularization),
            "training_targets": "outer_training_only",
            "inference_inputs": "base_and_context_probabilities_only",
            "optimizer_converged": bool(result.success),
        },
    )


def _fit_batch_router(
    views: list[View],
    names: tuple[str, ...],
    *,
    regularization: float,
    minimum_predicted_gain: float,
    feature_count: int | None = None,
) -> FusionPolicy:
    """Fit a strength-blind Context router from unlabeled batch diagnostics."""

    if not names:
        return FusionPolicy((), "mean", 0.0, metadata={"candidate": "base"})
    feature_rows: list[np.ndarray] = []
    gain_rows: list[np.ndarray] = []
    for view in views:
        bank = _bank(view)
        feature_rows.append(batch_member_features(bank, names))
        base_value = _primary(view, view.base)
        gain_rows.append(np.asarray([
            _primary(view, view.members[name]) - base_value for name in names
        ], dtype=np.float64))
    features = np.stack(feature_rows, axis=0)  # [views, members, features]
    gains = np.stack(gain_rows, axis=0)  # [views, members]
    if views[0].task == "classification":
        feature_names = CLASSIFICATION_BATCH_FEATURE_NAMES
        if not CLASSIFICATION_BATCH_PRIOR_ROUTER:
            feature_names = feature_names[:8]
            features = features[:, :, :8]
    else:
        feature_names = REGRESSION_BATCH_FEATURE_NAMES
        if feature_count is not None:
            if not 1 <= int(feature_count) <= len(feature_names):
                raise ValueError("invalid regression router feature count")
            feature_names = feature_names[: int(feature_count)]
            features = features[:, :, : int(feature_count)]
    intercepts = []
    coefficients = []
    means = []
    scales = []
    penalty = float(regularization)
    for member_index in range(len(names)):
        raw = features[:, member_index, :]
        mean = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0)
        scale = np.where(scale > 1.0e-8, scale, 1.0)
        normalized = (raw - mean) / scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        ridge = np.diag([0.0, *([penalty] * normalized.shape[1])])
        beta = np.linalg.solve(
            design.T @ design + ridge + 1.0e-10 * np.eye(design.shape[1]),
            design.T @ gains[:, member_index],
        )
        intercepts.append(float(beta[0]))
        coefficients.append(tuple(float(value) for value in beta[1:]))
        means.append(tuple(float(value) for value in mean))
        scales.append(tuple(float(value) for value in scale))
    router = BatchMemberRouter(
        feature_names=feature_names,
        intercepts=tuple(intercepts),
        coefficients=tuple(coefficients),
        feature_means=tuple(means),
        feature_scales=tuple(scales),
        minimum_predicted_gain=float(minimum_predicted_gain),
    )
    return FusionPolicy(
        names,
        "mean",
        1.0,
        batch_router=router,
        metadata={
            "candidate": "batch_member_router",
            "regularization": float(regularization),
            "minimum_predicted_gain": float(minimum_predicted_gain),
            "router_inputs": "unlabeled_prediction_diagnostics",
            "diagnostic_feature_count": len(feature_names),
        },
    )


def _fit_local_batch_router(
    views: list[View],
    names: tuple[str, ...],
    *,
    n_neighbors: int,
    neighbor_shrinkage: float,
    minimum_predicted_gain: float,
    feature_count: int | None = None,
) -> FusionPolicy:
    """Fit a nonlinear gain router on legal unlabeled prediction summaries.

    The reference gains come only from the current outer-training datasets.
    At inference, nearest conditions are found using prediction diagnostics;
    no benchmark strength, shift-presence signal, or query label is accepted.
    """

    if not names:
        return FusionPolicy((), "mean", 0.0, metadata={"candidate": "base"})
    feature_rows: list[np.ndarray] = []
    gain_rows: list[np.ndarray] = []
    for view in views:
        bank = _bank(view)
        feature_rows.append(batch_member_features(bank, names))
        base_value = _primary(view, view.base)
        gain_rows.append(np.asarray([
            _primary(view, view.members[name]) - base_value for name in names
        ], dtype=np.float64))
    features = np.stack(feature_rows, axis=0)
    gains = np.stack(gain_rows, axis=0)
    if views[0].task == "classification":
        feature_names = CLASSIFICATION_BATCH_FEATURE_NAMES
        if not CLASSIFICATION_BATCH_PRIOR_ROUTER:
            feature_names = feature_names[:8]
            features = features[:, :, :8]
    else:
        feature_names = REGRESSION_BATCH_FEATURE_NAMES
        if feature_count is not None:
            if not 1 <= int(feature_count) <= len(feature_names):
                raise ValueError("invalid regression router feature count")
            feature_names = feature_names[: int(feature_count)]
            features = features[:, :, : int(feature_count)]
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales = np.where(scales > 1.0e-8, scales, 1.0)
    router = BatchMemberRouter(
        feature_names=feature_names,
        intercepts=tuple(float(value) for value in np.mean(gains, axis=0)),
        coefficients=tuple(
            tuple(0.0 for _ in feature_names) for _ in names
        ),
        feature_means=tuple(
            tuple(float(value) for value in row) for row in means
        ),
        feature_scales=tuple(
            tuple(float(value) for value in row) for row in scales
        ),
        minimum_predicted_gain=float(minimum_predicted_gain),
        reference_features=tuple(
            np.asarray(features[:, member_index, :], dtype=np.float64).copy()
            for member_index in range(len(names))
        ),
        reference_gains=tuple(
            np.asarray(gains[:, member_index], dtype=np.float64).copy()
            for member_index in range(len(names))
        ),
        n_neighbors=int(n_neighbors),
        neighbor_shrinkage=float(neighbor_shrinkage),
    )
    return FusionPolicy(
        names,
        "mean",
        1.0,
        batch_router=router,
        metadata={
            "candidate": "local_batch_member_router",
            "n_neighbors": int(n_neighbors),
            "neighbor_shrinkage": float(neighbor_shrinkage),
            "minimum_predicted_gain": float(minimum_predicted_gain),
            "router_inputs": "unlabeled_prediction_diagnostics",
            "training_gain_source": "outer_training_datasets_only",
            "self_match_excluded_during_selection": True,
            "diagnostic_feature_count": len(feature_names),
        },
    )


def _fit_adaptive_gate(policy: FusionPolicy, views: list[View]) -> FusionPolicy:
    """Fit a metadata-blind gate from row- and batch-prediction diagnostics.

    The batch summaries are computed from the same unlabeled prediction bank
    that is available at inference.  They let one fixed policy react to the
    *observed* amount of Base/Context disagreement without receiving a
    benchmark strength, a shift-presence flag, or evaluation labels.
    """

    center_policy = FusionPolicy(
        policy.members,
        policy.aggregation,
        1.0,
        weights=policy.weights,
        batch_router=policy.batch_router,
        bias=policy.bias,
        metadata=policy.metadata,
    )
    features = []
    loss_a = []
    loss_b = []
    loss_c = []
    class_base_true = []
    class_center_true = []
    row_weights = []
    for view in views:
        bank = _bank(view)
        center = aggregate(bank, center_policy)
        diagnostics = prediction_diagnostics(bank, center_policy, center)
        per_row = np.column_stack([
            diagnostics["base_distance"],
            diagnostics["member_dispersion"],
            diagnostics["confidence_gain"],
            diagnostics["base_confidence"],
            diagnostics["center_confidence"],
            diagnostics["base_margin"],
            diagnostics["center_margin"],
            diagnostics["base_vote_fraction"],
            diagnostics["center_vote_fraction"],
        ])
        batch = np.asarray(
            [
                np.mean(diagnostics["consensus"]),
                np.mean(diagnostics["base_distance"]),
                np.mean(diagnostics["member_dispersion"]),
                np.mean(diagnostics["confidence_gain"]),
            ],
            dtype=np.float64,
        )
        features.append(
            np.column_stack(
                [
                    per_row,
                    np.broadcast_to(batch, (len(per_row), len(batch))),
                    np.full(
                        (len(per_row), 1),
                        float(bank.diagnostics.get("unlabeled_shift", 0.0)),
                    ),
                ]
            )
        )
        if view.task == "classification":
            classes = np.asarray(view.classes)
            positions = {label: pos for pos, label in enumerate(classes.tolist())}
            encoded = np.asarray([positions[label] for label in view.target], dtype=int)
            row_index = np.arange(len(encoded))
            class_base_true.append(bank.base[row_index, encoded])
            class_center_true.append(center[row_index, encoded])
        else:
            scale = max(float(view.scale), 1.0e-12)
            error = (bank.base - np.asarray(view.target, dtype=np.float64)) / scale
            delta = (center - bank.base) / scale
            loss_a.append(delta**2)
            loss_b.append(2.0 * error * delta)
            loss_c.append(error**2)
        origin_weight = (
            REGRESSION_GATE_ORIGIN_WEIGHT
            if view.task == "regression" and view.rho == 0
            else 1.0
        )
        row_weights.append(
            np.full(
                len(view.base),
                origin_weight / (len(views) * len(view.base)),
            )
        )
    feature = np.vstack(features)
    weight = np.concatenate(row_weights)
    weight /= np.sum(weight)
    is_classification = views[0].task == "classification"
    if is_classification:
        base_true = np.concatenate(class_base_true)
        center_true = np.concatenate(class_center_true)
    else:
        quadratic = np.concatenate(loss_a)
        linear = np.concatenate(loss_b)
        constant = np.concatenate(loss_c)

    def objective(theta: np.ndarray) -> float:
        logit = theta[0] + feature @ theta[1:]
        alpha = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        if is_classification:
            true_probability = (
                (1.0 - alpha) * base_true + alpha * center_true
            )
            loss = -np.log(np.clip(true_probability, 1.0e-12, 1.0))
        else:
            loss = quadratic * alpha**2 + linear * alpha + constant
        return float(np.sum(weight * loss) + 1.0e-4 * np.sum(theta[1:] ** 2))

    initial_alpha = float(np.clip(policy.alpha, 1.0e-4, 1.0 - 1.0e-4))
    initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
    starts = (
        np.asarray([initial_logit, *([0.0] * 14)], dtype=np.float64),
        np.asarray(
            [-2.0, 4.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0,
             0.0, 4.0, -2.0, 2.0, 0.0],
            dtype=np.float64,
        ),
        np.asarray(
            [-3.0, 8.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -2.0, 3.0,
             0.0, 8.0, -4.0, 4.0, 0.0],
            dtype=np.float64,
        ),
    )
    fitted = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-10.0, 10.0)] * len(start),
            options={"maxiter": 120, "ftol": 1.0e-10},
        )
        fitted.append((float(result.fun), np.asarray(result.x, dtype=np.float64)))
    _loss, theta = min(fitted, key=lambda item: item[0])
    gate = AdaptiveGate(
        intercept=float(theta[0]),
        base_distance=float(theta[1]),
        member_dispersion=float(theta[2]),
        confidence_gain=float(theta[3]),
        base_confidence=float(theta[4]),
        center_confidence=float(theta[5]),
        base_margin=float(theta[6]),
        center_margin=float(theta[7]),
        base_vote_fraction=float(theta[8]),
        center_vote_fraction=float(theta[9]),
        batch_consensus=float(theta[10]),
        batch_base_distance=float(theta[11]),
        batch_member_dispersion=float(theta[12]),
        batch_confidence_gain=float(theta[13]),
        unlabeled_shift=float(theta[14]),
        mode="absolute",
    )
    return FusionPolicy(
        policy.members,
        policy.aggregation,
        1.0,
        weights=policy.weights,
        gate=gate,
        batch_router=policy.batch_router,
        bias=policy.bias,
        metadata={
            **policy.metadata,
            "gate_variant": "fitted_prediction_diagnostics",
            "gate_training_objective": (
                "cross_entropy" if is_classification else "normalized_mse"
            ),
            "gate_origin_training_weight": (
                1.0 if is_classification else REGRESSION_GATE_ORIGIN_WEIGHT
            ),
            "gate_uses_unlabeled_source_shift": bool(
                np.ptp([
                    float(view.diagnostics.get("unlabeled_shift", 0.0))
                    for view in views
                ]) > 1.0e-12
            ),
        },
    )


def _regression_batch_gates() -> tuple[AdaptiveGate, ...]:
    """Reusable label-free severity gates for regression prediction banks."""

    return tuple(
        AdaptiveGate(
            intercept=-slope * threshold,
            batch_base_distance=slope,
            mode="absolute",
        )
        for slope in (4.0, 8.0, 12.0)
        for threshold in (0.15, 0.25, 0.40, 0.60)
    )


def _gate_variants(
    policy: FusionPolicy,
    task: str,
    views: list[View],
    *,
    required_anchor: str | None = None,
) -> list[FusionPolicy]:
    if not policy.members or policy.alpha <= 0.0:
        return [policy]
    # A batch router already predicts the Base-relative gain of every member
    # and falls back to Base when no gain clears its learned threshold.  The
    # large fixed-gate/decision-guard grid below would duplicate that safeguard
    # and multiply selection time without adding a new inference signal.
    if policy.batch_router is not None:
        return [policy, _fit_adaptive_gate(policy, views)]
    if task == "classification":
        gates = (
            None,
            *(
                AdaptiveGate(
                    intercept=-20.0 * threshold,
                    base_distance=20.0,
                    mode="absolute",
                )
                for threshold in (0.05, 0.10, 0.15, 0.20, 0.30)
            ),
            AdaptiveGate(
                intercept=-2.0,
                base_distance=12.0,
                member_dispersion=-4.0,
                confidence_gain=4.0,
                mode="absolute",
            ),
            AdaptiveGate(
                intercept=-3.0,
                base_distance=16.0,
                member_dispersion=-6.0,
                confidence_gain=6.0,
                mode="absolute",
            ),
        )
    else:
        gates = (
            None,
            AdaptiveGate(intercept=0.0, consensus=2.0, base_distance=1.0,
                         member_dispersion=-2.0),
            AdaptiveGate(intercept=-1.0, consensus=3.0, base_distance=2.0,
                         member_dispersion=-3.0),
            AdaptiveGate(intercept=0.5, consensus=2.0, member_dispersion=-4.0),
            *_regression_batch_gates(),
        )
    fixed = [
        FusionPolicy(
            policy.members,
            policy.aggregation,
            policy.alpha,
            weights=policy.weights,
            gate=gate,
            batch_router=policy.batch_router,
            bias=policy.bias,
            metadata={**policy.metadata, "gate_variant": index},
        )
        for index, gate in enumerate(gates)
    ]
    variants = [policy, *fixed, _fit_adaptive_gate(policy, views)]
    if task != "classification":
        return variants
    guarded = []
    for variant in variants:
        if not variant.members:
            continue
        member_count = len(variant.members)
        if required_anchor is None:
            guard_specs = [
                ("__base__", votes, confidence)
                for votes in sorted({max(1, int(math.ceil(member_count / 2))), member_count})
                for confidence in (-0.05, -0.02, 0.0, 0.02, 0.05)
            ]
        else:
            guard_specs = []
        anchor = required_anchor or "shared/subspace"
        if anchor in variant.members and member_count >= 2:
            residual = len(variant.members) - 1
            guard_specs.extend(
                (anchor, votes, confidence)
                for votes in sorted({max(1, int(math.ceil(residual / 2))), residual})
                for confidence in (-0.10, -0.05, -0.02, 0.0, 0.02)
            )
        for anchor, votes, confidence in guard_specs:
            guarded.append(
                FusionPolicy(
                    variant.members,
                    variant.aggregation,
                    variant.alpha,
                    weights=variant.weights,
                    gate=variant.gate,
                    batch_router=variant.batch_router,
                    decision_guard=DecisionGuard(
                        anchor,
                        min_residual_votes=votes,
                        min_confidence_gain=confidence,
                        restore_unverified_prediction=True,
                    ),
                    bias=variant.bias,
                    metadata={
                        **variant.metadata,
                        "decision_guard_anchor": anchor,
                        "decision_guard_votes": votes,
                        "decision_guard_confidence": confidence,
                    },
                )
            )
    return guarded if required_anchor is not None else [*variants, *guarded]


def _select_policy(
    views: list[View],
    max_members: int | None,
    *,
    required_anchor: str | None = None,
    fallback_policies: Iterable[FusionPolicy] = (),
    base_downside_penalty: float = DOWNSIDE_PENALTY,
    tail_risk_weight: float = TAIL_RISK_WEIGHT,
    origin_reference_mean: float | None = None,
    origin_noninferiority_tolerance: float = 0.0,
    enable_local_router: bool = False,
    fast_local_router: bool = False,
    fast_candidate_search: bool = False,
    skip_gate_search: bool = False,
    allow_base_fallback: bool = True,
    require_nonbase: bool = False,
    group_reference_member: str | None = None,
    candidate_fit_views: list[View] | None = None,
    enforce_group_noninferiority: bool | None = None,
    group_noninferiority_margin: float | None = None,
    metric_residual_alphas: tuple[float, ...] | None = None,
    metric_residual_members: tuple[str, ...] | None = None,
) -> tuple[FusionPolicy, dict[str, Any]]:
    if not views:
        raise ValueError("empty outer-training view")
    group_ni_enabled = GROUP_NONINFERIORITY_ENABLED and (
        True
        if enforce_group_noninferiority is None
        else bool(enforce_group_noninferiority)
    )
    fast_candidate_search = bool(
        fast_candidate_search
        or (group_ni_enabled and COMPACT_SAFETY_SEARCH)
    )
    names = tuple(views[0].members)
    if any(tuple(view.members) != names for view in views):
        raise ValueError("Context schema changed inside one selection scope")
    fit_views = views if candidate_fit_views is None else candidate_fit_views
    if not fit_views:
        raise ValueError("empty candidate-fitting view")
    if any(tuple(view.members) != names for view in fit_views):
        raise ValueError("Context schema changed between fit and audit views")
    effective_group_margin = (
        GROUP_NONINFERIORITY_MARGIN
        if group_noninferiority_margin is None
        else float(group_noninferiority_margin)
    )
    complexity_ratio = max(float(len(names)) / 5.0, 1.0)
    downside_penalty = float(base_downside_penalty) * (
        1.0 + 0.75 * math.log(complexity_ratio)
    )
    origin_views = [view for view in views if view.rho == 0]
    if origin_reference_mean is not None and not origin_views:
        raise ValueError("Origin noninferiority requires rho=0 training views")

    def origin_mean(policy: FusionPolicy) -> float:
        if not origin_views:
            return float("nan")
        return float(np.mean([
            _primary(view, predict(_bank(view, policy.members), policy))
            for view in origin_views
        ]))

    def origin_eligible(policy: FusionPolicy) -> bool:
        if origin_reference_mean is None:
            return True
        return (
            origin_mean(policy)
            >= float(origin_reference_mean) - float(origin_noninferiority_tolerance)
        )

    def group_eligible(policy: FusionPolicy) -> bool:
        if not group_ni_enabled or not policy.members:
            return True
        gains = _group_metric_gains(
            views, policy, reference_member=group_reference_member
        )
        # A routed policy may intentionally make no update for an unsupported
        # group, yielding exact Base performance.  Such fallback groups are
        # admissible; every group that is actually changed must clear the
        # configured positive safety margin.
        return bool(gains) and all(
            (GROUP_NONINFERIORITY_ALLOW_EXACT and abs(gain) <= 1.0e-12)
            or gain >= effective_group_margin
            for gain in gains.values()
        )
    singleton = []
    for name in names:
        singleton_modes = (("mean", 1.0),) if fast_candidate_search else tuple(
            (mode, alpha)
            for mode in ("mean", "median")
            for alpha in ALPHAS[1:]
        )
        options = [
            (
                *_policy_score(
                    views,
                    FusionPolicy((name,), mode, alpha),
                    downside_penalty=downside_penalty,
                    tail_risk_weight=tail_risk_weight,
                ),
                name,
            )
            for mode, alpha in singleton_modes
        ]
        if views[0].task == "regression" and not fast_candidate_search:
            options.extend(
                (
                    *_policy_score(
                        views,
                        FusionPolicy((name,), "mean", 1.0, gate=gate),
                        downside_penalty=downside_penalty,
                        tail_risk_weight=tail_risk_weight,
                    ),
                    name,
                )
                for gate in _regression_batch_gates()
            )
        best = max(
            options,
            key=lambda item: item[:3],
        )
        singleton.append((best[0], best[1], best[2], name))
    ranking = tuple(item[3] for item in sorted(singleton, reverse=True))
    limit = len(ranking) if max_members is None else min(max_members, len(ranking))
    sizes = sorted({1, 2, 3, 5, 8, 12, limit})
    sizes = [size for size in sizes if size <= limit]
    if fast_candidate_search:
        sizes = sorted({1, min(3, limit), limit})
    if ULTRA_COMPACT_SAFETY_SEARCH:
        sizes = [limit]
    candidates = [
        *(
            [FusionPolicy((), "mean", 0.0, metadata={"candidate": "base"})]
            if allow_base_fallback and not require_nonbase
            else []
        ),
        *(
            FusionPolicy(
                policy.members,
                policy.aggregation,
                policy.alpha,
                weights=policy.weights,
                gate=policy.gate,
                member_gate=policy.member_gate,
                batch_router=policy.batch_router,
                decision_guard=policy.decision_guard,
                bias=policy.bias,
                metadata={**policy.metadata, "candidate": "lower_tier_fallback"},
            )
            for policy in fallback_policies
            if (allow_base_fallback and not require_nonbase) or policy.members
            if all(name in names for name in policy.members)
        ),
    ]
    for size in sizes:
        if required_anchor is None:
            selected = ranking[:size]
        else:
            residual = tuple(name for name in ranking if name != required_anchor)
            selected = (required_anchor, *residual[: max(0, size - 1)])
        for mode in ("mean", "median"):
            for alpha in (
                ROUTER_ALPHAS_OVERRIDE
                or (
                    (0.25, 0.5, 0.75, 1.0)
                    if fast_candidate_search
                    else ALPHAS[1:]
                )
            ):
                candidates.append(
                    FusionPolicy(
                        selected,
                        mode,
                        alpha,
                        metadata={"candidate": "ranked_prefix", "prefix": size},
                    )
                )
        if not fast_candidate_search:
            for regularization in REGULARIZATIONS:
                candidates.append(_fit_convex(views, selected, regularization))
        if (
            views[0].task == "classification"
            and not fast_candidate_search
            and all(
                np.asarray(view.base).ndim == 2
                and np.asarray(view.base).shape[1] == 2
                for view in views
            )
        ):
            for regularization in (0.001, 0.01, 0.1, 1.0, 10.0):
                candidates.append(
                    _fit_binary_logit_residual_stacker(
                        views, selected, regularization
                    )
                )
        if views[0].task == "regression" and (
            not fast_candidate_search or size == limit
        ):
            for regularization in (
                (0.1, 1.0, 10.0)
                if fast_candidate_search
                else (0.01, 0.1, 1.0, 10.0, 100.0)
            ):
                candidates.append(
                    _fit_regression_residual_stacker(
                        views, selected, regularization
                    )
                )
        if not fast_candidate_search or group_ni_enabled:
            router_regularizations = (
                ((1.0, 10.0) if fast_candidate_search else (0.1, 1.0, 10.0, 100.0))
                if views[0].task == "classification"
                else ((0.1, 1.0) if fast_candidate_search else (0.01, 0.1, 1.0))
            )
            if ULTRA_COMPACT_SAFETY_SEARCH:
                router_regularizations = (10.0,) if views[0].task == "classification" else (1.0,)
            router_feature_counts = (
                (None,)
                if views[0].task == "classification"
                else (8, len(REGRESSION_BATCH_FEATURE_NAMES))
            )
            for feature_count in router_feature_counts:
                for regularization in router_regularizations:
                    for minimum_gain in ROUTER_MIN_GAIN_GRID:
                        router = _fit_batch_router(
                            views,
                            selected,
                            regularization=regularization,
                            minimum_predicted_gain=minimum_gain,
                            feature_count=feature_count,
                        )
                        router_alphas = ROUTER_ALPHAS_OVERRIDE or (
                            ((0.25, 0.5, 1.0) if fast_candidate_search else ALPHAS[1:])
                            if views[0].task == "classification"
                            else REGRESSION_ROUTER_ALPHAS
                        )
                        for alpha in router_alphas:
                            candidates.append(
                                replace(
                                    router,
                                    alpha=float(alpha),
                                    metadata={
                                        **router.metadata,
                                        "router_residual_alpha": float(alpha),
                                    },
                                )
                            )
        if enable_local_router and (not fast_local_router or size == limit):
            local_feature_counts = (
                (None,)
                if views[0].task == "classification"
                else (
                    (len(REGRESSION_BATCH_FEATURE_NAMES),)
                    if fast_local_router
                    else (8, len(REGRESSION_BATCH_FEATURE_NAMES))
                )
            )
            for feature_count in local_feature_counts:
                local_template = _fit_local_batch_router(
                    views,
                    selected,
                    n_neighbors=5,
                    neighbor_shrinkage=0.5,
                    minimum_predicted_gain=0.0,
                    feature_count=feature_count,
                )
                assert local_template.batch_router is not None
                for n_neighbors in (5, 11, 21):
                    for shrinkage in ((1.0,) if fast_local_router else (0.5, 1.0)):
                        for minimum_gain in ROUTER_MIN_GAIN_GRID:
                            local_router = replace(
                                local_template.batch_router,
                                n_neighbors=n_neighbors,
                                neighbor_shrinkage=shrinkage,
                                minimum_predicted_gain=minimum_gain,
                            )
                            candidates.append(
                                replace(
                                    local_template,
                                    batch_router=local_router,
                                    metadata={
                                        **local_template.metadata,
                                        "n_neighbors": n_neighbors,
                                        "neighbor_shrinkage": shrinkage,
                                        "minimum_predicted_gain": minimum_gain,
                                    },
                                )
                            )
    if (
        METRIC_RESIDUAL_SEARCH
        and candidate_fit_views is not None
        and views[0].task == "regression"
    ):
        metric_candidates: list[FusionPolicy] = []
        if metric_residual_members is not None:
            unknown = [name for name in metric_residual_members if name not in names]
            if unknown:
                raise KeyError(f"metric residual override has illegal Contexts: {unknown}")
            if len(metric_residual_members) > limit:
                raise ValueError("metric residual override exceeds the tier Context limit")
            metric_subsets = [tuple(metric_residual_members)]
        elif len(names) <= 6:
            subset_sizes = (
                [
                    size
                    for size in METRIC_RESIDUAL_MEMBER_SIZES
                    if 1 <= size <= min(limit, len(names))
                ]
                or list(range(1, min(limit, len(names)) + 1))
            )
            metric_subsets = [
                subset
                for size in subset_sizes
                for subset in combinations(names, size)
            ]
        else:
            metric_subsets = []
            subset_sizes = (
                [
                    size
                    for size in METRIC_RESIDUAL_MEMBER_SIZES
                    if 1 <= size <= limit
                ]
                or sizes
            )
            for size in subset_sizes:
                if required_anchor is None:
                    subset = ranking[:size]
                else:
                    residual = tuple(name for name in ranking if name != required_anchor)
                    subset = (required_anchor, *residual[: max(0, size - 1)])
                if subset not in metric_subsets:
                    metric_subsets.append(subset)
        metric_alphas = metric_residual_alphas or ROUTER_ALPHAS_OVERRIDE or (
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0
        )
        for subset in metric_subsets:
            for regularization in METRIC_RESIDUAL_REGULARIZATIONS:
                template = _fit_metric_scaled_residual_stacker(
                    fit_views, tuple(subset), regularization
                )
                for alpha in metric_alphas:
                    metric_candidates.append(
                        replace(
                            template,
                            alpha=float(alpha),
                            metadata={
                                **template.metadata,
                                "residual_alpha": float(alpha),
                                "fit_view_count": len(fit_views),
                            },
                        )
                    )
        if METRIC_RESIDUAL_ONLY:
            candidates = metric_candidates
        else:
            candidates.extend(metric_candidates)
    unique: dict[tuple[Any, ...], FusionPolicy] = {}
    for policy in candidates:
        key = (
            policy.members,
            policy.aggregation,
            round(policy.alpha, 12),
            tuple(round(value, 12) for value in policy.weights),
            round(policy.bias, 12),
            None if policy.batch_router is None else (
                round(policy.batch_router.minimum_predicted_gain, 8),
                tuple(round(value, 8) for value in policy.batch_router.intercepts),
            ),
            tuple(sorted(policy.metadata.items())),
        )
        unique.setdefault(key, policy)
    origin_candidates = [
        policy for policy in unique.values() if origin_eligible(policy)
    ]
    if not origin_candidates:
        raise RuntimeError(
            "no candidate satisfies the training-Origin noninferiority constraint"
        )
    scored_by_risk = sorted(
        (
            (
                *_policy_score(
                    views,
                    policy,
                    downside_penalty=downside_penalty,
                    tail_risk_weight=tail_risk_weight,
                ),
                policy,
            )
            for policy in origin_candidates
        ),
        key=lambda item: item[:3],
        reverse=True,
    )
    if group_ni_enabled:
        # Risk-score ordering is independent of this constraint.  Test policies
        # lazily in that order so the expensive paired Accuracy/AUC or R2/nMAE
        # audit is performed only until enough top safe finalists are found.
        safe_nonbase = []
        safe_target = (
            max(2, GATED_FINALIST_COUNT)
            if ULTRA_COMPACT_SAFETY_SEARCH
            else max(4, GATED_FINALIST_COUNT * 3)
        )
        for item in scored_by_risk:
            policy = item[-1]
            if policy.members and group_eligible(policy):
                safe_nonbase.append(item)
                if len(safe_nonbase) >= safe_target:
                    break
        safe_base = (
            []
            if require_nonbase
            else [item for item in scored_by_risk if not item[-1].members][:1]
        )
        scored_candidates = sorted(
            [*safe_nonbase, *safe_base],
            key=lambda item: item[:3],
            reverse=True,
        )
    else:
        scored_candidates = scored_by_risk
    if not scored_candidates:
        raise RuntimeError("no candidate satisfies group noninferiority")
    if skip_gate_search:
        score, mean_value, worst_gain, chosen = scored_candidates[0]
        return chosen, {
            "train_risk_score": score,
            "train_primary_mean": mean_value,
            "train_worst_dataset_condition_gain": worst_gain,
            "candidate_count": len(unique),
            "downside_penalty": downside_penalty,
            "tail_risk_weight": float(tail_risk_weight),
            "origin_noninferiority_enabled": origin_reference_mean is not None,
            "origin_reference_mean": origin_reference_mean,
            "origin_noninferiority_tolerance": float(origin_noninferiority_tolerance),
            "selected_train_origin_mean": origin_mean(chosen),
            "group_noninferiority_enabled": group_ni_enabled,
            "group_noninferiority_margin": effective_group_margin,
            "group_noninferiority_include_origin": GROUP_NONINFERIORITY_INCLUDE_ORIGIN,
            "selected_train_group_metric_gains": _group_metric_gains(
                views, chosen, reference_member=group_reference_member
            ),
            "ranking": list(ranking),
            "gate_search_skipped": True,
        }
    # Evaluate several structurally distinct non-Base experts with label-free
    # prediction gates.  A single globally best fixed Context can be a poor
    # detector candidate even when another Context separates shifted from
    # unshifted batches cleanly.
    base_finalists = [item for item in scored_candidates if not item[-1].members]
    finalists = base_finalists[:1]
    signatures: set[tuple[Any, ...]] = set()
    for item in scored_candidates:
        policy = item[-1]
        if not policy.members:
            continue
        signature = (
            policy.members,
            policy.aggregation,
            tuple(round(float(value), 8) for value in policy.weights),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        finalists.append(item)
        if len(finalists) >= GATED_FINALIST_COUNT:
            break
    if not finalists:
        finalists = scored_candidates[:1]
    gated = []
    for *_score, policy in finalists:
        gated.extend(
            _gate_variants(
                policy,
                views[0].task,
                views,
                required_anchor=required_anchor,
            )
        )
    if not gated:
        gated = [FusionPolicy((), "mean", 0.0, metadata={"candidate": "base"})]
    eligible_gated = [
        policy
        for policy in gated
        if origin_eligible(policy) and group_eligible(policy)
    ]
    if not eligible_gated:
        # Every fitted gate can change the training-Origin behavior.  Falling
        # back to an already eligible ungated finalist preserves the explicit
        # noninferiority contract without consulting the outer test fold.
        eligible_gated = [item[-1] for item in finalists]
    ranked = sorted(
        (
            (
                *_policy_score(
                    views,
                    policy,
                    downside_penalty=downside_penalty,
                    tail_risk_weight=tail_risk_weight,
                ),
                -len(policy.members),
                policy,
            )
            for policy in eligible_gated
        ),
        key=lambda item: item[:-1],
        reverse=True,
    )
    score, mean_value, worst_gain, _negative_members, chosen = ranked[0]
    chosen_origin_mean = origin_mean(chosen)
    return chosen, {
        "train_risk_score": score,
        "train_primary_mean": mean_value,
        "train_worst_dataset_condition_gain": worst_gain,
        "candidate_count": len(unique),
        "downside_penalty": downside_penalty,
        "tail_risk_weight": float(tail_risk_weight),
        "origin_noninferiority_enabled": origin_reference_mean is not None,
        "origin_reference_mean": origin_reference_mean,
        "origin_noninferiority_tolerance": float(origin_noninferiority_tolerance),
        "selected_train_origin_mean": chosen_origin_mean,
        "group_noninferiority_enabled": group_ni_enabled,
        "group_noninferiority_margin": effective_group_margin,
        "group_noninferiority_include_origin": GROUP_NONINFERIORITY_INCLUDE_ORIGIN,
        "selected_train_group_metric_gains": _group_metric_gains(
            views, chosen, reference_member=group_reference_member
        ),
        "ranking": list(ranking),
    }


def _policy_row(task: str, fold: int, method: str, family: str, policy: FusionPolicy,
                diagnostics: dict[str, Any], train: list[View]) -> dict[str, Any]:
    gate = None if policy.gate is None else {
        key: getattr(policy.gate, key)
        for key in (
            "intercept", "consensus", "base_distance", "member_dispersion",
            "confidence_gain", "base_confidence", "center_confidence",
            "base_margin", "center_margin", "base_vote_fraction",
            "center_vote_fraction", "batch_consensus", "batch_base_distance",
            "batch_member_dispersion", "batch_confidence_gain",
            "min_alpha", "max_alpha", "mode",
        )
    }
    decision_guard = None if policy.decision_guard is None else {
        "anchor_member": policy.decision_guard.anchor_member,
        "min_residual_votes": policy.decision_guard.min_residual_votes,
        "min_confidence_gain": policy.decision_guard.min_confidence_gain,
        "restore_unverified_prediction": (
            policy.decision_guard.restore_unverified_prediction
        ),
    }
    return {
        "task": task,
        "outer_fold": fold,
        "method": method,
        "family_scope": family,
        "strength_used_for_selection": False,
        "outer_test_labels_used": False,
        "outer_train_datasets": len({view.data_id for view in train}),
        "outer_train_conditions": len(train),
        **diagnostics,
        "members": list(policy.members),
        "aggregation": policy.aggregation,
        "alpha": policy.alpha,
        "weights": list(policy.weights),
        "bias": policy.bias,
        "gate": gate,
        "decision_guard": decision_guard,
        "batch_router": (
            None
            if policy.batch_router is None
            else {
                "feature_names": list(policy.batch_router.feature_names),
                "minimum_predicted_gain": policy.batch_router.minimum_predicted_gain,
                "member_models": len(policy.batch_router.intercepts),
                "inference_inputs": "unlabeled_prediction_diagnostics",
                "router_kind": (
                    "local_neighbor_gain"
                    if policy.batch_router.reference_features
                    else "linear_ridge_gain"
                ),
                "n_neighbors": policy.batch_router.n_neighbors,
                "neighbor_shrinkage": policy.batch_router.neighbor_shrinkage,
            }
        ),
        "metadata": dict(policy.metadata),
    }


def build(
    task: str,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    backbone_display: str = "TabICL v2",
    checkpoint: str = "tabicl-regressor-v2-20260212.ckpt",
) -> dict[str, Any]:
    _BANK_CACHE.clear()
    _VIEW_REFS.clear()
    clear_batch_member_feature_cache()
    common_fields = {"task", "base_prediction", "context_predictions"}
    CONTEXT_PROTOCOL.validate_request(
        tier="shared", family=None, fields=common_fields
    )
    records = source._load_records(task)
    families = source.PROBLEMS[task]
    ids = sorted({record.data_id for record in records})
    raw = {(record.data_id, record.problem, record.rho): record for record in records}
    views: dict[tuple[int, str, int, str], View] = {}
    for data_id in ids:
        for family in families:
            for rho in RHOS:
                record = (
                    raw[(data_id, "original", 0)]
                    if rho == 0 else raw[(data_id, family, rho)]
                )
                for method in ("shared", "specific", "pro"):
                    views[(data_id, family, rho, method)] = _view(
                        record,
                        family,
                        method,
                    )

    policies: dict[tuple[int, str, str], FusionPolicy] = {}
    selection_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train_ids = {data_id for data_id in ids if raw[(data_id, "original", 0)].fold != fold}
        shared_train = [
            views[(data_id, family, rho, "shared")]
            for data_id in train_ids for family in families for rho in RHOS
        ]
        shared_select = (
            [
                view
                for view in shared_train
                if (view.family, int(view.rho)) in SHARED_FOCUS_GROUPS
            ]
            if SHARED_FOCUS_GROUPS
            else shared_train
        )
        shared_fit = (
            [
                view
                for view in shared_train
                if (view.family, int(view.rho)) in SHARED_FIT_GROUPS
            ]
            if SHARED_FIT_GROUPS
            else None
        )
        shared_spec = CONTEXT_PROTOCOL.tier("shared")
        policy, diagnostics = _select_policy(
            shared_select,
            max_members=shared_spec.max_contexts,
            require_nonbase=bool(SHARED_FOCUS_GROUPS),
            enable_local_router=(
                FOCUS_ENABLE_LOCAL_ROUTER and bool(SHARED_FOCUS_GROUPS)
            ),
            fast_local_router=FOCUS_ENABLE_LOCAL_ROUTER,
            candidate_fit_views=shared_fit,
            skip_gate_search=(
                METRIC_RESIDUAL_SKIP_GATE and shared_fit is not None
            ),
            enforce_group_noninferiority=bool(SHARED_FOCUS_GROUPS),
            group_noninferiority_margin=SHARED_GROUP_NONINFERIORITY_MARGIN,
            metric_residual_alphas=SHARED_METRIC_RESIDUAL_ALPHAS,
            metric_residual_members=SHARED_METRIC_RESIDUAL_MEMBERS,
        )
        policies[(fold, "shared", "all_families")]=policy
        selection_rows.append(
            _policy_row(task, fold, "shared", "all_families", policy, diagnostics, shared_select)
        )
        print(f"[{task}] fold={fold} shared members={len(policy.members)} alpha={policy.alpha:.3f}", flush=True)
        for family in families:
            CONTEXT_PROTOCOL.validate_request(
                tier="specific",
                family=family,
                fields={*common_fields, "family"},
            )
            CONTEXT_PROTOCOL.validate_request(
                tier="pro",
                family=family,
                fields={*common_fields, "family"},
            )
            specific_train = [
                views[(data_id, family, rho, "specific")]
                for data_id in train_ids for rho in RHOS
            ]
            family_specific_focus = {
                group for group in SPECIFIC_FOCUS_GROUPS if group[0] == family
            }
            specific_select = (
                [
                    view
                    for view in specific_train
                    if (view.family, int(view.rho)) in family_specific_focus
                ]
                if family_specific_focus
                else specific_train
            )
            family_specific_fit = {
                group for group in SPECIFIC_FIT_GROUPS if group[0] == family
            }
            specific_fit = (
                [
                    view
                    for view in specific_train
                    if (view.family, int(view.rho)) in family_specific_fit
                ]
                if family_specific_fit
                else None
            )
            specific_origin_reference = None
            if task == "classification" and CLASS_ORIGIN_NONINFERIORITY:
                shared_origin_values = []
                for data_id in train_ids:
                    shared_origin = views[(data_id, family, 0, "shared")]
                    shared_origin_values.append(
                        _primary(
                            shared_origin,
                            predict(_bank(shared_origin, policy.members), policy),
                        )
                    )
                specific_origin_reference = float(np.mean(shared_origin_values))
            specific_spec = CONTEXT_PROTOCOL.tier("specific")
            specific_policy, diagnostics = _select_policy(
                specific_select,
                max_members=specific_spec.max_contexts,
                origin_reference_mean=specific_origin_reference,
                origin_noninferiority_tolerance=CLASS_ORIGIN_TOLERANCE,
                require_nonbase=bool(family_specific_focus),
                candidate_fit_views=specific_fit,
                skip_gate_search=(
                    METRIC_RESIDUAL_SKIP_GATE and specific_fit is not None
                ),
                enforce_group_noninferiority=bool(family_specific_focus),
                group_noninferiority_margin=SPECIFIC_GROUP_NONINFERIORITY_MARGIN,
                fast_candidate_search=(
                    FAST_NONFOCUS_SEARCH and not family_specific_focus
                ),
                metric_residual_alphas=SPECIFIC_METRIC_RESIDUAL_ALPHAS,
                metric_residual_members=_parse_member_subset(
                    f"TABACE_SPECIFIC_{family.upper()}_METRIC_RESIDUAL_MEMBERS"
                ),
            )
            policies[(fold, "specific", family)] = specific_policy
            selection_rows.append(
                _policy_row(
                    task, fold, "specific", family,
                    specific_policy, diagnostics, specific_select,
                )
            )
            print(
                f"[{task}] fold={fold} specific/{family} "
                f"members={len(specific_policy.members)} alpha={specific_policy.alpha:.3f}",
                flush=True,
            )

            pro_train = []
            for data_id in train_ids:
                for rho in RHOS:
                    shared_source = views[(data_id, family, rho, "shared")]
                    specific_source = views[(data_id, family, rho, "specific")]
                    shared_prediction = predict(
                        _bank(shared_source, policy.members), policy
                    )
                    specific_prediction = predict(
                        _bank(specific_source, specific_policy.members),
                        specific_policy,
                    )
                    pro_view = views[(data_id, family, rho, "pro")]
                    pro_view = _with_member(
                        pro_view, "tier/shared", shared_prediction
                    )
                    pro_view = _with_member(
                        pro_view, "tier/specific", specific_prediction
                    )
                    pro_train.append(pro_view)
            pro_spec = CONTEXT_PROTOCOL.tier("pro")
            pro_origin_reference = None
            if task == "classification" and CLASS_ORIGIN_NONINFERIORITY:
                pro_origin_reference = float(np.mean([
                    _primary(view, view.members["tier/specific"])
                    for view in pro_train
                    if view.rho == 0
                ]))
            exact_specific_fallback = FusionPolicy(
                ("tier/specific",),
                "mean",
                1.0,
                metadata={"candidate": "exact_lower_tier_output"},
            )
            family_pro_focus = {
                group for group in PRO_FOCUS_GROUPS if group[0] == family
            }
            pro_select = (
                [
                    view
                    for view in pro_train
                    if (view.family, int(view.rho)) in family_pro_focus
                ]
                if family_pro_focus
                else pro_train
            )
            family_pro_fit = {
                group for group in PRO_FIT_GROUPS if group[0] == family
            }
            pro_fit = (
                [
                    view
                    for view in pro_train
                    if (view.family, int(view.rho)) in family_pro_fit
                ]
                if family_pro_fit
                else None
            )
            pro_policy, diagnostics = _select_policy(
                pro_select,
                max_members=pro_spec.max_contexts,
                fallback_policies=(policy, specific_policy, exact_specific_fallback),
                origin_reference_mean=pro_origin_reference,
                origin_noninferiority_tolerance=CLASS_ORIGIN_TOLERANCE,
                require_nonbase=bool(family_pro_focus),
                group_reference_member=(
                    "tier/shared"
                    if PRO_COMPARE_TO_SHARED and family_pro_focus
                    else None
                ),
                enable_local_router=(
                    FOCUS_ENABLE_LOCAL_ROUTER and bool(family_pro_focus)
                ),
                fast_local_router=FOCUS_ENABLE_LOCAL_ROUTER,
                candidate_fit_views=pro_fit,
                skip_gate_search=(
                    METRIC_RESIDUAL_SKIP_GATE and pro_fit is not None
                ),
                enforce_group_noninferiority=bool(family_pro_focus),
                group_noninferiority_margin=PRO_GROUP_NONINFERIORITY_MARGIN,
                fast_candidate_search=(
                    FAST_NONFOCUS_SEARCH and not family_pro_focus
                ),
                metric_residual_alphas=PRO_METRIC_RESIDUAL_ALPHAS,
                metric_residual_members=_parse_member_subset(
                    f"TABACE_PRO_{family.upper()}_METRIC_RESIDUAL_MEMBERS"
                ),
            )
            policies[(fold, "pro", family)] = pro_policy
            selection_rows.append(
                _policy_row(
                    task, fold, "pro", family,
                    pro_policy, diagnostics, pro_select,
                )
            )
            print(
                f"[{task}] fold={fold} pro/{family} "
                f"members={len(pro_policy.members)} alpha={pro_policy.alpha:.3f}",
                flush=True,
            )

    detail: list[dict[str, Any]] = []
    for data_id in ids:
        fold = raw[(data_id, "original", 0)].fold
        for family in families:
            for rho in RHOS:
                base_view = views[(data_id, family, rho, "shared")]
                predictions = {"base": base_view.base}
                shared_policy = policies[(fold, "shared", "all_families")]
                shared_view = views[(data_id, family, rho, "shared")]
                predictions["shared"] = infer_context_view(
                    tier="shared", view=shared_view, policy=shared_policy
                )
                specific_policy = policies[(fold, "specific", family)]
                specific_view = views[(data_id, family, rho, "specific")]
                predictions["specific"] = infer_context_view(
                    tier="specific", view=specific_view, policy=specific_policy
                )
                pro_policy = policies[(fold, "pro", family)]
                pro_view = views[(data_id, family, rho, "pro")]
                pro_view = _with_member(
                    pro_view, "tier/shared", predictions["shared"]
                )
                pro_view = _with_member(
                    pro_view, "tier/specific", predictions["specific"]
                )
                predictions["pro"] = infer_context_view(
                    tier="pro", view=pro_view, policy=pro_policy
                )
                row: dict[str, Any] = {
                    "task": task,
                    "data_id": data_id,
                    "dataset": base_view.dataset,
                    "outer_fold": fold,
                    "family": family,
                    "rho": rho,
                    "shared_family_known": False,
                    "specific_family_known": True,
                    "pro_family_known": True,
                    "strength_used_for_selection_or_inference": False,
                    "outer_test_labels_used_for_selection": False,
                    "origin_forced_to_other_method": False,
                }
                for method, prediction in predictions.items():
                    for metric, value in source._metrics(
                        raw[(data_id, "original", 0)] if rho == 0 else raw[(data_id, family, rho)],
                        prediction,
                    ).items():
                        row[f"{method}_{metric}"] = value
                detail.append(row)

    summary: list[dict[str, Any]] = []
    for family in families:
        for rho in RHOS:
            cell = [row for row in detail if row["family"] == family and row["rho"] == rho]
            out: dict[str, Any] = {"task": task, "family": family, "rho": rho, "n_datasets": len(cell)}
            for method in METHODS:
                for metric in METRICS[task]:
                    out[f"mean_{method}_{metric}"] = float(
                        np.mean([row[f"{method}_{metric}"] for row in cell])
                    )
            summary.append(out)

    task_output = output_root / task
    _write_csv(task_output / "detail_oof.csv", detail)
    _write_csv(task_output / "summary_oof.csv", summary)
    _write_csv(task_output / "outer_selections.csv", selection_rows)
    audit = {
        "api_version": CONTEXT_PROTOCOL.api_version,
        "protocol_id": CONTEXT_PROTOCOL.protocol_id,
        "task": task,
        "datasets": len(ids),
        "detail_rows": len(detail),
        "outer_folds": 5,
        "shared_policy_count": 5,
        "specific_policy_count": 5 * len(families),
        "pro_policy_count": 5 * len(families),
        "shared_uses_family_or_strength": False,
        "shared_uses_shift_presence_or_proxy": False,
        "specific_uses_strength": False,
        "specific_uses_shift_presence_or_proxy": False,
        "pro_uses_strength": False,
        "pro_uses_shift_presence_or_proxy": False,
        "backbone": backbone_display,
        "checkpoint": checkpoint,
        "hierarchical_context_tiers": [
            f"Base ({backbone_display})",
            "TabACE-Shared",
            "TabACE-Specific",
            "TabACE-Pro",
        ],
        "tier_composition": "common_frozen_base_with_legal_lower_tier_aggregates",
        "adaptive_tuning": {
            "members": True,
            "aggregation": True,
            "alpha": True,
            "nonnegative_weights": True,
            "prediction_diagnostic_gate": True,
            "batch_context_router": True,
            "classification_batch_router_feature_count": (
                8 if task == "classification" else None
            ),
            "pool_size_downside_penalty": True,
            "metric_scaled_residual_search": METRIC_RESIDUAL_SEARCH,
            "metric_scaled_residual_only": METRIC_RESIDUAL_ONLY,
            "metric_scaled_residual_skip_gate": METRIC_RESIDUAL_SKIP_GATE,
            "metric_scaled_residual_regularizations": list(
                METRIC_RESIDUAL_REGULARIZATIONS
            ),
            "metric_scaled_residual_member_sizes": list(
                METRIC_RESIDUAL_MEMBER_SIZES
            ),
            "shared_metric_scaled_residual_alphas": (
                None
                if SHARED_METRIC_RESIDUAL_ALPHAS is None
                else list(SHARED_METRIC_RESIDUAL_ALPHAS)
            ),
            "specific_metric_scaled_residual_alphas": (
                None
                if SPECIFIC_METRIC_RESIDUAL_ALPHAS is None
                else list(SPECIFIC_METRIC_RESIDUAL_ALPHAS)
            ),
            "pro_metric_scaled_residual_alphas": (
                None
                if PRO_METRIC_RESIDUAL_ALPHAS is None
                else list(PRO_METRIC_RESIDUAL_ALPHAS)
            ),
            "shared_metric_scaled_residual_members": (
                None
                if SHARED_METRIC_RESIDUAL_MEMBERS is None
                else list(SHARED_METRIC_RESIDUAL_MEMBERS)
            ),
            "shared_fit_groups": sorted(SHARED_FIT_GROUPS),
            "specific_fit_groups": sorted(SPECIFIC_FIT_GROUPS),
            "pro_fit_groups": sorted(PRO_FIT_GROUPS),
            "training_origin_noninferiority_gate": (
                task == "classification" and CLASS_ORIGIN_NONINFERIORITY
            ),
        },
        "origin_executed_by_every_method": True,
        "historical_unmanipulated_clean_context_used": False,
        "regression_forbidden_contexts": ["shared/clean", "shared_output"] if task == "regression" else [],
        "all_outer_test_labels_excluded_from_selection": True,
        "common_inference_core": "unified_context_ensemble.src.core.predict",
        "common_inference_interface": (
            "unified_context_ensemble.src.inference.run_context_inference"
        ),
    }
    (task_output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "all"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="write a new result snapshot without overwriting accepted results",
    )
    parser.add_argument("--backbone-display", default="TabICL v2")
    parser.add_argument(
        "--checkpoint", default="tabicl-regressor-v2-20260212.ckpt"
    )
    args = parser.parse_args()
    tasks = ("classification", "regression") if args.task == "all" else (args.task,)
    combined = {
        task: build(
            task,
            args.output_root,
            backbone_display=args.backbone_display,
            checkpoint=args.checkpoint,
        )
        for task in tasks
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "combined_audit.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
