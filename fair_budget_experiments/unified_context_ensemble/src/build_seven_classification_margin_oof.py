#!/usr/bin/env python3
"""Seven-shift classification OOF with visible, strength-blind tier margins."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_budget_experiments.exp04_equal_cost_optimized.src import (  # noqa: E402
    build_adaptive_oof as source,
)
from fair_budget_experiments.exp05_context_pro.src import (  # noqa: E402
    build_pro_oof as pro_source,
)
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_context_only_five_shift as selection,
)
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_context_only_seven_shift as seven,
)
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_five_shift_candidate as metrics_source,
)
from fair_budget_experiments.unified_context_ensemble.src.build_source_only_classification_oof import (  # noqa: E402
    PRO_CONSENSUS_NAMES,
    PRO_GLOBAL_STAGEWISE_NAMES,
    PRO_ROUTE_NAME,
    REPEAT_SPECIFIC_NAME,
    SHARED_ROUTE_NAME,
    _cap_policy_update,
    _multimetric_residual_route_candidates,
    _pro_candidates,
    _residual_route_candidates,
    _route_prediction,
    _routed_views,
    _select_pro_multimetric,
    _with_consensus_experts,
)
from fair_budget_experiments.unified_context_ensemble.src.build_temporal_classification_oof import (  # noqa: E402
    PRO_STAGEWISE_NAMES,
    _select_pro_strength_blind,
    _select_shared_gain,
    _stagewise_probability,
)
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_temporal_classification_oof as temporal_selection,
)
from fair_budget_experiments.unified_context_ensemble.src.core import (  # noqa: E402
    FusionPolicy,
    aggregate,
    clear_batch_member_feature_cache,
    predict,
)


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = HERE / "results" / "seven_classification_margin_v1_candidate"
ACCEPTED_SHARED = (
    HERE / "results" / "context_only_seven_shift_v12"
    / "classification" / "detail_oof.csv"
)
METHODS = ("base", "shared", "specific", "pro")
METRICS = ("accuracy", "balanced_accuracy", "auc")
# Preserve the full selected Specific update.  Pro must earn its margin.
SPECIFIC_UPDATE_CAP = 1.0
# Re-fitting the prediction-diagnostic router can move one decision on a
# 128-row query batch (1 / 128 = 0.0078125) while retaining the exact Context
# registry and policy structure.  Record that metric drift explicitly and let
# the separate absolute-performance gate reject harmful changes.
RAW_SHARED_REPRODUCTION_TOLERANCE = 8.0e-3


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _accepted_shared_lookup() -> dict[tuple[str, str, float], dict[str, float]]:
    output = {}
    with ACCEPTED_SHARED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[(row["dataset"], row["shift"], float(row["strength"]))] = {
                metric: float(row[f"shared_{metric}"]) for metric in METRICS
            }
    return output


def _view(
    record: source.Record,
    members: dict[str, np.ndarray],
) -> selection.View:
    value = selection.View(
        task="classification",
        data_id=int(record.key),
        dataset=record.key,
        fold=int(record.fold),
        family=record.shift,
        rho=int(round(100 * float(record.strength))),
        target=np.asarray(record.y),
        classes=np.arange(np.asarray(record.base).shape[1]),
        scale=None,
        base=np.asarray(record.base),
        members={name: np.asarray(prediction) for name, prediction in members.items()},
    )
    selection._VIEW_REFS.append(value)
    return value


def _restrict(view: selection.View, names: tuple[str, ...]) -> selection.View:
    value = replace(view, members={name: view.members[name] for name in names})
    selection._VIEW_REFS.append(value)
    return value


def _canonical_views(
    records: list[source.Record],
) -> tuple[
    dict[tuple[str, str, float], selection.View],
    dict[tuple[str, str, float], selection.View],
    dict[tuple[str, str, float], selection.View],
    dict[str, tuple[str, ...]],
]:
    tiers = {
        (record.key, record.shift, float(record.strength)):
        seven._stable_members(record, "classification")
        for record in records
    }
    pro_schemas: dict[str, tuple[str, ...]] = {}
    for shift in source.SHIFTS:
        names: list[str] = []
        for record in records:
            if record.shift != shift:
                continue
            for name in tiers[(record.key, record.shift, float(record.strength))]["pro"]:
                if name not in names:
                    names.append(name)
        pro_schemas[shift] = tuple(names)

    shared_views = {}
    specific_views = {}
    pro_views = {}
    for record in records:
        key = (record.key, record.shift, float(record.strength))
        current = tiers[key]
        shared_views[key] = _view(record, current["shared"])
        specific_views[key] = _view(record, current["specific"])
        # Canonicalize the union across origin/shifted banks. Missing semantic
        # slots are exact Base fallbacks; this avoids an inference branch on
        # nominal strength while preserving every actually available Context.
        pro_views[key] = _view(record, {
            name: np.asarray(current["pro"].get(name, record.base))
            for name in pro_schemas[record.shift]
        })
    return shared_views, specific_views, pro_views, pro_schemas


def _pro_margin_candidates(
    views: list[selection.View], full_pool_policy: FusionPolicy
) -> list[FusionPolicy]:
    candidates = _pro_candidates(views, full_pool_policy)
    structured_groups = (
        (PRO_STAGEWISE_NAMES[0],),
        (PRO_STAGEWISE_NAMES[1],),
        PRO_STAGEWISE_NAMES,
        (REPEAT_SPECIFIC_NAME, PRO_STAGEWISE_NAMES[0]),
        *((name,) for name in PRO_CONSENSUS_NAMES),
        *((name,) for name in PRO_GLOBAL_STAGEWISE_NAMES[:3]),
        PRO_GLOBAL_STAGEWISE_NAMES[:2],
    )
    for names in structured_groups:
        for mode in ("mean", "median"):
            for alpha in (0.15, 0.25, 0.35, 0.50, 0.75, 1.0):
                candidates.append(FusionPolicy(
                    names,
                    mode,
                    alpha,
                    metadata={
                        "candidate": "stagewise_margin_expert",
                        "stagewise_members": list(names),
                    },
                ))
        for regularization in (1.0, 10.0):
            router = selection._fit_batch_router(
                views,
                names,
                regularization=regularization,
                minimum_predicted_gain=0.0,
            )
            for alpha in (0.25, 0.50, 0.75, 1.0):
                candidates.append(replace(
                    router,
                    alpha=alpha,
                    metadata={
                        **dict(router.metadata),
                        "candidate_group": "stagewise_margin_router",
                        "router_residual_alpha": alpha,
                    },
                ))
    stacker_groups = (
        tuple(views[0].members),
        tuple(name for name in (
            REPEAT_SPECIFIC_NAME,
            *PRO_STAGEWISE_NAMES,
            *PRO_GLOBAL_STAGEWISE_NAMES[:3],
            *PRO_CONSENSUS_NAMES,
        ) if name in views[0].members),
    )
    for names in stacker_groups:
        if not names:
            continue
        for regularization in (0.1, 1.0, 10.0, 100.0):
            candidates.append(selection._fit_convex(
                views, names, regularization
            ))
            candidates.append(selection._fit_binary_logit_residual_stacker(
                views, names, regularization
            ))
    return candidates


def build(
    output: Path,
    *,
    folds: tuple[int, ...] = tuple(range(5)),
    route_profile: str = "transfer_multimetric",
    enforce_accepted_shared_reproduction: bool = True,
    shared_gain_grid: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("folds must be a non-empty subset of 0..4")
    if route_profile not in {
        "legacy_nostrength", "pro_route_transfer", "pro_transfer_multimetric",
        "transfer_multimetric",
    }:
        raise ValueError(f"unknown route_profile: {route_profile}")
    if shared_gain_grid is not None:
        if not shared_gain_grid or any(
            not np.isfinite(value) or value <= 0.0
            for value in shared_gain_grid
        ):
            raise ValueError("shared_gain_grid must contain positive values")
        temporal_selection.SHARED_STAGEWISE_GAINS = tuple(
            float(value) for value in shared_gain_grid
        )
    pro_route_transfer = route_profile != "legacy_nostrength"
    pro_center_transfer = route_profile in {
        "pro_transfer_multimetric", "transfer_multimetric"
    }
    shared_transfer_multimetric = route_profile == "transfer_multimetric"
    selection.CLASSIFICATION_BATCH_PRIOR_ROUTER = True
    selection._BANK_CACHE.clear()
    selection._VIEW_REFS.clear()
    clear_batch_member_feature_cache()
    records = source._classification_records()
    pro_source._augment_classification(records)
    shared_views, specific_views, raw_pro_views, pro_schemas = _canonical_views(records)
    policies: dict[tuple[int, str, str], FusionPolicy] = {}
    shared_gains: dict[int, float] = {}
    rows: list[dict[str, Any]] = []

    for fold in folds:
        train = [record for record in records if int(record.fold) != fold]
        shared_train = [
            shared_views[(record.key, record.shift, float(record.strength))]
            for record in train
        ]
        shared_policy, diagnostics = selection._select_policy(
            shared_train,
            max_members=5,
            base_downside_penalty=0.0,
            tail_risk_weight=0.0,
        )
        raw_shared = [
            predict(selection._bank(view, shared_policy.members), shared_policy)
            for view in shared_train
        ]
        shared_gain, gain_diagnostics = _select_shared_gain(shared_train, raw_shared)
        shared_candidates = [
            _stagewise_probability(view.base, prediction, shared_gain)
            for view, prediction in zip(shared_train, raw_shared, strict=True)
        ]
        shared_route_train = _routed_views(
            shared_train, shared_candidates, name=SHARED_ROUTE_NAME
        )
        shared_route_candidates = [
            policy for policy in _residual_route_candidates(
                shared_route_train, name=SHARED_ROUTE_NAME
            )
            if str(policy.metadata.get("candidate", "")).startswith(
                ("direct_", "exact_", "multimetric_")
            )
        ]
        shared_route_policy, route_diagnostics = _select_pro_multimetric(
            shared_route_train, shared_route_candidates,
            prefer_dataset_transfer=shared_transfer_multimetric,
        )
        route_diagnostics = {
            **route_diagnostics,
            "route_member": SHARED_ROUTE_NAME,
            "route_training_targets": "outer_training_only",
            "route_inference_inputs": "prediction_diagnostics_only",
            "route_receives_family_or_strength": False,
            "primary_only_router_candidates_excluded": True,
        }
        shared_gains[fold] = shared_gain
        policies[(fold, "shared", "all_shifts")] = shared_policy
        policies[(fold, "shared_route", "all_shifts")] = shared_route_policy
        rows.append(selection._policy_row(
            "classification", fold, "shared", "all_shifts",
            shared_policy, {**diagnostics, **gain_diagnostics}, shared_train,
        ))
        rows.append(selection._policy_row(
            "classification", fold, "shared_route", "all_shifts",
            shared_route_policy, route_diagnostics, shared_route_train,
        ))

        pro_route_training: dict[str, list[selection.View]] = {}
        for shift in source.SHIFTS:
            shift_train = [record for record in train if record.shift == shift]
            specific_train: list[selection.View] = []
            for record in shift_train:
                key = (record.key, record.shift, float(record.strength))
                shared_view = shared_views[key]
                raw_prediction = predict(
                    selection._bank(shared_view, shared_policy.members), shared_policy
                )
                shared_prediction = _stagewise_probability(
                    shared_view.base, raw_prediction, shared_gain
                )
                shared_prediction = _route_prediction(
                    shared_view.base,
                    shared_prediction,
                    shared_route_policy,
                    name=SHARED_ROUTE_NAME,
                )
                specific_train.append(selection._with_base(
                    specific_views[key], shared_prediction
                ))
            specific_policy, diagnostics = selection._select_policy(
                specific_train,
                max_members=5,
                allow_base_fallback=False,
                base_downside_penalty=0.0,
                tail_risk_weight=0.0,
                fast_candidate_search=True,
                skip_gate_search=True,
            )
            specific_policy = _cap_policy_update(
                specific_policy, SPECIFIC_UPDATE_CAP
            )
            policies[(fold, "specific", shift)] = specific_policy
            rows.append(selection._policy_row(
                "classification", fold, "specific", shift,
                specific_policy, diagnostics, specific_train,
            ))

            pro_train: list[selection.View] = []
            for record, specific_view in zip(shift_train, specific_train, strict=True):
                key = (record.key, record.shift, float(record.strength))
                specific_prediction = predict(
                    selection._bank(specific_view, specific_policy.members),
                    specific_policy,
                )
                center_policy = replace(
                    specific_policy, alpha=1.0, gate=None,
                    batch_router=None, decision_guard=None,
                )
                center = aggregate(
                    selection._bank(specific_view, center_policy.members),
                    center_policy,
                )
                pro_view = selection._with_member(
                    selection._with_base(raw_pro_views[key], specific_prediction),
                    REPEAT_SPECIFIC_NAME,
                    center,
                )
                for name, gain in zip(PRO_STAGEWISE_NAMES, (2.0, 3.0), strict=True):
                    pro_view = selection._with_member(
                        pro_view,
                        name,
                        _stagewise_probability(
                            np.asarray(specific_view.base),
                            specific_prediction,
                            gain,
                            name=name,
                        ),
                    )
                for name, gain in zip(
                    PRO_GLOBAL_STAGEWISE_NAMES[:3], (1.25, 1.50, 2.00), strict=True
                ):
                    pro_view = selection._with_member(
                        pro_view,
                        name,
                        _stagewise_probability(
                            np.asarray(raw_pro_views[key].base),
                            specific_prediction, gain, name=name,
                        ),
                    )
                pro_view = _with_consensus_experts(
                    pro_view, raw_pro_views[key].members
                )
                pro_train.append(pro_view)
            full_pool_policy, _ = selection._select_policy(
                pro_train,
                max_members=None,
                allow_base_fallback=False,
                base_downside_penalty=0.0,
                tail_risk_weight=0.0,
                fast_candidate_search=True,
                skip_gate_search=True,
            )
            center_candidates = _pro_margin_candidates(pro_train, full_pool_policy)
            if pro_center_transfer:
                pro_center_policy, center_diagnostics = _select_pro_multimetric(
                    pro_train, center_candidates,
                    prefer_dataset_transfer=True,
                )
            else:
                pro_center_policy, center_diagnostics = _select_pro_strength_blind(
                    pro_train, center_candidates
                )
            pro_candidates = [
                predict(
                    selection._bank(view, pro_center_policy.members),
                    pro_center_policy,
                )
                for view in pro_train
            ]
            pro_route_train = _routed_views(
                pro_train, pro_candidates, name=PRO_ROUTE_NAME
            )
            pro_route_training[shift] = pro_route_train
            policies[(fold, "pro_center", shift)] = pro_center_policy
            rows.append(selection._policy_row(
                "classification", fold, "pro_center", shift,
                pro_center_policy, center_diagnostics, pro_train,
            ))

        pooled_route_train = [
            view for shift in source.SHIFTS for view in pro_route_training[shift]
        ]
        pooled_route_candidates = [
            replace(
                policy,
                metadata={
                    **dict(policy.metadata),
                    "route_fit_scope": "pooled_all_shifts",
                },
            )
            for policy in (
                _multimetric_residual_route_candidates
                if pro_route_transfer else _residual_route_candidates
            )(
                pooled_route_train, name=PRO_ROUTE_NAME
            )
        ]
        for shift in source.SHIFTS:
            pro_route_train = pro_route_training[shift]
            shift_route_candidates = [
                replace(
                    policy,
                    metadata={
                        **dict(policy.metadata),
                        "route_fit_scope": "registered_shift",
                    },
                )
                for policy in (
                    _multimetric_residual_route_candidates
                    if pro_route_transfer else _residual_route_candidates
                )(
                    pro_route_train, name=PRO_ROUTE_NAME
                )
            ]
            pro_policy, diagnostics = _select_pro_multimetric(
                pro_route_train,
                [*pooled_route_candidates, *shift_route_candidates],
                prefer_dataset_transfer=pro_route_transfer,
            )
            diagnostics = {
                **diagnostics,
                "route_member": PRO_ROUTE_NAME,
                "route_training_targets": "outer_training_only",
                "route_inference_inputs": "prediction_diagnostics_only",
                "route_receives_family_or_strength": False,
                "pooled_route_candidates_included": True,
            }
            policies[(fold, "pro", shift)] = pro_policy
            rows.append(selection._policy_row(
                "classification", fold, "pro", shift,
                pro_policy, diagnostics, pro_route_train,
            ))
            specific_policy = policies[(fold, "specific", shift)]
            print(
                f"fold={fold} {shift} shared_gain={shared_gain:.2f} "
                f"specific={len(specific_policy.members)}/{specific_policy.alpha:.2f} "
                f"pro={len(pro_policy.members)}/{pro_policy.alpha:.2f}",
                flush=True,
            )

    accepted_shared = _accepted_shared_lookup()
    detail: list[dict[str, Any]] = []
    raw_shared_errors: list[float] = []
    for record in records:
        fold = int(record.fold)
        if fold not in folds:
            continue
        key = (record.key, record.shift, float(record.strength))
        shared_policy = policies[(fold, "shared", "all_shifts")]
        shared_view = shared_views[key]
        raw_shared = predict(
            selection._bank(shared_view, shared_policy.members), shared_policy
        )
        raw_shared_metrics = metrics_source._classification_metrics(
            np.asarray(record.y), raw_shared,
            np.arange(np.asarray(record.base).shape[1]),
        )
        reference = accepted_shared[key]
        raw_shared_errors.extend(
            abs(float(raw_shared_metrics[metric]) - reference[metric])
            for metric in METRICS
        )
        shared_prediction = _stagewise_probability(
            shared_view.base, raw_shared, shared_gains[fold]
        )
        shared_prediction = _route_prediction(
            shared_view.base,
            shared_prediction,
            policies[(fold, "shared_route", "all_shifts")],
            name=SHARED_ROUTE_NAME,
        )
        specific_policy = policies[(fold, "specific", record.shift)]
        specific_view = selection._with_base(
            specific_views[key], shared_prediction
        )
        specific_prediction = predict(
            selection._bank(specific_view, specific_policy.members), specific_policy
        )
        center_policy = replace(
            specific_policy, alpha=1.0, gate=None,
            batch_router=None, decision_guard=None,
        )
        center = aggregate(
            selection._bank(specific_view, center_policy.members), center_policy
        )
        pro_view = selection._with_member(
            selection._with_base(raw_pro_views[key], specific_prediction),
            REPEAT_SPECIFIC_NAME,
            center,
        )
        for name, gain in zip(PRO_STAGEWISE_NAMES, (2.0, 3.0), strict=True):
            pro_view = selection._with_member(
                pro_view,
                name,
                _stagewise_probability(
                    np.asarray(specific_view.base), specific_prediction, gain,
                    name=name,
                ),
            )
        for name, gain in zip(
            PRO_GLOBAL_STAGEWISE_NAMES[:3], (1.25, 1.50, 2.00), strict=True
        ):
            pro_view = selection._with_member(
                pro_view,
                name,
                _stagewise_probability(
                    np.asarray(raw_pro_views[key].base), specific_prediction,
                    gain, name=name,
                ),
            )
        pro_view = _with_consensus_experts(
            pro_view, raw_pro_views[key].members
        )
        pro_center_policy = policies[(fold, "pro_center", record.shift)]
        pro_policy = policies[(fold, "pro", record.shift)]
        pro_candidate = predict(
            selection._bank(pro_view, pro_center_policy.members),
            pro_center_policy,
        )
        pro_prediction = _route_prediction(
            specific_prediction,
            pro_candidate,
            pro_policy,
            name=PRO_ROUTE_NAME,
        )
        predictions = {
            "base": np.asarray(record.base),
            "shared": shared_prediction,
            "specific": specific_prediction,
            "pro": pro_prediction,
        }
        row: dict[str, Any] = {
            "task": "classification",
            "dataset": record.key,
            "outer_fold": fold,
            "shift": record.shift,
            "strength": float(record.strength),
            "shared_context_registry_fixed": True,
            "shared_shift_known": False,
            "specific_shift_known": True,
            "pro_shift_known": True,
            "strength_used_for_selection_or_inference": False,
            "outer_test_labels_used_for_selection": False,
        }
        for method, prediction in predictions.items():
            values = metrics_source._classification_metrics(
                np.asarray(record.y), prediction,
                np.arange(np.asarray(record.base).shape[1]),
            )
            for metric in METRICS:
                row[f"{method}_{metric}"] = values[metric]
        detail.append(row)

    maximum_raw_shared_error = max(raw_shared_errors)
    accepted_shared_reproduced = (
        maximum_raw_shared_error <= RAW_SHARED_REPRODUCTION_TOLERANCE
    )
    if enforce_accepted_shared_reproduction and not accepted_shared_reproduced:
        raise RuntimeError(
            "accepted raw Shared was not reproduced: "
            f"max error={maximum_raw_shared_error}"
        )

    summary: list[dict[str, Any]] = []
    for shift in source.SHIFTS:
        for strength in source.STRENGTHS:
            cell = [
                row for row in detail
                if row["shift"] == shift
                and np.isclose(float(row["strength"]), float(strength))
            ]
            out: dict[str, Any] = {
                "task": "classification", "shift": shift,
                "strength": float(strength), "n_datasets": len(cell),
            }
            for method in METHODS:
                for metric in METRICS:
                    out[f"mean_{method}_{metric}"] = float(np.mean([
                        row[f"{method}_{metric}"] for row in cell
                    ]))
            summary.append(out)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "detail_oof.csv", detail)
    _write_csv(output / "summary_oof.csv", summary)
    _write_csv(output / "outer_selections.csv", rows)
    audit = {
        "protocol_id": "seven_classification_margin_v1",
        "task": "classification",
        "datasets": len({row["dataset"] for row in detail}),
        "detail_rows": len(detail),
        "outer_folds": len(folds),
        "selected_outer_folds": list(folds),
        "shared_context_registry_fixed": True,
        "shared_raw_policy_structure_locked": True,
        "accepted_shared_source": str(ACCEPTED_SHARED),
        "accepted_raw_shared_maximum_metric_error": maximum_raw_shared_error,
        "accepted_raw_shared_metric_tolerance": RAW_SHARED_REPRODUCTION_TOLERANCE,
        "accepted_raw_shared_check_enforced": (
            enforce_accepted_shared_reproduction
        ),
        "accepted_raw_shared_reproduced": accepted_shared_reproduced,
        "shared_selector_receives_family_or_strength": False,
        "nominal_strength_used_by_any_policy_selector": False,
        "policy_selection_unit": "outer_training_dataset_condition",
        "shared_stagewise_gains_by_fold": shared_gains,
        "shared_stagewise_gain_candidates": list(
            temporal_selection.SHARED_STAGEWISE_GAINS
        ),
        "shared_postfusion_router": "outer_train_prediction_diagnostics_only",
        "specific_context_count_limit": 5,
        "specific_update_cap": SPECIFIC_UPDATE_CAP,
        "pro_context_count_limit": None,
        "pro_postfusion_router": "outer_train_prediction_diagnostics_only",
        "pro_route_fit_scopes": ["pooled_all_shifts", "registered_shift"],
        "classification_router_training_target": (
            "minimum_accuracy_balanced_accuracy_auc_gain"
        ),
        "route_profile": route_profile,
        "pro_primary_only_route_candidates_excluded": pro_route_transfer,
        "route_selection_priority": (
            "cross_dataset_transfer_without_nominal_strength"
            if pro_route_transfer else "condition_consistency_without_nominal_strength"
        ),
        "shared_route_selection_priority": (
            "cross_dataset_transfer_without_nominal_strength"
            if shared_transfer_multimetric else
            "condition_consistency_without_nominal_strength"
        ),
        "binary_pro_stacker": "binary_logit_residual",
        "pro_raw_context_candidates_by_shift": {
            shift: len(names) for shift, names in pro_schemas.items()
        },
        "pro_stagewise_virtual_experts": list(PRO_STAGEWISE_NAMES),
        "pro_global_stagewise_virtual_experts": list(PRO_GLOBAL_STAGEWISE_NAMES[:3]),
        "pro_consensus_virtual_experts": list(PRO_CONSENSUS_NAMES),
        "specific_and_pro_receive_family_but_not_strength": True,
        "outer_test_labels_used_for_selection": False,
        "common_inference_core": "unified_context_ensemble.src.core.predict",
        "smoothing_or_curve_projection": False,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument(
        "--allow-shared-reselection",
        action="store_true",
        help=(
            "Write an explicitly audited candidate even when its newly "
            "selected Shared policy does not reproduce the accepted Shared."
        ),
    )
    parser.add_argument(
        "--shared-gain-grid",
        help=(
            "Comma-separated positive residual gains selected on each outer "
            "training partition; the default keeps the registered grid."
        ),
    )
    parser.add_argument(
        "--route-profile",
        choices=(
            "legacy_nostrength", "pro_route_transfer",
            "pro_transfer_multimetric",
            "transfer_multimetric",
        ),
        default="transfer_multimetric",
    )
    args = parser.parse_args()
    shared_gain_grid = (
        None
        if args.shared_gain_grid is None
        else tuple(
            float(value.strip())
            for value in args.shared_gain_grid.split(",")
            if value.strip()
        )
    )
    build(
        args.output,
        folds=(args.fold,) if args.fold is not None else tuple(range(5)),
        route_profile=args.route_profile,
        enforce_accepted_shared_reproduction=(
            not args.allow_shared_reselection
        ),
        shared_gain_grid=shared_gain_grid,
    )


if __name__ == "__main__":
    main()
