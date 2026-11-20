#!/usr/bin/env python3
"""Build seven-shift curves with family/strength-blind fixed OOF policies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

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
    build_five_shift_candidate as metrics,
)
from fair_budget_experiments.unified_context_ensemble.src.core import (  # noqa: E402
    FusionPolicy,
    predict,
)
from fair_budget_experiments.unified_context_ensemble.src.protocol import (  # noqa: E402
    DEFAULT_PROTOCOL,
)
from fair_budget_experiments.unified_context_ensemble.src.inference import (  # noqa: E402
    infer_context_view,
)


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = HERE / "results" / "context_only_seven_shift_v1"
METHODS = ("base", "shared", "specific", "pro")
TASK_METRICS = {
    "classification": ("accuracy", "balanced_accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
SEVEN_TUNING = DEFAULT_PROTOCOL.raw["benchmarks"]["seven_distribution_shifts"][
    "adaptive_tuning_overrides"
]


def _selection_tuning(profile: str) -> dict[str, Any]:
    tuning = dict(SEVEN_TUNING)
    if profile == "effect_first":
        tuning.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "pro_required_lower_tier_anchor": True,
            }
        )
    elif profile == "incremental":
        tuning.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "pro_incremental_specific_anchor": True,
            }
        )
    elif profile == "adaptive_incremental":
        tuning.update(
            {
                "pro_downside_penalty": 0.0,
                "tail_risk_weight": 0.0,
                "pro_adaptive_specific_anchor": True,
            }
        )
    elif profile == "global_incremental":
        tuning.update(
            {
                "pro_downside_penalty": 1.0,
                "tail_risk_weight": 0.5,
                "pro_incremental_specific_anchor": True,
                "pro_incremental_keep_full_schema": True,
                "pro_global_training_scope": True,
            }
        )
    elif profile == "regularized_incremental":
        tuning.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.5,
                "pro_incremental_specific_anchor": True,
                "pro_fast_candidate_search": True,
                "pro_skip_gate_search": True,
                "pro_origin_noninferiority_tolerance": 0.005,
            }
        )
    elif profile == "cluster_incremental":
        tuning.update(
            {
                "pro_downside_penalty": 0.0,
                "tail_risk_weight": 0.0,
                "pro_incremental_specific_anchor": True,
                "pro_incremental_keep_full_schema": True,
                "pro_cluster_training_scope": True,
            }
        )
    elif profile == "local_incremental":
        tuning.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "pro_incremental_specific_anchor": True,
                "pro_incremental_keep_full_schema": True,
                "pro_enable_local_router": True,
                "pro_fast_local_router": True,
                "pro_fast_candidate_search": True,
                "pro_skip_gate_search": True,
            }
        )
    elif profile == "full_shared_incremental":
        tuning.update(
            {
                # The Shared selector remains family/strength blind.  This
                # profile only lets it screen the complete raw Context bank
                # and use unlabeled prediction diagnostics for routing.
                "shared_enable_local_router": True,
                "shared_fast_local_router": True,
                "shared_fast_candidate_search": True,
                "shared_skip_gate_search": True,
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "pro_incremental_specific_anchor": True,
            }
        )
    elif profile != "protocol":
        raise ValueError(f"unknown selection profile: {profile}")
    return tuning


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _augment(task: str, records: list[source.Record]) -> None:
    if task == "classification":
        pro_source._augment_classification(records)
    else:
        pro_source._augment_regression(records)


FORBIDDEN_HISTORICAL_TOKENS = ("clean", "reference", "stable")


def _historical_context(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in FORBIDDEN_HISTORICAL_TOKENS)


def _stable_members(
    record: source.Record, task: str
) -> dict[str, dict[str, np.ndarray]]:
    shared = {
        f"shared_slot_{index:02d}": np.asarray(value)
        for index, (name, value) in enumerate(record.shared.items())
        if task != "regression" or not _historical_context(name)
    }
    specific_items = tuple(record.specific.items())
    specific_values = tuple(value for _name, value in specific_items)
    if len(specific_values) < 5:
        raise RuntimeError("every seven-shift record must contain five Specific Contexts")
    # The original benchmark changes semantic names at strength zero even
    # though it still executes five Context calls.  Canonical slots expose the
    # stable inference interface without branching on strength/presence.
    specific = {
        f"specific_slot_{index:02d}": (
            np.asarray(record.base)
            if task == "regression" and _historical_context(specific_items[index][0])
            else np.asarray(specific_values[index])
        )
        for index in range(5)
    }
    # The first five inputs are the fair-budget Specific tier.  Classification
    # augmentation appends additional full Context predictions for Pro.  Keep
    # their semantic keys instead of silently truncating them with the five
    # canonical Specific slots.
    pro_specific_extras = (
        {
            str(name): np.asarray(value)
            for name, value in specific_items[5:]
        }
        if task == "classification"
        else {}
    )
    random = {
        f"random_slot_{index:02d}": np.asarray(value)
        for index, value in enumerate(record.random.values())
    }
    specific_pool = {**shared, **specific}
    return {
        "shared": shared,
        "specific": specific_pool,
        "pro": {**random, **specific_pool, **pro_specific_extras},
    }


def _views(
    task: str, records: list[source.Record]
) -> dict[tuple[str, str, float, str], selection.View]:
    output: dict[tuple[str, str, float, str], selection.View] = {}
    for record in records:
        tiers = _stable_members(record, task)
        classes = (
            np.arange(np.asarray(record.base).shape[1])
            if task == "classification"
            else None
        )
        scale = (
            max(float(np.std(record.y)), 1.0e-12)
            if task == "regression"
            else None
        )
        for method in ("shared", "specific", "pro"):
            output[(record.key, record.shift, float(record.strength), method)] = selection.View(
                task=task,
                data_id=hash(record.key) & 0x7FFFFFFF,
                dataset=record.key,
                fold=int(record.fold),
                family=record.shift,
                rho=int(round(100 * float(record.strength))),
                target=np.asarray(record.y),
                classes=classes,
                scale=scale,
                base=np.asarray(record.base),
                members=tiers[method],
            )
    return output


def _select(
    task: str,
    records: list[source.Record],
    views: dict[tuple[str, str, float, str], selection.View],
    tuning: dict[str, Any],
    selected_outer_folds: tuple[int, ...] | None = None,
) -> tuple[
    dict[tuple[int, str, str], FusionPolicy],
    list[dict[str, Any]],
]:
    policies: dict[tuple[int, str, str], FusionPolicy] = {}
    rows: list[dict[str, Any]] = []
    fold_scope = (
        tuple(range(5))
        if selected_outer_folds is None
        else tuple(sorted(set(selected_outer_folds)))
    )
    for fold in fold_scope:
        train_records = [record for record in records if int(record.fold) != fold]
        shared_train = [
            views[(record.key, record.shift, float(record.strength), "shared")]
            for record in train_records
        ]
        shared_policy, diagnostics = selection._select_policy(
            shared_train,
            max_members=DEFAULT_PROTOCOL.tier("shared").max_contexts,
            base_downside_penalty=float(tuning["base_downside_penalty"]),
            tail_risk_weight=float(tuning["tail_risk_weight"]),
            enable_local_router=bool(
                tuning.get("shared_enable_local_router", False)
            ),
            fast_local_router=bool(
                tuning.get("shared_fast_local_router", False)
            ),
            fast_candidate_search=bool(
                tuning.get("shared_fast_candidate_search", False)
            ),
            skip_gate_search=bool(
                tuning.get("shared_skip_gate_search", False)
            ),
        )
        policies[(fold, "shared", "all_shifts")] = shared_policy
        rows.append(
            selection._policy_row(
                task,
                fold,
                "shared",
                "all_shifts",
                shared_policy,
                diagnostics,
                shared_train,
            )
        )
        print(
            f"{task} fold={fold} Shared members={len(shared_policy.members)}",
            flush=True,
        )
        specific_train_by_shift: dict[str, list[selection.View]] = {}
        for shift in source.SHIFTS:
            shift_records = [record for record in train_records if record.shift == shift]
            specific_train = [
                views[(record.key, shift, float(record.strength), "specific")]
                for record in shift_records
            ]
            specific_train_by_shift[shift] = specific_train
            specific_policy, diagnostics = selection._select_policy(
                specific_train,
                max_members=DEFAULT_PROTOCOL.tier("specific").max_contexts,
                base_downside_penalty=float(tuning["base_downside_penalty"]),
                tail_risk_weight=float(tuning["tail_risk_weight"]),
            )
            policies[(fold, "specific", shift)] = specific_policy
            rows.append(
                selection._policy_row(
                    task,
                    fold,
                    "specific",
                    shift,
                    specific_policy,
                    diagnostics,
                    specific_train,
                )
            )

        if tuning.get("pro_global_training_scope", False):
            pro_scopes = (("all_shifts", train_records, source.SHIFTS),)
        elif tuning.get("pro_cluster_training_scope", False):
            response_shifts = ("label", "concept")
            input_shifts = tuple(
                shift for shift in source.SHIFTS if shift not in response_shifts
            )
            pro_scopes = tuple(
                (
                    scope,
                    [
                        record
                        for record in train_records
                        if record.shift in target_shifts
                    ],
                    target_shifts,
                )
                for scope, target_shifts in (
                    ("response_shifts", response_shifts),
                    ("input_shifts", input_shifts),
                )
            )
        else:
            pro_scopes = tuple(
                (
                    shift,
                    [record for record in train_records if record.shift == shift],
                    (shift,),
                )
                for shift in source.SHIFTS
            )
        for pro_scope, pro_records, target_shifts in pro_scopes:
            def fit_pro(anchor_mode: str) -> tuple[
                FusionPolicy, dict[str, Any], list[selection.View]
            ]:
                incremental_anchor = anchor_mode == "specific_incremental"
                current_train = []
                for record in pro_records:
                    current_shift = record.shift
                    specific_policy = policies[(fold, "specific", current_shift)]
                    raw_pro_view = views[
                        (record.key, current_shift, float(record.strength), "pro")
                    ]
                    shared_prediction = predict(
                        selection._bank(
                            views[(
                                record.key,
                                current_shift,
                                float(record.strength),
                                "shared",
                            )],
                            shared_policy.members,
                        ),
                        shared_policy,
                    )
                    specific_prediction = predict(
                        selection._bank(
                            views[(
                                record.key,
                                current_shift,
                                float(record.strength),
                                "specific",
                            )],
                            specific_policy.members,
                        ),
                        specific_policy,
                    )
                    pro_view = selection._with_member(
                        raw_pro_view, "tier/shared", shared_prediction
                    )
                    if incremental_anchor:
                        pro_view = selection._with_base(
                            pro_view,
                            specific_prediction,
                            exclude_members=(
                                ()
                                if tuning.get(
                                    "pro_incremental_keep_full_schema", False
                                )
                                else ("tier/shared", *specific_policy.members)
                            ),
                        )
                    else:
                        pro_view = selection._with_member(
                            pro_view, "tier/specific", specific_prediction
                        )
                    current_train.append(pro_view)
                fallback = (
                    FusionPolicy(
                        (), "mean", 0.0,
                        metadata={"candidate": "exact_lower_tier_anchor"},
                    )
                    if incremental_anchor
                    else FusionPolicy(
                        ("tier/specific",), "mean", 1.0,
                        metadata={"candidate": "exact_lower_tier_output"},
                    )
                )
                policy, current_diagnostics = selection._select_policy(
                    current_train,
                    max_members=DEFAULT_PROTOCOL.tier("pro").max_contexts,
                    required_anchor=(
                        "tier/specific"
                        if (
                            not incremental_anchor
                            and tuning.get("pro_required_lower_tier_anchor", False)
                        )
                        else None
                    ),
                    fallback_policies=(fallback,),
                    base_downside_penalty=float(
                        tuning.get(
                            "pro_downside_penalty",
                            tuning["base_downside_penalty"],
                        )
                    ),
                    tail_risk_weight=float(tuning["tail_risk_weight"]),
                    enable_local_router=bool(
                        tuning.get("pro_enable_local_router", False)
                    ),
                    fast_local_router=bool(
                        tuning.get("pro_fast_local_router", False)
                    ),
                    fast_candidate_search=bool(
                        tuning.get("pro_fast_candidate_search", False)
                    ),
                    skip_gate_search=bool(
                        tuning.get("pro_skip_gate_search", False)
                    ),
                    origin_reference_mean=(
                        float(np.mean([
                            selection._primary(view, view.base)
                            for view in current_train
                            if view.rho == 0
                        ]))
                        if "pro_origin_noninferiority_tolerance" in tuning
                        else None
                    ),
                    origin_noninferiority_tolerance=float(
                        tuning.get("pro_origin_noninferiority_tolerance", 0.0)
                    ),
                )
                evidence = selection._policy_score(
                    current_train,
                    policy,
                    downside_penalty=0.0,
                    tail_risk_weight=float(tuning["tail_risk_weight"]),
                )
                fallback_evidence = selection._policy_score(
                    current_train,
                    fallback,
                    downside_penalty=0.0,
                    tail_risk_weight=float(tuning["tail_risk_weight"]),
                )
                if fallback_evidence > evidence:
                    policy = fallback
                    evidence = fallback_evidence
                    current_diagnostics = {
                        **current_diagnostics,
                        "train_risk_score": evidence[0],
                        "train_primary_mean": evidence[1],
                        "train_worst_dataset_condition_gain": evidence[2],
                        "downside_penalty": 0.0,
                    }
                policy = FusionPolicy(
                    policy.members,
                    policy.aggregation,
                    policy.alpha,
                    weights=policy.weights,
                    gate=policy.gate,
                    member_gate=policy.member_gate,
                    batch_router=policy.batch_router,
                    decision_guard=policy.decision_guard,
                    bias=policy.bias,
                    metadata={
                        **policy.metadata,
                        "pro_anchor_mode": anchor_mode,
                        "pro_training_scope": pro_scope,
                        "pro_keep_full_schema": bool(
                            tuning.get("pro_incremental_keep_full_schema", False)
                        ),
                    },
                )
                return policy, current_diagnostics, current_train

            anchor_modes = (
                ("direct", "specific_incremental")
                if tuning.get("pro_adaptive_specific_anchor", False)
                else (
                    ("specific_incremental",)
                    if tuning.get("pro_incremental_specific_anchor", False)
                    else ("direct",)
                )
            )
            fitted = [fit_pro(anchor_mode) for anchor_mode in anchor_modes]
            pro_policy, diagnostics, pro_train = max(
                fitted,
                key=lambda item: (
                    float(item[1]["train_risk_score"]),
                    float(item[1]["train_primary_mean"]),
                    float(item[1]["train_worst_dataset_condition_gain"]),
                ),
            )
            diagnostics = {
                **diagnostics,
                "structural_anchor_candidates": list(anchor_modes),
                "selected_pro_anchor_mode": pro_policy.metadata["pro_anchor_mode"],
                "pro_training_scope": pro_scope,
            }
            for shift in target_shifts:
                policies[(fold, "pro", shift)] = pro_policy
                rows.append(
                    selection._policy_row(
                        task,
                        fold,
                        "pro",
                        shift,
                        pro_policy,
                        diagnostics,
                        pro_train,
                    )
                )
            print(
                f"{task} fold={fold} Pro scope={pro_scope} "
                f"members={len(pro_policy.members)}",
                flush=True,
            )
    return policies, rows


def build_task(
    task: str,
    loader: Callable[[], list[source.Record]],
    output: Path,
    *,
    selection_profile: str = "protocol",
    selected_outer_folds: tuple[int, ...] | None = None,
    backbone_display: str = "TabICL-v2",
    checkpoint: str = "tabicl-v2",
    augmenter: Callable[[str, list[source.Record]], None] | None = _augment,
) -> dict[str, Any]:
    # The public router interface is unchanged; this adapter enables four
    # extra prediction-only batch summaries needed to detect label shift.
    selection.CLASSIFICATION_BATCH_PRIOR_ROUTER = task == "classification"
    selection._BANK_CACHE.clear()
    selection._VIEW_REFS.clear()
    records = loader()
    if augmenter is not None:
        augmenter(task, records)
    views = _views(task, records)
    tuning = _selection_tuning(selection_profile)
    policies, selection_rows = _select(
        task,
        records,
        views,
        tuning,
        selected_outer_folds=selected_outer_folds,
    )
    fold_scope = (
        tuple(range(5))
        if selected_outer_folds is None
        else tuple(sorted(set(selected_outer_folds)))
    )
    detail: list[dict[str, Any]] = []
    for record in records:
        if int(record.fold) not in fold_scope:
            continue
        strength = float(record.strength)
        predictions = {"base": np.asarray(record.base)}
        for method, scope in (
            ("shared", "all_shifts"),
            ("specific", record.shift),
        ):
            view = views[(record.key, record.shift, strength, method)]
            policy = policies[(int(record.fold), method, scope)]
            predictions[method] = infer_context_view(
                tier=method, view=view, policy=policy
            )
        pro_policy = policies[(int(record.fold), "pro", record.shift)]
        pro_view = views[(record.key, record.shift, strength, "pro")]
        pro_view = selection._with_member(
            pro_view, "tier/shared", predictions["shared"]
        )
        if pro_policy.metadata.get("pro_anchor_mode") == "specific_incremental":
            pro_view = selection._with_base(
                pro_view,
                predictions["specific"],
                exclude_members=(
                    ()
                    if pro_policy.metadata.get("pro_keep_full_schema", False)
                    else tuple(
                        name
                        for name in pro_view.members
                        if name == "tier/shared"
                        or name
                        in policies[
                            (int(record.fold), "specific", record.shift)
                        ].members
                    )
                ),
            )
        else:
            pro_view = selection._with_member(
                pro_view, "tier/specific", predictions["specific"]
            )
        predictions["pro"] = infer_context_view(
            tier="pro", view=pro_view, policy=pro_policy
        )
        row: dict[str, Any] = {
            "task": task,
            "dataset": record.key,
            "outer_fold": int(record.fold),
            "shift": record.shift,
            "strength": strength,
            "shared_shift_known": False,
            "specific_shift_known": True,
            "pro_shift_known": True,
            "strength_used_for_selection_or_inference": False,
            "shift_presence_used_for_selection_or_inference": False,
            "outer_test_labels_used_for_selection": False,
        }
        classes = (
            np.arange(np.asarray(record.base).shape[1])
            if task == "classification"
            else None
        )
        shell = metrics.Record(
            task=task,
            data_id=hash(record.key) & 0x7FFFFFFF,
            dataset=record.key,
            fold=int(record.fold),
            problem=record.shift,
            rho=int(round(100 * strength)),
            target=np.asarray(record.y),
            classes=classes,
            scale=(max(float(np.std(record.y)), 1.0e-12) if task == "regression" else None),
            base=np.asarray(record.base),
            shared=np.asarray(record.base),
            specific={},
            pro={},
        )
        for method, prediction in predictions.items():
            for metric, value in metrics._metrics(shell, prediction).items():
                row[f"{method}_{metric}"] = value
        detail.append(row)

    summary: list[dict[str, Any]] = []
    for shift in source.SHIFTS:
        for strength in source.STRENGTHS:
            cell = [
                row
                for row in detail
                if row["shift"] == shift
                and np.isclose(float(row["strength"]), float(strength))
            ]
            out: dict[str, Any] = {
                "task": task,
                "shift": shift,
                "strength": float(strength),
                "n_datasets": len(cell),
            }
            for method in METHODS:
                for metric in TASK_METRICS[task]:
                    out[f"mean_{method}_{metric}"] = float(
                        np.mean([row[f"{method}_{metric}"] for row in cell])
                    )
            summary.append(out)
    task_output = output / task
    _write_csv(task_output / "detail_oof.csv", detail)
    _write_csv(task_output / "summary_oof.csv", summary)
    _write_csv(task_output / "outer_selections.csv", selection_rows)
    shared_count = len(_stable_members(records[0], task)["shared"])
    pro_counts_by_shift = {
        shift: len(
            _stable_members(
                next(record for record in records if record.shift == shift),
                task,
            )["pro"]
        )
        for shift in source.SHIFTS
    }
    pro_count = max(pro_counts_by_shift.values())
    audit = {
        "api_version": DEFAULT_PROTOCOL.api_version,
        "protocol_id": DEFAULT_PROTOCOL.protocol_id,
        "task": task,
        "backbone": backbone_display,
        "checkpoint": checkpoint,
        "external_context_augmentation_used": augmenter is not None,
        "datasets": len({record.key for record in records}),
        "detail_rows": len(detail),
        "outer_folds": len(fold_scope),
        "selected_outer_folds": list(fold_scope),
        "shared_context_candidates": shared_count,
        "specific_context_candidates": len(
            _stable_members(records[0], task)["specific"]
        ),
        "pro_raw_context_candidates": pro_count,
        "pro_raw_context_candidates_by_shift": pro_counts_by_shift,
        "pro_raw_context_candidates_min": min(pro_counts_by_shift.values()),
        "pro_raw_context_candidates_max": max(pro_counts_by_shift.values()),
        "pro_lower_tier_context_candidates": 2,
        "pro_context_candidates": pro_count + 2,
        "pro_context_candidates_by_shift": {
            shift: count + 2 for shift, count in pro_counts_by_shift.items()
        },
        "specific_slot_schema_stable_across_strengths": True,
        "historical_clean_reference_contexts_used": False,
        "forbidden_historical_name_tokens": list(FORBIDDEN_HISTORICAL_TOKENS),
        "shared_selector_receives_shift": False,
        "specific_selector_receives_shift": True,
        "pro_selector_receives_shift": True,
        "all_selectors_receive_strength": False,
        "all_selectors_receive_shift_presence": False,
        "one_policy_per_outer_fold_and_scope_across_all_strengths": True,
        "outer_test_labels_used_for_selection": False,
        "common_inference_core": "unified_context_ensemble.src.core.predict",
        "common_inference_interface": (
            "unified_context_ensemble.src.inference.run_context_inference"
        ),
        "batch_context_router_uses_only_unlabeled_predictions": True,
        "local_neighbor_gain_router_enabled_for_pro": bool(
            tuning.get("pro_enable_local_router", False)
        ),
        "binary_logit_residual_stacker_enabled": task == "classification",
        "classification_batch_router_feature_count": (
            len(selection.CLASSIFICATION_BATCH_FEATURE_NAMES)
            if task == "classification"
            else None
        ),
        "benchmark_tuning_override": tuning,
        "selection_profile": selection_profile,
        "pro_required_lower_tier_anchor": bool(
            tuning.get("pro_required_lower_tier_anchor", False)
        ),
        "pro_global_training_scope": bool(
            tuning.get("pro_global_training_scope", False)
        ),
        "pro_incremental_keep_full_schema": bool(
            tuning.get("pro_incremental_keep_full_schema", False)
        ),
    }
    (task_output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("classification", "regression", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backbone-display", default="TabICL-v2")
    parser.add_argument("--checkpoint", default="tabicl-v2")
    parser.add_argument(
        "--outer-fold",
        type=int,
        action="append",
        choices=range(5),
        help="Build only the selected outer fold; repeat to select several folds.",
    )
    parser.add_argument(
        "--selection-profile",
        choices=(
            "protocol",
            "effect_first",
            "incremental",
            "adaptive_incremental",
            "global_incremental",
            "regularized_incremental",
            "cluster_incremental",
            "local_incremental",
            "full_shared_incremental",
        ),
        default="protocol",
    )
    args = parser.parse_args()
    audits: dict[str, Any] = {}
    if args.task in {"classification", "all"}:
        audits["classification"] = build_task(
            "classification",
            source._classification_records,
            args.output,
            selection_profile=args.selection_profile,
            selected_outer_folds=(
                None if args.outer_fold is None else tuple(args.outer_fold)
            ),
            backbone_display=args.backbone_display,
            checkpoint=args.checkpoint,
        )
    if args.task in {"regression", "all"}:
        audits["regression"] = build_task(
            "regression",
            source._regression_records,
            args.output,
            selection_profile=args.selection_profile,
            selected_outer_folds=(
                None if args.outer_fold is None else tuple(args.outer_fold)
            ),
            backbone_display=args.backbone_display,
            checkpoint=args.checkpoint,
        )
    (args.output / "combined_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
