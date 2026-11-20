#!/usr/bin/env python3
"""Build strict seven-shift regression curves from frozen-backbone Context banks."""

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

from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_context_only_five_shift as selection,
)
from fair_budget_experiments.unified_context_ensemble.src.core import (  # noqa: E402
    FusionPolicy,
    clear_batch_member_feature_cache,
    predict,
)
from fair_budget_experiments.unified_context_ensemble.src.protocol import (  # noqa: E402
    DEFAULT_PROTOCOL,
)
from fair_budget_experiments.unified_context_ensemble.src.inference import (  # noqa: E402
    infer_context_view,
)
from fair_budget_experiments.unified_context_ensemble.src.strict_seven_regression_contexts import (  # noqa: E402
    FAMILIES,
)


HERE = Path(__file__).resolve().parents[1]
DEFAULT_BANKS = HERE / "results" / "strict_seven_regression_banks_v1"
DEFAULT_OUTPUT = HERE / "results" / "strict_seven_regression_oof_v1"
FOLDS = ROOT / "fair_budget_experiments" / "exp02_regression" / "results" / "outer_folds.csv"
STRENGTHS = (0.0, 0.2, 0.5, 0.8, 1.0)
METHODS = ("base", "shared", "specific", "pro")
METRICS = ("r2", "nmae", "nrmse")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_family_bank_overrides(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if not separator or family not in FAMILIES or not raw_path:
            raise ValueError(
                "family bank override must be a registered FAMILY=PATH"
            )
        output[family] = Path(raw_path)
    return output


def _folds() -> dict[str, int]:
    with FOLDS.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["dataset"]): int(row["outer_fold"])
            for row in csv.DictReader(handle)
        }


def _views(
    bank_root: Path,
    selected_datasets: tuple[str, ...] | None = None,
    family_bank_overrides: dict[str, Path] | None = None,
) -> tuple[
    dict[tuple[str, str, float, str], selection.View],
    dict[str, int],
    dict[str, Any],
]:
    folds = _folds()
    views: dict[tuple[str, str, float, str], selection.View] = {}
    overrides = family_bank_overrides or {}
    shared_schema: tuple[str, ...] | None = None
    schemas_by_family: dict[str, dict[str, tuple[str, ...]]] = {}
    if selected_datasets is None:
        dataset_scope = tuple(folds)
    else:
        missing = sorted(set(selected_datasets) - set(folds))
        if missing:
            raise ValueError(f"unknown strict regression datasets: {missing}")
        dataset_scope = tuple(selected_datasets)
    for dataset in dataset_scope:
        fold = folds[dataset]
        source_roots = {bank_root, *overrides.values()}
        payloads: dict[Path, dict[str, np.ndarray]] = {}
        for source_root in source_roots:
            path = source_root / f"{dataset}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"strict regression bank is missing: {path}"
                )
            with np.load(path, allow_pickle=False) as stored:
                payloads[source_root] = {
                    key: np.asarray(stored[key]) for key in stored.files
                }
        default_target = np.asarray(
            payloads[bank_root]["y_target"], dtype=np.float64
        )
        for source_root, payload in payloads.items():
            np.testing.assert_array_equal(payload["y_target"], default_target)

        for family in FAMILIES:
            source_root = overrides.get(family, bank_root)
            payload = payloads[source_root]
            families = payload["family"].astype(str)
            strengths = payload["strength"].astype(float)
            y = np.asarray(payload["y_target"], dtype=np.float64)
            shared_names = tuple(payload["shared_names"].astype(str))
            specific_names = tuple(payload["specific_names"].astype(str))
            pro_extra_names = tuple(payload["pro_extra_names"].astype(str))
            current_schemas: dict[str, tuple[str, ...]] = {
                "shared": shared_names,
                "specific": (*shared_names, *specific_names),
                "pro": (*shared_names, *specific_names, *pro_extra_names),
            }
            if shared_schema is None:
                shared_schema = shared_names
            elif shared_names != shared_schema:
                raise RuntimeError(
                    "Shared Context schema differs across family bank sources"
                )
            if family not in schemas_by_family:
                schemas_by_family[family] = current_schemas
            elif schemas_by_family[family] != current_schemas:
                raise RuntimeError(
                    f"Context schema for {family} changed across datasets"
                )
            lookup = {
                (family, float(strength)): index
                for index, (family, strength) in enumerate(zip(families, strengths))
            }
            for strength in STRENGTHS:
                index = lookup[(family, strength)]
                # NRMSE is the outer-training selection loss.  Its target
                # scale is legal for outer-training views and offline
                # evaluation; `core.predict` deliberately ignores this
                # field, so held-out inference remains label-free.
                selection_scale = max(float(np.std(y)), 1.0e-12)
                shared = {
                    name: np.asarray(
                        payload["shared"][index, :, position], dtype=np.float64
                    )
                    for position, name in enumerate(shared_names)
                }
                specific = {
                    name: np.asarray(
                        payload["specific"][index, :, position], dtype=np.float64
                    )
                    for position, name in enumerate(specific_names)
                }
                pro_extra = {
                    name: np.asarray(
                        payload["pro_extra"][index, :, position], dtype=np.float64
                    )
                    for position, name in enumerate(pro_extra_names)
                }
                pools = {
                    "shared": shared,
                    "specific": {**shared, **specific},
                    "pro": {**shared, **specific, **pro_extra},
                }
                for method, members in pools.items():
                    views[(dataset, family, strength, method)] = selection.View(
                        task="regression",
                        data_id=hash(dataset) & 0x7FFFFFFF,
                        dataset=dataset,
                        fold=fold,
                        family=family,
                        rho=int(round(100 * strength)),
                        target=y,
                        classes=None,
                        scale=selection_scale,
                        base=np.asarray(payload["base"][index], dtype=np.float64),
                        members=members,
                    )
    if shared_schema is None:
        raise RuntimeError("no strict seven-shift Context schema was loaded")
    schemas: dict[str, Any] = {
        "shared": shared_schema,
        "by_family": schemas_by_family,
    }
    return views, folds, schemas


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    scale = max(float(np.std(target)), 1.0e-12)
    denominator = max(float(np.sum((target - np.mean(target)) ** 2)), 1.0e-12)
    return {
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "nmae": float(np.mean(np.abs(error))) / scale,
        "nrmse": float(np.sqrt(np.mean(error**2))) / scale,
    }


def _nested_upgrade_evidence(
    *,
    outer_train: list[str],
    family: str,
    method: str,
    views: dict[tuple[str, str, float, str], selection.View],
    folds: dict[str, int],
    downside: float,
    tail: float,
    origin_noninferiority: bool,
    origin_tolerance: float,
    specific_policy_cache: dict[tuple[tuple[str, ...], str, int], FusionPolicy],
) -> float:
    """Inner-fold evidence for upgrading Base→Specific or Specific→Pro."""

    gains: list[float] = []
    for inner_fold in sorted({folds[dataset] for dataset in outer_train}):
        inner_train = [
            dataset for dataset in outer_train if folds[dataset] != inner_fold
        ]
        inner_valid = [
            dataset for dataset in outer_train if folds[dataset] == inner_fold
        ]
        if not inner_train or not inner_valid:
            continue
        specific_train = [
            views[(dataset, family, strength, "specific")]
            for dataset in inner_train
            for strength in STRENGTHS
        ]
        specific_origin_reference = None
        if (
            origin_noninferiority
            and not override.get("shared_requires_incremental_policy", False)
        ):
            specific_origin_reference = float(np.mean([
                selection._primary(
                    views[(dataset, family, 0.0, "specific")],
                    views[(dataset, family, 0.0, "specific")].base,
                )
                for dataset in inner_train
            ]))
        cache_key = (tuple(outer_train), family, int(inner_fold))
        if cache_key not in specific_policy_cache:
            specific_policy_cache[cache_key], _diagnostics = selection._select_policy(
                specific_train,
                max_members=DEFAULT_PROTOCOL.tier("specific").max_contexts,
                base_downside_penalty=downside,
                tail_risk_weight=tail,
                origin_reference_mean=specific_origin_reference,
                origin_noninferiority_tolerance=origin_tolerance,
                group_noninferiority_margin=(
                    selection.SPECIFIC_GROUP_NONINFERIORITY_MARGIN
                ),
            )
        specific_policy = specific_policy_cache[cache_key]
        if method == "specific":
            candidate_policy = specific_policy
        elif method == "pro":
            pro_train = [
                views[(dataset, family, strength, "pro")]
                for dataset in inner_train
                for strength in STRENGTHS
            ]
            pro_origin_reference = None
            if origin_noninferiority:
                pro_origin_reference = float(np.mean([
                    selection._primary(
                        view,
                        predict(
                            selection._bank(view, specific_policy.members),
                            specific_policy,
                        ),
                    )
                    for view in pro_train
                    if view.rho == 0
                ]))
            candidate_policy, _diagnostics = selection._select_policy(
                pro_train,
                max_members=DEFAULT_PROTOCOL.tier("pro").max_contexts,
                fallback_policies=(specific_policy,),
                base_downside_penalty=downside,
                tail_risk_weight=tail,
                origin_reference_mean=pro_origin_reference,
                origin_noninferiority_tolerance=origin_tolerance,
                group_noninferiority_margin=(
                    selection.PRO_GROUP_NONINFERIORITY_MARGIN
                ),
            )
        else:
            raise ValueError(f"unknown nested upgrade method: {method}")

        for dataset in inner_valid:
            for strength in STRENGTHS:
                candidate_view = views[(dataset, family, strength, method)]
                candidate = predict(
                    selection._bank(candidate_view, candidate_policy.members),
                    candidate_policy,
                )
                if method == "specific":
                    fallback = candidate_view.base
                else:
                    fallback_view = views[(dataset, family, strength, "pro")]
                    fallback = predict(
                        selection._bank(fallback_view, specific_policy.members),
                        specific_policy,
                    )
                gains.append(
                    selection._primary(candidate_view, candidate)
                    - selection._primary(candidate_view, fallback)
                )
    if not gains:
        return float("-inf")
    values = np.asarray(gains, dtype=np.float64)
    return float(np.mean(values) - 0.10 * np.std(values))


def _fold_stability_evidence(
    views: list[selection.View],
    pro_policy: FusionPolicy,
    specific_policy: FusionPolicy,
) -> float:
    """Training-fold stability of the Specific→Pro upgrade."""

    fold_gains = []
    for fold in sorted({view.fold for view in views}):
        current = [view for view in views if view.fold == fold]
        gains = []
        for view in current:
            pro_prediction = predict(
                selection._bank(view, pro_policy.members), pro_policy
            )
            specific_prediction = predict(
                selection._bank(view, specific_policy.members), specific_policy
            )
            gains.append(
                selection._primary(view, pro_prediction)
                - selection._primary(view, specific_prediction)
            )
        fold_gains.append(float(np.mean(gains)))
    values = np.asarray(fold_gains, dtype=np.float64)
    return float(np.mean(values) - np.std(values))


def build(
    bank_root: Path,
    output: Path,
    *,
    selection_profile: str = "protocol",
    selected_outer_folds: tuple[int, ...] | None = None,
    selected_datasets: tuple[str, ...] | None = None,
    family_bank_overrides: dict[str, Path] | None = None,
    backbone_display: str = "TabICL-v2",
    checkpoint: str = "tabicl-regressor-v2-20260212.ckpt",
) -> dict[str, Any]:
    selection._BANK_CACHE.clear()
    selection._VIEW_REFS.clear()
    clear_batch_member_feature_cache()
    family_bank_overrides = family_bank_overrides or {}
    views, all_folds, schemas = _views(
        bank_root,
        selected_datasets,
        family_bank_overrides,
    )
    folds = {
        dataset: fold
        for dataset, fold in all_folds.items()
        if selected_datasets is None or dataset in selected_datasets
    }
    source_roots = {"default": bank_root, **family_bank_overrides}
    bank_audits: dict[str, dict[str, Any]] = {}
    for source_name, source_root in source_roots.items():
        bank_audit_path = source_root / "audit.json"
        bank_audits[source_name] = (
            json.loads(bank_audit_path.read_text(encoding="utf-8"))
            if bank_audit_path.exists()
            else {}
        )
    bank_audit = bank_audits["default"]
    bank_manipulation_protocol = str(
        bank_audit.get("manipulation_protocol", "original")
    )
    bank_base_refreshed = all(
        bool(audit.get("base_recomputed_for_each_condition", False))
        or (
            bool(audit.get("rows"))
            and all(
                row.get("refreshed_base_protocol")
                == audit.get("manipulation_protocol", "original")
                for row in audit.get("rows", [])
            )
        )
        for audit in bank_audits.values()
    )
    datasets = sorted(folds)
    override = dict(
        DEFAULT_PROTOCOL.raw["benchmarks"]["seven_distribution_shifts"][
            "adaptive_tuning_overrides"
        ]
    )
    if selection_profile == "effect_first":
        override.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
            }
        )
    elif selection_profile == "hierarchical_effect_first":
        override.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "shared_enable_local_router": True,
                "shared_requires_incremental_policy": True,
                "specific_uses_shared_anchor": True,
                "specific_enable_local_router": True,
                "specific_requires_incremental_context": True,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
            }
        )
    elif selection_profile == "hierarchical_fixed_pro":
        override.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "shared_enable_local_router": True,
                "shared_requires_incremental_policy": True,
                "specific_uses_shared_anchor": True,
                "specific_enable_local_router": True,
                "specific_requires_incremental_context": True,
                "pro_enable_local_router": False,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
            }
        )
    elif selection_profile == "hierarchical_safe_pro":
        override.update(
            {
                "pro_downside_penalty": 5.0,
                "tail_risk_weight": 0.25,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "shared_enable_local_router": True,
                "shared_requires_incremental_policy": True,
                "specific_uses_shared_anchor": True,
                "specific_enable_local_router": True,
                "specific_requires_incremental_context": True,
                "pro_enable_local_router": False,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
            }
        )
    elif selection_profile == "hierarchical_max_effect_pro":
        override.update(
            {
                "pro_downside_penalty": 0.0,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "shared_enable_local_router": True,
                "shared_requires_incremental_policy": True,
                "specific_uses_shared_anchor": True,
                "specific_enable_local_router": True,
                "specific_requires_incremental_context": True,
                "pro_enable_local_router": False,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
            }
        )
    elif selection_profile == "hierarchical_conservative_shared":
        override.update(
            {
                # Shared is family-blind.  A conservative update prevents a
                # high-effect repair for one family from overwhelming the
                # small, broadly useful deduplication/label-scale consensus
                # on an unseen family.  The member set is still selected on
                # each outer-training partition.
                "base_downside_penalty": 1.0,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "shared_enable_local_router": False,
                "shared_requires_incremental_policy": True,
                "shared_update_cap": 0.25,
                "specific_uses_shared_anchor": True,
                "specific_enable_local_router": True,
                "specific_requires_incremental_context": True,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
                "pro_update_cap": 0.25,
            }
        )
    elif selection_profile == "hierarchical_locked_shared":
        override.update(
            {
                "base_downside_penalty": 1.0,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "locked_shared_arithmetic_mean": True,
                "locked_shared_adaptive_residual_alpha": True,
                "specific_uses_shared_anchor": True,
                "specific_enable_local_router": True,
                "specific_requires_incremental_context": True,
                "pro_requires_incremental_context": True,
                "pro_origin_noninferiority_gate": False,
                "pro_update_cap": 0.25,
            }
        )
    elif selection_profile == "incremental":
        override.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                # Pro is already trained as a residual update to the frozen
                # Specific anchor below.  Keep the exact anchor fallback so
                # an unsupported increment is never forced onto a held-out
                # dataset merely to separate curves.
                "pro_requires_incremental_context": False,
                "pro_origin_noninferiority_gate": False,
            }
        )
    elif selection_profile == "incremental_origin_safe":
        override.update(
            {
                "pro_downside_penalty": 0.5,
                "tail_risk_weight": 0.0,
                "regression_pro_fold_stability_gate": False,
                "regression_origin_noninferiority_gate": True,
                "regression_origin_noninferiority_tolerance_nrmse": 0.002,
                "pro_requires_incremental_context": False,
                "pro_origin_noninferiority_gate": True,
            }
        )
    elif selection_profile != "protocol":
        raise ValueError(f"unknown selection profile: {selection_profile}")
    downside = float(override["base_downside_penalty"])
    pro_downside = float(override.get("pro_downside_penalty", downside))
    tail = float(override["tail_risk_weight"])
    nested_minimum = float(override["regression_nested_minimum_gain"])
    nested_upgrade_gate = bool(override.get("regression_nested_upgrade_gate", False))
    pro_nested_upgrade_gate = bool(
        override.get("regression_pro_nested_upgrade_gate", nested_upgrade_gate)
    )
    pro_fold_stability_gate = bool(
        override.get("regression_pro_fold_stability_gate", False)
    )
    origin_noninferiority = bool(
        override.get("regression_origin_noninferiority_gate", False)
    )
    origin_tolerance = float(
        override.get("regression_origin_noninferiority_tolerance_nrmse", 0.0)
    )
    policies: dict[tuple[int, str, str], FusionPolicy] = {}
    selection_rows: list[dict[str, Any]] = []
    fast_hierarchical = selection_profile in {
        "hierarchical_effect_first",
        "hierarchical_fixed_pro",
        "hierarchical_safe_pro",
        "hierarchical_max_effect_pro",
        "hierarchical_conservative_shared",
        "hierarchical_locked_shared",
    }

    fold_scope = tuple(range(5)) if selected_outer_folds is None else tuple(
        sorted(set(selected_outer_folds))
    )
    if not fold_scope or any(fold not in range(5) for fold in fold_scope):
        raise ValueError("selected outer folds must be a nonempty subset of 0..4")

    for fold in fold_scope:
        train = [dataset for dataset in datasets if folds[dataset] != fold]
        nested_specific_cache: dict[
            tuple[tuple[str, ...], str, int], FusionPolicy
        ] = {}
        shared_train = [
            views[(dataset, family, strength, "shared")]
            for dataset in train
            for family in FAMILIES
            for strength in STRENGTHS
        ]
        shared_select = (
            [
                view
                for view in shared_train
                if (view.family, int(view.rho))
                in selection.SHARED_FOCUS_GROUPS
            ]
            if selection.SHARED_FOCUS_GROUPS
            else shared_train
        )
        shared_fit = (
            [
                view
                for view in shared_train
                if (view.family, int(view.rho))
                in selection.SHARED_FIT_GROUPS
            ]
            if selection.SHARED_FIT_GROUPS
            else None
        )
        shared_origin_reference = None
        if (
            origin_noninferiority
            and not override.get("shared_requires_incremental_policy", False)
        ):
            shared_origin_reference = float(np.mean([
                selection._primary(view, view.base)
                for view in shared_train
                if view.rho == 0
            ]))
        if override.get("locked_shared_arithmetic_mean", False):
            shared_candidates = [
                FusionPolicy(
                    tuple(schemas["shared"]),
                    "mean",
                    alpha,
                    metadata={
                        "candidate": "locked_shared_five_context_arithmetic_mean",
                        "training_selected_residual_alpha": alpha,
                    },
                )
                for alpha in (
                    selection.ALPHAS[1:]
                    if override.get("locked_shared_adaptive_residual_alpha", False)
                    else (1.0,)
                )
            ]
            *shared_score, shared_policy = max(
                [
                (
                    *selection._policy_score(
                        shared_train,
                        candidate,
                        downside_penalty=downside,
                        tail_risk_weight=tail,
                    ),
                    candidate,
                )
                for candidate in shared_candidates
                ],
                key=lambda item: item[:3],
            )
            diagnostics = {
                "train_risk_score": shared_score[0],
                "train_primary_mean": shared_score[1],
                "train_worst_dataset_condition_gain": shared_score[2],
                "candidate_count": len(shared_candidates),
                "downside_penalty": downside,
                "tail_risk_weight": tail,
                "origin_noninferiority_enabled": False,
                "ranking": list(schemas["shared"]),
                "shared_policy_locked": True,
                "residual_alpha_selected_on_outer_training_datasets": bool(
                    override.get("locked_shared_adaptive_residual_alpha", False)
                ),
            }
        else:
            shared_policy, diagnostics = selection._select_policy(
                shared_select,
                max_members=DEFAULT_PROTOCOL.tier("shared").max_contexts,
                base_downside_penalty=downside,
                tail_risk_weight=tail,
                origin_reference_mean=shared_origin_reference,
                origin_noninferiority_tolerance=origin_tolerance,
                enable_local_router=bool(
                    override.get("shared_enable_local_router", False)
                ),
                fast_local_router=fast_hierarchical,
                fast_candidate_search=fast_hierarchical,
                skip_gate_search=fast_hierarchical,
                allow_base_fallback=not bool(
                    override.get("shared_requires_incremental_policy", False)
                ),
                require_nonbase=bool(selection.SHARED_FOCUS_GROUPS),
                candidate_fit_views=shared_fit,
                enforce_group_noninferiority=bool(
                    selection.SHARED_FOCUS_GROUPS
                ),
                group_noninferiority_margin=(
                    selection.SHARED_GROUP_NONINFERIORITY_MARGIN
                ),
                metric_residual_alphas=(
                    selection.SHARED_METRIC_RESIDUAL_ALPHAS
                ),
                metric_residual_members=(
                    selection.SHARED_METRIC_RESIDUAL_MEMBERS
                ),
            )
        if (
            override.get("shared_requires_incremental_policy", False)
            and not shared_policy.members
            and not selection.GROUP_NONINFERIORITY_ENABLED
        ):
            shared_singletons = [
                FusionPolicy(
                    (name,),
                    "mean",
                    alpha,
                    metadata={
                        "candidate": "required_shared_singleton_fallback",
                        "training_selected": True,
                    },
                )
                for name in shared_train[0].members
                for alpha in selection.ALPHAS[1:]
            ]
            *shared_score, shared_policy = max(
                [
                (
                    *selection._policy_score(
                        shared_train,
                        candidate,
                        downside_penalty=downside,
                        tail_risk_weight=tail,
                    ),
                    candidate,
                )
                for candidate in shared_singletons
                ],
                key=lambda item: item[:3],
            )
            diagnostics = {
                **diagnostics,
                "required_shared_singleton_fallback": True,
                "train_risk_score": shared_score[0],
                "train_primary_mean": shared_score[1],
                "train_worst_dataset_condition_gain": shared_score[2],
            }
        shared_update_cap = override.get("shared_update_cap")
        if (
            shared_update_cap is not None
            and shared_policy.members
            and float(shared_policy.alpha) > float(shared_update_cap)
        ):
            shared_policy = replace(
                shared_policy,
                alpha=float(shared_update_cap),
                metadata={
                    **shared_policy.metadata,
                    "training_selected_uncapped_alpha": float(shared_policy.alpha),
                    "adaptive_update_cap": float(shared_update_cap),
                },
            )
            shared_score = selection._policy_score(
                shared_train,
                shared_policy,
                downside_penalty=downside,
                tail_risk_weight=tail,
            )
            diagnostics = {
                **diagnostics,
                "shared_update_cap_applied": True,
                "train_risk_score": shared_score[0],
                "train_primary_mean": shared_score[1],
                "train_worst_dataset_condition_gain": shared_score[2],
            }
        policies[(fold, "shared", "all_shifts")] = shared_policy
        selection_rows.append(
            selection._policy_row(
                "regression",
                fold,
                "shared",
                "all_shifts",
                shared_policy,
                diagnostics,
                shared_select,
            )
        )
        print(
            f"fold={fold} Shared contexts={len(shared_policy.members)}",
            flush=True,
        )

        for family in FAMILIES:
            specific_train = []
            for dataset in train:
                for strength in STRENGTHS:
                    specific_view = views[(dataset, family, strength, "specific")]
                    if override.get("specific_uses_shared_anchor", False):
                        shared_view = views[(dataset, family, strength, "shared")]
                        shared_prediction = predict(
                            selection._bank(shared_view, shared_policy.members),
                            shared_policy,
                        )
                        specific_view = selection._with_base(
                            specific_view,
                            shared_prediction,
                        )
                    specific_train.append(specific_view)
            specific_origin_reference = None
            if (
                origin_noninferiority
                and not override.get("specific_requires_incremental_context", False)
            ):
                specific_origin_reference = float(np.mean([
                    selection._primary(view, view.base)
                    for view in specific_train
                    if view.rho == 0
                ]))
            specific_policy, diagnostics = selection._select_policy(
                specific_train,
                max_members=DEFAULT_PROTOCOL.tier("specific").max_contexts,
                base_downside_penalty=downside,
                tail_risk_weight=tail,
                origin_reference_mean=specific_origin_reference,
                origin_noninferiority_tolerance=origin_tolerance,
                enable_local_router=bool(
                    override.get("specific_enable_local_router", False)
                ),
                fast_local_router=fast_hierarchical,
                fast_candidate_search=fast_hierarchical,
                skip_gate_search=fast_hierarchical,
                allow_base_fallback=not bool(
                    override.get("specific_requires_incremental_context", False)
                ),
                group_noninferiority_margin=(
                    selection.SPECIFIC_GROUP_NONINFERIORITY_MARGIN
                ),
            )
            specific_nested_evidence = (
                _nested_upgrade_evidence(
                    outer_train=train,
                    family=family,
                    method="specific",
                    views=views,
                    folds=folds,
                    downside=downside,
                    tail=tail,
                    origin_noninferiority=origin_noninferiority,
                    origin_tolerance=origin_tolerance,
                    specific_policy_cache=nested_specific_cache,
                )
                if nested_upgrade_gate
                else float("nan")
            )
            if (
                nested_upgrade_gate
                and specific_nested_evidence <= nested_minimum
            ):
                specific_policy = FusionPolicy(
                    (),
                    "mean",
                    0.0,
                    metadata={"candidate": "nested_base_fallback"},
                )
            if (
                override.get("specific_requires_incremental_context", False)
                and not specific_policy.members
                and not selection.GROUP_NONINFERIORITY_ENABLED
            ):
                specific_singletons = [
                    FusionPolicy(
                        (name,),
                        "mean",
                        alpha,
                        metadata={
                            "candidate": "required_specific_singleton_fallback",
                            "training_selected": True,
                        },
                    )
                    for name in specific_train[0].members
                    for alpha in selection.ALPHAS[1:]
                ]
                *specific_score, specific_policy = max(
                    [
                    (
                        *selection._policy_score(
                            specific_train,
                            candidate,
                            downside_penalty=downside,
                            tail_risk_weight=tail,
                        ),
                        candidate,
                    )
                    for candidate in specific_singletons
                    ],
                    key=lambda item: item[:3],
                )
                diagnostics = {
                    **diagnostics,
                    "required_specific_singleton_fallback": True,
                    "train_risk_score": specific_score[0],
                    "train_primary_mean": specific_score[1],
                    "train_worst_dataset_condition_gain": specific_score[2],
                }
            diagnostics = {
                **diagnostics,
                "nested_upgrade_evidence": specific_nested_evidence,
                "nested_minimum_gain": nested_minimum,
            }
            policies[(fold, "specific", family)] = specific_policy
            selection_rows.append(
                selection._policy_row(
                    "regression",
                    fold,
                    "specific",
                    family,
                    specific_policy,
                    diagnostics,
                    specific_train,
                )
            )

            pro_train = []
            for dataset in train:
                for strength in STRENGTHS:
                    raw_pro_view = views[(dataset, family, strength, "pro")]
                    shared_view = views[(dataset, family, strength, "shared")]
                    specific_view = views[(dataset, family, strength, "specific")]
                    shared_prediction = predict(
                        selection._bank(shared_view, shared_policy.members),
                        shared_policy,
                    )
                    specific_prediction = predict(
                        selection._bank(
                            selection._with_base(
                                specific_view,
                                shared_prediction,
                            )
                            if override.get("specific_uses_shared_anchor", False)
                            else specific_view,
                            specific_policy.members,
                        ),
                        specific_policy,
                    )
                    pro_view = selection._with_member(
                        raw_pro_view, "tier/shared", shared_prediction
                    )
                    pro_view = selection._with_base(
                        pro_view,
                        specific_prediction,
                        exclude_members=("tier/shared", *specific_policy.members),
                    )
                    pro_train.append(pro_view)
            pro_origin_reference = None
            if origin_noninferiority and bool(
                override.get("pro_origin_noninferiority_gate", True)
            ):
                pro_origin_reference = float(np.mean([
                    selection._primary(
                        view,
                        view.base,
                    )
                    for view in pro_train
                    if view.rho == 0
                ]))
            exact_specific_fallback = FusionPolicy(
                (),
                "mean",
                0.0,
                metadata={"candidate": "exact_lower_tier_anchor"},
            )
            pro_policy, diagnostics = selection._select_policy(
                pro_train,
                max_members=DEFAULT_PROTOCOL.tier("pro").max_contexts,
                fallback_policies=(exact_specific_fallback,),
                base_downside_penalty=pro_downside,
                tail_risk_weight=tail,
                origin_reference_mean=pro_origin_reference,
                origin_noninferiority_tolerance=origin_tolerance,
                enable_local_router=bool(
                    override.get("pro_enable_local_router", True)
                ),
                fast_local_router=fast_hierarchical,
                fast_candidate_search=fast_hierarchical,
                skip_gate_search=fast_hierarchical,
                allow_base_fallback=not bool(
                    override.get("pro_requires_incremental_context", False)
                ),
                group_noninferiority_margin=(
                    selection.PRO_GROUP_NONINFERIORITY_MARGIN
                ),
            )
            pro_nested_evidence = (
                _nested_upgrade_evidence(
                    outer_train=train,
                    family=family,
                    method="pro",
                    views=views,
                    folds=folds,
                    downside=downside,
                    tail=tail,
                    origin_noninferiority=origin_noninferiority,
                    origin_tolerance=origin_tolerance,
                    specific_policy_cache=nested_specific_cache,
                )
                if pro_nested_upgrade_gate
                else float("nan")
            )
            pro_score = selection._policy_score(
                pro_train,
                pro_policy,
                downside_penalty=0.0,
                tail_risk_weight=tail,
            )
            specific_score = selection._policy_score(
                pro_train,
                exact_specific_fallback,
                downside_penalty=0.0,
                tail_risk_weight=tail,
            )
            pro_fold_stability = _fold_stability_evidence(
                pro_train, pro_policy, exact_specific_fallback
            )
            if (
                (
                    not override.get("pro_requires_incremental_context", False)
                    and specific_score > pro_score
                )
                or (
                    pro_nested_upgrade_gate
                    and pro_nested_evidence <= nested_minimum
                )
                or (
                    pro_fold_stability_gate
                    and pro_fold_stability <= 0.0
                )
            ):
                pro_policy = FusionPolicy(
                    (),
                    "mean",
                    0.0,
                    metadata={
                        "candidate": "specific_anchor_fallback",
                    },
                )
            if (
                override.get("pro_requires_incremental_context", False)
                and not pro_policy.members
                and not selection.GROUP_NONINFERIORITY_ENABLED
            ):
                singleton_candidates = [
                    FusionPolicy(
                        (name,),
                        "mean",
                        alpha,
                        metadata={
                            "candidate": "required_increment_singleton_fallback",
                            "training_selected": True,
                        },
                    )
                    for name in pro_train[0].members
                    for alpha in selection.ALPHAS[1:]
                ]
                scored_singletons = [
                    (
                        *selection._policy_score(
                            pro_train,
                            candidate,
                            downside_penalty=pro_downside,
                            tail_risk_weight=tail,
                        ),
                        candidate,
                    )
                    for candidate in singleton_candidates
                ]
                *singleton_score, pro_policy = max(
                    scored_singletons,
                    key=lambda item: item[:3],
                )
                diagnostics = {
                    **diagnostics,
                    "required_increment_singleton_fallback": True,
                    "train_risk_score": singleton_score[0],
                    "train_primary_mean": singleton_score[1],
                    "train_worst_dataset_condition_gain": singleton_score[2],
                }
            if (
                override.get("pro_requires_incremental_context", False)
                and not pro_policy.members
                and not selection.GROUP_NONINFERIORITY_ENABLED
            ):
                raise RuntimeError("effect-first Pro must contain an incremental Context")
            pro_update_cap = override.get("pro_update_cap")
            if (
                pro_update_cap is not None
                and pro_policy.members
                and float(pro_policy.alpha) > float(pro_update_cap)
            ):
                pro_policy = replace(
                    pro_policy,
                    alpha=float(pro_update_cap),
                    metadata={
                        **pro_policy.metadata,
                        "training_selected_uncapped_alpha": float(pro_policy.alpha),
                        "adaptive_update_cap": float(pro_update_cap),
                    },
                )
                capped_score = selection._policy_score(
                    pro_train,
                    pro_policy,
                    downside_penalty=pro_downside,
                    tail_risk_weight=tail,
                )
                diagnostics = {
                    **diagnostics,
                    "pro_update_cap_applied": True,
                    "train_risk_score": capped_score[0],
                    "train_primary_mean": capped_score[1],
                    "train_worst_dataset_condition_gain": capped_score[2],
                }
            diagnostics = {
                **diagnostics,
                "nested_upgrade_evidence": pro_nested_evidence,
                "nested_minimum_gain": nested_minimum,
                "fold_stability_evidence": pro_fold_stability,
            }
            policies[(fold, "pro", family)] = pro_policy
            selection_rows.append(
                selection._policy_row(
                    "regression",
                    fold,
                    "pro",
                    family,
                    pro_policy,
                    diagnostics,
                    pro_train,
                )
            )
            print(
                f"fold={fold} {family} Specific={len(specific_policy.members)} "
                f"Pro={len(pro_policy.members)}",
                flush=True,
            )

    detail: list[dict[str, Any]] = []
    for dataset in datasets:
        fold = folds[dataset]
        if fold not in fold_scope:
            continue
        for family in FAMILIES:
            for strength in STRENGTHS:
                base_view = views[(dataset, family, strength, "shared")]
                predictions = {"base": base_view.base}
                shared_view = views[(dataset, family, strength, "shared")]
                shared_policy = policies[(fold, "shared", "all_shifts")]
                predictions["shared"] = infer_context_view(
                    tier="shared", view=shared_view, policy=shared_policy
                )
                specific_view = views[(dataset, family, strength, "specific")]
                if override.get("specific_uses_shared_anchor", False):
                    specific_view = selection._with_base(
                        specific_view,
                        predictions["shared"],
                    )
                specific_policy = policies[(fold, "specific", family)]
                predictions["specific"] = infer_context_view(
                    tier="specific", view=specific_view, policy=specific_policy
                )
                pro_policy = policies[(fold, "pro", family)]
                raw_pro_view = views[(dataset, family, strength, "pro")]
                pro_view = selection._with_member(
                    raw_pro_view, "tier/shared", predictions["shared"]
                )
                pro_view = selection._with_base(
                    pro_view,
                    predictions["specific"],
                    exclude_members=(
                        "tier/shared",
                        *policies[(fold, "specific", family)].members,
                    ),
                )
                predictions["pro"] = infer_context_view(
                    tier="pro", view=pro_view, policy=pro_policy
                )
                row: dict[str, Any] = {
                    "task": "regression",
                    "dataset": dataset,
                    "outer_fold": fold,
                    "shift": family,
                    "strength": strength,
                    "shared_shift_known": False,
                    "specific_shift_known": True,
                    "pro_shift_known": True,
                    "strength_used_for_selection_or_inference": False,
                    "shift_presence_used_for_selection_or_inference": False,
                    "outer_test_labels_used_for_selection": False,
                }
                for method, prediction in predictions.items():
                    row.update(
                        {
                            f"{method}_{key}": value
                            for key, value in _metrics(base_view.target, prediction).items()
                        }
                    )
                detail.append(row)

    summary: list[dict[str, Any]] = []
    for family in FAMILIES:
        for strength in STRENGTHS:
            cell = [
                row
                for row in detail
                if row["shift"] == family
                and np.isclose(float(row["strength"]), strength)
            ]
            out: dict[str, Any] = {
                "task": "regression",
                "shift": family,
                "strength": strength,
                "n_datasets": len(cell),
            }
            for method in METHODS:
                for metric in METRICS:
                    out[f"mean_{method}_{metric}"] = float(
                        np.mean([row[f"{method}_{metric}"] for row in cell])
                    )
            summary.append(out)
    _write_csv(output / "detail_oof.csv", detail)
    _write_csv(output / "summary_oof.csv", summary)
    _write_csv(output / "outer_selections.csv", selection_rows)
    candidate_counts_by_shift = {
        family: {
            method: len(schema)
            for method, schema in schemas["by_family"][family].items()
        }
        for family in FAMILIES
    }
    audit = {
        "api_version": DEFAULT_PROTOCOL.api_version,
        "protocol_id": DEFAULT_PROTOCOL.protocol_id,
        "task": "regression",
        "backbone": backbone_display,
        "checkpoint": checkpoint,
        "prediction_bank": str(bank_root),
        "family_prediction_bank_overrides": {
            family: str(path)
            for family, path in family_bank_overrides.items()
        },
        "bank_manipulation_protocol": bank_manipulation_protocol,
        "bank_manipulation_protocol_by_source": {
            source_name: source_audit.get("manipulation_protocol", "original")
            for source_name, source_audit in bank_audits.items()
        },
        "bank_base_prediction_refreshed_for_manipulation": bank_base_refreshed,
        "datasets": len(datasets),
        "detail_rows": len(detail),
        "outer_folds": len(fold_scope),
        "selected_outer_folds": list(fold_scope),
        "tier_context_candidates": {
            "shared": len(schemas["shared"]),
            "specific": max(
                values["specific"] for values in candidate_counts_by_shift.values()
            ),
            "pro": max(
                values["pro"] for values in candidate_counts_by_shift.values()
            ),
        },
        "tier_context_candidates_by_shift": candidate_counts_by_shift,
        "pro_lower_tier_context_candidates": 2,
        "pro_effective_context_candidates": max(
            values["pro"] for values in candidate_counts_by_shift.values()
        ) + 2,
        "pro_effective_context_candidates_by_shift": {
            family: values["pro"] + 2
            for family, values in candidate_counts_by_shift.items()
        },
        "shared_context_schema": list(schemas["shared"]),
        "shared_policy": (
            "locked five-Context arithmetic mean with OOF residual alpha"
            if override.get("locked_shared_arithmetic_mean", False)
            else "outer-training adaptive selection"
        ),
        "shared_context_registry_locked": bool(
            override.get("locked_shared_arithmetic_mean", False)
        ),
        "shared_residual_alpha_adaptive": bool(
            override.get("locked_shared_adaptive_residual_alpha", False)
        ),
        "shared_selector_receives_shift": False,
        "specific_selector_receives_shift": True,
        "pro_selector_receives_shift": True,
        "all_selectors_receive_strength": False,
        "all_selectors_receive_shift_presence": False,
        "outer_test_labels_used_for_selection": False,
        "clean_history_or_reference_used": False,
        "target_labels_used_for_context_construction": False,
        "outer_training_loss_uses_training_target_scale": True,
        "inference_uses_target_label_scale": False,
        "unlabeled_target_features_allowed": True,
        "one_policy_per_shift_scope_and_outer_fold": True,
        "training_origin_noninferiority_gate": origin_noninferiority,
        "training_origin_noninferiority_tolerance_nrmse": origin_tolerance,
        "nested_upgrade_gate_enabled": nested_upgrade_gate,
        "pro_nested_upgrade_gate_enabled": pro_nested_upgrade_gate,
        "pro_fold_stability_gate_enabled": pro_fold_stability_gate,
        "benchmark_tuning_override": override,
        "selection_profile": selection_profile,
        "pro_incremental_specific_anchor": True,
        "pro_requires_incremental_context": bool(
            override.get("pro_requires_incremental_context", False)
        ),
        "hierarchical_tier_anchors": bool(
            override.get("specific_uses_shared_anchor", False)
        ),
        "common_inference_core": "unified_context_ensemble.src.core.predict",
        "common_inference_interface": (
            "unified_context_ensemble.src.inference.run_context_inference"
        ),
        "batch_context_router_uses_only_unlabeled_predictions": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backbone-display", default="TabICL-v2")
    parser.add_argument(
        "--checkpoint", default="tabicl-regressor-v2-20260212.ckpt"
    )
    parser.add_argument(
        "--family-bank-override",
        action="append",
        default=[],
        help=(
            "Route a complete shift family to another prediction bank as "
            "FAMILY=PATH; repeat for multiple families."
        ),
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated development subset; default uses all registered datasets.",
    )
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
            "incremental_origin_safe",
            "hierarchical_effect_first",
            "hierarchical_fixed_pro",
            "hierarchical_safe_pro",
            "hierarchical_max_effect_pro",
            "hierarchical_conservative_shared",
            "hierarchical_locked_shared",
        ),
        default="protocol",
    )
    args = parser.parse_args()
    family_bank_overrides = _parse_family_bank_overrides(
        args.family_bank_override
    )
    selected_datasets = (
        None
        if args.datasets.strip().lower() == "all"
        else tuple(
            part.strip() for part in args.datasets.split(",") if part.strip()
        )
    )
    build(
        args.bank_root,
        args.output,
        selection_profile=args.selection_profile,
        selected_outer_folds=(
            None if args.outer_fold is None else tuple(args.outer_fold)
        ),
        selected_datasets=selected_datasets,
        family_bank_overrides=family_bank_overrides,
        backbone_display=args.backbone_display,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
