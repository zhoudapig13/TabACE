#!/usr/bin/env python3
"""Merge five independently selected seven-shift classification OOF folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SHIFTS = ("covariate", "label", "concept", "selection", "domain", "temporal", "support")
STRENGTHS = (0.0, 0.2, 0.5, 0.8, 1.0)
METHODS = ("base", "shared", "specific", "pro")
METRICS = ("accuracy", "balanced_accuracy", "auc")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.part) != 5:
        raise ValueError("exactly five fold parts are required")

    audits = [
        json.loads((part / "audit.json").read_text(encoding="utf-8"))
        for part in args.part
    ]
    folds = [int(audit["selected_outer_folds"][0]) for audit in audits]
    if sorted(folds) != list(range(5)):
        raise RuntimeError(f"fold partition is incomplete or duplicated: {folds}")
    profiles = {audit.get("route_profile", "legacy_nostrength") for audit in audits}
    if len(profiles) != 1:
        raise RuntimeError(f"route profile changed across folds: {profiles}")
    if any(audit["outer_folds"] != 1 for audit in audits):
        raise RuntimeError("each part must contain exactly one outer fold")

    detail = [row for part in args.part for row in _read(part / "detail_oof.csv")]
    selections = [
        row for part in args.part for row in _read(part / "outer_selections.csv")
    ]
    keys = {
        (row["dataset"], row["shift"], float(row["strength"])) for row in detail
    }
    if len(detail) != 35 * 7 * 5 or len(keys) != len(detail):
        raise RuntimeError("merged seven-shift detail grid is incomplete or duplicated")
    if {int(row["outer_fold"]) for row in detail} != set(range(5)):
        raise RuntimeError("merged detail does not cover all outer folds")

    summary: list[dict[str, Any]] = []
    for shift in SHIFTS:
        for strength in STRENGTHS:
            cell = [
                row for row in detail
                if row["shift"] == shift
                and np.isclose(float(row["strength"]), strength)
            ]
            if len(cell) != 35:
                raise RuntimeError(f"unexpected cell size for {shift}/{strength}")
            output: dict[str, Any] = {
                "task": "classification", "shift": shift,
                "strength": strength, "n_datasets": len(cell),
            }
            for method in METHODS:
                for metric in METRICS:
                    output[f"mean_{method}_{metric}"] = float(np.mean([
                        float(row[f"{method}_{metric}"]) for row in cell
                    ]))
            summary.append(output)

    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "detail_oof.csv", detail)
    _write(args.output / "summary_oof.csv", summary)
    _write(args.output / "outer_selections.csv", selections)
    audit = {
        "protocol_id": "seven_classification_margin_v1_merged",
        "task": "classification",
        "datasets": 35,
        "detail_rows": len(detail),
        "outer_folds": 5,
        "selected_outer_folds": list(range(5)),
        "source_parts": [str(part) for part in args.part],
        "route_profile": next(iter(profiles)),
        "shared_context_registry_fixed": True,
        "shared_raw_policy_structure_locked": True,
        "accepted_shared_source": audits[0]["accepted_shared_source"],
        "accepted_raw_shared_maximum_metric_error": max(
            float(audit["accepted_raw_shared_maximum_metric_error"])
            for audit in audits
        ),
        "accepted_raw_shared_metric_tolerance": audits[0][
            "accepted_raw_shared_metric_tolerance"
        ],
        "accepted_raw_shared_check_enforced": all(
            bool(audit.get("accepted_raw_shared_check_enforced", True))
            for audit in audits
        ),
        "accepted_raw_shared_reproduced": all(
            bool(audit.get("accepted_raw_shared_reproduced", True))
            for audit in audits
        ),
        "shared_selector_receives_family_or_strength": False,
        "nominal_strength_used_by_any_policy_selector": False,
        "policy_selection_unit": "outer_training_dataset_condition",
        "shared_stagewise_gains_by_fold": {
            key: value for audit in audits
            for key, value in audit["shared_stagewise_gains_by_fold"].items()
        },
        "shared_stagewise_gain_candidates": audits[0].get(
            "shared_stagewise_gain_candidates"
        ),
        "shared_postfusion_router": audits[0].get(
            "shared_postfusion_router"
        ),
        "specific_context_count_limit": 5,
        "specific_update_cap": audits[0]["specific_update_cap"],
        "pro_context_count_limit": None,
        "pro_postfusion_router": audits[0].get("pro_postfusion_router"),
        "pro_route_fit_scopes": audits[0].get("pro_route_fit_scopes"),
        "classification_router_training_target": audits[0].get(
            "classification_router_training_target"
        ),
        "pro_primary_only_route_candidates_excluded": audits[0].get(
            "pro_primary_only_route_candidates_excluded", False
        ),
        "route_selection_priority": audits[0].get("route_selection_priority"),
        "binary_pro_stacker": audits[0].get("binary_pro_stacker"),
        "pro_raw_context_candidates_by_shift": audits[0][
            "pro_raw_context_candidates_by_shift"
        ],
        "pro_stagewise_virtual_experts": audits[0][
            "pro_stagewise_virtual_experts"
        ],
        "pro_global_stagewise_virtual_experts": audits[0][
            "pro_global_stagewise_virtual_experts"
        ],
        "specific_and_pro_receive_family_but_not_strength": True,
        "outer_test_labels_used_for_selection": False,
        "common_inference_core": "unified_context_ensemble.src.core.predict",
        "smoothing_or_curve_projection": False,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
