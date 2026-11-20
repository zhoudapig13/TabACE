#!/usr/bin/env python3
"""Audit strong seven-shift banks against the frozen LimiX Base banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--strong-classification", type=Path, required=True)
    parser.add_argument("--strong-regression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "classification": args.strong_classification,
        "regression": args.strong_regression,
    }
    expected = {"classification": 35, "regression": 33}
    report: dict[str, object] = {
        "protocol": "limix_strong_seven_context_bank_audit_v1",
        "target_labels_used_by_constructors": False,
        "single_seed": True,
        "tasks": {},
    }
    for task, root in roots.items():
        rows = []
        task_candidate_fallbacks = 0
        task_nonfinite_replacements = 0
        paths = sorted((root / task).glob("dataset_*.npz"))
        if len(paths) != expected[task]:
            raise RuntimeError(
                f"{task}: expected {expected[task]} banks, found {len(paths)}"
            )
        for path in paths:
            base_path = args.base / task / path.name
            if not base_path.exists():
                raise FileNotFoundError(base_path)
            with np.load(path, allow_pickle=False) as strong, np.load(
                base_path, allow_pickle=False
            ) as base:
                np.testing.assert_array_equal(strong["base"], base["base"])
                np.testing.assert_array_equal(strong["y_target"], base["y_target"])
                np.testing.assert_array_equal(strong["family"], base["family"])
                np.testing.assert_array_equal(strong["strength"], base["strength"])
                members = np.asarray(strong["members"])
                names = strong["member_names"].astype(str)
                shared = strong["shared_names"].astype(str)
                registry = json.loads(str(strong["specific_registry"]))
                diagnostics = (
                    strong["strong_context_diagnostics"].astype(str)
                    if "strong_context_diagnostics" in strong.files
                    else np.asarray([], dtype=str)
                )
                nonfinite_replacements = int(
                    strong["nonfinite_candidate_predictions_replaced_by_base"]
                ) if "nonfinite_candidate_predictions_replaced_by_base" in strong.files else 0
            if not np.isfinite(members).all():
                raise RuntimeError(f"non-finite prediction remains: {path}")
            if len(shared) != 5:
                raise RuntimeError(f"Shared budget is not five: {path}: {shared}")
            missing = sorted(set(shared) - set(names))
            missing += sorted(
                {
                    candidate
                    for candidates in registry.values()
                    for candidate in candidates
                }
                - set(names)
            )
            if missing:
                raise RuntimeError(f"registry members missing from {path}: {missing}")
            candidate_fallbacks = 0
            for item in diagnostics:
                parsed = json.loads(str(item))
                candidate_fallbacks += len(parsed.get("candidate_fallbacks", []))
            task_candidate_fallbacks += candidate_fallbacks
            task_nonfinite_replacements += nonfinite_replacements
            rows.append(
                {
                    "file": path.name,
                    "member_count": int(len(names)),
                    "shared_count": int(len(shared)),
                    "specific_counts": {
                        family: int(len(candidates))
                        for family, candidates in registry.items()
                    },
                    "base_exact": True,
                    "target_exact": True,
                    "finite": True,
                    "candidate_call_fallbacks_to_base": candidate_fallbacks,
                    "nonfinite_candidate_predictions_replaced_by_base": nonfinite_replacements,
                }
            )
        report["tasks"][task] = {
            "dataset_count": len(rows),
            "condition_count": len(rows) * 35,
            "base_predictions_exactly_frozen": True,
            "target_arrays_exactly_frozen": True,
            "shared_context_budget": 5,
            "candidate_call_fallbacks_to_base": task_candidate_fallbacks,
            "nonfinite_candidate_predictions_replaced_by_base": task_nonfinite_replacements,
            "rows": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        task: {
            "datasets": report["tasks"][task]["dataset_count"],
            "conditions": report["tasks"][task]["condition_count"],
        }
        for task in roots
    }, indent=2))


if __name__ == "__main__":
    main()
