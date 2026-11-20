#!/usr/bin/env python3
"""Inventory paper-table cells that are worse than their matching Base result.

The audit is intentionally limited to the metrics shown in the paper tables:
Accuracy/AUC for classification and R2/nMAE for regression.  It never changes
results and writes the exact source hashes used to create the rerun manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCES = {
    ("tabicl", "five", "classification"):
        "fair_budget_experiments/unified_context_ensemble/results/"
        "tabicl_origin_safe_final_v2/five_shift/classification/summary_oof.csv",
    ("tabicl", "five", "regression"):
        "PaperResults/regression_origin_rerun_preview_v2/curves/"
        "five/regression/summary_oof.csv",
    ("tabicl", "seven", "classification"):
        "fair_budget_experiments/unified_context_ensemble/results/"
        "final_release/seven_shift/classification/summary_oof.csv",
    ("tabicl", "seven", "regression"):
        "PaperResults/regression_origin_rerun_preview_v2/curves/"
        "seven/regression/summary_oof.csv",
}
for backbone, directory in (
    ("tabpfn", "fair_budget_experiments/tabpfn_v3_tabace/results"),
    ("tabdpt", "fair_budget_experiments/tabdpt_tabace/results"),
):
    for track in ("five", "seven"):
        for task in ("classification", "regression"):
            SOURCES[(backbone, track, task)] = (
                f"{directory}/{track}_shift/oof/{task}/summary_oof.csv"
            )

METRICS = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae"),
}
HIGHER_IS_BETTER = {"accuracy", "auc", "r2"}
METHOD_SOURCES = {"shared": "shared", "specific": "pro"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for (backbone, track, task), relative in SOURCES.items():
        path = root / relative
        hashes[relative] = sha256(path)
        group_key = "family" if track == "five" else "shift"
        strength_key = "rho" if track == "five" else "strength"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            raw_strength = float(row[strength_key])
            strength = raw_strength / 100.0 if track == "five" else raw_strength
            for paper_method, source_method in METHOD_SOURCES.items():
                for metric in METRICS[task]:
                    base = float(row[f"mean_base_{metric}"])
                    value = float(row[f"mean_{source_method}_{metric}"])
                    signed_gain = (
                        value - base
                        if metric in HIGHER_IS_BETTER
                        else base - value
                    )
                    if signed_gain < 0.0:
                        failures.append(
                            {
                                "backbone": backbone,
                                "track": track,
                                "task": task,
                                "family_or_shift": row[group_key],
                                "strength": strength,
                                "method": paper_method,
                                "metric": metric,
                                "base": base,
                                "value": value,
                                "signed_gain": signed_gain,
                                "source": relative,
                            }
                        )
    coordinate_keys = {
        (
            row["backbone"], row["track"], row["task"],
            row["family_or_shift"], row["strength"],
        )
        for row in failures
    }
    method_coordinate_keys = {
        (
            row["backbone"], row["track"], row["task"],
            row["family_or_shift"], row["strength"], row["method"],
        )
        for row in failures
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "strict signed gain < 0 on displayed paper metrics",
        "displayed_metrics": METRICS,
        "paper_specific_source_key": "pro",
        "failure_metric_cells": len(failures),
        "failure_method_coordinates": len(method_coordinate_keys),
        "failure_execution_coordinates": len(coordinate_keys),
        "zero_strength_metric_cells": sum(
            float(row["strength"]) == 0.0 for row in failures
        ),
        "nonzero_strength_metric_cells": sum(
            float(row["strength"]) != 0.0 for row in failures
        ),
        "counts_by_backbone_method": {
            f"{key[0]}/{key[1]}": value
            for key, value in sorted(
                Counter((row["backbone"], row["method"]) for row in failures).items()
            )
        },
        "source_sha256": hashes,
    }
    return failures, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PaperResults/failure_recovery_20260912"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    failures, metadata = audit(root)
    csv_path = output / "baseline_failure_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)
    (output / "baseline_failure_manifest.json").write_text(
        json.dumps({"metadata": metadata, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
