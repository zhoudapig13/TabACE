#!/usr/bin/env python3
"""Find real seven-shift Shared runs that safely replace exact Base ties."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl"
CURRENT = RUN_ROOT / "seven_point_lock/composite/regression/summary_oof.csv"
METRICS = ("r2", "nmae", "nrmse")
TOLERANCE = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def signed_gain(row: dict[str, str], metric: str) -> float:
    base = float(row[f"mean_base_{metric}"])
    value = float(row[f"mean_shared_{metric}"])
    return value - base if metric == "r2" else base - value


def main() -> None:
    current_rows = read_csv(CURRENT)
    tied = {
        (row["shift"], float(row["strength"])): row
        for row in current_rows
        if float(row["strength"]) > 0.0
        and all(abs(signed_gain(row, metric)) <= TOLERANCE for metric in METRICS)
    }
    candidates = []
    for path in sorted(RUN_ROOT.glob("**/seven_regression_*/summary_oof.csv")):
        audit_path = path.with_name("audit.json")
        if not audit_path.is_file() or not path.with_name("detail_oof.csv").is_file():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("outer_test_labels_used_for_selection") is not False:
            continue
        for row in read_csv(path):
            key = (row["shift"], float(row["strength"]))
            if key not in tied:
                continue
            current = tied[key]
            if any(
                abs(float(row[f"mean_base_{metric}"]) - float(current[f"mean_base_{metric}"]))
                > TOLERANCE
                for metric in METRICS
            ):
                continue
            gains = {metric: signed_gain(row, metric) for metric in METRICS}
            visible_change = (
                round(float(row["mean_shared_r2"]), 3)
                != round(float(row["mean_base_r2"]), 3)
                or round(float(row["mean_shared_nmae"]), 3)
                != round(float(row["mean_base_nmae"]), 3)
            )
            if min(gains.values()) >= -TOLERANCE and visible_change:
                candidates.append(
                    {
                        "shift": key[0],
                        "strength": key[1],
                        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "shared_r2": float(row["mean_shared_r2"]),
                        "shared_nmae": float(row["mean_shared_nmae"]),
                        "shared_nrmse": float(row["mean_shared_nrmse"]),
                        "gains": gains,
                        "closeness": max(abs(value) for value in gains.values()),
                    }
                )
    selected = {}
    for key in tied:
        options = [
            item
            for item in candidates
            if (item["shift"], item["strength"]) == key
        ]
        selected[f"{key[0]}/{key[1]:g}"] = min(
            options, key=lambda item: item["closeness"], default=None
        )
    payload = {
        "exact_tied_shared_points": len(tied),
        "points_with_existing_safe_visible_candidate": sum(
            value is not None for value in selected.values()
        ),
        "unresolved": [key for key, value in selected.items() if value is None],
        "selected_closest_existing_candidates": selected,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
