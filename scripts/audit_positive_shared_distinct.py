#!/usr/bin/env python3
"""Audit positive-strength Shared distinctness, safety, and zero-strength freeze."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
SUMMARY = LOCK_ROOT / "composite/regression/summary_oof.csv"
ORIGINAL = LOCK_ROOT / "original_strong_baseline/regression/summary_oof.csv"
OUTPUT = ROOT / "paper/audits/positive_shared_distinct_audit.json"
METRICS = ("r2", "nmae", "nrmse")
TOLERANCE = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gain(row: dict[str, str], method: str, metric: str) -> float:
    base = float(row[f"mean_base_{metric}"])
    value = float(row[f"mean_{method}_{metric}"])
    return value - base if metric == "r2" else base - value


def pair(row: dict[str, str], method: str) -> tuple[str, str]:
    return (
        f"{float(row[f'mean_{method}_r2']):.3f}",
        f"{float(row[f'mean_{method}_nmae']):.3f}",
    )


def main() -> None:
    rows = read_csv(SUMMARY)
    positive = [row for row in rows if float(row["strength"]) > 0.0]
    zero = [row for row in rows if float(row["strength"]) == 0.0]
    original = {
        (row["shift"], float(row["strength"])): row for row in read_csv(ORIGINAL)
    }
    failures = []
    exact_ties = []
    displayed_pair_ties = []
    for row in positive:
        key = f"{row['shift']}/{row['strength']}"
        gains = {metric: gain(row, "shared", metric) for metric in METRICS}
        if min(gains.values()) < -TOLERANCE:
            failures.append({"coordinate": key, "gains": gains})
        if all(abs(value) <= TOLERANCE for value in gains.values()):
            exact_ties.append(key)
        if pair(row, "shared") == pair(row, "base"):
            displayed_pair_ties.append(key)
    zero_max_difference = max(
        abs(
            float(row[f"mean_{method}_{metric}"])
            - float(original[(row["shift"], 0.0)][f"mean_{method}_{metric}"])
        )
        for row in zero
        for method in ("base", "shared", "specific", "pro")
        for metric in METRICS
    )
    mean_gains = {
        metric: sum(gain(row, "shared", metric) for row in positive) / len(positive)
        for metric in METRICS
    }
    seven_shift_average = {
        f"{method}_{metric}": sum(
            float(row[f"mean_{method}_{metric}"]) for row in positive
        )
        / len(positive)
        for method in ("base", "shared", "pro")
        for metric in ("r2", "nrmse")
    }
    payload = {
        "status": "PASS"
        if not failures
        and not exact_ties
        and not displayed_pair_ties
        and zero_max_difference <= TOLERANCE
        else "FAIL",
        "positive_strength_coordinates": len(positive),
        "raw_failure_coordinates": failures,
        "exact_shared_base_ties": exact_ties,
        "three_decimal_r2_nmae_pair_ties": displayed_pair_ties,
        "strength_zero_unchanged": zero_max_difference <= TOLERANCE,
        "strength_zero_max_absolute_difference": zero_max_difference,
        "positive_strength_mean_signed_gain": mean_gains,
        "seven_shift_average": seven_shift_average,
        "summary_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
        "criterion": (
            "At strength > 0, Shared must be non-inferior to Base on raw "
            "R2/nMAE/nRMSE and its displayed R2/nMAE pair must differ at "
            "three decimals; strength 0 must match the frozen original."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
