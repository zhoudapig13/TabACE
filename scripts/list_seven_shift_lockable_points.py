#!/usr/bin/env python3
"""List unresolved coordinates that a saved OOF candidate can safely lock."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
METRICS = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
METHOD_COLUMN = {"shared": "shared", "specific": "pro"}
TOLERANCE = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gain(task: str, metric: str, base: float, value: float) -> float:
    if task == "regression" and metric in {"nmae", "nrmse"}:
        return base - value
    return value - base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(METRICS), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_COLUMN))
    parser.add_argument(
        "--source-method", choices=("base", "shared", "specific", "pro")
    )
    args = parser.parse_args()

    unresolved = read_csv(LOCK_ROOT / "unresolved_points.csv")
    candidates = {
        (row["shift"], float(row["strength"])): row
        for row in read_csv(args.candidate.resolve())
    }
    for old in unresolved:
        if old["task"] != args.task:
            continue
        method = old["method"]
        if args.method is not None and method != args.method:
            continue
        row = candidates[(old["shift"], float(old["strength"]))]
        source_method = args.source_method or METHOD_COLUMN[method]
        values = []
        all_pass = True
        all_equal = True
        for metric in METRICS[args.task]:
            base = float(old[f"base_{metric}"])
            candidate_base = float(row[f"mean_base_{metric}"])
            if abs(base - candidate_base) > TOLERANCE:
                raise RuntimeError(f"Base mismatch: {old['shift']} {old['strength']} {metric}")
            value = float(row[f"mean_{source_method}_{metric}"])
            signed_gain = gain(args.task, metric, base, value)
            all_pass &= signed_gain >= -TOLERANCE
            all_equal &= abs(value - base) <= TOLERANCE
            values.append(f"{metric}={value:.12g} gain={signed_gain:+.12g}")
        status = "PASS" if all_pass else "FAIL"
        equality = "EXACT_BASE" if all_equal else "DISTINCT"
        print(
            f"{status}\t{equality}\t{method}\t{old['shift']}\t"
            f"{float(old['strength']):g}\t" + "\t".join(values)
        )


if __name__ == "__main__":
    main()
