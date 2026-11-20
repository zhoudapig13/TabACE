#!/usr/bin/env python3
"""Lock newly passing coordinates without changing any previously locked value."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
METRICS = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
METHOD_COLUMN = {"shared": "shared", "specific": "pro"}
TOLERANCE = 1.0e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _signed_gain(task: str, metric: str, base: float, value: float) -> float:
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
    parser.add_argument("--shift")
    parser.add_argument("--strength", type=float)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    audit = candidate.with_name("audit.json")
    if not audit.is_file():
        raise FileNotFoundError(f"candidate audit missing: {audit}")
    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    if audit_data.get("outer_test_labels_used_for_selection") is not False:
        raise RuntimeError("candidate does not certify outer-test-label exclusion")

    locked_path = LOCK_ROOT / "locked_points.csv"
    unresolved_path = LOCK_ROOT / "unresolved_points.csv"
    locked: list[dict[str, object]] = list(_read_csv(locked_path))
    unresolved: list[dict[str, object]] = list(_read_csv(unresolved_path))
    candidate_rows = {
        (row["shift"], float(row["strength"])): row
        for row in _read_csv(candidate)
    }
    candidate_hash = _sha256(candidate)
    try:
        source = str(candidate.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        source = str(candidate)

    newly_locked: list[dict[str, object]] = []
    still_unresolved: list[dict[str, object]] = []
    for old in unresolved:
        if old["task"] != args.task:
            still_unresolved.append(old)
            continue
        if args.method is not None and old["method"] != args.method:
            still_unresolved.append(old)
            continue
        if args.shift is not None and old["shift"] != args.shift:
            still_unresolved.append(old)
            continue
        if args.strength is not None and abs(
            float(old["strength"]) - args.strength
        ) > TOLERANCE:
            still_unresolved.append(old)
            continue
        if args.method is not None and old["method"] != args.method:
            still_unresolved.append(old)
            continue
        if args.shift is not None and old["shift"] != args.shift:
            still_unresolved.append(old)
            continue
        if args.strength is not None and abs(
            float(old["strength"]) - args.strength
        ) > TOLERANCE:
            still_unresolved.append(old)
            continue
        key = (old["shift"], float(old["strength"]))
        row = candidate_rows[key]
        method = old["method"]
        source_method = args.source_method or METHOD_COLUMN[method]
        gains: dict[str, float] = {}
        record: dict[str, object] = {
            "task": args.task,
            "shift": old["shift"],
            "strength": float(old["strength"]),
            "method": method,
            "source_method": source_method,
            "source": source,
            "source_sha256": candidate_hash,
        }
        for metric in METRICS[args.task]:
            expected_base = float(old[f"base_{metric}"])
            candidate_base = float(row[f"mean_base_{metric}"])
            if abs(expected_base - candidate_base) > TOLERANCE:
                raise RuntimeError(
                    f"Base changed at {key} {metric}: "
                    f"{expected_base} != {candidate_base}"
                )
            value = float(row[f"mean_{source_method}_{metric}"])
            gain = _signed_gain(args.task, metric, candidate_base, value)
            gains[metric] = gain
            record[f"base_{metric}"] = candidate_base
            record[f"value_{metric}"] = value
            record[f"signed_gain_{metric}"] = gain
        failed = [metric for metric, gain in gains.items() if gain < -TOLERANCE]
        if failed:
            still_unresolved.append(old)
        else:
            passed = {
                **record,
                "status": "LOCKED",
                "failed_metrics": "",
            }
            locked.append(passed)
            newly_locked.append(passed)

    _write_csv(locked_path, locked)
    _write_csv(unresolved_path, still_unresolved)
    event = {
        "schema_version": "1.0",
        "event": "candidate_lock_attempt",
        "task": args.task,
        "filters": {
            "method": args.method,
            "source_method": args.source_method,
            "shift": args.shift,
            "strength": args.strength,
        },
        "candidate": source,
        "candidate_sha256": candidate_hash,
        "newly_locked": [
            {
                "shift": row["shift"],
                "strength": row["strength"],
                "method": row["method"],
            }
            for row in newly_locked
        ],
        "remaining_unresolved": sum(
            row["task"] == args.task for row in still_unresolved
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with (LOCK_ROOT / "lock_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
