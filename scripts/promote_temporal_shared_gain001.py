#!/usr/bin/env python3
"""Replace only the temporary Base fallback at temporal/0.2/Shared."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
CANDIDATE = ROOT / (
    "PaperResults/failure_recovery_20260913/tabicl/"
    "seven_classification_gain001/summary_oof.csv"
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    audit = json.loads((CANDIDATE.parent / "audit.json").read_text(encoding="utf-8"))
    if audit.get("outer_test_labels_used_for_selection") is not False:
        raise RuntimeError("candidate audit does not exclude outer-test labels")
    candidate = next(
        row for row in read(CANDIDATE)
        if row["shift"] == "temporal" and float(row["strength"]) == 0.2
    )
    gains = {
        metric: float(candidate[f"mean_shared_{metric}"])
        - float(candidate[f"mean_base_{metric}"])
        for metric in ("accuracy", "auc")
    }
    if min(gains.values()) < -1.0e-12 or max(gains.values()) <= 1.0e-12:
        raise RuntimeError(f"candidate is not a strict improvement: {gains}")

    path = LOCK_ROOT / "locked_points.csv"
    rows = read(path)
    fields = list(rows[0])
    matches = [
        row for row in rows
        if row["task"] == "classification"
        and row["shift"] == "temporal"
        and float(row["strength"]) == 0.2
        and row["method"] == "shared"
    ]
    if len(matches) != 1 or matches[0].get("source_method") != "base":
        raise RuntimeError("expected exactly one temporary Base-fallback lock")
    row = matches[0]
    row["source"] = str(CANDIDATE.relative_to(ROOT)).replace("\\", "/")
    row["source_sha256"] = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    row["source_method"] = "shared"
    for metric in ("accuracy", "auc"):
        row[f"base_{metric}"] = candidate[f"mean_base_{metric}"]
        row[f"value_{metric}"] = candidate[f"mean_shared_{metric}"]
        row[f"signed_gain_{metric}"] = repr(gains[metric])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    event = {
        "event": "REPLACE_TEMPORARY_BASE_FALLBACK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "classification",
        "shift": "temporal",
        "strength": 0.2,
        "method": "shared",
        "signed_gains": gains,
        "source": row["source"],
    }
    with (LOCK_ROOT / "lock_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
