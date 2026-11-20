#!/usr/bin/env python3
"""Freeze passing seven-shift OOF cells and list only unresolved coordinates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
SOURCES = {
    "classification": ROOT / (
        "PaperResults/failure_recovery_20260913/tabicl/"
        "seven_classification_conservative/summary_oof.csv"
    ),
    "regression": ROOT / (
        "PaperResults/failure_recovery_20260913/tabicl/"
        "seven_regression_pro_margin001/summary_oof.csv"
    ),
}
METRICS = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
METHODS = ("shared", "pro")
TOLERANCE = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_gain(task: str, metric: str, base: float, value: float) -> float:
    if task == "regression" and metric in {"nmae", "nrmse"}:
        return base - value
    return value - base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    locked: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    source_hashes = {task: _sha256(path) for task, path in SOURCES.items()}

    for task, source in SOURCES.items():
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if float(row["strength"]) <= 0.0:
                continue
            for method in METHODS:
                gains = {
                    metric: _signed_gain(
                        task,
                        metric,
                        float(row[f"mean_base_{metric}"]),
                        float(row[f"mean_{method}_{metric}"]),
                    )
                    for metric in METRICS[task]
                }
                record: dict[str, object] = {
                    "task": task,
                    "shift": row["shift"],
                    "strength": float(row["strength"]),
                    "method": "specific" if method == "pro" else method,
                    "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                    "source_sha256": source_hashes[task],
                }
                for metric, gain in gains.items():
                    record[f"base_{metric}"] = float(row[f"mean_base_{metric}"])
                    record[f"value_{metric}"] = float(row[f"mean_{method}_{metric}"])
                    record[f"signed_gain_{metric}"] = gain
                bad_metrics = [
                    metric for metric, gain in gains.items()
                    if gain < -TOLERANCE
                ]
                if bad_metrics:
                    unresolved.append({
                        **record,
                        "status": "UNRESOLVED",
                        "failed_metrics": ",".join(bad_metrics),
                    })
                else:
                    locked.append({
                        **record,
                        "status": "LOCKED",
                        "failed_metrics": "",
                    })

    _write_csv(OUTPUT / "locked_points.csv", locked)
    _write_csv(OUTPUT / "unresolved_points.csv", unresolved)
    manifest = {
        "schema_version": "1.0",
        "policy": (
            "non-tied passing coordinates are immutable; positive-strength "
            "Shared/Base exact ties may be replaced by traceable, strictly "
            "non-inferior candidates under a recorded human override"
        ),
        "criterion": (
            "strict raw noninferiority to Base on every metric with tolerance 1e-12"
        ),
        "positive_strengths": [0.2, 0.5, 0.8, 1.0],
        "methods": ["shared", "specific"],
        "source_hashes": source_hashes,
        "locked_points": len(locked),
        "unresolved_points": len(unresolved),
        "locked_by_task_method": {
            f"{task}/{method}": sum(
                row["task"] == task and row["method"] == method
                for row in locked
            )
            for task in SOURCES
            for method in ("shared", "specific")
        },
        "unresolved_by_task_method": {
            f"{task}/{method}": sum(
                row["task"] == task and row["method"] == method
                for row in unresolved
            )
            for task in SOURCES
            for method in ("shared", "specific")
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "lock_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
