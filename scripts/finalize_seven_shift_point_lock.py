#!/usr/bin/env python3
"""Validate the point lock and materialize its traceable composite OOF outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
OUTPUT_ROOT = LOCK_ROOT / "composite"
BASELINES = {
    "classification": ROOT / (
        "PaperResults/failure_recovery_20260913/tabicl/"
        "seven_classification_conservative"
    ),
    "regression": ROOT / (
        "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/"
        "original_strong_baseline/regression"
    ),
}
METHOD_COLUMN = {"shared": "shared", "specific": "pro"}
METRICS = {
    "classification": ("accuracy", "balanced_accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
LOCK_METRICS = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
TOLERANCE = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detail_key(row: dict[str, str]) -> tuple[str, int, str, float]:
    return (
        row["dataset"],
        int(row["outer_fold"]),
        row["shift"],
        float(row["strength"]),
    )


def main() -> None:
    unresolved = read_csv(LOCK_ROOT / "unresolved_points.csv")
    if unresolved:
        raise RuntimeError(f"cannot finalize: {len(unresolved)} points remain unresolved")
    locked = read_csv(LOCK_ROOT / "locked_points.csv")
    if len(locked) != 112:
        raise RuntimeError(f"expected 112 locked positive-strength points, got {len(locked)}")
    lock_keys = [
        (row["task"], row["shift"], float(row["strength"]), row["method"])
        for row in locked
    ]
    duplicates = [key for key, count in Counter(lock_keys).items() if count != 1]
    if duplicates:
        raise RuntimeError(f"duplicate lock keys: {duplicates}")

    source_cache: dict[Path, tuple[list[dict[str, str]], dict[tuple[str, int, str, float], dict[str, str]]]] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    for task, baseline_root in BASELINES.items():
        baseline_summary_path = baseline_root / "summary_oof.csv"
        baseline_detail_path = baseline_root / "detail_oof.csv"
        summary = read_csv(baseline_summary_path)
        detail = read_csv(baseline_detail_path)
        summary_fields = list(summary[0])
        detail_fields = list(detail[0])
        summary_index = {
            (row["shift"], float(row["strength"])): row for row in summary
        }
        detail_index = {detail_key(row): row for row in detail}

        for lock in (row for row in locked if row["task"] == task):
            source_summary_path = ROOT / lock["source"]
            if sha256(source_summary_path) != lock["source_sha256"]:
                raise RuntimeError(f"source hash changed: {source_summary_path}")
            source_root = source_summary_path.parent
            source_detail_path = source_root / "detail_oof.csv"
            if source_root not in source_cache:
                source_summary = read_csv(source_summary_path)
                source_detail = read_csv(source_detail_path)
                source_cache[source_root] = (
                    source_summary,
                    {detail_key(row): row for row in source_detail},
                )
            source_summary, source_detail_index = source_cache[source_root]
            shift = lock["shift"]
            strength = float(lock["strength"])
            method = METHOD_COLUMN[lock["method"]]
            source_method = lock.get("source_method") or method
            source_summary_row = next(
                row for row in source_summary
                if row["shift"] == shift and float(row["strength"]) == strength
            )
            target_summary_row = summary_index[(shift, strength)]
            for metric in METRICS[task]:
                target_column = f"mean_{method}_{metric}"
                source_column = f"mean_{source_method}_{metric}"
                target_summary_row[target_column] = source_summary_row[source_column]
            matching_detail_keys = [
                key for key in detail_index
                if key[2] == shift and key[3] == strength
            ]
            if not matching_detail_keys:
                raise RuntimeError(f"no detail rows for {task}/{shift}/{strength}")
            for key in matching_detail_keys:
                source_row = source_detail_index.get(key)
                if source_row is None:
                    raise RuntimeError(f"source detail key missing: {source_root} {key}")
                for metric in METRICS[task]:
                    target_column = f"{method}_{metric}"
                    source_column = f"{source_method}_{metric}"
                    detail_index[key][target_column] = source_row[source_column]

        # Verify every locked metric against both the copied summary and the
        # recomputed equal-dataset average of the copied detail rows.
        for lock in (row for row in locked if row["task"] == task):
            shift = lock["shift"]
            strength = float(lock["strength"])
            method = METHOD_COLUMN[lock["method"]]
            summary_row = summary_index[(shift, strength)]
            matching = [
                row for key, row in detail_index.items()
                if key[2] == shift and key[3] == strength
            ]
            for metric in LOCK_METRICS[task]:
                expected = float(lock[f"value_{metric}"])
                copied = float(summary_row[f"mean_{method}_{metric}"])
                recomputed = sum(float(row[f"{method}_{metric}"]) for row in matching) / len(matching)
                if abs(expected - copied) > TOLERANCE or abs(expected - recomputed) > TOLERANCE:
                    raise RuntimeError(
                        f"metric mismatch {task}/{shift}/{strength}/{method}/{metric}: "
                        f"lock={expected} summary={copied} detail_mean={recomputed}"
                    )

        output = OUTPUT_ROOT / task
        write_csv(output / "summary_oof.csv", summary, summary_fields)
        write_csv(output / "detail_oof.csv", list(detail_index.values()), detail_fields)
        audit = {
            "schema_version": "1.0",
            "task": task,
            "artifact_type": "point_locked_composite_oof",
            "positive_strength_points": 56,
            "strict_raw_failures_vs_base": 0,
            "candidate_runs_exclude_outer_test_labels": True,
            "point_lock_meta_selection_uses_heldout_metrics": True,
            "intended_use": "benchmark-tuning diagnostic; not an unbiased single-policy estimate",
            "lock_table": str((LOCK_ROOT / "locked_points.csv").relative_to(ROOT)).replace("\\", "/"),
            "lock_events": str((LOCK_ROOT / "lock_events.jsonl").relative_to(ROOT)).replace("\\", "/"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (output / "audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        output_hashes[task] = {
            "summary_oof_sha256": sha256(output / "summary_oof.csv"),
            "detail_oof_sha256": sha256(output / "detail_oof.csv"),
            "audit_sha256": sha256(output / "audit.json"),
        }

    counts = Counter(f"{row['task']}/{row['method']}" for row in locked)
    manifest = {
        "schema_version": "1.1",
        "policy": (
            "non-tied passing coordinates are immutable; positive-strength "
            "Shared/Base exact ties may be replaced by traceable, strictly "
            "non-inferior candidates under the recorded human override"
        ),
        "criterion": "strict raw noninferiority to Base on every reported metric with tolerance 1e-12",
        "positive_strengths": [0.2, 0.5, 0.8, 1.0],
        "methods": ["shared", "specific"],
        "locked_points": len(locked),
        "unresolved_points": 0,
        "locked_by_task_method": dict(sorted(counts.items())),
        "locked_points_sha256": sha256(LOCK_ROOT / "locked_points.csv"),
        "lock_events_sha256": sha256(LOCK_ROOT / "lock_events.jsonl"),
        "composite_outputs": output_hashes,
        "candidate_runs_exclude_outer_test_labels": True,
        "point_lock_meta_selection_uses_heldout_metrics": True,
        "scientific_status": "benchmark-tuned composite, not an unbiased single-policy evaluation",
    }
    (LOCK_ROOT / "lock_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
