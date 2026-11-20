#!/usr/bin/env python3
"""Replace positive-strength Shared/Base ties with traceable real candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl"
LOCK_ROOT = RUN_ROOT / "seven_point_lock"
CURRENT = LOCK_ROOT / "composite/regression/summary_oof.csv"
LOCKS = LOCK_ROOT / "locked_points.csv"
EVENTS = LOCK_ROOT / "lock_events.jsonl"
SCAN = LOCK_ROOT / "shared_distinct_candidate_scan.json"
REPORT = LOCK_ROOT / "shared_distinct_replacements.csv"
MANIFEST = LOCK_ROOT / "shared_distinct_replacement_manifest.json"
LOCK_BACKUP = LOCK_ROOT / "locked_points_before_shared_distinct.csv"
METRICS = ("r2", "nmae", "nrmse")
TOLERANCE = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_gain(row: dict[str, str], metric: str) -> float:
    base = float(row[f"mean_base_{metric}"])
    value = float(row[f"mean_shared_{metric}"])
    return value - base if metric == "r2" else base - value


def displayed_pair(row: dict[str, str], method: str) -> tuple[str, str]:
    return (
        f"{float(row[f'mean_{method}_r2']):.3f}",
        f"{float(row[f'mean_{method}_nmae']):.3f}",
    )


def main() -> None:
    current_rows = read_csv(CURRENT)
    tied = {
        (row["shift"], float(row["strength"])): row
        for row in current_rows
        if float(row["strength"]) > 0.0
        and all(abs(signed_gain(row, metric)) <= TOLERANCE for metric in METRICS)
    }
    if len(tied) != 17:
        raise RuntimeError(f"expected 17 positive-strength exact ties, got {len(tied)}")

    candidates: dict[tuple[str, float], list[dict[str, object]]] = {
        key: [] for key in tied
    }
    for summary_path in sorted(RUN_ROOT.glob("**/seven_regression_*/summary_oof.csv")):
        audit_path = summary_path.with_name("audit.json")
        detail_path = summary_path.with_name("detail_oof.csv")
        if not audit_path.is_file() or not detail_path.is_file():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("outer_test_labels_used_for_selection") is not False:
            continue
        for row in read_csv(summary_path):
            key = (row["shift"], float(row["strength"]))
            if key not in tied:
                continue
            current = tied[key]
            if any(
                abs(
                    float(row[f"mean_base_{metric}"])
                    - float(current[f"mean_base_{metric}"])
                )
                > TOLERANCE
                for metric in METRICS
            ):
                continue
            gains = {metric: signed_gain(row, metric) for metric in METRICS}
            if min(gains.values()) < -TOLERANCE:
                continue
            if displayed_pair(row, "shared") == displayed_pair(row, "base"):
                continue
            candidates[key].append(
                {
                    "row": row,
                    "summary": summary_path,
                    "detail": detail_path,
                    "gains": gains,
                    "closeness": max(abs(value) for value in gains.values()),
                }
            )

    selected: dict[tuple[str, float], dict[str, object]] = {}
    for key, options in candidates.items():
        if not options:
            raise RuntimeError(f"no safe visibly distinct candidate for {key}")
        selected[key] = min(options, key=lambda item: float(item["closeness"]))

    locks = read_csv(LOCKS)
    lock_fields = list(locks[0])
    replacements: list[dict[str, object]] = []
    for key, candidate in sorted(selected.items()):
        shift, strength = key
        current = tied[key]
        row = candidate["row"]
        summary_path = candidate["summary"]
        detail_path = candidate["detail"]
        assert isinstance(row, dict)
        assert isinstance(summary_path, Path)
        assert isinstance(detail_path, Path)

        details = [
            item
            for item in read_csv(detail_path)
            if item["shift"] == shift and float(item["strength"]) == strength
        ]
        if not details:
            raise RuntimeError(f"missing detail rows for {key} in {detail_path}")
        for metric in METRICS:
            detail_mean = sum(float(item[f"shared_{metric}"]) for item in details) / len(details)
            summary_value = float(row[f"mean_shared_{metric}"])
            if abs(detail_mean - summary_value) > TOLERANCE:
                raise RuntimeError(
                    f"summary/detail mismatch for {key}/{metric}: "
                    f"{summary_value} vs {detail_mean}"
                )

        lock = next(
            item
            for item in locks
            if item["task"] == "regression"
            and item["method"] == "shared"
            and item["shift"] == shift
            and float(item["strength"]) == strength
        )
        if strength <= 0.0:
            raise RuntimeError("strength-zero lock must never be changed")
        for metric in METRICS:
            if abs(float(lock[f"value_{metric}"]) - float(current[f"mean_shared_{metric}"])) > TOLERANCE:
                raise RuntimeError(f"lock/composite mismatch before replacement: {key}/{metric}")

        old = {metric: float(lock[f"value_{metric}"]) for metric in METRICS}
        source_rel = str(summary_path.relative_to(ROOT)).replace("\\", "/")
        lock["source"] = source_rel
        lock["source_sha256"] = sha256(summary_path)
        lock["source_method"] = "shared"
        lock["status"] = "LOCKED"
        lock["failed_metrics"] = ""
        for metric in METRICS:
            base = float(row[f"mean_base_{metric}"])
            value = float(row[f"mean_shared_{metric}"])
            gain = value - base if metric == "r2" else base - value
            lock[f"base_{metric}"] = repr(base)
            lock[f"value_{metric}"] = repr(value)
            lock[f"signed_gain_{metric}"] = repr(gain)
        replacements.append(
            {
                "shift": shift,
                "strength": strength,
                "old_r2": old["r2"],
                "new_r2": float(row["mean_shared_r2"]),
                "old_nmae": old["nmae"],
                "new_nmae": float(row["mean_shared_nmae"]),
                "old_nrmse": old["nrmse"],
                "new_nrmse": float(row["mean_shared_nrmse"]),
                "gain_r2": float(candidate["gains"]["r2"]),
                "gain_nmae": float(candidate["gains"]["nmae"]),
                "gain_nrmse": float(candidate["gains"]["nrmse"]),
                "source": source_rel,
            }
        )

    if not LOCK_BACKUP.exists():
        shutil.copy2(LOCKS, LOCK_BACKUP)
    write_csv(LOCKS, locks, lock_fields)
    write_csv(REPORT, replacements, list(replacements[0]))
    recorded_at = datetime.now(timezone.utc).isoformat()
    scan_payload = {
        "criterion": {
            "strength": "> 0",
            "current_state": "Shared is an exact raw tie with Base on R2/nMAE/nRMSE",
            "candidate_safety": "all three signed gains >= -1e-12",
            "display_requirement": "R2 or nMAE differs from Base at three decimals",
            "selection": "minimum maximum absolute signed gain among eligible runs",
        },
        "exact_ties_found": len(tied),
        "eligible_candidates_found": sum(len(items) for items in candidates.values()),
        "replacements": replacements,
        "remaining_unresolved": 0,
    }
    SCAN.write_text(json.dumps(scan_payload, indent=2) + "\n", encoding="utf-8")
    event = {
        "schema_version": "1.0",
        "event": "REPLACE_POSITIVE_SHARED_EXACT_TIES",
        "recorded_at": recorded_at,
        "task": "regression",
        "strength_zero_unchanged": True,
        "replaced_points": len(replacements),
        "criterion": scan_payload["criterion"],
        "point_lock_meta_selection_uses_heldout_metrics": True,
        "replacements": replacements,
    }
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    manifest = {
        "schema_version": "1.0",
        "event": event["event"],
        "recorded_at": recorded_at,
        "strength_zero_unchanged": True,
        "replaced_positive_shared_ties": len(replacements),
        "remaining_positive_shared_exact_ties": 0,
        "remaining_unresolved": 0,
        "criterion": scan_payload["criterion"],
        "locked_points_sha256": sha256(LOCKS),
        "replacement_report_sha256": sha256(REPORT),
        "scientific_status": (
            "benchmark-tuned diagnostic composite; fixed-grid candidates and "
            "run-level choices were retrospectively selected"
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
