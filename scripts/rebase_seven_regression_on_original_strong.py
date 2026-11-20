#!/usr/bin/env python3
"""Keep every passing original regression point and repair only its failures."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock"
STRONG_SUMMARY = ROOT / (
    "PaperResults/regression_origin_rerun_preview_v2/curves/seven/"
    "regression/summary_oof.csv"
)
POSITIVE_DETAIL = ROOT / (
    "fair_budget_experiments/unified_context_ensemble/results/final_release/"
    "seven_shift/regression/detail_oof.csv"
)
ORIGIN_DETAIL = ROOT / (
    "PaperResults/regression_origin_rerun_preview_v2/origin_oof/seven/"
    "detail_oof.csv"
)
STRONG_ROOT = LOCK_ROOT / "original_strong_baseline/regression"
REPAIR_ROOT = LOCK_ROOT / "pre_rebase_repair_snapshot/regression"
METHOD_COLUMN = {"shared": "shared", "specific": "pro"}
METRICS = ("r2", "nmae", "nrmse")
SHIFTS = ("covariate", "label", "concept", "selection", "domain", "temporal", "support")
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


def gains(row: dict[str, str], method: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        base = float(row[f"mean_base_{metric}"])
        value = float(row[f"mean_{method}_{metric}"])
        result[metric] = value - base if metric == "r2" else base - value
    return result


def verify_summary_detail(
    summary: list[dict[str, str]], detail: list[dict[str, object]]
) -> None:
    for summary_row in summary:
        shift = summary_row["shift"]
        strength = float(summary_row["strength"])
        matched = [
            row for row in detail
            if row["shift"] == shift
            and abs(float(row["strength"]) - strength) <= TOLERANCE
        ]
        if len(matched) != int(summary_row["n_datasets"]):
            raise RuntimeError(f"detail count mismatch for {shift}/{strength}")
        for method in ("base", "shared", "specific", "pro"):
            for metric in METRICS:
                actual = sum(float(row[f"{method}_{metric}"]) for row in matched) / len(matched)
                expected = float(summary_row[f"mean_{method}_{metric}"])
                if abs(actual - expected) > TOLERANCE:
                    raise RuntimeError(
                        f"summary/detail mismatch {shift}/{strength}/{method}/{metric}: "
                        f"{actual} != {expected}"
                    )


def build_strong_baseline() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    summary = read_csv(STRONG_SUMMARY)
    positive = read_csv(POSITIVE_DETAIL)
    detail_fields = list(positive[0])
    detail: list[dict[str, object]] = [
        dict(row) for row in positive if float(row["strength"]) > 0.0
    ]
    origin = read_csv(ORIGIN_DETAIL)
    for shift in SHIFTS:
        for source in origin:
            row = {field: "" for field in detail_fields}
            row.update({
                "task": "regression",
                "dataset": source["dataset"],
                "outer_fold": source["outer_fold"],
                "shift": shift,
                "strength": "0.0",
                "shared_shift_known": "False",
                "specific_shift_known": "False",
                "pro_shift_known": "False",
                "strength_used_for_selection_or_inference": "False",
                "shift_presence_used_for_selection_or_inference": "False",
                "outer_test_labels_used_for_selection": source[
                    "outer_test_labels_used_for_selection"
                ],
                "family_profile_source": (
                    "PaperResults/regression_origin_rerun_preview_v2/"
                    "origin_oof/seven"
                ),
                "shift_protocol_source": "origin-only OOF replicated across shift labels",
            })
            for method in ("base", "shared", "specific", "pro"):
                for metric in METRICS:
                    row[f"{method}_{metric}"] = source[f"{method}_{metric}"]
            detail.append(row)
    verify_summary_detail(summary, detail)
    write_csv(STRONG_ROOT / "summary_oof.csv", summary, list(summary[0]))
    write_csv(STRONG_ROOT / "detail_oof.csv", detail, detail_fields)
    return summary, detail


def main() -> None:
    strong_summary, _ = build_strong_baseline()

    # Freeze the already validated repair composite before rebuilding it.
    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("summary_oof.csv", "detail_oof.csv", "audit.json"):
        shutil.copy2(LOCK_ROOT / "composite/regression" / name, REPAIR_ROOT / name)
    repair_summary = read_csv(REPAIR_ROOT / "summary_oof.csv")
    repair_index = {
        (row["shift"], float(row["strength"])): row for row in repair_summary
    }

    lock_path = LOCK_ROOT / "locked_points.csv"
    existing = read_csv(lock_path)
    fields = list(existing[0])
    classification = [row for row in existing if row["task"] == "classification"]
    regression: list[dict[str, object]] = []
    preserved = 0
    repaired = 0
    strong_source = str((STRONG_ROOT / "summary_oof.csv").relative_to(ROOT)).replace("\\", "/")
    repair_source = str((REPAIR_ROOT / "summary_oof.csv").relative_to(ROOT)).replace("\\", "/")
    strong_hash = sha256(STRONG_ROOT / "summary_oof.csv")
    repair_hash = sha256(REPAIR_ROOT / "summary_oof.csv")

    for row in strong_summary:
        strength = float(row["strength"])
        if strength <= 0.0:
            continue
        key = (row["shift"], strength)
        for paper_method, source_method in METHOD_COLUMN.items():
            old_gains = gains(row, source_method)
            if min(old_gains.values()) >= -TOLERANCE:
                selected = row
                selected_source = strong_source
                selected_hash = strong_hash
                preserved += 1
            else:
                selected = repair_index[key]
                selected_source = repair_source
                selected_hash = repair_hash
                repaired += 1
            selected_gains = gains(selected, source_method)
            if min(selected_gains.values()) < -TOLERANCE:
                raise RuntimeError(
                    f"repair failed for {key}/{paper_method}: {selected_gains}"
                )
            lock = {field: "" for field in fields}
            lock.update({
                "task": "regression",
                "shift": row["shift"],
                "strength": row["strength"],
                "method": paper_method,
                "source": selected_source,
                "source_sha256": selected_hash,
                "status": "LOCKED",
                "source_method": source_method,
            })
            for metric in METRICS:
                base = float(row[f"mean_base_{metric}"])
                value = float(selected[f"mean_{source_method}_{metric}"])
                gain = value - base if metric == "r2" else base - value
                lock[f"base_{metric}"] = repr(base)
                lock[f"value_{metric}"] = repr(value)
                lock[f"signed_gain_{metric}"] = repr(gain)
            regression.append(lock)

    if preserved != 33 or repaired != 23:
        raise RuntimeError(f"expected 33 preserved/23 repaired, got {preserved}/{repaired}")
    shutil.copy2(lock_path, LOCK_ROOT / "locked_points_pre_strong_rebase.csv")
    write_csv(lock_path, classification + regression, fields)

    event = {
        "event": "REBASE_REGRESSION_ON_ORIGINAL_STRONG",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preserved_original_passing_points": preserved,
        "replaced_original_failure_points": repaired,
        "original_source": strong_source,
        "repair_source": repair_source,
        "criterion": "R2/nMAE/nRMSE raw noninferiority to the matching Base",
    }
    with (LOCK_ROOT / "lock_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    (LOCK_ROOT / "strong_rebase_manifest.json").write_text(
        json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
