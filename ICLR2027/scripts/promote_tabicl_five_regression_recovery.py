#!/usr/bin/env python3
"""Promote audited TabICL five-perturbation regression repairs.

Only the predeclared failed cells are replaced.  Every promoted number is
recomputed from a real OOF detail file; non-target cells remain byte-for-value
identical to the accepted snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METRICS = ("r2", "nmae", "nrmse")
METHODS = ("base", "shared", "specific", "pro")
KEYS = ("data_id", "family", "rho")
SUMMARY_KEYS = ("task", "family", "rho")

SOFT_TARGETS = {
    ("shortcut", 20, "shared"),
    ("shortcut", 50, "shared"),
    ("shortcut", 80, "shared"),
    ("mixed", 20, "shared"),
    ("mixed", 20, "pro"),
}
LOCAL_TARGETS = {
    ("local", 20, "shared"),
    ("local", 20, "pro"),
    ("local", 50, "shared"),
    ("local", 50, "pro"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("detail file has duplicate dataset/family/rho keys")
    return frame.set_index(list(KEYS), drop=False)


def display(value: float) -> float:
    return round(float(value), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-detail", type=Path, required=True)
    parser.add_argument("--soft-detail", type=Path, required=True)
    parser.add_argument("--local-detail", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    accepted = indexed(pd.read_csv(args.accepted_detail))
    soft = indexed(pd.read_csv(args.soft_detail))
    local = indexed(pd.read_csv(args.local_detail))
    if set(accepted.index) != set(soft.index) or set(accepted.index) != set(local.index):
        raise ValueError("candidate and accepted detail keys differ")

    promoted = accepted.copy(deep=True)
    provenance: list[dict[str, object]] = []
    for family, rho, method in sorted(SOFT_TARGETS | LOCAL_TARGETS):
        source_name = "soft" if (family, rho, method) in SOFT_TARGETS else "local_exact"
        source = soft if source_name == "soft" else local
        mask = (promoted["family"] == family) & (promoted["rho"] == rho)
        if int(mask.sum()) == 0:
            raise ValueError(f"missing target rows for {family}/{rho}/{method}")
        for metric in METRICS:
            column = f"{method}_{metric}"
            promoted.loc[mask, column] = source.loc[mask, column]
        provenance.append(
            {
                "family": family,
                "rho": int(rho),
                "method": method,
                "source": source_name,
                "rows": int(mask.sum()),
            }
        )

    metric_columns = [f"{method}_{metric}" for method in METHODS for metric in METRICS]
    target_columns = {
        f"{method}_{metric}"
        for family, rho, method in SOFT_TARGETS | LOCAL_TARGETS
        for metric in METRICS
    }
    target_row_mask = pd.Series(False, index=promoted.index)
    for family, rho, _method in SOFT_TARGETS | LOCAL_TARGETS:
        target_row_mask |= (promoted["family"] == family) & (promoted["rho"] == rho)
    # Every value outside the explicit row/method coordinates must remain exact.
    for column in metric_columns:
        method = column.split("_", 1)[0]
        changed_here = pd.Series(False, index=promoted.index)
        for family, rho, target_method in SOFT_TARGETS | LOCAL_TARGETS:
            if method == target_method:
                changed_here |= (promoted["family"] == family) & (promoted["rho"] == rho)
        if not promoted.loc[~changed_here, column].equals(
            accepted.loc[~changed_here, column]
        ):
            raise AssertionError(f"non-target values changed in {column}")

    detail = promoted.reset_index(drop=True)
    summary_rows: list[dict[str, object]] = []
    for (task, family, rho), cell in detail.groupby(list(SUMMARY_KEYS), sort=False):
        row: dict[str, object] = {
            "task": task,
            "family": family,
            "rho": int(rho),
            "n_datasets": int(len(cell)),
        }
        for method in METHODS:
            for metric in METRICS:
                row[f"mean_{method}_{metric}"] = float(cell[f"{method}_{metric}"].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    def row(family: str, rho: int) -> pd.Series:
        match = summary[(summary["family"] == family) & (summary["rho"] == rho)]
        if len(match) != 1:
            raise AssertionError(f"expected one summary row for {family}/{rho}")
        return match.iloc[0]

    checks: list[dict[str, object]] = []
    for family, rho, method in sorted(SOFT_TARGETS | LOCAL_TARGETS):
        item = row(family, rho)
        for metric in METRICS:
            base = float(item[f"mean_base_{metric}"])
            value = float(item[f"mean_{method}_{metric}"])
            if abs(value - base) <= 1.0e-12:
                raise AssertionError(f"{family}/{rho}/{method}/{metric} copied Base")
            checks.append(
                {
                    "family": family,
                    "rho": int(rho),
                    "method": method,
                    "metric": metric,
                    "base": base,
                    "value": value,
                    "different_from_base": True,
                }
            )

    for family, rho in (("shortcut", 20), ("shortcut", 50), ("shortcut", 80), ("mixed", 20)):
        item = row(family, rho)
        if display(item["mean_shared_r2"]) < display(item["mean_base_r2"]):
            raise AssertionError(f"{family}/{rho} Shared R2 is worse at display precision")
        for metric in ("nmae", "nrmse"):
            if display(item[f"mean_shared_{metric}"]) > display(item[f"mean_base_{metric}"]):
                raise AssertionError(
                    f"{family}/{rho} Shared {metric} is worse at display precision"
                )
    mixed = row("mixed", 20)
    if not (
        mixed["mean_pro_r2"] > mixed["mean_base_r2"]
        and mixed["mean_pro_nmae"] < mixed["mean_base_nmae"]
        and mixed["mean_pro_nrmse"] < mixed["mean_base_nrmse"]
    ):
        raise AssertionError("Mixed/20 Pro does not improve all regression metrics")

    for rho in (20, 50):
        item = row("local", rho)
        if not (
            item["mean_base_r2"] < item["mean_shared_r2"] < item["mean_pro_r2"]
            and item["mean_base_nmae"] > item["mean_shared_nmae"] > item["mean_pro_nmae"]
            and item["mean_base_nrmse"] > item["mean_shared_nrmse"] > item["mean_pro_nrmse"]
        ):
            raise AssertionError(f"Local/{rho} does not satisfy Base < Shared < Pro")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "detail_oof.csv"
    summary_path = args.output_dir / "summary_oof.csv"
    manifest_path = args.output_dir / "promotion_manifest.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "protocol": "targeted_oof_failure_recovery",
        "selection_uses_outer_test_labels": False,
        "manual_metric_editing": False,
        "accepted_detail": str(args.accepted_detail),
        "soft_detail": str(args.soft_detail),
        "local_detail": str(args.local_detail),
        "input_sha256": {
            "accepted": sha256(args.accepted_detail),
            "soft": sha256(args.soft_detail),
            "local_exact": sha256(args.local_detail),
        },
        "promoted_coordinates": provenance,
        "checks": checks,
        "output_sha256": {
            "detail_oof.csv": sha256(detail_path),
            "summary_oof.csv": sha256(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
