#!/usr/bin/env python3
"""Fail closed unless a LimiX five-perturbation run is paper-complete."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_DATASETS = {"classification": 72, "regression": 35}
EXPECTED_CONDITIONS = 21
EXPECTED_SUMMARY_ROWS = {"classification": 50, "regression": 75}
EXPECTED_PIPELINES = {"classification": 4, "regression": 8}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = arguments()
    root = args.run.resolve()
    detail = read_csv(root / "detail.csv")
    summary = read_csv(root / "summary_long.csv")
    run_audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}

    by_task_dataset: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in detail:
        by_task_dataset[(row["task"], int(row["data_id"]))].append(row)

    dataset_counts = Counter(task for task, _data_id in by_task_dataset)
    checks["dataset_counts"] = dict(dataset_counts)
    checks["dataset_coverage"] = all(
        dataset_counts[task] == expected
        for task, expected in EXPECTED_DATASETS.items()
    )
    checks["condition_coverage"] = all(
        len(rows) == EXPECTED_CONDITIONS
        and len({(row["family"], int(row["rho"])) for row in rows})
        == EXPECTED_CONDITIONS
        for rows in by_task_dataset.values()
    )
    checks["fixed_partitions"] = all(
        len({row["source_indices_sha256"] for row in rows}) == 1
        and len({row["target_indices_sha256"] for row in rows}) == 1
        and len({row["target_features_sha256"] for row in rows}) == 1
        and len({row["evaluation_target_sha256"] for row in rows}) == 1
        for rows in by_task_dataset.values()
    )
    checks["classification_dispatch"] = all(
        (row["strategy"] == "native" and int(row["model_calls"]) == 1)
        if int(row["n_classes"]) <= 10
        else (
            row["strategy"] == "one_vs_rest"
            and int(row["model_calls"]) == int(row["n_classes"])
        )
        for row in detail
        if row["task"] == "classification"
    )
    ovr_datasets = {
        int(row["data_id"])
        for row in detail
        if row["task"] == "classification" and int(row["n_classes"]) > 10
    }
    checks["ovr_dataset_count"] = len(ovr_datasets)
    checks["five_ovr_datasets"] = len(ovr_datasets) == 5
    checks["regression_native"] = all(
        row["strategy"] == "native" and int(row["model_calls"]) == 1
        for row in detail
        if row["task"] == "regression"
    )
    threshold = int(run_audit["high_dimensional_adapter"]["threshold"])
    checks["feature_adapter_dispatch"] = all(
        (
            row.get("feature_adapter") == "truncated_svd_source_fit"
            and int(row["model_n_features"])
            <= int(run_audit["high_dimensional_adapter"]["maximum_components"])
        )
        if threshold > 0 and int(row["n_features"]) > threshold
        else row.get("feature_adapter") == "identity"
        for row in detail
    )
    checks["affected_row_counts"] = all(
        int(row["affected_rows"])
        == (
            0
            if row["family"] == "original"
            else min(
                int(row["n_source"]),
                max(1, math.ceil(int(row["rho"]) / 100 * int(row["n_source"]))),
            )
        )
        for row in detail
    )
    summary_counts = Counter(row["task"] for row in summary)
    checks["summary_row_counts"] = dict(summary_counts)
    checks["summary_coverage"] = all(
        summary_counts[task] == expected
        for task, expected in EXPECTED_SUMMARY_ROWS.items()
    ) and all(
        int(row["dataset_n"]) == EXPECTED_DATASETS[row["task"]]
        for row in summary
    )
    checks["official_ensemble"] = all(
        int(run_audit[f"{task}_pipelines"]) == expected
        for task, expected in EXPECTED_PIPELINES.items()
    )
    checks["no_failures"] = int(run_audit["failures"]) == 0
    checks["target_label_leakage_declared_false"] = (
        run_audit["controlled_perturbations"]["target_labels_used_for_construction"]
        is False
        and run_audit["high_dimensional_adapter"]["uses_target_labels"] is False
    )
    passed = all(
        value is True
        for key, value in checks.items()
        if key
        not in {
            "dataset_counts",
            "ovr_dataset_count",
            "summary_row_counts",
        }
    )
    output = {
        "paper_complete": passed,
        "checks": checks,
        "detail_rows": len(detail),
        "expected_detail_rows": 72 * 21 + 35 * 21,
        "run": str(root),
    }
    destination = root / "completion_audit.json"
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
