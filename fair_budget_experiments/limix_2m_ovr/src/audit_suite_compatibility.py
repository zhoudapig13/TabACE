#!/usr/bin/env python3
"""Inventory CC18/CTR23 tasks against native, OvR, and SVD dispatch rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import openml


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-threshold", type=int, default=1024)
    return parser.parse_args()


def number(qualities: dict[str, Any], name: str) -> int | None:
    value = qualities.get(name)
    return None if value is None else int(float(value))


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    openml.config.set_root_cache_directory(output / "openml_cache")
    rows: list[dict[str, Any]] = []
    for suite_id, task_name in ((99, "classification"), (353, "regression")):
        suite = openml.study.get_suite(suite_id)
        for position, task_id in enumerate(suite.tasks, start=1):
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = openml.datasets.get_dataset(
                task.dataset_id,
                download_data=False,
                download_qualities=True,
                download_features_meta_data=False,
            )
            features = number(dataset.qualities, "NumberOfFeatures")
            classes = (
                number(dataset.qualities, "NumberOfClasses")
                if task_name == "classification"
                else None
            )
            rows.append(
                {
                    "suite_id": suite_id,
                    "task": task_name,
                    "task_id": int(task_id),
                    "data_id": int(task.dataset_id),
                    "dataset": str(dataset.name),
                    "instances": number(dataset.qualities, "NumberOfInstances"),
                    "features_including_target": features,
                    "classes": classes,
                    "classification_adapter": (
                        "one_vs_rest" if classes is not None and classes > 10 else "native"
                    ),
                    "feature_adapter": (
                        "truncated_svd_source_fit"
                        if features is not None and max(0, features - 1) > args.feature_threshold
                        else "identity"
                    ),
                }
            )
            print(
                f"[{task_name} {position}/{len(suite.tasks)}] {dataset.name}",
                flush=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "suite_compatibility.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    classification = [row for row in rows if row["task"] == "classification"]
    regression = [row for row in rows if row["task"] == "regression"]
    report = {
        "cc18_tasks": len(classification),
        "ctr23_tasks": len(regression),
        "native_classification_tasks": sum(
            row["classification_adapter"] == "native" for row in classification
        ),
        "ovr_classification_tasks": sum(
            row["classification_adapter"] == "one_vs_rest" for row in classification
        ),
        "ovr_datasets": [
            {
                "data_id": row["data_id"],
                "dataset": row["dataset"],
                "classes": row["classes"],
            }
            for row in classification
            if row["classification_adapter"] == "one_vs_rest"
        ],
        "svd_tasks": [
            {
                "task": row["task"],
                "data_id": row["data_id"],
                "dataset": row["dataset"],
                "predictor_features": max(0, int(row["features_including_target"]) - 1),
            }
            for row in rows
            if row["feature_adapter"] == "truncated_svd_source_fit"
        ],
        "complete_coverage": len(classification) == 72 and len(regression) == 35,
    }
    (output / "suite_compatibility.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
