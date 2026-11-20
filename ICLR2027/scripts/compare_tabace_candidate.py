#!/usr/bin/env python3
"""Compare a candidate TabACE summary against Base and an accepted snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae"),
}
METHODS = {"shared": "shared", "specific": "pro"}
HIGHER_IS_BETTER = {"accuracy", "auc", "r2"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def signed_gain(row: dict[str, str], method: str, metric: str) -> float:
    base = float(row[f"mean_base_{metric}"])
    value = float(row[f"mean_{method}_{metric}"])
    return value - base if metric in HIGHER_IS_BETTER else base - value


def displayed_signed_gain(
    row: dict[str, str], method: str, metric: str, decimals: int = 3
) -> float:
    base = round(float(row[f"mean_base_{metric}"]), decimals)
    value = round(float(row[f"mean_{method}_{metric}"]), decimals)
    return value - base if metric in HIGHER_IS_BETTER else base - value


def summarize(rows: list[dict[str, str]], task: str) -> dict[str, object]:
    strength_key = "rho" if "rho" in rows[0] else "strength"
    result: dict[str, object] = {}
    for paper_method, source_method in METHODS.items():
        gains = {
            metric: [signed_gain(row, source_method, metric) for row in rows]
            for metric in METRICS[task]
        }
        positive = [
            index
            for index, row in enumerate(rows)
            if float(row[strength_key]) != 0.0
        ]
        result[paper_method] = {
            "failed_metric_cells": sum(
                gain < 0.0 for values in gains.values() for gain in values
            ),
            "failed_positive_metric_cells": sum(
                values[index] < 0.0
                for values in gains.values()
                for index in positive
            ),
            "failed_display_metric_cells": sum(
                displayed_signed_gain(row, source_method, metric) < 0.0
                for row in rows
                for metric in METRICS[task]
            ),
            "failed_positive_display_metric_cells": sum(
                displayed_signed_gain(rows[index], source_method, metric) < 0.0
                for metric in METRICS[task]
                for index in positive
            ),
            "mean_signed_gain": {
                metric: sum(values) / len(values)
                for metric, values in gains.items()
            },
            "positive_mean_signed_gain": {
                metric: (
                    sum(values[index] for index in positive) / len(positive)
                    if positive else None
                )
                for metric, values in gains.items()
            },
            "nonbase_metric_cells": sum(
                gain != 0.0 for values in gains.values() for gain in values
            ),
        }
        if task == "regression" and f"mean_{source_method}_nrmse" in rows[0]:
            nrmse_gains = [
                float(row["mean_base_nrmse"])
                - float(row[f"mean_{source_method}_nrmse"])
                for row in rows
            ]
            result[paper_method]["main_table_nrmse_gain"] = {
                "clean": (
                    sum(nrmse_gains[index] for index in range(len(rows)) if index not in positive)
                    / (len(rows) - len(positive))
                ),
                "positive": sum(nrmse_gains[index] for index in positive) / len(positive),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(METRICS))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--accepted", type=Path)
    args = parser.parse_args()
    payload = {"candidate": summarize(read_rows(args.candidate), args.task)}
    if args.accepted:
        payload["accepted"] = summarize(read_rows(args.accepted), args.task)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
