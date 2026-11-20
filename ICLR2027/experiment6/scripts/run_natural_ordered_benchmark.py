"""Run resumable ordered-split benchmarks on original public datasets.

Unlike the label-shift stress tests, this wrapper does not alter class ratios.
It preserves OpenML row order inside each sampled window: train comes before
test. This probes public datasets whose original ordering may contain natural
prior or concept drift.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


NATURAL_DATASETS = [
    "openml_151_electricity",
    "openml_179_adult",
    "openml_46910_bank_marketing",
    "openml_44_spambase",
    "openml_37_diabetes",
    "openml_31_credit_g",
    "openml_1067_kc1",
    "openml_1068_pc1",
    "openml_1050_pc3",
    "openml_1049_pc4",
    "openml_1464_blood_transfusion",
    "openml_3_kr_vs_kp",
    "openml_727_2dplanes",
    "openml_881_mv",
    "openml_56_vote",
]

METHODS = [
    "tabicl_baseline",
    "apa_cf_directional_router",
    "apa_cf_natural_guard_router",
    "apa_cf_local_shift_router",
    "apa_cf_local_shift_val_router",
    "apa_context_policy_val_selector",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(value: object, signed: bool = False, digits: int = 4) -> str:
    if value in {"", None}:
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def collect(prefix: str, result_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for detail_path in sorted(result_dir.glob(f"{prefix}_by_dataset_*_detail.csv")):
        rows.extend(read_csv(detail_path))
    return rows


def summarize_by_method(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        bacc_delta = [float(row["balanced_accuracy_delta"]) for row in group]
        out_rows.append(
            {
                "method": method,
                "n": len(group),
                "mean_accuracy": mean(group, "accuracy"),
                "mean_balanced_accuracy": mean(group, "balanced_accuracy"),
                "mean_macro_f1": mean(group, "macro_f1"),
                "mean_accuracy_delta": mean(group, "accuracy_delta"),
                "mean_balanced_accuracy_delta": mean(group, "balanced_accuracy_delta"),
                "mean_macro_f1_delta": mean(group, "macro_f1_delta"),
                "mean_minority_recall_delta": mean(group, "minority_recall_delta"),
                "mean_ece_delta": mean(group, "ece_delta"),
                "positive_bacc_groups": sum(v > 1e-12 for v in bacc_delta),
                "negative_bacc_groups": sum(v < -1e-12 for v in bacc_delta),
                "max_bacc_delta": max(bacc_delta) if bacc_delta else "",
                "min_bacc_delta": min(bacc_delta) if bacc_delta else "",
            }
        )
    return sorted(
        out_rows,
        key=lambda row: (
            float(row["mean_accuracy_delta"]),
            float(row["mean_balanced_accuracy_delta"]),
            -int(row["negative_bacc_groups"]),
        ),
        reverse=True,
    )


def summarize_by_dataset_method(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for dataset, method in sorted({(row["dataset"], row["method"]) for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
        bacc_delta = [float(row["balanced_accuracy_delta"]) for row in group]
        out_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(group),
                "mean_accuracy": mean(group, "accuracy"),
                "mean_balanced_accuracy": mean(group, "balanced_accuracy"),
                "mean_macro_f1": mean(group, "macro_f1"),
                "mean_accuracy_delta": mean(group, "accuracy_delta"),
                "mean_balanced_accuracy_delta": mean(group, "balanced_accuracy_delta"),
                "mean_macro_f1_delta": mean(group, "macro_f1_delta"),
                "mean_ece_delta": mean(group, "ece_delta"),
                "positive_bacc_groups": sum(v > 1e-12 for v in bacc_delta),
                "negative_bacc_groups": sum(v < -1e-12 for v in bacc_delta),
                "min_bacc_delta": min(bacc_delta) if bacc_delta else "",
                "max_bacc_delta": max(bacc_delta) if bacc_delta else "",
            }
        )
    return out_rows


def write_tables(prefix: str, result_dir: Path, failures: list[dict[str, object]]) -> None:
    rows = collect(prefix, result_dir)
    if not rows:
        return
    method_rows = summarize_by_method(rows)
    dataset_rows = summarize_by_dataset_method(rows)
    lines = [
        f"# {prefix}",
        "",
        "Ordered split on original public data. No artificial label-ratio intervention is applied.",
        "",
        "## 方法汇总",
        "",
        "| 方法 | 组数 | Acc | BAcc | Macro-F1 | Acc变化 | BAcc变化 | Macro-F1变化 | ECE变化 | BAcc正/负组 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {row['n']} | "
            f"{fmt(row['mean_accuracy'])} | {fmt(row['mean_balanced_accuracy'])} | {fmt(row['mean_macro_f1'])} | "
            f"{fmt(row['mean_accuracy_delta'], signed=True)} | "
            f"{fmt(row['mean_balanced_accuracy_delta'], signed=True)} | "
            f"{fmt(row['mean_macro_f1_delta'], signed=True)} | "
            f"{fmt(row['mean_ece_delta'], signed=True)} | "
            f"{row['positive_bacc_groups']} / {row['negative_bacc_groups']} |"
        )
    lines.extend(
        [
            "",
            "## 数据集 x 方法",
            "",
            "| 数据集 | 方法 | 组数 | Acc变化 | BAcc变化 | Macro-F1变化 | ECE变化 | BAcc正/负组 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_rows:
        if row["method"] == "tabicl_baseline":
            continue
        lines.append(
            f"| `{row['dataset']}` | `{row['method']}` | {row['n']} | "
            f"{fmt(row['mean_accuracy_delta'], signed=True)} | "
            f"{fmt(row['mean_balanced_accuracy_delta'], signed=True)} | "
            f"{fmt(row['mean_macro_f1_delta'], signed=True)} | "
            f"{fmt(row['mean_ece_delta'], signed=True)} | "
            f"{row['positive_bacc_groups']} / {row['negative_bacc_groups']} |"
        )
    if failures:
        lines.extend(["", "## 失败数据集", ""])
        for failure in failures:
            lines.append(f"- `{failure['dataset']}`: {failure['error_type']} - {failure['message']}")
    (result_dir / f"{prefix}_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(prefix: str, result_dir: Path, failures: list[dict[str, object]]) -> None:
    rows = collect(prefix, result_dir)
    if not rows:
        return
    write_csv(result_dir / f"{prefix}_detail.csv", rows)
    write_csv(result_dir / f"{prefix}_method_summary.csv", summarize_by_method(rows))
    write_csv(result_dir / f"{prefix}_dataset_method.csv", summarize_by_dataset_method(rows))
    write_tables(prefix, result_dir, failures)


def run_dataset(dataset: str, args: argparse.Namespace, result_dir: Path) -> dict[str, object] | None:
    dataset_prefix = f"{args.output_prefix}_by_dataset_{dataset}"
    detail_path = result_dir / f"{dataset_prefix}_detail.csv"
    if detail_path.exists() and not args.rerun:
        print(f"[skip] {dataset}: {detail_path.name} already exists", flush=True)
        return None

    cmd = [
        sys.executable,
        "-u",
        "experiment6/scripts/run_apa_tabicl.py",
        "--datasets",
        dataset,
        "--split-mode",
        "ordered",
        "--seeds",
        args.seeds,
        "--n-samples",
        str(args.n_samples),
        "--train-size",
        str(args.train_size),
        "--test-size",
        str(args.test_size),
        "--k",
        str(args.k),
        "--n-estimators",
        str(args.n_estimators),
        "--device",
        args.device,
        "--methods",
        args.methods,
        "--output-prefix",
        dataset_prefix,
    ]
    print(f"[run] {dataset}", flush=True)
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return {
            "dataset": dataset,
            "error_type": type(exc).__name__,
            "message": f"exit code {exc.returncode}",
            "elapsed_sec": time.time() - start,
        }
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(NATURAL_DATASETS))
    parser.add_argument("--output-prefix", default="apa_v17_natural_ordered_pilot")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path("experiment6/results")
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    failures: list[dict[str, object]] = []
    failure_path = result_dir / f"{args.output_prefix}_failures.csv"
    try:
        for dataset in datasets:
            failure = run_dataset(dataset, args, result_dir)
            if failure is not None:
                failures.append(failure)
                write_csv(failure_path, failures)
            aggregate(args.output_prefix, result_dir, failures)
    finally:
        aggregate(args.output_prefix, result_dir, failures)
        if failures:
            write_csv(failure_path, failures)


if __name__ == "__main__":
    main()
