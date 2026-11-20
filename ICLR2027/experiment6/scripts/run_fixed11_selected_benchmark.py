"""Run the fixed 11-dataset selected-method benchmark in resumable chunks.

This wrapper intentionally runs one dataset at a time. The underlying
`run_apa_tabicl.py` script writes artifacts only after a run completes, so a
single monolithic run can lose all intermediate work when interrupted.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


DATASETS = [
    "openml_ilpd",
    "openml_credit_g",
    "openml_blood_transfusion",
    "openml_kc1",
    "openml_diabetes_pima",
    "openml_pc1",
    "breast_cancer",
    "synthetic_balanced",
    "synthetic_imbalanced",
    "synthetic_label_shift",
    "synthetic_label_shift_hard",
]

METHODS = [
    "tabicl_baseline",
    "apa_full_prior_aware_tiny",
    "apa_dual_shift_router_v7",
    "apa_cf_directional_router",
    "apa_cf_strict_margin_router",
    "apa_cf_batch_margin_router",
    "apa_cf_tempered_rescue_router",
    "apa_cf_hybrid_tempered_router",
    "apa_cf_hybrid_calibrated_router",
    "apa_cf_hybrid_stability_router",
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


def summarize_by_method(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        bacc_delta = [float(row["balanced_accuracy_delta"]) for row in group]
        out: dict[str, object] = {
            "method": method,
            "n": len(group),
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
        out_rows.append(out)
    return sorted(
        out_rows,
        key=lambda row: (
            float(row["mean_balanced_accuracy_delta"]),
            float(row["mean_accuracy_delta"]),
            -int(row["negative_bacc_groups"]),
        ),
        reverse=True,
    )


def summarize_by_dataset_method(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    keys = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset, method in keys:
        group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
        bacc_delta = [float(row["balanced_accuracy_delta"]) for row in group]
        out_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(group),
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
    return out_rows


def collect(prefix: str, result_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for detail_path in sorted(result_dir.glob(f"{prefix}_by_dataset_*_detail.csv")):
        rows.extend(read_csv(detail_path))
    return rows


def aggregate(prefix: str, result_dir: Path) -> None:
    rows = collect(prefix, result_dir)
    if not rows:
        return
    write_csv(result_dir / f"{prefix}_detail.csv", rows)
    write_csv(result_dir / f"{prefix}_method_ranking.csv", summarize_by_method(rows))
    write_csv(result_dir / f"{prefix}_dataset_method.csv", summarize_by_dataset_method(rows))


def run_dataset(dataset: str, args: argparse.Namespace, result_dir: Path) -> None:
    dataset_prefix = f"{args.output_prefix}_by_dataset_{dataset}"
    detail_path = result_dir / f"{dataset_prefix}_detail.csv"
    if detail_path.exists() and not args.rerun:
        print(f"[skip] {dataset}: {detail_path.name} already exists", flush=True)
        return

    cmd = [
        sys.executable,
        "-u",
        "experiment6/scripts/run_apa_tabicl.py",
        "--datasets",
        dataset,
        "--force-label-shift",
        "--label-shift-strengths",
        args.label_shift_strengths,
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
        ",".join(METHODS),
        "--output-prefix",
        dataset_prefix,
    ]
    print(f"[run] {dataset}", flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--output-prefix", default="apa_v1_v16_fixed11_selected_benchmark")
    parser.add_argument("--label-shift-strengths", default="0.0,0.6,0.9")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--n-samples", type=int, default=180)
    parser.add_argument("--train-size", type=int, default=90)
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path("experiment6/results")
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    try:
        for dataset in datasets:
            run_dataset(dataset, args, result_dir)
            aggregate(args.output_prefix, result_dir)
    finally:
        aggregate(args.output_prefix, result_dir)


if __name__ == "__main__":
    main()
