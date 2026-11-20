"""Run a resumable v8-only OpenML expansion benchmark.

The wrapper runs one dataset at a time so interrupted long runs keep the
completed per-dataset artifacts. Each dataset still includes the TabICL
baseline because deltas are measured against it.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


BROAD48_DATASETS = [
    "openml_3_kr_vs_kp",
    "openml_784_newton_hema",
    "openml_1066_kc1_binary",
    "openml_902_sleuth_case2002",
    "openml_1006_lymph",
    "openml_1026_grub_damage",
    "openml_41976_tuning_svms",
    "openml_854_fishcatch",
    "openml_745_auto_price",
    "openml_40669_corral",
    "openml_1054_mc2",
    "openml_46875_divorce_prediction",
    "openml_973_wine",
    "openml_1490_planning_relax",
    "openml_788_triazines",
    "openml_941_lowbwt",
    "openml_753_wisconsin",
    "openml_1488_parkinsons",
    "openml_446_prnn_crabs",
    "openml_721_pwlinear",
    "openml_40_sonar",
    "openml_481_biomed",
    "openml_895_chscase_geyser1",
    "openml_999_audiology",
    "openml_851_tecator",
    "openml_46279_algerian_forest_fires",
    "openml_1495_qualitative_bankruptcy",
    "openml_336_spect",
    "openml_1600_spectf",
    "openml_53_heart_statlog",
    "openml_13_breast_cancer",
    "openml_51_heart_h",
    "openml_49_heart_c",
    "openml_43_haberman",
    "openml_59_ionosphere",
    "openml_27_colic",
    "openml_56_vote",
    "openml_1498_sa_heart",
    "openml_31_credit_g",
    "openml_37_diabetes",
    "openml_44_spambase",
    "openml_50_tic_tac_toe",
    "openml_1464_blood_transfusion",
    "openml_1480_ilpd",
    "openml_1067_kc1",
    "openml_1068_pc1",
    "openml_1050_pc3",
    "openml_1049_pc4",
]

LARGE13_DATASETS = [
    "openml_179_adult",
    "openml_151_electricity",
    "openml_46910_bank_marketing",
    "openml_44_spambase",
    "openml_3_kr_vs_kp",
    "openml_727_2dplanes",
    "openml_881_mv",
    "openml_1067_kc1",
    "openml_1068_pc1",
    "openml_1050_pc3",
    "openml_1049_pc4",
    "openml_31_credit_g",
    "openml_1464_blood_transfusion",
]

SUITES = {
    "broad48": BROAD48_DATASETS,
    "large13": LARGE13_DATASETS,
    "broad48_plus_large13": BROAD48_DATASETS + [d for d in LARGE13_DATASETS if d not in BROAD48_DATASETS],
}

METHODS = ["tabicl_baseline", "apa_cf_directional_router"]


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
                "mean_ece": mean(group, "ece"),
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
            float(row["mean_balanced_accuracy_delta"]),
            float(row["mean_accuracy_delta"]),
            -int(row["negative_bacc_groups"]),
        ),
        reverse=True,
    )


def summarize_by_dataset(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        base = [row for row in rows if row["dataset"] == dataset and row["method"] == "tabicl_baseline"]
        v8 = [row for row in rows if row["dataset"] == dataset and row["method"] == "apa_cf_directional_router"]
        if not v8:
            continue
        bacc_delta = [float(row["balanced_accuracy_delta"]) for row in v8]
        out_rows.append(
            {
                "dataset": dataset,
                "n": len(v8),
                "baseline_accuracy": mean(base, "accuracy"),
                "v8_accuracy": mean(v8, "accuracy"),
                "mean_accuracy_delta": mean(v8, "accuracy_delta"),
                "baseline_balanced_accuracy": mean(base, "balanced_accuracy"),
                "v8_balanced_accuracy": mean(v8, "balanced_accuracy"),
                "mean_balanced_accuracy_delta": mean(v8, "balanced_accuracy_delta"),
                "mean_macro_f1_delta": mean(v8, "macro_f1_delta"),
                "mean_minority_recall_delta": mean(v8, "minority_recall_delta"),
                "mean_ece_delta": mean(v8, "ece_delta"),
                "positive_bacc_groups": sum(v > 1e-12 for v in bacc_delta),
                "negative_bacc_groups": sum(v < -1e-12 for v in bacc_delta),
                "max_bacc_delta": max(bacc_delta) if bacc_delta else "",
                "min_bacc_delta": min(bacc_delta) if bacc_delta else "",
            }
        )
    return sorted(
        out_rows,
        key=lambda row: (
            float(row["mean_balanced_accuracy_delta"]),
            float(row["mean_accuracy_delta"]),
        ),
        reverse=True,
    )


def write_tables(prefix: str, result_dir: Path, failures: list[dict[str, object]]) -> None:
    rows = collect(prefix, result_dir)
    if not rows:
        return
    method_rows = summarize_by_method(rows)
    dataset_rows = summarize_by_dataset(rows)
    lines = [
        f"# {prefix}",
        "",
        "## 方法总览",
        "",
        "| 方法 | 组数 | Accuracy 变化 | Balanced Accuracy 变化 | Macro-F1 变化 | Minority Recall 变化 | ECE 变化 | BAcc 正/负组 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {row['n']} | "
            f"{fmt(row['mean_accuracy_delta'], signed=True)} | "
            f"{fmt(row['mean_balanced_accuracy_delta'], signed=True)} | "
            f"{fmt(row['mean_macro_f1_delta'], signed=True)} | "
            f"{fmt(row['mean_minority_recall_delta'], signed=True)} | "
            f"{fmt(row['mean_ece_delta'], signed=True)} | "
            f"{row['positive_bacc_groups']} / {row['negative_bacc_groups']} |"
        )
    lines.extend(
        [
            "",
            "## 数据集明细",
            "",
            "| 数据集 | 组数 | Baseline Acc | v8 Acc | Acc 变化 | Baseline BAcc | v8 BAcc | BAcc 变化 | BAcc 正/负组 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['n']} | "
            f"{fmt(row['baseline_accuracy'])} | {fmt(row['v8_accuracy'])} | "
            f"{fmt(row['mean_accuracy_delta'], signed=True)} | "
            f"{fmt(row['baseline_balanced_accuracy'])} | {fmt(row['v8_balanced_accuracy'])} | "
            f"{fmt(row['mean_balanced_accuracy_delta'], signed=True)} | "
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
    write_csv(result_dir / f"{prefix}_method_ranking.csv", summarize_by_method(rows))
    write_csv(result_dir / f"{prefix}_dataset_summary.csv", summarize_by_dataset(rows))
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
    parser.add_argument("--suite", choices=sorted(SUITES), default="broad48")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--output-prefix", default="apa_v8_openml_broad48")
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
    if not datasets:
        datasets = SUITES[args.suite]

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
