from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]


def to_float(value):
    if value in {"", None}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baseline_map(rows: list[dict[str, str]]):
    out = {}
    for row in rows:
        if row.get("method") != "baseline_n1":
            continue
        key = (row.get("dataset"), int(row.get("random_state", 0)))
        out[key] = row
    return out


def summarize(rows: list[dict[str, str]]):
    baselines = baseline_map(rows)
    groups = defaultdict(list)
    individual = []
    for row in rows:
        method = row.get("method")
        if method in {"baseline_n1", "load_failed"}:
            continue
        key = (row.get("dataset"), int(row.get("random_state", 0)))
        base = baselines.get(key)
        if base is None:
            continue
        result = {
            "dataset": row.get("dataset", ""),
            "source": row.get("source", ""),
            "method": method,
            "description": row.get("description", ""),
            "random_state": int(row.get("random_state", 0)),
            "n_samples": row.get("n_samples", ""),
            "n_features_original": row.get("n_features_original", ""),
        }
        any_metric = False
        for metric in METRICS:
            value = to_float(row.get(metric))
            base_value = to_float(base.get(metric))
            if value is None or base_value is None:
                continue
            result[metric] = value
            result[f"baseline_{metric}"] = base_value
            result[f"{metric}_delta"] = value - base_value
            any_metric = True
        if not any_metric:
            continue
        groups[(result["method"], result["dataset"])].append(result)
        individual.append(result)

    summary = []
    for (method, dataset), group in groups.items():
        row = {
            "method": method,
            "dataset": dataset,
            "runs": len(group),
            "description": group[0]["description"],
            "source": group[0]["source"],
            "n_samples": group[0]["n_samples"],
            "n_features_original": group[0]["n_features_original"],
        }
        for metric in METRICS:
            deltas = [item[f"{metric}_delta"] for item in group if f"{metric}_delta" in item]
            values = [item[metric] for item in group if metric in item]
            bases = [item[f"baseline_{metric}"] for item in group if f"baseline_{metric}" in item]
            if not deltas:
                continue
            arr = np.asarray(deltas, dtype=float)
            row[f"{metric}_positive_runs"] = int(np.sum(arr > 0))
            row[f"{metric}_nonnegative_runs"] = int(np.sum(arr >= 0))
            row[f"{metric}_mean_delta"] = float(np.mean(arr))
            row[f"{metric}_min_delta"] = float(np.min(arr))
            row[f"{metric}_max_delta"] = float(np.max(arr))
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"baseline_{metric}_mean"] = float(np.mean(bases))
        summary.append(row)
    summary.sort(
        key=lambda row: (
            row.get("accuracy_positive_runs", 0),
            row.get("balanced_accuracy_positive_runs", 0),
            row.get("macro_f1_positive_runs", 0),
            row.get("accuracy_mean_delta", -999),
        ),
        reverse=True,
    )
    individual.sort(
        key=lambda row: max([row.get(f"{metric}_delta", -999) for metric in METRICS]),
        reverse=True,
    )
    return summary, individual


def summarize_methods(individual: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method = defaultdict(list)
    for row in individual:
        by_method[row["method"]].append(row)

    out = []
    for method, group in by_method.items():
        method_row = {
            "method": method,
            "description": group[0].get("description", ""),
            "datasets": len({row["dataset"] for row in group}),
            "runs": len(group),
        }
        for metric in METRICS:
            deltas = [row[f"{metric}_delta"] for row in group if f"{metric}_delta" in row]
            if not deltas:
                continue
            arr = np.asarray(deltas, dtype=float)
            method_row[f"{metric}_mean_delta"] = float(np.mean(arr))
            method_row[f"{metric}_median_delta"] = float(np.median(arr))
            method_row[f"{metric}_positive_runs"] = int(np.sum(arr > 0))
            method_row[f"{metric}_nonnegative_runs"] = int(np.sum(arr >= 0))
            method_row[f"{metric}_min_delta"] = float(np.min(arr))
            method_row[f"{metric}_max_delta"] = float(np.max(arr))

            dataset_means = []
            for dataset in sorted({row["dataset"] for row in group}):
                dataset_deltas = [
                    row[f"{metric}_delta"]
                    for row in group
                    if row["dataset"] == dataset and f"{metric}_delta" in row
                ]
                if dataset_deltas:
                    dataset_means.append(float(np.mean(dataset_deltas)))
            if dataset_means:
                ds_arr = np.asarray(dataset_means, dtype=float)
                method_row[f"{metric}_datasets"] = int(len(ds_arr))
                method_row[f"{metric}_positive_datasets"] = int(np.sum(ds_arr > 0))
                method_row[f"{metric}_nonnegative_datasets"] = int(np.sum(ds_arr >= 0))
        out.append(method_row)

    out.sort(
        key=lambda row: (
            row.get("accuracy_positive_datasets", 0),
            row.get("roc_auc_positive_datasets", 0),
            row.get("accuracy_mean_delta", -999),
            row.get("roc_auc_mean_delta", -999),
        ),
        reverse=True,
    )
    return out


def fmt(value, signed=False):
    if value in {"", None}:
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: list[dict[str, object]],
    individual: list[dict[str, object]],
    method_summary: list[dict[str, object]],
) -> None:
    lines = [
        "# Experiment 2 Search Summary",
        "",
        "This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.",
        "",
        "## Overall Method Stability",
        "",
    ]
    if method_summary:
        lines.extend(
            [
                "| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in method_summary[:40]:
            auc_ds = row.get("roc_auc_datasets", 0)
            lines.append(
                f"| {row['method']} | {row['datasets']} | {row['runs']} | "
                f"{fmt(row.get('accuracy_mean_delta'), signed=True)} | "
                f"{row.get('accuracy_positive_datasets', 0)}/{row.get('accuracy_datasets', 0)} | "
                f"{fmt(row.get('roc_auc_mean_delta'), signed=True)} | "
                f"{row.get('roc_auc_positive_datasets', 0)}/{auc_ds} | "
                f"{row.get('description', '')} |"
            )
    else:
        lines.append("_No method-level summary rows._")

    lines.extend(
        [
            "",
        "## Strongest Dataset-Method Signals",
        "",
        ]
    )
    if summary:
        lines.extend(
            [
                "| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |",
                "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in summary[:60]:
            pos = (
                f"acc {row.get('accuracy_positive_runs', 0)}/{row['runs']}, "
                f"bacc {row.get('balanced_accuracy_positive_runs', 0)}/{row['runs']}, "
                f"f1 {row.get('macro_f1_positive_runs', 0)}/{row['runs']}"
            )
            acc_change = (
                f"{fmt(row.get('baseline_accuracy_mean'))} -> {fmt(row.get('accuracy_mean'))} "
                f"({fmt(row.get('accuracy_mean_delta'), signed=True)})"
                if "accuracy_mean_delta" in row
                else ""
            )
            bacc_change = (
                f"{fmt(row.get('baseline_balanced_accuracy_mean'))} -> {fmt(row.get('balanced_accuracy_mean'))} "
                f"({fmt(row.get('balanced_accuracy_mean_delta'), signed=True)})"
                if "balanced_accuracy_mean_delta" in row
                else ""
            )
            f1_change = (
                f"{fmt(row.get('baseline_macro_f1_mean'))} -> {fmt(row.get('macro_f1_mean'))} "
                f"({fmt(row.get('macro_f1_mean_delta'), signed=True)})"
                if "macro_f1_mean_delta" in row
                else ""
            )
            auc_change = (
                f"{fmt(row.get('baseline_roc_auc_mean'))} -> {fmt(row.get('roc_auc_mean'))} "
                f"({fmt(row.get('roc_auc_mean_delta'), signed=True)})"
                if "roc_auc_mean_delta" in row
                else ""
            )
            lines.append(
                f"| {row['method']} | {row['dataset']} | {row['runs']} | {acc_change} | "
                f"{bacc_change} | {f1_change} | {auc_change} | {pos} | {row.get('description', '')} |"
            )
    else:
        lines.append("_No comparable rows._")

    lines.extend(["", "## Top Single-Run Metric Improvements", ""])
    positives = [
        row
        for row in individual
        if any(row.get(f"{metric}_delta", 0) > 0 for metric in METRICS)
    ]
    if positives:
        lines.extend(
            [
                "| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |",
                "| --- | ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in positives[:80]:
            improved = [
                metric
                for metric in METRICS
                if row.get(f"{metric}_delta", 0) > 0
            ]
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['method']} | {', '.join(improved)} | "
                f"{fmt(row.get('baseline_accuracy'))} -> {fmt(row.get('accuracy'))} ({fmt(row.get('accuracy_delta'), signed=True)}) | "
                f"{fmt(row.get('baseline_balanced_accuracy'))} -> {fmt(row.get('balanced_accuracy'))} ({fmt(row.get('balanced_accuracy_delta'), signed=True)}) | "
                f"{fmt(row.get('baseline_macro_f1'))} -> {fmt(row.get('macro_f1'))} ({fmt(row.get('macro_f1_delta'), signed=True)}) | "
                f"{fmt(row.get('baseline_roc_auc'))} -> {fmt(row.get('roc_auc'))} ({fmt(row.get('roc_auc_delta'), signed=True)}) |"
            )
    else:
        lines.append("_No positive metric deltas._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment2 search results by metric deltas.")
    parser.add_argument("--input", default="search_results.csv")
    parser.add_argument("--summary-output", default="search_summary.csv")
    parser.add_argument("--method-summary-output", default="search_method_summary.csv")
    parser.add_argument("--individual-output", default="search_individual_deltas.csv")
    parser.add_argument("--report-output", default="search_report.md")
    args = parser.parse_args()

    rows = load_rows(RESULTS_DIR / args.input)
    summary, individual = summarize(rows)
    method_summary = summarize_methods(individual)
    write_csv(RESULTS_DIR / args.summary_output, summary)
    write_csv(RESULTS_DIR / args.method_summary_output, method_summary)
    write_csv(RESULTS_DIR / args.individual_output, individual)
    write_report(RESULTS_DIR / args.report_output, summary, individual, method_summary)
    print(f"[experiment2] Wrote {RESULTS_DIR / args.report_output}")


if __name__ == "__main__":
    main()
