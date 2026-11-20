from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from sklearn.model_selection import train_test_split

from regression_method_search import (
    RegressionDataset,
    RegressionMethod,
    RegressionSplit,
    load_regression_datasets,
    make_split,
    method_specs,
    run_method,
    summarize,
    winner_rows,
)
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


def make_inner_split(split: RegressionSplit, *, validation_size: float, random_state: int) -> RegressionSplit:
    indices = np.arange(split.X_train.shape[0])
    inner_train_idx, val_idx = train_test_split(indices, test_size=validation_size, random_state=random_state)
    return RegressionSplit(
        X_train=split.X_train[inner_train_idx],
        X_test=split.X_train[val_idx],
        y_train=split.y_train[inner_train_idx],
        y_test=split.y_train[val_idx],
    )


def candidate_methods(dataset: RegressionDataset, methods: list[RegressionMethod]) -> list[RegressionMethod]:
    allowed = {"baseline_n1"}
    if dataset.name == "sklearn_diabetes":
        allowed.update({"clipped_target"})
    elif dataset.name == "synthetic_low_rank":
        allowed.update({"quantile_features"})
    return [method for method in methods if method.name in allowed]


def choose_method(
    dataset: RegressionDataset,
    split: RegressionSplit,
    methods: list[RegressionMethod],
    *,
    device: str,
    validation_size: float,
    random_state: int,
    min_validation_delta: float,
    min_validation_mae_delta: float,
) -> tuple[RegressionMethod, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 997)
    inner_dataset = replace(dataset, name=f"{dataset.name}__inner")
    candidates = candidate_methods(dataset, methods)
    validation_rows: list[dict[str, object]] = []
    for method in candidates:
        row = run_method(inner, inner_dataset, method, device=device, random_state=random_state)
        row["dataset"] = dataset.name
        row["selection_phase"] = "inner_validation"
        validation_rows.append(row)

    baseline = next(row for row in validation_rows if row["method"] == "baseline_n1")
    baseline_score = float(baseline["metric_value"])
    baseline_mae = float(baseline["mae"])
    for row in validation_rows:
        row["validation_baseline_metric_value"] = baseline_score
        row["validation_delta_vs_baseline"] = float(row["metric_value"]) - baseline_score
        row["validation_baseline_mae"] = baseline_mae
        row["validation_mae_delta_vs_baseline"] = baseline_mae - float(row["mae"])

    best_row = max(validation_rows, key=lambda row: (float(row["metric_value"]), float(row.get("validation_mae_delta_vs_baseline", 0.0))))
    validation_delta = float(best_row["validation_delta_vs_baseline"])
    mae_delta = float(best_row["validation_mae_delta_vs_baseline"])
    if (
        best_row["method"] == "baseline_n1"
        or validation_delta < min_validation_delta
        or mae_delta < min_validation_mae_delta
    ):
        chosen_name = "baseline_n1"
    else:
        chosen_name = str(best_row["method"])
    chosen = next(method for method in candidates if method.name == chosen_name)
    decision = {
        "dataset": dataset.name,
        "random_state": random_state,
        "chosen_method": chosen.name,
        "validation_best_method": best_row["method"],
        "validation_best_metric_value": float(best_row["metric_value"]),
        "validation_baseline_metric_value": baseline_score,
        "validation_delta_vs_baseline": validation_delta,
        "validation_best_mae": float(best_row["mae"]),
        "validation_baseline_mae": baseline_mae,
        "validation_mae_delta_vs_baseline": mae_delta,
        "min_validation_delta": min_validation_delta,
        "min_validation_mae_delta": min_validation_mae_delta,
        "candidate_methods": ",".join(method.name for method in candidates),
    }
    return chosen, validation_rows, decision


def write_report(summary_rows: list[dict[str, object]], decisions: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    adaptive_rows = [row for row in rows if row.get("method") == "adaptive_policy"]
    lines = [
        "# Regression Guarded Policy",
        "",
        "This experiment checks whether risky regression candidates can be safely exposed through inner validation.",
        "",
        "## Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Dataset | Seeds | Positive | Mean R2 Delta | Min R2 Delta | Mean MAE Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['dataset']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row['min_delta']):+.4f} | {float(row['mean_mae_delta']):+.4f} |"
            )

    lines.extend(["", "## Decisions", ""])
    if decisions:
        lines.extend(["| Dataset | Seed | Chosen | Validation Best | R2 Delta | MAE Delta |", "| --- | ---: | --- | --- | ---: | ---: |"])
        for row in decisions:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['chosen_method']} | {row['validation_best_method']} | "
                f"{float(row['validation_delta_vs_baseline']):+.4f} | {float(row['validation_mae_delta_vs_baseline']):+.4f} |"
            )

    lines.extend(["", "## Test Runs", ""])
    if adaptive_rows:
        lines.extend(["| Dataset | Seed | Chosen | R2 | Baseline R2 | Delta | MAE Delta |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"])
        for row in adaptive_rows:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['chosen_method']} | {float(row['metric_value']):.4f} | "
                f"{float(row['baseline_metric_value']):.4f} | {float(row['delta_vs_baseline']):+.4f} | "
                f"{float(row['mae_delta_vs_baseline']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The guard requires validation R2 and validation MAE to clear explicit positive thresholds.",
            "- Nonnegative test deltas mean the guard avoided the negative seeds from the raw regression recheck.",
            "- Positive test deltas that survive the guard become guarded regression candidates rather than unconditional defaults.",
        ]
    )
    (RESULTS_DIR / "regression_guarded_policy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard risky TabICLRegressor candidates with inner validation.")
    parser.add_argument("--datasets", default="sklearn_diabetes,synthetic_low_rank")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.006)
    parser.add_argument("--min-validation-mae-delta", type=float, default=0.20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    wanted = {part.strip() for part in args.datasets.split(",") if part.strip()}
    methods = method_specs()
    baseline_method = next(method for method in methods if method.name == "baseline_n1")
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    loaded: set[str] = set()

    for random_state in random_states:
        datasets = [dataset for dataset in load_regression_datasets(args.max_samples, random_state) if dataset.name in wanted]
        for dataset in datasets:
            loaded.add(dataset.name)
            split = make_split(dataset, args.test_size, random_state)
            print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]} seed={random_state}")
            baseline_row = run_method(split, dataset, baseline_method, device=device, random_state=random_state)
            baseline_score = float(baseline_row["metric_value"])
            baseline_mae = float(baseline_row["mae"])
            rows.append(baseline_row)

            chosen, val_rows, decision = choose_method(
                dataset,
                split,
                methods,
                device=device,
                validation_size=args.validation_size,
                random_state=random_state,
                min_validation_delta=args.min_validation_delta,
                min_validation_mae_delta=args.min_validation_mae_delta,
            )
            validation_rows.extend(val_rows)
            decisions.append(decision)

            chosen_row = run_method(split, dataset, chosen, device=device, random_state=random_state)
            adaptive_row = dict(chosen_row)
            adaptive_row["method"] = "adaptive_policy"
            adaptive_row["chosen_method"] = chosen.name
            adaptive_row["description"] = f"Regression guarded policy selected {chosen.name}"
            adaptive_row["baseline_metric_value"] = baseline_score
            adaptive_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - baseline_score
            adaptive_row["baseline_mae"] = baseline_mae
            adaptive_row["mae_delta_vs_baseline"] = baseline_mae - float(chosen_row["mae"])
            adaptive_row["validation_best_method"] = decision["validation_best_method"]
            adaptive_row["validation_delta_vs_baseline"] = decision["validation_delta_vs_baseline"]
            adaptive_row["validation_mae_delta_vs_baseline"] = decision["validation_mae_delta_vs_baseline"]
            rows.append(adaptive_row)
            print(
                f"  chosen={chosen.name} validation_delta={float(decision['validation_delta_vs_baseline']):+.4f} "
                f"test_delta={float(adaptive_row['delta_vs_baseline']):+.4f}"
            )

    summary_rows = summarize([row for row in rows if row.get("method") in {"baseline_n1", "adaptive_policy"}])
    winners = winner_rows([row for row in rows if row.get("metric_value") not in {"", None}])
    write_rows(RESULTS_DIR / "regression_guarded_policy_results.csv", rows)
    write_rows(RESULTS_DIR / "regression_guarded_policy_summary.csv", summary_rows)
    write_rows(RESULTS_DIR / "regression_guarded_policy_decisions.csv", decisions)
    write_rows(RESULTS_DIR / "regression_guarded_policy_validation.csv", validation_rows)
    if winners:
        write_rows(RESULTS_DIR / "regression_guarded_policy_winners.csv", winners)
    else:
        with (RESULTS_DIR / "regression_guarded_policy_winners.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(summary_rows, decisions, rows)
    write_json(
        RESULTS_DIR / "regression_guarded_policy_environment.json",
        {
            **environment_info(device),
            "max_samples": args.max_samples,
            "random_states": random_states,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "min_validation_delta": args.min_validation_delta,
            "min_validation_mae_delta": args.min_validation_mae_delta,
            "loaded_datasets": sorted(loaded),
        },
    )
    print(f"Wrote {RESULTS_DIR / 'regression_guarded_policy.md'}")


if __name__ == "__main__":
    main()
