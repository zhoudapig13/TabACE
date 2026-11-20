from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, make_friedman1, make_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from tabicl import TabICLRegressor
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


@dataclass(frozen=True)
class RegressionDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    source: str


@dataclass(frozen=True)
class RegressionSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class RegressionMethod:
    name: str
    description: str
    transform: Callable[[RegressionSplit, int], tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], dict[str, object]]]
    model_kwargs: dict[str, object]


def subset_rows(X: np.ndarray, y: np.ndarray, max_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_samples:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(np.arange(X.shape[0]), size=max_samples, replace=False)
    return X[idx], y[idx]


def load_regression_datasets(max_samples: int, random_state: int) -> list[RegressionDataset]:
    datasets: list[RegressionDataset] = []

    diabetes = load_diabetes()
    X, y = subset_rows(np.asarray(diabetes.data, dtype=np.float32), np.asarray(diabetes.target, dtype=np.float32), max_samples, random_state)
    datasets.append(RegressionDataset("sklearn_diabetes", X, y, "sklearn:diabetes"))

    try:
        california = fetch_california_housing()
        X, y = subset_rows(
            np.asarray(california.data, dtype=np.float32),
            np.asarray(california.target, dtype=np.float32),
            max_samples,
            random_state,
        )
        datasets.append(RegressionDataset("sklearn_california_housing", X, y, "sklearn:california_housing"))
    except Exception as exc:
        print(f"California housing load failed: {exc!r}")

    X, y = make_friedman1(n_samples=max_samples, n_features=12, noise=1.0, random_state=random_state)
    datasets.append(RegressionDataset("synthetic_friedman1", X.astype(np.float32), y.astype(np.float32), "synthetic:friedman1"))

    X, y = make_regression(
        n_samples=max_samples,
        n_features=30,
        n_informative=8,
        noise=12.0,
        effective_rank=10,
        random_state=random_state,
    )
    datasets.append(RegressionDataset("synthetic_low_rank", X.astype(np.float32), y.astype(np.float32), "synthetic:low_rank"))

    rng = np.random.default_rng(random_state)
    X, y = make_regression(n_samples=max_samples, n_features=20, n_informative=6, noise=8.0, random_state=random_state + 17)
    outlier_idx = rng.choice(np.arange(max_samples), size=max(4, max_samples // 20), replace=False)
    y = y.astype(np.float32)
    y[outlier_idx] += rng.normal(0, np.std(y) * 4.0, size=len(outlier_idx)).astype(np.float32)
    datasets.append(RegressionDataset("synthetic_outlier_target", X.astype(np.float32), y, "synthetic:outlier_target"))
    return datasets


def make_split(dataset: RegressionDataset, test_size: float, random_state: int) -> RegressionSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
    )
    return RegressionSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def identity(split: RegressionSplit, random_state: int):
    return split.X_train, split.y_train, split.X_test, lambda pred: pred, {"target_transform": "identity"}


def standard_features(split: RegressionSplit, random_state: int):
    scaler = StandardScaler()
    return scaler.fit_transform(split.X_train), split.y_train, scaler.transform(split.X_test), lambda pred: pred, {"feature_transform": "standard"}


def robust_features(split: RegressionSplit, random_state: int):
    scaler = RobustScaler()
    return scaler.fit_transform(split.X_train), split.y_train, scaler.transform(split.X_test), lambda pred: pred, {"feature_transform": "robust"}


def quantile_features(split: RegressionSplit, random_state: int):
    n_quantiles = min(100, split.X_train.shape[0])
    transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=random_state)
    return (
        transformer.fit_transform(split.X_train),
        split.y_train,
        transformer.transform(split.X_test),
        lambda pred: pred,
        {"feature_transform": "quantile_normal", "n_quantiles": n_quantiles},
    )


def target_standardized(split: RegressionSplit, random_state: int):
    mean = float(np.mean(split.y_train))
    std = float(np.std(split.y_train) + 1e-8)
    y_train = (split.y_train - mean) / std
    return split.X_train, y_train, split.X_test, lambda pred: pred * std + mean, {"target_transform": "standard"}


def target_yeo_johnson(split: RegressionSplit, random_state: int):
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    y_train = transformer.fit_transform(split.y_train.reshape(-1, 1)).ravel()
    return (
        split.X_train,
        y_train,
        split.X_test,
        lambda pred: transformer.inverse_transform(pred.reshape(-1, 1)).ravel(),
        {"target_transform": "yeo_johnson"},
    )


def clipped_target(split: RegressionSplit, random_state: int):
    lo, hi = np.quantile(split.y_train, [0.02, 0.98])
    return split.X_train, np.clip(split.y_train, lo, hi), split.X_test, lambda pred: pred, {"target_transform": "clip_2_98"}


def method_specs() -> list[RegressionMethod]:
    return [
        RegressionMethod("baseline_n1", "TabICLRegressor one-estimator baseline", identity, {"n_estimators": 1}),
        RegressionMethod("ensemble_n4", "Increase TabICLRegressor ensemble size to 4", identity, {"n_estimators": 4}),
        RegressionMethod("standard_features", "StandardScaler on features", standard_features, {"n_estimators": 1}),
        RegressionMethod("robust_features", "RobustScaler on features", robust_features, {"n_estimators": 1}),
        RegressionMethod("quantile_features", "Quantile-normal feature transform", quantile_features, {"n_estimators": 1}),
        RegressionMethod("target_standardized", "Standardize target and invert predictions", target_standardized, {"n_estimators": 1}),
        RegressionMethod("target_yeo_johnson", "Yeo-Johnson target transform and inverse predictions", target_yeo_johnson, {"n_estimators": 1}),
        RegressionMethod("clipped_target", "Clip extreme training targets", clipped_target, {"n_estimators": 1}),
    ]


def run_method(split: RegressionSplit, dataset: RegressionDataset, method: RegressionMethod, *, device: str, random_state: int) -> dict[str, object]:
    X_train, y_train, X_test, inverse_pred, details = method.transform(split, random_state)
    start = time.perf_counter()
    model = TabICLRegressor(device=device, random_state=random_state, verbose=False, **method.model_kwargs)
    model.fit(X_train, y_train)
    pred = inverse_pred(np.asarray(model.predict(X_test), dtype=np.float32))
    runtime = time.perf_counter() - start
    return {
        "dataset": dataset.name,
        "source": dataset.source,
        "method": method.name,
        "description": method.description,
        "metric_name": "r2",
        "metric_value": float(r2_score(split.y_test, pred)),
        "mae": float(mean_absolute_error(split.y_test, pred)),
        "n_samples": int(dataset.X.shape[0]),
        "n_features_original": int(dataset.X.shape[1]),
        "train_samples_original": int(split.X_train.shape[0]),
        "test_samples": int(split.X_test.shape[0]),
        "n_features_used": int(X_train.shape[1]),
        "n_estimators": int(method.model_kwargs.get("n_estimators", 1)),
        "device": device,
        "random_state": random_state,
        "runtime_sec": runtime,
        **details,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        if row.get("method") == "baseline_n1" and row.get("metric_value") not in {"", None}:
            baseline[(str(row["dataset"]), int(row["random_state"]))] = row
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("metric_value") in {"", None} or row.get("method") in {"baseline_n1", "load_failed"}:
            continue
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)

    out = []
    for (dataset, method), group in grouped.items():
        deltas = []
        mae_deltas = []
        for row in group:
            base = baseline.get((dataset, int(row["random_state"])))
            if not base:
                continue
            deltas.append(float(row["metric_value"]) - float(base["metric_value"]))
            mae_deltas.append(float(base["mae"]) - float(row["mae"]))
        if not deltas:
            continue
        values = np.asarray(deltas, dtype=float)
        mae_values = np.asarray(mae_deltas, dtype=float)
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": len(values),
                "positive_seeds": int(np.sum(values > 0)),
                "nonnegative_seeds": int(np.sum(values >= 0)),
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "mean_mae_delta": float(np.mean(mae_values)),
                "min_mae_delta": float(np.min(mae_values)),
                "max_mae_delta": float(np.max(mae_values)),
            }
        )
    out.sort(key=lambda row: (float(row["mean_delta"]), int(row["positive_seeds"])), reverse=True)
    return out


def winner_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        if row.get("method") == "baseline_n1" and row.get("metric_value") not in {"", None}:
            baseline[(str(row["dataset"]), int(row["random_state"]))] = row
    winners = []
    for row in rows:
        if row.get("metric_value") in {"", None} or row.get("method") in {"baseline_n1", "load_failed"}:
            continue
        base = baseline.get((str(row["dataset"]), int(row["random_state"])))
        if not base:
            continue
        delta = float(row["metric_value"]) - float(base["metric_value"])
        if delta > 0:
            out = dict(row)
            out["baseline_metric_value"] = float(base["metric_value"])
            out["delta_vs_baseline"] = delta
            out["baseline_mae"] = float(base["mae"])
            out["mae_delta_vs_baseline"] = float(base["mae"]) - float(row["mae"])
            winners.append(out)
    winners.sort(key=lambda row: (float(row["delta_vs_baseline"]), float(row["metric_value"])), reverse=True)
    return winners


def write_report(summary_rows: list[dict[str, object]], winners: list[dict[str, object]], report_output: Path) -> None:
    lines = [
        "# Regression Method Search",
        "",
        "This run extends the method search to TabICLRegressor using R2 as the primary metric and MAE as a secondary check.",
        "",
        "## Mean Delta Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Dataset | Method | Seeds | Positive | Mean R2 Delta | Min R2 Delta | Mean MAE Delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows[:40]:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row['min_delta']):+.4f} | {float(row['mean_mae_delta']):+.4f} |"
            )
    else:
        lines.append("_No valid summary rows._")

    lines.extend(["", "## Top Individual Improvements", ""])
    if winners:
        lines.extend(
            [
                "| Dataset | Seed | Method | R2 | Baseline R2 | Delta | MAE Delta |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in winners[:40]:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['method']} | {float(row['metric_value']):.4f} | "
                f"{float(row['baseline_metric_value']):.4f} | {float(row['delta_vs_baseline']):+.4f} | "
                f"{float(row['mae_delta_vs_baseline']):+.4f} |"
            )
    else:
        lines.append("_No positive individual improvements._")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive R2 deltas indicate a method beat the one-estimator TabICLRegressor baseline on the same split.",
            "- Positive MAE deltas mean MAE decreased, so a method improved absolute error as well.",
            "- Regression methods need separate evidence from classification because target transforms can help or hurt independently of feature transforms.",
        ]
    )
    report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search TabICLRegressor improvement methods.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--random-states", default="42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--methods", default="all")
    parser.add_argument("--output", default="regression_method_search_results.csv")
    parser.add_argument("--summary-output", default="regression_method_search_summary.csv")
    parser.add_argument("--winners-output", default="regression_method_search_winners.csv")
    parser.add_argument("--report-output", default="regression_method_search.md")
    parser.add_argument("--environment-output", default="regression_method_search_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    wanted = None if args.datasets == "all" else {part.strip() for part in args.datasets.split(",") if part.strip()}
    all_methods = method_specs()
    if args.methods == "all":
        methods = all_methods
    else:
        names = {part.strip() for part in args.methods.split(",") if part.strip()}
        methods = [method for method in all_methods if method.name in names]
    rows: list[dict[str, object]] = []
    loaded: set[str] = set()

    for random_state in random_states:
        datasets = load_regression_datasets(args.max_samples, random_state)
        for dataset in datasets:
            if wanted is not None and dataset.name not in wanted:
                continue
            loaded.add(dataset.name)
            split = make_split(dataset, args.test_size, random_state)
            print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]} seed={random_state}")
            for method in methods:
                print(f"  method={method.name}")
                try:
                    rows.append(run_method(split, dataset, method, device=device, random_state=random_state))
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "source": dataset.source,
                            "method": method.name,
                            "description": method.description,
                            "metric_name": "r2",
                            "metric_value": "",
                            "mae": "",
                            "error": repr(exc),
                            "n_samples": int(dataset.X.shape[0]),
                            "n_features_original": int(dataset.X.shape[1]),
                            "device": device,
                            "random_state": random_state,
                        }
                    )
                    print(f"    failed: {exc!r}")

    valid_rows = [row for row in rows if row.get("metric_value") not in {"", None}]
    summary_rows = summarize(valid_rows)
    winners = winner_rows(valid_rows)
    write_rows(RESULTS_DIR / args.output, rows)
    write_rows(RESULTS_DIR / args.summary_output, summary_rows)
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(summary_rows, winners, RESULTS_DIR / args.report_output)
    write_json(
        RESULTS_DIR / args.environment_output,
        {
            **environment_info(device),
            "max_samples": args.max_samples,
            "random_states": random_states,
            "test_size": args.test_size,
            "loaded_datasets": sorted(loaded),
            "methods": [method.name for method in methods],
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.summary_output}")
    print(f"Wrote {RESULTS_DIR / args.report_output}")


if __name__ == "__main__":
    main()
