from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from context_selection_search import ClassificationDataset, SplitData, load_datasets, make_split
from query_context_selection import QueryMethod, predict_with_query_context, query_methods, subset_dataset
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


DEFAULT_CANDIDATES = [
    "random_context",
    "propensity_score_context",
    "propensity_feature_context",
]


def make_inner_split(split: SplitData, validation_size: float, random_state: int) -> SplitData:
    indices = np.arange(split.X_train.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=split.y_train if len(np.unique(split.y_train)) > 1 else None,
    )
    return SplitData(
        X_train=split.X_train[train_idx],
        X_test=split.X_train[val_idx],
        y_train=split.y_train[train_idx],
        y_test=split.y_train[val_idx],
        r_train=split.r_train[train_idx],
        r_test=split.r_train[val_idx],
        p_train=split.p_train[train_idx],
        p_test=split.p_train[val_idx],
        causal_weights=split.causal_weights,
    )


def selected_methods(candidate_names: list[str]) -> list[QueryMethod]:
    methods_by_name = {method.name: method for method in query_methods()}
    return [methods_by_name[name] for name in candidate_names]


def choose_query_method(
    dataset: ClassificationDataset,
    split: SplitData,
    candidates: list[QueryMethod],
    *,
    device: str,
    context_size: int,
    validation_size: float,
    max_validation_queries: int,
    random_state: int,
    min_validation_delta: float,
) -> tuple[QueryMethod, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size, random_state + 149)
    y_val = inner.y_test[: min(max_validation_queries, inner.y_test.shape[0])]
    rows = []
    for method in candidates:
        pred, details = predict_with_query_context(
            inner,
            method,
            device=device,
            context_size=context_size,
            max_test=max_validation_queries,
            random_state=random_state,
        )
        rows.append(
            {
                "dataset": dataset.name,
                "method": method.name,
                "metric_value": float(accuracy_score(y_val, pred)),
                "random_state": random_state,
                "selection_phase": "inner_validation",
                **details,
            }
        )

    baseline = next(row for row in rows if row["method"] == "random_context")
    baseline_score = float(baseline["metric_value"])
    for row in rows:
        row["validation_baseline_metric_value"] = baseline_score
        row["validation_delta_vs_random"] = float(row["metric_value"]) - baseline_score
    best = max(rows, key=lambda row: (float(row["metric_value"]), query_method_priority(str(row["method"]))))
    if best["method"] == "random_context" or float(best["validation_delta_vs_random"]) < min_validation_delta:
        chosen_name = "random_context"
    else:
        chosen_name = str(best["method"])
    chosen = next(method for method in candidates if method.name == chosen_name)
    decision = {
        "dataset": dataset.name,
        "random_state": random_state,
        "chosen_method": chosen.name,
        "validation_best_method": best["method"],
        "validation_best_metric_value": float(best["metric_value"]),
        "validation_random_metric_value": baseline_score,
        "validation_delta_vs_random": float(best["validation_delta_vs_random"]),
        "min_validation_delta": min_validation_delta,
    }
    return chosen, rows, decision


def query_method_priority(method_name: str) -> int:
    order = {
        "propensity_feature_context": 5,
        "propensity_score_context": 3,
        "random_context": 1,
    }
    return order.get(method_name, 0)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("method") != "guarded_query_policy":
            continue
        grouped.setdefault((int(row["max_samples"]), str(row["dataset"])), []).append(row)
    summary = []
    for (max_samples, dataset), group in grouped.items():
        deltas = np.asarray([float(row["delta_vs_random"]) for row in group], dtype=float)
        summary.append(
            {
                "max_samples": max_samples,
                "dataset": dataset,
                "seeds": len(group),
                "positive_seeds": int(np.sum(deltas > 0)),
                "nonnegative_seeds": int(np.sum(deltas >= 0)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "mean_metric_value": float(np.mean([float(row["metric_value"]) for row in group])),
                "mean_random_metric_value": float(np.mean([float(row["random_metric_value"]) for row in group])),
            }
        )
    summary.sort(key=lambda row: (int(row["max_samples"]), float(row["mean_delta"])), reverse=True)
    return summary


def write_report(summary_rows: list[dict[str, object]], decisions: list[dict[str, object]]) -> None:
    lines = [
        "# Query Guarded Policy",
        "",
        "This experiment chooses a query-level context selection method using an inner validation split, then evaluates the selected method on held-out query rows.",
        "",
        "## Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Seeds | Positive Seeds | Mean Delta | Min Delta | Max Delta |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )

    lines.extend(["", "## Decisions", ""])
    if decisions:
        lines.extend(["| Max Samples | Dataset | Seed | Chosen | Validation Best | Validation Delta |", "| ---: | --- | ---: | --- | --- | ---: |"])
        for row in decisions:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['chosen_method']} | "
                f"{row['validation_best_method']} | {float(row['validation_delta_vs_random']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Deltas are measured against the limited random-context baseline, not full context.",
            "- The policy is useful when validation-selected query context transfers to held-out query rows.",
            "- Because each query refits TabICL, this is a slower but more direct test of the slide-style context selection idea.",
        ]
    )
    (RESULTS_DIR / "query_guarded_policy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate guarded query-level context selection.")
    parser.add_argument("--datasets", default="breast_cancer,synth_selection_hard_overlap,synth_high_dim_sparse")
    parser.add_argument("--scales", default="300,500")
    parser.add_argument("--random-states", default="42")
    parser.add_argument("--context-size", type=int, default=96)
    parser.add_argument("--max-test", type=int, default=20)
    parser.add_argument("--max-validation-queries", type=int, default=12)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.05)
    parser.add_argument("--synthetic-samples", type=int, default=900)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    wanted = {part.strip() for part in args.datasets.split(",") if part.strip()}
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    candidates = selected_methods(DEFAULT_CANDIDATES)
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for random_state in random_states:
        datasets = [dataset for dataset in load_datasets(args.synthetic_samples, random_state) if dataset.name in wanted]
        for max_samples in scales:
            for dataset in datasets:
                small_dataset = subset_dataset(dataset, max_samples, random_state)
                print(f"Dataset={small_dataset.name} max_samples={max_samples} seed={random_state}")
                split = make_split(small_dataset, test_size=0.25, random_state=random_state)
                random_method = next(method for method in candidates if method.name == "random_context")
                y_eval = split.y_test[: min(args.max_test, split.y_test.shape[0])]
                random_pred, random_details = predict_with_query_context(
                    split,
                    random_method,
                    device=device,
                    context_size=args.context_size,
                    max_test=args.max_test,
                    random_state=random_state,
                )
                random_score = float(accuracy_score(y_eval, random_pred))
                rows.append(
                    {
                        "max_samples": max_samples,
                        "dataset": small_dataset.name,
                        "source": small_dataset.source,
                        "method": "random_context",
                        "metric_name": "accuracy",
                        "metric_value": random_score,
                        "random_state": random_state,
                        "n_samples": int(small_dataset.X.shape[0]),
                        "n_features": int(small_dataset.X.shape[1]),
                        "context_size": args.context_size,
                        "device": device,
                        **random_details,
                    }
                )

                chosen, val_rows, decision = choose_query_method(
                    small_dataset,
                    split,
                    candidates,
                    device=device,
                    context_size=args.context_size,
                    validation_size=args.validation_size,
                    max_validation_queries=args.max_validation_queries,
                    random_state=random_state,
                    min_validation_delta=args.min_validation_delta,
                )
                for row in val_rows:
                    row["max_samples"] = max_samples
                validation_rows.extend(val_rows)
                decision["max_samples"] = max_samples
                decisions.append(decision)
                chosen_pred, chosen_details = predict_with_query_context(
                    split,
                    chosen,
                    device=device,
                    context_size=args.context_size,
                    max_test=args.max_test,
                    random_state=random_state,
                )
                chosen_score = float(accuracy_score(y_eval, chosen_pred))
                rows.append(
                    {
                        "max_samples": max_samples,
                        "dataset": small_dataset.name,
                        "source": small_dataset.source,
                        "method": "guarded_query_policy",
                        "chosen_method": chosen.name,
                        "metric_name": "accuracy",
                        "metric_value": chosen_score,
                        "random_metric_value": random_score,
                        "delta_vs_random": chosen_score - random_score,
                        "validation_best_method": decision["validation_best_method"],
                        "validation_delta_vs_random": decision["validation_delta_vs_random"],
                        "random_state": random_state,
                        "n_samples": int(small_dataset.X.shape[0]),
                        "n_features": int(small_dataset.X.shape[1]),
                        "context_size": args.context_size,
                        "device": device,
                        **chosen_details,
                    }
                )
                print(
                    f"  chosen={chosen.name} validation_delta={float(decision['validation_delta_vs_random']):+.4f} "
                    f"test_delta={chosen_score - random_score:+.4f}"
                )

    summary_rows = summarize(rows)
    write_rows(RESULTS_DIR / "query_guarded_policy_results.csv", rows)
    write_rows(RESULTS_DIR / "query_guarded_policy_summary.csv", summary_rows)
    write_rows(RESULTS_DIR / "query_guarded_policy_decisions.csv", decisions)
    write_rows(RESULTS_DIR / "query_guarded_policy_validation.csv", validation_rows)
    write_report(summary_rows, decisions)
    write_json(
        RESULTS_DIR / "query_guarded_policy_environment.json",
        {
            **environment_info(device),
            "datasets": sorted(wanted),
            "scales": scales,
            "random_states": random_states,
            "context_size": args.context_size,
            "max_test": args.max_test,
            "max_validation_queries": args.max_validation_queries,
            "min_validation_delta": args.min_validation_delta,
            "candidates": DEFAULT_CANDIDATES,
        },
    )
    print(f"Wrote {RESULTS_DIR / 'query_guarded_policy.md'}")


if __name__ == "__main__":
    main()
