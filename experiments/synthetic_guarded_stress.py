from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from context_selection_search import ClassificationDataset, load_datasets, make_split, run_method
from openml_adaptive_policy import choose_method
from openml_propensity_variant_search import variant_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


DEFAULT_DATASETS = [
    "synth_selection_linear",
    "synth_selection_hard_overlap",
    "synth_rare_boundary",
    "synth_high_dim_sparse",
]


def subset_dataset(dataset: ClassificationDataset, max_samples: int, random_state: int) -> ClassificationDataset:
    if dataset.X.shape[0] <= max_samples:
        return dataset
    rng = np.random.default_rng(random_state)
    selected = []
    classes = np.unique(dataset.y)
    per_class = max(1, max_samples // len(classes))
    for cls in classes:
        cls_idx = np.flatnonzero(dataset.y == cls)
        take = min(per_class, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    remaining = max_samples - len(selected)
    if remaining > 0:
        used = np.zeros(dataset.X.shape[0], dtype=bool)
        used[selected] = True
        candidates = np.flatnonzero(~used)
        selected.extend(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False).tolist())
    selected = np.asarray(selected)
    rng.shuffle(selected)
    return ClassificationDataset(
        name=dataset.name,
        X=dataset.X[selected],
        y=dataset.y[selected],
        source=f"{dataset.source}_subset",
    )


def summarize_by_dataset(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("method") == "adaptive_policy":
            grouped[(int(row["max_samples"]), str(row["dataset"]))].append(row)
    summary = []
    for (max_samples, dataset), group in grouped.items():
        deltas = np.asarray([float(row["delta_vs_baseline"]) for row in group], dtype=float)
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
                "mean_baseline_metric_value": float(np.mean([float(row["baseline_metric_value"]) for row in group])),
            }
        )
    summary.sort(key=lambda row: (int(row["max_samples"]), float(row["mean_delta"])), reverse=True)
    return summary


def summarize_by_scale(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("method") == "adaptive_policy":
            grouped[int(row["max_samples"])].append(row)
    summary = []
    for max_samples, group in grouped.items():
        deltas = np.asarray([float(row["delta_vs_baseline"]) for row in group], dtype=float)
        summary.append(
            {
                "max_samples": max_samples,
                "runs": len(group),
                "positive_runs": int(np.sum(deltas > 0)),
                "nonnegative_runs": int(np.sum(deltas >= 0)),
                "macro_mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
            }
        )
    summary.sort(key=lambda row: int(row["max_samples"]))
    return summary


def stress_winners(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("metric_value") not in {"", None}:
            grouped[(int(row["max_samples"]), str(row["dataset"]), int(row["random_state"]))].append(row)
    out = []
    for _, group in grouped.items():
        baseline = next((row for row in group if row["method"] == "baseline_n1"), None)
        adaptive = next((row for row in group if row["method"] == "adaptive_policy"), None)
        if baseline is None or adaptive is None:
            continue
        delta = float(adaptive["metric_value"]) - float(baseline["metric_value"])
        if delta > 0:
            row = dict(adaptive)
            row["baseline_metric_value"] = float(baseline["metric_value"])
            row["delta_vs_baseline"] = delta
            out.append(row)
    out.sort(key=lambda row: (float(row["delta_vs_baseline"]), float(row["metric_value"])), reverse=True)
    return out


def write_report(scale_summary: list[dict[str, object]], dataset_summary: list[dict[str, object]], decisions: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Guarded Stress Test",
        "",
        "This experiment applies the guarded adaptive policy to synthetic special datasets designed to stress selection bias, rare boundaries, and sparse high-dimensional features.",
        "",
        "## Overall By Scale",
        "",
    ]
    if scale_summary:
        lines.extend(
            [
                "| Max Samples | Runs | Positive Runs | Macro Mean Delta | Median Delta | Min Delta | Max Delta |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in scale_summary:
            lines.append(
                f"| {row['max_samples']} | {row['runs']} | {row['positive_runs']} | "
                f"{float(row['macro_mean_delta']):+.4f} | {float(row['median_delta']):+.4f} | "
                f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )

    lines.extend(["", "## Dataset By Scale", ""])
    if dataset_summary:
        lines.extend(
            [
                "| Max Samples | Dataset | Seeds | Positive Seeds | Mean Delta | Min Delta | Max Delta |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dataset_summary:
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
                f"{row['validation_best_method']} | {float(row['validation_delta_vs_baseline']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive deltas mean guarded method selection beat the baseline TabICL fit on the same split.",
            "- Synthetic stress datasets are intentionally harsher than ordinary demos, so negative deltas reveal where policy guards are still incomplete.",
            "- The current guarded policy treats `synthetic_special` sources as stress tests and falls back to baseline by default.",
            "- Stable positive signals should be promoted into the method catalog; unstable ones remain research candidates.",
        ]
    )
    (RESULTS_DIR / "synthetic_guarded_stress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test guarded adaptive policy on synthetic special datasets.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--scales", default="300,650,900")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--synthetic-samples", type=int, default=1200)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    wanted = {part.strip() for part in args.datasets.split(",") if part.strip()}
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    methods = variant_specs()
    baseline_method = next(method for method in methods if method.name == "baseline_n1")
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for random_state in random_states:
        datasets = [dataset for dataset in load_datasets(args.synthetic_samples, random_state) if dataset.name in wanted]
        for max_samples in scales:
            for dataset in datasets:
                small_dataset = subset_dataset(dataset, max_samples, random_state)
                print(f"Dataset={small_dataset.name} max_samples={max_samples} seed={random_state}")
                split = make_split(small_dataset, args.test_size, random_state)
                baseline_row = run_method(split, small_dataset, baseline_method, device=device, random_state=random_state)
                baseline_score = float(baseline_row["metric_value"])
                baseline_row["max_samples"] = max_samples
                rows.append(baseline_row)

                chosen, val_rows, decision = choose_method(
                    small_dataset,
                    split,
                    methods,
                    device=device,
                    validation_size=args.validation_size,
                    random_state=random_state,
                    min_validation_delta=args.min_validation_delta,
                    policy="guarded",
                )
                for row in val_rows:
                    row["max_samples"] = max_samples
                validation_rows.extend(val_rows)
                decision["max_samples"] = max_samples
                decisions.append(decision)

                chosen_row = run_method(split, small_dataset, chosen, device=device, random_state=random_state)
                adaptive_row = dict(chosen_row)
                adaptive_row["method"] = "adaptive_policy"
                adaptive_row["chosen_method"] = chosen.name
                adaptive_row["description"] = f"Synthetic stress guarded policy selected {chosen.name}"
                adaptive_row["baseline_metric_value"] = baseline_score
                adaptive_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - baseline_score
                adaptive_row["validation_best_method"] = decision["validation_best_method"]
                adaptive_row["validation_delta_vs_baseline"] = decision["validation_delta_vs_baseline"]
                adaptive_row["max_samples"] = max_samples
                rows.append(adaptive_row)
                print(
                    f"  chosen={chosen.name} validation_delta={float(decision['validation_delta_vs_baseline']):+.4f} "
                    f"test_delta={float(adaptive_row['delta_vs_baseline']):+.4f}"
                )

    scale_summary = summarize_by_scale(rows)
    dataset_summary = summarize_by_dataset(rows)
    write_rows(RESULTS_DIR / "synthetic_guarded_stress_results.csv", rows)
    write_rows(RESULTS_DIR / "synthetic_guarded_stress_scale_summary.csv", scale_summary)
    write_rows(RESULTS_DIR / "synthetic_guarded_stress_dataset_summary.csv", dataset_summary)
    write_rows(RESULTS_DIR / "synthetic_guarded_stress_decisions.csv", decisions)
    write_rows(RESULTS_DIR / "synthetic_guarded_stress_validation.csv", validation_rows)
    write_report(scale_summary, dataset_summary, decisions)
    winners = stress_winners(rows)
    if winners:
        write_rows(RESULTS_DIR / "synthetic_guarded_stress_winners.csv", winners)
    else:
        with (RESULTS_DIR / "synthetic_guarded_stress_winners.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_json(
        RESULTS_DIR / "synthetic_guarded_stress_environment.json",
        {
            **environment_info(device),
            "datasets": sorted(wanted),
            "scales": scales,
            "random_states": random_states,
            "synthetic_samples": args.synthetic_samples,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "min_validation_delta": args.min_validation_delta,
        },
    )
    print(f"Wrote {RESULTS_DIR / 'synthetic_guarded_stress.md'}")


if __name__ == "__main__":
    main()
