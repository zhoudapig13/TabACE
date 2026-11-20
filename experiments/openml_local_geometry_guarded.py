from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from sklearn.model_selection import train_test_split

from context_selection_search import ClassificationDataset, MethodSpec, SplitData, make_split, method_specs, run_method, winner_rows
from openml_broad_sweep import summarize
from openml_method_search import OpenMLSpec, load_openml_dataset, parse_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


DEFAULT_SPECS = [
    OpenMLSpec(38, "openml_sick", "Thyroid sick dataset with mixed features"),
    OpenMLSpec(23, "openml_cmc", "Contraceptive method choice with mixed social features"),
    OpenMLSpec(29, "openml_credit_approval", "Credit approval with mixed categorical/numeric features"),
    OpenMLSpec(31, "openml_credit_g", "Credit risk with mixed categorical/numeric features"),
]


def specs_from_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "default":
        return DEFAULT_SPECS
    return parse_specs(raw)


def local_geometry_methods() -> list[MethodSpec]:
    names = {"baseline_n1", "local_density_feature", "centroid_distance_feature", "knn_label_feature"}
    return [method for method in method_specs() if method.name in names]


def make_inner_split(split: SplitData, *, validation_size: float, random_state: int) -> SplitData:
    indices = np.arange(split.X_train.shape[0])
    inner_train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=split.y_train if len(np.unique(split.y_train)) > 1 else None,
    )
    return SplitData(
        X_train=split.X_train[inner_train_idx],
        X_test=split.X_train[val_idx],
        y_train=split.y_train[inner_train_idx],
        y_test=split.y_train[val_idx],
        r_train=split.r_train[inner_train_idx],
        r_test=split.r_train[val_idx],
        p_train=split.p_train[inner_train_idx],
        p_test=split.p_train[val_idx],
        causal_weights=split.causal_weights,
    )


def method_priority(method_name: str) -> int:
    return {
        "centroid_distance_feature": 4,
        "knn_label_feature": 3,
        "local_density_feature": 2,
        "baseline_n1": 1,
    }.get(method_name, 0)


def choose_local_geometry_method(
    dataset: ClassificationDataset,
    split: SplitData,
    methods: list[MethodSpec],
    *,
    device: str,
    validation_size: float,
    min_validation_delta: float,
    random_state: int,
) -> tuple[MethodSpec, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 1703)
    inner_dataset = replace(dataset, name=f"{dataset.name}__inner_local_geometry")
    validation_rows = []
    for method in methods:
        row = run_method(inner, inner_dataset, method, device=device, random_state=random_state)
        row["dataset"] = dataset.name
        row["selection_phase"] = "inner_validation"
        validation_rows.append(row)

    baseline = next(row for row in validation_rows if row["method"] == "baseline_n1")
    baseline_score = float(baseline["metric_value"])
    for row in validation_rows:
        row["validation_baseline_metric_value"] = baseline_score
        row["validation_delta_vs_baseline"] = float(row["metric_value"]) - baseline_score

    best_row = max(validation_rows, key=lambda row: (float(row["metric_value"]), method_priority(str(row["method"]))))
    if best_row["method"] == "baseline_n1" or float(best_row["validation_delta_vs_baseline"]) < min_validation_delta:
        chosen_name = "baseline_n1"
    else:
        chosen_name = str(best_row["method"])
    chosen = next(method for method in methods if method.name == chosen_name)
    decision = {
        "dataset": dataset.name,
        "random_state": random_state,
        "chosen_method": chosen.name,
        "validation_best_method": best_row["method"],
        "validation_best_metric_value": float(best_row["metric_value"]),
        "validation_baseline_metric_value": baseline_score,
        "validation_delta_vs_baseline": float(best_row["validation_delta_vs_baseline"]),
        "min_validation_delta": min_validation_delta,
        "candidate_methods": ",".join(method.name for method in methods),
    }
    return chosen, validation_rows, decision


def write_report(summary_rows: list[dict[str, object]], decisions: list[dict[str, object]], rows: list[dict[str, object]], prefix: str) -> None:
    lines = [
        "# OpenML Local Geometry Guarded Policy",
        "",
        "This experiment checks whether risky local-geometry features can be made safer through an inner validation gate.",
        "",
        "## Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Seeds | Positive | Mean Delta | Min Delta | Max Delta |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )
    else:
        lines.append("_No valid summary rows._")

    lines.extend(["", "## Decisions", ""])
    if decisions:
        lines.extend(
            [
                "| Dataset | Seed | Chosen Method | Validation Best | Validation Delta |",
                "| --- | ---: | --- | --- | ---: |",
            ]
        )
        for row in decisions:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['chosen_method']} | "
                f"{row['validation_best_method']} | {float(row['validation_delta_vs_baseline']):+.4f} |"
            )

    lines.extend(["", "## Test Runs", ""])
    adaptive_rows = [row for row in rows if row.get("method") == "adaptive_policy"]
    if adaptive_rows:
        lines.extend(["| Dataset | Seed | Chosen Method | Accuracy | Baseline | Delta |", "| --- | ---: | --- | ---: | ---: | ---: |"])
        for row in adaptive_rows:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['chosen_method']} | "
                f"{float(row['metric_value']):.4f} | {float(row['baseline_metric_value']):.4f} | "
                f"{float(row['delta_vs_baseline']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the guarded policy mostly returns baseline, the local-geometry signal is too weak or too noisy for default use.",
            "- Positive rows with nonnegative minima can become candidates for the main guarded policy.",
            "- Negative rows show that validation still fails to detect some split-specific local-structure overfitting.",
        ]
    )
    (RESULTS_DIR / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard risky local geometry features with inner validation.")
    parser.add_argument("--openml", default="default")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.02)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-prefix", default="openml_local_geometry_guarded")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    specs = specs_from_arg(args.openml)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    methods = local_geometry_methods()
    baseline_method = next(method for method in methods if method.name == "baseline_n1")
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    loaded_datasets: set[str] = set()

    for random_state in random_states:
        for spec in specs:
            print(f"Loading OpenML data_id={spec.data_id} alias={spec.alias} seed={random_state}")
            try:
                dataset = load_openml_dataset(spec, max_samples=args.max_samples, random_state=random_state)
            except Exception as exc:
                rows.append(
                    {
                        "max_samples": args.max_samples,
                        "dataset": spec.alias,
                        "source": f"openml:{spec.data_id}",
                        "method": "load_failed",
                        "metric_name": "accuracy",
                        "metric_value": "",
                        "error": repr(exc),
                        "random_state": random_state,
                        "device": device,
                    }
                )
                print(f"  failed: {exc!r}")
                continue

            loaded_datasets.add(dataset.name)
            split = make_split(dataset, args.test_size, random_state)
            baseline_row = run_method(split, dataset, baseline_method, device=device, random_state=random_state)
            baseline_row["max_samples"] = args.max_samples
            baseline_score = float(baseline_row["metric_value"])
            rows.append(baseline_row)

            chosen, val_rows, decision = choose_local_geometry_method(
                dataset,
                split,
                methods,
                device=device,
                validation_size=args.validation_size,
                min_validation_delta=args.min_validation_delta,
                random_state=random_state,
            )
            for row in val_rows:
                row["max_samples"] = args.max_samples
            validation_rows.extend(val_rows)
            decisions.append(decision)

            chosen_row = run_method(split, dataset, chosen, device=device, random_state=random_state)
            adaptive_row = dict(chosen_row)
            adaptive_row["max_samples"] = args.max_samples
            adaptive_row["method"] = "adaptive_policy"
            adaptive_row["chosen_method"] = chosen.name
            adaptive_row["description"] = f"Local-geometry guarded policy selected {chosen.name}"
            adaptive_row["baseline_metric_value"] = baseline_score
            adaptive_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - baseline_score
            adaptive_row["validation_best_method"] = decision["validation_best_method"]
            adaptive_row["validation_delta_vs_baseline"] = decision["validation_delta_vs_baseline"]
            rows.append(adaptive_row)
            print(
                f"  chosen={chosen.name} validation_delta={float(decision['validation_delta_vs_baseline']):+.4f} "
                f"test_delta={float(adaptive_row['delta_vs_baseline']):+.4f}"
            )

    summary_rows = summarize(rows)
    prefix = args.output_prefix
    write_rows(RESULTS_DIR / f"{prefix}_results.csv", rows)
    write_rows(RESULTS_DIR / f"{prefix}_summary.csv", summary_rows)
    write_rows(RESULTS_DIR / f"{prefix}_decisions.csv", decisions)
    write_rows(RESULTS_DIR / f"{prefix}_validation.csv", validation_rows)
    winners = winner_rows([row for row in rows if row.get("metric_value") not in {"", None}])
    if winners:
        write_rows(RESULTS_DIR / f"{prefix}_winners.csv", winners)
    else:
        with (RESULTS_DIR / f"{prefix}_winners.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(summary_rows, decisions, rows, prefix)
    write_json(
        RESULTS_DIR / f"{prefix}_environment.json",
        {
            **environment_info(device),
            "max_samples": args.max_samples,
            "random_states": random_states,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "min_validation_delta": args.min_validation_delta,
            "openml_specs": [spec.__dict__ for spec in specs],
            "loaded_datasets": sorted(loaded_datasets),
            "candidate_methods": [method.name for method in methods],
        },
    )
    print(f"Wrote {RESULTS_DIR / f'{prefix}.md'}")


if __name__ == "__main__":
    main()
