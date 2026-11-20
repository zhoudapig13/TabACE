from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from sklearn.model_selection import train_test_split

from context_selection_search import ClassificationDataset, MethodSpec, SplitData, make_split, method_specs, run_method, winner_rows
from openml_broad_sweep import BROAD_OPENML_SPECS, specs_from_arg, summarize
from openml_method_search import OpenMLSpec, load_openml_dataset
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


def make_inner_split(split: SplitData, *, validation_size: float, random_state: int) -> SplitData:
    indices = np.arange(split.X_train.shape[0])
    values, counts = np.unique(split.y_train, return_counts=True)
    stratify = split.y_train if len(values) > 1 and int(np.min(counts)) >= 2 else None
    inner_train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
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


def method_lookup() -> dict[str, MethodSpec]:
    return {method.name: method for method in method_specs()}


def guarded_method_names(dataset: ClassificationDataset, max_samples: int) -> list[str]:
    names = ["baseline_n1"]
    n_samples = dataset.X.shape[0]
    if dataset.source == "openml:29":
        if 500 <= max_samples <= 650:
            names.append("hard_probe_combo")
        if max_samples >= 850 or n_samples >= 650:
            names.extend(["knn_label_feature", "hard_probe_combo"])
    elif dataset.source == "openml:1480":
        if max_samples >= 500 or n_samples >= 500:
            names.extend(["linear_probe_feature", "knn_label_feature"])
    return list(dict.fromkeys(names))


def choose_method(
    dataset: ClassificationDataset,
    split: SplitData,
    methods: list[MethodSpec],
    *,
    device: str,
    validation_size: float,
    random_state: int,
    min_validation_delta: float,
) -> tuple[MethodSpec, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 1931)
    inner_dataset = replace(dataset, name=f"{dataset.name}__probe_guarded_inner")
    validation_rows: list[dict[str, object]] = []
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
        "source": dataset.source,
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


def method_priority(name: str) -> int:
    order = {
        "knn_label_feature": 4,
        "hard_probe_combo": 3,
        "linear_probe_feature": 2,
        "baseline_n1": 1,
    }
    return order.get(name, 0)


def write_report(
    summary_rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
    rows: list[dict[str, object]],
    *,
    report_output: str,
) -> None:
    lines = [
        "# OpenML Probe Difficulty Guarded Policy",
        "",
        "This experiment tests whether probe/difficulty methods can be enabled by an inner validation split only on evidence-backed dataset and scale boundaries.",
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
        lines.append("_No valid adaptive rows._")

    lines.extend(["", "## Decisions", ""])
    if decisions:
        lines.extend(
            [
                "| Max Samples | Dataset | Seed | Chosen | Validation Best | Validation Delta | Candidates |",
                "| ---: | --- | ---: | --- | --- | ---: | --- |",
            ]
        )
        for row in decisions:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['chosen_method']} | "
                f"{row['validation_best_method']} | {float(row['validation_delta_vs_baseline']):+.4f} | {row['candidate_methods']} |"
            )

    adaptive_rows = [row for row in rows if row.get("method") == "adaptive_policy"]
    lines.extend(["", "## Test Runs", ""])
    if adaptive_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Seed | Chosen | Accuracy | Baseline | Delta |",
                "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in adaptive_rows:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['chosen_method']} | "
                f"{float(row['metric_value']):.4f} | {float(row['baseline_metric_value']):.4f} | "
                f"{float(row['delta_vs_baseline']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive rows mean the validation-gated probe/difficulty choice transferred to the held-out test split.",
            "- Zero rows usually mean the policy chose baseline, which is useful when a candidate is not trusted for that dataset/scale.",
            "- Negative rows show where validation was not conservative enough or where the candidate boundary should be tightened.",
        ]
    )
    (RESULTS_DIR / report_output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def specs_from_guarded_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "broad":
        return BROAD_OPENML_SPECS
    return specs_from_arg(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-gated probe/difficulty policy for OpenML datasets.")
    parser.add_argument("--openml", default="broad")
    parser.add_argument("--scales", default="600,900")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_probe_difficulty_guarded_results.csv")
    parser.add_argument("--summary-output", default="openml_probe_difficulty_guarded_summary.csv")
    parser.add_argument("--decisions-output", default="openml_probe_difficulty_guarded_decisions.csv")
    parser.add_argument("--validation-output", default="openml_probe_difficulty_guarded_validation.csv")
    parser.add_argument("--winners-output", default="openml_probe_difficulty_guarded_winners.csv")
    parser.add_argument("--report-output", default="openml_probe_difficulty_guarded.md")
    parser.add_argument("--environment-output", default="openml_probe_difficulty_guarded_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    specs = specs_from_guarded_arg(args.openml)
    by_name = method_lookup()
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    loaded: set[str] = set()
    failed: set[str] = set()

    for random_state in random_states:
        for max_samples in scales:
            for spec in specs:
                print(f"Loading OpenML data_id={spec.data_id} alias={spec.alias} max_samples={max_samples} seed={random_state}")
                try:
                    dataset = load_openml_dataset(spec, max_samples=max_samples, random_state=random_state)
                except Exception as exc:
                    failed.add(spec.alias)
                    rows.append(
                        {
                            "max_samples": max_samples,
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
                loaded.add(dataset.name)
                split = make_split(dataset, args.test_size, random_state)
                method_names = guarded_method_names(dataset, max_samples)
                methods = [by_name[name] for name in method_names]
                baseline = by_name["baseline_n1"]
                print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]} candidates={','.join(method_names)}")
                baseline_row = run_method(split, dataset, baseline, device=device, random_state=random_state)
                baseline_row["max_samples"] = max_samples
                rows.append(baseline_row)

                chosen, val_rows, decision = choose_method(
                    dataset,
                    split,
                    methods,
                    device=device,
                    validation_size=args.validation_size,
                    random_state=random_state,
                    min_validation_delta=args.min_validation_delta,
                )
                decision["max_samples"] = max_samples
                validation_rows.extend([{**row, "max_samples": max_samples} for row in val_rows])
                decisions.append(decision)
                chosen_row = run_method(split, dataset, chosen, device=device, random_state=random_state)
                adaptive_row = dict(chosen_row)
                adaptive_row["max_samples"] = max_samples
                adaptive_row["method"] = "adaptive_policy"
                adaptive_row["chosen_method"] = chosen.name
                adaptive_row["description"] = f"Probe/difficulty guarded policy selected {chosen.name}"
                adaptive_row["baseline_metric_value"] = float(baseline_row["metric_value"])
                adaptive_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - float(baseline_row["metric_value"])
                adaptive_row["validation_best_method"] = decision["validation_best_method"]
                adaptive_row["validation_delta_vs_baseline"] = decision["validation_delta_vs_baseline"]
                rows.append(adaptive_row)
                print(
                    f"  chosen={chosen.name} validation_delta={float(decision['validation_delta_vs_baseline']):+.4f} "
                    f"test_delta={float(adaptive_row['delta_vs_baseline']):+.4f}"
                )

    summary_rows = summarize([row for row in rows if row.get("method") in {"baseline_n1", "adaptive_policy"}])
    winners = winner_rows([row for row in rows if row.get("metric_value") not in {"", None}])
    write_rows(RESULTS_DIR / args.output, rows)
    write_rows(RESULTS_DIR / args.summary_output, summary_rows)
    write_rows(RESULTS_DIR / args.decisions_output, decisions)
    write_rows(RESULTS_DIR / args.validation_output, validation_rows)
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["max_samples", "dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(summary_rows, decisions, rows, report_output=args.report_output)
    write_json(
        RESULTS_DIR / args.environment_output,
        {
            **environment_info(device),
            "openml": args.openml,
            "scales": scales,
            "random_states": random_states,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "min_validation_delta": args.min_validation_delta,
            "loaded_datasets": sorted(loaded),
            "failed_datasets": sorted(failed),
            "openml_specs": [spec.__dict__ for spec in specs],
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.summary_output}")
    print(f"Wrote {RESULTS_DIR / args.report_output}")


if __name__ == "__main__":
    main()
