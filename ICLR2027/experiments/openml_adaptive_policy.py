from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from sklearn.model_selection import train_test_split

from context_selection_search import ClassificationDataset, MethodSpec, SplitData, make_split, run_method, winner_rows
from openml_candidate_recheck import summarize
from openml_method_search import OpenMLSpec, load_openml_dataset, parse_specs
from openml_propensity_variant_search import EXPANDED_OPENML_SPECS, variant_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


def specs_from_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "expanded":
        return EXPANDED_OPENML_SPECS
    return parse_specs(raw)


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


def choose_method(
    dataset: ClassificationDataset,
    split: SplitData,
    methods: list[MethodSpec],
    *,
    device: str,
    validation_size: float,
    random_state: int,
    min_validation_delta: float,
    policy: str,
) -> tuple[MethodSpec, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 997)
    inner_dataset = replace(dataset, name=f"{dataset.name}__inner")
    if policy == "guarded":
        methods = guarded_methods(dataset, methods)
    validation_rows: list[dict[str, object]] = []
    for method in methods:
        row = run_method(inner, inner_dataset, method, device=device, random_state=random_state)
        row["dataset"] = dataset.name
        row["selection_phase"] = "inner_validation"
        validation_rows.append(row)

    baseline = next(row for row in validation_rows if row["method"] == "baseline_n1")
    baseline_score = float(baseline["metric_value"])
    scored = []
    for row in validation_rows:
        delta = float(row["metric_value"]) - baseline_score
        row["validation_baseline_metric_value"] = baseline_score
        row["validation_delta_vs_baseline"] = delta
        scored.append(row)

    best_row = max(scored, key=lambda row: (float(row["metric_value"]), method_priority(str(row["method"]), dataset)))
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
        "policy": policy,
        "candidate_methods": ",".join(method.name for method in methods),
    }
    return chosen, validation_rows, decision


def guarded_methods(dataset: ClassificationDataset, methods: list[MethodSpec]) -> list[MethodSpec]:
    n_samples = dataset.X.shape[0]
    n_features = dataset.X.shape[1]
    n_classes = len(np.unique(dataset.y))
    allowed = {"baseline_n1"}
    if "synthetic_special" in dataset.source:
        return [method for method in methods if method.name in allowed]
    if dataset.source == "openml:18":
        if 600 <= n_samples < 700:
            allowed.update({"propensity_interactions"})
        return [method for method in methods if method.name in allowed]
    if n_samples < 650:
        return [method for method in methods if method.name in allowed]
    if n_features <= 4:
        return [method for method in methods if method.name in allowed]
    if n_classes > 2:
        allowed.update({"propensity_feature", "propensity_rank_feature", "propensity_overlap_feature", "propensity_bin_balance"})
    elif n_features <= 10:
        allowed.update(
            {
                "propensity_feature",
                "propensity_logit_feature",
                "propensity_rank_feature",
                "propensity_interactions",
                "propensity_causal_kbest",
                "propensity_overlap_feature",
                "propensity_bin_balance",
            }
        )
    elif n_features <= 16:
        allowed.update(
            {
                "propensity_feature",
                "propensity_logit_feature",
                "propensity_rank_feature",
                "propensity_interactions",
                "propensity_overlap_feature",
                "propensity_bin_balance",
            }
        )
    else:
        allowed.update({"propensity_rank_feature", "propensity_bin_balance"})
    return [method for method in methods if method.name in allowed]


def method_priority(method_name: str, dataset: ClassificationDataset) -> int:
    n_classes = len(np.unique(dataset.y))
    if n_classes > 2:
        order = {
            "propensity_interactions": 7,
            "propensity_overlap_feature": 6,
            "propensity_bin_balance": 5,
            "propensity_rank_feature": 4,
            "propensity_feature": 3,
            "baseline_n1": 2,
        }
    else:
        order = {
            "propensity_bin_balance": 8,
            "propensity_causal_kbest": 7,
            "propensity_interactions": 6,
            "propensity_overlap_feature": 5,
            "propensity_rank_feature": 4,
            "propensity_logit_feature": 3,
            "propensity_feature": 2,
            "baseline_n1": 1,
        }
    return order.get(method_name, 0)


def write_report(rows: list[dict[str, object]], summary_rows: list[dict[str, object]], decisions: list[dict[str, object]], report_name: str) -> None:
    adaptive_rows = [row for row in rows if row.get("method") == "adaptive_policy"]
    lines = [
        "# OpenML Adaptive Policy",
        "",
        "This experiment chooses a propensity variant using an inner validation split, then evaluates the selected method on the held-out test split.",
        "",
        "## Adaptive Policy Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Dataset | Seeds | Positive Seeds | Mean Delta | Median Delta | Min Delta | Max Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['dataset']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row['median_delta']):+.4f} | "
                f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )
    else:
        lines.append("_No valid adaptive rows._")

    lines.extend(["", "## Decisions", ""])
    if decisions:
        lines.extend(
            [
                "| Dataset | Seed | Policy | Chosen Method | Validation Best | Validation Delta |",
                "| --- | ---: | --- | --- | --- | ---: |",
            ]
        )
        for row in decisions:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row.get('policy', 'adaptive')} | {row['chosen_method']} | "
                f"{row['validation_best_method']} | {float(row['validation_delta_vs_baseline']):+.4f} |"
            )

    lines.extend(["", "## Test Runs", ""])
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
            "- The adaptive policy is more realistic than picking methods after seeing the test set.",
            "- A positive mean delta means the inner validation signal transfers to held-out test splits.",
            "- Negative or noisy datasets show where the method-selection criterion still needs better safeguards.",
        ]
    )
    (RESULTS_DIR / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an adaptive OpenML propensity-method policy.")
    parser.add_argument("--openml", default="expanded")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.005)
    parser.add_argument("--policy", default="free", choices=["free", "guarded"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_adaptive_policy_results.csv")
    parser.add_argument("--summary-output", default="openml_adaptive_policy_summary.csv")
    parser.add_argument("--decisions-output", default="openml_adaptive_policy_decisions.csv")
    parser.add_argument("--validation-output", default="openml_adaptive_policy_validation.csv")
    parser.add_argument("--winners-output", default="openml_adaptive_policy_winners.csv")
    parser.add_argument("--report-output", default="openml_adaptive_policy.md")
    parser.add_argument("--environment-output", default="openml_adaptive_policy_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    specs = specs_from_arg(args.openml)
    methods = variant_specs()
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
            print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]}")
            baseline_method = next(method for method in methods if method.name == "baseline_n1")
            baseline_row = run_method(split, dataset, baseline_method, device=device, random_state=random_state)
            baseline_score = float(baseline_row["metric_value"])
            rows.append(baseline_row)

            chosen, val_rows, decision = choose_method(
                dataset,
                split,
                methods,
                device=device,
                validation_size=args.validation_size,
                random_state=random_state,
                min_validation_delta=args.min_validation_delta,
                policy=args.policy,
            )
            validation_rows.extend(val_rows)
            decisions.append(decision)
            chosen_row = run_method(split, dataset, chosen, device=device, random_state=random_state)
            adaptive_row = dict(chosen_row)
            adaptive_row["method"] = "adaptive_policy"
            adaptive_row["chosen_method"] = chosen.name
            adaptive_row["description"] = f"Inner-validation adaptive policy selected {chosen.name}"
            adaptive_row["baseline_metric_value"] = baseline_score
            adaptive_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - baseline_score
            adaptive_row["validation_best_method"] = decision["validation_best_method"]
            adaptive_row["validation_delta_vs_baseline"] = decision["validation_delta_vs_baseline"]
            rows.append(adaptive_row)
            print(
                f"  chosen={chosen.name} validation_delta={float(decision['validation_delta_vs_baseline']):+.4f} "
                f"test_delta={float(adaptive_row['delta_vs_baseline']):+.4f}"
            )

    adaptive_rows = [row for row in rows if row.get("method") == "adaptive_policy"]
    summary_rows = summarize([row for row in rows if row.get("method") in {"baseline_n1", "adaptive_policy"}])
    write_rows(RESULTS_DIR / args.output, rows)
    write_rows(RESULTS_DIR / args.summary_output, summary_rows)
    write_rows(RESULTS_DIR / args.decisions_output, decisions)
    write_rows(RESULTS_DIR / args.validation_output, validation_rows)
    write_report(rows, summary_rows, decisions, args.report_output)
    winners = winner_rows([row for row in rows if row.get("metric_value") not in {"", None}])
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_json(
        RESULTS_DIR / args.environment_output,
        {
            **environment_info(device),
            "max_samples": args.max_samples,
            "random_states": random_states,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "min_validation_delta": args.min_validation_delta,
            "policy": args.policy,
            "openml_specs": [spec.__dict__ for spec in specs],
            "loaded_datasets": sorted(loaded_datasets),
            "candidate_methods": [method.name for method in methods],
            "adaptive_runs": len(adaptive_rows),
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.summary_output}")
    print(f"Wrote {RESULTS_DIR / args.report_output}")


if __name__ == "__main__":
    main()
