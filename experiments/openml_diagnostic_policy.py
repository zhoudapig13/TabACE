from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import numpy as np
from sklearn.model_selection import train_test_split

from context_selection_search import (
    ClassificationDataset,
    MethodSpec,
    SplitData,
    estimate_domain_probability,
    make_split,
    method_specs,
    run_method,
    winner_rows,
)
from openml_broad_sweep import BROAD_OPENML_SPECS, specs_from_arg, summarize
from openml_method_search import OpenMLSpec, load_openml_dataset
from openml_propensity_variant_search import variant_specs
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
    return {method.name: method for method in [*variant_specs(), *method_specs()]}


def outlier_rates(split: SplitData) -> tuple[float, float]:
    median = np.nanmedian(split.X_train, axis=0)
    q1 = np.nanquantile(split.X_train, 0.25, axis=0)
    q3 = np.nanquantile(split.X_train, 0.75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    train_outlier = np.any(np.abs((split.X_train - median) / iqr) > 3.0, axis=1)
    test_outlier = np.any(np.abs((split.X_test - median) / iqr) > 3.0, axis=1)
    return float(np.mean(train_outlier)), float(np.mean(test_outlier))


def feature_shape_score(X: np.ndarray) -> float:
    q05 = np.nanquantile(X, 0.05, axis=0)
    q25 = np.nanquantile(X, 0.25, axis=0)
    q50 = np.nanquantile(X, 0.50, axis=0)
    q75 = np.nanquantile(X, 0.75, axis=0)
    q95 = np.nanquantile(X, 0.95, axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    tail = (q95 - q05) / iqr
    asym = np.abs((q75 + q25 - 2 * q50) / iqr)
    score = np.nanmedian(np.clip(tail, 0, 50)) + np.nanmedian(np.clip(asym, 0, 20))
    return float(score)


def diagnostics(dataset: ClassificationDataset, split: SplitData, max_samples: int) -> dict[str, object]:
    classes, counts = np.unique(dataset.y, return_counts=True)
    train_domain_prob, test_domain_prob, domain_details = estimate_domain_probability(split.X_train, split.X_test)
    train_outlier_rate, test_outlier_rate = outlier_rates(split)
    class_min_frac = float(np.min(counts) / np.sum(counts))
    class_imbalance = float(np.max(counts) / max(np.min(counts), 1))
    return {
        "dataset": dataset.name,
        "source": dataset.source,
        "max_samples": max_samples,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "n_classes": int(len(classes)),
        "class_min_frac": class_min_frac,
        "class_imbalance": class_imbalance,
        "train_outlier_rate": train_outlier_rate,
        "test_outlier_rate": test_outlier_rate,
        "outlier_rate_gap": float(test_outlier_rate - train_outlier_rate),
        "shape_score": feature_shape_score(split.X_train),
        "domain_gap": float(domain_details["domain_gap"]),
        "domain_abs_gap": float(abs(domain_details["domain_gap"])),
        "domain_train_mean": float(np.mean(train_domain_prob)),
        "domain_test_mean": float(np.mean(test_domain_prob)),
    }


def diagnostic_method_names(info: dict[str, object]) -> list[str]:
    names = ["baseline_n1"]
    n_samples = int(info["n_samples"])
    n_features = int(info["n_features"])
    n_classes = int(info["n_classes"])
    class_min_frac = float(info["class_min_frac"])
    domain_abs_gap = float(info["domain_abs_gap"])
    train_outlier_rate = float(info["train_outlier_rate"])
    outlier_rate_gap = abs(float(info["outlier_rate_gap"]))
    shape_score = float(info["shape_score"])

    if 450 <= n_samples <= 750 and 0.008 <= domain_abs_gap <= 0.045 and n_features <= 20 and n_classes <= 10:
        if n_classes == 2 and class_min_frac >= 0.35 and train_outlier_rate < 0.15:
            pass
        else:
            names.extend(["domain_shift_feature", "quantile_domain_shift_feature"])
            if n_classes >= 3:
                names.append("quantile_features")
    if 0.15 <= train_outlier_rate <= 0.55 and n_features <= 25:
        names.extend(["outlier_winsor_combo", "winsorized_features", "quantile_features"])
    if shape_score >= 8.0:
        names.extend(["signed_log_features", "outlier_winsor_combo"])
    if n_samples >= 500 and 6 <= n_features <= 16 and n_classes == 2 and class_min_frac < 0.35:
        names.extend(["linear_probe_feature", "knn_label_feature"])
    if n_samples >= 500 and class_min_frac < 0.20:
        names.extend(["rare_class_oversample", "rare_boundary_context"])
    if outlier_rate_gap >= 0.05 and n_features <= 25:
        names.append("outlier_indicator_feature")
    return list(dict.fromkeys(names))


def method_priority(name: str) -> int:
    order = {
        "quantile_domain_shift_feature": 10,
        "domain_shift_feature": 9,
        "linear_probe_feature": 8,
        "knn_label_feature": 7,
        "outlier_winsor_combo": 6,
        "quantile_features": 5,
        "winsorized_features": 4,
        "signed_log_features": 3,
        "outlier_indicator_feature": 2,
        "rare_class_oversample": 1,
        "rare_boundary_context": 1,
        "baseline_n1": 0,
    }
    return order.get(name, 0)


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
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 2473)
    inner_dataset = replace(dataset, name=f"{dataset.name}__diagnostic_inner")
    validation_rows = []
    for method in methods:
        row = run_method(inner, inner_dataset, method, device=device, random_state=random_state)
        row["dataset"] = dataset.name
        row["selection_phase"] = "inner_validation"
        validation_rows.append(row)

    baseline = next(row for row in validation_rows if row["method"] == "baseline_n1")
    baseline_score = float(baseline["metric_value"])
    baseline_bacc = float(baseline.get("balanced_accuracy") or baseline_score)
    baseline_f1 = float(baseline.get("macro_f1") or baseline_score)
    for row in validation_rows:
        row["validation_baseline_metric_value"] = baseline_score
        row["validation_delta_vs_baseline"] = float(row["metric_value"]) - baseline_score
        row["validation_balanced_accuracy_delta"] = float(row.get("balanced_accuracy") or 0.0) - baseline_bacc
        row["validation_macro_f1_delta"] = float(row.get("macro_f1") or 0.0) - baseline_f1

    def validation_key(row: dict[str, object]) -> tuple[float, float, int]:
        return (
            float(row["validation_delta_vs_baseline"]),
            float(row.get("validation_balanced_accuracy_delta") or 0.0),
            method_priority(str(row["method"])),
        )

    best_row = max(validation_rows, key=validation_key)
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
        "validation_balanced_accuracy_delta": float(best_row.get("validation_balanced_accuracy_delta") or 0.0),
        "validation_macro_f1_delta": float(best_row.get("validation_macro_f1_delta") or 0.0),
        "min_validation_delta": min_validation_delta,
        "candidate_methods": ",".join(method.name for method in methods),
    }
    return chosen, validation_rows, decision


def write_report(
    summary_rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
    diagnostics_rows: list[dict[str, object]],
    rows: list[dict[str, object]],
    *,
    report_output: str,
) -> None:
    lines = [
        "# OpenML Diagnostic Policy",
        "",
        "This experiment opens TabICL enhancement candidates from dataset diagnostics rather than from dataset names, then uses an inner validation split before testing.",
        "",
        "## Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Seeds | Positive | Mean Delta | BAcc Delta | Macro F1 Delta | AUC Delta | Min Delta | Max Delta |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row.get('mean_balanced_accuracy_delta', 0.0)):+.4f} | "
                f"{float(row.get('mean_macro_f1_delta', 0.0)):+.4f} | {float(row.get('mean_roc_auc_delta', 0.0)):+.4f} | "
                f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
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

    lines.extend(["", "## Diagnostics", ""])
    if diagnostics_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Seed | Classes | Min Class | Domain Gap | Outlier Rate | Shape | Candidates |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in diagnostics_rows[:80]:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['n_classes']} | "
                f"{float(row['class_min_frac']):.3f} | {float(row['domain_abs_gap']):.3f} | "
                f"{float(row['train_outlier_rate']):.3f} | {float(row['shape_score']):.2f} | {row['candidate_methods']} |"
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
            "- Positive rows mean the diagnostic gate plus inner validation improved held-out accuracy over baseline.",
            "- Zero rows mean the policy safely returned baseline.",
            "- Negative rows expose diagnostics or validation thresholds that are too permissive.",
        ]
    )
    (RESULTS_DIR / report_output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def specs_from_diagnostic_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "broad":
        return BROAD_OPENML_SPECS
    return specs_from_arg(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic-gated TabICL method policy for OpenML datasets.")
    parser.add_argument("--openml", default="broad")
    parser.add_argument("--scales", default="600,900")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.015)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_diagnostic_policy_results.csv")
    parser.add_argument("--summary-output", default="openml_diagnostic_policy_summary.csv")
    parser.add_argument("--decisions-output", default="openml_diagnostic_policy_decisions.csv")
    parser.add_argument("--diagnostics-output", default="openml_diagnostic_policy_diagnostics.csv")
    parser.add_argument("--validation-output", default="openml_diagnostic_policy_validation.csv")
    parser.add_argument("--winners-output", default="openml_diagnostic_policy_winners.csv")
    parser.add_argument("--report-output", default="openml_diagnostic_policy.md")
    parser.add_argument("--environment-output", default="openml_diagnostic_policy_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    specs = specs_from_diagnostic_arg(args.openml)
    by_name = method_lookup()
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
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
                info = diagnostics(dataset, split, max_samples)
                method_names = diagnostic_method_names(info)
                methods = [by_name[name] for name in method_names if name in by_name]
                if "baseline_n1" not in {method.name for method in methods}:
                    methods.insert(0, by_name["baseline_n1"])
                info.update({"random_state": random_state, "candidate_methods": ",".join(method.name for method in methods)})
                diagnostics_rows.append(info)
                baseline = by_name["baseline_n1"]
                print(
                    f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]} "
                    f"gap={float(info['domain_abs_gap']):.3f} out={float(info['train_outlier_rate']):.3f} "
                    f"candidates={','.join(method.name for method in methods)}"
                )
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
                decision.update({key: info[key] for key in ["n_samples", "n_features", "n_classes", "domain_abs_gap", "train_outlier_rate", "shape_score"]})
                validation_rows.extend([{**row, "max_samples": max_samples} for row in val_rows])
                decisions.append(decision)
                chosen_row = run_method(split, dataset, chosen, device=device, random_state=random_state)
                adaptive_row = dict(chosen_row)
                adaptive_row["max_samples"] = max_samples
                adaptive_row["method"] = "adaptive_policy"
                adaptive_row["chosen_method"] = chosen.name
                adaptive_row["description"] = f"Diagnostic-gated policy selected {chosen.name}"
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
    write_rows(RESULTS_DIR / args.diagnostics_output, diagnostics_rows)
    write_rows(RESULTS_DIR / args.validation_output, validation_rows)
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["max_samples", "dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(summary_rows, decisions, diagnostics_rows, rows, report_output=args.report_output)
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
