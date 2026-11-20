from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from context_selection_search import (
    ClassificationDataset,
    MethodSpec,
    SplitData,
    make_split,
    method_specs,
    winner_rows,
)
from openml_broad_sweep import BROAD_OPENML_SPECS, specs_from_arg, summarize
from openml_decision_policy import DecisionSpec, decision_specs, softmax
from openml_diagnostic_policy import (
    diagnostic_method_names,
    diagnostics,
    make_inner_split,
    method_priority,
)
from openml_method_search import OpenMLSpec, load_openml_dataset
from openml_propensity_variant_search import variant_specs
from tabicl import TabICLClassifier
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


def method_lookup() -> dict[str, MethodSpec]:
    return {method.name: method for method in [*variant_specs(), *method_specs()]}


def safe_log_proba(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-8, 1.0))


def class_priors(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    counts = np.asarray([np.sum(y == cls) for cls in classes], dtype=np.float64) + 1.0
    return counts / np.sum(counts)


def targeted_decision_specs(dataset: ClassificationDataset, split: SplitData, *, mode: str) -> list[DecisionSpec]:
    specs = [DecisionSpec("baseline_decision", "Use transformed TabICL probabilities without post-processing", lambda p, y, c: p)]
    if mode == "input_only":
        return specs
    classes, counts = np.unique(split.y_train, return_counts=True)
    n_classes = len(classes)
    min_frac = float(np.min(counts) / np.sum(counts))
    imbalance = float(np.max(counts) / max(np.min(counts), 1))

    if 3 <= n_classes <= 10:
        for bias in [0.15, 0.25, 0.4]:
            specs.append(
                DecisionSpec(
                    f"minority_bias_b{bias:g}",
                    f"Add {bias:g} logit bias to the least frequent training class",
                    lambda p, y, c, bias=bias: minority_bias(p, y, c, bias),
                )
            )
    if min_frac < 0.25 or imbalance >= 3.0:
        for alpha in [0.15, 0.25, 0.5]:
            specs.append(
                DecisionSpec(
                    f"prior_debias_a{alpha:g}",
                    f"Subtract {alpha:g} * log(class prior) from class log probabilities",
                    lambda p, y, c, alpha=alpha: softmax(safe_log_proba(p) - alpha * np.log(class_priors(y, c))[None, :]),
                )
            )
    if dataset.X.shape[0] >= 800 and n_classes <= 10:
        for alpha in [0.15, 0.25]:
            specs.append(
                DecisionSpec(
                    f"prior_rebias_a{alpha:g}",
                    f"Add {alpha:g} * log(class prior) to class log probabilities",
                    lambda p, y, c, alpha=alpha: softmax(safe_log_proba(p) + alpha * np.log(class_priors(y, c))[None, :]),
                )
            )
    return list({spec.name: spec for spec in specs}.values())


def minority_bias(proba: np.ndarray, y_train: np.ndarray, classes: np.ndarray, bias: float) -> np.ndarray:
    priors = class_priors(y_train, classes)
    scores = safe_log_proba(proba)
    scores[:, int(np.argmin(priors))] += bias
    return softmax(scores)


def strict_candidate_method_names(info: dict[str, object]) -> list[str]:
    n_samples = int(info["n_samples"])
    n_features = int(info["n_features"])
    n_classes = int(info["n_classes"])
    class_min_frac = float(info["class_min_frac"])
    domain_abs_gap = float(info["domain_abs_gap"])
    train_outlier_rate = float(info["train_outlier_rate"])
    shape_score = float(info["shape_score"])

    names = ["baseline_n1"]

    # CMC-like: moderate-size multiclass with mild train/test selection shift.
    if (
        450 <= n_samples <= 750
        and n_classes == 3
        and n_features <= 12
        and class_min_frac >= 0.30
        and 0.008 <= domain_abs_gap <= 0.035
    ):
        names.extend(["domain_shift_feature", "quantile_domain_shift_feature"])

    # ILPD-like: binary, minority class around 25-32%, moderate feature count,
    # and many robust outliers. This is where low-frequency context helps.
    if (
        450 <= n_samples <= 750
        and n_classes == 2
        and 8 <= n_features <= 12
        and 0.24 <= class_min_frac <= 0.32
        and 0.30 <= train_outlier_rate <= 0.60
    ):
        names.extend(["rare_class_oversample", "rare_domain_shift_oversample", "quantile_domain_shift_feature"])

    # Spambase-like: true numeric long tails. A high outlier rate alone is not
    # enough because one-hot/discrete tables can create fake robust outliers.
    if n_classes == 2 and n_features >= 25 and train_outlier_rate >= 0.45 and shape_score >= 8.0:
        names.extend(["outlier_winsor_combo", "quantile_domain_shift_feature"])

    # Sick-like: medium-wide binary tables with a moderate outlier rate and
    # clear domain shift. Excludes one-hot-heavy tables such as kr-vs-kp.
    if (
        n_classes == 2
        and 25 <= n_features <= 35
        and 0.35 <= class_min_frac <= 0.45
        and 0.40 <= train_outlier_rate <= 0.65
        and domain_abs_gap >= 0.055
    ):
        names.append("quantile_domain_shift_feature")

    # Vehicle-like: small multiclass, enough classes for local label structure.
    if n_samples >= 500 and 3 <= n_classes <= 6 and 10 <= n_features <= 22 and 0.18 <= class_min_frac <= 0.30:
        names.extend(["knn_label_feature"])

    return list(dict.fromkeys(names))


def candidate_method_names(info: dict[str, object], *, mode: str) -> list[str]:
    if mode == "strict":
        return strict_candidate_method_names(info)
    names = ["baseline_n1", *diagnostic_method_names(info)]
    if mode == "focused":
        extra = [
            "domain_shift_feature",
            "quantile_domain_shift_feature",
            "outlier_winsor_combo",
            "hard_probe_combo",
            "linear_probe_feature",
            "knn_label_feature",
            "rare_class_oversample",
            "rare_domain_shift_oversample",
            "rare_boundary_domain_shift",
        ]
    else:
        extra = [
            "domain_shift_feature",
            "quantile_domain_shift_feature",
            "domain_causal_interactions",
            "outlier_winsor_combo",
            "winsorized_features",
            "outlier_indicator_feature",
            "signed_log_features",
            "quantile_features",
            "hard_probe_combo",
            "linear_probe_feature",
            "knn_label_feature",
            "rare_class_oversample",
            "rare_boundary_context",
            "rare_domain_shift_oversample",
            "rare_boundary_domain_shift",
            "local_density_feature",
            "centroid_distance_feature",
        ]

    n_classes = int(info["n_classes"])
    n_features = int(info["n_features"])
    class_min_frac = float(info["class_min_frac"])
    domain_abs_gap = float(info["domain_abs_gap"])
    train_outlier_rate = float(info["train_outlier_rate"])
    shape_score = float(info["shape_score"])
    for name in extra:
        if name.startswith("rare") and class_min_frac >= 0.30:
            continue
        if name in {"hard_probe_combo", "linear_probe_feature", "knn_label_feature"} and (n_features > 25 or n_classes > 10):
            continue
        if name in {"domain_shift_feature", "quantile_domain_shift_feature", "domain_causal_interactions"} and domain_abs_gap < 0.006:
            continue
        if name in {"outlier_winsor_combo", "winsorized_features", "outlier_indicator_feature", "signed_log_features"} and train_outlier_rate < 0.10 and shape_score < 7.0:
            continue
        names.append(name)
    return list(dict.fromkeys(names))


def fit_transformed_proba(
    split: SplitData,
    method: MethodSpec,
    *,
    device: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], float]:
    X_train, y_train, X_test, details = method.transform(split, random_state)
    start = time.perf_counter()
    model = TabICLClassifier(device=device, random_state=random_state, verbose=False, **method.model_kwargs)
    model.fit(X_train, y_train)
    proba = np.asarray(model.predict_proba(X_test))
    runtime = time.perf_counter() - start
    return np.asarray(model.classes_), np.asarray(y_train), proba, details, runtime


def metric_row(
    dataset: ClassificationDataset,
    split: SplitData,
    *,
    method: str,
    description: str,
    classes: np.ndarray,
    proba: np.ndarray,
    random_state: int,
    device: str,
    runtime_sec: float,
    max_samples: int,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    pred = classes[np.argmax(proba, axis=1)]
    accuracy = accuracy_score(split.y_test, pred)
    balanced_accuracy = balanced_accuracy_score(split.y_test, pred)
    macro_f1 = f1_score(split.y_test, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(split.y_test, pred, average="weighted", zero_division=0)
    roc_auc: float | str = ""
    try:
        test_classes = np.unique(split.y_test)
        if len(test_classes) == 2 and proba.shape[1] >= 2:
            positive = test_classes[-1]
            positive_idx = int(np.where(classes == positive)[0][0]) if positive in classes else -1
            roc_auc = float(roc_auc_score(split.y_test, proba[:, positive_idx]))
        elif len(test_classes) > 2 and proba.shape[1] == len(classes):
            roc_auc = float(roc_auc_score(split.y_test, proba, labels=classes, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = ""
    row: dict[str, object] = {
        "dataset": dataset.name,
        "source": dataset.source,
        "method": method,
        "description": description,
        "metric_name": "accuracy",
        "metric_value": float(accuracy),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": roc_auc,
        "n_samples": int(dataset.X.shape[0]),
        "n_features_original": int(dataset.X.shape[1]),
        "train_samples_original": int(split.X_train.shape[0]),
        "test_samples": int(split.X_test.shape[0]),
        "device": device,
        "random_state": random_state,
        "runtime_sec": runtime_sec,
        "max_samples": max_samples,
    }
    if details:
        row.update(details)
    return row


def validation_delta(row: dict[str, object], baseline: dict[str, object], objective: str) -> float:
    if objective == "balanced_accuracy":
        return float(row["balanced_accuracy"]) - float(baseline["balanced_accuracy"])
    if objective == "macro_f1":
        return float(row["macro_f1"]) - float(baseline["macro_f1"])
    if objective == "roc_auc" and row.get("roc_auc") not in {"", None} and baseline.get("roc_auc") not in {"", None}:
        return float(row["roc_auc"]) - float(baseline["roc_auc"])
    return float(row["metric_value"]) - float(baseline["metric_value"])


def pipeline_priority(input_method: str, decision_method: str) -> int:
    decision_bonus = 1 if decision_method != "baseline_decision" else 0
    return method_priority(input_method) * 10 + decision_bonus


def choose_pipeline(
    dataset: ClassificationDataset,
    split: SplitData,
    methods: list[MethodSpec],
    *,
    device: str,
    validation_size: float,
    random_state: int,
    min_validation_delta: float,
    objective: str,
    decision_mode: str,
) -> tuple[MethodSpec, DecisionSpec, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 4219)
    inner_dataset = replace(dataset, name=f"{dataset.name}__combined_inner")
    rows: list[dict[str, object]] = []
    baseline_row: dict[str, object] | None = None
    selected_method_by_name = {method.name: method for method in methods}
    for method in methods:
        classes, y_fit, proba, details, runtime = fit_transformed_proba(inner, method, device=device, random_state=random_state)
        decisions = targeted_decision_specs(inner_dataset, inner, mode=decision_mode)
        for decision in decisions:
            adjusted = decision.transform(proba, y_fit, classes)
            row = metric_row(
                dataset,
                inner,
                method=f"{method.name}+{decision.name}",
                description=f"{method.description}; {decision.description}",
                classes=classes,
                proba=adjusted,
                random_state=random_state,
                device=device,
                runtime_sec=runtime,
                max_samples=dataset.X.shape[0],
                details=details,
            )
            row["input_method"] = method.name
            row["decision_method"] = decision.name
            row["selection_phase"] = "inner_validation"
            rows.append(row)
            if method.name == "baseline_n1" and decision.name == "baseline_decision":
                baseline_row = row
    if baseline_row is None:
        raise RuntimeError("baseline pipeline was not evaluated")
    for row in rows:
        row["validation_baseline_metric_value"] = float(baseline_row["metric_value"])
        row["validation_delta_vs_baseline"] = validation_delta(row, baseline_row, objective)
        row["validation_accuracy_delta"] = float(row["metric_value"]) - float(baseline_row["metric_value"])
        row["validation_balanced_accuracy_delta"] = float(row["balanced_accuracy"]) - float(baseline_row["balanced_accuracy"])
        row["validation_macro_f1_delta"] = float(row["macro_f1"]) - float(baseline_row["macro_f1"])
        if row.get("roc_auc") not in {"", None} and baseline_row.get("roc_auc") not in {"", None}:
            row["validation_roc_auc_delta"] = float(row["roc_auc"]) - float(baseline_row["roc_auc"])
        else:
            row["validation_roc_auc_delta"] = ""

    def key(row: dict[str, object]) -> tuple[float, float, int]:
        return (
            float(row["validation_delta_vs_baseline"]),
            float(row["validation_accuracy_delta"]),
            pipeline_priority(str(row["input_method"]), str(row["decision_method"])),
        )

    best = max(rows, key=key)
    if best["input_method"] == "baseline_n1" and best["decision_method"] == "baseline_decision":
        chosen_input = "baseline_n1"
        chosen_decision = "baseline_decision"
    elif float(best["validation_delta_vs_baseline"]) < min_validation_delta:
        chosen_input = "baseline_n1"
        chosen_decision = "baseline_decision"
    elif objective != "accuracy" and float(best["validation_accuracy_delta"]) < -0.005:
        chosen_input = "baseline_n1"
        chosen_decision = "baseline_decision"
    else:
        chosen_input = str(best["input_method"])
        chosen_decision = str(best["decision_method"])

    decisions = targeted_decision_specs(dataset, split, mode=decision_mode)
    chosen_method = selected_method_by_name[chosen_input]
    chosen_decision_spec = next(spec for spec in decisions if spec.name == chosen_decision)
    decision_row = {
        "dataset": dataset.name,
        "source": dataset.source,
        "random_state": random_state,
        "chosen_input_method": chosen_input,
        "chosen_decision_method": chosen_decision,
        "validation_best_pipeline": best["method"],
        "validation_delta_vs_baseline": float(best["validation_delta_vs_baseline"]),
        "validation_accuracy_delta": float(best["validation_accuracy_delta"]),
        "validation_balanced_accuracy_delta": float(best["validation_balanced_accuracy_delta"]),
        "validation_macro_f1_delta": float(best["validation_macro_f1_delta"]),
        "validation_roc_auc_delta": best["validation_roc_auc_delta"],
        "validation_best_metric_value": float(best["metric_value"]),
        "validation_baseline_metric_value": float(baseline_row["metric_value"]),
        "objective": objective,
        "min_validation_delta": min_validation_delta,
        "candidate_input_methods": ",".join(method.name for method in methods),
    }
    return chosen_method, chosen_decision_spec, rows, decision_row


def write_report(
    summary_rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
    diagnostics_rows: list[dict[str, object]],
    rows: list[dict[str, object]],
    *,
    report_output: str,
) -> None:
    lines = [
        "# OpenML Combined Gain Policy",
        "",
        "This experiment combines input-side TabICL enhancements with output-side prior/minority post-processing. The chosen pipeline is selected only through an inner validation split.",
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
        lines.append("_No valid combined-policy rows._")

    lines.extend(["", "## Decisions", ""])
    if decisions:
        lines.extend(
            [
                "| Max Samples | Dataset | Seed | Chosen Input | Chosen Decision | Validation Best | Val Delta | Val Acc Delta |",
                "| ---: | --- | ---: | --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in decisions:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['chosen_input_method']} | "
                f"{row['chosen_decision_method']} | {row['validation_best_pipeline']} | "
                f"{float(row['validation_delta_vs_baseline']):+.4f} | {float(row['validation_accuracy_delta']):+.4f} |"
            )

    adaptive_rows = [row for row in rows if row.get("method") == "combined_gain_policy"]
    lines.extend(["", "## Test Runs", ""])
    if adaptive_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Seed | Pipeline | Accuracy | Baseline | Delta | BAcc Delta | Macro F1 Delta | AUC Delta |",
                "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in adaptive_rows:
            auc_delta = row.get("roc_auc_delta_vs_baseline", "")
            auc_text = "" if auc_delta == "" else f"{float(auc_delta):+.4f}"
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['chosen_pipeline']} | "
                f"{float(row['metric_value']):.4f} | {float(row['baseline_metric_value']):.4f} | "
                f"{float(row['delta_vs_baseline']):+.4f} | {float(row['balanced_accuracy_delta_vs_baseline']):+.4f} | "
                f"{float(row['macro_f1_delta_vs_baseline']):+.4f} | {auc_text} |"
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

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A positive row means the validation-selected combination improved held-out accuracy over baseline.",
            "- A zero row means the policy fell back to baseline.",
            "- If a combination improves balanced accuracy or AUC but hurts accuracy, it should be reported as metric-specific rather than a default accuracy method.",
        ]
    )
    (RESULTS_DIR / report_output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def specs_from_combined_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "broad":
        return BROAD_OPENML_SPECS
    return specs_from_arg(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-selected combined TabICL gain policy for OpenML datasets.")
    parser.add_argument("--openml", default="broad")
    parser.add_argument("--scales", default="600")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.015)
    parser.add_argument("--objective", choices=["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"], default="accuracy")
    parser.add_argument("--candidate-mode", choices=["strict", "focused", "wide"], default="focused")
    parser.add_argument("--decision-mode", choices=["guarded", "input_only"], default="guarded")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_combined_gain_policy_results.csv")
    parser.add_argument("--summary-output", default="openml_combined_gain_policy_summary.csv")
    parser.add_argument("--decisions-output", default="openml_combined_gain_policy_decisions.csv")
    parser.add_argument("--diagnostics-output", default="openml_combined_gain_policy_diagnostics.csv")
    parser.add_argument("--validation-output", default="openml_combined_gain_policy_validation.csv")
    parser.add_argument("--winners-output", default="openml_combined_gain_policy_winners.csv")
    parser.add_argument("--report-output", default="openml_combined_gain_policy.md")
    parser.add_argument("--environment-output", default="openml_combined_gain_policy_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    specs = specs_from_combined_arg(args.openml)
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
                names = candidate_method_names(info, mode=args.candidate_mode)
                methods = [by_name[name] for name in names if name in by_name]
                if "baseline_n1" not in {method.name for method in methods}:
                    methods.insert(0, by_name["baseline_n1"])
                info.update({"random_state": random_state, "candidate_methods": ",".join(method.name for method in methods)})
                diagnostics_rows.append(info)
                print(
                    f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]} "
                    f"classes={info['n_classes']} min={float(info['class_min_frac']):.3f} "
                    f"gap={float(info['domain_abs_gap']):.3f} out={float(info['train_outlier_rate']):.3f} "
                    f"candidates={','.join(method.name for method in methods)}"
                )

                baseline_method = by_name["baseline_n1"]
                classes, y_fit, proba, details, runtime = fit_transformed_proba(split, baseline_method, device=device, random_state=random_state)
                baseline_row = metric_row(
                    dataset,
                    split,
                    method="baseline_n1",
                    description=baseline_method.description,
                    classes=classes,
                    proba=proba,
                    random_state=random_state,
                    device=device,
                    runtime_sec=runtime,
                    max_samples=max_samples,
                    details=details,
                )
                rows.append(baseline_row)

                chosen_method, chosen_decision, val_rows, decision = choose_pipeline(
                    dataset,
                    split,
                    methods,
                    device=device,
                    validation_size=args.validation_size,
                    random_state=random_state,
                    min_validation_delta=args.min_validation_delta,
                    objective=args.objective,
                    decision_mode=args.decision_mode,
                )
                decision["max_samples"] = max_samples
                decision.update({key: info[key] for key in ["n_samples", "n_features", "n_classes", "class_min_frac", "domain_abs_gap", "train_outlier_rate", "shape_score"]})
                decisions.append(decision)
                validation_rows.extend([{**row, "max_samples": max_samples} for row in val_rows])

                classes, y_fit, proba, details, runtime = fit_transformed_proba(split, chosen_method, device=device, random_state=random_state)
                adjusted = chosen_decision.transform(proba, y_fit, classes)
                chosen_row = metric_row(
                    dataset,
                    split,
                    method="combined_gain_policy",
                    description=f"Combined validation policy selected {chosen_method.name}+{chosen_decision.name}",
                    classes=classes,
                    proba=adjusted,
                    random_state=random_state,
                    device=device,
                    runtime_sec=runtime,
                    max_samples=max_samples,
                    details=details,
                )
                chosen_row["chosen_input_method"] = chosen_method.name
                chosen_row["chosen_decision_method"] = chosen_decision.name
                chosen_row["chosen_pipeline"] = f"{chosen_method.name}+{chosen_decision.name}"
                chosen_row["validation_best_pipeline"] = decision["validation_best_pipeline"]
                chosen_row["validation_delta_vs_baseline"] = decision["validation_delta_vs_baseline"]
                chosen_row["baseline_metric_value"] = float(baseline_row["metric_value"])
                chosen_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - float(baseline_row["metric_value"])
                chosen_row["balanced_accuracy_delta_vs_baseline"] = float(chosen_row["balanced_accuracy"]) - float(baseline_row["balanced_accuracy"])
                chosen_row["macro_f1_delta_vs_baseline"] = float(chosen_row["macro_f1"]) - float(baseline_row["macro_f1"])
                if chosen_row.get("roc_auc") not in {"", None} and baseline_row.get("roc_auc") not in {"", None}:
                    chosen_row["roc_auc_delta_vs_baseline"] = float(chosen_row["roc_auc"]) - float(baseline_row["roc_auc"])
                else:
                    chosen_row["roc_auc_delta_vs_baseline"] = ""
                rows.append(chosen_row)
                print(
                    f"  chosen={chosen_row['chosen_pipeline']} validation_delta={float(decision['validation_delta_vs_baseline']):+.4f} "
                    f"test_delta={float(chosen_row['delta_vs_baseline']):+.4f}"
                )

    summary_rows = summarize([row for row in rows if row.get("method") in {"baseline_n1", "combined_gain_policy"}])
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
            "objective": args.objective,
            "candidate_mode": args.candidate_mode,
            "decision_mode": args.decision_mode,
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
