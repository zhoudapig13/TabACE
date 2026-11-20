from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from context_selection_search import ClassificationDataset, SplitData, make_split, winner_rows
from openml_broad_sweep import BROAD_OPENML_SPECS, specs_from_arg, summarize
from openml_method_search import OpenMLSpec, load_openml_dataset
from tabicl import TabICLClassifier
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


@dataclass(frozen=True)
class DecisionSpec:
    name: str
    description: str
    transform: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


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


def safe_log_proba(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-8, 1.0))


def class_priors(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    counts = np.asarray([np.sum(y == cls) for cls in classes], dtype=np.float64)
    counts = counts + 1.0
    return counts / np.sum(counts)


def softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / np.sum(exp, axis=1, keepdims=True)


def decision_specs() -> list[DecisionSpec]:
    specs = [DecisionSpec("baseline_decision", "Use TabICL probabilities without post-processing", lambda p, y, c: p)]
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        specs.append(
            DecisionSpec(
                f"prior_debias_a{alpha:g}",
                f"Subtract {alpha:g} * log(class prior) from class log probabilities",
                lambda p, y, c, alpha=alpha: softmax(safe_log_proba(p) - alpha * np.log(class_priors(y, c))[None, :]),
            )
        )
    for alpha in [0.25, 0.5]:
        specs.append(
            DecisionSpec(
                f"prior_rebias_a{alpha:g}",
                f"Add {alpha:g} * log(class prior) to class log probabilities",
                lambda p, y, c, alpha=alpha: softmax(safe_log_proba(p) + alpha * np.log(class_priors(y, c))[None, :]),
            )
        )
    for bias in [0.25, 0.5, 0.75, 1.0]:
        specs.append(
            DecisionSpec(
                f"minority_bias_b{bias:g}",
                f"Add {bias:g} logit bias to the least frequent training class",
                lambda p, y, c, bias=bias: minority_bias(p, y, c, bias),
            )
        )
    return specs


def guarded_decision_specs(dataset: ClassificationDataset, policy: str) -> list[DecisionSpec]:
    specs = decision_specs()
    if policy == "free":
        return specs
    n_classes = len(np.unique(dataset.y))
    if n_classes < 3 or n_classes > 10:
        return [spec for spec in specs if spec.name == "baseline_decision"]
    allowed_prefixes = ("baseline_decision", "prior_debias", "minority_bias")
    return [spec for spec in specs if spec.name == "baseline_decision" or spec.name.startswith(allowed_prefixes)]


def minority_bias(proba: np.ndarray, y_train: np.ndarray, classes: np.ndarray, bias: float) -> np.ndarray:
    priors = class_priors(y_train, classes)
    scores = safe_log_proba(proba)
    scores[:, int(np.argmin(priors))] += bias
    return softmax(scores)


def fit_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    device: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    model = TabICLClassifier(device=device, random_state=random_state, verbose=False, n_estimators=1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    runtime = time.perf_counter() - start
    return np.asarray(model.classes_), np.asarray(proba), runtime


def row_metrics(
    dataset: ClassificationDataset,
    split: SplitData,
    *,
    method: str,
    description: str,
    proba: np.ndarray,
    classes: np.ndarray,
    runtime_sec: float,
    random_state: int,
    device: str,
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
        "n_classes": int(len(classes)),
        "device": device,
        "random_state": random_state,
        "runtime_sec": runtime_sec,
        "max_samples": max_samples,
    }
    if details:
        row.update(details)
    return row


def validation_score(row: dict[str, object], baseline: dict[str, object], *, objective: str) -> float:
    if objective == "balanced_accuracy":
        return float(row["balanced_accuracy"]) - float(baseline["balanced_accuracy"])
    if objective == "macro_f1":
        return float(row["macro_f1"]) - float(baseline["macro_f1"])
    return float(row["metric_value"]) - float(baseline["metric_value"])


def choose_decision(
    dataset: ClassificationDataset,
    split: SplitData,
    *,
    device: str,
    validation_size: float,
    random_state: int,
    min_validation_delta: float,
    objective: str,
    policy: str,
) -> tuple[DecisionSpec, list[dict[str, object]], dict[str, object]]:
    inner = make_inner_split(split, validation_size=validation_size, random_state=random_state + 3191)
    classes, val_proba, runtime = fit_proba(inner.X_train, inner.y_train, inner.X_test, device=device, random_state=random_state)
    specs = guarded_decision_specs(dataset, policy)
    rows = []
    for spec in specs:
        adjusted = spec.transform(val_proba, inner.y_train, classes)
        rows.append(
            row_metrics(
                dataset,
                inner,
                method=spec.name,
                description=spec.description,
                proba=adjusted,
                classes=classes,
                runtime_sec=runtime,
                random_state=random_state,
                device=device,
                max_samples=int(dataset.X.shape[0]),
                details={"selection_phase": "inner_validation"},
            )
        )
    baseline = next(row for row in rows if row["method"] == "baseline_decision")
    for row in rows:
        row["validation_delta_vs_baseline"] = float(row["metric_value"]) - float(baseline["metric_value"])
        row["validation_balanced_accuracy_delta"] = float(row["balanced_accuracy"]) - float(baseline["balanced_accuracy"])
        row["validation_macro_f1_delta"] = float(row["macro_f1"]) - float(baseline["macro_f1"])

    best = max(rows, key=lambda row: (validation_score(row, baseline, objective=objective), float(row["metric_value"])))
    best_delta = validation_score(best, baseline, objective=objective)
    if best["method"] == "baseline_decision" or best_delta < min_validation_delta:
        chosen_name = "baseline_decision"
    else:
        chosen_name = str(best["method"])
    chosen = next(spec for spec in specs if spec.name == chosen_name)
    decision = {
        "dataset": dataset.name,
        "source": dataset.source,
        "random_state": random_state,
        "chosen_method": chosen.name,
        "validation_best_method": best["method"],
        "validation_objective": objective,
        "validation_objective_delta": float(best_delta),
        "validation_delta_vs_baseline": float(best["validation_delta_vs_baseline"]),
        "validation_balanced_accuracy_delta": float(best["validation_balanced_accuracy_delta"]),
        "validation_macro_f1_delta": float(best["validation_macro_f1_delta"]),
        "min_validation_delta": min_validation_delta,
        "policy": policy,
        "candidate_methods": ",".join(spec.name for spec in specs),
    }
    return chosen, rows, decision


def write_report(
    summary_rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
    rows: list[dict[str, object]],
    *,
    report_output: str,
) -> None:
    lines = [
        "# OpenML Decision Policy",
        "",
        "This experiment keeps TabICL inputs fixed and uses inner validation to choose probability/logit post-processing for class prior and minority-class decision bias.",
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
                "| Max Samples | Dataset | Seed | Chosen | Validation Best | Objective Delta | Accuracy Delta | BAcc Delta |",
                "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in decisions:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['chosen_method']} | "
                f"{row['validation_best_method']} | {float(row['validation_objective_delta']):+.4f} | "
                f"{float(row['validation_delta_vs_baseline']):+.4f} | {float(row['validation_balanced_accuracy_delta']):+.4f} |"
            )

    adaptive_rows = [row for row in rows if row.get("method") == "adaptive_decision_policy"]
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
            "- Positive rows mean probability post-processing transferred from inner validation to the held-out test split.",
            "- Negative rows show where validation prefers a threshold or prior correction that does not generalize.",
            "- This branch is intentionally separate from input/context transforms so both can be compared later.",
        ]
    )
    (RESULTS_DIR / report_output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def specs_from_decision_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "broad":
        return BROAD_OPENML_SPECS
    return specs_from_arg(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-gated probability post-processing policy for TabICL.")
    parser.add_argument("--openml", default="broad")
    parser.add_argument("--scales", default="600")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.01)
    parser.add_argument("--objective", choices=["accuracy", "balanced_accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--policy", choices=["guarded", "free"], default="guarded")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_decision_policy_results.csv")
    parser.add_argument("--summary-output", default="openml_decision_policy_summary.csv")
    parser.add_argument("--decisions-output", default="openml_decision_policy_decisions.csv")
    parser.add_argument("--validation-output", default="openml_decision_policy_validation.csv")
    parser.add_argument("--winners-output", default="openml_decision_policy_winners.csv")
    parser.add_argument("--report-output", default="openml_decision_policy.md")
    parser.add_argument("--environment-output", default="openml_decision_policy_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    specs = specs_from_decision_arg(args.openml)
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
                print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]}")
                chosen, val_rows, decision = choose_decision(
                    dataset,
                    split,
                    device=device,
                    validation_size=args.validation_size,
                    random_state=random_state,
                    min_validation_delta=args.min_validation_delta,
                    objective=args.objective,
                    policy=args.policy,
                )
                decision["max_samples"] = max_samples
                decisions.append(decision)
                validation_rows.extend([{**row, "max_samples": max_samples} for row in val_rows])

                classes, test_proba, runtime = fit_proba(split.X_train, split.y_train, split.X_test, device=device, random_state=random_state)
                baseline_spec = next(spec for spec in decision_specs() if spec.name == "baseline_decision")
                baseline_proba = baseline_spec.transform(test_proba, split.y_train, classes)
                baseline_row = row_metrics(
                    dataset,
                    split,
                    method="baseline_n1",
                    description=baseline_spec.description,
                    proba=baseline_proba,
                    classes=classes,
                    runtime_sec=runtime,
                    random_state=random_state,
                    device=device,
                    max_samples=max_samples,
                )
                rows.append(baseline_row)

                chosen_proba = chosen.transform(test_proba, split.y_train, classes)
                chosen_row = row_metrics(
                    dataset,
                    split,
                    method=chosen.name,
                    description=chosen.description,
                    proba=chosen_proba,
                    classes=classes,
                    runtime_sec=runtime,
                    random_state=random_state,
                    device=device,
                    max_samples=max_samples,
                )
                adaptive_row = dict(chosen_row)
                adaptive_row["method"] = "adaptive_decision_policy"
                adaptive_row["chosen_method"] = chosen.name
                adaptive_row["description"] = f"Decision policy selected {chosen.name}"
                adaptive_row["baseline_metric_value"] = float(baseline_row["metric_value"])
                adaptive_row["delta_vs_baseline"] = float(chosen_row["metric_value"]) - float(baseline_row["metric_value"])
                adaptive_row["validation_best_method"] = decision["validation_best_method"]
                adaptive_row["validation_objective_delta"] = decision["validation_objective_delta"]
                rows.append(adaptive_row)
                print(
                    f"  chosen={chosen.name} validation_objective_delta={float(decision['validation_objective_delta']):+.4f} "
                    f"test_delta={float(adaptive_row['delta_vs_baseline']):+.4f}"
                )

    summary_rows = summarize([row for row in rows if row.get("method") in {"baseline_n1", "adaptive_decision_policy"}])
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
            "objective": args.objective,
            "policy": args.policy,
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
