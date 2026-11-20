from __future__ import annotations

import argparse
import csv
import hashlib
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from tabicl import TabICLClassifier

from context_selection_search import (
    ClassificationDataset,
    SplitData,
    causal_proxy_weights,
    estimate_propensity,
    load_datasets,
    make_split,
)
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


@dataclass(frozen=True)
class QueryMethod:
    name: str
    description: str
    alpha: float = 1.0
    beta: float = 0.0
    balanced: bool = False
    random_context: bool = False
    stratified_random: bool = False
    causal_weighted: bool = False
    add_propensity_feature: bool = False


def query_methods() -> list[QueryMethod]:
    return [
        QueryMethod("full_context", "Use all training rows as TabICL context"),
        QueryMethod("random_context", "Random limited-size context baseline", random_context=True),
        QueryMethod("balanced_random_context", "Random limited-size context with balanced R=0/R=1", stratified_random=True),
        QueryMethod("knn_context", "Per-query nearest context by standardized feature similarity"),
        QueryMethod(
            "propensity_score_context",
            "Per-query context score alpha*feature_similarity - beta*propensity_distance",
            alpha=1.0,
            beta=0.8,
        ),
        QueryMethod(
            "balanced_propensity_context",
            "Per-query propensity-aware context with balanced R=0/R=1 selection",
            alpha=1.0,
            beta=0.8,
            balanced=True,
        ),
        QueryMethod(
            "causal_weighted_context",
            "Per-query nearest context using supervised causal-proxy feature weights",
            causal_weighted=True,
        ),
        QueryMethod(
            "causal_propensity_context",
            "Per-query causal-weighted feature similarity plus propensity matching",
            alpha=1.0,
            beta=0.8,
            causal_weighted=True,
        ),
        QueryMethod(
            "propensity_feature_context",
            "Use all rows but append estimated propensity and selection flag as features",
            add_propensity_feature=True,
        ),
    ]


def subset_dataset(dataset: ClassificationDataset, max_samples: int, random_state: int) -> ClassificationDataset:
    if dataset.X.shape[0] <= max_samples:
        return dataset
    rng = np.random.default_rng(random_state)
    indices = rng.choice(dataset.X.shape[0], size=max_samples, replace=False)
    return ClassificationDataset(
        name=dataset.name,
        X=dataset.X[indices],
        y=dataset.y[indices],
        source=dataset.source,
    )


def selected_indices_for_query(
    split: SplitData,
    x_query_scaled: np.ndarray,
    p_query: float,
    *,
    method: QueryMethod,
    scaler: StandardScaler,
    context_size: int,
) -> np.ndarray:
    n_train = split.X_train.shape[0]
    if method.name == "full_context" or method.add_propensity_feature or context_size >= n_train:
        return np.arange(n_train)
    seed_text = f"{method.name}:{float(p_query):.6f}:{n_train}:{context_size}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if method.random_context:
        return rng.choice(n_train, size=context_size, replace=False)
    if method.stratified_random:
        selected_parts = []
        take_each = max(1, context_size // 2)
        for r_value in [0, 1]:
            candidates = np.flatnonzero(split.r_train == r_value)
            if len(candidates) == 0:
                continue
            take = min(take_each, len(candidates))
            selected_parts.append(rng.choice(candidates, size=take, replace=False))
        selected = np.concatenate(selected_parts) if selected_parts else rng.choice(n_train, size=context_size, replace=False)
        if len(selected) < context_size:
            used = np.zeros(n_train, dtype=bool)
            used[selected] = True
            extras = rng.choice(np.flatnonzero(~used), size=context_size - len(selected), replace=False)
            selected = np.concatenate([selected, extras])
        return selected

    X_train_scaled = scaler.transform(split.X_train)
    if method.causal_weighted:
        weights = split.causal_weights / (np.mean(split.causal_weights) + 1e-9)
        diff = (X_train_scaled - x_query_scaled) * np.sqrt(weights)
    else:
        diff = X_train_scaled - x_query_scaled
    feature_distance = np.mean(diff * diff, axis=1)
    feature_similarity = -feature_distance
    propensity_distance = np.abs(split.p_train - p_query)
    score = method.alpha * feature_similarity - method.beta * propensity_distance

    if not method.balanced:
        return np.argsort(score)[-context_size:]

    take_each = max(1, context_size // 2)
    selected_parts = []
    for r_value in [0, 1]:
        candidates = np.flatnonzero(split.r_train == r_value)
        if len(candidates) == 0:
            continue
        take = min(take_each, len(candidates))
        selected_parts.append(candidates[np.argsort(score[candidates])[-take:]])
    selected = np.concatenate(selected_parts) if selected_parts else np.argsort(score)[-context_size:]
    if len(selected) < context_size:
        missing = context_size - len(selected)
        used = np.zeros(n_train, dtype=bool)
        used[selected] = True
        extras = np.argsort(score[~used])[-missing:]
        selected = np.concatenate([selected, np.flatnonzero(~used)[extras]])
    return selected


def features_for_method(
    split: SplitData,
    train_idx: np.ndarray,
    test_idx: int,
    method: QueryMethod,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = split.X_train[train_idx]
    X_test = split.X_test[[test_idx]]
    if method.add_propensity_feature:
        X_train = np.column_stack([X_train, split.p_train[train_idx], split.r_train[train_idx]])
        X_test = np.column_stack([X_test, split.p_test[[test_idx]], split.r_test[[test_idx]]])
    return X_train, X_test


def predict_with_query_context(
    split: SplitData,
    method: QueryMethod,
    *,
    device: str,
    context_size: int,
    max_test: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    scaler = StandardScaler().fit(split.X_train)
    X_test_scaled = scaler.transform(split.X_test)
    limit = min(max_test, split.X_test.shape[0])
    predictions = []
    selected_sizes = []
    selected_r1_rates = []
    start = time.perf_counter()
    for test_idx in range(limit):
        selected = selected_indices_for_query(
            split,
            X_test_scaled[test_idx],
            float(split.p_test[test_idx]),
            method=method,
            scaler=scaler,
            context_size=context_size,
        )
        X_train_ctx, X_test_one = features_for_method(split, selected, test_idx, method)
        model = TabICLClassifier(
            n_estimators=1,
            device=device,
            random_state=random_state,
            verbose=False,
        )
        model.fit(X_train_ctx, split.y_train[selected])
        predictions.append(model.predict(X_test_one)[0])
        selected_sizes.append(len(selected))
        selected_r1_rates.append(float(np.mean(split.r_train[selected])))
    runtime = time.perf_counter() - start
    details = {
        "evaluated_test": limit,
        "avg_selected_train": float(np.mean(selected_sizes)),
        "avg_selected_r1_rate": float(np.mean(selected_r1_rates)),
        "runtime_sec": runtime,
    }
    return np.asarray(predictions), details


def run_dataset(
    dataset: ClassificationDataset,
    *,
    device: str,
    random_state: int,
    test_size: float,
    context_size: int,
    max_test: int,
) -> list[dict[str, object]]:
    split = make_split(dataset, test_size, random_state)
    rows = []
    y_eval = split.y_test[: min(max_test, split.y_test.shape[0])]
    for method in query_methods():
        print(f"  method={method.name}")
        pred, details = predict_with_query_context(
            split,
            method,
            device=device,
            context_size=context_size,
            max_test=max_test,
            random_state=random_state,
        )
        rows.append(
            {
                "dataset": dataset.name,
                "source": dataset.source,
                "method": method.name,
                "description": method.description,
                "metric_name": "accuracy",
                "metric_value": float(accuracy_score(y_eval, pred)),
                "n_samples": int(dataset.X.shape[0]),
                "n_features": int(dataset.X.shape[1]),
                "train_samples": int(split.X_train.shape[0]),
                "test_samples": int(split.X_test.shape[0]),
                "context_size": context_size,
                "random_state": random_state,
                "device": device,
                **details,
            }
        )
    return rows


def winners(rows: list[dict[str, object]], baseline_method: str = "random_context") -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["random_state"])), []).append(row)
    winner_rows = []
    for _, group in grouped.items():
        baseline = next((row for row in group if row["method"] == baseline_method), None)
        if baseline is None:
            baseline = next(row for row in group if row["method"] == "full_context")
        base_score = float(baseline["metric_value"])
        for row in group:
            if row["method"] in {baseline["method"], "full_context"}:
                continue
            delta = float(row["metric_value"]) - base_score
            if delta > 0:
                out = dict(row)
                out["baseline_metric_value"] = base_score
                out["baseline_method"] = baseline["method"]
                out["delta_vs_baseline"] = delta
                winner_rows.append(out)
    winner_rows.sort(key=lambda row: (float(row["delta_vs_baseline"]), float(row["metric_value"])), reverse=True)
    return winner_rows


def write_report(rows: list[dict[str, object]], winner_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Query-Level Context Selection Search",
        "",
        "This experiment fits a fresh TabICL context for each query row, directly testing score-based context selection.",
        "",
        "## Positive Deltas",
        "",
    ]
    if winner_rows:
        lines.extend(["| Dataset | Method | Accuracy | Baseline | Baseline Method | Delta |", "| --- | --- | ---: | ---: | --- | ---: |"])
        for row in winner_rows:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {float(row['metric_value']):.4f} | "
                f"{float(row['baseline_metric_value']):.4f} | {row['baseline_method']} | {float(row['delta_vs_baseline']):+.4f} |"
            )
    else:
        lines.append("_No positive deltas found in this run._")

    lines.extend(["", "## Best Per Dataset", ""])
    lines.extend(["| Dataset | Best Method | Best Accuracy | Full Context |", "| --- | --- | ---: | ---: |"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    for dataset, group in sorted(grouped.items()):
        baseline = next(row for row in group if row["method"] == "full_context")
        best = max(group, key=lambda row: float(row["metric_value"]))
        lines.append(
            f"| {dataset} | {best['method']} | {float(best['metric_value']):.4f} | "
            f"{float(baseline['metric_value']):.4f} |"
        )

    lines.extend(["", "## Limited-Context Baselines", ""])
    lines.extend(["| Dataset | Random Context | Balanced Random | KNN Context | Full Context |", "| --- | ---: | ---: | ---: | ---: |"])
    for dataset, group in sorted(grouped.items()):
        lookup = {row["method"]: row for row in group}
        lines.append(
            f"| {dataset} | {float(lookup['random_context']['metric_value']):.4f} | "
            f"{float(lookup['balanced_random_context']['metric_value']):.4f} | "
            f"{float(lookup['knn_context']['metric_value']):.4f} | "
            f"{float(lookup['full_context']['metric_value']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is closer to the slide formula than global subset selection, but it is slower because TabICL is refit per query.",
            "- Positive deltas are measured against `random_context`, which is the fair limited-context baseline.",
            "- `propensity_score_context` uses `alpha*Sim(x_i,x*) - beta*|s_i-s*|`.",
            "- `balanced_propensity_context` additionally forces roughly balanced selected `R=0/R=1` context.",
            "- `causal_weighted_context` replaces plain distance with supervised causal-proxy weighted distance.",
        ]
    )
    (RESULTS_DIR / "query_context_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-query TabICL context selection search.")
    parser.add_argument("--datasets", default="breast_cancer,synth_selection_hard_overlap,synth_high_dim_sparse")
    parser.add_argument("--synthetic-samples", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=700)
    parser.add_argument("--context-size", type=int, default=96)
    parser.add_argument("--max-test", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="query_context_selection_results.csv")
    parser.add_argument("--winners-output", default="query_context_selection_winners.csv")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    wanted = {part.strip() for part in args.datasets.split(",") if part.strip()}
    datasets = [
        subset_dataset(dataset, args.max_samples, args.random_state)
        for dataset in load_datasets(args.synthetic_samples, args.random_state)
        if dataset.name in wanted
    ]
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]}")
        rows.extend(
            run_dataset(
                dataset,
                device=device,
                random_state=args.random_state,
                test_size=args.test_size,
                context_size=args.context_size,
                max_test=args.max_test,
            )
        )
    winner_rows = winners(rows)
    write_rows(RESULTS_DIR / args.output, rows)
    if winner_rows:
        write_rows(RESULTS_DIR / args.winners_output, winner_rows)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(rows, winner_rows)
    write_json(
        RESULTS_DIR / "query_context_selection_environment.json",
        {
            **environment_info(device),
            "datasets": sorted(wanted),
            "synthetic_samples": args.synthetic_samples,
            "max_samples": args.max_samples,
            "context_size": args.context_size,
            "max_test": args.max_test,
            "random_state": args.random_state,
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.winners_output}")
    print(f"Wrote {RESULTS_DIR / 'query_context_selection.md'}")


if __name__ == "__main__":
    main()
