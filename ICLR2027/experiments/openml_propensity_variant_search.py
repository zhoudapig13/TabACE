from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from context_selection_search import MethodSpec, SplitData, identity, make_split, run_method, winner_rows
from openml_candidate_recheck import summarize
from openml_method_search import DEFAULT_OPENML_SPECS, OpenMLSpec, load_openml_dataset, parse_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


EXPANDED_OPENML_SPECS = [
    *DEFAULT_OPENML_SPECS,
    OpenMLSpec(11, "openml_balance_scale", "Small multiclass balance-scale dataset"),
    OpenMLSpec(23, "openml_cmc", "Contraceptive method choice with mixed social features"),
    OpenMLSpec(29, "openml_credit_approval", "Credit approval with mixed categorical/numeric features"),
    OpenMLSpec(50, "openml_tic_tac_toe", "Categorical game-end classification"),
    OpenMLSpec(54, "openml_vehicle", "Vehicle silhouette multiclass classification"),
]


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def standardized(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def propensity_logit_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    p_train = safe_logit(split.p_train)
    p_test = safe_logit(split.p_test)
    X_train = np.column_stack([split.X_train, p_train, split.r_train])
    X_test = np.column_stack([split.X_test, p_test, split.r_test])
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "added_features": 2}


def propensity_rank_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    order = np.argsort(split.p_train)
    ranks = np.empty_like(split.p_train, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(split.p_train), dtype=np.float32)
    test_ranks = np.searchsorted(np.sort(split.p_train), split.p_test, side="left") / max(len(split.p_train), 1)
    X_train = np.column_stack([split.X_train, ranks, split.r_train])
    X_test = np.column_stack([split.X_test, test_ranks, split.r_test])
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "added_features": 2}


def propensity_interactions(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train_scaled, X_test_scaled = standardized(split.X_train, split.X_test)
    k = min(max(2, split.X_train.shape[1] // 4), 8, split.X_train.shape[1])
    top = np.argsort(split.causal_weights)[-k:]
    p_train = (split.p_train - np.mean(split.p_train))[:, None]
    p_test = (split.p_test - np.mean(split.p_train))[:, None]
    interactions_train = X_train_scaled[:, top] * p_train
    interactions_test = X_test_scaled[:, top] * p_test
    X_train = np.column_stack([split.X_train, split.p_train, split.r_train, interactions_train])
    X_test = np.column_stack([split.X_test, split.p_test, split.r_test, interactions_test])
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "added_features": int(2 + k),
        "interaction_features": int(k),
    }


def propensity_causal_kbest(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    k = min(max(4, split.X_train.shape[1] // 2), split.X_train.shape[1])
    top = np.argsort(split.causal_weights)[-k:]
    X_train = np.column_stack([split.X_train[:, top], split.p_train, split.r_train])
    X_test = np.column_stack([split.X_test[:, top], split.p_test, split.r_test])
    return X_train, split.y_train, X_test, {
        "selected_train": len(split.y_train),
        "selected_features": int(k),
        "added_features": 2,
    }


def propensity_overlap_feature(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    lo = max(np.quantile(split.p_train, 0.08), np.quantile(split.p_test, 0.04))
    hi = min(np.quantile(split.p_train, 0.92), np.quantile(split.p_test, 0.96))
    mask = (split.p_train >= lo) & (split.p_train <= hi)
    if np.sum(mask) < max(32, len(split.y_train) // 4):
        X_train = np.column_stack([split.X_train, split.p_train, split.r_train])
        X_test = np.column_stack([split.X_test, split.p_test, split.r_test])
        return X_train, split.y_train, X_test, {
            "selected_train": len(split.y_train),
            "added_features": 2,
            "fallback": "no_trim",
        }
    X_train = np.column_stack([split.X_train[mask], split.p_train[mask], split.r_train[mask]])
    X_test = np.column_stack([split.X_test, split.p_test, split.r_test])
    return X_train, split.y_train[mask], X_test, {
        "selected_train": int(np.sum(mask)),
        "added_features": 2,
        "propensity_lo": float(lo),
        "propensity_hi": float(hi),
    }


def propensity_bin_balance(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    quantiles = np.unique(np.quantile(split.p_train, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(quantiles) < 3:
        return identity(split, random_state)
    bins = np.digitize(split.p_train, quantiles[1:-1], right=True)
    groups = [np.flatnonzero(bins == bin_id) for bin_id in np.unique(bins)]
    if min(len(group) for group in groups) < 8:
        return identity(split, random_state)
    take = min(len(group) for group in groups)
    selected = np.concatenate([rng.choice(group, size=take, replace=False) for group in groups])
    rng.shuffle(selected)
    X_train = np.column_stack([split.X_train[selected], split.p_train[selected], split.r_train[selected]])
    X_test = np.column_stack([split.X_test, split.p_test, split.r_test])
    return X_train, split.y_train[selected], X_test, {
        "selected_train": int(len(selected)),
        "added_features": 2,
        "bins": int(len(groups)),
        "per_bin": int(take),
    }


def variant_specs() -> list[MethodSpec]:
    return [
        MethodSpec("baseline_n1", "TabICL default one-estimator baseline", identity, {"n_estimators": 1}),
        MethodSpec("propensity_feature", "Append propensity score and treatment flag", propensity_logit_passthrough, {"n_estimators": 1}),
        MethodSpec("propensity_logit_feature", "Append logit propensity score and treatment flag", propensity_logit_feature, {"n_estimators": 1}),
        MethodSpec("propensity_rank_feature", "Append propensity percentile rank and treatment flag", propensity_rank_feature, {"n_estimators": 1}),
        MethodSpec("propensity_interactions", "Append propensity interactions with top causal-proxy features", propensity_interactions, {"n_estimators": 1}),
        MethodSpec("propensity_causal_kbest", "Keep top causal-proxy features and append propensity features", propensity_causal_kbest, {"n_estimators": 1}),
        MethodSpec("propensity_overlap_feature", "Trim to propensity overlap and append propensity features", propensity_overlap_feature, {"n_estimators": 1}),
        MethodSpec("propensity_bin_balance", "Balance training context across propensity bins", propensity_bin_balance, {"n_estimators": 1}),
    ]


def propensity_logit_passthrough(split: SplitData, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X_train = np.column_stack([split.X_train, split.p_train, split.r_train])
    X_test = np.column_stack([split.X_test, split.p_test, split.r_test])
    return X_train, split.y_train, X_test, {"selected_train": len(split.y_train), "added_features": 2}


def specs_from_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "expanded":
        return EXPANDED_OPENML_SPECS
    if raw == "default":
        return DEFAULT_OPENML_SPECS
    return parse_specs(raw)


def write_report(summary_rows: list[dict[str, object]], winners: list[dict[str, object]], path) -> None:
    lines = [
        "# OpenML Propensity Variant Search",
        "",
        "This run tests several selection-bias and propensity-score variants on a broader OpenML suite.",
        "",
        "## Mean Delta Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Dataset | Method | Seeds | Positive Seeds | Mean Delta | Median Delta | Min Delta | Max Delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row['median_delta']):+.4f} | "
                f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )
    else:
        lines.append("_No valid rows._")

    lines.extend(["", "## Top Individual Improvements", ""])
    if winners:
        lines.extend(["| Dataset | Seed | Method | Accuracy | Baseline | Delta |", "| --- | ---: | --- | ---: | ---: | ---: |"])
        for row in winners[:40]:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['method']} | "
                f"{float(row['metric_value']):.4f} | {float(row['baseline_metric_value']):.4f} | "
                f"{float(row['delta_vs_baseline']):+.4f} |"
            )
    else:
        lines.append("_No positive individual improvements._")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `propensity_feature` is the control variant from the previous OpenML recheck.",
            "- `propensity_causal_kbest` asks whether the signal improves when noisy features are removed before adding selection-bias features.",
            "- `propensity_interactions` tests whether propensity should modulate only likely causal features rather than entering as a standalone column.",
            "- Broad wins should be retested with more seeds; this is a screening pass for new directions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search OpenML propensity feature variants.")
    parser.add_argument("--openml", default="expanded")
    parser.add_argument("--random-states", default="42")
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_propensity_variant_results.csv")
    parser.add_argument("--summary-output", default="openml_propensity_variant_summary.csv")
    parser.add_argument("--winners-output", default="openml_propensity_variant_winners.csv")
    parser.add_argument("--report-output", default="openml_propensity_variant_search.md")
    parser.add_argument("--environment-output", default="openml_propensity_variant_environment.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    specs = specs_from_arg(args.openml)
    methods = variant_specs()
    rows: list[dict[str, object]] = []
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
                            "metric_name": "accuracy",
                            "metric_value": "",
                            "error": repr(exc),
                            "n_samples": int(dataset.X.shape[0]),
                            "n_features_original": int(dataset.X.shape[1]),
                            "device": device,
                            "random_state": random_state,
                        }
                    )
                    print(f"    failed: {exc!r}")

    valid_rows = [row for row in rows if row.get("metric_value") not in {"", None} and row["method"] != "load_failed"]
    winners = winner_rows(valid_rows)
    summary_rows = summarize(rows)
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
            "openml_specs": [spec.__dict__ for spec in specs],
            "loaded_datasets": sorted(loaded_datasets),
            "methods": [method.name for method in methods],
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.summary_output}")
    print(f"Wrote {RESULTS_DIR / args.winners_output}")
    print(f"Wrote {RESULTS_DIR / args.report_output}")


if __name__ == "__main__":
    main()
