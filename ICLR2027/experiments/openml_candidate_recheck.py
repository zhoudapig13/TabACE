from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from context_selection_search import make_split, method_specs, run_method, winner_rows
from openml_method_search import DEFAULT_OPENML_SPECS, OpenMLSpec, load_openml_dataset, parse_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


DEFAULT_CANDIDATES = {
    "openml_ilpd": ["quantile_features", "ensemble_n4", "propensity_feature"],
    "openml_credit_g": ["causal_kbest"],
    "openml_blood_transfusion": ["propensity_feature"],
    "openml_diabetes_pima": ["ensemble_n4"],
    "openml_spambase": ["quantile_features"],
}


def parse_candidates(raw: str) -> dict[str, list[str]]:
    if raw == "default":
        return {dataset: list(methods) for dataset, methods in DEFAULT_CANDIDATES.items()}
    candidates: dict[str, list[str]] = defaultdict(list)
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        dataset, method = item.split(":", 1)
        candidates[dataset.strip()].append(method.strip())
    return dict(candidates)


def selected_specs(openml_arg: str, candidates: dict[str, list[str]]) -> list[OpenMLSpec]:
    specs = parse_specs(openml_arg)
    if openml_arg != "default":
        return specs
    return [spec for spec in specs if spec.alias in candidates]


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("metric_value") not in {"", None}]
    baselines = {
        (str(row["dataset"]), int(row["random_state"])): float(row["metric_value"])
        for row in valid
        if row["method"] == "baseline_n1"
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in valid:
        if row["method"] == "baseline_n1":
            continue
        key = (str(row["dataset"]), int(row["random_state"]))
        if key not in baselines:
            continue
        out = dict(row)
        out["baseline_metric_value"] = baselines[key]
        out["delta_vs_baseline"] = float(row["metric_value"]) - baselines[key]
        grouped[(str(row["dataset"]), str(row["method"]))].append(out)

    summary_rows: list[dict[str, object]] = []
    for (dataset, method), group in grouped.items():
        deltas = np.asarray([float(row["delta_vs_baseline"]) for row in group], dtype=float)
        scores = np.asarray([float(row["metric_value"]) for row in group], dtype=float)
        baselines_arr = np.asarray([float(row["baseline_metric_value"]) for row in group], dtype=float)
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": len(group),
                "positive_seeds": int(np.sum(deltas > 0)),
                "nonnegative_seeds": int(np.sum(deltas >= 0)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "mean_metric_value": float(np.mean(scores)),
                "mean_baseline_metric_value": float(np.mean(baselines_arr)),
            }
        )
    summary_rows.sort(key=lambda row: (int(row["positive_seeds"]), float(row["mean_delta"])), reverse=True)
    return summary_rows


def write_report(summary_rows: list[dict[str, object]], winners: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# OpenML Candidate Recheck",
        "",
        "This report re-runs the strongest OpenML candidates across multiple random seeds.",
        "",
        "## Stability Summary",
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
        lines.append("_No valid candidate rows were produced._")

    lines.extend(["", "## Individual Positive Runs", ""])
    if winners:
        lines.extend(["| Dataset | Seed | Method | Accuracy | Baseline | Delta |", "| --- | ---: | --- | ---: | ---: | ---: |"])
        for row in winners[:40]:
            lines.append(
                f"| {row['dataset']} | {row['random_state']} | {row['method']} | "
                f"{float(row['metric_value']):.4f} | {float(row['baseline_metric_value']):.4f} | "
                f"{float(row['delta_vs_baseline']):+.4f} |"
            )
    else:
        lines.append("_No positive individual runs._")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `positive_seeds` is the first reliability filter: a method that only wins once may be a random split artifact.",
            "- `min_delta` shows downside risk. A strong average with a very negative minimum should be retested before becoming a default method.",
            "- Candidates that keep positive or near-zero deltas across seeds are the best next targets for larger samples and more OpenML datasets.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recheck positive OpenML candidates across random seeds.")
    parser.add_argument("--openml", default="default")
    parser.add_argument("--candidates", default="default")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_candidate_recheck_results.csv")
    parser.add_argument("--summary-output", default="openml_candidate_recheck_summary.csv")
    parser.add_argument("--winners-output", default="openml_candidate_recheck_winners.csv")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    candidates = parse_candidates(args.candidates)
    methods_by_name = {method.name: method for method in method_specs()}
    specs = selected_specs(args.openml, candidates)
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
            method_names = ["baseline_n1", *candidates.get(dataset.name, [])]
            for method_name in dict.fromkeys(method_names):
                method = methods_by_name[method_name]
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
    write_report(summary_rows, winners, RESULTS_DIR / "openml_candidate_recheck.md")
    write_json(
        RESULTS_DIR / "openml_candidate_recheck_environment.json",
        {
            **environment_info(device),
            "max_samples": args.max_samples,
            "random_states": random_states,
            "candidates": candidates,
            "openml_specs": [spec.__dict__ for spec in specs],
            "loaded_datasets": sorted(loaded_datasets),
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.summary_output}")
    print(f"Wrote {RESULTS_DIR / args.winners_output}")
    print(f"Wrote {RESULTS_DIR / 'openml_candidate_recheck.md'}")


if __name__ == "__main__":
    main()
