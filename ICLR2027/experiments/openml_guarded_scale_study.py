from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from context_selection_search import make_split, run_method, winner_rows
from openml_adaptive_policy import choose_method
from openml_candidate_recheck import summarize
from openml_method_search import OpenMLSpec, load_openml_dataset, parse_specs
from openml_propensity_variant_search import variant_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


DEFAULT_SCALE_SPECS = [
    OpenMLSpec(23, "openml_cmc", "Contraceptive method choice with mixed social features"),
    OpenMLSpec(29, "openml_credit_approval", "Credit approval with mixed categorical/numeric features"),
    OpenMLSpec(37, "openml_diabetes_pima", "Pima diabetes binary medical classification"),
    OpenMLSpec(31, "openml_credit_g", "Credit risk with mixed categorical/numeric features"),
]


def parse_scales(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_specs_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "default":
        return DEFAULT_SCALE_SPECS
    return parse_specs(raw)


def summarize_by_scale(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("method") == "adaptive_policy"]
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in valid:
        grouped[(int(row["max_samples"]), str(row["dataset"]))].append(row)
    out: list[dict[str, object]] = []
    for (max_samples, dataset), group in grouped.items():
        deltas = np.asarray([float(row["delta_vs_baseline"]) for row in group], dtype=float)
        out.append(
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
    out.sort(key=lambda row: (int(row["max_samples"]), float(row["mean_delta"])), reverse=True)
    return out


def summarize_scales(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("method") == "adaptive_policy"]
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in valid:
        grouped[int(row["max_samples"])].append(row)
    out: list[dict[str, object]] = []
    for max_samples, group in grouped.items():
        deltas = np.asarray([float(row["delta_vs_baseline"]) for row in group], dtype=float)
        out.append(
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
    out.sort(key=lambda row: int(row["max_samples"]))
    return out


def write_report(scale_summary: list[dict[str, object]], dataset_summary: list[dict[str, object]]) -> None:
    lines = [
        "# OpenML Guarded Policy Scale Study",
        "",
        "This experiment checks whether the guarded adaptive policy remains useful as sample size changes.",
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
                f"{float(row['mean_delta']):+.4f} | {float(row['min_delta']):+.4f} | "
                f"{float(row['max_delta']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a scale robustness check, not a new method search.",
            "- A useful guarded policy should keep `min_delta` near zero while retaining some positive runs.",
            "- If a method only helps at one sample size, it should be treated as a conditional candidate rather than a default.",
        ]
    )
    (RESULTS_DIR / "openml_guarded_scale_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded OpenML adaptive policy across sample scales.")
    parser.add_argument("--openml", default="default")
    parser.add_argument("--scales", default="300,600,900")
    parser.add_argument("--random-states", default="11,23,42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--min-validation-delta", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    specs = parse_specs_arg(args.openml)
    scales = parse_scales(args.scales)
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    methods = variant_specs()
    baseline_method = next(method for method in methods if method.name == "baseline_n1")
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    loaded_datasets: set[str] = set()

    for max_samples in scales:
        for random_state in random_states:
            for spec in specs:
                print(f"Loading OpenML data_id={spec.data_id} alias={spec.alias} max_samples={max_samples} seed={random_state}")
                try:
                    dataset = load_openml_dataset(spec, max_samples=max_samples, random_state=random_state)
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
                            "max_samples": max_samples,
                            "device": device,
                        }
                    )
                    print(f"  failed: {exc!r}")
                    continue
                loaded_datasets.add(dataset.name)
                split = make_split(dataset, args.test_size, random_state)
                baseline_row = run_method(split, dataset, baseline_method, device=device, random_state=random_state)
                baseline_score = float(baseline_row["metric_value"])
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
                    policy="guarded",
                )
                for row in val_rows:
                    row["max_samples"] = max_samples
                validation_rows.extend(val_rows)
                decision["max_samples"] = max_samples
                decisions.append(decision)
                chosen_row = run_method(split, dataset, chosen, device=device, random_state=random_state)
                adaptive_row = dict(chosen_row)
                adaptive_row["method"] = "adaptive_policy"
                adaptive_row["chosen_method"] = chosen.name
                adaptive_row["description"] = f"Scale-study guarded policy selected {chosen.name}"
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

    scale_summary = summarize_scales(rows)
    dataset_summary = summarize_by_scale(rows)
    write_rows(RESULTS_DIR / "openml_guarded_scale_study_results.csv", rows)
    write_rows(RESULTS_DIR / "openml_guarded_scale_study_scale_summary.csv", scale_summary)
    write_rows(RESULTS_DIR / "openml_guarded_scale_study_dataset_summary.csv", dataset_summary)
    write_rows(RESULTS_DIR / "openml_guarded_scale_study_decisions.csv", decisions)
    write_rows(RESULTS_DIR / "openml_guarded_scale_study_validation.csv", validation_rows)
    write_report(scale_summary, dataset_summary)
    adaptive_for_winners = [row for row in rows if row.get("method") in {"baseline_n1", "adaptive_policy"}]
    winners = winner_rows([row for row in adaptive_for_winners if row.get("metric_value") not in {"", None}])
    if winners:
        write_rows(RESULTS_DIR / "openml_guarded_scale_study_winners.csv", winners)
    else:
        with (RESULTS_DIR / "openml_guarded_scale_study_winners.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_json(
        RESULTS_DIR / "openml_guarded_scale_study_environment.json",
        {
            **environment_info(device),
            "scales": scales,
            "random_states": random_states,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "min_validation_delta": args.min_validation_delta,
            "openml_specs": [spec.__dict__ for spec in specs],
            "loaded_datasets": sorted(loaded_datasets),
        },
    )
    print(f"Wrote {RESULTS_DIR / 'openml_guarded_scale_study.md'}")


if __name__ == "__main__":
    main()
