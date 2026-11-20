from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from context_selection_search import MethodSpec, make_split, method_specs, run_method
from openml_method_search import OpenMLSpec, load_openml_dataset, parse_specs
from openml_propensity_variant_search import EXPANDED_OPENML_SPECS, variant_specs
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


BROAD_OPENML_SPECS = [
    *EXPANDED_OPENML_SPECS,
    OpenMLSpec(3, "openml_kr_vs_kp", "Chess king-rook versus king-pawn classification"),
    OpenMLSpec(6, "openml_letter", "Letter recognition multiclass dataset"),
    OpenMLSpec(12, "openml_mfeat_factors", "Multiple features factors dataset"),
    OpenMLSpec(14, "openml_mfeat_fourier", "Multiple features Fourier dataset"),
    OpenMLSpec(15, "openml_breast_w", "Wisconsin breast cancer classification"),
    OpenMLSpec(16, "openml_mfeat_karhunen", "Multiple features Karhunen dataset"),
    OpenMLSpec(18, "openml_mfeat_morphological", "Multiple features morphological dataset"),
    OpenMLSpec(22, "openml_mfeat_zernike", "Multiple features Zernike dataset"),
    OpenMLSpec(36, "openml_segment", "Image segmentation multiclass dataset"),
    OpenMLSpec(38, "openml_sick", "Thyroid sick dataset with mixed features"),
]

EXTRA_OPENML_SPECS = [
    OpenMLSpec(28, "openml_optdigits", "Optical handwritten digit recognition"),
    OpenMLSpec(32, "openml_pendigits", "Pen-based handwritten digit recognition"),
    OpenMLSpec(46, "openml_splice", "DNA splice junction classification"),
    OpenMLSpec(151, "openml_electricity", "Electricity market direction classification"),
    OpenMLSpec(182, "openml_satimage", "Satellite image multiclass classification"),
    OpenMLSpec(188, "openml_eucalyptus", "Eucalyptus seedlot classification"),
    OpenMLSpec(307, "openml_vowel", "Vowel recognition multiclass dataset"),
    OpenMLSpec(458, "openml_authorship", "Authorship attribution with text-derived features"),
    OpenMLSpec(469, "openml_dmft", "Dental caries DMFT multiclass dataset"),
    OpenMLSpec(1049, "openml_pc4", "Software defect classification pc4"),
    OpenMLSpec(1050, "openml_pc3", "Software defect classification pc3"),
    OpenMLSpec(1063, "openml_kc2", "Software defect classification kc2"),
    OpenMLSpec(1067, "openml_kc1", "Software defect classification kc1"),
    OpenMLSpec(1068, "openml_pc1", "Software defect classification pc1"),
    OpenMLSpec(1590, "openml_adult", "Adult income classification"),
    OpenMLSpec(40975, "openml_car", "Car evaluation classification"),
    OpenMLSpec(40982, "openml_steel_plates_fault", "Steel plates fault multiclass classification"),
    OpenMLSpec(40984, "openml_segment_challenge", "Segment challenge multiclass classification"),
]

WIDE_OPENML_SPECS = [*BROAD_OPENML_SPECS, *EXTRA_OPENML_SPECS]


DEFAULT_NEW_SPECS = [
    spec
    for spec in BROAD_OPENML_SPECS
    if spec.alias
    in {
        "openml_kr_vs_kp",
        "openml_letter",
        "openml_mfeat_factors",
        "openml_mfeat_fourier",
        "openml_breast_w",
        "openml_mfeat_karhunen",
        "openml_mfeat_morphological",
        "openml_segment",
    }
]


def specs_from_arg(raw: str) -> list[OpenMLSpec]:
    if raw == "new":
        return DEFAULT_NEW_SPECS
    if raw == "broad":
        return BROAD_OPENML_SPECS
    if raw == "extra":
        return EXTRA_OPENML_SPECS
    if raw == "wide":
        return WIDE_OPENML_SPECS
    if raw == "expanded":
        return EXPANDED_OPENML_SPECS
    return parse_specs(raw)


def quick_methods(method_names: str = "quick") -> list[MethodSpec]:
    by_name = {method.name: method for method in [*variant_specs(), *method_specs()]}
    if method_names == "quick":
        names = [
            "baseline_n1",
            "propensity_rank_feature",
            "propensity_interactions",
            "propensity_causal_kbest",
            "propensity_overlap_feature",
            "propensity_bin_balance",
            "quantile_features",
            "causal_kbest",
        ]
    else:
        names = [part.strip() for part in method_names.split(",") if part.strip()]
    return [by_name[name] for name in names if name in by_name]


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("metric_value") in {"", None} or row.get("method") in {"baseline_n1", "load_failed"}:
            continue
        grouped[(int(row["max_samples"]), str(row["dataset"]), str(row["method"]))].append(row)

    baseline: dict[tuple[int, str, int], float] = {}
    for row in rows:
        if row.get("method") == "baseline_n1" and row.get("metric_value") not in {"", None}:
            baseline[(int(row["max_samples"]), str(row["dataset"]), int(row["random_state"]))] = float(row["metric_value"])

    summary = []
    extra_metrics = ["balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]
    for (max_samples, dataset, method), group in grouped.items():
        deltas = []
        scores = []
        baselines = []
        extra_deltas: dict[str, list[float]] = {metric: [] for metric in extra_metrics}
        extra_scores: dict[str, list[float]] = {metric: [] for metric in extra_metrics}
        extra_baselines: dict[str, list[float]] = {metric: [] for metric in extra_metrics}
        for row in group:
            key = (max_samples, dataset, int(row["random_state"]))
            if key not in baseline:
                continue
            baseline_row = next(
                (
                    base_row
                    for base_row in rows
                    if base_row.get("method") == "baseline_n1"
                    and base_row.get("metric_value") not in {"", None}
                    and int(base_row["max_samples"]) == max_samples
                    and str(base_row["dataset"]) == dataset
                    and int(base_row["random_state"]) == int(row["random_state"])
                ),
                None,
            )
            scores.append(float(row["metric_value"]))
            baselines.append(baseline[key])
            deltas.append(float(row["metric_value"]) - baseline[key])
            if baseline_row is not None:
                for metric in extra_metrics:
                    if row.get(metric) not in {"", None} and baseline_row.get(metric) not in {"", None}:
                        extra_scores[metric].append(float(row[metric]))
                        extra_baselines[metric].append(float(baseline_row[metric]))
                        extra_deltas[metric].append(float(row[metric]) - float(baseline_row[metric]))
        if not deltas:
            continue
        values = np.asarray(deltas, dtype=float)
        out = {
            "max_samples": max_samples,
            "dataset": dataset,
            "method": method,
            "seeds": len(values),
            "positive_seeds": int(np.sum(values > 0)),
            "nonnegative_seeds": int(np.sum(values >= 0)),
            "mean_delta": float(np.mean(values)),
            "median_delta": float(np.median(values)),
            "min_delta": float(np.min(values)),
            "max_delta": float(np.max(values)),
            "mean_metric_value": float(np.mean(scores)),
            "mean_baseline_metric_value": float(np.mean(baselines)),
        }
        for metric in extra_metrics:
            metric_values = np.asarray(extra_deltas[metric], dtype=float)
            if len(metric_values):
                out[f"mean_{metric}_delta"] = float(np.mean(metric_values))
                out[f"min_{metric}_delta"] = float(np.min(metric_values))
                out[f"max_{metric}_delta"] = float(np.max(metric_values))
                out[f"positive_{metric}_seeds"] = int(np.sum(metric_values > 0))
                out[f"nonnegative_{metric}_seeds"] = int(np.sum(metric_values >= 0))
                out[f"mean_{metric}"] = float(np.mean(extra_scores[metric]))
                out[f"mean_baseline_{metric}"] = float(np.mean(extra_baselines[metric]))
        summary.append(out)
    summary.sort(key=lambda row: (float(row["mean_delta"]), int(row["positive_seeds"])), reverse=True)
    return summary


def positive_winners(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline: dict[tuple[int, str, int], float] = {}
    for row in rows:
        if row.get("method") == "baseline_n1" and row.get("metric_value") not in {"", None}:
            baseline[(int(row["max_samples"]), str(row["dataset"]), int(row["random_state"]))] = float(row["metric_value"])

    winners = []
    for row in rows:
        if row.get("metric_value") in {"", None} or row.get("method") in {"baseline_n1", "load_failed"}:
            continue
        key = (int(row["max_samples"]), str(row["dataset"]), int(row["random_state"]))
        if key not in baseline:
            continue
        delta = float(row["metric_value"]) - baseline[key]
        if delta > 0:
            out = dict(row)
            out["baseline_metric_value"] = baseline[key]
            out["delta_vs_baseline"] = delta
            winners.append(out)
    winners.sort(key=lambda row: (float(row["delta_vs_baseline"]), float(row["metric_value"])), reverse=True)
    return winners


def write_report(
    summary_rows: list[dict[str, object]],
    winners: list[dict[str, object]],
    loaded: list[str],
    failed: list[str],
    *,
    report_output: str,
) -> None:
    lines = [
        "# OpenML Broad Sweep",
        "",
        "This is a fast screening pass on additional OpenML datasets. It keeps prior promising propensity and causal-proxy transforms, then checks which dataset-method pairs deserve heavier multi-seed rechecks.",
        "",
        "## Coverage",
        "",
        f"- Loaded datasets: {len(loaded)}",
        f"- Failed datasets: {len(failed)}",
        f"- Loaded: {', '.join(sorted(loaded)) if loaded else 'none'}",
        f"- Failed: {', '.join(sorted(failed)) if failed else 'none'}",
        "",
        "## Strongest Mean Deltas",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| Max Samples | Dataset | Method | Seeds | Positive | Acc Delta | BAcc Delta | Macro F1 Delta | AUC Delta | Min Acc Delta | Max Acc Delta |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows[:40]:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['method']} | {row['seeds']} | {row['positive_seeds']} | "
                f"{float(row['mean_delta']):+.4f} | {float(row.get('mean_balanced_accuracy_delta', 0.0)):+.4f} | "
                f"{float(row.get('mean_macro_f1_delta', 0.0)):+.4f} | "
                f"{float(row.get('mean_roc_auc_delta', 0.0)):+.4f} | "
                f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )
    else:
        lines.append("_No valid summary rows._")

    lines.extend(["", "## Top Individual Improvements", ""])
    if winners:
        lines.extend(
            [
                "| Max Samples | Dataset | Seed | Method | Accuracy | Baseline | Delta |",
                "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in winners[:40]:
            lines.append(
                f"| {row['max_samples']} | {row['dataset']} | {row['random_state']} | {row['method']} | "
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
            "- This is a broad candidate search, not a final default policy.",
            "- Dataset-method pairs with positive deltas should be rechecked across more random seeds before promotion.",
            "- Negative rows are useful too: they identify where a transform should stay behind guarded selection rather than becoming a default.",
        ]
    )
    (RESULTS_DIR / report_output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast broad OpenML sweep for TabICL method candidates.")
    parser.add_argument("--openml", default="new", help="new, broad, expanded, default, or comma list like 31:credit")
    parser.add_argument("--scales", default="600")
    parser.add_argument("--random-states", default="42")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_broad_sweep_results.csv")
    parser.add_argument("--summary-output", default="openml_broad_sweep_summary.csv")
    parser.add_argument("--winners-output", default="openml_broad_sweep_winners.csv")
    parser.add_argument("--report-output", default="openml_broad_sweep.md")
    parser.add_argument("--environment-output", default="openml_broad_sweep_environment.json")
    parser.add_argument("--methods", default="quick", help="quick or comma-separated method names")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    specs = specs_from_arg(args.openml)
    scales = [int(part.strip()) for part in args.scales.split(",") if part.strip()]
    random_states = [int(part.strip()) for part in args.random_states.split(",") if part.strip()]
    methods = quick_methods(args.methods)
    rows: list[dict[str, object]] = []
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
                print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]}")
                split = make_split(dataset, args.test_size, random_state)
                for method in methods:
                    print(f"  method={method.name}")
                    try:
                        row = run_method(split, dataset, method, device=device, random_state=random_state)
                        row["max_samples"] = max_samples
                        rows.append(row)
                    except Exception as exc:
                        rows.append(
                            {
                                "max_samples": max_samples,
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

    summary_rows = summarize(rows)
    winners = positive_winners(rows)
    write_rows(RESULTS_DIR / args.output, rows)
    write_rows(RESULTS_DIR / args.summary_output, summary_rows)
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["max_samples", "dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(summary_rows, winners, sorted(loaded), sorted(failed), report_output=args.report_output)
    write_json(
        RESULTS_DIR / args.environment_output,
        {
            **environment_info(device),
            "openml": args.openml,
            "scales": scales,
            "random_states": random_states,
            "test_size": args.test_size,
            "openml_specs": [spec.__dict__ for spec in specs],
            "loaded_datasets": sorted(loaded),
            "failed_datasets": sorted(failed),
            "methods": [method.name for method in methods],
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.summary_output}")
    print(f"Wrote {RESULTS_DIR / args.winners_output}")
    print(f"Wrote {RESULTS_DIR / args.report_output}")


if __name__ == "__main__":
    main()
