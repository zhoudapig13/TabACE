#!/usr/bin/env python3
"""Audit the manuscript's OpenML numbers, claims, trace data, and palette."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
OUTPUT = PAPER / "audits" / "cross_media_consistency_data.json"
TOL = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_overall_generator():
    path = PAPER / "scripts" / "generate_overall_results_table.py"
    spec = importlib.util.spec_from_file_location("overall_results_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signed_gain(metric: str, base: float, value: float) -> float:
    return value - base if metric in {"accuracy", "auc", "r2"} else base - value


def main() -> None:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, **details: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", **details})

    generator = load_overall_generator()
    expected_table = generator.build_table()
    actual_table = (PAPER / "tables" / "overall_results.tex").read_text(encoding="utf-8")
    record(
        "table_1_exactly_matches_current_generators_and_sources",
        actual_table == expected_table,
        compared_characters=len(actual_table),
    )

    values = generator.backbone_values()
    limix = generator.limix_values()
    if limix is not None:
        values["LimiX"] = limix
    nonzero_columns = {
        4: "controlled/classification/accuracy",
        5: "controlled/classification/auc",
        6: "controlled/regression/r2",
        7: "controlled/regression/nrmse",
        8: "shift/classification/accuracy",
        9: "shift/classification/auc",
        10: "shift/regression/r2",
        11: "shift/regression/nrmse",
    }
    shared_gains: list[float] = []
    specific_gains: list[float] = []
    for backbone, tier_values in values.items():
        for index, label in nonzero_columns.items():
            metric = label.rsplit("/", 1)[-1]
            shared_gains.append(
                signed_gain(metric, tier_values["base"][index], tier_values["shared"][index])
            )
            specific_gains.append(
                signed_gain(
                    metric,
                    tier_values["shared"][index],
                    tier_values["specific"][index],
                )
            )
    record(
        "table_1_nonzero_claims",
        min(shared_gains) > 0.0 and min(specific_gains) > 0.0,
        shared_over_base_comparisons=len(shared_gains),
        specific_over_shared_comparisons=len(specific_gains),
        minimum_shared_gain=min(shared_gains),
        minimum_specific_increment=min(specific_gains),
    )

    clean_differences = [
        abs(tier_values[method][index] - tier_values["base"][index])
        for tier_values in values.values()
        for method in ("shared", "specific")
        for index in range(4)
    ]
    record(
        "table_1_clean_scores_remain_close",
        max(clean_differences) < 0.005,
        maximum_absolute_clean_difference=max(clean_differences),
        threshold=0.005,
    )
    record(
        "table_1_baseline_count",
        len(generator.BASELINE_ORDER) == 13,
        baseline_count=len(generator.BASELINE_ORDER),
    )

    trace_path = PAPER / "figures" / "tabace_strength_comparison_data.csv"
    trace_rows = read_csv(trace_path)
    source_cache: dict[str, list[dict[str, str]]] = {}
    trace_errors: list[float] = []
    for row in trace_rows:
        source_name = row["summary_validation_source"].replace("\\", "/")
        source = source_cache.setdefault(source_name, read_csv(ROOT / source_name))
        benchmark = row["benchmark"]
        strength_key = "rho" if benchmark == "controlled" else "strength"
        scale = 0.01 if benchmark == "controlled" else 1.0
        strength = float(row["strength"])
        selected = [
            item
            for item in source
            if math.isclose(float(item[strength_key]) * scale, strength, abs_tol=1e-12)
        ]
        metric = row["metric"]
        method = row["source_method"]
        expected = math.fsum(float(item[f"mean_{method}_{metric}"]) for item in selected) / len(
            selected
        )
        trace_errors.append(abs(expected - float(row["macro_mean"])))
    record(
        "main_strength_figure_matches_canonical_summaries",
        len(trace_rows) == 120 and max(trace_errors) <= TOL,
        trace_rows=len(trace_rows),
        maximum_absolute_error=max(trace_errors),
        tolerance=TOL,
    )

    grouped_trace: dict[tuple[str, str, str, float], dict[str, float]] = defaultdict(dict)
    for row in trace_rows:
        grouped_trace[
            (row["benchmark"], row["task"], row["metric"], float(row["strength"]))
        ][row["method"]] = float(row["macro_mean"])
    specific_best_failures: list[str] = []
    for key, method_values in grouped_trace.items():
        benchmark, task, metric, strength = key
        if strength == 0.0:
            continue
        specific = method_values["specific"]
        competitors = [method_values["base"], method_values["shared"]]
        is_best = specific > max(competitors) if metric != "nmae" else specific < min(competitors)
        if not is_best:
            specific_best_failures.append(f"{benchmark}/{task}/{metric}/{strength:g}")
    record(
        "specific_is_best_in_all_nonzero_main_figure_panels",
        not specific_best_failures,
        comparisons=32,
        failures=specific_best_failures,
    )

    detail_path = PAPER / "figures" / "tabace_backbone_strength_details_data.csv"
    detail_rows = read_csv(detail_path)
    detail_audit = json.loads(
        (PAPER / "figures" / "tabace_backbone_strength_details_audit.json").read_text(
            encoding="utf-8"
        )
    )
    detail_errors: list[float] = []
    display_mean_errors: list[float] = []
    cross_table_errors: list[float] = []
    table_column = {
        ("five", "classification", "accuracy"): 4,
        ("five", "classification", "auc"): 5,
        ("five", "regression", "r2"): 6,
        ("five", "regression", "nrmse"): 7,
        ("seven", "classification", "accuracy"): 8,
        ("seven", "classification", "auc"): 9,
        ("seven", "regression", "r2"): 10,
        ("seven", "regression", "nrmse"): 11,
    }
    display_to_tier = {"base": "base", "shared": "shared", "pro": "specific"}
    audit_backbone_key = {
        "LimiX-2M": "LimiX",
        "TabICL-v2": "TabICL-v2",
        "TabPFN-v3": "TabPFN-v3",
        "TabDPT v1.2": "TabDPT-v1.2",
    }
    for item in detail_audit["results"]:
        matching = [
            row
            for row in detail_rows
            if row["backbone"] == item["backbone"]
            and row["track"] == item["track"]
            and row["task"] == item["task"]
            and row["metric"] == item["metric"]
        ]
        for source_method in ("base", "shared", "pro"):
            method_rows = [row for row in matching if row["source_method"] == source_method]
            family_rows = [row for row in method_rows if row["family"] != "Mean"]
            for band, selector in (
                ("origin", lambda value: value == 0.0),
                ("positive", lambda value: value > 0.0),
            ):
                selected = [row for row in family_rows if selector(float(row["strength"]))]
                observed = math.fsum(float(row["mean"]) for row in selected) / len(selected)
                expected = item["canonical_summary_aggregates"][band][source_method]
                detail_errors.append(abs(observed - expected))

            display_override = item.get("display_origin_override") or {}
            for mean_row in (row for row in method_rows if row["family"] == "Mean"):
                strength = float(mean_row["strength"])
                matching_family_rows = [
                    row
                    for row in family_rows
                    if float(row["strength"]) == strength
                ]
                expected_mean = math.fsum(
                    float(row["mean"]) for row in matching_family_rows
                ) / len(matching_family_rows)
                if strength == 0.0 and source_method in display_override:
                    expected_mean = float(display_override[source_method])
                display_mean_errors.append(
                    abs(float(mean_row["mean"]) - expected_mean)
                )
            key = (item["track"], item["task"], item["metric"])
            if key in table_column:
                backbone = audit_backbone_key[item["backbone"]]
                tier = display_to_tier[source_method]
                observed = values[backbone][tier][table_column[key]]
                expected = item["canonical_summary_aggregates"]["positive"][source_method]
                cross_table_errors.append(abs(observed - expected))
    record(
        "appendix_strength_data_matches_canonical_summaries",
        len(detail_rows) == 3360
        and max(detail_errors) <= TOL
        and max(display_mean_errors) <= TOL,
        detail_rows=len(detail_rows),
        aggregate_comparisons=len(detail_errors),
        maximum_absolute_error=max(detail_errors),
        displayed_mean_comparisons=len(display_mean_errors),
        displayed_mean_maximum_absolute_error=max(display_mean_errors),
    )
    record(
        "table_1_and_appendix_strength_aggregates_agree",
        len(cross_table_errors) == 78 and max(cross_table_errors) <= TOL,
        aggregate_comparisons=len(cross_table_errors),
        maximum_absolute_error=max(cross_table_errors),
    )

    robustness = json.loads(
        (ROOT / "robustness" / "Q2" / "q2_robustness_summary.json").read_text(
            encoding="utf-8"
        )
    )
    classification_path = (
        ROOT
        / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/"
        "composite/classification/summary_oof.csv"
    )
    regression_path = (
        ROOT
        / "PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/"
        "composite/regression/summary_oof.csv"
    )
    hashes_match = (
        sha256(classification_path) == robustness["artifacts"]["classification_summary_sha256"]
        and sha256(regression_path) == robustness["artifacts"]["regression_summary_sha256"]
    )
    record("locked_summary_hashes_match_robustness_record", hashes_match)

    failure_cells: list[str] = []
    tie_cells: list[str] = []
    method_coordinates = 0
    for task, path, metrics in (
        ("classification", classification_path, ("accuracy", "auc")),
        ("regression", regression_path, ("r2", "nmae", "nrmse")),
    ):
        for row in read_csv(path):
            if float(row["strength"]) == 0.0:
                continue
            for method in ("shared", "pro"):
                method_coordinates += 1
                for metric in metrics:
                    gain = signed_gain(
                        metric,
                        float(row[f"mean_base_{metric}"]),
                        float(row[f"mean_{method}_{metric}"]),
                    )
                    label = f"{task}/{row['shift']}/{row['strength']}/{method}/{metric}"
                    if gain < -1e-12:
                        failure_cells.append(label)
                    elif abs(gain) <= 1e-12:
                        tie_cells.append(label)
    record(
        "seven_shift_point_lock_has_no_raw_failure",
        method_coordinates == 112 and not failure_cells,
        method_coordinates=method_coordinates,
        failure_metric_cells=len(failure_cells),
        exact_tie_metric_cells=len(tie_cells),
    )

    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (PAPER / "sections").glob("*.tex")
    )
    text_claims_ok = all(
        phrase in manuscript_text
        for phrase in (
            "72 OpenML-CC18 classification and 35",
            "distribution-shift track contains 35\nand 33",
            "67.0\\% for classification and 87.6\\% for",
            "$5.7\\times$ faster repeated inference",
            "$3.9\\times$ amortized speedup",
        )
    )
    record("repeated_numeric_claims_are_consistent_in_manuscript", text_claims_ok)

    record(
        "classification_class_symbol_is_consistent",
        "\\ldots,K" not in manuscript_text
        and "K-class probability" not in manuscript_text
        and "\\ldots,L" in manuscript_text,
        canonical_symbol="L",
    )

    companion_table = (PAPER / "tables" / "result_tables_3_4.tex").read_text(
        encoding="utf-8"
    )
    natural_segment = companion_table.split("Geographic:", 1)[1].split(
        "\\bottomrule", 1
    )[0]
    natural_values = [float(value) for value in re.findall(r"(?<!\d)0\.\d{4}", natural_segment)]
    natural_groups = [natural_values[index : index + 6] for index in range(0, 36, 6)]
    natural_claim_passes = len(natural_values) == 36 and all(
        specific > max(base, shared)
        for base_group, shared_group, specific_group in (
            natural_groups[0:3],
            natural_groups[3:6],
        )
        for base, shared, specific in zip(base_group, shared_group, specific_group)
    )
    record(
        "natural_shift_best_method_claim",
        natural_claim_passes,
        compared_settings=12,
    )

    traditional_segment = companion_table.split(
        "Classification: $\\Delta$Accuracy", 1
    )[1]
    traditional_values = [
        float(value) for value in re.findall(r"\+(0\.\d{4})", traditional_segment)
    ]
    traditional_pairs = list(zip(traditional_values[0::2], traditional_values[1::2]))
    record(
        "conventional_predictor_gain_claim",
        len(traditional_pairs) == 8
        and all(shared > 0.0 and specific > shared for shared, specific in traditional_pairs),
        comparisons=len(traditional_pairs),
    )

    supplemental = json.loads(
        (PAPER / "figures" / "tabace_supplemental_figures_audit.json").read_text(
            encoding="utf-8"
        )
    )
    cost = supplemental["inference_cost"]
    repeated_specific = cost["cache_ms"]["original"][1] / cost["cache_ms"]["cached"][1]
    amortized_specific = cost["amortized_speedup"]["specific_context_6"][-1]
    record(
        "inference_cost_claim",
        round(repeated_specific, 1) == 5.7 and round(amortized_specific, 1) == 3.9,
        repeated_speedup=repeated_specific,
        ten_batch_amortized_speedup=amortized_specific,
    )

    context_count = json.loads(
        (PAPER / "figures" / "tabace_context_count_ablation_audit.json").read_text(
            encoding="utf-8"
        )
    )["values"]
    context_monotonic = True
    context_r5_beats_base = True
    for metric, item in context_count.items():
        r_values = item["r"]
        lower_is_better = metric in {"regression_nmae", "regression_nrmse"}
        if lower_is_better:
            context_monotonic &= all(left > right for left, right in zip(r_values, r_values[1:]))
            context_r5_beats_base &= r_values[-1] < item["base"]
        else:
            context_monotonic &= all(left < right for left, right in zip(r_values, r_values[1:]))
            context_r5_beats_base &= r_values[-1] > item["base"]
    record(
        "context_count_ablation_claim",
        context_monotonic and context_r5_beats_base,
        metrics=len(context_count),
        monotonic_from_R1_to_R5=context_monotonic,
        R5_beats_base=context_r5_beats_base,
    )

    textual_extensions = {".py", ".tex", ".svg", ".md", ".json", ".csv"}
    paper_text_files = [
        path
        for path in PAPER.rglob("*")
        if path.is_file()
        and path.suffix in textual_extensions
        and path.resolve() != Path(__file__).resolve()
    ]
    old_red_hits = [
        str(path.relative_to(ROOT))
        for path in paper_text_files
        if any(token in path.read_text(encoding="utf-8", errors="ignore").lower() for token in ("b2182b", "178,52,45"))
    ]
    palette_targets = (
        PAPER / "scripts" / "generate_strength_comparison.py",
        PAPER / "scripts" / "generate_backbone_strength_details.py",
        PAPER / "scripts" / "generate_overall_results_table.py",
        PAPER / "tables" / "overall_results.tex",
        PAPER / "tables" / "tabicl_strength_details.tex",
        PAPER / "figures" / "tabace_strength_comparison.svg",
        PAPER / "figures" / "tabace_limix_2m_strength_summary.svg",
        PAPER / "figures" / "tabace_tabpfn_v3_strength_summary.svg",
        PAPER / "figures" / "tabace_tabdpt_v1_2_strength_summary.svg",
    )
    missing_orange = [
        str(path.relative_to(ROOT))
        for path in palette_targets
        if "d55e00" not in path.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    record(
        "original_palette_is_consistent",
        not old_red_hits and not missing_orange,
        old_red_hits=old_red_hits,
        missing_orange_targets=missing_orange,
        base="#555555",
        shared="#56B4E9",
        specific="#D55E00",
    )

    referenced_outputs = [
        PAPER / "tables" / "overall_results.tex",
        PAPER / "tables" / "tabicl_strength_details.tex",
        PAPER / "figures" / "tabace_strength_comparison.pdf",
        PAPER / "figures" / "tabace_limix_2m_strength_summary.pdf",
        PAPER / "figures" / "tabace_tabpfn_v3_strength_summary.pdf",
        PAPER / "figures" / "tabace_tabdpt_v1_2_strength_summary.pdf",
    ]
    record(
        "affected_tables_and_figures_exist",
        all(path.is_file() for path in referenced_outputs),
        files=[str(path.relative_to(ROOT)) for path in referenced_outputs],
    )

    missing_synthetic_sources = []
    synthetic_audit = json.loads(
        (PAPER / "figures" / "tabace_synthetic_figures_audit.json").read_text(encoding="utf-8")
    )
    for source in synthetic_audit.get("sources", []):
        if not (ROOT / source).is_file():
            missing_synthetic_sources.append(source)
    record(
        "synthetic_figure_raw_sources_are_present",
        not missing_synthetic_sources,
        missing_sources=missing_synthetic_sources,
        note="This is a provenance gap, not a detected disagreement among displayed values.",
    )

    output = {
        "scope": "final manuscript cross-media consistency, with emphasis on the updated OpenML results",
        "overall_status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "checks": checks,
        "information_flow_caveat": robustness["information_flow"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
