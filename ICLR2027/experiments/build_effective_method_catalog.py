from __future__ import annotations

import csv
from pathlib import Path

from utils import RESULTS_DIR, write_rows


SUMMARY_FILES = [
    ("openml_propensity_variant_recheck", RESULTS_DIR / "openml_propensity_variant_recheck_summary.csv"),
    ("openml_guarded_policy", RESULTS_DIR / "openml_guarded_policy_summary.csv"),
    ("openml_guarded_scale", RESULTS_DIR / "openml_guarded_scale_study_dataset_summary.csv"),
    ("openml_broad_recheck", RESULTS_DIR / "openml_broad_recheck_summary.csv"),
    ("openml_promoted_candidate", RESULTS_DIR / "openml_promoted_candidate_summary.csv"),
    ("openml_promoted_guarded", RESULTS_DIR / "openml_promoted_guarded_summary.csv"),
    ("openml_remaining_recheck", RESULTS_DIR / "openml_remaining_recheck_summary.csv"),
    ("openml_label_balance_sweep", RESULTS_DIR / "openml_label_balance_sweep_summary.csv"),
    ("openml_label_balance_recheck", RESULTS_DIR / "openml_label_balance_recheck_summary.csv"),
    ("openml_local_geometry_sweep", RESULTS_DIR / "openml_local_geometry_sweep_summary.csv"),
    ("openml_local_geometry_recheck", RESULTS_DIR / "openml_local_geometry_recheck_summary.csv"),
    ("openml_local_geometry_guarded", RESULTS_DIR / "openml_local_geometry_guarded_summary.csv"),
    ("openml_local_geometry_guarded_strict", RESULTS_DIR / "openml_local_geometry_guarded_strict_summary.csv"),
    ("openml_local_geometry_guarded_ultra", RESULTS_DIR / "openml_local_geometry_guarded_ultra_summary.csv"),
    ("openml_domain_shift_sweep", RESULTS_DIR / "openml_domain_shift_sweep_summary.csv"),
    ("openml_domain_shift_recheck", RESULTS_DIR / "openml_domain_shift_recheck_summary.csv"),
    ("openml_domain_shift_scale", RESULTS_DIR / "openml_domain_shift_scale_summary.csv"),
    ("openml_rare_tail", RESULTS_DIR / "openml_rare_tail_summary.csv"),
    ("openml_rare_tail_combo", RESULTS_DIR / "openml_rare_tail_combo_summary.csv"),
    ("openml_probe_difficulty_sweep", RESULTS_DIR / "openml_probe_difficulty_sweep_summary.csv"),
    ("openml_probe_difficulty_recheck", RESULTS_DIR / "openml_probe_difficulty_recheck_summary.csv"),
    ("openml_probe_difficulty_scale", RESULTS_DIR / "openml_probe_difficulty_scale_summary.csv"),
    ("openml_probe_difficulty_guarded", RESULTS_DIR / "openml_probe_difficulty_guarded_summary.csv"),
    ("openml_outlier_shape_sweep", RESULTS_DIR / "openml_outlier_shape_sweep_summary.csv"),
    ("openml_outlier_shape_recheck", RESULTS_DIR / "openml_outlier_shape_recheck_summary.csv"),
    ("openml_combined_methods", RESULTS_DIR / "openml_combined_methods_summary.csv"),
    ("openml_diagnostic_policy", RESULTS_DIR / "openml_diagnostic_policy_summary.csv"),
    ("openml_diagnostic_policy_scale", RESULTS_DIR / "openml_diagnostic_policy_scale_summary.csv"),
    ("openml_decision_policy", RESULTS_DIR / "openml_decision_policy_summary.csv"),
    ("openml_decision_policy_scale", RESULTS_DIR / "openml_decision_policy_scale_summary.csv"),
    ("openml_combined_gain_policy", RESULTS_DIR / "openml_combined_gain_policy_summary.csv"),
    ("openml_combined_gain_policy_strict", RESULTS_DIR / "openml_combined_gain_policy_strict_summary.csv"),
    ("openml_extra_combined_gain_policy", RESULTS_DIR / "openml_extra_combined_gain_policy_summary.csv"),
    ("openml_extra_method_sweep", RESULTS_DIR / "openml_extra_method_sweep_summary.csv"),
    ("regression_recheck", RESULTS_DIR / "regression_recheck_summary.csv"),
    ("regression_guarded_policy", RESULTS_DIR / "regression_guarded_policy_summary.csv"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify(row: dict[str, str]) -> str:
    seeds = int(row.get("seeds") or row.get("runs") or 0)
    positive = int(row.get("positive_seeds") or row.get("positive_runs") or 0)
    nonnegative = int(row.get("nonnegative_seeds") or row.get("nonnegative_runs") or 0)
    mean_delta = float(row.get("mean_delta") or row.get("macro_mean_delta") or 0.0)
    min_delta = float(row.get("min_delta") or 0.0)
    if seeds >= 2 and positive == seeds and mean_delta > 0 and min_delta >= 0:
        return "stable_positive"
    if seeds >= 2 and nonnegative == seeds and mean_delta > 0:
        return "guarded_positive"
    if mean_delta > 0:
        return "risky_positive"
    if mean_delta == 0 and min_delta >= 0:
        return "safe_baseline_or_tie"
    return "negative_do_not_default"


def normalize_row(source_name: str, row: dict[str, str]) -> dict[str, object]:
    max_samples = row.get("max_samples", "")
    dataset = row.get("dataset", source_name)
    method = row.get("method", "adaptive_policy" if "guarded" in source_name else "")
    seeds = int(row.get("seeds") or row.get("runs") or 0)
    positive = int(row.get("positive_seeds") or row.get("positive_runs") or 0)
    nonnegative = int(row.get("nonnegative_seeds") or row.get("nonnegative_runs") or 0)
    mean_delta = float(row.get("mean_delta") or row.get("macro_mean_delta") or 0.0)
    min_delta = float(row.get("min_delta") or 0.0)
    max_delta = float(row.get("max_delta") or 0.0)
    return {
        "status": classify(row),
        "source": source_name,
        "max_samples": max_samples,
        "dataset": dataset,
        "method": method,
        "seeds_or_runs": seeds,
        "positive": positive,
        "nonnegative": nonnegative,
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
    }


def build_catalog() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_name, path in SUMMARY_FILES:
        for row in read_csv(path):
            if row.get("method") == "baseline_n1":
                continue
            rows.append(normalize_row(source_name, row))
    status_rank = {
        "stable_positive": 0,
        "guarded_positive": 1,
        "risky_positive": 2,
        "safe_baseline_or_tie": 3,
        "negative_do_not_default": 4,
    }
    rows.sort(key=lambda row: (status_rank[str(row["status"])], -float(row["mean_delta"]), float(row["min_delta"])))
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["status"]), []).append(row)

    lines = [
        "# Effective Method Catalog",
        "",
        "This catalog consolidates the current experiment evidence into stable positives, guarded positives, risky positives, ties, and negative combinations that should not become defaults.",
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status in ["stable_positive", "guarded_positive", "risky_positive", "safe_baseline_or_tie", "negative_do_not_default"]:
        lines.append(f"| {status} | {len(groups.get(status, []))} |")

    for status in ["stable_positive", "guarded_positive", "risky_positive", "negative_do_not_default"]:
        lines.extend(["", f"## {status}", ""])
        subset = groups.get(status, [])
        if not subset:
            lines.append("_None._")
            continue
        lines.extend(
            [
                "| Source | Max Samples | Dataset | Method | Runs | Positive | Mean Delta | Min Delta | Max Delta |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subset[:30]:
            lines.append(
                f"| {row['source']} | {row['max_samples']} | {row['dataset']} | {row['method']} | "
                f"{row['seeds_or_runs']} | {row['positive']} | {float(row['mean_delta']):+.4f} | "
                f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Practical Takeaways",
            "",
            "- Promote `stable_positive` rows first; these show positive gains across all checked seeds/runs.",
            "- Use `guarded_positive` rows through validation or explicit policy rules, not as unconditional defaults.",
            "- Treat `risky_positive` rows as research leads only; they need stronger guards or more seeds.",
            "- Keep `negative_do_not_default` rows as anti-patterns to prevent repeating failed combinations.",
        ]
    )
    (RESULTS_DIR / "effective_method_catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_catalog()
    write_rows(RESULTS_DIR / "effective_method_catalog.csv", rows)
    write_report(rows)
    print(f"Wrote {RESULTS_DIR / 'effective_method_catalog.csv'}")
    print(f"Wrote {RESULTS_DIR / 'effective_method_catalog.md'}")


if __name__ == "__main__":
    main()
