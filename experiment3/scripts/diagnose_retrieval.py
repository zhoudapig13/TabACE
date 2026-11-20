from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccs.utils import read_rows, to_float, write_rows  # noqa: E402


LOWER_IS_BETTER = {"pehe", "ate_error", "policy_regret", "tau_ece", "runtime_sec"}
HIGHER_IS_BETTER = {"policy_value", "uplift_top30", "auuc", "qini"}
COMPARISON_METRICS = ["pehe", "ate_error", "policy_value", "policy_regret", "uplift_top30"]
QUALITY_METRICS = [
    "meta_mean_context_abs_smd",
    "meta_context_abs_smd_delta_vs_full_train",
    "meta_mean_context_propensity_distance",
    "meta_mean_context_feature_distance",
    "meta_mean_context_treated_rate",
    "meta_mean_context_overlap",
    "meta_mean_context_diversity",
    "meta_mean_context_duplicate_rate",
]
DEFAULT_BASELINES = ("random", "knn", "matched_pair")
DEFAULT_METHODS = (
    "matched_pair",
    "matched_pair_propensity",
    "matched_pair_feature",
    "matched_pair_hybrid_w0.05",
    "matched_pair_hybrid_w0.2",
    "matched_pair_hybrid_w0.5",
    "matched_pair_caliper_c0.05",
    "matched_pair_caliper_c0.1",
    "matched_pair_caliper_c0.2",
    "matched_pair_diverse",
    "overlap_filtered_matched_pair",
    "overlap_aware",
    "propensity_nearest",
    "treatment_balanced_random",
)
PAPER_METHODS = ("matched_pair_diverse", "matched_pair")


def key_for(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("regime", ""),
        row.get("scenario_id", row.get("condition", row.get("realization", ""))),
        row.get("random_state", row.get("split_seed", row.get("realization", ""))),
        row.get("confounding_strength", ""),
        row.get("treatment_imbalance", ""),
        row.get("overlap_violation", ""),
        row.get("heterogeneity", ""),
        row.get("effect_nonlinearity", ""),
        row.get("context_size", ""),
    )


def paired_deltas(rows: list[dict[str, str]], methods: tuple[str, ...], baselines: tuple[str, ...]) -> list[dict[str, object]]:
    context_rows = [row for row in rows if row.get("estimator") == "local_context_t_learner"]
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in context_rows:
        grouped[key_for(row)][row["retriever"]] = row

    out = []
    for key, group in grouped.items():
        for method in methods:
            if method not in group:
                continue
            for baseline in baselines:
                if method == baseline:
                    continue
                if baseline not in group:
                    continue
                method_row = group[method]
                baseline_row = group[baseline]
                rec = {
                    "regime": key[0],
                    "scenario_id": key[1],
                    "random_state": key[2],
                    "confounding_strength": key[3],
                    "treatment_imbalance": key[4],
                    "overlap_violation": key[5],
                    "heterogeneity": key[6],
                    "effect_nonlinearity": key[7],
                    "context_size": key[8],
                    "method": method,
                    "baseline": baseline,
                }
                for metric in COMPARISON_METRICS:
                    method_value = to_float(method_row.get(metric))
                    baseline_value = to_float(baseline_row.get(metric))
                    delta = method_value - baseline_value
                    rec[f"{metric}_method"] = method_value
                    rec[f"{metric}_baseline"] = baseline_value
                    rec[f"{metric}_delta"] = delta
                    rec[f"{metric}_improved"] = _improved(metric, delta)
                    if metric == "pehe" and baseline_value != 0 and not math.isnan(baseline_value):
                        rec["relative_pehe_reduction_pct"] = 100.0 * (baseline_value - method_value) / baseline_value
                for metric in QUALITY_METRICS:
                    method_value = to_float(method_row.get(metric))
                    baseline_value = to_float(baseline_row.get(metric))
                    rec[f"{metric}_method"] = method_value
                    rec[f"{metric}_baseline"] = baseline_value
                    rec[f"{metric}_delta"] = method_value - baseline_value
                out.append(rec)
    return out


def _improved(metric: str, delta: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return bool(delta < 0)
    if metric in HIGHER_IS_BETTER:
        return bool(delta > 0)
    return bool(delta < 0)


def aggregate(rows: list[dict[str, object]], group_fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        rec = {field: value for field, value in zip(group_fields, key)}
        rec["runs"] = len(group)
        for metric in COMPARISON_METRICS:
            deltas = np.asarray([float(row[f"{metric}_delta"]) for row in group], dtype=float)
            deltas = deltas[~np.isnan(deltas)]
            if deltas.size == 0:
                continue
            rec[f"{metric}_mean_delta"] = float(np.mean(deltas))
            rec[f"{metric}_median_delta"] = float(np.median(deltas))
            rec[f"{metric}_std_delta"] = float(np.std(deltas))
            rec[f"{metric}_improved_runs"] = int(np.sum([row[f"{metric}_improved"] for row in group]))
            if metric == "pehe":
                method_values = np.asarray([to_float(row.get("pehe_method")) for row in group], dtype=float)
                baseline_values = np.asarray([to_float(row.get("pehe_baseline")) for row in group], dtype=float)
                mask = ~np.isnan(method_values) & ~np.isnan(baseline_values) & (baseline_values != 0)
                if int(np.sum(mask)) > 0:
                    rec["relative_pehe_reduction_pct_aggregate"] = float(
                        100.0 * (np.mean(baseline_values[mask]) - np.mean(method_values[mask])) / np.mean(baseline_values[mask])
                    )
        rel = np.asarray([to_float(row.get("relative_pehe_reduction_pct")) for row in group], dtype=float)
        rel = rel[~np.isnan(rel)]
        if rel.size:
            rec["relative_pehe_reduction_pct_per_run_mean"] = float(np.mean(rel))
            rec["relative_pehe_reduction_pct_per_run_median"] = float(np.median(rel))
        for quality in QUALITY_METRICS:
            quality_deltas = np.asarray([to_float(row.get(f"{quality}_delta")) for row in group], dtype=float)
            quality_deltas = quality_deltas[~np.isnan(quality_deltas)]
            if quality_deltas.size:
                rec[f"{quality}_mean_delta"] = float(np.mean(quality_deltas))
                rec[f"{quality}_median_delta"] = float(np.median(quality_deltas))
        out.append(rec)
    out.sort(key=lambda row: (str(row.get("baseline", "")), str(row.get("method", "")), str(row.get("context_size", ""))))
    return out


def bootstrap_tests(rows: list[dict[str, object]], group_fields: tuple[str, ...], n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        rec_base = {field: value for field, value in zip(group_fields, key)}
        for metric in ["pehe", "ate_error", "policy_value", "policy_regret"]:
            deltas = np.asarray([float(row[f"{metric}_delta"]) for row in group], dtype=float)
            deltas = deltas[~np.isnan(deltas)]
            if deltas.size < 2:
                continue
            means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(deltas, size=deltas.size, replace=True)
                means.append(float(np.mean(sample)))
            means_arr = np.asarray(means)
            std = float(np.std(deltas, ddof=1))
            se = std / math.sqrt(deltas.size) if deltas.size else float("nan")
            t_stat = float(np.mean(deltas) / se) if se > 0 else float("inf")
            p_normal = math.erfc(abs(t_stat) / math.sqrt(2.0)) if math.isfinite(t_stat) else 0.0
            wilcoxon_stat, wilcoxon_p = wilcoxon_signed_rank_normal_approx(deltas)
            rec = {
                **rec_base,
                "metric": metric,
                "runs": int(deltas.size),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "bootstrap_ci_low": float(np.quantile(means_arr, 0.025)),
                "bootstrap_ci_high": float(np.quantile(means_arr, 0.975)),
                "paired_t_stat": t_stat,
                "paired_t_normal_approx_p": p_normal,
                "wilcoxon_signed_rank_stat": wilcoxon_stat,
                "wilcoxon_normal_approx_p": wilcoxon_p,
                "improved_runs": int(np.sum([_improved(metric, delta) for delta in deltas])),
                "win_rate": float(np.mean([_improved(metric, delta) for delta in deltas])),
            }
            out.append(rec)
    return out


def wilcoxon_signed_rank_normal_approx(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values != 0]
    n = values.size
    if n == 0:
        return float("nan"), float("nan")
    abs_values = np.abs(values)
    order = np.argsort(abs_values)
    ranks = np.empty(n, dtype=float)
    sorted_abs = abs_values[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_abs[end] == sorted_abs[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    w_positive = float(np.sum(ranks[values > 0]))
    w_negative = float(np.sum(ranks[values < 0]))
    stat = min(w_positive, w_negative)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return stat, float("nan")
    z = (w_positive - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return stat, float(p)


def correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["baseline"])].append(row)
    for (method, baseline), group in sorted(groups.items()):
        improvement = np.asarray([-float(row["pehe_delta"]) for row in group], dtype=float)
        for quality in QUALITY_METRICS:
            values = np.asarray([to_float(row.get(f"{quality}_delta")) for row in group], dtype=float)
            mask = ~np.isnan(values) & ~np.isnan(improvement)
            if int(np.sum(mask)) < 3 or float(np.std(values[mask])) == 0 or float(np.std(improvement[mask])) == 0:
                continue
            corr = float(np.corrcoef(values[mask], improvement[mask])[0, 1])
            out.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "quality_delta": quality,
                    "runs": int(np.sum(mask)),
                    "pearson_corr_with_pehe_improvement": corr,
                }
            )
    out.sort(key=lambda row: abs(float(row["pearson_corr_with_pehe_improvement"])), reverse=True)
    return out


def write_reports(output_dir: Path, deltas: list[dict[str, object]], grouped: list[dict[str, object]], tests: list[dict[str, object]], corr: list[dict[str, object]]) -> None:
    matched = [row for row in grouped if row.get("method") in PAPER_METHODS and row.get("baseline") in DEFAULT_BASELINES]
    lines = [
        "# Diagnostic Report",
        "",
        "Negative PEHE/ATE/policy-regret deltas are better. Positive policy-value deltas are better.",
        "Relative PEHE reduction in the main tables is computed from aggregate means; the per-run percentage is kept only as a diagnostic because it can be unstable when a baseline PEHE is very small.",
        "",
        "## Paper Methods Overall",
        "",
        "| Method | Baseline | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in matched:
        lines.append(
            f"| `{row['method']}` | `{row['baseline']}` | {row['runs']} | {row.get('pehe_improved_runs', '')}/{row['runs']} | "
            f"{float(row.get('pehe_mean_delta', float('nan'))):+.4f} | {float(row.get('relative_pehe_reduction_pct_aggregate', float('nan'))):+.2f}% | "
            f"{row.get('ate_error_improved_runs', '')}/{row['runs']} | {float(row.get('ate_error_mean_delta', float('nan'))):+.4f} | "
            f"{row.get('policy_value_improved_runs', '')}/{row['runs']} | {float(row.get('policy_value_mean_delta', float('nan'))):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap / Paired Tests",
            "",
            "| Method | Baseline | Metric | Runs | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate | Significant? |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in tests:
        if row.get("method") not in {"matched_pair", "matched_pair_diverse"} or row.get("baseline") not in DEFAULT_BASELINES or row.get("metric") not in {"pehe", "ate_error", "policy_value", "policy_regret"}:
            continue
        significant = "yes" if float(row["bootstrap_ci_low"]) * float(row["bootstrap_ci_high"]) > 0 and float(row["paired_t_normal_approx_p"]) < 0.05 and float(row["wilcoxon_normal_approx_p"]) < 0.05 else "mixed/no"
        lines.append(
            f"| `{row['method']}` | `{row['baseline']}` | `{row['metric']}` | {row['runs']} | "
            f"{float(row['mean_delta']):+.4f} | {float(row['median_delta']):+.4f} | "
            f"[{float(row['bootstrap_ci_low']):+.4f}, {float(row['bootstrap_ci_high']):+.4f}] | "
            f"{float(row['paired_t_normal_approx_p']):.4g} | {float(row['wilcoxon_normal_approx_p']):.4g} | "
            f"{float(row['win_rate']):.2%} | {significant} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval Quality Correlations",
            "",
            "| Method | Baseline | Quality Delta | Runs | Corr With PEHE Improvement |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in corr[:20]:
        lines.append(
            f"| `{row['method']}` | `{row['baseline']}` | `{row['quality_delta']}` | {row['runs']} | "
            f"{float(row['pearson_corr_with_pehe_improvement']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `matched_pair_diverse` is the promoted method when its PEHE bootstrap CI stays below zero against random, kNN, and the original matched pair.",
            "- ATE/policy results should be treated as secondary unless their paired intervals exclude zero.",
            "- `overlap_aware` should remain a negative/control result unless its redesigned overlap-filtered matching variant is consistently better.",
        ]
    )
    (output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Paper-Style Summary",
        "",
        "## Strongest Claim",
        "",
        "Diverse matched treated-control context selection consistently improves PEHE for context-limited tabular ICL-style CATE estimation over random and kNN retrieval. The original matched pair remains the simpler baseline; the diverse variant is the promoted method.",
        "",
        "## Weak Claim",
        "",
        "ATE and policy-value gains are more modest and scenario-dependent. They should be reported as secondary diagnostics rather than the main claim.",
        "",
        "## Negative Result",
        "",
        "The first `overlap_aware` retriever often hurts PEHE because filtering toward common support alone can discard local similarity. The redesigned `overlap_filtered_matched_pair` should be compared against it, but plain overlap filtering is not promoted.",
        "",
        "## What Should Not Be Claimed",
        "",
        "- Do not claim a new causal PFN.",
        "- Do not claim superiority over full-data classical causal learners.",
        "- Do not claim ATE/policy improvements are as stable as PEHE improvements unless paired intervals support that.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    pehe_tests = [
        row
        for row in tests
        if row.get("method") == "matched_pair_diverse"
        and row.get("baseline") in {"random", "knn", "matched_pair"}
        and row.get("metric") == "pehe"
    ]
    audit_lines = [
        "# Paper-Ready Audit",
        "",
        "## Strongest Claim",
        "",
        "The supported main claim is about retrieval: `matched_pair_diverse` improves PEHE for context-limited ICL-style CATE estimation compared with random and kNN context retrieval. This is not evidence for a newly trained causal PFN and not yet final evidence for a real TabICL/TabPFN backend.",
        "",
        "## PEHE Significance Summary",
        "",
        "| Comparison | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(pehe_tests, key=lambda item: item["baseline"]):
        audit_lines.append(
            f"| `matched_pair_diverse` vs `{row['baseline']}` | {float(row['mean_delta']):+.4f} | "
            f"{float(row['median_delta']):+.4f} | [{float(row['bootstrap_ci_low']):+.4f}, {float(row['bootstrap_ci_high']):+.4f}] | "
            f"{float(row['paired_t_normal_approx_p']):.4g} | {float(row['wilcoxon_normal_approx_p']):.4g} | {float(row['win_rate']):.2%} |"
        )
    audit_lines.extend(
        [
            "",
            "## Weak Claim",
            "",
            "ATE error and policy value are secondary. They should be described as modest or scenario-dependent unless their paired confidence intervals and signed tests are consistently significant.",
            "",
            "## Negative Result",
            "",
            "`overlap_aware` and `overlap_filtered_matched_pair` are negative controls. Their failure supports the mechanism explanation that common-support filtering alone sacrifices local similarity and does not solve context selection.",
            "",
            "## Relative PEHE Reduction Audit",
            "",
            "Main tables use aggregate relative PEHE reduction: `(mean baseline PEHE - mean method PEHE) / mean baseline PEHE`. Per-run relative percentages are retained only as diagnostics because they can flip sign when baseline PEHE is tiny or highly variable.",
            "",
            "## Backend Boundary",
            "",
            "Current evidence uses the local ridge T-learner fallback behind the ICL-style wrapper. It isolates context retrieval, not final TabICL/TabPFN model-level performance.",
            "",
            "## What Should Not Be Claimed",
            "",
            "- Do not claim a new causal PFN.",
            "- Do not claim full-data classical causal learners are beaten overall.",
            "- Do not claim ATE/policy gains are as robust as PEHE gains.",
            "- Do not describe fallback runs as real TabICL/TabPFN backend evidence.",
        ]
    )
    (output_dir.parent / "paper_ready_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def plot_pehe_vs_context(rows: list[dict[str, str]], output: Path) -> None:
    context_rows = [row for row in rows if row.get("estimator") == "local_context_t_learner"]
    groups = defaultdict(list)
    for row in context_rows:
        groups[(row["retriever"], int(row["context_size"]))].append(to_float(row["pehe"]))
    plt.figure(figsize=(9, 5.5))
    for retriever in sorted({key[0] for key in groups}):
        if retriever not in {"random", "knn", "treatment_balanced_random", "propensity_nearest", "matched_pair", "matched_pair_diverse", "overlap_aware"}:
            continue
        xs = sorted({key[1] for key in groups if key[0] == retriever})
        ys = [float(np.mean(groups[(retriever, x)])) for x in xs]
        plt.plot(xs, ys, marker="o", label=retriever)
    plt.xlabel("Context size")
    plt.ylabel("Mean PEHE")
    plt.title("PEHE vs context size")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=170)
    plt.close()


def plot_heatmap(deltas: list[dict[str, object]], output: Path, method: str = "matched_pair_diverse", baseline: str = "random") -> None:
    rows = [row for row in deltas if row["method"] == method and row["baseline"] == baseline]
    numeric_rows = [
        row
        for row in rows
        if str(row.get("treatment_imbalance", "")) != "" and str(row.get("confounding_strength", "")) != ""
    ]
    xs = sorted({float(row["treatment_imbalance"]) for row in numeric_rows})
    ys = sorted({float(row["confounding_strength"]) for row in numeric_rows})
    if not xs or not ys:
        return
    grid = np.full((len(ys), len(xs)), np.nan)
    for yi, conf in enumerate(ys):
        for xi, imb in enumerate(xs):
            vals = [float(row["pehe_delta"]) for row in numeric_rows if float(row["confounding_strength"]) == conf and float(row["treatment_imbalance"]) == imb]
            if vals:
                grid[yi, xi] = float(np.mean(vals))
    plt.figure(figsize=(6.5, 5))
    plt.imshow(grid, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="PEHE delta")
    plt.xticks(range(len(xs)), [str(x) for x in xs])
    plt.yticks(range(len(ys)), [str(y) for y in ys])
    plt.xlabel("Treatment imbalance")
    plt.ylabel("Confounding strength")
    plt.title(f"{method} vs {baseline}: PEHE delta")
    for yi in range(len(ys)):
        for xi in range(len(xs)):
            if not np.isnan(grid[yi, xi]):
                plt.text(xi, yi, f"{grid[yi, xi]:+.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=170)
    plt.close()


def plot_delta_by_scenario(deltas: list[dict[str, object]], output: Path, metric: str, method: str = "matched_pair_diverse", baseline: str = "random") -> None:
    rows = [row for row in deltas if row["method"] == method and row["baseline"] == baseline]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("regime") or f"c{row['confounding_strength']}_i{row['treatment_imbalance']}_o{row['overlap_violation']}"].append(float(row[f"{metric}_delta"]))
    labels = sorted(groups)
    means = [float(np.mean(groups[label])) for label in labels]
    plt.figure(figsize=(8, 4.5))
    colors = ["#2f7d32" if (value < 0 if metric != "policy_value" else value > 0) else "#b23a2f" for value in means]
    plt.bar(labels, means, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel(f"{metric} delta")
    plt.title(f"{method} vs {baseline}: {metric} delta by scenario")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=170)
    plt.close()


def plot_pehe_delta_by_context(deltas: list[dict[str, object]], output: Path, method: str = "matched_pair_diverse") -> None:
    rows = [row for row in deltas if row.get("method") == method and row.get("baseline") in {"random", "knn", "matched_pair"}]
    if not rows:
        return
    groups = defaultdict(list)
    for row in rows:
        groups[(row["baseline"], int(row["context_size"]))].append(float(row["pehe_delta"]))
    plt.figure(figsize=(8, 5))
    for baseline in sorted({key[0] for key in groups}):
        xs = sorted({key[1] for key in groups if key[0] == baseline})
        ys = [float(np.mean(groups[(baseline, x)])) for x in xs]
        plt.plot(xs, ys, marker="o", label=f"vs {baseline}")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Context size")
    plt.ylabel("Mean PEHE delta")
    plt.title(f"{method}: PEHE delta by context size")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=170)
    plt.close()


def plot_bootstrap_forest(tests: list[dict[str, object]], output: Path) -> None:
    rows = [
        row
        for row in tests
        if row.get("method") in {"matched_pair_diverse", "matched_pair"}
        and row.get("baseline") in {"random", "knn", "matched_pair"}
        and row.get("metric") in {"pehe", "ate_error", "policy_value"}
    ]
    if not rows:
        return
    rows.sort(key=lambda row: (row["metric"], row["method"], row["baseline"]))
    labels = [f"{row['method']} vs {row['baseline']} / {row['metric']}" for row in rows]
    means = np.asarray([float(row["mean_delta"]) for row in rows])
    lows = np.asarray([float(row["bootstrap_ci_low"]) for row in rows])
    highs = np.asarray([float(row["bootstrap_ci_high"]) for row in rows])
    y = np.arange(len(rows))
    plt.figure(figsize=(9, max(4.5, 0.35 * len(rows))))
    plt.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", color="#2f5d8c", ecolor="#6d8fb3")
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.yticks(y, labels, fontsize=7)
    plt.xlabel("Mean paired delta with 95% bootstrap CI")
    plt.title("Bootstrap CI forest plot")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=170)
    plt.close()


def plot_quality_scatter(deltas: list[dict[str, object]], output_dir: Path, method: str = "matched_pair_diverse", baseline: str = "random") -> None:
    plot_specs = [
        ("meta_mean_context_abs_smd_delta", "SMD delta"),
        ("meta_mean_context_propensity_distance_delta", "Propensity-distance delta"),
        ("meta_mean_context_feature_distance_delta", "Feature-distance delta"),
        ("meta_mean_context_diversity_delta", "Diversity delta"),
        ("meta_mean_context_duplicate_rate_delta", "Duplicate-rate delta"),
    ]
    rows = [row for row in deltas if row.get("method") == method and row.get("baseline") == baseline]
    if not rows:
        return
    y = np.asarray([-float(row["pehe_delta"]) for row in rows], dtype=float)
    for key, label in plot_specs:
        x = np.asarray([to_float(row.get(key)) for row in rows], dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if int(np.sum(mask)) < 3:
            continue
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if float(np.std(x[mask])) > 0 and float(np.std(y[mask])) > 0 else float("nan")
        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(x[mask], y[mask], alpha=0.55, s=18, color="#2f5d8c")
        plt.axhline(0.0, color="black", linewidth=0.7)
        plt.axvline(0.0, color="black", linewidth=0.7)
        plt.xlabel(label)
        plt.ylabel("PEHE improvement (-delta)")
        plt.title(f"{method} vs {baseline}: corr={corr:+.2f}")
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        safe = key.replace("meta_", "").replace("_delta", "")
        plt.savefig(output_dir / f"quality_scatter_{safe}.png", dpi=170)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose causal context retrieval gains and failure modes.")
    parser.add_argument("--input", default=str(ROOT / "results" / "synthetic_default" / "synthetic_results.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    plots_dir = output_dir / "plots"
    rows = read_rows(input_path)

    methods = tuple(name for name in DEFAULT_METHODS if any(row.get("retriever") == name for row in rows))
    deltas = paired_deltas(rows, methods=methods, baselines=DEFAULT_BASELINES)
    write_rows(output_dir / "scenario_level_deltas.csv", deltas)

    grouped_overall = aggregate(deltas, ("method", "baseline"))
    grouped_context = aggregate(deltas, ("method", "baseline", "context_size"))
    grouped_regime = aggregate(deltas, ("method", "baseline", "regime"))
    grouped_confounding = aggregate(deltas, ("method", "baseline", "confounding_strength"))
    grouped_imbalance = aggregate(deltas, ("method", "baseline", "treatment_imbalance"))
    grouped_overlap = aggregate(deltas, ("method", "baseline", "overlap_violation"))
    write_rows(output_dir / "diagnostic_overall.csv", grouped_overall)
    write_rows(output_dir / "diagnostic_by_context_size.csv", grouped_context)
    write_rows(output_dir / "diagnostic_by_regime.csv", grouped_regime)
    write_rows(output_dir / "diagnostic_by_confounding.csv", grouped_confounding)
    write_rows(output_dir / "diagnostic_by_imbalance.csv", grouped_imbalance)
    write_rows(output_dir / "diagnostic_by_overlap.csv", grouped_overlap)

    tests = bootstrap_tests(deltas, ("method", "baseline"), n_bootstrap=args.bootstrap, seed=123)
    write_rows(output_dir / "paired_bootstrap_tests.csv", tests)
    corr = correlations(deltas)
    write_rows(output_dir / "retrieval_quality_correlations.csv", corr)

    plot_pehe_vs_context(rows, plots_dir / "pehe_vs_context_size.png")
    plot_heatmap(deltas, plots_dir / "pehe_delta_heatmap_confounding_imbalance.png")
    plot_delta_by_scenario(deltas, plots_dir / "ate_error_delta_by_scenario.png", metric="ate_error")
    plot_delta_by_scenario(deltas, plots_dir / "policy_value_delta_by_scenario.png", metric="policy_value")
    plot_delta_by_scenario(deltas, plots_dir / "pehe_delta_by_regime.png", metric="pehe")
    plot_pehe_delta_by_context(deltas, plots_dir / "pehe_delta_by_context_size.png")
    plot_bootstrap_forest(tests, plots_dir / "bootstrap_ci_forest.png")
    plot_quality_scatter(deltas, plots_dir)
    write_reports(output_dir, deltas, grouped_overall, tests, corr)
    print(f"[experiment3] wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
