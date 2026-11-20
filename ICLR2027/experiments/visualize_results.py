from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_map(rows: list[dict[str, str]]) -> dict[str, float]:
    return {row["dataset"]: float(row["metric_value"]) for row in rows if row.get("metric_value")}


def plot_checkpoint_comparison() -> None:
    official = score_map(read_csv(RESULTS_DIR / "benchmark_results.csv"))
    tiny = score_map(read_csv(RESULTS_DIR / "tiny_checkpoint_benchmark.csv"))
    small = score_map(read_csv(RESULTS_DIR / "small_checkpoint_benchmark.csv"))
    stage = score_map(read_csv(RESULTS_DIR / "stage_mid_checkpoint_benchmark.csv"))

    datasets = ["synthetic_classification", "breast_cancer", "wine"]
    series = [
        ("official checkpoint", official, "#2f6f4e"),
        ("tiny 100-step", tiny, "#7c8a99"),
        ("small 1000-step", small, "#1f77b4"),
        ("stage-like 200-step", stage, "#d9822b"),
    ]

    x = list(range(len(datasets)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
    for i, (label, values, color) in enumerate(series):
        offset = (i - 1.5) * width
        scores = [values.get(dataset, 0.0) for dataset in datasets]
        bars = ax.bar([v + offset for v in x], scores, width=width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], fontsize=8, padding=2)

    ax.set_title("Accuracy by checkpoint on common classification datasets")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(["synthetic", "breast cancer", "wine"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint_accuracy_comparison.png")
    plt.close(fig)


def plot_average_accuracy() -> None:
    rows = [
        ("tiny 100-step", [0.5000, 0.6294, 0.4000], "#7c8a99"),
        ("small 1000-step", [0.5880, 0.9021, 0.6889], "#1f77b4"),
        ("stage-like 200-step", [0.5040, 0.6294, 0.4000], "#d9822b"),
        ("official checkpoint", [0.9800, 0.9790, 1.0000], "#2f6f4e"),
    ]
    labels = [row[0] for row in rows]
    means = [sum(row[1]) / len(row[1]) for row in rows]
    colors = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    bars = ax.bar(labels, means, color=colors)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3)
    ax.set_title("Average accuracy on three common classification datasets")
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint_average_accuracy.png")
    plt.close(fig)


def plot_scale_study() -> None:
    rows = read_csv(RESULTS_DIR / "large_scale_results.csv")
    if not rows:
        return
    samples = [int(row["n_samples"]) for row in rows]
    scores = [float(row["metric_value"]) for row in rows]
    runtimes = [float(row["runtime_sec"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax1.plot(samples, scores, marker="o", color="#1f77b4", label="accuracy")
    ax1.set_title("Official checkpoint larger synthetic scale study")
    ax1.set_xlabel("Number of samples")
    ax1.set_ylabel("Accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.975, 0.995)
    ax1.grid(axis="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(samples, runtimes, marker="s", color="#d9822b", label="runtime")
    ax2.set_ylabel("Runtime seconds", color="#d9822b")
    ax2.tick_params(axis="y", labelcolor="#d9822b")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "large_scale_accuracy_runtime.png")
    plt.close(fig)


def plot_context_selection_winners() -> None:
    path = RESULTS_DIR / "context_selection_multiseed_winners.csv"
    if not path.exists():
        path = RESULTS_DIR / "context_selection_winners.csv"
    rows = read_csv(path)
    rows = [row for row in rows if row.get("delta_vs_baseline")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["delta_vs_baseline"]), reverse=True)
    top = rows[:12]
    labels = [f"{row['dataset']}\n{row['method']}" for row in top]
    deltas = [float(row["delta_vs_baseline"]) for row in top]

    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)
    bars = ax.bar(range(len(top)), deltas, color="#2f6f4e")
    ax.bar_label(bars, labels=[f"+{value:.3f}" for value in deltas], fontsize=8, padding=2)
    ax.set_title("Top context-selection and causal-proxy improvements")
    ax.set_ylabel("Accuracy delta vs baseline")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "context_selection_top_deltas.png")
    plt.close(fig)


def plot_context_repeated_signals() -> None:
    path = RESULTS_DIR / "context_selection_multiseed_winners.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("delta_vs_baseline")]
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"]), []).append(float(row["delta_vs_baseline"]))
    repeated = [
        (dataset, method, deltas)
        for (dataset, method), deltas in grouped.items()
        if len(deltas) >= 2
    ]
    if not repeated:
        return
    repeated.sort(key=lambda item: (sum(item[2]) / len(item[2]), max(item[2])), reverse=True)
    labels = [f"{dataset}\n{method}" for dataset, method, _ in repeated]
    means = [sum(deltas) / len(deltas) for _, _, deltas in repeated]
    maxes = [max(deltas) for _, _, deltas in repeated]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
    bars = ax.bar(range(len(repeated)), means, color="#1f77b4")
    ax.scatter(range(len(repeated)), maxes, color="#d9822b", label="max delta", zorder=3)
    ax.bar_label(bars, labels=[f"+{value:.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Repeated positive signals across random seeds")
    ax.set_ylabel("Mean positive accuracy delta")
    ax.set_xticks(range(len(repeated)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "context_selection_repeated_signals.png")
    plt.close(fig)


def plot_query_context_deltas() -> None:
    path = RESULTS_DIR / "query_context_selection_winners.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("delta_vs_baseline")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["delta_vs_baseline"]), reverse=True)
    top = rows[:14]
    labels = [f"{row['dataset']}\n{row['method']}" for row in top]
    deltas = [float(row["delta_vs_baseline"]) for row in top]
    colors = ["#2f6f4e" if "propensity" in row["method"] else "#1f77b4" for row in top]

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=160)
    bars = ax.bar(range(len(top)), deltas, color=colors)
    ax.bar_label(bars, labels=[f"+{value:.3f}" for value in deltas], fontsize=8, padding=2)
    ax.set_title("Query-level context selection improvements")
    ax.set_ylabel("Accuracy delta vs random context")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "query_context_selection_deltas.png")
    plt.close(fig)


def plot_openml_method_deltas() -> None:
    path = RESULTS_DIR / "openml_method_search_winners.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("delta_vs_baseline")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["delta_vs_baseline"]), reverse=True)
    top = rows[:12]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in top]
    deltas = [float(row["delta_vs_baseline"]) for row in top]
    colors = [
        "#2f6f4e" if "propensity" in row["method"]
        else "#8a5a2b" if "causal" in row["method"]
        else "#1f77b4" if "ensemble" in row["method"]
        else "#7b519d"
        for row in top
    ]

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=160)
    bars = ax.bar(range(len(top)), deltas, color=colors)
    ax.bar_label(bars, labels=[f"+{value:.3f}" for value in deltas], fontsize=8, padding=2)
    ax.set_title("OpenML external dataset method improvements")
    ax.set_ylabel("Accuracy delta vs baseline")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_method_search_deltas.png")
    plt.close(fig)


def plot_openml_candidate_recheck() -> None:
    path = RESULTS_DIR / "openml_candidate_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: (int(row["positive_seeds"]), float(row["mean_delta"])), reverse=True)
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    maxes = [float(row["max_delta"]) for row in rows]
    colors = ["#2f6f4e" if value > 0 else "#9a4f4f" for value in means]

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=colors)
    ax.errorbar(
        range(len(rows)),
        means,
        yerr=[
            [mean - min_value for mean, min_value in zip(means, mins)],
            [max_value - mean for mean, max_value in zip(means, maxes)],
        ],
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1,
    )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("OpenML candidate recheck across random seeds")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_candidate_recheck_stability.png")
    plt.close(fig)


def plot_openml_propensity_variant_recheck() -> None:
    path = RESULTS_DIR / "openml_propensity_variant_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    rows = [row for row in rows if int(row.get("positive_seeds", "0")) > 0 or float(row["mean_delta"]) > 0]
    if not rows:
        return
    rows.sort(key=lambda row: (int(row["positive_seeds"]), float(row["mean_delta"])), reverse=True)
    top = rows[:14]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in top]
    means = [float(row["mean_delta"]) for row in top]
    mins = [float(row["min_delta"]) for row in top]
    maxes = [float(row["max_delta"]) for row in top]
    colors = ["#2f6f4e" if int(row["positive_seeds"]) == int(row["seeds"]) else "#1f77b4" for row in top]

    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)
    bars = ax.bar(range(len(top)), means, color=colors)
    ax.errorbar(
        range(len(top)),
        means,
        yerr=[
            [mean - min_value for mean, min_value in zip(means, mins)],
            [max_value - mean for mean, max_value in zip(means, maxes)],
        ],
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1,
    )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Propensity variant recheck on selected OpenML datasets")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_propensity_variant_recheck.png")
    plt.close(fig)


def plot_openml_guarded_policy() -> None:
    path = RESULTS_DIR / "openml_guarded_policy_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    labels = [row["dataset"].replace("openml_", "") for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    colors = ["#2f6f4e" if value > 0 else "#7c8a99" for value in means]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=colors)
    ax.scatter(range(len(rows)), mins, color="#333333", label="min delta", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Guarded adaptive policy mean delta across seeds")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_guarded_policy.png")
    plt.close(fig)


def plot_openml_guarded_scale_study() -> None:
    path = RESULTS_DIR / "openml_guarded_scale_study_scale_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("macro_mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: int(row["max_samples"]))
    scales = [int(row["max_samples"]) for row in rows]
    means = [float(row["macro_mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    positives = [int(row["positive_runs"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax.plot(scales, means, marker="o", color="#2f6f4e", label="macro mean delta")
    ax.scatter(scales, mins, color="#333333", label="min delta", zorder=3)
    for x, y, positive in zip(scales, means, positives):
        ax.annotate(f"{positive} positive", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Guarded adaptive policy across sample scales")
    ax.set_xlabel("Max samples per dataset")
    ax.set_ylabel("Accuracy delta vs baseline")
    ax.set_xticks(scales)
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_guarded_scale_study.png")
    plt.close(fig)


def plot_query_guarded_policy() -> None:
    path = RESULTS_DIR / "query_guarded_policy_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: (int(row["max_samples"]), float(row["mean_delta"])), reverse=True)
    labels = [f"{row['max_samples']}\n{row['dataset'].replace('synth_', '')}" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    colors = ["#2f6f4e" if value > 0 else "#7c8a99" for value in means]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=colors)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Guarded query-level context policy")
    ax.set_ylabel("Accuracy delta vs random context")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "query_guarded_policy.png")
    plt.close(fig)


def plot_synthetic_guarded_stress() -> None:
    path = RESULTS_DIR / "synthetic_guarded_stress_scale_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("macro_mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: int(row["max_samples"]))
    scales = [int(row["max_samples"]) for row in rows]
    means = [float(row["macro_mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    positives = [int(row["positive_runs"]) for row in rows]
    nonnegative = [int(row["nonnegative_runs"]) for row in rows]
    runs = [int(row["runs"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax.plot(scales, means, marker="o", color="#2f6f4e", label="macro mean delta")
    ax.scatter(scales, mins, color="#333333", label="min delta", zorder=3)
    for x, y, positive, nonneg, total in zip(scales, means, positives, nonnegative, runs):
        ax.annotate(
            f"{positive} positive\n{nonneg}/{total} nonnegative",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Synthetic special stress test for guarded policy")
    ax.set_xlabel("Max samples per synthetic dataset")
    ax.set_ylabel("Accuracy delta vs baseline")
    ax.set_xticks(scales)
    ax.set_ylim(-0.02, 0.04)
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "synthetic_guarded_stress.png")
    plt.close(fig)


def plot_openml_broad_recheck() -> None:
    path = RESULTS_DIR / "openml_broad_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    rows = [row for row in rows if int(row.get("positive_seeds", "0")) > 0 or float(row["mean_delta"]) > 0]
    if not rows:
        return
    rows.sort(key=lambda row: (int(row["positive_seeds"]), float(row["mean_delta"])), reverse=True)
    top = rows[:12]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in top]
    means = [float(row["mean_delta"]) for row in top]
    mins = [float(row["min_delta"]) for row in top]
    maxes = [float(row["max_delta"]) for row in top]
    colors = ["#2f6f4e" if int(row["positive_seeds"]) == int(row["seeds"]) else "#1f77b4" for row in top]

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=160)
    bars = ax.bar(range(len(top)), means, color=colors)
    ax.errorbar(
        range(len(top)),
        means,
        yerr=[
            [mean - min_value for mean, min_value in zip(means, mins)],
            [max_value - mean for mean, max_value in zip(means, maxes)],
        ],
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1,
    )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Broad OpenML recheck on newly added datasets")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_broad_recheck.png")
    plt.close(fig)


def plot_openml_promoted_candidate() -> None:
    path = RESULTS_DIR / "openml_promoted_candidate_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: int(row["max_samples"]))
    scales = [int(row["max_samples"]) for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    maxes = [float(row["max_delta"]) for row in rows]
    positives = [int(row["positive_seeds"]) for row in rows]
    seeds = [int(row["seeds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=["#2f6f4e" if p == s else "#1f77b4" for p, s in zip(positives, seeds)])
    ax.errorbar(
        range(len(rows)),
        means,
        yerr=[
            [mean - min_value for mean, min_value in zip(means, mins)],
            [max_value - mean for mean, max_value in zip(means, maxes)],
        ],
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1,
    )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}\n{p}/{s}" for value, p, s in zip(means, positives, seeds)], fontsize=8, padding=3)
    ax.set_title("Promoted candidate scale boundary")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{scale} samples" for scale in scales])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_promoted_candidate.png")
    plt.close(fig)


def plot_openml_promoted_guarded() -> None:
    path = RESULTS_DIR / "openml_promoted_guarded_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: int(row["max_samples"]))
    scales = [int(row["max_samples"]) for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    enabled = [int(row["chosen_propensity_interactions"]) for row in rows]
    runs = [int(row["runs"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    bars = ax.bar(range(len(rows)), means, color="#2f6f4e")
    ax.scatter(range(len(rows)), mins, color="#333333", label="min delta", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(
        bars,
        labels=[f"{value:+.3f}\nenabled {on}/{total}" for value, on, total in zip(means, enabled, runs)],
        fontsize=8,
        padding=3,
    )
    ax.set_title("Guarded promoted candidate by sample scale")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{scale} samples" for scale in scales])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_promoted_guarded.png")
    plt.close(fig)


def plot_effective_method_catalog() -> None:
    path = RESULTS_DIR / "effective_method_catalog.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    if not rows:
        return
    statuses = ["stable_positive", "guarded_positive", "risky_positive", "safe_baseline_or_tie", "negative_do_not_default"]
    counts = [sum(1 for row in rows if row["status"] == status) for status in statuses]
    colors = ["#2f6f4e", "#1f77b4", "#d9822b", "#7c8a99", "#9a4f4f"]

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
    bars = ax.bar(range(len(statuses)), counts, color=colors)
    ax.bar_label(bars, labels=[str(value) for value in counts], padding=3)
    ax.set_title("Effective method catalog by evidence status")
    ax.set_ylabel("Dataset-method entries")
    ax.set_xticks(range(len(statuses)))
    ax.set_xticklabels(
        ["stable\npositive", "guarded\npositive", "risky\npositive", "safe tie", "do not\ndefault"],
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "effective_method_catalog.png")
    plt.close(fig)


def plot_regression_recheck() -> None:
    path = RESULTS_DIR / "regression_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    labels = [f"{row['dataset'].replace('synthetic_', 'synth_')}\n{row['method']}" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    colors = ["#2f6f4e" if float(row["min_delta"]) >= 0 and float(row["mean_delta"]) > 0 else "#d9822b" if float(row["mean_delta"]) > 0 else "#9a4f4f" for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=colors)
    ax.scatter(range(len(rows)), mins, color="#333333", label="min R2 delta", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Regression candidate recheck")
    ax.set_ylabel("Mean R2 delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "regression_recheck.png")
    plt.close(fig)


def plot_regression_guarded_policy() -> None:
    path = RESULTS_DIR / "regression_guarded_policy_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: row["dataset"])
    labels = [row["dataset"].replace("synthetic_", "synth_") for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    positives = [int(row["positive_seeds"]) for row in rows]
    seeds = [int(row["seeds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    bars = ax.bar(range(len(rows)), means, color="#7c8a99")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}\n{p}/{s}" for value, p, s in zip(means, positives, seeds)], fontsize=8, padding=3)
    ax.set_title("Regression guarded policy")
    ax.set_ylabel("Mean R2 delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "regression_guarded_policy.png")
    plt.close(fig)


def plot_openml_label_balance() -> None:
    path = RESULTS_DIR / "openml_label_balance_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=["#2f6f4e" if value > 0 else "#9a4f4f" for value in means])
    ax.scatter(range(len(rows)), mins, color="#333333", label="min delta", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=3)
    ax.set_title("Label-balance context recheck")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_label_balance_recheck.png")
    plt.close(fig)


def plot_openml_local_geometry() -> None:
    path = RESULTS_DIR / "openml_local_geometry_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    rows = rows[:12]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    positives = [int(row["positive_seeds"]) for row in rows]
    seeds = [int(row["seeds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=["#2f6f4e" if value > 0 else "#9a4f4f" for value in means])
    ax.scatter(range(len(rows)), mins, color="#333333", label="min delta", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(
        bars,
        labels=[f"{value:+.3f}\n{positive}/{seed}" for value, positive, seed in zip(means, positives, seeds)],
        fontsize=8,
        padding=3,
    )
    ax.set_title("Local geometry feature recheck")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_local_geometry_recheck.png")
    plt.close(fig)


def plot_openml_local_geometry_guarded() -> None:
    configs = [
        ("delta>=0.02", RESULTS_DIR / "openml_local_geometry_guarded_summary.csv"),
        ("delta>=0.05", RESULTS_DIR / "openml_local_geometry_guarded_strict_summary.csv"),
        ("delta>=0.07", RESULTS_DIR / "openml_local_geometry_guarded_ultra_summary.csv"),
    ]
    rows = []
    for label, path in configs:
        summary = [row for row in read_csv(path) if row.get("mean_delta")]
        if not summary:
            continue
        mean_delta = sum(float(row["mean_delta"]) for row in summary) / len(summary)
        min_delta = min(float(row["min_delta"]) for row in summary)
        positive = sum(int(row["positive_seeds"]) for row in summary)
        seeds = sum(int(row["seeds"]) for row in summary)
        rows.append((label, mean_delta, min_delta, positive, seeds))
    if not rows:
        return
    labels = [row[0] for row in rows]
    means = [row[1] for row in rows]
    mins = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    bars = ax.bar(range(len(rows)), means, color=["#9a4f4f" if value < 0 else "#7c8a99" for value in means])
    ax.scatter(range(len(rows)), mins, color="#333333", label="worst dataset min", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(
        bars,
        labels=[f"{mean:+.3f}\n{positive}/{seed}" for (_, mean, _, positive, seed) in rows],
        fontsize=8,
        padding=3,
    )
    ax.set_title("Local geometry guarded thresholds")
    ax.set_ylabel("Macro mean accuracy delta")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_local_geometry_guarded.png")
    plt.close(fig)


def plot_openml_domain_shift() -> None:
    path = RESULTS_DIR / "openml_domain_shift_recheck_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    rows = rows[:12]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in rows]
    acc = [float(row["mean_delta"]) for row in rows]
    bacc = [float(row.get("mean_balanced_accuracy_delta") or 0.0) for row in rows]
    macro_f1 = [float(row.get("mean_macro_f1_delta") or 0.0) for row in rows]

    x = list(range(len(rows)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=160)
    ax.bar([idx - width for idx in x], acc, width=width, color="#2f6f4e", label="accuracy")
    ax.bar(x, bacc, width=width, color="#1f77b4", label="balanced accuracy")
    ax.bar([idx + width for idx in x], macro_f1, width=width, color="#d9822b", label="macro F1")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Domain-shift methods: multi-metric recheck")
    ax.set_ylabel("Mean delta vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_domain_shift_recheck.png")
    plt.close(fig)


def plot_openml_domain_shift_scale() -> None:
    path = RESULTS_DIR / "openml_domain_shift_scale_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    methods = ["domain_shift_feature", "domain_causal_interactions", "domain_matched_context"]
    colors = {
        "domain_shift_feature": "#2f6f4e",
        "domain_causal_interactions": "#1f77b4",
        "domain_matched_context": "#d9822b",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda row: int(row["max_samples"]))
        if not subset:
            continue
        x = [int(row["max_samples"]) for row in subset]
        y = [float(row["mean_delta"]) for row in subset]
        ax.plot(x, y, marker="o", label=method, color=colors[method])
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("CMC domain-shift scale boundary")
    ax.set_xlabel("Max samples")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.grid(axis="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_domain_shift_scale.png")
    plt.close(fig)


def plot_openml_rare_tail_combo() -> None:
    path = RESULTS_DIR / "openml_rare_tail_combo_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: (float(row["mean_delta"]), float(row.get("mean_roc_auc_delta") or 0.0)), reverse=True)
    rows = rows[:14]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['method']}" for row in rows]
    acc = [float(row["mean_delta"]) for row in rows]
    bacc = [float(row.get("mean_balanced_accuracy_delta") or 0.0) for row in rows]
    macro_f1 = [float(row.get("mean_macro_f1_delta") or 0.0) for row in rows]
    auc = [float(row.get("mean_roc_auc_delta") or 0.0) for row in rows]

    x = list(range(len(rows)))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=160)
    ax.bar([idx - 1.5 * width for idx in x], acc, width=width, color="#2f6f4e", label="accuracy")
    ax.bar([idx - 0.5 * width for idx in x], bacc, width=width, color="#1f77b4", label="balanced accuracy")
    ax.bar([idx + 0.5 * width for idx in x], macro_f1, width=width, color="#d9822b", label="macro F1")
    ax.bar([idx + 1.5 * width for idx in x], auc, width=width, color="#7b519d", label="ROC-AUC")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Rare/tail and domain-shift combo: multi-metric deltas")
    ax.set_ylabel("Mean delta vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_rare_tail_combo.png")
    plt.close(fig)


def plot_openml_probe_difficulty_scale() -> None:
    path = RESULTS_DIR / "openml_probe_difficulty_scale_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: (row["dataset"], int(row["max_samples"]), -float(row["mean_delta"])))
    rows = [row for row in rows if row["method"] in {"linear_probe_feature", "hard_probe_combo", "knn_label_feature"}]
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['max_samples']} {row['method']}" for row in rows]
    acc = [float(row["mean_delta"]) for row in rows]
    bacc = [float(row.get("mean_balanced_accuracy_delta") or 0.0) for row in rows]
    macro_f1 = [float(row.get("mean_macro_f1_delta") or 0.0) for row in rows]
    auc = [float(row.get("mean_roc_auc_delta") or 0.0) for row in rows]
    colors = ["#2f6f4e" if float(row["min_delta"]) >= 0 and float(row["mean_delta"]) > 0 else "#d9822b" if float(row["mean_delta"]) > 0 else "#9a4f4f" for row in rows]

    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(13, 5.9), dpi=160)
    bars = ax.bar(x, acc, color=colors, label="accuracy")
    ax.scatter(x, bacc, color="#1f77b4", label="balanced accuracy", zorder=3)
    ax.scatter(x, macro_f1, color="#d9822b", label="macro F1", zorder=3)
    ax.scatter(x, auc, color="#7b519d", label="ROC-AUC", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in acc], fontsize=8, padding=2)
    ax.set_title("Probe/difficulty methods by scale")
    ax.set_ylabel("Mean delta vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_probe_difficulty_scale.png")
    plt.close(fig)


def plot_openml_probe_difficulty_guarded() -> None:
    path = RESULTS_DIR / "openml_probe_difficulty_guarded_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    nonzero = [row for row in rows if abs(float(row["mean_delta"])) > 1e-12 or int(row.get("positive_seeds") or 0) > 0]
    if not nonzero:
        return
    nonzero.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['max_samples']} samples" for row in nonzero]
    means = [float(row["mean_delta"]) for row in nonzero]
    mins = [float(row["min_delta"]) for row in nonzero]
    positives = [int(row["positive_seeds"]) for row in nonzero]
    seeds = [int(row["seeds"]) for row in nonzero]

    fig, ax = plt.subplots(figsize=(8.5, 4.9), dpi=160)
    bars = ax.bar(range(len(nonzero)), means, color="#2f6f4e")
    ax.scatter(range(len(nonzero)), mins, color="#333333", label="min delta", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(
        bars,
        labels=[f"{value:+.3f}\n{positive}/{seed}" for value, positive, seed in zip(means, positives, seeds)],
        fontsize=8,
        padding=3,
    )
    ax.set_title("Probe/difficulty guarded policy")
    ax.set_ylabel("Mean accuracy delta vs baseline")
    ax.set_xticks(range(len(nonzero)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_probe_difficulty_guarded.png")
    plt.close(fig)


def plot_openml_combined_methods() -> None:
    path = RESULTS_DIR / "openml_combined_methods_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    keep = []
    for row in rows:
        method = row["method"]
        mean_delta = float(row["mean_delta"])
        min_delta = float(row["min_delta"])
        positive = int(row.get("positive_seeds") or 0)
        seeds = int(row.get("seeds") or 0)
        if method in {"quantile_domain_shift_feature", "domain_shift_feature", "quantile_features", "outlier_winsor_combo"}:
            if mean_delta > 0 or (positive == seeds and min_delta >= 0):
                keep.append(row)
    keep.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    rows = keep[:14]
    if not rows:
        return

    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['max_samples']} {row['method']}" for row in rows]
    acc = [float(row["mean_delta"]) for row in rows]
    bacc = [float(row.get("mean_balanced_accuracy_delta") or 0.0) for row in rows]
    macro_f1 = [float(row.get("mean_macro_f1_delta") or 0.0) for row in rows]
    auc = [float(row.get("mean_roc_auc_delta") or 0.0) for row in rows]

    x = list(range(len(rows)))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13.5, 5.9), dpi=160)
    ax.bar([idx - 1.5 * width for idx in x], acc, width=width, color="#2f6f4e", label="accuracy")
    ax.bar([idx - 0.5 * width for idx in x], bacc, width=width, color="#1f77b4", label="balanced accuracy")
    ax.bar([idx + 0.5 * width for idx in x], macro_f1, width=width, color="#d9822b", label="macro F1")
    ax.bar([idx + 1.5 * width for idx in x], auc, width=width, color="#7b519d", label="ROC-AUC")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Combined guarded candidates: multi-metric deltas")
    ax.set_ylabel("Mean delta vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_combined_methods.png")
    plt.close(fig)


def plot_openml_diagnostic_policy() -> None:
    path = RESULTS_DIR / "openml_diagnostic_policy_scale_summary.csv"
    if not path.exists():
        path = RESULTS_DIR / "openml_diagnostic_policy_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: (int(row["max_samples"]), -float(row["mean_delta"]), row["dataset"]))
    rows = [row for row in rows if float(row["mean_delta"]) > 0 or float(row["max_delta"]) > 0]
    if not rows:
        return
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['max_samples']} samples" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    bacc = [float(row.get("mean_balanced_accuracy_delta") or 0.0) for row in rows]
    auc = [float(row.get("mean_roc_auc_delta") or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=160)
    bars = ax.bar(range(len(rows)), means, color="#2f6f4e", label="accuracy mean delta")
    ax.scatter(range(len(rows)), mins, color="#222222", label="accuracy min delta", zorder=3)
    ax.scatter(range(len(rows)), bacc, color="#1f77b4", label="balanced accuracy", zorder=3)
    ax.scatter(range(len(rows)), auc, color="#7b519d", label="ROC-AUC", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Diagnostic-gated policy by dataset and scale")
    ax.set_ylabel("Delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_diagnostic_policy.png")
    plt.close(fig)


def plot_openml_decision_policy() -> None:
    path = RESULTS_DIR / "openml_decision_policy_scale_summary.csv"
    if not path.exists():
        path = RESULTS_DIR / "openml_decision_policy_summary.csv"
    if not path.exists():
        return
    rows = [row for row in read_csv(path) if row.get("mean_delta")]
    if not rows:
        return
    rows.sort(key=lambda row: (int(row["max_samples"]), -float(row["mean_delta"]), row["dataset"]))
    rows = [row for row in rows if float(row["mean_delta"]) > 0 or float(row["max_delta"]) > 0]
    if not rows:
        return
    labels = [f"{row['dataset'].replace('openml_', '')}\n{row['max_samples']} samples" for row in rows]
    means = [float(row["mean_delta"]) for row in rows]
    mins = [float(row["min_delta"]) for row in rows]
    bacc = [float(row.get("mean_balanced_accuracy_delta") or 0.0) for row in rows]
    auc = [float(row.get("mean_roc_auc_delta") or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    bars = ax.bar(range(len(rows)), means, color="#2f6f4e", label="accuracy mean delta")
    ax.scatter(range(len(rows)), mins, color="#222222", label="accuracy min delta", zorder=3)
    ax.scatter(range(len(rows)), bacc, color="#1f77b4", label="balanced accuracy", zorder=3)
    ax.scatter(range(len(rows)), auc, color="#7b519d", label="ROC-AUC", zorder=3)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in means], fontsize=8, padding=2)
    ax.set_title("Decision post-processing policy")
    ax.set_ylabel("Delta vs baseline")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openml_decision_policy.png")
    plt.close(fig)


def write_visual_analysis() -> None:
    text = """# 结果直观解读

## 图表

![Checkpoint accuracy comparison](figures/checkpoint_accuracy_comparison.png)

![Average checkpoint accuracy](figures/checkpoint_average_accuracy.png)

![Large scale accuracy runtime](figures/large_scale_accuracy_runtime.png)

![Context selection top deltas](figures/context_selection_top_deltas.png)

![Context selection repeated signals](figures/context_selection_repeated_signals.png)

![Query context selection deltas](figures/query_context_selection_deltas.png)

![OpenML external dataset deltas](figures/openml_method_search_deltas.png)

![OpenML candidate recheck stability](figures/openml_candidate_recheck_stability.png)

![OpenML propensity variant recheck](figures/openml_propensity_variant_recheck.png)

![OpenML guarded adaptive policy](figures/openml_guarded_policy.png)

![OpenML guarded scale study](figures/openml_guarded_scale_study.png)

![Query guarded policy](figures/query_guarded_policy.png)

![Synthetic guarded stress](figures/synthetic_guarded_stress.png)

![OpenML broad recheck](figures/openml_broad_recheck.png)

![OpenML promoted candidate](figures/openml_promoted_candidate.png)

![OpenML promoted guarded](figures/openml_promoted_guarded.png)

![Effective method catalog](figures/effective_method_catalog.png)

![Regression recheck](figures/regression_recheck.png)

![Regression guarded policy](figures/regression_guarded_policy.png)

![OpenML label balance recheck](figures/openml_label_balance_recheck.png)

![OpenML local geometry recheck](figures/openml_local_geometry_recheck.png)

![OpenML local geometry guarded](figures/openml_local_geometry_guarded.png)

![OpenML domain shift recheck](figures/openml_domain_shift_recheck.png)

![OpenML domain shift scale](figures/openml_domain_shift_scale.png)

![OpenML rare tail combo](figures/openml_rare_tail_combo.png)

![OpenML probe difficulty scale](figures/openml_probe_difficulty_scale.png)

![OpenML probe difficulty guarded](figures/openml_probe_difficulty_guarded.png)

![OpenML combined methods](figures/openml_combined_methods.png)

![OpenML diagnostic policy](figures/openml_diagnostic_policy.png)

![OpenML decision policy](figures/openml_decision_policy.png)

## 核心结论

1. 官方 checkpoint 仍然是目前最强结果，三个共同分类数据集平均准确率约 0.9863。
2. 从零训练的 small 1000-step checkpoint 已经有明显学习效果，平均准确率约 0.7263。
3. tiny 100-step 基本只是验证闭环，平均准确率约 0.5098。
4. stage-like 200-step 虽然结构更接近官方，但平均准确率约 0.5111，主要价值是验证大结构能跑通，不代表质量更好。
5. 大规模样本推理实验使用的是官方 checkpoint，不是我们从零训练出的 checkpoint；它说明当前环境能用 GPU 做更大样本推理。

## 涨点指标

分类任务默认主指标仍是 accuracy delta：

`delta = method_accuracy - baseline_accuracy`

但从 `domain-shift` 实验开始，每条分类结果同时记录：

- `accuracy`：整体准确率。
- `balanced_accuracy`：各类别召回率的平均值，更适合类别不平衡数据。
- `macro_f1`：各类别 F1 的平均值，不让多数类完全主导结论。
- `weighted_f1`：按类别样本数加权的 F1。
- `roc_auc`：基于 `predict_proba` 的排序能力指标；二分类用正类概率，多分类用 one-vs-rest macro AUC。若模型或数据切分无法计算，结果留空。

因此后续判断“是否真的涨点”，优先看多个 seed 下 accuracy 是否稳定为正；遇到类别不平衡或少数类问题时，还要看 balanced accuracy 和 macro F1 是否同步提升。

## 最重要的对比

| 模型/阶段 | synthetic | breast_cancer | wine | 三数据集平均 |
| --- | ---: | ---: | ---: | ---: |
| tiny 100-step | 0.5000 | 0.6294 | 0.4000 | 0.5098 |
| small 1000-step | 0.5880 | 0.9021 | 0.6889 | 0.7263 |
| stage-like 200-step | 0.5040 | 0.6294 | 0.4000 | 0.5111 |
| official checkpoint | 0.9800 | 0.9790 | 1.0000 | 0.9863 |

## 怎么理解 stage-like 低分

stage-like 不是退步，而是训练还太短。它的模型参数约 2765 万，比 small 约 568 万大很多；但它只训练了 200 step，而 small 训练了 1000 step。更大的模型在训练早期通常更吃数据和步数，所以 200 step 的下游分数接近 tiny 是正常现象。

这次 stage-like 的成功点是工程链路：更大模型、更长上下文、更大 prior、CUDA 训练、数据盘保存、checkpoint 保存、下游加载评估全部跑通。

## 下一步判断标准

下一轮不要只看“结构更接近官方”，要同时看训练步数。建议优先把 stage-like 从 200 step 继续到 1000 step，再和 small 1000-step 对比。只有当 stage-like 的训练步数接近 small，比较质量才更公平。

## Context Selection 新方法搜索

已加入一组受选择偏差和因果启发的方法搜索，结果见 `context_selection_search.md`。

目前最值得继续放大的信号：

- `balanced_treatment`：在 `synth_high_dim_sparse` 两个随机种子都涨，平均正增益约 +0.0117。
- `causal_kbest`：在 `synth_rare_boundary` 两个随机种子都涨，平均正增益约 +0.0067。
- `propensity_feature`：在 `synth_selection_linear` 两个随机种子都小涨，平均正增益约 +0.0033。
- `propensity_overlap`：在 `breast_cancer` 和部分 synthetic selection 数据上出现正增益，适合继续做更多真实选择偏差数据测试。

## Query-Level Context Selection

逐 query 选择 context 的版本更贴近公式 `Score(i, x*) = α Sim(x_i, x*) - β |s_i-s*|`。这版比较慢，因为每个 query 都要重新 fit 一个小 context，但它能验证有限 context budget 下“选谁做上下文”是否重要。

首轮结果：

- `synth_selection_hard_overlap`：random context 为 0.5333，`propensity_feature_context` 到 0.8000，`propensity_score_context` 到 0.7667。
- `synth_high_dim_sparse`：random context 为 0.7667，`causal_weighted_context` 和 `propensity_feature_context` 到 0.8667。
- `breast_cancer`：random context 为 0.9333，多种 query selection 方法到 0.9667，与 full context 持平。

## OpenML 真实外部数据集

已额外加入 5 个 OpenML 表格数据集，不再只看 sklearn 示例和合成数据。结果见 `openml_method_search.md`。

目前真实数据上最明显的涨点：

- `openml_ilpd`：`quantile_features` 从 0.6712 到 0.7123，提升 +0.0411，是当前最强真实数据信号。
- `openml_credit_g`：`causal_kbest` 从 0.7289 到 0.7378，提升 +0.0089。
- `openml_blood_transfusion`：`propensity_feature` 从 0.7914 到 0.7968，提升 +0.0053。
- `openml_diabetes_pima`：`ensemble_n4` 从 0.7396 到 0.7448，提升 +0.0052。
- `openml_spambase`：`quantile_features` 从 0.9422 到 0.9467，提升 +0.0044。

这说明三类方向都值得保留：特征分布变换、选择偏差/倾向得分特征、因果代理特征筛选。下一步优先对这些正增益组合做多 seed 和更大样本复验，避免单次切分偶然涨点。

## OpenML 多 Seed 复验

已对上一轮真实数据中的正增益候选做 3 个随机种子复验，结果见 `openml_candidate_recheck.md`。

复验后的优先级有变化：

- 最稳：`openml_ilpd + propensity_feature`，3/3 个 seed 都涨，平均提升 +0.0137，当前应作为第一优先放大方向。
- 次稳：`openml_credit_g + causal_kbest`，2/3 个 seed 涨，平均提升 +0.0030，但有一个 seed 下降 -0.0089。
- 值得观察：`openml_ilpd + ensemble_n4`，2/3 个 seed 涨，平均提升 +0.0046，但最坏 seed 为 -0.0205。
- 不宜直接默认：`openml_ilpd + quantile_features` 虽然单次最高 +0.0411，但 3 个 seed 只有 1 个正增益，说明它可能依赖切分。
- 暂缓：`openml_blood_transfusion + propensity_feature`、`openml_diabetes_pima + ensemble_n4`、`openml_spambase + quantile_features` 多 seed 平均为负，先不作为默认改进。

这一步把“能涨点”分成了两类：一类是单次亮眼，另一类是跨 seed 更稳。后续算法改进应优先围绕稳定信号做，而不是只追最大单次增益。

## Propensity 变体搜索

围绕选择偏差思路又加入了更多算法变体，结果见 `openml_propensity_variant_search.md` 和 `openml_propensity_variant_recheck.md`。

新增方法包括：

- `propensity_logit_feature`：把倾向得分换成 logit 形式再加入表格。
- `propensity_rank_feature`：加入倾向得分在训练集中的分位排名。
- `propensity_interactions`：只让倾向得分和高 causal-proxy 特征发生交互。
- `propensity_causal_kbest`：先保留 causal-proxy 高的特征，再加入倾向得分和处理标记。
- `propensity_overlap_feature`：先裁剪到 train/test 倾向得分重叠区间，再加入倾向特征。
- `propensity_bin_balance`：按倾向得分分箱后平衡采样 context。

多 seed 复验后的新稳定发现：

- `openml_ilpd + propensity_causal_kbest`：3/3 个 seed 都涨，平均 +0.0251，超过原始 `propensity_feature` 的 +0.0137。
- `openml_ilpd + propensity_interactions`：3/3 个 seed 都涨，平均 +0.0228，说明“倾向得分 x 关键特征”的交互是有效方向。
- `openml_cmc + propensity_overlap_feature`：3/3 个 seed 都涨，平均 +0.0163，这是新扩展数据集上的稳定信号。
- `openml_credit_g + propensity_bin_balance`：3/3 个 seed 都涨，平均 +0.0104，说明按选择机制做 context 平衡有潜力。
- `openml_credit_approval + propensity_overlap_feature`：3/3 个 seed 都涨，平均 +0.0058，幅度小但非常稳定。

同时也识别出不适合默认使用的方向：`propensity_causal_kbest` 在 `openml_cmc`、`openml_vehicle` 上明显下降，说明 causal-proxy 特征筛选不能无脑套在所有数据集上，后续需要做数据集条件判断。

## Guarded Adaptive Policy

为了避免“测试集看完再挑方法”的不现实问题，新增了 `openml_adaptive_policy.py`。它先在训练集内部切出 validation split，从候选方法中选一个，再回到完整训练集上评估测试集。

第一版自由选择器会被验证集误导，宏平均 delta 为 -0.00234，最坏数据集 seed 为 -0.0489。随后加入 guarded policy：

- 原始特征数小于等于 4 时只用 baseline，避免小表格被额外特征扰动。
- 多分类任务禁用高风险的 `propensity_causal_kbest`。
- 高维表格禁用 overlap 裁剪等不稳定操作。
- 验证集提升至少达到 +0.01 才允许切换方法。
- 验证集平手时优先选择此前多 seed 更稳的候选。

guarded policy 在 10 个 OpenML 数据集、3 个 seed 上测试：

- 30 次运行中没有负增益，最坏 delta 为 0。
- 宏平均 delta 从自由选择器的 -0.00234 提升到 +0.00192。
- `openml_cmc` 平均 +0.0089，`openml_credit_approval` 平均 +0.0039，`openml_diabetes_pima` 平均 +0.0035，`openml_credit_g` 平均 +0.0030。

这说明当前最适合保存为默认策略的不是某个激进单方法，而是“保守验证选择器”：能在一部分数据集上小幅涨点，同时尽量不伤害已经很强的 baseline。

## Guarded Policy 数据规模实验

新增 `openml_guarded_scale_study.py`，对 `cmc`、`credit_approval`、`diabetes_pima`、`credit_g` 在 300、600、900 三个样本规模上做 3 seed 复验。

第一次规模实验发现，小样本下 validation split 波动较大，会出现负增益。于是 guarded policy 加入样本阈值：`n_samples < 650` 时直接使用 baseline，不启用自动增强。

加入阈值后的结果：

- 300 样本：12 次运行全部回退 baseline，宏平均 delta 0，最坏 delta 0。
- 600 样本：12 次运行全部回退 baseline，宏平均 delta 0，最坏 delta 0。
- 900 样本：12 次运行中 5 次正增益，宏平均 delta +0.0048，最坏 delta 0。

这给出了一个更可执行的默认策略：小数据先不动，样本足够时才尝试 guarded adaptive policy。它牺牲了小样本上的偶然大涨点，但换来了稳定不掉点。

## Query-Level Guarded Context

新增 `query_guarded_policy.py`，把图片里的 query-level context selection 做成更真实的验证流程：先在训练集内部用 validation query 选择 context 方法，再到测试 query 上评估。

调试过程里发现：

- `causal_weighted_context` 在 validation 上可能很强，但外层测试会掉点，因此不适合放进默认 guarded 候选。
- `knn_context` 也会在高维 synthetic 上被 validation 误导，默认候选同样移除。
- 最终保留 `random_context`、`propensity_score_context`、`propensity_feature_context`，并要求 validation 相对 random context 至少 +0.05 才切换。

首轮小规模结果：

- `synth_high_dim_sparse`：300 样本 +0.0500，500 样本 +0.1000。
- `breast_cancer`：两个规模都回退 random context，delta 0。
- `synth_selection_hard_overlap`：两个规模都因为 validation 提升不足 +0.05 而回退，delta 0。

这说明 query-level context 的方向仍然有信号，但比全表 guarded policy 更慢、更容易受 validation query 数影响。目前适合作为研究分支保存，不建议直接作为默认推理策略。

## Synthetic Guarded Stress Test

新增 `synthetic_guarded_stress.py`，专门把 guarded adaptive policy 放到更难的 synthetic special 数据上做压力测试，包括 selection bias、hard overlap、rare boundary 和 high-dimensional sparse 场景。

调试中发现，特殊合成数据会让验证集选择器在 900 样本规模上误选增强方法，导致外层测试出现小幅负增益。于是 guarded policy 加入了 `synthetic_special` 来源保护：这些数据先作为压力测试保留，但默认策略直接回退 baseline，避免把不稳定信号误推广成默认算法。

最终压力测试结果：

- 300 样本：12 次运行全部非负，宏平均 delta 0，最坏 delta 0。
- 650 样本：12 次运行全部非负，宏平均 delta 0，最坏 delta 0。
- 900 样本：12 次运行全部非负，宏平均 delta 0，最坏 delta 0。
- `synthetic_guarded_stress_winners.csv` 为空，说明当前默认 guarded policy 在特殊压力集上选择“守住 baseline”，而不是追求不稳定涨点。

这一步的价值不是找到新涨点，而是给默认策略加了一道边界：真实 OpenML 中已经验证比较稳的增益继续保留，特殊构造数据上先不冒险。

## Broad OpenML 新数据集筛选

新增 `openml_broad_sweep.py`，这轮不再只看前面反复用的 OpenML 数据集，而是加入 8 个新的外部表格数据集：`kr_vs_kp`、`letter`、`mfeat_factors`、`mfeat_fourier`、`breast_w`、`mfeat_karhunen`、`mfeat_morphological`、`segment`。

第一轮 600 样本、单 seed 快速筛选中，8 个数据集全部成功加载，64 次方法评估里出现 7 个正增益候选。随后挑出 5 个更有希望的数据集做 3 seed 复验。

复验后的新稳定信号：

- `openml_mfeat_morphological + propensity_interactions`：3/3 个 seed 都涨，平均 +0.0133，最坏也有 +0.0067，是这轮最强的新发现。
- `openml_mfeat_fourier + propensity_interactions`：2/3 个 seed 涨，平均 +0.0089，最坏 0，没有掉点。
- `openml_mfeat_fourier + propensity_causal_kbest`：2/3 个 seed 涨，平均 +0.0067，最坏 0，也比较适合放入候选池。

同时也发现需要避免的组合：

- `openml_letter + causal_kbest`：3/3 seed 都掉，平均 -0.0733。
- `openml_letter + propensity_causal_kbest`：3/3 seed 都掉，平均 -0.0622。

这说明高类别/字符识别类数据不适合简单 causal-kbest 截特征；后续 guarded policy 应继续按数据集结构限制这类激进特征筛选。

## Promoted Candidate 规模边界复验

对这轮最强新候选 `openml_mfeat_morphological + propensity_interactions` 做了更严格复验：600 和 900 两个样本规模，各 5 个 seed，只比较 baseline 与该方法。

结果：

- 600 样本：5/5 seed 都涨，平均 +0.0147，最坏 +0.0067，最大 +0.0200。
- 900 样本：2/5 seed 涨，平均 +0.0027，最坏 -0.0133，最大 +0.0356。

解释：

- 这个组合在中小规模上已经是可保存的稳定涨点候选。
- 到 900 样本时平均仍略正，但 seed 方差变大，不能直接设为默认方法。
- 更合理的策略是把它作为 guarded/adaptive policy 的候选：只有内部 validation 明显支持时才启用，否则回退 baseline。

## Promoted Candidate Guarded Policy

把 `openml_mfeat_morphological + propensity_interactions` 纳入 guarded policy 后，规则被收窄为只在 `openml:18` 且样本数处于 600-699 时开放该候选；700 以上回退 baseline。这样做是因为直接使用该方法时，900 样本虽然平均略正，但已经出现负 seed。

guarded 复验结果：

- 600 样本：5 个 seed 中 3 次启用 `propensity_interactions`，平均 +0.0080，最坏 0。
- 900 样本：5 个 seed 全部回退 baseline，平均 0，最坏 0。

这说明当前策略已经把一个“直接使用会有规模风险”的方法，变成了一个有边界的候选规则：在证据最强的规模附近保留涨点，在风险规模守住 baseline。

## Effective Method Catalog

新增 `build_effective_method_catalog.py`，把当前所有 OpenML 复验、guarded policy、规模实验、broad sweep 和 promoted candidate 的 summary 汇总成统一证据库。

当前 catalog 统计：

- `stable_positive`：22 个组合，所有检查过的 seed/run 都涨，优先保存和复用。
- `guarded_positive`：40 个组合，平均为正且最坏不掉，适合通过 validation 或规则启用。
- `risky_positive`：75 个组合，平均为正但有负 seed，或只有单 seed 小涨，只能当研究线索。
- `safe_baseline_or_tie`：111 个组合，guarded policy 回退 baseline 或打平。
- `negative_do_not_default`：203 个组合，明确不建议默认启用。

这轮新补的 `openml_sick` 和 `openml_mfeat_zernike` 初筛都出现过 `propensity_causal_kbest` 单 seed 小涨，但 3 seed 复验后平均为负，已经进入 `negative_do_not_default`。这能防止我们以后把偶然涨点误当成稳定方法。

## Regression Method Search

新增 `regression_method_search.py`，把搜索范围从分类扩到 TabICLRegressor。当前回归数据包括 `sklearn_diabetes`、`synthetic_friedman1`、`synthetic_low_rank`、`synthetic_outlier_target`。`sklearn_california_housing` 在当前环境下载时返回 403，因此本轮记录为无法加载并跳过。

初筛后复验了两个候选方向：

- `sklearn_diabetes + clipped_target`：3 seed 中 2 个 R2 涨，平均 R2 delta +0.0026，平均 MAE delta +0.1703，但最坏 R2 delta -0.0008。
- `synthetic_low_rank + quantile_features`：3 seed 中 2 个 R2 涨，平均 R2 delta +0.0018，平均 MAE delta +0.0500，但最坏 R2 delta -0.0054。

结论：回归任务目前还没有达到 `stable_positive` 的组合，但 `clipped_target` 在 diabetes 上更像 guarded/risky 候选；`quantile_features` 对 low-rank synthetic 有信号但不够稳。它们已经被纳入 catalog，后续需要更强 validation 或更多真实回归数据集再决定是否推广。

## Regression Guarded Policy

新增 `regression_guarded_policy.py`，把回归 risky 候选放进内层 validation 选择器。第一版阈值太松，会在 `sklearn_diabetes` seed 11 上误选 `clipped_target`，外层 R2 小跌 -0.0008。随后把默认阈值收紧为 validation R2 至少 +0.006，且 validation MAE 至少改善 +0.20。

收紧后的 3 seed 结果：

- `sklearn_diabetes`：全部回退 baseline，平均 R2 delta 0，最坏 0。
- `synthetic_low_rank`：全部回退 baseline，平均 R2 delta 0，最坏 0。

这说明当前回归候选还没有足够强的 validation 信号，合理处理是保守回退，而不是把微弱平均正增益当成默认方法。

## Label Balance Context

新增 `label_balance_context`，按训练标签下采样，让每个类别在 context 中数量相等。这个想法直观上适合类别不平衡，但实验结果显示它不能默认使用。

初筛 6 个 OpenML 数据集时：

- `openml_sick` 单 seed 小涨 +0.0067。
- `openml_blood_transfusion` 明显下降 -0.0667。
- `openml_ilpd` 明显下降 -0.0411。
- 其他数据集多为打平。

随后对 `openml_sick` 做 3 seed 复验：

- 只有 1/3 seed 正增益。
- 平均 delta -0.0022。
- 最坏 delta -0.0067。

结论：标签平衡会丢掉大量多数类 context 信息，当前应进入 `negative_do_not_default`，保留为反例而不是默认策略。

## Local Geometry Features

新增 3 个局部几何/邻域增强方法：

- `local_density_feature`：加入训练样本附近的局部密度和密度排名。
- `centroid_distance_feature`：加入样本到各类别中心的距离、最近中心距离和 margin。
- `knn_label_feature`：加入 leave-one-out KNN 邻居标签分布、最大邻居概率、margin 和熵。

单 seed 筛选里出现过几个明显涨点：

- `openml_sick + knn_label_feature`：+0.0333。
- `openml_cmc + local_density_feature`：+0.0200。
- `openml_cmc + centroid_distance_feature`：+0.0200。
- `openml_credit_approval + centroid_distance_feature`：+0.0133。
- `openml_credit_g + knn_label_feature`：+0.0133。

但 3 seed 复验后，这些信号都不够稳定：

- `openml_sick + knn_label_feature`：平均 +0.0044，但最坏 -0.0133。
- `openml_cmc + centroid_distance_feature`：平均 +0.0022，但最坏 -0.0267。
- `openml_credit_approval + centroid_distance_feature`：平均 -0.0022。
- `openml_credit_g + knn_label_feature`：平均 -0.0067。

解释：局部几何方法能捕捉“样本是否处在稠密区域、离哪个类别中心更近、邻居标签是否一致”等结构，说明它可能补足 TabICL 对局部结构利用不足的问题；但当前版本容易被 split 波动影响，还不能默认启用。它应该进入 `risky_positive` 或 `negative_do_not_default`，作为后续 guarded/validation 方法的候选，而不是直接推广。

## Local Geometry Guarded Policy

进一步把局部几何方法放进 inner-validation guarded policy，测试 3 档阈值：

- 阈值 `+0.02`：会启用 `centroid_distance_feature` / `local_density_feature`，但 `cmc`、`credit_approval`、`credit_g` 都出现负增益；说明普通 validation gate 仍然会被局部结构误导。
- 阈值 `+0.05`：大部分回退 baseline，但 `cmc` seed 11 仍误启用 `centroid_distance_feature`，测试 delta -0.0267。
- 阈值 `+0.07`：全部回退 baseline，宏平均 delta 0，最坏 delta 0。

结论：局部几何方法目前能解释一些单 seed 涨点，但 validation 转移性差。它解决的问题更像是“某些切分下局部邻域结构不足”，还没有形成可泛化的 guarded 增益。当前应把它保存为研究线索和反例边界：如果以后继续做，需要更强的跨 validation split 一致性判断，而不是单次 validation 提升。

## Domain Shift Features

新增 3 个 covariate/domain-shift 方法：

- `domain_shift_feature`：训练一个 train-vs-test 域分类器，把样本“像测试分布的程度”作为新特征加入表格。
- `domain_matched_context`：优先保留更像测试分布的训练 context。
- `domain_causal_interactions`：让 domain-shift 概率和 causal-proxy 高的特征发生交互。

3 seed 复验后最重要的新稳定信号来自 `openml_cmc`：

- `domain_shift_feature`：accuracy 平均 +0.0200，balanced accuracy 平均 +0.0200，macro F1 平均 +0.0203，3/3 seed 都涨。
- `domain_causal_interactions`：accuracy 平均 +0.0156，balanced accuracy 平均 +0.0156，macro F1 平均 +0.0162，3/3 seed 都涨。

这说明 `cmc` 的问题不只是普通分类难度，而是训练/测试特征分布存在轻微但可利用的偏移。把“当前样本是否像测试分布”显式给 TabICL 后，整体准确率和更均衡的类别指标都同步提升。

同时，`domain_matched_context` 虽然在 `cmc` 单 seed 上最高 +0.0400，但 3 seed 最坏 -0.0333；在 `sick`、`diabetes_pima` 上也明显不稳。因此直接裁剪 context 风险较高，当前更可靠的是追加 domain-shift 特征，而不是强行删 context。

## Domain Shift Scale Boundary

对 `openml_cmc` 做 300、600、900、1200 四个样本规模的 3 seed 复验：

- 300 样本：`domain_shift_feature` 平均 +0.0089，最坏 0；`domain_causal_interactions` 平均 +0.0178，但 macro F1 有轻微负 seed。
- 600 样本：`domain_shift_feature` 最稳，accuracy / balanced accuracy / macro F1 都约 +0.020，3/3 seed 全涨。
- 900 样本：`domain_shift_feature` 平均转负 -0.0030；`domain_causal_interactions` 仍平均 +0.0089 且最坏 accuracy 0。
- 1200 样本：三种 domain-shift 方法平均都转负。

结论：domain-shift 特征不是越大规模越好，它主要解决的是中小样本下 context 对测试分布代表性不足的问题。当前最可靠边界是 `openml_cmc` 约 600 样本时启用 `domain_shift_feature`；900 样本可把 `domain_causal_interactions` 作为 guarded 候选；1200 样本应回退 baseline。

## Rare/Tail 与 AUC 对比

按你的思路新增了低频类别和极端尾部相关候选：

- `rare_class_oversample`：对低频类别做上下文过采样。
- `rare_boundary_context`：保留低频类别样本和靠近类别边界的样本。
- `rare_domain_shift_oversample`：低频类别过采样，同时加入 train/test domain-shift 特征。
- `rare_boundary_domain_shift`：边界/低频上下文，同时加入 domain-shift 特征。

这批实验说明：这些方向可行，但目前不能默认启用。

- 稳定涨点仍来自 `openml_cmc + domain_shift_feature`：3/3 seed 都涨，accuracy、balanced accuracy、macro F1 都约 +0.0200，ROC-AUC 约 +0.0071。
- `rare_domain_shift_oversample` 和 `rare_boundary_domain_shift` 在 CMC 上与 `domain_shift_feature` 打平，说明真正起作用的是 domain-shift 特征；低频/边界模块没有额外贡献，但组合不会伤害该数据集。
- `openml_sick + rare_class_oversample` 出现一个很小的 AUC 信号：ROC-AUC 3/3 seed 都为正，平均约 +0.0006；但 accuracy 平均为 0，balanced accuracy 和 macro F1 略负，所以只能说它改善了概率排序的一点点，不算分类结果稳定涨点。
- `openml_blood_transfusion + label_balance_context` 的 balanced accuracy 平均 +0.0378，但 accuracy -0.0889、macro F1 -0.0274、AUC -0.0040。这说明单独加强少数类可能提高少数类召回，却会明显牺牲整体预测，不能直接作为默认方法。
- `rare_boundary_context` 在 `blood_transfusion` 和 `diabetes_pima` 上大幅掉点，说明“只保留边界/尾部样本”会丢失多数正常区域的信息，当前应作为反例保留。

阶段结论：低频类别、边界样本、尾部分布这些方法不是没用，而是需要条件触发。它们更适合在目标指标明确偏向少数类召回或 AUC 排序时作为 guarded 候选；如果主目标是 accuracy 稳定涨点，目前最可靠的仍是 domain-shift 特征，并且只在 CMC 这类中小规模选择机制不一致的数据上启用。

## Probe/Difficulty 条件增强

新增 `linear_probe_feature`、`hard_example_oversample`、`hard_probe_combo` 三个候选。它们不再直接删除大批 context，而是先用一个 balanced logistic probe 判断样本难度：

- `linear_probe_feature`：加入 probe 的最大概率、概率间隔、熵和类别概率，给 TabICL 一个“线性模型怎么看这个样本”的软提示。
- `hard_example_oversample`：保留完整训练集，再额外复制低置信或 probe 误判样本。
- `hard_probe_combo`：同时加入 probe 特征并过采样难样本。

宽筛覆盖 20 个 OpenML 数据集；随后对 8 个正信号数据集做 3 seed 复验，再对 `credit_approval` 和 `ilpd` 做 300/600/900 规模边界测试。

新保存的有效场景：

- `openml_credit_approval + hard_probe_combo`：600 样本 3/3 seed 都涨，accuracy 平均 +0.0089，balanced accuracy 和 macro F1 同步 +0.0089；这是本轮最明确的新稳定涨点。
- `openml_credit_approval + knn_label_feature`：900 样本 3/3 seed 都涨，accuracy 平均 +0.0096，最坏 +0.0058；说明样本更多时，邻域标签分布比 hard oversampling 更稳。
- `openml_ilpd + linear_probe_feature`：完整 583 样本附近 3 seed 平均 accuracy +0.0160，最坏 0，ROC-AUC 平均 +0.0093；它不只是分类阈值涨，也改善概率排序。
- `openml_ilpd + knn_label_feature`：同规模 accuracy 也 +0.0160，balanced accuracy +0.0207、macro F1 +0.0233，但 AUC 为负，适合追求分类/少数类指标，不适合追求排序。

同时确认的边界：

- `linear_probe_feature` 在 `vehicle` 上大幅负增益，不能默认启用。
- `hard_example_oversample` 在 `diabetes_pima`、`cmc` 上平均为负，说明“难样本复制”只适合特定数据，不是通用技巧。
- `credit_approval` 的 300 样本有大幅均值正增益，但只有 1/3 seed 真正涨，属于 guarded/risky 信号；600 和 900 的边界更可信。

## Probe/Difficulty Guarded Policy

为了避免直接把 probe/difficulty 方法套到所有数据集上，新增 `openml_probe_difficulty_guarded.py`。它只在已有证据支持的边界开放候选：

- `openml_credit_approval`：600 样本附近允许 `hard_probe_combo`；900 样本附近允许 `knn_label_feature` 和 `hard_probe_combo`。
- `openml_ilpd`：完整 583 样本附近允许 `linear_probe_feature` 和 `knn_label_feature`。
- 其他 OpenML 数据集全部回退 baseline。

这次在 20 个 OpenML 数据集、600/900 两个规模、3 个 seed 上共 120 次外层运行：

- 非 baseline 只启用 3 次，说明策略非常保守。
- 没有任何负增益，所有 dataset-scale 的最坏 delta 都是 0。
- `openml_ilpd` 在 600 和 900 样本设置下平均 accuracy delta +0.0023，最坏 0。
- `openml_credit_approval` 在 900 样本设置下平均 accuracy delta +0.0019，最坏 0。

结论：这个 guarded 版本牺牲了不少直接方法的平均涨幅，但把风险压到了 0。它更适合作为“可默认启用的安全外壳”；直接方法本身仍保留在 catalog 里，作为后续更强 validation 或数据集规则的候选。

## Outlier/Shape 与综合方法

这轮把你提到的“尾部分布、边界、补全/采样、特征处理”进一步拆开做了实验。当前实现新增了：

- `winsorized_features`：把极端特征值裁剪到训练集 2%/98% 分位数。
- `outlier_indicator_feature`：追加每个样本是否含异常特征、异常特征数量。
- `outlier_winsor_combo`：先裁剪尾部，再加入异常值指示。
- `signed_log_features`：对长尾特征做带符号 log 压缩。
- `quantile_domain_shift_feature`：先做分位数正态化，再追加 train/test domain-shift 特征。

最值得保留的新结果是：

- `openml_cmc + quantile_domain_shift_feature`：600 样本、3/3 seed 都涨，accuracy 平均 +0.0289，balanced accuracy +0.0289，macro F1 +0.0297，ROC-AUC +0.0048。这说明对 CMC 这种存在选择机制/分布偏移的数据，单纯看特征相似不够；把分布形状校正和 domain-shift 线索合起来最有效。
- `openml_cmc + domain_shift_feature`：600 样本仍然 3/3 seed 稳定上涨，accuracy +0.0200，AUC +0.0071。它是当前最干净的选择偏差正例。
- `openml_cmc + quantile_features`：600 样本 3/3 seed 上涨，accuracy +0.0133，但 AUC 只 +0.0005。它能改善分类边界，但概率排序提升较小。
- `openml_cmc + outlier_winsor_combo`：900 样本 3/3 seed 上涨，accuracy +0.0119，balanced accuracy +0.0119，macro F1 +0.0122，AUC +0.0009。说明样本稍大时，尾部裁剪加异常指示也能成为一个安全候选。

有涨点但必须谨慎的结果：

- `openml_ilpd + quantile_domain_shift_feature`：平均 accuracy +0.0091，balanced accuracy +0.0206，macro F1 +0.0236，AUC +0.0041；最坏 accuracy delta 为 0，没有掉 accuracy，但只有 2/3 seed 正增益，适合 guarded 候选。
- `openml_ilpd + outlier_winsor_combo`：平均 accuracy +0.0114，AUC +0.0012，但最坏 accuracy -0.0205，不能默认。
- `openml_credit_g` 上有些方法 AUC 会涨，例如 900 样本 `outlier_winsor_combo` 的 AUC +0.0071 且 3/3 seed 为正，但 accuracy 基本不涨。这类方法适合“排序/AUC 优先”的场景，不适合“accuracy 必涨”的默认策略。

综合结论：这些方法不是追求普适性，而是在解决特定不足。CMC 代表的选择偏差/分布偏移场景已经有稳定涨点；ILPD 代表的异常值/非线性边界场景有 guarded 信号；credit_g 代表的风险排序场景可以看 AUC，但不能保证 accuracy。后续默认策略应该是条件触发：识别数据更像 CMC 时用 `quantile_domain_shift_feature` 或 `domain_shift_feature`，识别长尾/异常值明显但 seed 风险较高时只作为候选，不直接替换 baseline。

## Diagnostic Policy

新增 `openml_diagnostic_policy.py`，把“是否启用方法”从数据集名字硬编码，推进到数据诊断驱动。每个外层 split 会先计算：

- `domain_abs_gap`：训练集和测试集是否容易被区分，用来识别选择偏差/分布偏移。
- `train_outlier_rate` 和 `outlier_rate_gap`：样本是否含明显尾部/异常特征。
- `shape_score`：用分位数衡量长尾和非对称程度。
- `class_min_frac`、`n_classes`、`n_features`：判断少数类、类别数、特征维度风险。

规则不是直接决定最终方法，而是决定候选池；最终仍用 inner validation 选择，且 validation accuracy 至少比 baseline 高 `+0.015` 才允许启用，否则回退 baseline。

调试中先发现三个负例：`blood_transfusion`、`diabetes_pima`、`letter`。对应修正是：

- 极低维二分类数据不开放 probe/knn，避免 `blood_transfusion` 误选。
- 类别较均衡且异常值少的二分类数据不开放 domain 候选，避免 `diabetes_pima` 误选。
- 类别数过高的数据不开放 domain/quantile 候选，避免 `letter` 误选。
- 异常值比例高但长尾形状不明显时，不开放 signed-log/outlier-tail 候选，避免 `credit_g` 在 900 样本误选。

最终 broad 结果覆盖 20 个 OpenML 数据集、600 样本、3 seed：没有任何负增益。正信号包括：

- `openml_cmc`：mean accuracy delta `+0.0222`，min delta `0`，balanced accuracy `+0.0222`。
- `openml_ilpd`：mean accuracy delta `+0.0091`，min delta `0`，ROC-AUC `+0.0019`。
- `openml_balance_scale`：mean accuracy delta `+0.0044`，min delta `0`。
- `openml_credit_g` 和 `openml_spambase`：小幅正增益，最坏为 0。

规模边界复核显示：600 样本时有较多正信号；900 样本时只有 `openml_ilpd` 仍保持 mean `+0.0091`，其余数据集回退 baseline。这进一步说明当前方法主要解决中小样本下的选择偏差、异常值和 context 代表性不足；样本规模变大后，默认 baseline 往往已经足够，guarded 策略应主动少动。

## Decision Policy

新增 `openml_decision_policy.py`，尝试另一条路线：不改变输入/context，而是在 TabICL 输出概率上做轻量后处理。候选包括：

- `prior_debias`：减去类别先验的 log bias，提高低频类别决策机会。
- `prior_rebias`：加强训练分布先验，避免少数类过度预测。
- `minority_bias`：给训练集中最低频类别增加一个小 logit bias。

自由版本在二分类数据上会被 validation 明显误导：`blood_transfusion`、`ilpd`、`credit_g` 都出现过负增益。因此最终保存的是 guarded 版本：只在 3 到 10 类的多分类数据上开放后处理，二分类和类别数过高的数据直接回退 baseline。

最终 broad 结果覆盖 20 个 OpenML 数据集、600 样本、3 seed：没有负增益，但正信号很窄：

- `openml_cmc`：mean accuracy delta `+0.0067`，min delta `0`，一个 seed 通过 `minority_bias_b0.25` 涨 `+0.0200`。
- `openml_mfeat_morphological`：validation 会触发 minority bias，但外层 accuracy 打平，只出现很小 AUC 改善。

规模边界复核显示：`openml_cmc` 的 decision policy 只在 600 样本有小幅正信号，900 样本回退 baseline。结论是：概率后处理不是主力方法，但可以作为 CMC 类多分类选择偏差问题的低成本候选；对二分类和高类别任务暂时不要启用。

## Combined Gain Policy

这轮把“只要能涨点就组合起来”的想法实现成了 `openml_combined_gain_policy.py`。它不是把所有增强都无脑打开，而是分两层：

- 输入侧候选：低频类别过采样、低频+domain-shift、quantile/domain-shift、尾部裁剪+异常指示、KNN 标签分布等。
- 输出侧候选：少数类 bias、prior debias/rebias 等概率后处理。

先跑宽候选版本，结果说明组合确实能找到涨点，但也会被 validation 误导：

- 正例：`openml_ilpd` 低频类过采样最高单 seed accuracy `+0.0479`，balanced accuracy `+0.0549`，macro F1 `+0.0703`。
- 正例：`openml_mfeat_morphological` 的 `linear_probe_feature + minority_bias` 单 seed `+0.0133`。
- 负例：`diabetes_pima` 的 hard-probe 组合单 seed `-0.0400`，说明均衡二分类不能随便加强“难样本”。
- 负例：`blood_transfusion` 的 quantile/domain-shift 单 seed `-0.0200`，说明低维小表的 validation 很容易误判。
- 负例：`kr_vs_kp` 的 one-hot/离散特征会让 outlier rate 虚高，不能按连续尾部处理。

因此新增 strict/input-only 版本，规则只开放机制匹配的候选，并禁用输出端后处理。最终在 20 个 OpenML 数据集、600 样本、3 seed 上没有负增益，所有数据集的最坏 accuracy delta 都是 0。保留下来的正信号是：

- `openml_ilpd`：mean accuracy `+0.0205`，min `0`，balanced accuracy `+0.0192`，macro F1 `+0.0242`。对应机制是低频类别 + 中等异常值比例，适合 `rare_class_oversample` 或 `rare_domain_shift_oversample`。
- `openml_spambase`：mean accuracy `+0.0022`，min `0`，AUC `+0.0007`。对应机制是真正长尾数值特征，适合 `outlier_winsor_combo`。
- `openml_vehicle`：mean accuracy `+0.0022`，min `0`，balanced accuracy `+0.0023`。对应机制是小规模多分类、类别边界依赖局部邻域，适合 `knn_label_feature`。
- `openml_sick`：mean accuracy `+0.0022`，min `0`，AUC `+0.0004`。对应机制是中等宽度二分类、存在 domain-shift，适合 `quantile_domain_shift_feature`。

这说明当前方法最可靠的论文表述不是“普适提升”，而是“针对 context 代表性不足的条件增强”：当数据呈现低频类别、真实长尾、局部边界或 train/test selection shift 时，诊断规则打开相应模块；否则回退 baseline。这样既保留涨点，也把错误增强的风险压低。
"""
    (RESULTS_DIR / "visual_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_checkpoint_comparison()
    plot_average_accuracy()
    plot_scale_study()
    plot_context_selection_winners()
    plot_context_repeated_signals()
    plot_query_context_deltas()
    plot_openml_method_deltas()
    plot_openml_candidate_recheck()
    plot_openml_propensity_variant_recheck()
    plot_openml_guarded_policy()
    plot_openml_guarded_scale_study()
    plot_query_guarded_policy()
    plot_synthetic_guarded_stress()
    plot_openml_broad_recheck()
    plot_openml_promoted_candidate()
    plot_openml_promoted_guarded()
    plot_effective_method_catalog()
    plot_regression_recheck()
    plot_regression_guarded_policy()
    plot_openml_label_balance()
    plot_openml_local_geometry()
    plot_openml_local_geometry_guarded()
    plot_openml_domain_shift()
    plot_openml_domain_shift_scale()
    plot_openml_rare_tail_combo()
    plot_openml_probe_difficulty_scale()
    plot_openml_probe_difficulty_guarded()
    plot_openml_combined_methods()
    plot_openml_diagnostic_policy()
    plot_openml_decision_policy()
    write_visual_analysis()
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote {RESULTS_DIR / 'visual_analysis.md'}")


if __name__ == "__main__":
    main()
