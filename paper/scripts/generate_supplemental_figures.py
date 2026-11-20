#!/usr/bin/env python3
"""Generate publication figures for the Section 5.4 supplemental analyses.

Every plotted value is loaded from the experiment CSV files listed in the
audit record.  The script emits PDF/SVG with live text and a PNG preview.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper" / "figures"
COUNT_SOURCE = (
    ROOT
    / "PaperResults"
    / "tabace_comprehensive_ablation"
    / "average_context_count_summary.csv"
)
AGG_SOURCE = (
    ROOT
    / "PaperResults"
    / "tabace_k5_aggregation_preview"
    / "summary.csv"
)
K5_REPRODUCTION = (
    ROOT
    / "fair_budget_experiments"
    / "k5_inference_cost_reproduction"
)
COST_SOURCE = (
    K5_REPRODUCTION
    / "exp44_k5_component_efficiency"
    / "results"
    / "component_timing_summary.csv"
)
CACHE_SOURCE = (
    K5_REPRODUCTION
    / "exp45_k5_propensity_cache"
    / "results"
    / "timing_summary.csv"
)

COUNT_STEM = FIGURES / "tabace_context_count_ablation"
AGG_STEM = FIGURES / "tabace_aggregation_ablation"
COST_STEM = FIGURES / "tabace_inference_cost_ablation"
AUDIT_PATH = FIGURES / "tabace_supplemental_figures_audit.json"

FONT_ROOTS = (
    Path("/usr/share/texmf/fonts/opentype/public/lm"),
    Path("/usr/share/fonts/opentype/lmodern"),
)


def register_font(filename: str, fallback: str) -> str:
    for root in FONT_ROOTS:
        path = root / filename
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=path).get_name()
    return fallback


PLOT_FONT = register_font("lmsans10-regular.otf", "DejaVu Sans")
PANEL_FONT = register_font("lmroman10-regular.otf", "DejaVu Serif")
register_font("lmroman10-italic.otf", "DejaVu Serif")

BLUE = "#4477AA"
GREEN = "#228833"
GRAY = "#A7A7A7"
DARK_GRAY = "#4D4D4D"
LIGHT_GRAY = "#D5D5D5"
ORANGE = "#B85C18"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": PLOT_FONT,
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save_figure(figure: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(
            stem.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            pad_inches=0.025,
        )
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


def polish_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D8D8D8", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)


def panel_caption(axis: plt.Axes, marker: str, text: str, y: float = -0.28) -> None:
    axis.text(
        0.5,
        y,
        rf"$\mathit{{{marker}}}$ {text}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontfamily=PANEL_FONT,
        fontsize=9.0,
    )


def context_count_audit() -> dict[str, object]:
    rows = read_rows(COUNT_SOURCE)
    row_map = {
        (row["benchmark"], row["task"], row["metric"]): row for row in rows
    }
    panels = (
        ("classification", "accuracy"),
        ("classification", "auc"),
        ("regression", "r2"),
        ("regression", "nmae"),
    )
    keys = (
        "base",
        "avg_k1",
        "avg_k2",
        "avg_k3",
        "avg_k4",
        "avg_k5",
        "avg_k6",
        "avg_k7",
        "pro",
    )
    audit_values: dict[str, dict[str, list[float | None]]] = {}
    for task, metric in panels:
        panel_key = f"{task}_{metric}"
        audit_values[panel_key] = {}
        for benchmark in ("five", "seven"):
            row = row_map[(benchmark, task, metric)]
            values = [float(row[key]) if row.get(key, "") else None for key in keys]
            audit_values[panel_key][benchmark] = values
    return {"source": relative(COUNT_SOURCE), "columns": list(keys), "values": audit_values}


def aggregation_figure() -> dict[str, object]:
    rows = [
        row
        for row in read_rows(AGG_SOURCE)
        if row["benchmark"] == "five+seven_equal_macro"
    ]
    row_map = {(row["task"], row["operator"]): row for row in rows}
    variants = (
        "mean",
        "median",
        "trimmed_mean",
        "oof_weighted",
    )
    labels = ("Mean", "Median", "Trimmed", "OOF\nweighted")
    colors = (GRAY, GRAY, GRAY, ORANGE)
    panels = (
        ("classification", "accuracy", "Accuracy", "(a)"),
        ("regression", "nrmse", "nRMSE", "(b)"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.08, 1.95))
    audit_values: dict[str, list[float]] = {}
    for axis, (task, metric, ylabel, marker) in zip(axes, panels):
        values = [float(row_map[(task, variant)]["value"]) for variant in variants]
        audit_values[f"{task}_{metric}"] = values
        x = np.arange(len(variants))
        lower = 0.625 if metric == "accuracy" else 0.730
        upper = 0.678 if metric == "accuracy" else 0.840
        bars = axis.bar(
            x,
            np.asarray(values) - lower,
            bottom=lower,
            width=0.67,
            color=colors,
            edgecolor="white",
            linewidth=0.55,
        )
        axis.set_ylim(lower, upper)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        annotation_offset = (upper - lower) * 0.025
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + annotation_offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=DARK_GRAY,
            )
        polish_axis(axis)
        panel_caption(
            axis,
            marker,
            "Classification" if task == "classification" else "Regression",
            y=-0.40,
        )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.96, bottom=0.30, wspace=0.24)
    save_figure(figure, AGG_STEM)
    return {"source": relative(AGG_SOURCE), "variants": list(variants), "values": audit_values}


def inference_cost_figure() -> dict[str, object]:
    cost_rows = read_rows(COST_SOURCE)
    cache_rows = read_rows(CACHE_SOURCE)
    cost_map = {(row["task"], row["method"]): row for row in cost_rows}
    cache_map = {row["family"]: row for row in cache_rows}

    methods = ("tabicl_base", "random_context_6", "shared_context_6", "specific_context_6")
    method_labels = ("Base", "Random", "Shared", "Specific")
    figure, axes = plt.subplots(1, 4, figsize=(7.08, 2.12))
    audit: dict[str, object] = {
        "context_protocol": {
            "alternative_contexts": 5,
            "base_evaluated_separately": True,
            "total_backbone_calls": 6,
        },
        "components_ms": {},
        "cache_ms": {},
        "amortized_speedup": {},
    }

    for axis, task, marker in zip(axes[:2], ("classification", "regression"), ("(a)", "(b)")):
        backbone = np.asarray(
            [float(cost_map[(task, method)]["median_backbone_seconds"]) * 1000 for method in methods]
        )
        measured_context = np.asarray(
            [float(cost_map[(task, method)]["median_context_and_fusion_seconds"]) * 1000 for method in methods]
        )
        total = np.asarray(
            [float(cost_map[(task, method)]["median_total_seconds"]) * 1000 for method in methods]
        )
        # These quantities are marginal medians over the 21 cells.  Derive the
        # displayed residual from the median total so every stacked bar ends at
        # its annotated end-to-end latency.
        displayed_context = np.maximum(total - backbone, 0.0)
        y = np.arange(len(methods))
        axis.barh(y, backbone, color=BLUE, height=0.58, label="Backbone")
        axis.barh(
            y,
            displayed_context,
            left=backbone,
            color="#C89B3C",
            height=0.58,
            label="Context + fusion",
        )
        axis.set_yticks(y, method_labels)
        axis.invert_yaxis()
        axis.set_xlabel("Latency (ms)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        for yi, value in zip(y, total):
            axis.text(value + max(total) * 0.018, yi, f"{value:.0f}", va="center", fontsize=5.8)
        axis.set_xlim(0, max(total) * 1.18)
        polish_axis(axis, grid_axis="x")
        panel_caption(axis, marker, "Classification" if task == "classification" else "Regression", y=-0.31)
        audit["components_ms"][task] = {
            "methods": list(methods),
            "total": total.tolist(),
            "backbone": backbone.tolist(),
            "measured_context_and_fusion_median": measured_context.tolist(),
            "displayed_context_and_fusion": displayed_context.tolist(),
        }

    family_order = ("shared_context_6", "specific_context_6")
    family_labels = ("Shared", "Specific")
    original = np.asarray([float(cache_map[key]["median_original_seconds"]) * 1000 for key in family_order])
    cached = np.asarray([float(cache_map[key]["median_cached_seconds"]) * 1000 for key in family_order])
    x = np.arange(len(family_order))
    width = 0.34
    axes[2].bar(x - width / 2, original, width, color=DARK_GRAY, label="Original")
    axes[2].bar(x + width / 2, cached, width, color="#62A87C", label="Cached")
    axes[2].set_xticks(x, family_labels)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4))
    for xi, before, after in zip(x, original, cached):
        axes[2].text(
            xi + width / 2,
            after + max(original) * 0.04,
            f"{before / after:.1f}" + r"$\times$",
            ha="center",
            fontsize=6.0,
        )
    polish_axis(axes[2])
    panel_caption(axes[2], "(c)", "Propensity cache", y=-0.31)
    audit["cache_ms"] = {
        "families": list(family_order),
        "original": original.tolist(),
        "cached": cached.tolist(),
    }

    batches = np.asarray([1, 2, 5, 10], dtype=float)
    setup = np.asarray([float(cache_map[key]["median_propensity_setup_seconds"]) for key in family_order])
    original_seconds = np.asarray([float(cache_map[key]["median_original_seconds"]) for key in family_order])
    cached_seconds = np.asarray([float(cache_map[key]["median_cached_seconds"]) for key in family_order])
    for idx, (label, color, marker) in enumerate(zip(family_labels, (BLUE, GREEN), ("o", "s"))):
        speedup = batches * original_seconds[idx] / (setup[idx] + batches * cached_seconds[idx])
        axes[3].plot(batches, speedup, color=color, marker=marker, markersize=3.0, linewidth=1.35, label=label)
        audit["amortized_speedup"][family_order[idx]] = speedup.tolist()
    axes[3].axhline(1.0, color="#888888", linewidth=0.65, linestyle=(0, (2, 2)))
    axes[3].set_xticks(batches.astype(int))
    axes[3].set_xlabel("Batches")
    axes[3].set_ylabel("Speedup")
    axes[3].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[3].legend(frameon=False, loc="lower right", handlelength=1.3, borderaxespad=0.1)
    polish_axis(axes[3])
    panel_caption(axes[3], "(d)", "Amortized speedup", y=-0.31)

    handles = [
        Line2D([0], [0], color=BLUE, linewidth=5, label="Backbone"),
        Line2D([0], [0], color="#C89B3C", linewidth=5, label="Context + fusion"),
        Line2D([0], [0], color=DARK_GRAY, linewidth=5, label="Original"),
        Line2D([0], [0], color="#62A87C", linewidth=5, label="Cached"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.065, right=0.995, top=0.83, bottom=0.23, wspace=0.53)
    save_figure(figure, COST_STEM)
    return {
        "component_source": relative(COST_SOURCE),
        "cache_source": relative(CACHE_SOURCE),
        **audit,
    }


def main() -> None:
    configure_style()
    audit = {
        "context_count": context_count_audit(),
        "context_bank": aggregation_figure(),
        "inference_cost": inference_cost_figure(),
        "outputs": [
            relative(path.with_suffix(suffix))
            for path in (AGG_STEM, COST_STEM)
            for suffix in (".pdf", ".svg", ".png")
        ],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Generated supplemental figures; audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
