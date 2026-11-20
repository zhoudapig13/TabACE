#!/usr/bin/env python3
"""Create an eight-panel review figure for Context performance and cost."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
TABLE_SOURCE = ROOT / "paper" / "tables" / "context_count_ablation.tex"
TIMING_SOURCE = EXP / "results" / "timing_summary.csv"
FIGURES = EXP / "figures"
OUTPUT_STEM = FIGURES / "context_budget_performance_cost_review"
AUDIT_PATH = FIGURES / "context_budget_performance_cost_review_audit.json"

CLASSIFICATION = "#0072B2"
REGRESSION = "#D55E00"
COST = "#E69F00"
BASE_GRAY = "#737373"
GRID_GRAY = "#D9D9D9"
TEXT = "#222222"

PANELS = (
    ("classification", "accuracy", "Classification & Accuracy", "Accuracy", "up"),
    (
        "classification",
        "balanced_accuracy",
        "Balanced Accuracy",
        "Balanced accuracy",
        "up",
    ),
    ("classification", "auc", "AUC", "AUC", "up"),
    ("regression", "r2", "$R^2$", r"$R^2$", "up"),
    ("regression", "nmae", "nMAE", "nMAE", "down"),
    ("regression", "nrmse", "nRMSE", "nRMSE", "down"),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "text.color": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def table_values() -> dict[tuple[str, str], list[float]]:
    lines = TABLE_SOURCE.read_text(encoding="utf-8").splitlines()
    parsed: dict[tuple[str, str], list[float]] = {}
    for task, metric, needle, _title, _direction in PANELS:
        matches = [line for line in lines if needle in line]
        if len(matches) != 1:
            raise ValueError(f"expected one row containing {needle!r}; found {len(matches)}")
        values = [float(value) for value in re.findall(r"-?\d+\.\d+", matches[0])]
        if len(values) != 6:
            raise ValueError(f"expected six numeric values for {metric}; found {values}")
        parsed[(task, metric)] = values
    return parsed


def timing_values() -> dict[str, dict[str, list[float]]]:
    with TIMING_SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, list[float]]] = {}
    for task in ("classification", "regression"):
        selected = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: int(row["k"]),
        )
        if [int(row["k"]) for row in selected] != list(range(6)):
            raise ValueError(f"incomplete timing curve for {task}")
        output[task] = {
            "median": [float(row["median_relative_to_paired_base"]) for row in selected],
            "p25": [float(row["p25_relative_to_paired_base"]) for row in selected],
            "p75": [float(row["p75_relative_to_paired_base"]) for row in selected],
            "seconds": [float(row["median_total_seconds"]) for row in selected],
        }
    return output


def polish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID_GRAY, linewidth=0.45, alpha=0.82)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlim(-0.25, 5.22)
    axis.set_xticks(np.arange(6), ("Base", "1", "2", "3", "4", "5"))


def performance_limits(values: np.ndarray) -> tuple[float, float]:
    span = float(np.max(values) - np.min(values))
    padding = max(span * 0.18, 0.012)
    return float(np.min(values) - padding), float(np.max(values) + padding)


def draw_performance(
    axis: plt.Axes,
    values: list[float],
    title: str,
    direction: str,
    color: str,
    letter: str,
) -> None:
    curve = np.asarray(values, dtype=np.float64)
    positions = np.arange(6)
    axis.axhline(curve[0], color=BASE_GRAY, linewidth=0.9, linestyle=(0, (3, 2)))
    axis.plot(
        positions,
        curve,
        color=color,
        linewidth=1.55,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    axis.scatter([0], [curve[0]], marker="D", s=25, color=BASE_GRAY, edgecolor="white")
    axis.scatter([5], [curve[-1]], s=27, color=color, edgecolor="white", linewidth=0.6)
    arrow = "↑" if direction == "up" else "↓"
    axis.set_title(f"({letter})  {title} {arrow}", loc="left", pad=3.0, fontweight="semibold")
    axis.set_ylim(*performance_limits(curve))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    polish_axis(axis)


def draw_cost(
    axis: plt.Axes,
    task: str,
    values: dict[str, list[float]],
    letter: str,
) -> None:
    positions = np.arange(6)
    median = np.asarray(values["median"])
    p25 = np.asarray(values["p25"])
    p75 = np.asarray(values["p75"])
    axis.fill_between(positions, p25, p75, color=COST, alpha=0.18, linewidth=0)
    axis.plot(
        positions,
        median,
        color=COST,
        linewidth=1.65,
        marker="o",
        markersize=3.7,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    axis.scatter([0], [1.0], marker="D", s=25, color=BASE_GRAY, edgecolor="white")
    axis.scatter([5], [median[-1]], s=28, color=COST, edgecolor="white", linewidth=0.6)
    axis.axhline(1.0, color=BASE_GRAY, linewidth=0.9, linestyle=(0, (3, 2)))
    axis.set_title(f"({letter})  Relative latency ↓", loc="left", pad=3.0, fontweight="semibold")
    if task == "classification":
        axis.set_yscale("log", base=2)
        axis.set_yticks((1, 2, 4, 8, 16, 32), ("1", "2", "4", "8", "16", "32"))
        axis.set_ylim(0.88, max(48.0, float(np.max(p75)) * 1.12))
    else:
        axis.set_ylim(0.75, float(np.max(p75)) * 1.16)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    axis.set_ylabel("× Base", labelpad=1.2)
    annotation_offset = -8 if task == "classification" else 5
    axis.annotate(
        f"{median[-1]:.2f}×",
        xy=(5, median[-1]),
        xytext=(-3, annotation_offset),
        textcoords="offset points",
        ha="right",
        va="top" if annotation_offset < 0 else "bottom",
        fontsize=6.0,
        fontweight="semibold",
        color="#9A5B00",
    )
    polish_axis(axis)


def main() -> None:
    configure_style()
    performance = table_values()
    timing = timing_values()
    figure, axes = plt.subplots(2, 4, figsize=(7.08, 3.5))

    for index, panel in enumerate(PANELS[:3]):
        task, metric, _needle, title, direction = panel
        draw_performance(
            axes[0, index],
            performance[(task, metric)],
            title,
            direction,
            CLASSIFICATION,
            chr(ord("a") + index),
        )
    draw_cost(axes[0, 3], "classification", timing["classification"], "d")

    for index, panel in enumerate(PANELS[3:]):
        task, metric, _needle, title, direction = panel
        draw_performance(
            axes[1, index],
            performance[(task, metric)],
            title,
            direction,
            REGRESSION,
            chr(ord("e") + index),
        )
    draw_cost(axes[1, 3], "regression", timing["regression"], "h")

    figure.text(
        0.012,
        0.712,
        "Classification",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="semibold",
    )
    figure.text(
        0.012,
        0.295,
        "Regression",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="semibold",
    )
    figure.text(
        0.53,
        0.017,
        r"Configuration (Base; Context budget $K$)",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.965,
        bottom=0.13,
        wspace=0.38,
        hspace=0.5,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(
            OUTPUT_STEM.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            pad_inches=0.025,
        )
    figure.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=360,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)

    audit = {
        "performance_source": TABLE_SOURCE.relative_to(ROOT).as_posix(),
        "timing_source": TIMING_SOURCE.relative_to(ROOT).as_posix(),
        "timing_statistic": "median paired-to-Base ratio with interquartile band",
        "timing_cells_per_task_and_configuration": 63,
        "classification_latency_scale": "base-2 logarithmic",
        "regression_latency_scale": "linear",
        "performance_values": {
            f"{task}_{metric}": values
            for (task, metric), values in performance.items()
        },
        "timing_values": timing,
        "outputs": [
            OUTPUT_STEM.with_suffix(f".{suffix}").relative_to(ROOT).as_posix()
            for suffix in ("pdf", "svg", "png")
        ],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
