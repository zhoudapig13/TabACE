#!/usr/bin/env python3
"""Draw the six-metric Context-count ablation directly from its LaTeX table."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
TABLE_SOURCE = PAPER / "tables" / "context_count_ablation.tex"
FIGURES = PAPER / "figures"
OUTPUT_STEM = FIGURES / "tabace_context_count_ablation"
AUDIT_PATH = FIGURES / "tabace_context_count_ablation_audit.json"

# Okabe--Ito accents plus neutral manuscript colors.
CLASSIFICATION = "#0072B2"
REGRESSION = "#D55E00"
BASE_GRAY = "#737373"
GRID_GRAY = "#D9D9D9"
TEXT = "#222222"

PANELS = (
    ("classification", "accuracy", "Classification & Accuracy", "Accuracy", "up", CLASSIFICATION),
    ("classification", "balanced_accuracy", "Balanced Accuracy", "Balanced accuracy", "up", CLASSIFICATION),
    ("classification", "auc", "AUC", "AUC", "up", CLASSIFICATION),
    ("regression", "r2", "$R^2$", r"$R^2$", "up", REGRESSION),
    ("regression", "nmae", "nMAE", "nMAE", "down", REGRESSION),
    ("regression", "nrmse", "nRMSE", "nRMSE", "down", REGRESSION),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.3,
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
    """Extract Base and R=1,...,5 values from the approved LaTeX table."""
    lines = TABLE_SOURCE.read_text(encoding="utf-8").splitlines()
    parsed: dict[tuple[str, str], list[float]] = {}
    for task, metric, needle, _title, _direction, _color in PANELS:
        matches = [line for line in lines if needle in line]
        if len(matches) != 1:
            raise ValueError(f"Expected one row containing {needle!r}; found {len(matches)}")
        values = [float(value) for value in re.findall(r"-?\d+\.\d+", matches[0])]
        if len(values) != 6:
            raise ValueError(f"Expected six numeric values for {metric}; found {values}")
        parsed[(task, metric)] = values
    return parsed


def limits(values: np.ndarray) -> tuple[float, float]:
    """Use a metric-specific raw scale with enough padding for endpoint labels."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = max(span * 0.18, 0.012)
    return lower - padding, upper + padding


def draw_panel(
    axis: plt.Axes,
    values: list[float],
    title: str,
    direction: str,
    color: str,
    letter: str,
) -> None:
    base = values[0]
    positions = np.arange(6)
    curve = np.asarray(values)

    axis.axhline(
        base,
        color=BASE_GRAY,
        linewidth=1.0,
        linestyle=(0, (3.2, 2.2)),
        zorder=1,
    )
    axis.plot(
        positions,
        curve,
        color=color,
        linewidth=1.65,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.05,
        zorder=2,
    )
    axis.scatter(
        [5],
        [curve[-1]],
        s=31,
        color=color,
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )
    axis.scatter(
        [0],
        [base],
        s=29,
        marker="D",
        color=BASE_GRAY,
        edgecolor="white",
        linewidth=0.65,
        zorder=4,
    )

    arrow = "↑" if direction == "up" else "↓"
    axis.set_title(f"({letter})  {title} {arrow}", loc="left", pad=3.2, fontweight="semibold")
    axis.set_xlim(-0.25, 5.22)
    axis.set_ylim(*limits(np.asarray(values)))
    axis.set_xticks(positions, ("Base", "1", "2", "3", "4", "5"))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(axis="y", color=GRID_GRAY, linewidth=0.45, alpha=0.82)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # Place the Base value away from the first segment of the curve.
    base_offset = -5 if curve[1] > base else 5
    axis.annotate(
        f"{base:.4f}",
        xy=(0, base),
        xytext=(3, base_offset),
        textcoords="offset points",
        ha="left",
        va="top" if base_offset < 0 else "bottom",
        fontsize=5.8,
        color=BASE_GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.45, "alpha": 0.88},
    )
    endpoint_offset = -5 if direction == "down" else 5
    axis.annotate(
        f"{curve[-1]:.4f}",
        xy=(5, curve[-1]),
        xytext=(-3, endpoint_offset),
        textcoords="offset points",
        ha="right",
        va="top" if endpoint_offset < 0 else "bottom",
        fontsize=5.9,
        fontweight="semibold",
        color=color,
    )


def save_figure(figure: plt.Figure) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(
            OUTPUT_STEM.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            pad_inches=0.025,
        )
    figure.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def main() -> None:
    configure_style()
    data = table_values()
    figure, axes = plt.subplots(2, 3, figsize=(7.08, 3.42))

    for index, (axis, panel) in enumerate(zip(axes.flat, PANELS)):
        task, metric, _needle, title, direction, color = panel
        draw_panel(
            axis,
            data[(task, metric)],
            title,
            direction,
            color,
            chr(ord("a") + index),
        )

    figure.text(
        0.013,
        0.707,
        "Classification",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="semibold",
    )
    figure.text(
        0.013,
        0.292,
        "Regression",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="semibold",
    )
    figure.text(
        0.53,
        0.018,
        r"Configuration (Base; context budget $R$)",
        ha="center",
        va="bottom",
        fontsize=7.3,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.965,
        bottom=0.125,
        wspace=0.36,
        hspace=0.48,
    )
    save_figure(figure)
    plt.close(figure)

    audit = {
        "source": TABLE_SOURCE.relative_to(PAPER).as_posix(),
        "selection": "Base and R=1,...,5 only; no Pro series",
        "averaging": (
            "four nonzero strengths; equal macro weight for the five-perturbation "
            "and seven-shift tracks"
        ),
        "display": (
            "six raw-metric small multiples with independent y-axis ranges; "
            "each curve begins at Base and continues through R=1,...,5"
        ),
        "values": {
            f"{task}_{metric}": {"base": values[0], "r": values[1:]}
            for (task, metric), values in data.items()
        },
        "outputs": [
            OUTPUT_STEM.with_suffix(suffix).relative_to(PAPER).as_posix()
            for suffix in (".pdf", ".svg", ".png")
        ],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Generated {OUTPUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
