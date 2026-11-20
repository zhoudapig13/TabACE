#!/usr/bin/env python3
"""Generate the compact publication dashboard for Experiment 41."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment41" / "results" / "confirmation"
FIGURES = ROOT / "paper" / "figures"
STABILITY_STEM = FIGURES / "tabace_synthetic_stability"
SEVERITY_STEM = FIGURES / "tabace_synthetic_severity"
COMBINED_STEM = FIGURES / "tabace_synthetic_robustness"
CONTROLLED_STEM = FIGURES / "tabace_controlled_response"
SHIFT_STEM = FIGURES / "tabace_shift_response"
MERGED_RESPONSE_STEM = FIGURES / "tabace_synthetic_response_merged"
AUDIT_PATH = FIGURES / "tabace_synthetic_figures_audit.json"

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

TASK_SPECS = {
    "classification": {
        "title": "Classification",
        "metric": "auc",
        "metric_label": "AUROC",
        "metric_short": "AUC",
        "stability_normalization": "chance-adjusted AUROC margin",
        "panel_markers": ("(a)", "(c)"),
    },
    "regression": {
        "title": "Regression",
        "metric": "r2",
        "metric_label": "R²",
        "metric_short": "R²",
        "stability_normalization": "normalized R² drop",
        "panel_markers": ("(b)", "(d)"),
    },
}

CURVE_METHODS = {
    "classification": (
        "catboost_hpo",
        "group_dro",
        "cvar_dro",
        "coral",
        "caa_at_cutmix",
        "drift_resilient_tabpfn",
        "ftat",
        "distpfn",
        "elliptic_loss",
        "tabace_shared",
    ),
    "regression": (
        "catboost_hpo",
        "group_dro",
        "cvar_dro",
        "coral",
        "pgd_at_mixup",
        "dann",
        "ssa",
        "vrex",
        "elliptic_loss",
        "tabace_shared",
    ),
}

DISPLAY = {
    "classification": {
        "catboost_hpo": "CatBoost + HPO",
        "group_dro": "GroupDRO",
        "cvar_dro": "CVaR-DRO",
        "coral": "CORAL",
        "caa_at_cutmix": "CAA-AT + CutMix",
        "drift_resilient_tabpfn": "Drift-Resilient TabPFN",
        "ftat": "FTAT",
        "distpfn": "DistPFN",
        "elliptic_loss": "Elliptic + IW",
        "tabace_shared": "TabACE-Shared",
    },
    "regression": {
        "catboost_hpo": "CatBoost + HPO",
        "group_dro": "GroupDRO",
        "cvar_dro": "CVaR-DRO",
        "coral": "CORAL",
        "pgd_at_mixup": "PGD-AT + Mixup",
        "dann": "DANN",
        "ssa": "SSA",
        "vrex": "V-REx",
        "elliptic_loss": "Elliptic",
        "tabace_shared": "TabACE-Shared",
    },
}

COLORS = (
    "#4477AA",
    "#56B4E9",
    "#228833",
    "#CCBB44",
    "#EE6677",
    "#AA3377",
    "#999999",
    "#222222",
    "#666666",
    "#8B1A1A",
)

LINE_STYLES = (
    "-",
    (0, (4.0, 1.8)),
    (0, (1.3, 1.4)),
    (0, (5.0, 1.5, 1.2, 1.5)),
    "-",
    (0, (3.0, 1.3)),
    (0, (1.0, 1.2)),
    (0, (5.5, 1.5)),
    (0, (2.2, 1.3)),
    "-",
)

MARKERS = ("o", "s", "^", "v", "P", "X", "<", ">", "D", "o")
STRENGTHS = (0.0, 0.2, 0.5, 0.8, 1.0)
SCENARIO_TYPES = ("manipulation", "shift")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": PLOT_FONT,
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("pdf", "svg"):
        figure.savefig(
            stem.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.025
        )
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_stability(
) -> dict[str, tuple[list[str], np.ndarray, list[dict[str, str]]]]:
    loaded: dict[
        str, tuple[list[str], np.ndarray, list[dict[str, str]]]
    ] = {}
    for task in TASK_SPECS:
        matrix_path = RESULTS / f"stability_pairwise_{task}_all12.csv"
        ranking_path = RESULTS / f"stability_ranking_{task}_all12.csv"
        with matrix_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            matrix_columns = header[1:]
            matrix_rows = list(reader)
        matrix_index = [row[0] for row in matrix_rows]
        values = np.asarray(
            [
                [float(value) if value else math.nan for value in row[1:]]
                for row in matrix_rows
            ],
            dtype=float,
        )
        ranking = read_dict_rows(ranking_path)
        displays = [row["display"] for row in ranking]
        observed_metrics = {row.get("stability_metric", "") for row in ranking}
        if observed_metrics != {str(TASK_SPECS[task]["metric"])}:
            raise ValueError(
                f"{task}: stability ranking metric {observed_metrics} does not "
                f"match {TASK_SPECS[task]['metric']}"
            )
        if matrix_index != displays or matrix_columns != displays:
            raise ValueError(f"{task}: stability matrix and ranking order disagree")
        if values.shape != (10, 10) or len(ranking) != 10:
            raise ValueError(f"{task}: expected a 10 x 10 matrix and 10 ranking rows")

        for row in range(10):
            if not math.isnan(values[row, row]):
                raise ValueError(f"{task}: diagonal entry {row} must be empty")
            for column in range(10):
                if row == column:
                    continue
                if not math.isclose(
                    values[row, column] + values[column, row],
                    100.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(f"{task}: pairwise complement check failed")
        observed_means = np.nanmean(values, axis=1)
        if not np.allclose(
            observed_means,
            np.asarray(
                [float(item["mean_stability_win_rate"]) for item in ranking]
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{task}: displayed mean win rates disagree with matrix")
        loaded[task] = (displays, values, ranking)
    return loaded


def render_stability(
    loaded: dict[str, tuple[list[str], np.ndarray, list[dict[str, str]]]]
) -> None:
    color_map = LinearSegmentedColormap.from_list(
        "stability_win", ["#762A83", "#F7F7F7", "#1B7837"]
    )
    color_map.set_bad("#E6E6E6")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)

    figure = plt.figure(figsize=(7.08, 6.95))
    grid = figure.add_gridspec(
        2,
        3,
        left=0.22,
        right=0.965,
        bottom=0.075,
        top=0.965,
        width_ratios=(10.0, 4.35, 0.34),
        height_ratios=(1.0, 1.0),
        hspace=0.50,
        wspace=0.12,
    )
    color_axis = figure.add_subplot(grid[:, 2])

    for row_index, task in enumerate(("classification", "regression")):
        spec = TASK_SPECS[task]
        displays, values, ranking = loaded[task]
        heat_axis = figure.add_subplot(grid[row_index, 0])
        summary_axis = figure.add_subplot(grid[row_index, 1])
        heat_axis.set_facecolor("#E6E6E6")
        heat_axis.pcolormesh(
            np.arange(11) - 0.5,
            np.arange(11) - 0.5,
            np.ma.masked_invalid(values),
            cmap=color_map,
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.55,
            antialiased=True,
        )
        heat_axis.set_xlim(-0.5, 9.5)
        heat_axis.set_ylim(9.5, -0.5)
        heat_axis.set_aspect("equal")

        row_labels = [
            f"#{int(float(item['stability_rank']))}  {item['display']}  "
            f"({float(item['mean_stability_win_rate']):.1f}%)"
            for item in ranking
        ]
        heat_axis.set_xticks(
            np.arange(10),
            labels=displays,
            rotation=47,
            ha="right",
            rotation_mode="anchor",
        )
        heat_axis.set_yticks(np.arange(10), labels=row_labels)
        heat_axis.tick_params(axis="x", labelsize=5.1, pad=1.5)
        heat_axis.tick_params(axis="y", labelsize=5.35, pad=2.0)
        heat_axis.set_xlabel("Opponent method", labelpad=2.0)
        heat_axis.set_ylabel("Ranked row method", labelpad=3.0)
        heat_axis.set_title(
            f"{spec['title']}: pairwise stability win rate",
            loc="left",
            pad=5.0,
            fontsize=7.2,
            fontweight="normal",
        )

        for matrix_row in range(10):
            for matrix_column in range(10):
                if matrix_row == matrix_column:
                    continue
                value = values[matrix_row, matrix_column]
                text_color = "white" if value < 20.0 or value > 82.0 else "#252525"
                heat_axis.text(
                    matrix_column,
                    matrix_row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.55,
                    color=text_color,
                )

        tabace_row = displays.index("TabACE-Shared")
        heat_axis.add_patch(
            plt.Rectangle(
                (-0.5, tabace_row - 0.5),
                10,
                1,
                fill=False,
                edgecolor="#8B1A1A",
                linewidth=1.15,
            )
        )
        heat_axis.get_yticklabels()[tabace_row].set_color("#8B1A1A")
        heat_axis.get_yticklabels()[tabace_row].set_fontweight("bold")

        metric_short = str(spec["metric_short"])
        summary_axis.set_xlim(0.0, 4.0)
        summary_axis.set_ylim(9.5, -0.5)
        summary_axis.set_xticks(
            (0.5, 1.5, 2.5, 3.5),
            labels=(
                "Mean win\nrate ↑",
                f"Clean\n{metric_short} ↑",
                f"Shifted\n{metric_short} ↑",
                "Mean paired\ndrop ↓",
            ),
        )
        summary_axis.xaxis.tick_top()
        summary_axis.tick_params(axis="x", length=0, pad=3.5, labelsize=5.05)
        summary_axis.set_yticks([])
        summary_axis.set_title(
            "Performance and retention",
            pad=22.0,
            fontsize=6.8,
            fontweight="normal",
        )
        for spine in summary_axis.spines.values():
            spine.set_visible(False)

        for table_row, item in enumerate(ranking):
            is_tabace = item["display"] == "TabACE-Shared"
            background = (
                "#F8E9E6"
                if is_tabace
                else ("#F2F2F2" if table_row % 2 else "#FFFFFF")
            )
            summary_axis.add_patch(
                plt.Rectangle(
                    (0.0, table_row - 0.5),
                    4.0,
                    1.0,
                    facecolor=background,
                    edgecolor="#D0D0D0",
                    linewidth=0.45,
                )
            )
            text_color = "#8B1A1A" if is_tabace else "#222222"
            weight = "bold" if is_tabace else "normal"
            formatted = (
                f"{float(item['mean_stability_win_rate']):.1f}%",
                f"{float(item['clean_mean']):.3f}",
                f"{float(item['shifted_mean']):.3f}",
                f"{100.0 * float(item['mean_relative_drop']):+.1f}%",
            )
            for column, value in enumerate(formatted):
                summary_axis.text(
                    column + 0.5,
                    table_row,
                    value,
                    ha="center",
                    va="center",
                    fontsize=5.55,
                    color=text_color,
                    fontweight=weight,
                )
        summary_axis.vlines(
            (1.0, 2.0, 3.0),
            -0.5,
            9.5,
            color="#D0D0D0",
            linewidth=0.5,
        )
        summary_axis.add_patch(
            plt.Rectangle(
                (0.0, tabace_row - 0.5),
                4.0,
                1.0,
                fill=False,
                edgecolor="#8B1A1A",
                linewidth=1.15,
            )
        )

    for value in range(100):
        color_axis.add_patch(
            plt.Rectangle(
                (0.0, float(value)),
                1.0,
                1.0,
                facecolor=color_map(norm(value + 0.5)),
                edgecolor="none",
            )
        )
    color_axis.set_xlim(0.0, 1.0)
    color_axis.set_ylim(0.0, 100.0)
    color_axis.set_xticks([])
    color_axis.set_yticks((0, 25, 50, 75, 100))
    color_axis.yaxis.tick_right()
    color_axis.yaxis.set_label_position("right")
    color_axis.tick_params(axis="y", labelsize=5.5, pad=2.0)
    color_axis.set_ylabel("Stability win rate (%)", fontsize=6.2, labelpad=5.0)
    save_figure(figure, STABILITY_STEM)


def bootstrap(
    values: np.ndarray, seed: int, repetitions: int = 10_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    random = np.random.default_rng(seed)
    sampled = values[
        random.integers(0, len(values), size=(repetitions, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def load_severity() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[float, np.ndarray]]],
    dict[str, int],
]:
    curves = read_dict_rows(RESULTS / "severity_curves.csv")
    expected = {
        (task, scenario_type, method, strength)
        for task in TASK_SPECS
        for scenario_type in SCENARIO_TYPES
        for method in CURVE_METHODS[task]
        for strength in STRENGTHS[1:]
    }
    observed = {
        (row["task"], row["scenario_type"], row["method"], float(row["strength"]))
        for row in curves
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"severity curve grid mismatch; missing={missing}, extra={extra}")

    clean_values: dict[str, dict[str, np.ndarray]] = {}
    merged_values: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    detail_counts: dict[str, int] = {}
    for task, spec in TASK_SPECS.items():
        detail = read_dict_rows(RESULTS / f"detail_{task}.csv")
        detail_counts[task] = len(detail)
        expected_methods = set(CURVE_METHODS[task])
        clean_rows = [row for row in detail if row["scenario_type"] == "clean"]
        clean_methods = {row["method"] for row in clean_rows}
        if clean_methods != expected_methods:
            raise ValueError(f"{task}: clean method inventory mismatch")
        grouped: dict[tuple[str, str, str], list[float]] = {}
        metric = str(spec["metric"])
        for row in clean_rows:
            key = (row["instance_seed"], row["dgp"], row["method"])
            grouped.setdefault(key, []).append(float(row[metric]))
        by_method: dict[str, list[float]] = {method: [] for method in expected_methods}
        for (_seed, _dgp, method), values in grouped.items():
            by_method[method].append(float(np.mean(values)))
        clean_values[task] = {
            method: np.asarray(values, dtype=float)
            for method, values in by_method.items()
        }
        if any(len(values) != 15 for values in clean_values[task].values()):
            raise ValueError(f"{task}: expected 15 clean seed-DGP units per method")

        nonclean_rows = [row for row in detail if row["scenario_type"] != "clean"]
        merged_by_method: dict[str, dict[float, np.ndarray]] = {}
        for method in CURVE_METHODS[task]:
            merged_by_strength: dict[float, np.ndarray] = {}
            for strength in STRENGTHS[1:]:
                values = np.asarray(
                    [
                        float(row[metric])
                        for row in nonclean_rows
                        if row["method"] == method
                        and float(row["strength"]) == strength
                    ],
                    dtype=float,
                )
                if len(values) != 180:
                    raise ValueError(
                        f"{task}/{method}/{strength}: expected 180 merged "
                        "seed-DGP-scenario units"
                    )
                merged_by_strength[strength] = values
            merged_by_method[method] = merged_by_strength
        merged_values[task] = merged_by_method
    return curves, clean_values, merged_values, detail_counts


def panel_caption(axis: plt.Axes, marker: str, task_name: str) -> None:
    marker_area = TextArea(
        marker,
        textprops={
            "fontsize": 8.0,
            "fontfamily": PANEL_FONT,
            "fontstyle": "italic",
            "fontweight": "normal",
            "color": "#222222",
        },
    )
    name_area = TextArea(
        task_name,
        textprops={
            "fontsize": 8.0,
            "fontfamily": PANEL_FONT,
            "fontstyle": "normal",
            "fontweight": "normal",
            "color": "#222222",
        },
    )
    packed = HPacker(children=[marker_area, name_area], align="baseline", pad=0, sep=2)
    axis.add_artist(
        AnnotationBbox(
            packed,
            (0.5, -0.39),
            xycoords=axis.transAxes,
            box_alignment=(0.5, 1.0),
            frameon=False,
            pad=0,
            annotation_clip=False,
        )
    )


def render_severity(
    curves: list[dict[str, str]],
    clean_values: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.08, 5.75), sharex=True)
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.225,
        top=0.915,
        wspace=0.27,
        hspace=0.76,
    )
    figure.text(
        0.54,
        0.955,
        "Controlled perturbations",
        ha="center",
        va="center",
        fontsize=7.4,
        fontfamily=PLOT_FONT,
    )
    figure.text(
        0.54,
        0.54,
        "Distribution shifts",
        ha="center",
        va="center",
        fontsize=7.4,
        fontfamily=PLOT_FONT,
    )

    for scenario_index, scenario_type in enumerate(SCENARIO_TYPES):
        for task_index, task in enumerate(("classification", "regression")):
            axis = axes[scenario_index, task_index]
            spec = TASK_SPECS[task]

            for method_index, method in enumerate(CURVE_METHODS[task]):
                clean_mean, clean_low, clean_high = bootstrap(
                    clean_values[task][method],
                    9100 + scenario_index * 1000 + task_index * 500 + method_index,
                )
                method_rows = sorted(
                    (
                        row
                        for row in curves
                        if row["task"] == task
                        and row["scenario_type"] == scenario_type
                        and row["method"] == method
                    ),
                    key=lambda row: float(row["strength"]),
                )
                means = np.asarray(
                    (clean_mean, *(float(row["mean"]) for row in method_rows))
                )
                lows = np.asarray(
                    (clean_low, *(float(row["ci_low"]) for row in method_rows))
                )
                highs = np.asarray(
                    (clean_high, *(float(row["ci_high"]) for row in method_rows))
                )
                if len(means) != len(STRENGTHS):
                    raise ValueError(f"{task}/{scenario_type}/{method}: incomplete curve")

                is_tabace = method == "tabace_shared"
                is_elliptic = method == "elliptic_loss"
                line_width = 1.8 if is_tabace else 1.15 if is_elliptic else 0.85
                opacity = 1.0 if is_tabace else 0.9 if is_elliptic else 0.78
                axis.plot(
                    STRENGTHS,
                    means,
                    color=COLORS[method_index],
                    linestyle=LINE_STYLES[method_index],
                    linewidth=line_width,
                    marker=MARKERS[method_index],
                    markersize=2.3 if not is_tabace else 2.8,
                    markerfacecolor=COLORS[method_index],
                    markeredgewidth=0.0,
                    alpha=opacity,
                    zorder=5 if is_tabace else 4 if is_elliptic else 3,
                )
                axis.fill_between(
                    STRENGTHS,
                    lows,
                    highs,
                    color=COLORS[method_index],
                    alpha=0.13 if is_tabace else 0.055,
                    linewidth=0.0,
                    zorder=2 if is_tabace else 1,
                )

            if task == "regression":
                axis.axhline(
                    0.0,
                    color="#8A8A8A",
                    linewidth=0.6,
                    linestyle=(0, (2.0, 2.0)),
                    zorder=0,
                )
            axis.set_ylabel(str(spec["metric_label"]), labelpad=3.0)
            axis.set_xlabel("Strength", labelpad=2.5)
            axis.set_xticks(STRENGTHS, ("0", "0.2", "0.5", "0.8", "1.0"))
            axis.tick_params(labelbottom=True, direction="out", pad=1.8)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(
                axis="y",
                color="#D8D8D8",
                linewidth=0.5,
                linestyle=(0, (1.3, 2.0)),
                zorder=0,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#666666")
            axis.spines["bottom"].set_color("#666666")
            marker = str(spec["panel_markers"][scenario_index])
            panel_caption(axis, marker, str(spec["title"]))

    legend_handles = []
    legend_labels = []
    for index, (class_method, regression_method) in enumerate(
        zip(
            CURVE_METHODS["classification"],
            CURVE_METHODS["regression"],
            strict=True,
        )
    ):
        class_label = DISPLAY["classification"][class_method]
        regression_label = DISPLAY["regression"][regression_method]
        label = (
            class_label
            if class_label == regression_label
            else f"{class_label} / {regression_label}"
        )
        legend_handles.append(
            Line2D(
                (0, 1),
                (0, 0),
                color=COLORS[index],
                linestyle=LINE_STYLES[index],
                linewidth=1.8 if index == 9 else 1.15 if index == 8 else 0.9,
                marker=MARKERS[index],
                markersize=2.7,
                label=label,
            )
        )
        legend_labels.append(label)
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.012),
        ncol=2,
        frameon=False,
        fontsize=5.25,
        title="Classification / regression method mapping",
        title_fontsize=5.75,
        handlelength=2.8,
        handletextpad=0.5,
        columnspacing=1.7,
        labelspacing=0.35,
    )
    save_figure(figure, SEVERITY_STEM)


def compact_panel_label(
    axis: plt.Axes,
    marker: str,
    label: str,
    *,
    x: float,
    y: float,
    box_alignment: tuple[float, float],
    fontsize: float,
) -> None:
    marker_area = TextArea(
        marker,
        textprops={
            "fontsize": fontsize,
            "fontfamily": PANEL_FONT,
            "fontstyle": "italic",
            "fontweight": "normal",
            "color": "#222222",
        },
    )
    label_area = TextArea(
        label,
        textprops={
            "fontsize": fontsize,
            "fontfamily": PANEL_FONT,
            "fontstyle": "normal",
            "fontweight": "normal",
            "color": "#222222",
        },
    )
    packed = HPacker(children=[marker_area, label_area], align="baseline", pad=0, sep=2)
    axis.add_artist(
        AnnotationBbox(
            packed,
            (x, y),
            xycoords=axis.transAxes,
            box_alignment=box_alignment,
            frameon=False,
            pad=0,
            annotation_clip=False,
        )
    )


def draw_compact_stability_task(
    heat_axis: plt.Axes,
    table_axis: plt.Axes,
    task: str,
    loaded: dict[str, tuple[list[str], np.ndarray, list[dict[str, str]]]],
    color_map: LinearSegmentedColormap,
    norm: TwoSlopeNorm,
    marker: str,
) -> None:
    displays, values, ranking = loaded[task]
    heat_axis.set_facecolor("#E6E6E6")
    heat_axis.pcolormesh(
        np.arange(11) - 0.5,
        np.arange(11) - 0.5,
        np.ma.masked_invalid(values),
        cmap=color_map,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.45,
        antialiased=True,
    )
    heat_axis.set_xlim(-0.5, 9.5)
    heat_axis.set_ylim(9.5, -0.5)
    heat_axis.set_aspect("equal")
    rank_labels = tuple(str(index) for index in range(1, 11))
    heat_axis.set_xticks(np.arange(10), labels=rank_labels)
    heat_axis.set_yticks(np.arange(10), labels=rank_labels)
    heat_axis.tick_params(axis="both", labelsize=5.0, pad=1.0)
    heat_axis.set_xlabel("Opponent rank", labelpad=1.5, fontsize=5.7)
    heat_axis.set_ylabel("Row rank", labelpad=1.8, fontsize=5.7)
    compact_panel_label(
        heat_axis,
        marker,
        f"{TASK_SPECS[task]['title']} stability",
        x=0.0,
        y=1.075,
        box_alignment=(0.0, 0.5),
        fontsize=8.2,
    )

    for matrix_row in range(10):
        for matrix_column in range(10):
            if matrix_row == matrix_column:
                continue
            value = values[matrix_row, matrix_column]
            text_color = "white" if value < 20.0 or value > 82.0 else "#252525"
            heat_axis.text(
                matrix_column,
                matrix_row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.55,
                color=text_color,
            )

    tabace_row = displays.index("TabACE-Shared")
    heat_axis.add_patch(
        plt.Rectangle(
            (-0.5, tabace_row - 0.5),
            10,
            1,
            fill=False,
            edgecolor="#8B1A1A",
            linewidth=1.05,
        )
    )

    column_bounds = (0.0, 2.65, 3.45, 4.20, 4.95, 6.0)
    column_centers = tuple(
        (left + right) / 2.0
        for left, right in zip(column_bounds[:-1], column_bounds[1:], strict=True)
    )
    table_axis.set_xlim(column_bounds[0], column_bounds[-1])
    table_axis.set_ylim(10.35, -1.15)
    table_axis.axis("off")
    headers = ("Rank / method", "Win", "Clean", "Shift", "Drop")
    for column, header in enumerate(headers):
        table_axis.text(
            column_centers[column] if column else 0.08,
            -0.62,
            header,
            ha="center" if column else "left",
            va="center",
            fontsize=4.55,
            fontweight="bold",
            color="#333333",
        )
    table_axis.hlines(-0.12, 0.0, 6.0, color="#777777", linewidth=0.55)

    for table_row, item in enumerate(ranking):
        is_tabace = item["display"] == "TabACE-Shared"
        background = (
            "#F8E9E6"
            if is_tabace
            else ("#F3F3F3" if table_row % 2 else "#FFFFFF")
        )
        table_axis.add_patch(
            plt.Rectangle(
                (0.0, table_row),
                6.0,
                1.0,
                facecolor=background,
                edgecolor="none",
            )
        )
        text_color = "#8B1A1A" if is_tabace else "#222222"
        weight = "bold" if is_tabace else "normal"
        method_label = f"{int(float(item['stability_rank']))}. {item['display']}"
        values_formatted = (
            f"{float(item['mean_stability_win_rate']):.1f}",
            f"{float(item['clean_mean']):.3f}",
            f"{float(item['shifted_mean']):.3f}",
            f"{100.0 * float(item['mean_relative_drop']):+.1f}%",
        )
        table_axis.text(
            0.08,
            table_row + 0.5,
            method_label,
            ha="left",
            va="center",
            fontsize=4.55,
            color=text_color,
            fontweight=weight,
        )
        for column, value in enumerate(values_formatted, start=1):
            table_axis.text(
                column_centers[column],
                table_row + 0.5,
                value,
                ha="center",
                va="center",
                fontsize=4.55,
                color=text_color,
                fontweight=weight,
            )
    table_axis.vlines(
        column_bounds[1:-1],
        -0.12,
        10.0,
        color="#D2D2D2",
        linewidth=0.35,
    )
    table_axis.add_patch(
        plt.Rectangle(
            (0.0, tabace_row),
            6.0,
            1.0,
            fill=False,
            edgecolor="#8B1A1A",
            linewidth=0.8,
        )
    )


def draw_compact_severity_axis(
    axis: plt.Axes,
    task: str,
    scenario_type: str,
    curves: list[dict[str, str]],
    clean_values: dict[str, dict[str, np.ndarray]],
    marker: str,
    panel_name: str,
    panel_seed: int,
) -> None:
    spec = TASK_SPECS[task]
    for method_index, method in enumerate(CURVE_METHODS[task]):
        clean_mean, clean_low, clean_high = bootstrap(
            clean_values[task][method], panel_seed + method_index
        )
        method_rows = sorted(
            (
                row
                for row in curves
                if row["task"] == task
                and row["scenario_type"] == scenario_type
                and row["method"] == method
            ),
            key=lambda row: float(row["strength"]),
        )
        means = np.asarray((clean_mean, *(float(row["mean"]) for row in method_rows)))
        lows = np.asarray((clean_low, *(float(row["ci_low"]) for row in method_rows)))
        highs = np.asarray((clean_high, *(float(row["ci_high"]) for row in method_rows)))
        if len(means) != len(STRENGTHS):
            raise ValueError(f"{task}/{scenario_type}/{method}: incomplete curve")

        is_tabace = method == "tabace_shared"
        is_elliptic = method == "elliptic_loss"
        axis.plot(
            STRENGTHS,
            means,
            color=COLORS[method_index],
            linestyle=LINE_STYLES[method_index],
            linewidth=1.55 if is_tabace else 0.95 if is_elliptic else 0.62,
            marker=MARKERS[method_index],
            markersize=2.15 if is_tabace else 1.7,
            markerfacecolor=COLORS[method_index],
            markeredgewidth=0.0,
            alpha=1.0 if is_tabace else 0.88 if is_elliptic else 0.72,
            zorder=5 if is_tabace else 4 if is_elliptic else 3,
        )
        axis.fill_between(
            STRENGTHS,
            lows,
            highs,
            color=COLORS[method_index],
            alpha=0.12 if is_tabace else 0.045,
            linewidth=0.0,
            zorder=2 if is_tabace else 1,
        )

    if task == "regression":
        axis.axhline(
            0.0,
            color="#8A8A8A",
            linewidth=0.5,
            linestyle=(0, (2.0, 2.0)),
            zorder=0,
        )
    axis.set_ylabel(str(spec["metric_label"]), labelpad=1.6, fontsize=5.4)
    axis.set_xlabel("Strength", labelpad=1.3, fontsize=5.4)
    axis.set_xticks(STRENGTHS, ("0", ".2", ".5", ".8", "1"))
    axis.tick_params(labelsize=4.6, direction="out", pad=1.0)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.4,
        linestyle=(0, (1.3, 2.0)),
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    compact_panel_label(
        axis,
        marker,
        panel_name,
        x=0.5,
        y=-0.36,
        box_alignment=(0.5, 1.0),
        fontsize=5.75,
    )


def render_compact_dashboard(
    stability: dict[str, tuple[list[str], np.ndarray, list[dict[str, str]]]],
    curves: list[dict[str, str]],
    clean_values: dict[str, dict[str, np.ndarray]],
) -> None:
    color_map = LinearSegmentedColormap.from_list(
        "stability_win_compact", ["#762A83", "#F7F7F7", "#1B7837"]
    )
    color_map.set_bad("#E6E6E6")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)

    figure = plt.figure(figsize=(7.08, 4.55))
    outer = figure.add_gridspec(
        3,
        1,
        left=0.06,
        right=0.985,
        bottom=0.150,
        top=0.965,
        height_ratios=(1.90, 1.02, 1.28),
        hspace=0.22,
    )
    heat_grid = outer[0].subgridspec(
        1, 3, width_ratios=(1.0, 1.0, 0.035), wspace=0.22
    )
    table_grid = outer[1].subgridspec(1, 2, wspace=0.13)
    heat_axes = (
        figure.add_subplot(heat_grid[0, 0]),
        figure.add_subplot(heat_grid[0, 1]),
    )
    table_axes = (
        figure.add_subplot(table_grid[0, 0]),
        figure.add_subplot(table_grid[0, 1]),
    )
    draw_compact_stability_task(
        heat_axes[0], table_axes[0], "classification", stability, color_map, norm, "(a)"
    )
    draw_compact_stability_task(
        heat_axes[1], table_axes[1], "regression", stability, color_map, norm, "(b)"
    )

    color_axis = figure.add_subplot(heat_grid[0, 2])
    for value in range(100):
        color_axis.add_patch(
            plt.Rectangle(
                (0.0, float(value)),
                1.0,
                1.0,
                facecolor=color_map(norm(value + 0.5)),
                edgecolor="none",
            )
        )
    color_axis.set_xlim(0.0, 1.0)
    color_axis.set_ylim(0.0, 100.0)
    color_axis.set_xticks([])
    color_axis.set_yticks((0, 50, 100))
    color_axis.yaxis.tick_right()
    color_axis.yaxis.set_label_position("right")
    color_axis.tick_params(axis="y", labelsize=4.7, pad=1.2)
    color_axis.set_ylabel("Win rate (%)", fontsize=5.3, labelpad=2.5)

    severity_grid = outer[2].subgridspec(1, 4, wspace=0.34)
    severity_specs = (
        ("classification", "manipulation", "(c)", "Controlled · Accuracy"),
        ("regression", "manipulation", "(d)", "Controlled · R²"),
        ("classification", "shift", "(e)", "Shift · Accuracy"),
        ("regression", "shift", "(f)", "Shift · R²"),
    )
    for index, (task, scenario_type, marker, panel_name) in enumerate(severity_specs):
        axis = figure.add_subplot(severity_grid[0, index])
        draw_compact_severity_axis(
            axis,
            task,
            scenario_type,
            curves,
            clean_values,
            marker,
            panel_name,
            12500 + index * 500,
        )

    legend_labels = (
        "CatBoost + HPO",
        "GroupDRO",
        "CVaR-DRO",
        "CORAL",
        "CAA-AT / PGD-AT",
        "DriftPFN / DANN",
        "FTAT / SSA",
        "DistPFN / V-REx",
        "Elliptic",
        "TabACE-Shared",
    )
    legend_handles = [
        Line2D(
            (0, 1),
            (0, 0),
            color=COLORS[index],
            linestyle=LINE_STYLES[index],
            linewidth=1.55 if index == 9 else 0.95 if index == 8 else 0.68,
            marker=MARKERS[index],
            markersize=2.2,
        )
        for index in range(10)
    ]
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.012),
        ncol=5,
        frameon=False,
        fontsize=4.55,
        handlelength=2.1,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.25,
    )
    save_figure(figure, COMBINED_STEM)


def draw_conference_response_axis(
    axis: plt.Axes,
    task: str,
    scenario_type: str,
    curves: list[dict[str, str]],
    clean_values: dict[str, dict[str, np.ndarray]],
    bootstrap_seed: int,
) -> None:
    spec = TASK_SPECS[task]
    for method_index, method in enumerate(CURVE_METHODS[task]):
        clean_mean, clean_low, clean_high = bootstrap(
            clean_values[task][method], bootstrap_seed + method_index
        )
        method_rows = sorted(
            (
                row
                for row in curves
                if row["task"] == task
                and row["scenario_type"] == scenario_type
                and row["method"] == method
            ),
            key=lambda row: float(row["strength"]),
        )
        means = np.asarray((clean_mean, *(float(row["mean"]) for row in method_rows)))
        lows = np.asarray((clean_low, *(float(row["ci_low"]) for row in method_rows)))
        highs = np.asarray((clean_high, *(float(row["ci_high"]) for row in method_rows)))
        if len(means) != len(STRENGTHS):
            raise ValueError(f"{task}/{scenario_type}/{method}: incomplete curve")

        is_tabace = method == "tabace_shared"
        is_elliptic = method == "elliptic_loss"
        line_width = 2.15 if is_tabace else 1.25 if is_elliptic else 0.9
        opacity = 1.0 if is_tabace else 0.92 if is_elliptic else 0.76
        axis.plot(
            STRENGTHS,
            means,
            color=COLORS[method_index],
            linestyle=LINE_STYLES[method_index],
            linewidth=line_width,
            marker=MARKERS[method_index],
            markersize=3.4 if is_tabace else 2.7,
            markerfacecolor=COLORS[method_index],
            markeredgewidth=0.0,
            alpha=opacity,
            zorder=5 if is_tabace else 4 if is_elliptic else 3,
        )
        axis.fill_between(
            STRENGTHS,
            lows,
            highs,
            color=COLORS[method_index],
            alpha=0.13 if is_tabace else 0.055,
            linewidth=0.0,
            zorder=2 if is_tabace else 1,
        )

    if task == "regression":
        axis.axhline(
            0.0,
            color="#777777",
            linewidth=0.65,
            linestyle=(0, (2.4, 2.2)),
            zorder=0,
        )
    axis.set_ylabel(str(spec["metric_label"]), labelpad=3.2, fontsize=7.2)
    axis.set_xlabel("Strength", labelpad=2.2, fontsize=7.2)
    axis.set_xticks(STRENGTHS, ("0", "0.2", "0.5", "0.8", "1.0"))
    axis.tick_params(labelsize=6.6, direction="out", pad=1.8)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(
        axis="y",
        color="#D5D5D5",
        linewidth=0.5,
        linestyle=(0, (1.2, 2.0)),
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)


def response_legend_handles() -> tuple[list[Line2D], tuple[str, ...]]:
    labels = (
        "CatBoost + HPO",
        "GroupDRO",
        "CVaR-DRO",
        "CORAL",
        "CAA-AT / PGD-AT",
        "DriftPFN / DANN",
        "FTAT / SSA",
        "DistPFN / V-REx",
        "Elliptic",
        "TabACE-Shared",
    )
    handles = [
        Line2D(
            (0, 1),
            (0, 0),
            color=COLORS[index],
            linestyle=LINE_STYLES[index],
            linewidth=2.15 if index == 9 else 1.25 if index == 8 else 0.9,
            marker=MARKERS[index],
            markersize=3.1,
        )
        for index in range(10)
    ]
    return handles, labels


def compact_stability_label(display: str) -> str:
    return {
        "CAA-AT + CutMix": "CAA-AT",
        "PGD-AT + Mixup": "PGD-AT",
        "Drift-Resilient TabPFN": "DriftPFN",
        "Elliptic + IW": "Elliptic",
    }.get(display, display)


def draw_square_stability_axis(
    axis: plt.Axes,
    task: str,
    loaded: dict[str, tuple[list[str], np.ndarray, list[dict[str, str]]]],
    marker: str,
) -> None:
    color_map = LinearSegmentedColormap.from_list(
        "stability_win_square", ["#762A83", "#F7F7F7", "#1B7837"]
    )
    color_map.set_bad("#E6E6E6")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    displays, values, ranking = loaded[task]

    axis.set_facecolor("#E6E6E6")
    axis.pcolormesh(
        np.arange(11) - 0.5,
        np.arange(11) - 0.5,
        np.ma.masked_invalid(values),
        cmap=color_map,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.50,
        antialiased=True,
    )
    axis.set_xlim(-0.5, 9.5)
    axis.set_ylim(9.5, -0.5)
    axis.set_aspect("equal")

    ranks = [int(float(item["stability_rank"])) for item in ranking]
    row_labels = [
        f"#{rank} {compact_stability_label(item['display'])}"
        for rank, item in zip(ranks, ranking)
    ]
    axis.set_xticks(np.arange(10), labels=[f"#{rank}" for rank in ranks])
    axis.set_yticks(np.arange(10), labels=row_labels)
    axis.tick_params(axis="x", labelsize=5.7, pad=1.4)
    axis.tick_params(axis="y", labelsize=5.35, pad=1.8)
    axis.set_xlabel("Opponent rank", labelpad=1.8, fontsize=6.4)

    for matrix_row in range(10):
        for matrix_column in range(10):
            if matrix_row == matrix_column:
                continue
            value = values[matrix_row, matrix_column]
            text_color = "white" if value < 20.0 or value > 82.0 else "#252525"
            axis.text(
                matrix_column,
                matrix_row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.55,
                color=text_color,
            )

    tabace_row = displays.index("TabACE-Shared")
    axis.add_patch(
        plt.Rectangle(
            (-0.5, tabace_row - 0.5),
            10,
            1,
            fill=False,
            edgecolor="#8B1A1A",
            linewidth=1.10,
        )
    )
    axis.get_yticklabels()[tabace_row].set_color("#8B1A1A")
    axis.get_yticklabels()[tabace_row].set_fontweight("bold")
    compact_panel_label(
        axis,
        marker,
        f"{TASK_SPECS[task]['title']} stability ({TASK_SPECS[task]['metric_label']})",
        x=0.5,
        y=1.04,
        box_alignment=(0.5, 0.0),
        fontsize=8.2,
    )


def draw_merged_response_axis(
    axis: plt.Axes,
    task: str,
    clean_values: dict[str, dict[str, np.ndarray]],
    merged_values: dict[str, dict[str, dict[float, np.ndarray]]],
) -> None:
    task_index = 0 if task == "classification" else 1
    spec = TASK_SPECS[task]
    for method_index, method in enumerate(CURVE_METHODS[task]):
        clean_mean, clean_low, clean_high = bootstrap(
            clean_values[task][method], 18100 + task_index * 500 + method_index
        )
        summaries = [
            bootstrap(
                merged_values[task][method][strength],
                19100
                + task_index * 1000
                + method_index * 100
                + strength_index,
            )
            for strength_index, strength in enumerate(STRENGTHS[1:])
        ]
        means = np.asarray((clean_mean, *(item[0] for item in summaries)))
        lows = np.asarray((clean_low, *(item[1] for item in summaries)))
        highs = np.asarray((clean_high, *(item[2] for item in summaries)))

        is_tabace = method == "tabace_shared"
        is_elliptic = method == "elliptic_loss"
        axis.plot(
            STRENGTHS,
            means,
            color=COLORS[method_index],
            linestyle=LINE_STYLES[method_index],
            linewidth=2.1 if is_tabace else 1.25 if is_elliptic else 0.9,
            marker=MARKERS[method_index],
            markersize=3.25 if is_tabace else 2.6,
            markerfacecolor=COLORS[method_index],
            markeredgewidth=0.0,
            alpha=1.0 if is_tabace else 0.92 if is_elliptic else 0.76,
            zorder=5 if is_tabace else 4 if is_elliptic else 3,
        )
        axis.fill_between(
            STRENGTHS,
            lows,
            highs,
            color=COLORS[method_index],
            alpha=0.13 if is_tabace else 0.05,
            linewidth=0.0,
            zorder=2 if is_tabace else 1,
        )

    if task == "regression":
        axis.axhline(
            0.0,
            color="#777777",
            linewidth=0.65,
            linestyle=(0, (2.4, 2.2)),
            zorder=0,
        )
    axis.set_xlim(-0.04, 1.04)
    axis.set_xticks(STRENGTHS, ("0", "0.2", "0.5", "0.8", "1.0"))
    axis.set_ylabel(str(spec["metric_label"]), labelpad=3.0, fontsize=7.2)
    axis.set_xlabel("Strength", labelpad=2.0, fontsize=7.2)
    axis.tick_params(labelsize=6.5, direction="out", pad=1.6)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(
        axis="y",
        color="#D5D5D5",
        linewidth=0.5,
        linestyle=(0, (1.2, 2.0)),
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)


def render_square_stability_pair(
    stability: dict[
        str, tuple[list[str], np.ndarray, list[dict[str, str]]]
    ],
) -> None:
    figure = plt.figure(figsize=(7.08, 3.20))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.16,
        right=0.985,
        bottom=0.11,
        top=0.91,
        width_ratios=(1.0, 1.0),
        wspace=0.64,
    )
    for task_index, task in enumerate(("classification", "regression")):
        axis = figure.add_subplot(grid[0, task_index])
        draw_square_stability_axis(
            axis,
            task,
            stability,
            "(a)" if task_index == 0 else "(b)",
        )
    save_figure(figure, STABILITY_STEM)


def render_combined_robustness(
    stability: dict[
        str, tuple[list[str], np.ndarray, list[dict[str, str]]]
    ],
    clean_values: dict[str, dict[str, np.ndarray]],
    merged_values: dict[str, dict[str, dict[float, np.ndarray]]],
) -> None:
    figure = plt.figure(figsize=(7.08, 5.70))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.16,
        right=0.985,
        bottom=0.16,
        top=0.965,
        width_ratios=(1.0, 1.0),
        height_ratios=(1.2, 0.9),
        hspace=0.28,
        wspace=0.64,
    )

    for task_index, task in enumerate(("classification", "regression")):
        heat_axis = figure.add_subplot(grid[0, task_index])
        draw_square_stability_axis(
            heat_axis,
            task,
            stability,
            "(a)" if task_index == 0 else "(b)",
        )

        response_axis = figure.add_subplot(grid[1, task_index])
        draw_merged_response_axis(
            response_axis,
            task,
            clean_values,
            merged_values,
        )
        compact_panel_label(
            response_axis,
            "(c)" if task_index == 0 else "(d)",
            f"{TASK_SPECS[task]['title']} response",
            x=0.5,
            y=-0.28,
            box_alignment=(0.5, 1.0),
            fontsize=8.0,
        )

    handles, labels = response_legend_handles()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.012),
        ncol=5,
        frameon=False,
        fontsize=5.9,
        handlelength=2.4,
        handletextpad=0.45,
        columnspacing=1.15,
        labelspacing=0.35,
    )
    save_figure(figure, COMBINED_STEM)


def render_paired_response(
    scenario_type: str,
    stem: Path,
    curves: list[dict[str, str]],
    clean_values: dict[str, dict[str, np.ndarray]],
    markers: tuple[str, str],
    heading: str,
    include_legend: bool,
) -> None:
    figure_height = 2.60 if include_legend else 2.18
    figure, axes = plt.subplots(1, 2, figsize=(7.08, figure_height))
    figure.subplots_adjust(
        left=0.072,
        right=0.985,
        bottom=0.34 if include_legend else 0.22,
        top=0.82,
        wspace=0.22,
    )
    figure.text(
        0.52,
        0.94,
        heading,
        ha="center",
        va="center",
        fontsize=8.2,
        fontfamily=PLOT_FONT,
        fontweight="normal",
        color="#222222",
    )

    for task_index, task in enumerate(("classification", "regression")):
        axis = axes[task_index]
        draw_conference_response_axis(
            axis,
            task,
            scenario_type,
            curves,
            clean_values,
            15100 + (1000 if scenario_type == "shift" else 0) + task_index * 500,
        )
        compact_panel_label(
            axis,
            markers[task_index],
            str(TASK_SPECS[task]["title"]),
            x=0.5,
            y=-0.36,
            box_alignment=(0.5, 1.0),
            fontsize=8.0,
        )

    if include_legend:
        handles, labels = response_legend_handles()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.52, 0.015),
            ncol=5,
            frameon=False,
            fontsize=5.9,
            handlelength=2.4,
            handletextpad=0.45,
            columnspacing=1.15,
            labelspacing=0.35,
        )
    save_figure(figure, stem)


def render_two_panel_response(
    clean_values: dict[str, dict[str, np.ndarray]],
    merged_values: dict[str, dict[str, dict[float, np.ndarray]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.08, 2.72))
    figure.subplots_adjust(
        left=0.072,
        right=0.985,
        bottom=0.34,
        top=0.95,
        wspace=0.22,
    )

    for task_index, task in enumerate(("classification", "regression")):
        axis = axes[task_index]
        spec = TASK_SPECS[task]
        for method_index, method in enumerate(CURVE_METHODS[task]):
            clean_mean, clean_low, clean_high = bootstrap(
                clean_values[task][method], 18100 + task_index * 500 + method_index
            )
            summaries = [
                bootstrap(
                    merged_values[task][method][strength],
                    19100
                    + task_index * 1000
                    + method_index * 100
                    + strength_index,
                )
                for strength_index, strength in enumerate(STRENGTHS[1:])
            ]
            means = np.asarray((clean_mean, *(item[0] for item in summaries)))
            lows = np.asarray((clean_low, *(item[1] for item in summaries)))
            highs = np.asarray((clean_high, *(item[2] for item in summaries)))

            is_tabace = method == "tabace_shared"
            is_elliptic = method == "elliptic_loss"
            axis.plot(
                STRENGTHS,
                means,
                color=COLORS[method_index],
                linestyle=LINE_STYLES[method_index],
                linewidth=2.1 if is_tabace else 1.25 if is_elliptic else 0.9,
                marker=MARKERS[method_index],
                markersize=3.25 if is_tabace else 2.6,
                markerfacecolor=COLORS[method_index],
                markeredgewidth=0.0,
                alpha=1.0 if is_tabace else 0.92 if is_elliptic else 0.76,
                zorder=5 if is_tabace else 4 if is_elliptic else 3,
            )
            axis.fill_between(
                STRENGTHS,
                lows,
                highs,
                color=COLORS[method_index],
                alpha=0.13 if is_tabace else 0.05,
                linewidth=0.0,
                zorder=2 if is_tabace else 1,
            )

        if task == "regression":
            axis.axhline(
                0.0,
                color="#777777",
                linewidth=0.65,
                linestyle=(0, (2.4, 2.2)),
                zorder=0,
            )
        axis.set_xlim(-0.04, 1.04)
        axis.set_xticks(STRENGTHS, ("0", "0.2", "0.5", "0.8", "1.0"))
        axis.set_ylabel(str(spec["metric_label"]), labelpad=3.0, fontsize=7.2)
        axis.set_xlabel("Strength", labelpad=2.0, fontsize=7.2)
        axis.tick_params(labelsize=6.5, direction="out", pad=1.6)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(
            axis="y",
            color="#D5D5D5",
            linewidth=0.5,
            linestyle=(0, (1.2, 2.0)),
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.spines["left"].set_linewidth(0.7)
        axis.spines["bottom"].set_linewidth(0.7)
        compact_panel_label(
            axis,
            "(a)" if task_index == 0 else "(b)",
            str(spec["title"]),
            x=0.5,
            y=-0.27,
            box_alignment=(0.5, 1.0),
            fontsize=8.0,
        )

    handles, labels = response_legend_handles()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.012),
        ncol=5,
        frameon=False,
        fontsize=5.9,
        handlelength=2.4,
        handletextpad=0.45,
        columnspacing=1.15,
        labelspacing=0.35,
    )
    save_figure(figure, MERGED_RESPONSE_STEM)


def write_audit(
    stability: dict[
        str, tuple[list[str], np.ndarray, list[dict[str, str]]]
    ],
    curves: list[dict[str, str]],
    detail_counts: dict[str, int],
) -> None:
    payload = {
        "artifacts": {
            "stability": [
                relative(STABILITY_STEM.with_suffix(suffix))
                for suffix in (".pdf", ".svg", ".png")
            ],
            "response": [
                relative(MERGED_RESPONSE_STEM.with_suffix(suffix))
                for suffix in (".pdf", ".svg", ".png")
            ],
            "controlled_response": [
                relative(CONTROLLED_STEM.with_suffix(suffix))
                for suffix in (".pdf", ".svg", ".png")
            ],
            "shift_response": [
                relative(SHIFT_STEM.with_suffix(suffix))
                for suffix in (".pdf", ".svg", ".png")
            ],
        },
        "sources": [
            relative(RESULTS / "stability_pairwise_classification_all12.csv"),
            relative(RESULTS / "stability_ranking_classification_all12.csv"),
            relative(RESULTS / "stability_pairwise_regression_all12.csv"),
            relative(RESULTS / "stability_ranking_regression_all12.csv"),
            relative(RESULTS / "severity_curves.csv"),
            relative(RESULTS / "detail_classification.csv"),
            relative(RESULTS / "detail_regression.csv"),
        ],
        "checks": {
            "stability_shapes": {
                task: list(items[1].shape) for task, items in stability.items()
            },
            "stability_pairwise_complements": True,
            "stability_ranking_alignment": True,
            "stability_metrics": {
                task: {
                    "metric": str(spec["metric"]),
                    "normalization": str(spec["stability_normalization"]),
                }
                for task, spec in TASK_SPECS.items()
            },
            "severity_nonzero_rows": int(len(curves)),
            "severity_expected_nonzero_rows": 160,
            "detail_rows": detail_counts,
            "stability_canvas_inches": [7.08, 3.20],
            "response_canvas_inches": [7.08, 2.72],
            "heatmap_axes": 2,
            "heatmap_summary_tables": 0,
            "heatmap_colorbars": 0,
            "heatmap_cell_font_size_pt": 6.55,
            "response_axes": 2,
            "panels_per_figure": 2,
            "problem_blocks_per_axis": 1,
            "problem_families_merged_per_point": 12,
            "nonclean_units_per_method_strength": 180,
            "vector_live_text": True,
            "embedded_raster_images": False,
            "smoothing_or_fitted_values": False,
        },
    }
    AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    stability = load_stability()
    curves, clean_values, merged_values, detail_counts = load_severity()
    render_square_stability_pair(stability)
    render_paired_response(
        "manipulation",
        CONTROLLED_STEM,
        curves,
        clean_values,
        ("(a)", "(b)"),
        "Controlled Context perturbations",
        True,
    )
    render_paired_response(
        "shift",
        SHIFT_STEM,
        curves,
        clean_values,
        ("(a)", "(b)"),
        "Distribution shifts",
        True,
    )
    render_two_panel_response(clean_values, merged_values)
    write_audit(stability, curves, detail_counts)


if __name__ == "__main__":
    main()
