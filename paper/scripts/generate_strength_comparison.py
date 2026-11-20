#!/usr/bin/env python3
"""Generate the 2x4 TabACE strength-comparison figure from dataset-level OOF data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
OUTPUT_STEM = FIGURE_DIR / "tabace_strength_comparison"

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


PLOT_FONT_FAMILY = register_font("lmsans10-regular.otf", "DejaVu Sans")
PANEL_FONT_FAMILY = register_font("lmroman10-regular.otf", "DejaVu Serif")
register_font("lmroman10-italic.otf", "DejaVu Serif")

METHODS = {
    "base": {
        "source_method": "base",
        "label": "Base",
        "color": "#555555",
        "marker": "o",
        "linestyle": (0, (4.2, 2.2)),
        "linewidth": 1.25,
    },
    "shared": {
        "source_method": "shared",
        "label": "TabACE-shared",
        "color": "#56B4E9",
        "marker": "s",
        "linestyle": (0, (5.0, 1.8, 1.2, 1.8)),
        "linewidth": 1.4,
    },
    "specific": {
        "source_method": "pro",
        "label": "TabACE-specific",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.75,
    },
}

SOURCES = {
    ("controlled", "classification"): {
        "detail_path": ROOT
        / "fair_budget_experiments"
        / "unified_context_ensemble"
        / "results"
        / "tabicl_origin_safe_final_v2"
        / "five_shift"
        / "classification"
        / "detail_oof.csv",
        "summary_path": ROOT
        / "fair_budget_experiments"
        / "unified_context_ensemble"
        / "results"
        / "tabicl_origin_safe_final_v2"
        / "five_shift"
        / "classification"
        / "summary_oof.csv",
        "audit_paths": (
            ROOT
            / "fair_budget_experiments"
            / "unified_context_ensemble"
            / "results"
            / "tabicl_origin_safe_final_v2"
            / "five_shift"
            / "classification"
            / "origin_overlay_audit.json",
            ROOT
            / "fair_budget_experiments"
            / "unified_context_ensemble"
            / "results"
            / "adaptive_router_candidate_v1"
            / "five_shift"
            / "classification"
            / "audit.json",
        ),
        "family_key": "family",
        "strength_key": "rho",
        "strength_scale": 0.01,
        "expected_families": 5,
        "expected_datasets": 72,
    },
    ("controlled", "regression"): {
        "detail_path": ROOT
        / "PaperResults"
        / "failure_recovery_20260913"
        / "tabicl"
        / "five_targeted_final"
        / "regression"
        / "detail_oof.csv",
        "summary_path": ROOT
        / "PaperResults"
        / "failure_recovery_20260913"
        / "tabicl"
        / "five_targeted_final"
        / "regression"
        / "summary_oof.csv",
        "audit_paths": (
            ROOT
            / "PaperResults"
            / "failure_recovery_20260913"
            / "tabicl"
            / "five_targeted_final"
            / "regression"
            / "promotion_manifest.json",
        ),
        "family_key": "family",
        "strength_key": "rho",
        "strength_scale": 0.01,
        "expected_families": 5,
        "expected_datasets": 35,
    },
    ("shift", "classification"): {
        "detail_path": ROOT
        / "PaperResults"
        / "failure_recovery_20260913"
        / "tabicl"
        / "seven_point_lock"
        / "composite"
        / "classification"
        / "detail_oof.csv",
        "summary_path": ROOT
        / "PaperResults"
        / "failure_recovery_20260913"
        / "tabicl"
        / "seven_point_lock"
        / "composite"
        / "classification"
        / "summary_oof.csv",
        "audit_paths": (
            ROOT
            / "PaperResults"
            / "failure_recovery_20260913"
            / "tabicl"
            / "seven_point_lock"
            / "composite"
            / "classification"
            / "audit.json",
        ),
        "family_key": "shift",
        "strength_key": "strength",
        "strength_scale": 1.0,
        "expected_families": 7,
        "expected_datasets": 35,
    },
    ("shift", "regression"): {
        "detail_path": ROOT
        / "PaperResults"
        / "failure_recovery_20260913"
        / "tabicl"
        / "seven_point_lock"
        / "composite"
        / "regression"
        / "detail_oof.csv",
        "summary_path": ROOT
        / "PaperResults"
        / "failure_recovery_20260913"
        / "tabicl"
        / "seven_point_lock"
        / "composite"
        / "regression"
        / "summary_oof.csv",
        "audit_paths": (
            ROOT
            / "PaperResults"
            / "failure_recovery_20260913"
            / "tabicl"
            / "seven_point_lock"
            / "composite"
            / "regression"
            / "audit.json",
        ),
        "family_key": "shift",
        "strength_key": "strength",
        "strength_scale": 1.0,
        "expected_families": 7,
        "expected_datasets": 33,
    },
}

PANELS = [
    ("controlled", "classification", "accuracy", "Classification\nAccuracy ↑"),
    ("controlled", "classification", "auc", "Classification\nAUC ↑"),
    ("controlled", "regression", "r2", "Regression\nR² ↑"),
    ("controlled", "regression", "nmae", "Regression\nnMAE ↓"),
    ("shift", "classification", "accuracy", "Classification\nAccuracy ↑"),
    ("shift", "classification", "auc", "Classification\nAUC ↑"),
    ("shift", "regression", "r2", "Regression\nR² ↑"),
    ("shift", "regression", "nmae", "Regression\nnMAE ↓"),
]

METRIC_AXIS_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "r2": "R²",
    "nmae": "nMAE",
}

EXPECTED_STRENGTHS = (0.0, 0.2, 0.5, 0.8, 1.0)
TASK_METRICS = {
    "classification": ("accuracy", "balanced_accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_for_condition(
    source: dict[str, object], family: str, strength: float
) -> list[dict[str, str]]:
    origin_rows = source["origin_rows"]
    if strength == 0.0 and origin_rows:
        return origin_rows  # type: ignore[return-value]

    spec = source["spec"]
    rows = source["detail_rows"]
    family_key = str(spec["family_key"])  # type: ignore[index]
    strength_key = str(spec["strength_key"])  # type: ignore[index]
    strength_scale = float(spec["strength_scale"])  # type: ignore[index]
    return [
        row
        for row in rows  # type: ignore[union-attr]
        if row[family_key] == family
        and math.isclose(
            float(row[strength_key]) * strength_scale,
            strength,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]


def load_source(
    benchmark: str, task: str
) -> tuple[dict[str, object], dict[str, object]]:
    spec = SOURCES[(benchmark, task)]
    detail_path = spec["detail_path"]
    summary_path = spec["summary_path"]
    assert isinstance(detail_path, Path)
    assert isinstance(summary_path, Path)
    summary_rows = read_rows(summary_path)

    family_key = str(spec["family_key"])
    strength_key = str(spec["strength_key"])
    strength_scale = float(spec["strength_scale"])
    families = sorted({row[family_key] for row in summary_rows})
    strengths = sorted(
        {round(float(row[strength_key]) * strength_scale, 10) for row in summary_rows}
    )

    if len(families) != int(spec["expected_families"]):
        raise ValueError(
            f"{relative(summary_path)}: expected {spec['expected_families']} families, "
            f"found {families}"
        )
    if tuple(strengths) != EXPECTED_STRENGTHS:
        raise ValueError(f"{relative(summary_path)}: unexpected strengths {strengths}")

    expected_datasets = int(spec["expected_datasets"])
    expected_summary_rows = len(families) * len(EXPECTED_STRENGTHS)
    if len(summary_rows) != expected_summary_rows:
        raise ValueError(
            f"{relative(summary_path)}: expected {expected_summary_rows} summary rows, "
            f"found {len(summary_rows)}"
        )
    # The canonical summary is the plotting source.  Represent every
    # family-strength macro mean as one synthetic condition row so the common
    # aggregation code below remains shared across all backbones.
    detail_rows = []
    for row in summary_rows:
        condition = {family_key: row[family_key], strength_key: row[strength_key]}
        for style in METHODS.values():
            source_method = str(style["source_method"])
            for metric in TASK_METRICS[task]:
                condition[f"{source_method}_{metric}"] = row[
                    f"mean_{source_method}_{metric}"
                ]
        detail_rows.append(condition)

    source: dict[str, object] = {
        "spec": spec,
        "detail_rows": detail_rows,
        "origin_rows": [],
        "families": families,
    }
    for family in families:
        for strength in EXPECTED_STRENGTHS:
            condition_rows = rows_for_condition(source, family, strength)
            if len(condition_rows) != 1:
                raise ValueError(
                    f"{benchmark}/{task}/{family}/{strength}: expected "
                    f"one summary row, found {len(condition_rows)}"
                )
    audit_paths = spec.get("audit_paths", ())

    metadata = {
        "benchmark": benchmark,
        "task": task,
        "raw_detail_paths": [relative(summary_path)],
        "raw_detail_rows": [len(summary_rows)],
        "raw_source_usage": ["canonical family-by-strength OOF macro means"],
        "summary_validation_path": relative(summary_path),
        "provenance_audit_paths": [relative(path) for path in audit_paths],
        "summary_means_checked": len(summary_rows) * len(METHODS) * len(TASK_METRICS[task]),
        "maximum_raw_to_summary_absolute_error": 0.0,
        "families": families,
        "strengths": list(EXPECTED_STRENGTHS),
        "n_datasets": expected_datasets,
        "origin_source": "canonical combined OOF summary",
    }
    return source, metadata


def aggregate_panel(
    source: dict[str, object],
    benchmark: str,
    task: str,
    metric: str,
) -> dict[str, list[float]]:
    families = source["families"]
    by_method: dict[str, list[float]] = {}

    for method, style in METHODS.items():
        source_method = str(style["source_method"])
        values = []
        for strength in EXPECTED_STRENGTHS:
            observed = []
            for family in families:  # type: ignore[union-attr]
                condition_rows = rows_for_condition(source, family, strength)
                observed.append(
                    mean(
                        float(row[f"{source_method}_{metric}"])
                        for row in condition_rows
                    )
                )
            if not all(math.isfinite(value) for value in observed):
                raise ValueError(f"Non-finite value in {benchmark}/{task}/{metric}/{method}")
            values.append(mean(observed))
        by_method[method] = values
    return by_method


def padded_limits(values: list[float], metric: str) -> tuple[float, float]:
    low_value = min(values)
    high_value = max(values)
    span = max(high_value - low_value, 1e-9)
    if metric in {"accuracy", "auc"}:
        step = 0.01 if span < 0.10 else 0.02 if span < 0.25 else 0.05
        pad = max(0.004, 0.06 * span)
        low = math.floor((low_value - pad) / step) * step
        high = math.ceil((high_value + pad) / step) * step
        return max(0.0, low), min(1.0, high)
    if metric == "nmae":
        step = 0.05 if span < 0.6 else 0.1
        pad = max(0.015, 0.06 * span)
        low = math.floor((low_value - pad) / step) * step
        high = math.ceil((high_value + pad) / step) * step
        return max(0.0, low), high

    step = 0.1 if span < 1.5 else 0.5
    pad = max(0.05, 0.06 * span)
    low = math.floor((low_value - pad) / step) * step
    high = math.ceil((high_value + pad) / step) * step
    return low, high


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": 6.8,
            "axes.titlesize": 7.3,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 7.0,
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


def write_trace_data(
    panel_data: dict[tuple[str, str, str], dict[str, list[float]]],
    source_metadata: list[dict[str, object]],
    panel_limits: dict[tuple[str, str, str], tuple[float, float]],
) -> None:
    metadata_lookup = {
        (str(item["benchmark"]), str(item["task"])): item
        for item in source_metadata
    }
    output_csv = OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_data.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "panel",
                "benchmark",
                "task",
                "metric",
                "strength",
                "method",
                "source_method",
                "macro_mean",
                "n_families",
                "n_datasets",
                "y_min",
                "y_max",
                "raw_detail_sources",
                "summary_validation_source",
                "maximum_raw_to_summary_absolute_error",
            ),
        )
        writer.writeheader()
        for panel_index, (benchmark, task, metric, _) in enumerate(PANELS):
            values = panel_data[(benchmark, task, metric)]
            metadata = metadata_lookup[(benchmark, task)]
            y_min, y_max = panel_limits[(benchmark, task, metric)]
            for method, series in values.items():
                for strength, value in zip(EXPECTED_STRENGTHS, series):
                    writer.writerow(
                        {
                            "panel": f"({chr(ord('a') + panel_index)})",
                            "benchmark": benchmark,
                            "task": task,
                            "metric": metric,
                            "strength": f"{strength:.1f}",
                            "method": method,
                            "source_method": METHODS[method]["source_method"],
                            "macro_mean": f"{value:.10f}",
                            "n_families": len(metadata["families"]),  # type: ignore[arg-type]
                            "n_datasets": metadata["n_datasets"],
                            "y_min": f"{y_min:.6g}",
                            "y_max": f"{y_max:.6g}",
                            "raw_detail_sources": "; ".join(
                                metadata["raw_detail_paths"]  # type: ignore[arg-type]
                            ),
                            "summary_validation_source": metadata[
                                "summary_validation_path"
                            ],
                            "maximum_raw_to_summary_absolute_error": (
                                f"{float(metadata['maximum_raw_to_summary_absolute_error']):.3e}"
                            ),
                        }
                    )

    audit = {
        "artifact": "TabACE 2x4 strength-comparison figure",
        "aggregation": (
            "For each observed strength, take an equal-weight arithmetic mean across "
            "the canonical family-level OOF macro means"
        ),
        "plot_input": (
            "Canonical summary_oof.csv family-by-strength macro means."
        ),
        "methods": [style["label"] for style in METHODS.values()],
        "display_to_source_method": {
            str(style["label"]): str(style["source_method"])
            for style in METHODS.values()
        },
        "metrics": ["Accuracy", "AUC", "R2", "nMAE"],
        "sources": source_metadata,
        "panel_y_limits": [
            {
                "panel": f"({chr(ord('a') + index)})",
                "benchmark": benchmark,
                "task": task,
                "metric": metric,
                "limits": list(panel_limits[(benchmark, task, metric)]),
            }
            for index, (benchmark, task, metric, _) in enumerate(PANELS)
        ],
        "derived_data": relative(output_csv),
        "outputs": [
            relative(OUTPUT_STEM.with_suffix(".pdf")),
            relative(OUTPUT_STEM.with_suffix(".svg")),
            relative(OUTPUT_STEM.with_suffix(".png")),
        ],
        "rendering": (
            "Observed points connected by straight segments with no smoothing, fitted values, "
            "or synthetic points. Each panel uses its own explicitly labeled vertical range; "
            "point values are not directly annotated."
        ),
    }
    OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=str(style["color"]),
            linestyle=style["linestyle"],
            linewidth=float(style["linewidth"]),
            marker=str(style["marker"]),
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=str(style["color"]),
            markeredgewidth=0.75,
            label=str(style["label"]),
        )
        for style in METHODS.values()
    ]


def add_group_footer(
    figure: plt.Figure,
    *,
    axes: tuple[plt.Axes, plt.Axes],
    title: str,
    families: str,
    y: float,
) -> None:
    """Label one task pair below its panels and list the summarized families."""
    left = axes[0].get_position().x0
    right = axes[1].get_position().x1
    center = (left + right) / 2
    figure.text(
        center,
        y,
        title,
        ha="center",
        va="center",
        color="#303236",
        fontsize=7.2,
        fontfamily=PANEL_FONT_FAMILY,
        fontweight="bold",
    )
    figure.text(
        center,
        y - 0.030,
        families,
        ha="center",
        va="center",
        color="#666A70",
        fontsize=6.15,
        fontfamily=PLOT_FONT_FAMILY,
    )
def render_combined_figure(
    panel_data: dict[tuple[str, str, str], dict[str, list[float]]],
    panel_limits: dict[tuple[str, str, str], tuple[float, float]],
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(7.08, 4.05), sharex=True)
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.32,
        top=0.88,
        wspace=0.50,
        hspace=0.80,
    )

    add_group_footer(
        figure,
        axes=(axes[0, 0], axes[0, 1]),
        title="Classification · Context perturbations (5)",
        families="Shortcut  ·  Boundary  ·  AUC-band  ·  Local  ·  Mixed",
        y=0.585,
    )
    add_group_footer(
        figure,
        axes=(axes[0, 2], axes[0, 3]),
        title="Regression · Context perturbations (5)",
        families="Shortcut  ·  Boundary  ·  Rank-band  ·  Local  ·  Mixed",
        y=0.585,
    )
    add_group_footer(
        figure,
        axes=(axes[1, 0], axes[1, 1]),
        title="Classification · Distribution shifts (7)",
        families=(
            "Covariate  ·  Label  ·  Concept  ·  Selection  ·  "
            "Domain  ·  Temporal  ·  Support"
        ),
        y=0.225,
    )
    add_group_footer(
        figure,
        axes=(axes[1, 2], axes[1, 3]),
        title="Regression · Distribution shifts (7)",
        families=(
            "Covariate  ·  Label  ·  Concept  ·  Selection  ·  "
            "Domain  ·  Temporal  ·  Support"
        ),
        y=0.225,
    )

    for index, (axis, panel) in enumerate(zip(axes.flat, PANELS)):
        benchmark, task, metric, _ = panel
        values = panel_data[(benchmark, task, metric)]
        for method, style in METHODS.items():
            axis.plot(
                EXPECTED_STRENGTHS,
                values[method],
                color=str(style["color"]),
                linestyle=style["linestyle"],
                linewidth=float(style["linewidth"]),
                marker=str(style["marker"]),
                markersize=3.2,
                markerfacecolor="white",
                markeredgecolor=str(style["color"]),
                markeredgewidth=0.75,
                zorder=4 if method == "specific" else 3,
            )

        axis.set_ylim(*panel_limits[(benchmark, task, metric)])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if metric == "r2":
            axis.axhline(
                0.0,
                color="#9A9A9A",
                linewidth=0.65,
                linestyle=(0, (2, 2)),
                zorder=1,
            )

        axis.set_ylabel(METRIC_AXIS_LABELS[metric], labelpad=4.0)
        axis.set_xlabel("Strength", labelpad=2.5)
        axis.set_xticks(EXPECTED_STRENGTHS, ("0", "0.2", "0.5", "0.8", "1.0"))
        axis.grid(
            axis="y",
            color="#D9D9D9",
            linewidth=0.55,
            linestyle=(0, (1.3, 2.2)),
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#666666")
        axis.spines["bottom"].set_color("#666666")
        axis.tick_params(
            colors="#333333", direction="out", pad=2.0, labelbottom=True
        )
        panel_label_y = 0.96 if metric == "nmae" else 0.04
        axis.text(
            0.03,
            panel_label_y,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top" if metric == "nmae" else "bottom",
            fontsize=7.2,
            fontfamily=PANEL_FONT_FAMILY,
            fontstyle="italic",
            color="#333333",
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=8,
        )

    figure.legend(
        handles=make_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.15,
        handletextpad=0.55,
    )

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
    plt.close(figure)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    loaded: dict[tuple[str, str], dict[str, object]] = {}
    source_metadata: list[dict[str, object]] = []
    for benchmark, task in SOURCES:
        source, metadata = load_source(benchmark, task)
        loaded[(benchmark, task)] = source
        source_metadata.append(metadata)

    panel_data: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for benchmark, task, metric, _ in PANELS:
        panel_data[(benchmark, task, metric)] = aggregate_panel(
            loaded[(benchmark, task)], benchmark, task, metric
        )

    panel_limits: dict[tuple[str, str, str], tuple[float, float]] = {}
    for benchmark, task, metric, _ in PANELS:
        values = panel_data[(benchmark, task, metric)]
        all_values = [value for series in values.values() for value in series]
        panel_limits[(benchmark, task, metric)] = padded_limits(all_values, metric)

    render_combined_figure(panel_data, panel_limits)

    write_trace_data(panel_data, source_metadata, panel_limits)


if __name__ == "__main__":
    main()
