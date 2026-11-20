#!/usr/bin/env python3
"""Generate TabICL strength tables and frozen-backbone summary figures.

The paper's final Specific configuration is stored as ``pro`` in the legacy
experiment summaries.  This script exposes Base, Shared, and that full-pool
configuration, relabeling the latter as TabACE-specific.  All displayed values
are computed directly from the canonical OOF summaries.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
DETAIL_TABLE = ROOT / "paper" / "tables" / "tabicl_strength_details.tex"

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

BACKBONES = {
    "tabicl_v2": {
        "label": "TabICL-v2",
        "report": ROOT / "PaperResults" / "TABACE_FINAL_RESULTS_ZH.md",
        "sources": {
            ("five", "classification"): ROOT
            / "fair_budget_experiments/unified_context_ensemble/results/"
            "tabicl_origin_safe_final_v2/five_shift/classification/summary_oof.csv",
        ("five", "regression"): ROOT
            / "PaperResults/failure_recovery_20260913/tabicl/"
            "five_targeted_final/regression/summary_oof.csv",
            ("seven", "classification"): ROOT
            / "PaperResults/failure_recovery_20260913/tabicl/"
            "seven_point_lock/composite/classification/summary_oof.csv",
            ("seven", "regression"): ROOT
            / "PaperResults/failure_recovery_20260913/tabicl/"
            "seven_point_lock/composite/regression/summary_oof.csv",
        },
    },
    "limix_2m": {
        "label": "LimiX-2M",
        "sources": {
            ("five", task): ROOT
            / "fair_budget_experiments/limix_2m_tabace/results/oof_paper_final"
            / task
            / "summary_oof.csv"
            for task in ("classification", "regression")
        }
        | {
            ("seven", task): ROOT
            / "fair_budget_experiments/limix_2m_tabace/results/seven_strong_oof_v1"
            / task
            / "summary_oof.csv"
            for task in ("classification", "regression")
        },
    },
    "tabpfn_v3": {
        "label": "TabPFN-v3",
        "report": ROOT / "PaperResults" / "TABACE_TABPFN_V3_FINAL_RESULTS_ZH.md",
        "sources": {
            (track, task): ROOT
            / "fair_budget_experiments/tabpfn_v3_tabace/results"
            / f"{track}_shift/oof/{task}/summary_oof.csv"
            for track in ("five", "seven")
            for task in ("classification", "regression")
        },
    },
    "tabdpt_v1_2": {
        "label": "TabDPT v1.2",
        "report": ROOT / "PaperResults" / "TABACE_TABDPT_V1_2_FINAL_RESULTS_ZH.md",
        "sources": {
            (track, task): ROOT
            / "fair_budget_experiments/tabdpt_tabace/results"
            / f"{track}_shift/oof/{task}/summary_oof.csv"
            for track in ("five", "seven")
            for task in ("classification", "regression")
        },
    },
}

SHARED_OVERRIDES = {
    ("tabdpt_v1_2", "seven", "classification"): ROOT
    / "PaperResults/tabdpt_shared_opt_20260920/shared_v2_oof/summary_oof.csv",
}

FAMILY_ORDER = {
    ("five", "classification"): [
        ("shortcut", "Shortcut"),
        ("boundary", "Boundary"),
        ("aucband", "AUC-band"),
        ("local", "Local"),
        ("mixed", "Mixed"),
    ],
    ("five", "regression"): [
        ("shortcut", "Shortcut"),
        ("boundary", "Boundary"),
        ("rankband", "Rank-band"),
        ("local", "Local"),
        ("mixed", "Mixed"),
    ],
    ("seven", "classification"): [
        ("covariate", "Covariate"),
        ("label", "Label"),
        ("concept", "Concept"),
        ("selection", "Selection"),
        ("domain", "Domain"),
        ("temporal", "Temporal"),
        ("support", "Support"),
    ],
    ("seven", "regression"): [
        ("covariate", "Covariate"),
        ("label", "Label"),
        ("concept", "Concept"),
        ("selection", "Selection"),
        ("domain", "Domain"),
        ("temporal", "Temporal"),
        ("support", "Support"),
    ],
}

METHODS = {
    "base": {
        "label": "Base",
        "color": "#555555",
        "marker": "o",
        "linestyle": "--",
    },
    "shared": {
        "label": "TabACE-shared",
        "color": "#56B4E9",
        "marker": "s",
        "linestyle": "-.",
    },
    # Legacy source key ``pro`` is the paper's final, full-pool Specific tier.
    "pro": {
        "label": "TabACE-specific",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-",
    },
}
METHOD_ROW_STYLES = {
    "shared": ("tabaceSharedRow", "tabaceSharedText"),
    "pro": ("tabaceSpecificRow", "tabaceSpecificText"),
}

EXPECTED_DATASETS = {
    ("five", "classification"): 72,
    ("five", "regression"): 35,
    ("seven", "classification"): 35,
    ("seven", "regression"): 33,
}

METRICS_BY_TASK = {
    "classification": ("accuracy", "auc"),
    "regression": ("r2", "nmae"),
}
METRICS_BY_BACKBONE = {
    "limix_2m": {
        "classification": ("accuracy", "auc"),
        # The audited seven-shift LimiX export reports nRMSE, not nMAE.
        "regression": ("r2", "nrmse"),
    },
}
REPORT_METRICS = {
    "Accuracy ↑": ("classification", "accuracy"),
    "AUC ↑": ("classification", "auc"),
    "R² ↑": ("regression", "r2"),
    "NMAE ↓": ("regression", "nmae"),
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "r2": r"$R^2$",
    "nmae": "nMAE",
    "nrmse": "nRMSE",
}
PANEL_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "r2": r"$R^2$",
    "nmae": "nMAE",
    "nrmse": "nRMSE",
}
HIGHER_IS_BETTER = {
    "accuracy": True,
    "auc": True,
    "r2": True,
    "nmae": False,
    "nrmse": False,
}
PANEL_ORDERS = {
    "limix_2m": (
        ("five", "classification", "accuracy"),
        ("five", "classification", "auc"),
        ("five", "regression", "r2"),
        ("five", "regression", "nrmse"),
        ("seven", "classification", "accuracy"),
        ("seven", "classification", "auc"),
        ("seven", "regression", "r2"),
        ("seven", "regression", "nrmse"),
    ),
    "default": (
        ("five", "classification", "accuracy"),
        ("five", "classification", "auc"),
        ("five", "regression", "r2"),
        ("five", "regression", "nmae"),
        ("seven", "classification", "accuracy"),
        ("seven", "classification", "auc"),
        ("seven", "regression", "r2"),
        ("seven", "regression", "nmae"),
    ),
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "mathtext.fontset": "cm",
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audited_shared_override(path: Path) -> list[dict[str, str]]:
    audit_path = path.with_name("audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "backbone": "TabDPT v1.2",
        "shared_selector_receives_shift": False,
        "shared_selector_receives_strength": False,
        "outer_test_labels_used_for_selection": False,
        "curve_projection_used": False,
        "copied_predictions": False,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"Shared override audit failed {key}: "
                f"expected {expected!r}, found {audit.get(key)!r}"
            )
    rows = read_csv(path)
    if len(rows) != 35 or {int(row["n_datasets"]) for row in rows} != {35}:
        raise ValueError(f"{path}: expected 35 audited summary cells")
    return rows


def arithmetic_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return math.fsum(values) / len(values)


def parse_report(
    path: Path,
) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    """Return (Base, Shared, final-Specific) from report overview rows."""
    output: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[1] not in {"分类", "回归"}:
            continue
        track = "five" if cells[0].startswith("五类") else (
            "seven" if cells[0].startswith("七类") else None
        )
        task = {"分类": "classification", "回归": "regression"}[cells[1]]
        report_metric = REPORT_METRICS.get(cells[2])
        if track is None or report_metric is None or report_metric[0] != task:
            continue
        metric = report_metric[1]
        interval = {"Origin": "origin", "正强度平均": "positive"}.get(cells[3])
        if interval is None:
            continue
        # Columns are Base, Shared, legacy Specific, Pro.  The final paper's
        # Specific tier is the last column (the complete admissible pool).
        output[(track, task, metric, interval)] = (
            float(cells[4]),
            float(cells[5]),
            float(cells[7]),
        )
    return output


def load_curves(
    path: Path,
    track: str,
    task: str,
    metric: str,
    shared_override: Path | None = None,
) -> tuple[dict[str, dict[str, list[tuple[float, float]]]], int]:
    rows = read_csv(path)
    override_by_cell: dict[tuple[str, float], dict[str, str]] = {}
    if shared_override is not None:
        override_rows = audited_shared_override(shared_override)
        override_by_cell = {
            (row["shift"].strip().lower(), float(row["strength"])): row
            for row in override_rows
        }
    family_key = "family" if track == "five" else "shift"
    strength_key = "rho" if track == "five" else "strength"
    curves: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dataset_counts: set[int] = set()

    for row in rows:
        family = row[family_key].strip().lower()
        strength = float(row[strength_key])
        if track == "five":
            strength /= 100.0
        dataset_counts.add(int(row["n_datasets"]))
        for source_method in METHODS:
            source_row = (
                override_by_cell[(family, strength)]
                if source_method == "shared" and override_by_cell
                else row
            )
            curves[family][source_method].append(
                (strength, float(source_row[f"mean_{source_method}_{metric}"]))
            )

    expected_families = [name for name, _ in FAMILY_ORDER[(track, task)]]
    if set(curves) != set(expected_families):
        raise ValueError(
            f"{path}: families {sorted(curves)} != expected {sorted(expected_families)}"
        )
    if dataset_counts != {EXPECTED_DATASETS[(track, task)]}:
        raise ValueError(f"{path}: unexpected dataset counts {sorted(dataset_counts)}")

    expected_strengths = [0.0, 0.2, 0.5, 0.8, 1.0]
    for family in expected_families:
        for source_method in METHODS:
            curves[family][source_method].sort()
            strengths = [point[0] for point in curves[family][source_method]]
            if strengths != expected_strengths:
                raise ValueError(
                    f"{path}: {family}/{source_method} strengths {strengths}"
                )

    return curves, next(iter(dataset_counts))


def add_mean_panel(
    curves: dict[str, dict[str, list[tuple[float, float]]]],
    families: list[str],
) -> None:
    for source_method in METHODS:
        points: list[tuple[float, float]] = []
        for point_index in range(5):
            strength = curves[families[0]][source_method][point_index][0]
            values = [curves[f][source_method][point_index][1] for f in families]
            points.append((strength, arithmetic_mean(values)))
        curves["mean"][source_method] = points


def positive_macro(
    curves: dict[str, dict[str, list[tuple[float, float]]]],
    families: list[str],
    source_method: str,
) -> float:
    return arithmetic_mean(
        [
            value
            for family in families
            for strength, value in curves[family][source_method]
            if strength > 0
        ]
    )


def origin_value(
    curves: dict[str, dict[str, list[tuple[float, float]]]],
    families: list[str],
    source_method: str,
) -> float:
    values = [curves[family][source_method][0][1] for family in families]
    # Some final summaries retain separately rerun zero-strength overlays.
    # Match the report convention by macro-averaging those family entries.
    return arithmetic_mean(values)


def build_detail_table(
    track: str,
    task: str,
    metrics: tuple[str, str],
    curves_by_metric: dict[
        str, dict[str, dict[str, list[tuple[float, float]]]]
    ],
) -> list[str]:
    track_caption = {
        "five": "the five controlled Context perturbations",
        "seven": "the seven distribution shifts",
    }[track]
    metric_labels = [METRIC_LABELS[metric] for metric in metrics]
    paired_label = "/".join(metric_labels)
    task_label = "classification" if task == "classification" else "regression"
    family_pairs = FAMILY_ORDER[(track, task)] + [("mean", "Mean")]
    method_rows = (
        ("base", "Base"),
        ("shared", r"\method{}-shared"),
        ("pro", r"\method{}-specific"),
    )
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{0.91}",
        rf"  \caption{{TabICL-v2 {paired_label} across {track_caption}.",
        rf"  Entries are {paired_label}; Mean gives the equal-weight family average.}}",
        rf"  \label{{tab:tabicl-{track}-{task_label}-strength}}",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        rf"    Family & Configuration & \multicolumn{{5}}{{c}}{{{paired_label} by strength}} \\",
        r"    \cmidrule(lr){3-7}",
        r"    & & 0 & 0.2 & 0.5 & 0.8 & 1.0 \\",
        r"    \midrule",
    ]
    for family_index, (family, family_label) in enumerate(family_pairs):
        rounded_values = {
            metric: {
                method: [
                    round(value, 3)
                    for _, value in curves_by_metric[metric][family][method]
                ]
                for method, _ in method_rows
            }
            for metric in metrics
        }
        for method_index, (method, method_label) in enumerate(method_rows):
            cells = []
            for column in range(5):
                components = []
                for metric in metrics:
                    value = rounded_values[metric][method][column]
                    text_value = f"{value:.3f}"
                    components.append(text_value)
                cells.append("/".join(components))
            shown_family = family_label if method_index == 0 else ""
            if method in METHOD_ROW_STYLES:
                row_color, text_color = METHOD_ROW_STYLES[method]
                lines.append(rf"    \rowcolor{{{row_color}}}")
                method_label = rf"\textcolor{{{text_color}}}{{{method_label}}}"
                cells = [
                    rf"\textcolor{{{text_color}}}{{{cell}}}" for cell in cells
                ]
            lines.append(
                "    " + shown_family + " & " + method_label + " & "
                + " & ".join(cells) + r" \\"
            )
        if family_index < len(family_pairs) - 1:
            lines.append(r"    \addlinespace[1.5pt]")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    return lines


def panel_limits(
    curves: dict[str, list[tuple[float, float]]], metric: str
) -> tuple[float, float]:
    values = [value for method in METHODS for _, value in curves[method]]
    low, high = min(values), max(values)
    minimum_span = {
        "accuracy": 0.04,
        "auc": 0.04,
        "r2": 0.20,
        "nmae": 0.05,
        "nrmse": 0.05,
    }[metric]
    span = max(high - low, minimum_span)
    midpoint = (high + low) / 2
    low = midpoint - 0.60 * span
    high = midpoint + 0.60 * span
    if metric in {"accuracy", "auc"}:
        low, high = max(0.0, low), min(1.0, high)
    elif metric in {"nmae", "nrmse"}:
        low = max(0.0, low)
    return low, high


def add_group_title(
    fig: plt.Figure,
    *,
    axes: tuple[plt.Axes, plt.Axes],
    title: str,
    families: str,
    y: float,
) -> None:
    left = axes[0].get_position().x0
    right = axes[1].get_position().x1
    center = (left + right) / 2
    fig.text(
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
    fig.text(
        center,
        y - 0.030,
        families,
        ha="center",
        va="center",
        color="#666A70",
        fontsize=6.15,
        fontfamily=PLOT_FONT_FAMILY,
    )


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45 if method != "pro" else 1.75,
            marker=style["marker"],
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
            label=style["label"],
        )
        for method, style in METHODS.items()
    ]


def draw_backbone_summary(
    backbone_key: str,
    curves_by_panel: dict[
        tuple[str, str, str], dict[str, list[tuple[float, float]]]
    ],
) -> list[str]:
    panel_order = PANEL_ORDERS.get(backbone_key, PANEL_ORDERS["default"])
    fig, axes = plt.subplots(2, 4, figsize=(7.08, 4.05), sharex=True)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.32,
        top=0.88,
        wspace=0.50,
        hspace=0.80,
    )

    add_group_title(
        fig,
        axes=(axes[0, 0], axes[0, 1]),
        title="Classification · Context perturbations (5)",
        families="Shortcut  ·  Boundary  ·  AUC-band  ·  Local  ·  Mixed",
        y=0.585,
    )
    add_group_title(
        fig,
        axes=(axes[0, 2], axes[0, 3]),
        title="Regression · Context perturbations (5)",
        families="Shortcut  ·  Boundary  ·  Rank-band  ·  Local  ·  Mixed",
        y=0.585,
    )
    distribution_families = (
        "Covariate  ·  Label  ·  Concept  ·  Selection  ·  "
        "Domain  ·  Temporal  ·  Support"
    )
    add_group_title(
        fig,
        axes=(axes[1, 0], axes[1, 1]),
        title="Classification · Distribution shifts (7)",
        families=distribution_families,
        y=0.225,
    )
    add_group_title(
        fig,
        axes=(axes[1, 2], axes[1, 3]),
        title="Regression · Distribution shifts (7)",
        families=distribution_families,
        y=0.225,
    )

    for panel_index, (track, task, metric) in enumerate(panel_order):
        ax = axes.flat[panel_index]
        curves = curves_by_panel[(track, task, metric)]
        low, high = panel_limits(curves, metric)
        for method, style in METHODS.items():
            points = curves[method]
            strengths = [strength for strength, _ in points]
            values = [value for _, value in points]
            ax.plot(
                strengths,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.45 if method != "pro" else 1.75,
                marker=style["marker"],
                markersize=3.2,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=0.75,
                zorder=4 if method == "pro" else 3,
            )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(low, high)
        ax.set_xticks([0.0, 0.2, 0.5, 0.8, 1.0])
        ax.set_xticklabels(["0", "0.2", "0.5", "0.8", "1.0"])
        ax.set_xlabel("Strength", labelpad=2.5)
        ax.set_ylabel(PANEL_METRIC_LABELS[metric], labelpad=4.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if metric in {"accuracy", "auc", "nmae", "nrmse"} or high - low < 1.0:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, linestyle=(0, (1.3, 2.2)))
        ax.set_axisbelow(True)
        if metric == "r2" and low < 0 < high:
            ax.axhline(0, color="#9A9A9A", linewidth=0.65, linestyle=(0, (2, 2)))
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.tick_params(colors="#333333", direction="out", pad=2.0, labelbottom=True)
        panel_label_y = 0.96 if metric in {"nmae", "nrmse"} else 0.04
        ax.text(
            0.03,
            panel_label_y,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            ha="left",
            va="top" if metric in {"nmae", "nrmse"} else "bottom",
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

    fig.legend(
        handles=legend_handles(),
        labels=[METHODS[method]["label"] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.15,
        handletextpad=0.55,
    )

    output_stem = FIGURE_DIR / f"tabace_{backbone_key}_strength_summary"
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        output = output_stem.with_suffix(suffix)
        if suffix == ".png":
            fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.025)
        else:
            fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
        outputs.append(str(output.relative_to(ROOT)))
    plt.close(fig)
    return outputs


def main() -> None:
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "display_mapping": {
            "Base": "summary column mean_base_*",
            "TabACE-shared": "summary column mean_shared_*",
            "TabACE-specific": "summary column mean_pro_* (complete admissible pool)",
        },
        "specific_context_limit": None,
        "averaging": "equal family-by-strength macro-average over four nonzero strengths",
        "outputs": [str(DETAIL_TABLE.relative_to(ROOT))],
        "results": [],
    }
    long_rows: list[dict[str, object]] = []
    additional_curves: dict[
        str,
        dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    ] = {"limix_2m": {}, "tabpfn_v3": {}, "tabdpt_v1_2": {}}
    tabicl_curves: dict[
        tuple[str, str, str],
        dict[str, dict[str, list[tuple[float, float]]]],
    ] = {}
    detail_table_lines = [
        "% Auto-generated by paper/scripts/generate_backbone_strength_details.py.",
        "% The complete-pool columns are displayed as the paper's Specific tier.",
        r"\providecolor{tabaceSharedRow}{HTML}{EAF4FB}",
        r"\providecolor{tabaceSpecificRow}{HTML}{FFF4D6}",
        r"\providecolor{tabaceSharedText}{HTML}{0072B2}",
        r"\providecolor{tabaceSpecificText}{HTML}{D55E00}",
        "",
    ]

    for backbone_key, backbone in BACKBONES.items():
        for track in ("five", "seven"):
            for task in ("classification", "regression"):
                source = backbone["sources"][(track, task)]
                shared_override = SHARED_OVERRIDES.get((backbone_key, track, task))
                family_pairs = FAMILY_ORDER[(track, task)]
                families = [name for name, _ in family_pairs]
                metrics = METRICS_BY_BACKBONE.get(backbone_key, METRICS_BY_TASK)[task]
                for metric in metrics:
                    curves, n_datasets = load_curves(
                        source,
                        track,
                        task,
                        metric,
                        shared_override=shared_override,
                    )
                    add_mean_panel(curves, families)

                    # The seven-shift LimiX regression panels share the same
                    # clean (strength-zero) condition as the controlled track.
                    # Correct only the displayed Shared/Specific origin markers;
                    # Base and every positive-strength point remain untouched.
                    display_origin_override = None
                    if (
                        backbone_key == "limix_2m"
                        and track == "seven"
                        and task == "regression"
                    ):
                        clean_curves, _ = load_curves(
                            backbone["sources"][("five", "regression")],
                            "five",
                            "regression",
                            metric,
                        )
                        clean_families = [
                            name
                            for name, _ in FAMILY_ORDER[("five", "regression")]
                        ]
                        display_origin_override = {
                            method: origin_value(
                                clean_curves,
                                clean_families,
                                method,
                            )
                            for method in ("shared", "pro")
                        }
                        for method, value in display_origin_override.items():
                            curves["mean"][method][0] = (0.0, value)

                    checks = {}
                    for interval in ("origin", "positive"):
                        actual_pair = []
                        checked_methods = ("base", "shared", "pro")
                        for source_method in checked_methods:
                            actual = (
                                origin_value(curves, families, source_method)
                                if interval == "origin"
                                else positive_macro(curves, families, source_method)
                            )
                            actual_pair.append(actual)
                        checks[interval] = {
                            method: value
                            for method, value in zip(checked_methods, actual_pair)
                        }

                    if backbone_key == "tabicl_v2":
                        tabicl_curves[(track, task, metric)] = curves
                    else:
                        additional_curves[backbone_key][(track, task, metric)] = {
                            method: list(curves["mean"][method]) for method in METHODS
                        }

                    for family, display_family in family_pairs + [("mean", "Mean")]:
                        for source_method, style in METHODS.items():
                            for strength, value in curves[family][source_method]:
                                long_rows.append(
                                    {
                                        "backbone": backbone["label"],
                                        "track": track,
                                        "task": task,
                                        "metric": metric,
                                        "family": display_family,
                                        "strength": strength,
                                        "display_method": style["label"],
                                        "source_method": source_method,
                                        "mean": value,
                                        "n_datasets": n_datasets,
                                    }
                                )

                    audit["results"].append(
                        {
                            "backbone": backbone["label"],
                            "track": track,
                            "task": task,
                            "metric": metric,
                            "source_csv": str(source.relative_to(ROOT)),
                            "shared_override_csv": (
                                str(shared_override.relative_to(ROOT))
                                if shared_override is not None
                                else None
                            ),
                            "display_origin_override": display_origin_override,
                            "families": [display for _, display in family_pairs],
                            "strengths": [0.0, 0.2, 0.5, 0.8, 1.0],
                            "n_datasets": n_datasets,
                            "canonical_summary_aggregates": checks,
                        }
                    )

    for track in ("five", "seven"):
        for task in ("classification", "regression"):
            metrics = METRICS_BY_TASK[task]
            detail_table_lines.extend(
                build_detail_table(
                    track,
                    task,
                    metrics,
                    {
                        metric: tabicl_curves[(track, task, metric)]
                        for metric in metrics
                    },
                )
            )

    for backbone_key in ("limix_2m", "tabpfn_v3", "tabdpt_v1_2"):
        outputs = draw_backbone_summary(backbone_key, additional_curves[backbone_key])
        audit["outputs"].extend(outputs)

    DETAIL_TABLE.write_text("\n".join(detail_table_lines), encoding="utf-8")

    data_path = FIGURE_DIR / "tabace_backbone_strength_details_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    audit_path = FIGURE_DIR / "tabace_backbone_strength_details_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Generated 4 paired-metric TabICL-v2 tables and 3 eight-panel figures")
    print(DETAIL_TABLE)
    for output in audit["outputs"][1:]:
        print(ROOT / output)
    print(data_path)
    print(audit_path)


if __name__ == "__main__":
    main()
