#!/usr/bin/env python3
"""Generate the paper-wide overall-results table from canonical OOF summaries."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "tables" / "overall_results.tex"

BASELINE_SUMMARY = (
    ROOT
    / "fair_budget_experiments"
    / "baseline_shift_comparison"
    / "outputs"
    / "summary_long.csv"
)

LIMIX_TABACE_SUMMARIES = {
    ("five", task): ROOT
    / "fair_budget_experiments/limix_2m_tabace/results/oof_paper_final"
    / task
    / "summary_oof.csv"
    for task in ("classification", "regression")
}
LIMIX_TABACE_SUMMARIES.update(
    {
        ("seven", task): ROOT
        / "fair_budget_experiments/limix_2m_tabace/results/seven_strong_oof_v1"
        / task
        / "summary_oof.csv"
        for task in ("classification", "regression")
    }
)

BACKBONE_SUMMARIES = {
    "TabICL-v2": {
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
    "TabPFN-v3": {
        (track, task): ROOT
        / "fair_budget_experiments/tabpfn_v3_tabace/results"
        / f"{track}_shift/oof/{task}/summary_oof.csv"
        for track in ("five", "seven")
        for task in ("classification", "regression")
    },
    "TabDPT-v1.2": {
        (track, task): ROOT
        / "fair_budget_experiments/tabdpt_tabace/results"
        / f"{track}_shift/oof/{task}/summary_oof.csv"
        for track in ("five", "seven")
        for task in ("classification", "regression")
    },
}

TABDPT_SHARED_V2_SUMMARY = (
    ROOT
    / "PaperResults/tabdpt_shared_opt_20260920/shared_v2_oof/summary_oof.csv"
)
TABDPT_SHARED_V2_AUDIT = TABDPT_SHARED_V2_SUMMARY.with_name("audit.json")

BASELINE_ORDER = [
    ("linear", "LogReg/Ridge + HPO"),
    ("svm", "SVM/SVR + HPO"),
    ("knn", "kNN + HPO"),
    ("random_forest", "RF + HPO"),
    ("lightgbm", "LightGBM + HPO"),
    ("catboost", "CatBoost + HPO"),
    ("xgboost", "XGBoost + HPO"),
    ("mlp", "MLP + HPO"),
    ("ft_transformer", "FT-Transformer"),
    ("tabm", "TabM"),
    ("tabularnn", "TabulaRNN"),
    ("mambatab", "MambaTab"),
    ("realmlp", "RealMLP"),
]

TIERS = ("base", "shared", "specific")
SOURCE_TIER_COLUMN = {
    "base": "base",
    "shared": "shared",
    # The full admissible pool is the paper's final Specific configuration.
    "specific": "pro",
}
TIER_LABELS = {
    "shared": r"+ \method{}-shared",
    "specific": r"+ \method{}-specific",
}

BACKBONE_ROW_LABELS = {
    "TabICL-v2": "TabICL-v2",
    "TabPFN-v3": "TabPFN-v3",
    "TabDPT-v1.2": "TabDPT v1.2",
}

TIER_ROW_COLORS = {
    "shared": "tabaceSharedRow",
    "specific": "tabaceSpecificRow",
}

TIER_TEXT_COLORS = {
    "shared": "tabaceSharedText",
    "specific": "tabaceSpecificText",
}

# Columns: Clean classification/regression, controlled perturbations, and
# distribution shifts.  Higher is better except for nRMSE.
COLUMNS = (
    ("clean", "classification", "accuracy"),
    ("clean", "classification", "auc"),
    ("clean", "regression", "r2"),
    ("clean", "regression", "nrmse"),
    ("controlled", "classification", "accuracy"),
    ("controlled", "classification", "auc"),
    ("controlled", "regression", "r2"),
    ("controlled", "regression", "nrmse"),
    ("shift", "classification", "accuracy"),
    ("shift", "classification", "auc"),
    ("shift", "regression", "r2"),
    ("shift", "regression", "nrmse"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tabdpt_shared_v2_rows() -> list[dict[str, str]]:
    """Read the audited family-blind TabDPT seven-shift Shared overlay."""

    audit = json.loads(TABDPT_SHARED_V2_AUDIT.read_text(encoding="utf-8"))
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
                f"TabDPT Shared-v2 audit failed {key}: "
                f"expected {expected!r}, found {audit.get(key)!r}"
            )
    rows = read_rows(TABDPT_SHARED_V2_SUMMARY)
    if len(rows) != 35 or {int(row["n_datasets"]) for row in rows} != {35}:
        raise ValueError("TabDPT Shared-v2 summary must contain 35 audited cells")
    return rows


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty collection")
    return math.fsum(values) / len(values)


def baseline_values() -> dict[str, list[float]]:
    rows = read_rows(BASELINE_SUMMARY)
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["family"] == "mean":
            continue
        suite = row["suite"]
        intensity = float(row["intensity"])
        if suite == "five_shift" and intensity == 0:
            condition = "clean"
        elif suite == "five_shift" and intensity > 0:
            condition = "controlled"
        elif suite == "seven_shift" and intensity > 0:
            condition = "shift"
        else:
            continue
        grouped[(row["model"], condition, row["task"], row["metric"])].append(
            float(row["mean"])
        )

    expected_cells = {"clean": 5, "controlled": 20, "shift": 28}
    output: dict[str, list[float]] = {}
    for model, _ in BASELINE_ORDER:
        values = []
        for condition, task, metric in COLUMNS:
            cells = grouped[(model, condition, task, metric)]
            if len(cells) != expected_cells[condition]:
                raise ValueError(
                    f"{model}/{condition}/{task}/{metric}: expected "
                    f"{expected_cells[condition]} cells, found {len(cells)}"
                )
            values.append(mean(cells))
        output[model] = values
    return output


def limix_values() -> dict[str, list[float]] | None:
    """Read the audited Base/Shared/Specific LimiX result block."""

    present = {key: path.is_file() for key, path in LIMIX_TABACE_SUMMARIES.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = [f"{track}/{task}" for (track, task), exists in present.items() if not exists]
        raise FileNotFoundError(
            "Refusing to emit a partial LimiX block; missing summaries: "
            + ", ".join(missing)
        )

    five_root = LIMIX_TABACE_SUMMARIES[("five", "classification")].parents[1]
    seven_root = LIMIX_TABACE_SUMMARIES[("seven", "classification")].parents[1]
    five_audit_path = five_root / "combined_audit.json"
    seven_audit_path = seven_root / "combined_audit.json"
    for path in (five_audit_path, seven_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing LimiX + TabACE audit: {path}")
    five_audit = json.loads(five_audit_path.read_text(encoding="utf-8"))
    seven_audit = json.loads(seven_audit_path.read_text(encoding="utf-8"))
    for task in ("classification", "regression"):
        if five_audit[task].get("single_seed") is not True:
            raise ValueError(f"LimiX five/{task} is not the frozen single-seed run")
        if five_audit[task].get("nonzero_rows_exactly_equal_to_frozen_default") is not True:
            raise ValueError(f"LimiX five/{task} nonzero audit did not pass")
        if seven_audit[task].get("single_seed") is not True:
            raise ValueError(f"LimiX seven/{task} is not the single-seed run")
        if seven_audit[task].get("target_labels_used_for_outer_selection") is not False:
            raise ValueError(f"LimiX seven/{task} target-label audit did not pass")

    dataset_counts = {
        "five": {"classification": 72, "regression": 35},
        "seven": {"classification": 35, "regression": 33},
    }
    expected_cells = {"clean": 5, "controlled": 20, "shift": 28}
    output = {tier: [] for tier in TIERS}
    for condition, task, metric in COLUMNS:
        track = "seven" if condition == "shift" else "five"
        rows = read_rows(LIMIX_TABACE_SUMMARIES[(track, task)])
        strength_key = "rho" if "rho" in rows[0] else "strength"
        selected = [
            row
            for row in rows
            if (float(row[strength_key]) == 0.0) == (condition == "clean")
        ]
        if len(selected) != expected_cells[condition]:
            raise ValueError(
                f"LimiX/{condition}/{task}/{metric}: expected "
                f"{expected_cells[condition]} cells, found {len(selected)}"
            )
        for row in selected:
            if int(row["n_datasets"]) != dataset_counts[track][task]:
                raise ValueError(
                    f"LimiX {track}/{task} has unexpected n_datasets={row['n_datasets']}"
                )
        for tier, source_tier in SOURCE_TIER_COLUMN.items():
            output[tier].append(
                mean([float(row[f"mean_{source_tier}_{metric}"]) for row in selected])
            )
    return output


def backbone_values() -> dict[str, dict[str, list[float]]]:
    expected_cells = {"clean": 5, "controlled": 20, "shift": 28}
    output: dict[str, dict[str, list[float]]] = {}
    tabdpt_shared_v2 = tabdpt_shared_v2_rows()
    for backbone, sources in BACKBONE_SUMMARIES.items():
        tier_values = {tier: [] for tier in TIERS}
        for condition, task, metric in COLUMNS:
            track = "seven" if condition == "shift" else "five"
            rows = read_rows(sources[(track, task)])
            shared_rows = (
                tabdpt_shared_v2
                if (backbone, track, task)
                == ("TabDPT-v1.2", "seven", "classification")
                else rows
            )
            strength_key = "rho" if "rho" in rows[0] else "strength"
            selected = [
                row
                for row in rows
                if (float(row[strength_key]) == 0.0) == (condition == "clean")
            ]
            if len(selected) != expected_cells[condition]:
                raise ValueError(
                    f"{backbone}/{condition}/{task}/{metric}: expected "
                    f"{expected_cells[condition]} cells, found {len(selected)}"
                )
            for tier, source_tier in SOURCE_TIER_COLUMN.items():
                tier_rows = selected
                if tier == "shared" and shared_rows is not rows:
                    tier_rows = [
                        row
                        for row in shared_rows
                        if (float(row["strength"]) == 0.0) == (condition == "clean")
                    ]
                    if len(tier_rows) != expected_cells[condition]:
                        raise ValueError(
                            f"{backbone}/{condition}/{task}/{metric}: expected "
                            f"{expected_cells[condition]} Shared-v2 cells, "
                            f"found {len(tier_rows)}"
                        )
                tier_values[tier].append(
                    mean(
                        [
                            float(row[f"mean_{source_tier}_{metric}"])
                            for row in tier_rows
                        ]
                    )
                )
        output[backbone] = tier_values
    return output


def latex_value(value: float) -> str:
    return f"{value:.3f}"


def build_table() -> str:
    baselines = baseline_values()
    limix = limix_values()
    backbones = backbone_values()
    lines = [
        "% Auto-generated by paper/scripts/generate_overall_results_table.py.",
        "% TFM rows come from the canonical summary_oof.csv files.",
        "% TabDPT seven-shift Shared classification uses the audited Shared-v2 OOF overlay.",
        "% Non-TFM rows come from the audited baseline CSV summary.",
        "% The LimiX rows are emitted only after all completion audits pass.",
        r"\definecolor{tabaceSharedRow}{HTML}{EAF4FB}",
        r"\definecolor{tabaceSpecificRow}{HTML}{FFF4D6}",
        r"\definecolor{tabaceSharedText}{HTML}{0072B2}",
        r"\definecolor{tabaceSpecificText}{HTML}{D55E00}",
        r"\ifdefined\ICLRLayout",
        r"\begin{table}[H]",
        r"\else",
        r"\begin{table}[H]",
        r"\fi",
        r"  \caption{Overall performance on OpenML-CC18 classification and OpenML-CTR23",
        r"  regression tasks. The perturbation and shift blocks macro-average all",
        r"  nonzero strengths over five and seven families, respectively.}",
        r"  \label{tab:overall-performance}",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.0pt}",
        r"  \renewcommand{\arraystretch}{0.98}",
        r"  \begin{tabular}{@{}l*{12}{c}@{}}",
        r"    \toprule",
        r"    & \multicolumn{4}{c}{Clean}",
        r"    & \multicolumn{4}{c}{Five Perturbations (Avg.)}",
        r"    & \multicolumn{4}{c}{Seven Shifts (Avg.)} \\",
        r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        r"    & \multicolumn{2}{c}{Classification} & \multicolumn{2}{c}{Regression}",
        r"    & \multicolumn{2}{c}{Classification} & \multicolumn{2}{c}{Regression}",
        r"    & \multicolumn{2}{c}{Classification} & \multicolumn{2}{c}{Regression} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"    \cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"    \cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r"    Method",
        r"    & Acc$\uparrow$ & AUC$\uparrow$ & $R^2\uparrow$ & nRMSE$\downarrow$",
        r"    & Acc$\uparrow$ & AUC$\uparrow$ & $R^2\uparrow$ & nRMSE$\downarrow$",
        r"    & Acc$\uparrow$ & AUC$\uparrow$ & $R^2\uparrow$ & nRMSE$\downarrow$ \\",
        r"    \midrule",
        r"    \multicolumn{13}{@{}l}{\textit{Non-TFM baselines}} \\",
    ]

    for model, label in BASELINE_ORDER:
        cells = [latex_value(value) for value in baselines[model]]
        lines.append("    " + label + " & " + " & ".join(cells) + r" \\")

    if limix is not None:
        lines.append(r"    \midrule")
        for tier in TIERS:
            cells = [latex_value(value) for value in limix[tier]]
            if tier in TIER_ROW_COLORS:
                lines.append(rf"    \rowcolor{{{TIER_ROW_COLORS[tier]}}}")
            label = "LimiX-2M" if tier == "base" else TIER_LABELS[tier]
            if tier != "base":
                text_color = TIER_TEXT_COLORS[tier]
                label = rf"\textcolor{{{text_color}}}{{\textbf{{{label}}}}}"
                cells = [
                    rf"\textcolor{{{text_color}}}{{\textbf{{{cell}}}}}"
                    for cell in cells
                ]
            lines.append("    " + label + " & " + " & ".join(cells) + r" \\")

    for backbone in BACKBONE_SUMMARIES:
        lines.append(r"    \midrule")
        for tier in TIERS:
            cells = [latex_value(value) for value in backbones[backbone][tier]]
            if tier in TIER_ROW_COLORS:
                lines.append(rf"    \rowcolor{{{TIER_ROW_COLORS[tier]}}}")
            label = BACKBONE_ROW_LABELS[backbone] if tier == "base" else TIER_LABELS[tier]
            if tier != "base":
                text_color = TIER_TEXT_COLORS[tier]
                label = rf"\textcolor{{{text_color}}}{{\textbf{{{label}}}}}"
                cells = [
                    rf"\textcolor{{{text_color}}}{{\textbf{{{cell}}}}}"
                    for cell in cells
                ]
            lines.append("    " + label + " & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
        r"\ifdefined\ICLRLayout",
        r"\end{table}",
        r"\else",
        r"\end{table}",
            r"\fi",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(build_table(), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
