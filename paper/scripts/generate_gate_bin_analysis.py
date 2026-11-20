#!/usr/bin/env python3
"""Summarize TabACE Full behavior over label-free U/E quantile bins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


TASKS = ("classification", "regression")
BIN_LABELS = ("Low", "Mid", "High")
METRICS = (
    ("mean_alpha", r"Mean $\alpha_i$", "Blues", "sequential"),
    ("loss_improvement_vs_base", "Loss gain vs. Base", "RdBu", "diverging"),
    ("modified_fraction", "Modified fraction", "Purples", "sequential"),
    (
        "worsened_given_modified",
        "Worsened after modification",
        "Oranges",
        "sequential",
    ),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    output = []
    thresholds: dict[str, object] = {}
    for task in TASKS:
        part = frame.loc[frame["task"].eq(task)].copy()
        thresholds[task] = {}
        for column in ("u", "e"):
            q1, q2 = np.quantile(part[column].to_numpy(float), [1.0 / 3.0, 2.0 / 3.0])
            if not q1 < q2:
                raise ValueError(f"non-distinct {column} thresholds for {task}: {q1}, {q2}")
            part[f"{column}_bin"] = pd.cut(
                part[column],
                bins=(-np.inf, q1, q2, np.inf),
                labels=BIN_LABELS,
                include_lowest=True,
                ordered=True,
            )
            thresholds[task][column] = {"q33": float(q1), "q67": float(q2)}
        output.append(part)
    return pd.concat(output, ignore_index=True), thresholds


def summarize(frame: pd.DataFrame) -> list[dict[str, object]]:
    grouped = (
        frame.groupby(["task", "u_bin", "e_bin"], observed=True)
        .agg(
            samples=("alpha", "size"),
            mean_u=("u", "mean"),
            mean_e=("e", "mean"),
            mean_alpha=("alpha", "mean"),
            loss_improvement_vs_base=("loss_improvement_vs_base", "mean"),
            modified_fraction=("material_modified", "mean"),
            worsened_count=("worsened_material", "sum"),
            modified_count=("material_modified", "sum"),
        )
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [TASKS, BIN_LABELS, BIN_LABELS], names=["task", "u_bin", "e_bin"]
    )
    grouped = (
        grouped.set_index(["task", "u_bin", "e_bin"])
        .reindex(complete_index)
        .reset_index()
    )
    for column in ("samples", "worsened_count", "modified_count"):
        grouped[column] = grouped[column].fillna(0).astype(int)
    grouped["worsened_given_modified"] = (
        grouped["worsened_count"]
        / grouped["modified_count"].replace(0, np.nan)
    )
    if len(grouped) != 18:
        raise ValueError("expected a complete 3x3 U/E grid for each task")
    rows = grouped.to_dict(orient="records")
    for row in rows:
        row["u_bin"] = str(row["u_bin"])
        row["e_bin"] = str(row["e_bin"])
    return rows


def matrix(rows: list[dict[str, object]], task: str, metric: str) -> np.ndarray:
    lookup = {
        (str(row["u_bin"]), str(row["e_bin"])): float(row[metric])
        for row in rows
        if row["task"] == task
    }
    return np.asarray(
        [[lookup[(u_bin, e_bin)] for e_bin in BIN_LABELS] for u_bin in BIN_LABELS],
        dtype=float,
    )


def plot(rows: list[dict[str, object]], output_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.15), constrained_layout=False)
    for row_index, task in enumerate(TASKS):
        for column_index, (metric, title, cmap, scale) in enumerate(METRICS):
            ax = axes[row_index, column_index]
            values = matrix(rows, task, metric)
            color_map = mpl.colormaps[cmap].copy()
            color_map.set_bad("#F2F2F2")
            if scale == "diverging":
                limit = max(float(np.nanmax(np.abs(values))), 1.0e-6)
                image = ax.imshow(
                    values,
                    origin="lower",
                    cmap=color_map,
                    norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                    aspect="equal",
                )
            else:
                image = ax.imshow(
                    values,
                    origin="lower",
                    cmap=color_map,
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                )
            for u_index in range(3):
                for e_index in range(3):
                    value = values[u_index, e_index]
                    if np.isnan(value):
                        ax.text(
                            e_index,
                            u_index,
                            "--",
                            ha="center",
                            va="center",
                            fontsize=6.4,
                            color="#777777",
                        )
                        continue
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    ax.text(
                        e_index,
                        u_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.4,
                        color="white" if luminance < 0.53 else "#222222",
                    )
            ax.set_xticks(range(3), BIN_LABELS, fontsize=6.8)
            ax.set_yticks(range(3), BIN_LABELS, fontsize=6.8)
            ax.tick_params(length=0)
            ax.set_xlabel(r"Context evidence $E_i$", fontsize=7.2, labelpad=2)
            if column_index == 0:
                task_label = "Classification" if task == "classification" else "Regression"
                ax.set_ylabel(task_label + "\n" + r"Base unreliability $U_i$", fontsize=7.2)
            else:
                ax.set_yticklabels([])
            if row_index == 0:
                ax.set_title(title, fontsize=7.5, pad=5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#555555")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.08, top=0.93, wspace=0.22, hspace=0.34)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def mechanism_checks(rows: list[dict[str, object]]) -> dict[str, object]:
    checks: dict[str, object] = {}
    for task in TASKS:
        lookup = {
            (str(row["u_bin"]), str(row["e_bin"])): row
            for row in rows
            if row["task"] == task
        }
        low_low = lookup[("Low", "Low")]
        high_high = lookup[("High", "High")]
        checks[task] = {
            "alpha_high_high_exceeds_low_low": float(high_high["mean_alpha"])
            > float(low_low["mean_alpha"]),
            "modified_high_high_exceeds_low_low": float(high_high["modified_fraction"])
            > float(low_low["modified_fraction"]),
            "gain_high_high_exceeds_low_low": float(
                high_high["loss_improvement_vs_base"]
            )
            > float(low_low["loss_improvement_vs_base"]),
            "low_low": {
                key: float(low_low[key])
                for key in (
                    "mean_alpha",
                    "loss_improvement_vs_base",
                    "modified_fraction",
                    "worsened_given_modified",
                )
            },
            "high_high": {
                key: float(high_high[key])
                for key in (
                    "mean_alpha",
                    "loss_improvement_vs_base",
                    "modified_fraction",
                    "worsened_given_modified",
                )
            },
        }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--diagnostics", type=Path)
    source.add_argument("--summary-input", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--material-alpha", type=float, default=0.5)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_input is not None:
        rows = pd.read_csv(args.summary_input).to_dict(orient="records")
        if len(rows) != 18:
            raise ValueError("precomputed summary must contain two complete 3x3 grids")
        write_csv(args.results_dir / "gate_bin_summary.csv", rows)
        audit = {
            "status": "complete_from_precomputed_summary",
            "input": str(args.summary_input),
            "samples": int(sum(int(row["samples"]) for row in rows)),
            "binning": "task-wise U/E tertiles computed without target labels",
            "material_modification": f"alpha >= {args.material_alpha:g}",
            "worsened_denominator": "samples satisfying the material-modification rule",
            "checks": mechanism_checks(rows),
        }
        (args.results_dir / "audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        plot(rows, args.figure_dir / "tabace_gate_bin_analysis")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    assert args.diagnostics is not None
    frame = pd.read_csv(args.diagnostics, low_memory=False)
    required = {
        "task",
        "u",
        "e",
        "alpha",
        "loss_improvement_vs_base",
        "modified",
        "worsened",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing diagnostic columns: {sorted(missing)}")
    if not set(frame["task"]) == set(TASKS):
        raise ValueError(f"unexpected task set: {sorted(set(frame['task']))}")
    for column in ("u", "e", "alpha"):
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} leaves [0, 1]")
    if not 0.0 < args.material_alpha < 1.0:
        raise ValueError("--material-alpha must lie strictly between zero and one")
    frame["material_modified"] = frame["alpha"].ge(args.material_alpha)
    frame["worsened_material"] = (
        frame["worsened"].astype(bool) & frame["material_modified"]
    )

    frame, thresholds = add_bins(frame)
    rows = summarize(frame)
    checks = mechanism_checks(rows)
    write_csv(args.results_dir / "gate_bin_summary.csv", rows)
    audit = {
        "status": "complete",
        "input": str(args.diagnostics),
        "samples": int(len(frame)),
        "binning": "task-wise U/E tertiles computed without target labels",
        "thresholds": thresholds,
        "material_modification": f"alpha >= {args.material_alpha:g}",
        "worsened_denominator": "samples satisfying the material-modification rule",
        "empty_cells": int(sum(int(row["samples"]) == 0 for row in rows)),
        "checks": checks,
    }
    (args.results_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot(rows, args.figure_dir / "tabace_gate_bin_analysis")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
