#!/usr/bin/env python3
"""Generate the appendix fixed-bank formula diagnostic from frozen CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ORDER = {
    "classification": [
        "no_confidence",
        "no_agreement",
        "uniform_prior",
        "no_base_gate",
        "no_context_gate",
        "static_oof_reference",
    ],
    "regression": [
        "no_agreement",
        "uniform_prior",
        "no_base_gate",
        "no_context_gate",
        "static_oof_reference",
    ],
}

LABELS = {
    "static_oof_reference": "Full reference",
    "no_confidence": "w/o Confidence",
    "no_agreement": "w/o Agreement",
    "uniform_prior": "Uniform prior",
    "no_base_gate": "w/o Base gate",
    "no_context_gate": "w/o Context gate",
}

TICK_LABELS = {
    "static_oof_reference": "Full",
    "no_confidence": "w/o Conf.",
    "no_agreement": "w/o Agree.",
    "uniform_prior": "Uniform prior",
    "no_base_gate": "w/o Base",
    "no_context_gate": "w/o Context",
}

REFERENCE = {
    "classification": 0.6720678212691327,
    "regression": 0.7392303537291853,
}


def read_summary(path: Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["benchmark"] == "five+seven_equal_macro":
                values[(row["task"], row["variant"])] = float(row["value"])
    for task, value in REFERENCE.items():
        values[(task, "static_oof_reference")] = value
    return values


def read_tests(path: Path) -> dict[tuple[str, str], dict[str, float | bool]]:
    tests: dict[tuple[str, str], dict[str, float | bool]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tests[(row["task"], row["variant"])] = {
                "degradation_low": float(row["ci95_low"]),
                "degradation_high": float(row["ci95_high"]),
                "holm_p": float(row["holm_p_two_sided_within_task"]),
                "significant": row["significant_holm_0p05"].strip().lower() == "true",
            }
    return tests


def metric_interval(
    task: str, ref: float, test: dict[str, float | bool]
) -> tuple[float, float]:
    lo = float(test["degradation_low"])
    hi = float(test["degradation_high"])
    if task == "classification":
        return ref - hi, ref - lo
    return ref + lo, ref + hi


def write_plot_data(
    path: Path,
    values: dict[tuple[str, str], float],
    tests: dict[tuple[str, str], dict[str, float | bool]],
) -> None:
    fields = [
        "task",
        "variant",
        "label",
        "value",
        "reference",
        "ci95_metric_low",
        "ci95_metric_high",
        "holm_p",
        "significant_holm_0p05",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ("classification", "regression"):
            ref = REFERENCE[task]
            for variant in ORDER[task]:
                value = values.get((task, variant))
                if value is None:
                    writer.writerow(
                        {
                            "task": task,
                            "variant": variant,
                            "label": LABELS[variant],
                            "reference": f"{ref:.12f}",
                        }
                    )
                    continue
                test = tests.get((task, variant))
                row = {
                    "task": task,
                    "variant": variant,
                    "label": LABELS[variant],
                    "value": f"{value:.12f}",
                    "reference": f"{ref:.12f}",
                }
                if test is not None:
                    ci_low, ci_high = metric_interval(task, ref, test)
                    row.update(
                        {
                            "ci95_metric_low": f"{ci_low:.12f}",
                            "ci95_metric_high": f"{ci_high:.12f}",
                            "holm_p": f"{float(test['holm_p']):.12g}",
                            "significant_holm_0p05": str(bool(test["significant"])),
                        }
                    )
                writer.writerow(row)


def draw_panel(
    ax: plt.Axes,
    task: str,
    values: dict[tuple[str, str], float],
    tests: dict[tuple[str, str], dict[str, float | bool]],
) -> None:
    order = ORDER[task]
    x = np.arange(len(order))
    ref = REFERENCE[task]
    baseline = 0.62 if task == "classification" else 0.72

    for xi, variant in zip(x, order):
        value = values.get((task, variant))
        if value is None:
            raise KeyError(f"missing plotted value for {task}/{variant}")

        if variant == "static_oof_reference":
            face, edge, hatch = "#56B4E9", "#0072B2", None
        else:
            face, edge, hatch = "#B8B8B8", "#777777", None

        ax.bar(
            xi,
            value - baseline,
            bottom=baseline,
            width=0.73 if variant == "static_oof_reference" else 0.67,
            color=face,
            edgecolor=edge,
            linewidth=0.8,
            hatch=hatch,
            zorder=2,
        )

        test = tests.get((task, variant))
        significant = bool(test["significant"]) if test is not None else False
        suffix = "*" if significant else ""
        offset = 0.0022 if task == "classification" else 0.0050
        ax.text(
            xi,
            value + offset,
            f"{value:.3f}{suffix}",
            va="bottom",
            ha="center",
            fontsize=7.2 if variant == "static_oof_reference" else 6.8,
            fontweight="bold" if variant == "static_oof_reference" else "normal",
            color="#0072B2" if variant == "static_oof_reference" else "#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TICK_LABELS[v] for v in order], fontsize=6.9, rotation=27, ha="right"
    )
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=7.5)
    break_kwargs = dict(transform=ax.transAxes, color="#444444", clip_on=False, linewidth=0.9)
    ax.plot((-0.010, 0.010), (-0.012, 0.012), **break_kwargs)
    ax.plot((-0.010, 0.010), (0.014, 0.038), **break_kwargs)
    if task == "classification":
        ax.set_ylim(0.62, 0.686)
        ax.set_yticks([0.62, 0.64, 0.66, 0.68])
        ax.set_ylabel("Accuracy", fontsize=8.5)
        ax.set_title("(a) Classification (higher is better)", fontsize=9.0, pad=7)
    else:
        ax.set_ylim(0.72, 0.872)
        ax.set_yticks([0.72, 0.76, 0.80, 0.84])
        ax.set_ylabel("nRMSE", fontsize=8.5)
        ax.set_title("(b) Regression (lower is better)", fontsize=9.0, pad=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = read_summary(args.results_dir / "summary.csv")
    tests = read_tests(args.results_dir / "paired_tests.csv")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    draw_panel(axes[0], "classification", values, tests)
    draw_panel(axes[1], "regression", values, tests)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.26, top=0.90, wspace=0.20)

    stem = args.output_dir / "tabace_k5_formula_diagnostic"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    write_plot_data(
        args.output_dir / "tabace_k5_formula_diagnostic_data.csv", values, tests
    )


if __name__ == "__main__":
    main()
