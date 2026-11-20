from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccs.utils import read_rows, to_float, write_rows  # noqa: E402


METRICS = ["pehe", "ate_error", "policy_value", "policy_regret", "uplift_top30", "auuc", "qini", "tau_ece", "runtime_sec"]


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["estimator"], row["retriever"], row["context_size"])
        groups[key].append(row)

    summary = []
    for (estimator, retriever, context_size), group in sorted(groups.items()):
        out = {
            "estimator": estimator,
            "retriever": retriever,
            "context_size": context_size,
            "runs": len(group),
        }
        for metric in METRICS:
            values = np.asarray([to_float(row.get(metric)) for row in group], dtype=float)
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values))
            out[f"{metric}_min"] = float(np.min(values))
            out[f"{metric}_max"] = float(np.max(values))
        summary.append(out)
    summary.sort(key=lambda row: (to_float(row.get("pehe_mean")), to_float(row.get("ate_error_mean"))))
    return summary


def write_report(path: Path, summary: list[dict[str, object]]) -> None:
    lines = [
        "# Experiment 3 Synthetic Summary",
        "",
        "Lower PEHE and ATE error are better. Higher policy value and uplift are better.",
        "",
        "| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary[:40]:
        lines.append(
            "| {estimator} | {retriever} | {context_size} | {runs} | {pehe:.4f} | {ate:.4f} | {policy:.4f} | {runtime:.4f} |".format(
                estimator=row.get("estimator", ""),
                retriever=row.get("retriever", ""),
                context_size=row.get("context_size", ""),
                runs=row.get("runs", ""),
                pehe=to_float(row.get("pehe_mean")),
                ate=to_float(row.get("ate_error_mean")),
                policy=to_float(row.get("policy_value_mean")),
                runtime=to_float(row.get("runtime_sec_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pehe(path: Path, summary: list[dict[str, object]]) -> None:
    context_rows = [
        row
        for row in summary
        if row.get("retriever") != "global_all_data" and row.get("estimator") == "local_context_t_learner"
    ]
    if not context_rows:
        return
    by_retriever = defaultdict(list)
    for row in context_rows:
        by_retriever[row["retriever"]].append(row)

    plt.figure(figsize=(8, 5))
    for retriever, rows in sorted(by_retriever.items()):
        rows = sorted(rows, key=lambda row: int(row["context_size"]))
        xs = [int(row["context_size"]) for row in rows]
        ys = [to_float(row.get("pehe_mean")) for row in rows]
        plt.plot(xs, ys, marker="o", label=retriever)
    plt.xlabel("Context size")
    plt.ylabel("PEHE")
    plt.title("Causal context selection sensitivity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize synthetic causal context selection results.")
    parser.add_argument("--input", default=str(ROOT / "results" / "synthetic_default" / "synthetic_results.csv"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    rows = read_rows(input_path)
    summary = summarize(rows)
    write_rows(output_dir / "synthetic_summary.csv", summary)
    write_report(output_dir / "synthetic_report.md", summary)
    plot_pehe(output_dir / "pehe_by_context_size.png", summary)
    print(f"[experiment3] wrote summary to {output_dir / 'synthetic_summary.csv'}")


if __name__ == "__main__":
    main()
