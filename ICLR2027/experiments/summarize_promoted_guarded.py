from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from utils import RESULTS_DIR, write_rows


INPUTS = [
    (600, RESULTS_DIR / "openml_promoted_guarded_600_results.csv", RESULTS_DIR / "openml_promoted_guarded_600_decisions.csv"),
    (900, RESULTS_DIR / "openml_promoted_guarded_900_results.csv", RESULTS_DIR / "openml_promoted_guarded_900_decisions.csv"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for max_samples, result_path, decision_path in INPUTS:
        results = [row for row in read_csv(result_path) if row.get("method") == "adaptive_policy"]
        decisions = read_csv(decision_path)
        deltas = np.asarray([float(row["delta_vs_baseline"]) for row in results], dtype=float)
        chosen = [row["chosen_method"] for row in results]
        summary_rows.append(
            {
                "max_samples": max_samples,
                "runs": len(results),
                "chosen_propensity_interactions": int(sum(method == "propensity_interactions" for method in chosen)),
                "chosen_baseline": int(sum(method == "baseline_n1" for method in chosen)),
                "positive_runs": int(np.sum(deltas > 0)),
                "nonnegative_runs": int(np.sum(deltas >= 0)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
            }
        )
        for row in decisions:
            row = dict(row)
            row["max_samples"] = max_samples
            decision_rows.append(row)
    return summary_rows, decision_rows


def write_report(summary_rows: list[dict[str, object]], decision_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Promoted Candidate Guarded Policy",
        "",
        "This report checks whether the promoted `openml_mfeat_morphological + propensity_interactions` candidate can be safely exposed through guarded validation rather than used as an unconditional default.",
        "",
        "## Summary",
        "",
        "| Max Samples | Runs | Chose Interactions | Chose Baseline | Positive | Nonnegative | Mean Delta | Min Delta | Max Delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['max_samples']} | {row['runs']} | {row['chosen_propensity_interactions']} | {row['chosen_baseline']} | "
            f"{row['positive_runs']} | {row['nonnegative_runs']} | {float(row['mean_delta']):+.4f} | "
            f"{float(row['min_delta']):+.4f} | {float(row['max_delta']):+.4f} |"
        )

    lines.extend(["", "## Decisions", ""])
    lines.extend(["| Max Samples | Seed | Chosen | Validation Best | Validation Delta |", "| ---: | ---: | --- | --- | ---: |"])
    for row in decision_rows:
        lines.append(
            f"| {row['max_samples']} | {row['random_state']} | {row['chosen_method']} | "
            f"{row['validation_best_method']} | {float(row['validation_delta_vs_baseline']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- At 600 samples, the guarded policy enables `propensity_interactions` only when validation supports it, keeping all runs nonnegative while preserving most of the stable gain.",
            "- At 900 samples, the policy now falls back to baseline for `openml:18`, avoiding the negative outer-test deltas observed when the method was applied unconditionally.",
            "- This turns a raw candidate into a bounded policy rule: use it only inside the scale range where multi-seed evidence supports it.",
        ]
    )
    (RESULTS_DIR / "openml_promoted_guarded.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_rows, decision_rows = summarize()
    write_rows(RESULTS_DIR / "openml_promoted_guarded_summary.csv", summary_rows)
    write_rows(RESULTS_DIR / "openml_promoted_guarded_decisions.csv", decision_rows)
    write_report(summary_rows, decision_rows)
    print(f"Wrote {RESULTS_DIR / 'openml_promoted_guarded_summary.csv'}")
    print(f"Wrote {RESULTS_DIR / 'openml_promoted_guarded.md'}")


if __name__ == "__main__":
    main()
