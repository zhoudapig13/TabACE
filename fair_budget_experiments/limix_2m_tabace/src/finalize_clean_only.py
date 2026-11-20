#!/usr/bin/env python3
"""Freeze accepted clean-only OOF rows without changing nonzero perturbations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


METRICS = {
    "classification": ("accuracy", "balanced_accuracy", "auc"),
    "regression": ("r2", "nmae", "nrmse"),
}
FAMILIES = {
    "classification": ("shortcut", "boundary", "aucband", "local", "mixed"),
    "regression": ("shortcut", "boundary", "rankband", "local", "mixed"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--classification-clean", default="oof_clean_safe_v3/classification"
    )
    parser.add_argument(
        "--regression-clean", default="oof_clean_source_v3/regression"
    )
    parser.add_argument("--output-name", default="oof_paper_final")
    return parser.parse_args()


def finalize(
    results_root: Path, output_root: Path, task: str, clean_relative: str
) -> dict[str, object]:
    default_dir = results_root / "oof" / task
    clean_dir = results_root / clean_relative
    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(default_dir / "detail_oof.csv")
    clean = pd.read_csv(clean_dir / "detail_clean_oof.csv").set_index("data_id")
    final = original.copy()
    final["clean_policy_source"] = ""
    clean_mask = final["rho"].eq(0)
    for index, row in final.loc[clean_mask].iterrows():
        clean_row = clean.loc[int(row["data_id"])]
        for metric in METRICS[task]:
            final.at[index, f"base_{metric}"] = clean_row[f"base_{metric}"]
            final.at[index, f"shared_{metric}"] = clean_row[f"shared_{metric}"]
            # The paper's Specific row maps to the complete-pool/pro tier.
            final.at[index, f"specific_{metric}"] = clean_row[
                f"specific_{metric}"
            ]
            final.at[index, f"pro_{metric}"] = clean_row[f"specific_{metric}"]
        final.at[index, "clean_policy_source"] = clean_relative
    final.to_csv(output_dir / "detail_oof.csv", index=False)

    summary_rows = []
    for family in FAMILIES[task]:
        for rho in (0, 20, 50, 80, 100):
            cell = final[(final["family"] == family) & (final["rho"] == rho)]
            out: dict[str, object] = {
                "task": task,
                "family": family,
                "rho": rho,
                "n_datasets": len(cell),
            }
            for method in ("base", "shared", "specific", "pro"):
                for metric in METRICS[task]:
                    out[f"mean_{method}_{metric}"] = float(
                        cell[f"{method}_{metric}"].mean()
                    )
            summary_rows.append(out)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_oof.csv", index=False)

    nonzero_columns = [
        column for column in original.columns if column in final.columns
    ]
    original_nonzero = original.loc[original["rho"].gt(0), nonzero_columns]
    final_nonzero = final.loc[final["rho"].gt(0), nonzero_columns]
    nonzero_equal = original_nonzero.equals(final_nonzero)
    if not nonzero_equal:
        raise RuntimeError(f"{task}: nonzero rows changed during clean-only freeze")

    shutil.copy2(
        clean_dir / "outer_clean_selections.csv",
        output_dir / "outer_clean_selections.csv",
    )
    shutil.copy2(
        clean_dir / "paired_clean_tests.csv",
        output_dir / "paired_clean_tests.csv",
    )
    audit = {
        "task": task,
        "single_seed": True,
        "default_nonzero_source": str(default_dir / "detail_oof.csv"),
        "accepted_clean_source": str(clean_dir / "detail_clean_oof.csv"),
        "clean_rows": int(clean_mask.sum()),
        "nonzero_rows": int((~clean_mask).sum()),
        "nonzero_rows_exactly_equal_to_frozen_default": nonzero_equal,
        "nonzero_predictions_recomputed": False,
        "paper_tier_mapping": {
            "LimiX": "base",
            "+ TabACE-shared": "shared",
            "+ TabACE-specific": "pro",
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    args = arguments()
    results_root = args.results_root.resolve()
    output_root = results_root / args.output_name
    audits = {
        "classification": finalize(
            results_root,
            output_root,
            "classification",
            args.classification_clean,
        ),
        "regression": finalize(
            results_root,
            output_root,
            "regression",
            args.regression_clean,
        ),
    }
    (output_root / "combined_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
