#!/usr/bin/env python3
"""Build the benchmark-tuned, family-blind TabDPT Shared-v2 OOF result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_budget_experiments.tabdpt_tabace.src import (  # noqa: E402
    build_seven_classification as source,
)
from fair_budget_experiments.unified_context_ensemble.src.core import (  # noqa: E402
    FusionPolicy,
    PredictionBank,
    predict,
)


DEFAULT_OUTPUT = ROOT / "PaperResults/tabdpt_shared_opt_20260920/shared_v2_oof"
SHARED_V2 = (
    "target_matched_a",
    "target_matched_b",
    "target_matched_c",
    "recent_heavy",
    "prior_context",
)
ALPHA_CANDIDATES = (0.0, 0.95, 1.0)
TARGET_ACCURACY = 0.780
TARGET_AUC = 0.830


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _prediction(record: Any, alpha: float) -> np.ndarray:
    bank = PredictionBank(
        task="classification",
        base=np.asarray(record.base),
        members=record.shared,
    )
    policy = FusionPolicy(
        members=tuple(record.shared),
        aggregation="mean",
        alpha=alpha,
        metadata={"profile": "tabdpt_shared_v2"},
    )
    return predict(bank, policy)


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError("the seven-shift classification suite must be binary")
    hard = np.argmax(probability, axis=1)
    return {
        "accuracy": float(accuracy_score(target, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(target, hard)),
        "auc": float(roc_auc_score(target, probability[:, 1])),
    }


def _select_alpha(train_records: list[Any]) -> tuple[float, list[dict[str, float]]]:
    diagnostics: list[dict[str, float]] = []
    for alpha in ALPHA_CANDIDATES:
        scores = []
        accuracies = []
        aucs = []
        for record in train_records:
            metric = _metrics(record.y, _prediction(record, alpha))
            scores.append(0.5 * (metric["accuracy"] + metric["auc"]))
            accuracies.append(metric["accuracy"])
            aucs.append(metric["auc"])
        diagnostics.append(
            {
                "alpha": alpha,
                "selection_score": float(np.mean(scores)),
                "train_accuracy": float(np.mean(accuracies)),
                "train_auc": float(np.mean(aucs)),
            }
        )
    # A deterministic tie break keeps more Base weight.
    best = max(
        diagnostics,
        key=lambda row: (row["selection_score"], -row["alpha"]),
    )
    return float(best["alpha"]), diagnostics


def build(output: Path) -> dict[str, Any]:
    started = time.time()
    source.SHARED = SHARED_V2
    records = source.load_records()
    detail: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for fold in range(5):
        train_records = [record for record in records if int(record.fold) != fold]
        heldout_records = [record for record in records if int(record.fold) == fold]
        alpha, diagnostics = _select_alpha(train_records)
        selections.append(
            {
                "outer_fold": fold,
                "outer_train_datasets": len({record.key for record in train_records}),
                "outer_train_conditions": len(train_records),
                "members": list(SHARED_V2),
                "selected_alpha": alpha,
                "candidate_diagnostics": diagnostics,
                "shift_used_for_selection": False,
                "strength_used_for_selection": False,
                "outer_test_labels_used_for_selection": False,
            }
        )
        for record in heldout_records:
            base_metric = _metrics(record.y, np.asarray(record.base))
            shared_metric = _metrics(record.y, _prediction(record, alpha))
            detail.append(
                {
                    "task": "classification",
                    "dataset": record.key,
                    "outer_fold": fold,
                    "shift": record.shift,
                    "strength": float(record.strength),
                    "selected_alpha": alpha,
                    **{f"base_{key}": value for key, value in base_metric.items()},
                    **{f"shared_{key}": value for key, value in shared_metric.items()},
                }
            )

    summary: list[dict[str, Any]] = []
    for shift in source.records_api.SHIFTS:
        for strength in source.records_api.STRENGTHS:
            cell = [
                row
                for row in detail
                if row["shift"] == shift
                and np.isclose(float(row["strength"]), float(strength))
            ]
            row: dict[str, Any] = {
                "task": "classification",
                "shift": shift,
                "strength": strength,
                "n_datasets": len(cell),
            }
            for method in ("base", "shared"):
                for metric in ("accuracy", "balanced_accuracy", "auc"):
                    row[f"mean_{method}_{metric}"] = float(
                        np.mean([item[f"{method}_{metric}"] for item in cell])
                    )
            summary.append(row)

    positive = [row for row in detail if float(row["strength"]) > 0.0]
    origin = [row for row in detail if float(row["strength"]) == 0.0]
    metrics = {
        "origin": {
            metric: float(np.mean([row[f"shared_{metric}"] for row in origin]))
            for metric in ("accuracy", "balanced_accuracy", "auc")
        },
        "positive_strength_mean": {
            metric: float(np.mean([row[f"shared_{metric}"] for row in positive]))
            for metric in ("accuracy", "balanced_accuracy", "auc")
        },
        "base_positive_strength_mean": {
            metric: float(np.mean([row[f"base_{metric}"] for row in positive]))
            for metric in ("accuracy", "balanced_accuracy", "auc")
        },
    }
    passed = (
        metrics["positive_strength_mean"]["accuracy"] > TARGET_ACCURACY
        and metrics["positive_strength_mean"]["auc"] > TARGET_AUC
    )
    audit = {
        "status": "passed" if passed else "failed",
        "backbone": "TabDPT v1.2",
        "checkpoint": "tabdpt1_2.safetensors",
        "profile": "shared_v2_oof",
        "shared_context_registry": list(SHARED_V2),
        "alpha_candidates": list(ALPHA_CANDIDATES),
        "selection_objective": "equal mean of Accuracy and AUROC over all outer-training conditions",
        "shared_selector_receives_shift": False,
        "shared_selector_receives_strength": False,
        "outer_test_labels_used_for_selection": False,
        "target_labels_used_only_for_outer_training_selection_and_heldout_scoring": True,
        "common_inference_core": "unified_context_ensemble.src.core.predict",
        "curve_projection_used": False,
        "copied_predictions": False,
        "datasets": len({record.key for record in records}),
        "detail_rows": len(detail),
        "positive_strength_rows": len(positive),
        "targets": {"accuracy_strictly_greater_than": TARGET_ACCURACY, "auc_strictly_greater_than": TARGET_AUC},
        "metrics": metrics,
        "outer_selections": selections,
        "elapsed_seconds": time.time() - started,
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "detail_oof.csv", detail)
    _write_csv(output / "summary_oof.csv", summary)
    _write_csv(
        output / "outer_selections.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "candidate_diagnostics"},
                "members": json.dumps(row["members"]),
                "candidate_diagnostics": json.dumps(row["candidate_diagnostics"]),
            }
            for row in selections
        ],
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output / "run_summary.json").write_text(
        json.dumps(
            {
                "status": audit["status"],
                "input": "experiment17/results/tabdpt_v1_2/{unified_bank,final_bank}",
                "output": str(output.relative_to(ROOT)),
                "backbone": audit["backbone"],
                "checkpoint": audit["checkpoint"],
                "shared_context_registry": audit["shared_context_registry"],
                "metrics": metrics,
                "warnings": [],
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(
            "Shared-v2 did not meet the requested positive-strength thresholds"
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    audit = build(output.resolve())
    print(json.dumps(audit["metrics"], indent=2))


if __name__ == "__main__":
    main()
