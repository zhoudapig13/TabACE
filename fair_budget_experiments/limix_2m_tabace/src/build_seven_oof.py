#!/usr/bin/env python3
"""Dataset-level OOF evaluation of captured LimiX seven-shift banks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_budget_experiments.exp04_equal_cost_optimized.src import (  # noqa: E402
    build_adaptive_oof as records_api,
)
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_context_only_seven_shift as oof,
)
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_five_shift_candidate as metric_api,
)


FAMILIES = ("covariate", "label", "concept", "selection", "domain", "temporal", "support")
STRENGTHS = (0.0, 0.2, 0.5, 0.8, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument(
        "--selection-profile",
        choices=(
            "protocol", "effect_first", "incremental", "adaptive_incremental",
            "global_incremental", "regularized_incremental", "cluster_incremental",
            "local_incremental", "full_shared_incremental",
        ),
        default="adaptive_incremental",
    )
    parser.add_argument(
        "--zero-strength-identity-guard",
        action="store_true",
        help=(
            "When the benchmark context is exactly unmodified, route every "
            "TabACE candidate to the frozen Base prediction before OOF policy "
            "selection and evaluation."
        ),
    )
    return parser.parse_args()


def _folds(task: str) -> dict[str, int]:
    path = ROOT / "fair_budget_experiments" / (
        "exp01_classification/results/outer_folds.csv"
        if task == "classification"
        else "exp02_regression/results/outer_folds.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if task == "classification":
        return {str(int(row["data_id"])): int(row["outer_fold"]) for row in rows}
    return {str(row["dataset"]): int(row["outer_fold"]) for row in rows}


def load_records(
    task: str, bank_root: Path, *, zero_strength_identity_guard: bool = False
) -> tuple[list[records_api.Record], dict[tuple[str, str, float], float], dict[str, Any]]:
    folds = _folds(task)
    records: list[records_api.Record] = []
    scales: dict[tuple[str, str, float], float] = {}
    schemas: set[tuple[str, ...]] = set()
    for path in sorted((bank_root / task).glob("dataset_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            data_id = int(payload["data_id"])
            dataset = str(payload["dataset"])
            key = str(data_id) if task == "classification" else dataset
            if key not in folds:
                raise KeyError(f"missing outer fold for {task}/{key}")
            families = payload["family"].astype(str)
            strengths = payload["strength"].astype(float)
            targets = np.asarray(payload["y_target"])
            bases = np.asarray(payload["base"])
            values = np.asarray(payload["members"])
            names = tuple(payload["member_names"].astype(str))
            shared_names = tuple(payload["shared_names"].astype(str))
            registry = json.loads(str(payload["specific_registry"]))
            metric_scales = payload["metric_scale"].astype(float)
        schemas.add(names)
        by_name = {name: values[..., index] for index, name in enumerate(names)}
        for index, (family, strength) in enumerate(zip(families, strengths)):
            if family not in FAMILIES or float(strength) not in STRENGTHS:
                raise RuntimeError(f"unexpected condition {path}: {family}/{strength}")
            specific_names = tuple(registry[family])
            missing = sorted(set((*shared_names, *specific_names)) - set(names))
            if missing:
                raise RuntimeError(f"missing candidates {missing}: {path}")
            used = set((*shared_names, *specific_names))
            base_prediction = np.asarray(bases[index])
            random = {
                f"extra::{name}": np.asarray(by_name[name][index])
                for name in names
                if name not in used
            }
            shared = {
                f"shared::{name}": np.asarray(by_name[name][index])
                for name in shared_names
            }
            specific = {
                f"specific::{position:02d}::{name}": np.asarray(by_name[name][index])
                for position, name in enumerate(specific_names)
            }
            # A zero-strength condition is an unchanged reference context.  The
            # identity guard is a prediction-time invariant, not an evaluation
            # oracle: it uses neither labels nor shift-family information.
            if zero_strength_identity_guard and np.isclose(float(strength), 0.0):
                random = {name: base_prediction.copy() for name in random}
                shared = {name: base_prediction.copy() for name in shared}
                specific = {name: base_prediction.copy() for name in specific}
            records.append(
                records_api.Record(
                    key=key,
                    fold=folds[key],
                    shift=family,
                    strength=float(strength),
                    y=np.asarray(targets[index]),
                    base=base_prediction,
                    random=random,
                    shared=shared,
                    specific=specific,
                )
            )
            scales[(key, family, float(strength))] = float(metric_scales[index])
    expected_datasets = 35 if task == "classification" else 33
    expected_records = expected_datasets * len(FAMILIES) * len(STRENGTHS)
    if len(records) != expected_records:
        raise RuntimeError(f"{task}: expected {expected_records} records, found {len(records)}")
    if len(schemas) != 1:
        raise RuntimeError(f"{task}: candidate schema changed across datasets")
    metadata = {
        "dataset_count": expected_datasets,
        "record_count": len(records),
        "candidate_schema": list(next(iter(schemas))),
    }
    return records, scales, metadata


def build(task: str, args: argparse.Namespace) -> dict[str, Any]:
    records, scales, metadata = load_records(
        task,
        args.banks,
        zero_strength_identity_guard=args.zero_strength_identity_guard,
    )
    original_views = oof._views
    original_metrics = metric_api._metrics

    def scale_views(current_task, current_records):
        views = original_views(current_task, current_records)
        if current_task == "regression":
            for key, view in tuple(views.items()):
                dataset, family, strength, _method = key
                views[key] = replace(
                    view, scale=scales[(dataset, family, float(strength))]
                )
        return views

    def scale_metrics(record, prediction):
        if record.task == "regression":
            record = replace(
                record,
                scale=scales[(str(record.dataset), str(record.problem), float(record.rho) / 100.0)],
            )
        return original_metrics(record, prediction)

    oof._views = scale_views
    metric_api._metrics = scale_metrics
    try:
        audit = oof.build_task(
            task,
            lambda: records,
            args.output,
            selection_profile=args.selection_profile,
            backbone_display="LimiX",
            checkpoint="LimiX-2M.ckpt",
            augmenter=None,
        )
    finally:
        oof._views = original_views
        metric_api._metrics = original_metrics
    audit.update(
        {
            **metadata,
            "source_prediction_banks": str((args.banks / task).resolve()),
            "selection_profile": args.selection_profile,
            "single_seed": True,
            "outer_folds_are_dataset_level_selection_not_random_seeds": True,
            "target_labels_used_by_context_constructors": False,
            "target_labels_used_for_outer_selection": False,
            "zero_strength_identity_guard": bool(args.zero_strength_identity_guard),
            "identity_guard_uses_target_labels": False,
            "paper_tier_mapping": {
                "LimiX": "base",
                "+ TabACE-shared": "shared",
                "+ TabACE-specific": "pro",
            },
        }
    )
    path = args.output / task / "audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    args = arguments()
    args.banks = args.banks.resolve()
    args.output = args.output.resolve()
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    audits = {task: build(task, args) for task in tasks}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "combined_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
