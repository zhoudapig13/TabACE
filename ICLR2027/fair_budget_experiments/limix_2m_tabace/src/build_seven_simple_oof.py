#!/usr/bin/env python3
"""Fast nested dataset-OOF selector for LimiX seven-shift Context banks."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_seven_oof import FAMILIES, STRENGTHS, load_records  # noqa: E402


ALPHAS = (0.10, 0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class Policy:
    members: tuple[str, ...]
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {"members": list(self.members), "alpha": float(self.alpha)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(value: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(value, dtype=np.float64), 1.0e-12, None)
    return output / output.sum(axis=1, keepdims=True)


def _prediction(base: np.ndarray, members: dict[str, np.ndarray], policy: Policy) -> np.ndarray:
    if not policy.members or policy.alpha <= 0.0:
        return np.asarray(base, dtype=np.float64)
    context = np.mean([np.asarray(members[name], dtype=np.float64) for name in policy.members], axis=0)
    value = (1.0 - policy.alpha) * np.asarray(base, dtype=np.float64) + policy.alpha * context
    return _normalize(value) if value.ndim == 2 else value


def _metrics(task: str, target: np.ndarray, prediction: np.ndarray, scale: float | None) -> dict[str, float]:
    if task == "classification":
        probability = _normalize(prediction)
        hard = probability.argmax(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = float(roc_auc_score(np.asarray(target, dtype=int), probability[:, 1]))
        return {"accuracy": float(np.mean(hard == np.asarray(target, dtype=int))), "auc": auc}
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    error = target - prediction
    denominator = max(float(np.sum((target - np.mean(target)) ** 2)), 1.0e-12)
    return {
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "nrmse": float(np.sqrt(np.mean(error**2))) / max(float(scale), 1.0e-12),
    }


def _utility(task: str, values: dict[str, float]) -> float:
    if task == "classification":
        return 0.5 * (values["accuracy"] + values["auc"])
    return 0.5 * (values["r2"] - values["nrmse"])


def _candidates(names: tuple[str, ...]) -> list[Policy]:
    subsets = [(name,) for name in names]
    if len(names) > 2:
        subsets.append(names)
    return [Policy((), 0.0), *(Policy(tuple(subset), alpha) for subset in subsets for alpha in ALPHAS)]


def _select(
    task: str,
    records: list[Any],
    scales: dict[tuple[str, str, float], float],
    member_getter,
    reference_getter,
) -> tuple[Policy, dict[str, Any]]:
    if not records:
        return Policy((), 0.0), {"score": 0.0}
    member_names = tuple(member_getter(records[0]))
    prepared = []
    for record in records:
        reference = np.asarray(reference_getter(record), dtype=np.float64)
        scale = scales[(record.key, record.shift, float(record.strength))]
        prepared.append(
            (
                record,
                reference,
                member_getter(record),
                scale,
                _metrics(task, record.y, reference, scale),
            )
        )
    best: tuple[float, float, int, Policy, dict[str, Any]] | None = None
    for policy in _candidates(member_names):
        by_dataset: dict[str, list[tuple[float, dict[str, float], dict[str, float]]]] = {}
        for record, reference, members, scale, reference_metrics in prepared:
            candidate = _prediction(reference, members, policy)
            candidate_metrics = _metrics(task, record.y, candidate, scale)
            gain = _utility(task, candidate_metrics) - _utility(task, reference_metrics)
            by_dataset.setdefault(record.key, []).append((gain, candidate_metrics, reference_metrics))
        dataset_gains = np.asarray([np.mean([item[0] for item in rows]) for rows in by_dataset.values()])
        mean_gain = float(np.mean(dataset_gains))
        downside = float(np.mean(np.clip(-dataset_gains, 0.0, None)))
        score = mean_gain - 0.35 * downside - 0.05 * float(np.std(dataset_gains))
        metric_gains: dict[str, float] = {}
        keys = tuple(next(iter(by_dataset.values()))[0][1])
        for key in keys:
            sign = -1.0 if key == "nrmse" else 1.0
            metric_gains[key] = sign * float(np.mean([
                candidate[key] - reference[key]
                for rows in by_dataset.values()
                for _gain, candidate, reference in rows
            ]))
        # Reject a policy that gains only by sacrificing one paper metric.
        noninferiority = min(metric_gains.values())
        rank = (score + 2.0 * min(noninferiority, 0.0), mean_gain, -len(policy.members), policy)
        diagnostics = {
            "score": score,
            "mean_dataset_utility_gain": mean_gain,
            "mean_dataset_downside": downside,
            "metric_gains": metric_gains,
        }
        if best is None or rank[:3] > best[:3]:
            best = (rank[0], rank[1], rank[2], policy, diagnostics)
    assert best is not None
    return best[3], best[4]


def _member_pools(record: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shared = {str(name): np.asarray(value) for name, value in record.shared.items()}
    full = {
        **shared,
        **{str(name): np.asarray(value) for name, value in record.specific.items()},
    }
    return shared, full


def build(task: str, banks: Path, output: Path) -> dict[str, Any]:
    records, scales, bank_metadata = load_records(task, banks)
    policies: dict[tuple[int, str, str], Policy] = {}
    policy_rows: list[dict[str, Any]] = []
    shared_predictions: dict[tuple[str, str, float], np.ndarray] = {}
    for fold in range(5):
        train = [record for record in records if record.fold != fold and float(record.strength) > 0.0]
        shared, diagnostics = _select(
            task, train, scales,
            lambda record: _member_pools(record)[0],
            lambda record: record.base,
        )
        policies[(fold, "shared", "all")] = shared
        policy_rows.append({"task": task, "outer_fold": fold, "tier": "shared", "scope": "all", **shared.to_dict(), **diagnostics})
        for record in records:
            if record.fold == fold:
                shared_predictions[(record.key, record.shift, float(record.strength))] = _prediction(
                    record.base, _member_pools(record)[0], shared
                )
        for family in FAMILIES:
            family_train = [record for record in train if record.shift == family]
            train_shared = {
                (record.key, record.shift, float(record.strength)): _prediction(
                    record.base, _member_pools(record)[0], shared
                )
                for record in family_train
            }
            specific, diagnostics = _select(
                task, family_train, scales,
                lambda record: _member_pools(record)[1],
                lambda record: train_shared[(record.key, record.shift, float(record.strength))],
            )
            policies[(fold, "pro", family)] = specific
            policy_rows.append({"task": task, "outer_fold": fold, "tier": "pro", "scope": family, **specific.to_dict(), **diagnostics})

    detail: list[dict[str, Any]] = []
    for record in records:
        key = (record.key, record.shift, float(record.strength))
        shared = shared_predictions[key]
        pro_policy = policies[(record.fold, "pro", record.shift)]
        pro = _prediction(shared, _member_pools(record)[1], pro_policy)
        row: dict[str, Any] = {
            "task": task, "dataset": record.key, "outer_fold": record.fold,
            "shift": record.shift, "strength": float(record.strength),
            "outer_test_labels_used_for_selection": False,
        }
        scale = scales[key]
        for method, prediction in (("base", record.base), ("shared", shared), ("specific", pro), ("pro", pro)):
            for metric, value in _metrics(task, record.y, prediction, scale).items():
                row[f"{method}_{metric}"] = value
        detail.append(row)

    metric_names = ("accuracy", "auc") if task == "classification" else ("r2", "nrmse")
    summary: list[dict[str, Any]] = []
    for family in FAMILIES:
        for strength in STRENGTHS:
            cell = [row for row in detail if row["shift"] == family and np.isclose(row["strength"], strength)]
            row = {"task": task, "shift": family, "strength": strength, "n_datasets": len(cell)}
            for method in ("base", "shared", "specific", "pro"):
                for metric in metric_names:
                    row[f"mean_{method}_{metric}"] = float(np.mean([value[f"{method}_{metric}"] for value in cell]))
            summary.append(row)

    nonzero = [row for row in detail if float(row["strength"]) > 0.0]
    overall: list[dict[str, Any]] = []
    for method in ("base", "shared", "pro"):
        row: dict[str, Any] = {"task": task, "method": method, "conditions": len(nonzero)}
        for metric in metric_names:
            row[metric] = float(np.mean([value[f"{method}_{metric}"] for value in nonzero]))
        overall.append(row)

    tests: list[dict[str, Any]] = []
    for lower, upper in (("base", "shared"), ("shared", "pro")):
        for metric in metric_names:
            by_dataset: dict[str, list[float]] = {}
            for row in nonzero:
                sign = -1.0 if metric == "nrmse" else 1.0
                by_dataset.setdefault(row["dataset"], []).append(
                    sign * (row[f"{upper}_{metric}"] - row[f"{lower}_{metric}"])
                )
            differences = np.asarray([np.mean(value) for value in by_dataset.values()])
            try:
                statistic, pvalue = wilcoxon(differences, alternative="greater", zero_method="wilcox")
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            tests.append({
                "task": task, "comparison": f"{upper}-{lower}", "metric": metric,
                "mean_improvement": float(np.mean(differences)), "median_improvement": float(np.median(differences)),
                "improved_datasets": int(np.sum(differences > 0)), "dataset_n": len(differences),
                "wilcoxon_statistic": float(statistic), "pvalue_one_sided": float(pvalue),
            })

    task_output = output / task
    _write_csv(task_output / "detail_oof.csv", detail)
    _write_csv(task_output / "summary_oof.csv", summary)
    _write_csv(task_output / "outer_selections.csv", policy_rows)
    _write_csv(task_output / "overall_nonzero.csv", overall)
    _write_csv(task_output / "paired_tests.csv", tests)
    audit = {
        **bank_metadata,
        "task": task,
        "protocol": "nested_dataset_oof_single_seed_v1",
        "single_seed": True,
        "outer_folds_are_dataset_splits_not_seeds": True,
        "target_labels_used_for_outer_selection": False,
        "shared_family_blind": True,
        "specific_family_aware": True,
        "specific_incremental_anchor": "shared",
        "paper_tier_mapping": {"LimiX": "base", "+ TabACE-shared": "shared", "+ TabACE-specific": "pro"},
        "paired_tests": tests,
    }
    (task_output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    args = arguments()
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    audits = {task: build(task, args.banks.resolve(), args.output.resolve()) for task in tasks}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "combined_audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
