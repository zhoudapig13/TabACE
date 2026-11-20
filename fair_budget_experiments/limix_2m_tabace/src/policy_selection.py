"""Source-only selection of compact LimiX + TabACE fusion policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Literal, Sequence

import numpy as np


Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class PredictionRecord:
    family: str
    strength: int
    target: np.ndarray
    base: np.ndarray
    members: dict[str, np.ndarray]
    classes: np.ndarray | None = None


@dataclass(frozen=True)
class ContextPolicy:
    members: tuple[str, ...] = ()
    alpha: float = 0.0
    validation_utility: float = 0.0
    mean_base_relative_gain: float = 0.0
    worst_cell_gain: float = 0.0
    selection_scope: str = "base"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["members"] = list(self.members)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ContextPolicy":
        return cls(
            members=tuple(str(item) for item in value.get("members", [])),
            alpha=float(value.get("alpha", 0.0)),
            validation_utility=float(value.get("validation_utility", 0.0)),
            mean_base_relative_gain=float(value.get("mean_base_relative_gain", 0.0)),
            worst_cell_gain=float(value.get("worst_cell_gain", 0.0)),
            selection_scope=str(value.get("selection_scope", "base")),
        )


def fuse(base: np.ndarray, members: dict[str, np.ndarray], policy: ContextPolicy) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    if not policy.members or policy.alpha <= 0.0:
        return base.copy()
    center = np.mean(
        np.stack([np.asarray(members[name], dtype=np.float64) for name in policy.members]),
        axis=0,
    )
    output = (1.0 - policy.alpha) * base + policy.alpha * center
    if output.ndim == 2:
        output = np.clip(output, 1.0e-12, None)
        output /= output.sum(axis=1, keepdims=True)
    return output


def _classification_auc(
    target: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> float:
    positions = {label: index for index, label in enumerate(classes.tolist())}
    encoded = np.asarray([positions[label] for label in target], dtype=int)
    values = []
    for index in range(len(classes)):
        binary = (encoded == index).astype(int)
        if np.any(binary == 1) and np.any(binary == 0):
            positive = probability[binary == 1, index]
            negative = probability[binary == 0, index]
            comparisons = positive[:, None] - negative[None, :]
            values.append(
                float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))
            )
    return float(np.mean(values)) if values else 0.5


def _record_score(record: PredictionRecord, prediction: np.ndarray, task: Task) -> float:
    target = np.asarray(record.target)
    if task == "classification":
        if record.classes is None:
            raise ValueError("classification record is missing classes")
        probability = np.asarray(prediction, dtype=np.float64)
        predicted = record.classes[np.argmax(probability, axis=1)]
        accuracy = float(np.mean(predicted == target))
        auc = _classification_auc(target, probability, record.classes)
        # Both reported classification metrics have equal influence.
        return 0.5 * (accuracy + auc)
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    scale = max(float(np.std(target)), 1.0e-12)
    return -float(np.mean(error**2) / (scale**2))


def policy_diagnostics(
    records: Sequence[PredictionRecord], policy: ContextPolicy, *, task: Task
) -> tuple[float, float, float]:
    gains = []
    for record in records:
        base_score = _record_score(record, record.base, task)
        value = fuse(record.base, record.members, policy)
        gains.append(_record_score(record, value, task) - base_score)
    array = np.asarray(gains, dtype=np.float64)
    mean = float(np.mean(array))
    worst = float(np.min(array))
    # Modest dispersion and downside penalties keep per-dataset selection from
    # buying a high average with a catastrophic simulated perturbation cell.
    utility = mean - 0.10 * float(np.std(array)) + 0.05 * min(worst, 0.0)
    return utility, mean, worst


def candidate_policies(
    member_names: Sequence[str],
    *,
    max_members: int = 3,
    alpha_grid: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
) -> list[ContextPolicy]:
    names = tuple(dict.fromkeys(str(name) for name in member_names))
    output = [ContextPolicy()]
    for width in range(1, min(max_members, len(names)) + 1):
        for subset in combinations(names, width):
            output.extend(
                ContextPolicy(members=subset, alpha=float(alpha)) for alpha in alpha_grid
            )
    return output


def select_policy(
    records: Sequence[PredictionRecord],
    *,
    task: Task,
    member_names: Sequence[str],
    selection_scope: str,
    reference: ContextPolicy | None = None,
    max_members: int = 3,
    minimum_gain: float = 1.0e-6,
) -> ContextPolicy:
    """Select on source-only records, optionally requiring gain over a policy."""

    if not records:
        raise ValueError("policy selection needs at least one record")
    reference = reference or ContextPolicy()
    reference_values = [
        _record_score(record, fuse(record.base, record.members, reference), task)
        for record in records
    ]
    best: ContextPolicy | None = None
    best_key: tuple[float, float, float, float] | None = None
    for raw in candidate_policies(member_names, max_members=max_members):
        values = [
            _record_score(record, fuse(record.base, record.members, raw), task)
            for record in records
        ]
        gains = np.asarray(values, dtype=np.float64) - np.asarray(reference_values)
        mean = float(np.mean(gains))
        worst = float(np.min(gains))
        utility = mean - 0.10 * float(np.std(gains)) + 0.05 * min(worst, 0.0)
        # Tie-break toward smaller/safer interventions.
        key = (utility, mean, -float(len(raw.members)), -float(raw.alpha))
        if best_key is None or key > best_key:
            best_key = key
            best = ContextPolicy(
                members=raw.members,
                alpha=raw.alpha,
                validation_utility=utility,
                mean_base_relative_gain=mean,
                worst_cell_gain=worst,
                selection_scope=selection_scope,
            )
    assert best is not None
    if best.mean_base_relative_gain <= minimum_gain:
        # The reference is a fail-closed choice.  For Specific this means exact
        # inheritance of Shared rather than using the outer benchmark to rescue
        # an unsupported family-specific change.
        utility, mean, worst = policy_diagnostics(records, reference, task=task)
        return ContextPolicy(
            members=reference.members,
            alpha=reference.alpha,
            validation_utility=utility,
            mean_base_relative_gain=mean,
            worst_cell_gain=worst,
            selection_scope=f"{selection_scope}:reference_fallback",
        )
    return best
