#!/usr/bin/env python3
"""Expose the original admissible candidates to the strong Specific tier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_seven_banks import (  # noqa: E402
    CLASS_SHARED,
    CLASS_SPECIFIC,
    REG_SHARED,
    REG_SPECIFIC,
    atomic_save,
)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    args = parser.parse_args()
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    for task in tasks:
        old_shared = CLASS_SHARED if task == "classification" else REG_SHARED
        old_specific = CLASS_SPECIFIC if task == "classification" else REG_SPECIFIC
        repaired = 0
        nonfinite_replaced = 0
        for path in sorted((args.banks / task).glob("dataset_*.npz")):
            with np.load(path, allow_pickle=False) as stored:
                payload = {key: np.asarray(stored[key]) for key in stored.files}
            available = set(payload["member_names"].astype(str).tolist())
            registry = json.loads(str(payload["specific_registry"]))
            for family, current in registry.items():
                additions = [
                    *[name for name in old_shared if name in available],
                    *[name for name in old_specific[family] if name in available],
                ]
                registry[family] = _deduplicate([*current, *additions])
            payload["specific_registry"] = np.asarray(
                json.dumps(registry, sort_keys=True)
            )
            members = np.asarray(payload["members"], dtype=np.float64).copy()
            base = np.asarray(payload["base"], dtype=np.float64)
            if task == "classification":
                invalid = ~np.isfinite(members).all(axis=2)
                fallback = np.broadcast_to(base[..., None], members.shape)
            else:
                invalid = ~np.isfinite(members)
                fallback = np.broadcast_to(base[..., None], members.shape)
            nonfinite_replaced += int(np.sum(invalid))
            members = np.where(
                invalid[:, :, None, :] if task == "classification" else invalid,
                fallback,
                members,
            )
            payload["members"] = members
            payload["specific_includes_original_admissible_candidates"] = np.asarray(True)
            previous_replacements = int(
                payload.get(
                    "nonfinite_candidate_predictions_replaced_by_base",
                    np.asarray(0, dtype=np.int64),
                )
            )
            payload["nonfinite_candidate_predictions_replaced_by_base"] = np.asarray(
                previous_replacements + int(np.sum(invalid)), dtype=np.int64
            )
            atomic_save(path, **payload)
            repaired += 1
        print(
            f"{task}: repaired {repaired} banks; "
            f"nonfinite_predictions_replaced={nonfinite_replaced}",
            flush=True,
        )


if __name__ == "__main__":
    main()
