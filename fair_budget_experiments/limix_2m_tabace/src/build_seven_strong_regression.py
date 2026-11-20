#!/usr/bin/env python3
"""Build strong, label-safe LimiX regression Contexts for seven shifts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (
    ROOT,
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "experiment6" / "scripts",
    ROOT / "experiment7" / "scripts",
    ROOT / "experiment8" / "scripts",
    ROOT / "experiment11" / "scripts",
    ROOT / "experiment13",
    ROOT / "experiment14",
    ROOT / "experiment15",
    ROOT / "experiment17",
    ROOT / "experiment28",
    ROOT / "fair_budget_experiments" / "baseline_shift_comparison" / "src",
    HERE,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_seven_banks import (  # noqa: E402
    _adapt,
    _projector,
    _regression_base,
    atomic_save,
    load_config,
)
import run_direct_baselines as frozen  # noqa: E402


contexts: Any


FAMILIES = frozen.SEVEN_SHIFTS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--pipelines", type=int, default=8)
    parser.add_argument("--profile", choices=("original", "hierarchy_v4"), default="hierarchy_v4")
    parser.add_argument("--high-dimensional-threshold", type=int, default=1024)
    parser.add_argument("--high-dimensional-components", type=int, default=256)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _candidate_sets(
    profile: str,
    X: np.ndarray,
    y: np.ndarray,
    Q: np.ndarray,
    family: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if profile == "hierarchy_v4":
        return (
            contexts.shared_contexts_hierarchy_v4(X, y, Q),
            contexts.specific_contexts_hierarchy_v4(X, y, Q, family),
        )
    return (
        contexts.shared_contexts(X, y, Q),
        contexts.specific_contexts(X, y, Q, family),
    )


def main() -> None:
    args = arguments()
    import openml
    import run_apa_tabicl as apa

    args.input = args.input.resolve()
    args.output = args.output.resolve()
    openml.config.set_root_cache_directory(args.base_run.resolve() / "openml_cache")
    apa.ROOT = args.base_run.resolve() / "protocol_input"
    limix_root = args.limix_root.resolve()
    sys.path.insert(0, str(limix_root))
    # Some benchmark helpers expose a top-level ``inference.py`` module.
    # Remove that cached module before importing LimiX's ``inference`` package.
    sys.modules.pop("inference.predictor", None)
    sys.modules.pop("inference", None)
    from inference.predictor import LimiXPredictor
    global contexts
    from fair_budget_experiments.unified_context_ensemble.src import (
        strict_seven_regression_contexts as contexts_module,
    )
    contexts = contexts_module

    predictor = LimiXPredictor(
        device=torch.device("cuda"),
        model_path=str(args.model_path.resolve()),
        inference_config=load_config(
            limix_root, "reg_default_noretrieval.json", args.pipelines
        ),
        mix_precision=True,
        seed=2718,
    )
    requested = {
        value.strip().casefold() for value in args.dataset.split(",") if value.strip()
    }
    datasets = list(frozen._regression_names("all"))
    if requested:
        datasets = [value for value in datasets if str(value).casefold() in requested]

    for dataset_position, dataset in enumerate(datasets, start=1):
        input_path = args.input / "regression" / f"dataset_{dataset}.npz"
        output_path = args.output / "regression" / f"dataset_{dataset}.npz"
        if output_path.exists() and not args.force:
            print(f"[{dataset_position}/{len(datasets)}] cached {dataset}", flush=True)
            continue
        with np.load(input_path, allow_pickle=False) as stored:
            payload = {key: np.asarray(stored[key]) for key in stored.files}
        conditions = frozen._seven_conditions("regression", dataset)
        clean = next(
            row
            for row in conditions
            if row.family == FAMILIES[0] and float(row.strength) == 0.0
        )
        projector = _projector(
            np.asarray(clean.X_train),
            args.high_dimensional_threshold,
            args.high_dimensional_components,
        )
        shared_blocks: list[np.ndarray] = []
        specific_blocks: list[np.ndarray] = []
        diagnostics: list[str] = []
        shared_names: tuple[str, ...] | None = None
        specific_names: tuple[str, ...] | None = None
        started = time.perf_counter()
        for index, condition in enumerate(conditions):
            if args.max_conditions and index >= args.max_conditions:
                break
            X = np.asarray(condition.X_train, dtype=np.float32)
            y = np.asarray(condition.y_train, dtype=np.float64)
            Q = np.asarray(condition.X_target, dtype=np.float32)
            shared, specific = _candidate_sets(
                args.profile, X, y, Q, str(condition.family)
            )
            current_shared_names = tuple(
                f"strong/shared::{candidate.name}" for candidate in shared
            )
            current_specific_names = tuple(
                f"strong/specific::{position:02d}"
                for position in range(len(specific))
            )
            if shared_names is None:
                shared_names = current_shared_names
                specific_names = current_specific_names
            elif (
                shared_names != current_shared_names
                or specific_names != current_specific_names
            ):
                raise RuntimeError("strong regression Context schema changed")

            before = time.perf_counter()
            candidate_fallbacks: list[dict[str, str | int]] = []

            def predict_candidates(candidates, tier: str) -> np.ndarray:
                predictions = []
                for position, candidate in enumerate(candidates):
                    try:
                        value = _regression_base(
                            predictor,
                            _adapt(projector, candidate.X_context),
                            np.asarray(candidate.y_context, dtype=np.float64),
                            _adapt(projector, Q),
                        )
                    except (ValueError, FloatingPointError, OverflowError) as error:
                        # Some extreme covariate-shift Contexts can overflow a
                        # fitted power transform inside the frozen LimiX
                        # preprocessor.  A failed candidate is inadmissible for
                        # this condition, so retain the already frozen Base
                        # prediction instead of changing the target protocol.
                        value = np.asarray(payload["base"][index], dtype=np.float64)
                        candidate_fallbacks.append(
                            {
                                "tier": tier,
                                "position": position,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    predictions.append(value)
                return np.stack(predictions, axis=-1)

            shared_prediction = predict_candidates(shared, "shared")
            specific_prediction = predict_candidates(specific, "specific")
            # As in classification, an exactly unmodified Context is an
            # identity case.  Apply the Base-anchor before the OOF selector
            # ever receives candidate predictions; no target labels or shift
            # family are used by this guard.
            identity_guard = bool(np.isclose(float(condition.strength), 0.0))
            if identity_guard:
                base_prediction = np.asarray(payload["base"][index], dtype=np.float64)
                shared_prediction = np.repeat(
                    base_prediction[:, None], shared_prediction.shape[1], axis=1
                )
                specific_prediction = np.repeat(
                    base_prediction[:, None], specific_prediction.shape[1], axis=1
                )
            shared_blocks.append(shared_prediction)
            specific_blocks.append(specific_prediction)
            diagnostics.append(
                json.dumps(
                    {
                        "family": str(condition.family),
                        "strength": float(condition.strength),
                        "profile": args.profile,
                        "shared": [dict(candidate.metadata) for candidate in shared],
                        "specific": [dict(candidate.metadata) for candidate in specific],
                        "candidate_fallbacks": candidate_fallbacks,
                        "zero_strength_identity_guard": identity_guard,
                        "target_labels_used": False,
                    },
                    sort_keys=True,
                )
            )
            print(
                f"  {dataset} {index + 1}/35 {condition.family} s={condition.strength:g} "
                f"seconds={time.perf_counter() - before:.1f}",
                flush=True,
            )

        if args.max_conditions:
            print(
                f"pilot dataset={dataset} conditions={len(shared_blocks)} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
            continue

        assert shared_names is not None and specific_names is not None
        added_names = (*shared_names, *specific_names)
        added = np.concatenate(
            [np.stack(shared_blocks), np.stack(specific_blocks)], axis=-1
        )
        payload["members"] = np.concatenate([payload["members"], added], axis=-1)
        payload["member_names"] = np.asarray(
            (*payload["member_names"].astype(str).tolist(), *added_names)
        )
        payload["shared_names"] = np.asarray(shared_names)
        payload["specific_registry"] = np.asarray(
            json.dumps(
                {family: list(specific_names) for family in FAMILIES},
                sort_keys=True,
            )
        )
        payload["strong_context_diagnostics"] = np.asarray(diagnostics)
        atomic_save(output_path, **payload)
        print(
            f"[{dataset_position}/{len(datasets)}] built {dataset} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
