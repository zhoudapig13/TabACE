#!/usr/bin/env python3
"""Build strong, label-safe LimiX Context candidates for seven shifts.

This runner reuses the Context constructors used by the other frozen
backbones.  Candidate generation can inspect the labeled source/reference
Context and the unlabeled target features, but never target labels.  The
existing LimiX Base prediction is retained byte-for-byte in the output bank.
"""

from __future__ import annotations

import argparse
import json
import os
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

from build_seven_banks import _adapt, _projector, atomic_save, load_config  # noqa: E402
from limix_compat import predict_proba_native_or_ovr  # noqa: E402
import run_direct_baselines as frozen  # noqa: E402
import run_unified_context_bank as E14  # noqa: E402
import run_shift_specific_contexts as E15  # noqa: E402


FAMILIES = frozen.SEVEN_SHIFTS
STRENGTHS = frozen.SEVEN_STRENGTHS
CORE_SHARED_SOURCE = (
    "uniform_bootstrap",
    "class_balanced",
    "target_matched_a",
    "mean_transport",
    "affine_transport",
)
FAMILY_SOURCE = {
    "covariate": (
        "target_matched_a", "target_matched_b", "target_matched_c",
        "mean_transport", "affine_transport",
    ),
    "label": (
        "class_balanced", "target_matched_a", "target_matched_b",
        "target_matched_c", "prior_context",
    ),
    "concept": (
        "recent_heavy", "class_balanced", "target_matched_a",
        "target_matched_b", "target_matched_c",
    ),
    "selection": (
        "target_matched_a", "target_matched_b", "target_matched_c",
        "balanced_target_matched", "nearest_target_pseudo",
    ),
    "domain": (
        "mean_transport", "affine_transport", "target_matched_a",
        "target_matched_b", "nearest_target_pseudo",
    ),
    "temporal": (
        "recent_heavy", "mean_transport", "affine_transport",
        "target_matched_b", "target_matched_c",
    ),
    "support": (
        "mean_transport", "affine_transport", "nearest_target_pseudo",
        "target_matched_b", "target_matched_c",
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--pipelines", type=int, default=4)
    parser.add_argument("--high-dimensional-threshold", type=int, default=1024)
    parser.add_argument("--high-dimensional-components", type=int, default=256)
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _observed(data_id: int, family: str, strength: float, split: Any) -> Any:
    policy = (
        E14.CovariatePolicy(
            "lr_probability",
            sign=1.0 if np.mean(split.y_historical) >= 0.5 else -1.0,
        )
        if family == "covariate"
        else None
    )
    return E14.construct_natural_condition(
        split,
        family,
        float(strength),
        covariate_policy=policy,
        label_hard_class=None,
        label_max_log_odds_shift=E15.LABEL_MAX_LOG_ODDS_SHIFT,
        label_observation_scale=E15.LABEL_OBSERVATION_SCALE,
    )


def _normalize(value: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(value, dtype=np.float64), 1.0e-12, None)
    return output / output.sum(axis=1, keepdims=True)


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
    from inference.predictor import LimiXPredictor

    predictor = LimiXPredictor(
        device=torch.device("cuda"),
        model_path=str(args.model_path.resolve()),
        inference_config=load_config(
            limix_root, "cls_default_noretrieval.json", args.pipelines
        ),
        mix_precision=True,
        seed=0,
    )

    requested = {
        value.strip().casefold() for value in args.dataset.split(",") if value.strip()
    }
    datasets = [int(value) for value in frozen.CC18_BINARY_DATASETS]
    if requested:
        names = {int(key): str(value) for key, value in frozen.CC18_DATASET_NAMES.items()}
        datasets = [
            value
            for value in datasets
            if str(value).casefold() in requested or names[value].casefold() in requested
        ]

    for dataset_position, data_id in enumerate(datasets, start=1):
        input_path = args.input / "classification" / f"dataset_{data_id}.npz"
        output_path = args.output / "classification" / f"dataset_{data_id}.npz"
        if output_path.exists() and not args.force:
            print(f"[{dataset_position}/{len(datasets)}] cached {data_id}", flush=True)
            continue
        with np.load(input_path, allow_pickle=False) as stored:
            payload = {key: np.asarray(stored[key]) for key in stored.files}

        split = E14.load_real_split(data_id)
        clean_observed = _observed(data_id, FAMILIES[0], 0.0, split)
        clean_X, _clean_y = E14._base_context(
            clean_observed, 3000007 + 137 * data_id
        )
        projector = _projector(
            np.asarray(clean_X),
            args.high_dimensional_threshold,
            args.high_dimensional_components,
        )

        call_count = 0

        def fit_predict(X: np.ndarray, y: np.ndarray, Q: np.ndarray, **_kwargs: Any) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            labels = np.unique(np.asarray(y))
            if len(labels) < 2:
                # A severe source shift can make a legal bootstrap Context
                # single-class.  LimiX requires two classes, so use the
                # corresponding smoothed empirical prior for this candidate.
                positive = 1.0 - 1.0e-6 if int(labels[0]) == 1 else 1.0e-6
                return np.repeat(
                    np.asarray([[1.0 - positive, positive]], dtype=np.float64),
                    len(Q),
                    axis=0,
                )
            result = predict_proba_native_or_ovr(
                predictor,
                _adapt(projector, np.asarray(X)),
                np.asarray(y),
                _adapt(projector, np.asarray(Q)),
                max_native_classes=10,
            )
            return _normalize(result.probabilities)

        E14._fit_predict = fit_predict
        E15._fit_predict = fit_predict
        # _context_ensemble is defined in the experiment-13 helper module and
        # therefore resolves _fit_predict in that module's globals.
        sys.modules[E14._context_ensemble.__module__]._fit_predict = fit_predict

        shared_blocks: list[np.ndarray] = []
        specific_blocks: list[np.ndarray] = []
        diagnostics: list[str] = []
        shared_names: tuple[str, ...] | None = None
        specific_width = 0
        started = time.perf_counter()
        condition_count = 0
        stop = False
        for family in FAMILIES:
            for strength in STRENGTHS:
                if args.max_conditions and condition_count >= args.max_conditions:
                    stop = True
                    break
                index = condition_count
                observed = _observed(data_id, family, float(strength), split)
                base = np.asarray(payload["base"][index], dtype=np.float64)
                before = call_count
                shared = E14.build_unified_context_bank(
                    observed,
                    seed=3000007 + 137 * data_id,
                    replicate_seed=2000003 + 131 * data_id,
                    device="cuda",
                )
                current_shared_names = tuple(f"strong/shared::{name}" for name in shared.names[1:])
                if shared_names is None:
                    shared_names = current_shared_names
                elif shared_names != current_shared_names:
                    raise RuntimeError("strong Shared schema changed")
                current_shared = np.transpose(
                    np.asarray(shared.predictions[:, 1:], dtype=np.float64),
                    (0, 2, 1),
                )

                if args.shared_only:
                    current_specific = np.empty((len(base), base.shape[1], 0), dtype=np.float64)
                    current_specific_names: tuple[str, ...] = ()
                else:
                    unified = _normalize(np.mean(shared.predictions, axis=1))
                    specific = E15.build_shift_specific_bank(
                        observed,
                        shift=family,
                        seed=5000011 + 149 * data_id,
                        device="cuda",
                        base_prediction=base,
                        unified_prediction=unified,
                        upstream_context_predictions=shared.predictions,
                    )
                    current_specific = np.transpose(
                        np.asarray(specific.predictions, dtype=np.float64),
                        (0, 2, 1),
                    )
                    current_specific_names = tuple(
                        f"strong/specific::{position:02d}"
                        for position in range(current_specific.shape[2])
                    )
                # The exact zero-strength condition has the original reference
                # Context.  Route all alternate branches to frozen Base here so
                # the guard is enforced before OOF policy fitting; it does not
                # inspect target labels or the shift family.
                identity_guard = bool(np.isclose(float(strength), 0.0))
                if identity_guard:
                    current_shared = np.repeat(
                        base[:, :, None], current_shared.shape[2], axis=2
                    )
                    if current_specific.shape[2]:
                        current_specific = np.repeat(
                            base[:, :, None], current_specific.shape[2], axis=2
                        )
                shared_blocks.append(current_shared)
                specific_width = max(specific_width, current_specific.shape[2])
                specific_blocks.append(current_specific)
                diagnostics.append(
                    json.dumps(
                        {
                            "family": family,
                            "strength": float(strength),
                            "shared_names": list(shared.names[1:]),
                            "specific_source_names": list(current_specific_names),
                            "zero_strength_identity_guard": identity_guard,
                            "model_calls": call_count - before,
                            "target_labels_used": False,
                        },
                        sort_keys=True,
                    )
                )
                condition_count += 1
                print(
                    f"  {data_id} {condition_count}/35 {family} s={strength:g} "
                    f"calls={call_count - before} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if stop:
                break

        if args.max_conditions:
            print(
                f"pilot dataset={data_id} conditions={condition_count} calls={call_count} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
            continue

        assert shared_names is not None
        padded_specific: list[np.ndarray] = []
        for value in specific_blocks:
            if value.shape[2] < specific_width:
                padding = np.repeat(
                    value[:, :, -1:], specific_width - value.shape[2], axis=2
                )
                value = np.concatenate([value, padding], axis=2)
            padded_specific.append(value)
        added_names = (*shared_names, *(
            f"strong/specific::{position:02d}" for position in range(specific_width)
        ))
        added = np.concatenate(
            [np.stack(shared_blocks), np.stack(padded_specific)], axis=-1
        )
        payload["members"] = np.concatenate([payload["members"], added], axis=-1)
        payload["member_names"] = np.asarray(
            (*payload["member_names"].astype(str).tolist(), *added_names)
        )
        # Shared stays family-blind; Specific receives the Shared pool plus its
        # family-indexed slots.  The selector remains dataset-level OOF.
        core_shared = tuple(
            f"strong/shared::{name}" for name in CORE_SHARED_SOURCE
        )
        payload["shared_names"] = np.asarray(core_shared)
        registry = {
            family: [
                *(f"strong/shared::{name}" for name in FAMILY_SOURCE[family]),
                *(
                    f"strong/specific::{position:02d}"
                    for position in range(specific_width)
                ),
            ]
            for family in FAMILIES
        }
        payload["specific_registry"] = np.asarray(json.dumps(registry, sort_keys=True))
        payload["strong_context_diagnostics"] = np.asarray(diagnostics)
        atomic_save(output_path, **payload)
        print(
            f"[{dataset_position}/{len(datasets)}] built {data_id} calls={call_count} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
