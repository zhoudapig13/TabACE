#!/usr/bin/env python3
"""Append true frozen-LimiX per-query Context predictions to seven-shift banks."""

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
    ROOT, ROOT / "src", ROOT / "scripts", ROOT / "experiment6" / "scripts",
    ROOT / "experiment7" / "scripts", ROOT / "experiment8" / "scripts",
    ROOT / "experiment11" / "scripts", ROOT / "experiment13", ROOT / "experiment14",
    ROOT / "experiment15", ROOT / "experiment17", ROOT / "experiment28",
    ROOT / "fair_budget_experiments" / "baseline_shift_comparison" / "src", HERE,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_seven_banks import _adapt, _projector  # noqa: E402
from context_builders import classification_contexts, regression_contexts  # noqa: E402
from contextual_predictor import (  # noqa: E402
    predict_context_bank_classification,
    predict_context_bank_native,
)
import run_direct_baselines as frozen  # noqa: E402


CLASS_SPECS = {
    "limix/knn": ("knn", 20),
    "limix/balanced": ("balanced", 20),
    "limix/prototype": ("prototype", 20),
    "limix/clean": ("clean", 20),
    "limix/boundary": ("boundary", 20),
    "limix/minority": ("minority", 20),
}
REG_SPECS = {
    "limix/knn": ("knn", 32),
    "limix/clean": ("clean", 32),
    "limix/prototype": ("prototype", 32),
    "limix/bracket": ("bracket", 32),
    "limix/trimmed": ("trimmed", 32),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--pipelines", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--high-dimensional-threshold", type=int, default=1024)
    parser.add_argument("--high-dimensional-components", type=int, default=256)
    return parser.parse_args()


def load_config(root: Path, name: str, count: int) -> list[dict[str, Any]]:
    return json.loads((root / "config" / name).read_text(encoding="utf-8"))[:count]


def atomic_save(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    args = arguments()
    import openml
    import run_apa_tabicl as apa

    openml.config.set_root_cache_directory(args.base_run.resolve() / "openml_cache")
    apa.ROOT = args.base_run.resolve() / "protocol_input"
    limix_root = args.limix_root.resolve()
    sys.path.insert(0, str(limix_root))
    from inference.predictor import LimiXPredictor

    config_name = "cls_default_noretrieval.json" if args.task == "classification" else "reg_default_noretrieval.json"
    predictor = LimiXPredictor(
        device=torch.device("cuda"), model_path=str(args.model_path.resolve()),
        inference_config=load_config(limix_root, config_name, args.pipelines),
        mix_precision=True, seed=0 if args.task == "classification" else 2718,
    )
    requested = {value.strip().casefold() for value in args.dataset.split(",") if value.strip()}
    if args.task == "classification":
        datasets: list[str | int] = [int(value) for value in frozen.CC18_BINARY_DATASETS]
    else:
        datasets = list(frozen._regression_names("all"))
    if requested:
        datasets = [value for value in datasets if str(value).casefold() in requested]
    for dataset_position, dataset in enumerate(datasets, start=1):
        input_path = args.input / args.task / f"dataset_{dataset}.npz"
        output_path = args.output / args.task / f"dataset_{dataset}.npz"
        if output_path.exists():
            print(f"[{dataset_position}/{len(datasets)}] cached {dataset}", flush=True)
            continue
        with np.load(input_path, allow_pickle=False) as stored:
            payload = {key: np.asarray(stored[key]) for key in stored.files}
        conditions = frozen._seven_conditions(args.task, dataset)
        clean = next(row for row in conditions if row.family == frozen.SEVEN_SHIFTS[0] and float(row.strength) == 0.0)
        projector = _projector(np.asarray(clean.X_train), args.high_dimensional_threshold, args.high_dimensional_components)
        context_blocks: list[np.ndarray] = []
        true_names: tuple[str, ...] | None = None
        started = time.perf_counter()
        for position, condition in enumerate(conditions, start=1):
            X = _adapt(projector, condition.X_train)
            Q = _adapt(projector, condition.X_target)
            y = np.asarray(condition.y_train)
            base = np.asarray(payload["base"][position - 1])
            if args.task == "classification":
                classes = np.unique(y)
                contexts = classification_contexts(X, y, Q, base, classes, CLASS_SPECS)
                bank, bank_classes, _strategy, _calls = predict_context_bank_classification(
                    predictor, X, y, Q, contexts, batch_size=args.batch_size
                )
                np.testing.assert_array_equal(classes, bank_classes)
            else:
                contexts = regression_contexts(X, y, Q, base, REG_SPECS)
                center = float(np.mean(y))
                scale = max(float(np.std(y)), 1.0e-12)
                bank = predict_context_bank_native(
                    predictor, X, (np.asarray(y, dtype=np.float64) - center) / scale,
                    Q, contexts, task="regression", batch_size=args.batch_size,
                )
                bank = {name: np.asarray(value) * scale + center for name, value in bank.items()}
            names = tuple(bank)
            if true_names is None:
                true_names = names
            elif true_names != names:
                raise RuntimeError("true Context schema changed")
            context_blocks.append(np.stack([bank[name] for name in names], axis=-1))
            print(f"  {dataset} {position}/35 {condition.family} s={condition.strength:g}", flush=True)
        assert true_names is not None
        payload["members"] = np.concatenate([payload["members"], np.stack(context_blocks)], axis=-1)
        payload["member_names"] = np.asarray((*payload["member_names"].astype(str).tolist(), *true_names))
        payload["shared_names"] = np.asarray(true_names[:5])
        registry = json.loads(str(payload["specific_registry"]))
        for family in registry:
            if args.task == "classification":
                if family == "label":
                    registry[family] = ["prior/em", "prior/pcc", "limix/minority", "limix/balanced", "limix/boundary"]
                else:
                    registry[family] = list(true_names[:5])
            else:
                registry[family] = list(true_names[:5])
        payload["specific_registry"] = np.asarray(json.dumps(registry, sort_keys=True))
        atomic_save(output_path, payload)
        print(f"[{dataset_position}/{len(datasets)}] augmented {dataset} seconds={time.perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
