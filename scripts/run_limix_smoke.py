#!/usr/bin/env python3
"""Run reproducible LimiX-2M classification and regression smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_diabetes, load_digits, make_classification
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limix-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--classification-pipelines",
        type=int,
        default=1,
        help="Number of official classification ensemble members (default: 1).",
    )
    parser.add_argument(
        "--regression-pipelines",
        type=int,
        default=1,
        help="Number of official regression ensemble members (default: 1).",
    )
    parser.add_argument(
        "--high-dimensional-features",
        type=int,
        default=0,
        help="If positive, also run a synthetic binary compatibility probe.",
    )
    parser.add_argument(
        "--ovr-classes",
        type=int,
        default=0,
        help="If above 10, run an OvR compatibility probe with this many classes.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _config(path: Path, pipelines: int) -> list[dict[str, object]]:
    members = json.loads(path.read_text(encoding="utf-8"))
    if not 1 <= pipelines <= len(members):
        raise ValueError(f"pipelines must be in [1, {len(members)}]")
    return members[:pipelines]


def _classification(predictor_type, root: Path, model: Path, pipelines: int) -> dict[str, object]:
    features, target = load_digits(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        train_size=200,
        test_size=100,
        random_state=42,
        stratify=target,
    )
    config = _config(root / "config" / "cls_default_noretrieval.json", pipelines)
    started = time.perf_counter()
    predictor = predictor_type(
        device=torch.device("cpu"),
        model_path=str(model),
        inference_config=config,
        mix_precision=False,
        seed=42,
    )
    probabilities = np.asarray(
        predictor.predict(train_x, train_y, test_x, task_type="Classification")
    )
    elapsed = time.perf_counter() - started
    classes = np.asarray(predictor.classes)
    predicted = classes[np.argmax(probabilities, axis=1)]
    encoded = label_binarize(test_y, classes=classes)
    return {
        "dataset": "sklearn_digits",
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_features": int(train_x.shape[1]),
        "n_classes": int(len(classes)),
        "probability_shape": list(probabilities.shape),
        "probability_row_sum_max_error": float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        ),
        "accuracy": float(accuracy_score(test_y, predicted)),
        "auc_ovr_macro": float(
            roc_auc_score(encoded, probabilities, multi_class="ovr", average="macro")
        ),
        "seconds": elapsed,
    }


def _regression(predictor_type, root: Path, model: Path, pipelines: int) -> dict[str, object]:
    features, target = load_diabetes(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        train_size=200,
        test_size=100,
        random_state=42,
    )
    target_mean = float(np.mean(train_y))
    target_std = float(np.std(train_y))
    normalized_train_y = (train_y - target_mean) / target_std
    config = _config(root / "config" / "reg_default_noretrieval.json", pipelines)
    started = time.perf_counter()
    predictor = predictor_type(
        device=torch.device("cpu"),
        model_path=str(model),
        inference_config=config,
        mix_precision=False,
        seed=42,
    )
    normalized_prediction = np.asarray(
        predictor.predict(
            train_x,
            normalized_train_y,
            test_x,
            task_type="Regression",
        )
    ).reshape(-1)
    prediction = normalized_prediction * target_std + target_mean
    elapsed = time.perf_counter() - started
    rmse = float(np.sqrt(np.mean((test_y - prediction) ** 2)))
    return {
        "dataset": "sklearn_diabetes",
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_features": int(train_x.shape[1]),
        "prediction_shape": list(prediction.shape),
        "r2": float(r2_score(test_y, prediction)),
        "rmse": rmse,
        "nrmse_train_std": rmse / target_std,
        "seconds": elapsed,
    }


def _high_dimensional_probe(
    predictor_type,
    root: Path,
    model: Path,
    pipelines: int,
    n_features: int,
) -> dict[str, object]:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, n_features)).astype(np.float32)
    score = features[:, : min(8, n_features)].sum(axis=1)
    target = (score >= np.median(score)).astype(np.int64)
    train_x, test_x = features[:64], features[64:]
    train_y, test_y = target[:64], target[64:]
    config = _config(root / "config" / "cls_default_noretrieval.json", pipelines)
    started = time.perf_counter()
    predictor = predictor_type(
        device=torch.device("cpu"),
        model_path=str(model),
        inference_config=config,
        mix_precision=False,
        seed=42,
    )
    probabilities = np.asarray(
        predictor.predict(train_x, train_y, test_x, task_type="Classification")
    )
    predicted = np.asarray(predictor.classes)[np.argmax(probabilities, axis=1)]
    return {
        "kind": "synthetic_compatibility_probe_not_an_accuracy_benchmark",
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_features": n_features,
        "n_classes": 2,
        "probability_shape": list(probabilities.shape),
        "all_finite": bool(np.isfinite(probabilities).all()),
        "accuracy": float(accuracy_score(test_y, predicted)),
        "seconds": time.perf_counter() - started,
    }


def _ovr_probe(
    predictor_type,
    root: Path,
    model: Path,
    pipelines: int,
    n_classes: int,
) -> dict[str, object]:
    from limix_compat import predict_proba_native_or_ovr

    n_samples = max(15 * n_classes, 180)
    features, target = make_classification(
        n_samples=n_samples,
        n_features=24,
        n_informative=18,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    config = _config(root / "config" / "cls_default_noretrieval.json", pipelines)
    predictor = predictor_type(
        device=torch.device("cpu"),
        model_path=str(model),
        inference_config=config,
        mix_precision=False,
        seed=42,
    )
    started = time.perf_counter()
    result = predict_proba_native_or_ovr(
        predictor, train_x, train_y, test_x, max_native_classes=10
    )
    predicted = result.classes[np.argmax(result.probabilities, axis=1)]
    return {
        "kind": "synthetic_compatibility_probe_not_an_accuracy_benchmark",
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_features": int(train_x.shape[1]),
        "n_classes": int(len(result.classes)),
        "strategy": result.strategy,
        "model_calls": result.model_calls,
        "probability_shape": list(result.probabilities.shape),
        "probability_row_sum_max_error": float(
            np.max(np.abs(result.probabilities.sum(axis=1) - 1.0))
        ),
        "accuracy": float(accuracy_score(test_y, predicted)),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = _arguments()
    root = args.limix_root.resolve()
    model = args.model_path.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    sys.path.insert(0, str(root))
    from inference.predictor import LimiXPredictor

    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    report = {
        "model": model.name,
        "torch": torch.__version__,
        "device": "cpu",
        "official_ensemble_members_used": {
            "classification": args.classification_pipelines,
            "regression": args.regression_pipelines,
        },
        "max_supported_classes": int(model_config["decoder_config"]["num_classes"]),
        "classification": _classification(
            LimiXPredictor, root, model, args.classification_pipelines
        ),
        "regression": _regression(
            LimiXPredictor, root, model, args.regression_pipelines
        ),
    }
    if args.high_dimensional_features > 0:
        report["high_dimensional_classification"] = _high_dimensional_probe(
            LimiXPredictor,
            root,
            model,
            args.classification_pipelines,
            args.high_dimensional_features,
        )
    if args.ovr_classes > 10:
        report["ovr_classification"] = _ovr_probe(
            LimiXPredictor,
            root,
            model,
            args.classification_pipelines,
            args.ovr_classes,
        )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
