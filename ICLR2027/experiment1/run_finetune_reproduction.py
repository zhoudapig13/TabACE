from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor, TabICLClassifier, TabICLRegressor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

sys.path.insert(0, str(ROOT / "experiments"))

from utils import add_common_args, environment_info, load_named_dataset, select_device  # noqa: E402


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate(task: str, y_true, y_pred) -> tuple[str, float, dict[str, float]]:
    if task == "classification":
        return "accuracy", float(accuracy_score(y_true, y_pred)), {}
    return "r2", float(r2_score(y_true, y_pred)), {"mae": float(mean_absolute_error(y_true, y_pred))}


def resolve_output_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return RESULTS_DIR / path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a reproduced TabICL checkpoint on one downstream dataset.")
    parser.add_argument("--dataset", default="synthetic_classification")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--output", default="finetune_reproduction_results.csv")
    parser.add_argument("--checkpoint-dir", default=str(EXPERIMENT_DIR / "finetune_ckpts"))
    parser.add_argument("--classifier-model-path", default=None)
    parser.add_argument("--regressor-model-path", default=None)
    add_common_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    spec = load_named_dataset(args.dataset, n_samples=args.n_samples, random_state=args.random_state)
    stratify = spec.y if spec.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        spec.X,
        spec.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model_path = args.classifier_model_path if spec.task == "classification" else args.regressor_model_path
    rows: list[dict[str, object]] = []
    checkpoint_dir = Path(args.checkpoint_dir) / spec.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    baseline_start = time.perf_counter()
    if spec.task == "classification":
        baseline = TabICLClassifier(
            n_estimators=args.n_estimators,
            device=device,
            random_state=args.random_state,
            kv_cache=args.kv_cache,
            model_path=model_path,
        )
    else:
        baseline = TabICLRegressor(
            n_estimators=args.n_estimators,
            device=device,
            random_state=args.random_state,
            kv_cache=args.kv_cache,
            model_path=model_path,
        )
    baseline.fit(X_train, y_train)
    metric_name, metric_value, extra = evaluate(spec.task, y_test, baseline.predict(X_test))
    rows.append(
        {
            "dataset": spec.name,
            "task": spec.task,
            "stage": "zero_shot",
            "n_samples": int(spec.X.shape[0]),
            "n_features": int(spec.X.shape[1]),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "model_path": model_path,
            "n_estimators": int(args.n_estimators),
            "epochs": 0,
            "device": device,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "runtime_sec": time.perf_counter() - baseline_start,
            "checkpoint": "",
            **extra,
        }
    )

    finetune_start = time.perf_counter()
    if spec.task == "classification":
        model = FinetunedTabICLClassifier(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=args.n_estimators,
            early_stopping=True,
            patience=max(2, args.epochs),
            time_limit=args.time_limit,
            device=device,
            random_state=args.random_state,
            verbose=False,
            eval_metric="accuracy",
            model_path=model_path,
        )
    else:
        model = FinetunedTabICLRegressor(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=args.n_estimators,
            early_stopping=True,
            patience=max(2, args.epochs),
            time_limit=args.time_limit,
            device=device,
            random_state=args.random_state,
            verbose=False,
            eval_metric="r2",
            model_path=model_path,
        )
    model.fit(X_train, y_train, output_dir=checkpoint_dir)
    metric_name, metric_value, extra = evaluate(spec.task, y_test, model.predict(X_test))
    rows.append(
        {
            "dataset": spec.name,
            "task": spec.task,
            "stage": "finetuned",
            "n_samples": int(spec.X.shape[0]),
            "n_features": int(spec.X.shape[1]),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "model_path": model_path,
            "n_estimators": int(args.n_estimators),
            "epochs": int(args.epochs),
            "device": device,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "runtime_sec": time.perf_counter() - finetune_start,
            "checkpoint": str(checkpoint_dir / "best.ckpt"),
            **extra,
        }
    )

    output_path = resolve_output_path(args.output)
    write_rows(output_path, rows)
    write_json(output_path.with_name(output_path.stem + "_environment.json"), environment_info(device))
    print(f"[experiment1] Wrote {output_path}")


if __name__ == "__main__":
    main()
