from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from methods import METHOD_REGISTRY, apply_method  # noqa: E402
from utils import (  # noqa: E402
    add_common_args,
    environment_info,
    load_named_dataset,
    run_tabicl_experiment,
    select_device,
)


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


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TabICL baseline with experiment2 candidate methods.")
    parser.add_argument(
        "--datasets",
        default="breast_cancer,wine,synthetic_classification",
        help="Comma-separated dataset names supported by experiments/utils.py.",
    )
    parser.add_argument(
        "--methods",
        default="identity,row_norm_feature",
        help=f"Comma-separated candidate methods. Available: {', '.join(sorted(METHOD_REGISTRY))}",
    )
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--output", default="innovation_results.csv")
    parser.add_argument(
        "--classifier-model-path",
        default=None,
        help="Optional classifier checkpoint path. If omitted, TabICL uses the default checkpoint.",
    )
    parser.add_argument(
        "--regressor-model-path",
        default=None,
        help="Optional regressor checkpoint path. If omitted, TabICL uses the default checkpoint.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    rows: list[dict[str, object]] = []

    for dataset_name in parse_csv(args.datasets):
        n_samples = args.synthetic_samples if dataset_name.startswith("synthetic_") else None
        base_spec = load_named_dataset(dataset_name, n_samples=n_samples, random_state=args.random_state)

        print(f"[experiment2] Running baseline on dataset={dataset_name}...")
        baseline = run_tabicl_experiment(
            base_spec,
            n_estimators=args.n_estimators,
            device=device,
            random_state=args.random_state,
            test_size=args.test_size,
            kv_cache=args.kv_cache,
            classifier_model_path=args.classifier_model_path,
            regressor_model_path=args.regressor_model_path,
        )
        baseline.update(
            {
                "experiment": "experiment2",
                "method": "baseline",
                "source_dataset": dataset_name,
                "delta_vs_baseline": 0.0,
            }
        )
        rows.append(baseline)

        baseline_metric = float(baseline["metric_value"])
        for method_name in parse_csv(args.methods):
            print(f"[experiment2] Running method={method_name} on dataset={dataset_name}...")
            candidate_spec = apply_method(base_spec, method_name)
            candidate = run_tabicl_experiment(
                candidate_spec,
                n_estimators=args.n_estimators,
                device=device,
                random_state=args.random_state,
                test_size=args.test_size,
                kv_cache=args.kv_cache,
                classifier_model_path=args.classifier_model_path,
                regressor_model_path=args.regressor_model_path,
            )
            candidate.update(
                {
                    "experiment": "experiment2",
                    "method": method_name,
                    "source_dataset": dataset_name,
                    "delta_vs_baseline": float(candidate["metric_value"]) - baseline_metric,
                }
            )
            rows.append(candidate)

    output_path = RESULTS_DIR / args.output
    write_rows(output_path, rows)
    write_json(RESULTS_DIR / "innovation_environment.json", environment_info(device))
    print(f"[experiment2] Wrote {output_path}")


if __name__ == "__main__":
    main()

