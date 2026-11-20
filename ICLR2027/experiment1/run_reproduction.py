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


def parse_dataset_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_output_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return RESULTS_DIR / path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run a clean TabICL reproduction baseline.")
    parser.add_argument(
        "--datasets",
        default="breast_cancer,wine,diabetes,synthetic_classification,synthetic_regression",
        help="Comma-separated dataset names supported by experiments/utils.py.",
    )
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--output", default="reproduction_results.csv")
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

    for name in parse_dataset_names(args.datasets):
        n_samples = args.synthetic_samples if name.startswith("synthetic_") else None
        print(f"[experiment1] Running TabICL baseline on dataset={name}...")
        spec = load_named_dataset(name, n_samples=n_samples, random_state=args.random_state)
        result = run_tabicl_experiment(
            spec,
            n_estimators=args.n_estimators,
            device=device,
            random_state=args.random_state,
            test_size=args.test_size,
            kv_cache=args.kv_cache,
            classifier_model_path=args.classifier_model_path,
            regressor_model_path=args.regressor_model_path,
        )
        result["experiment"] = "experiment1_reproduction"
        rows.append(result)

    output_path = resolve_output_path(args.output)
    write_rows(output_path, rows)
    write_json(RESULTS_DIR / "reproduction_environment.json", environment_info(device))
    print(f"[experiment1] Wrote {output_path}")


if __name__ == "__main__":
    main()
