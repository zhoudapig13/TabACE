from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    RESULTS_DIR,
    add_common_args,
    environment_info,
    load_named_dataset,
    run_tabicl_experiment,
    select_device,
    write_json,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one reproducible TabICL experiment.")
    parser.add_argument("--dataset", default="synthetic_classification")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--output", default="single_dataset_results.csv")
    add_common_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    spec = load_named_dataset(args.dataset, n_samples=args.n_samples, random_state=args.random_state)
    result = run_tabicl_experiment(
        spec,
        n_estimators=args.n_estimators,
        device=device,
        random_state=args.random_state,
        test_size=args.test_size,
        kv_cache=args.kv_cache,
    )

    output_path = RESULTS_DIR / args.output
    write_rows(output_path, [result])
    write_json(RESULTS_DIR / "single_dataset_environment.json", environment_info(device))

    print(f"Wrote {output_path}")
    print(
        f"{result['dataset']} | {result['task']} | n={result['n_samples']} | "
        f"{result['metric_name']}={result['metric_value']:.4f} | "
        f"runtime={result['runtime_sec']:.2f}s"
    )


if __name__ == "__main__":
    main()
