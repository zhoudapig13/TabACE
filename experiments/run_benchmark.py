from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Run TabICL on multiple built-in datasets.")
    parser.add_argument(
        "--datasets",
        default="breast_cancer,wine,diabetes,synthetic_classification,synthetic_regression",
    )
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument(
        "--classifier-model-path",
        default=None,
        help="Optional classifier checkpoint path. If omitted, TabICL downloads/uses the default pretrained checkpoint.",
    )
    parser.add_argument(
        "--regressor-model-path",
        default=None,
        help="Optional regressor checkpoint path. If omitted, TabICL downloads/uses the default pretrained checkpoint.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    rows = []
    for name in [part.strip() for part in args.datasets.split(",") if part.strip()]:
        n_samples = args.synthetic_samples if name.startswith("synthetic_") else None
        print(f"Running dataset={name}...")
        spec = load_named_dataset(name, n_samples=n_samples, random_state=args.random_state)
        rows.append(
            run_tabicl_experiment(
                spec,
                n_estimators=args.n_estimators,
                device=device,
                random_state=args.random_state,
                test_size=args.test_size,
                kv_cache=args.kv_cache,
                classifier_model_path=args.classifier_model_path,
                regressor_model_path=args.regressor_model_path,
            )
        )

    output_path = RESULTS_DIR / args.output
    write_rows(output_path, rows)
    write_json(RESULTS_DIR / "benchmark_environment.json", environment_info(device))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
