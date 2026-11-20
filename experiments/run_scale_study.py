from __future__ import annotations

import argparse

from utils import (
    RESULTS_DIR,
    add_common_args,
    environment_info,
    load_named_dataset,
    parse_scales,
    run_tabicl_experiment,
    select_device,
    write_json,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabICL experiments while increasing dataset size.")
    parser.add_argument("--dataset", default="synthetic_classification")
    parser.add_argument("--scales", default="100,300,1000,3000")
    parser.add_argument("--output", default="scale_study_results.csv")
    add_common_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    rows = []
    for n_samples in parse_scales(args.scales):
        print(f"Running scale n={n_samples} on {args.dataset}...")
        spec = load_named_dataset(args.dataset, n_samples=n_samples, random_state=args.random_state)
        rows.append(
            run_tabicl_experiment(
                spec,
                n_estimators=args.n_estimators,
                device=device,
                random_state=args.random_state,
                test_size=args.test_size,
                kv_cache=args.kv_cache,
            )
        )

    output_path = RESULTS_DIR / args.output
    write_rows(output_path, rows)
    write_json(RESULTS_DIR / "scale_study_environment.json", environment_info(device))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
