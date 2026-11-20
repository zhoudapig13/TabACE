from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccs.estimators import make_estimator  # noqa: E402
from ccs.metrics import ate_error, pehe, policy_value, uplift_at_fraction  # noqa: E402
from ccs.real_data import load_acic2016_split  # noqa: E402
from ccs.retrievers import make_retriever  # noqa: E402
from ccs.types import CausalDataset, CausalSplit  # noqa: E402
from ccs.utils import environment_info, write_json, write_rows  # noqa: E402


ACIC2016_BASE_URL = (
    "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
    "causallib/datasets/data/acic_challenge_2016"
)


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def ensure_acic2016_files(data_dir: Path, conditions: tuple[int, ...]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    required = ["README.md", "LICENSE", "x.csv"] + [f"zymu_{condition}.csv" for condition in conditions]
    for name in required:
        path = data_dir / name
        if path.exists():
            continue
        url = f"{ACIC2016_BASE_URL}/{name}"
        print(f"[experiment3] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)


def subset_dataset(dataset: CausalDataset, indices: np.ndarray, name_suffix: str) -> CausalDataset:
    return CausalDataset(
        name=f"{dataset.name}_{name_suffix}",
        X=dataset.X[indices],
        treatment=dataset.treatment[indices],
        outcome=dataset.outcome[indices],
        mu0=dataset.mu0[indices],
        mu1=dataset.mu1[indices],
        tau=dataset.tau[indices],
        propensity=dataset.propensity[indices],
    )


def limit_test(split: CausalSplit, max_test_eval: int, seed: int) -> CausalSplit:
    if max_test_eval <= 0 or split.test.n_samples <= max_test_eval:
        return split
    rng = np.random.default_rng(seed)
    idx = rng.choice(split.test.n_samples, size=max_test_eval, replace=False)
    return CausalSplit(train=split.train, test=subset_dataset(split.test, idx, "eval_subset"), random_state=split.random_state)


def result_row(
    split: CausalSplit,
    result,
    estimator: str,
    retriever: str,
    context_size: int,
    condition: int,
    split_seed: int,
) -> dict[str, object]:
    value = policy_value(split.test.mu0, split.test.mu1, result.tau_hat)
    oracle_policy_value = float(np.mean(np.maximum(split.test.mu0, split.test.mu1)))
    return {
        "dataset": "acic2016",
        "condition": condition,
        "split_seed": split_seed,
        "estimator": estimator,
        "retriever": retriever,
        "context_size": int(context_size),
        "n_train": split.train.n_samples,
        "n_test": split.test.n_samples,
        "n_features": split.train.n_features,
        "train_treated_rate": float(np.mean(split.train.treatment)),
        "test_treated_rate": float(np.mean(split.test.treatment)),
        "true_ate": split.test.ate,
        "pehe": pehe(split.test.tau, result.tau_hat),
        "ate_error": ate_error(split.test.tau, result.ate_hat),
        "policy_value": value,
        "oracle_policy_value": oracle_policy_value,
        "policy_regret": oracle_policy_value - value,
        "uplift_top30": uplift_at_fraction(split.test.mu0, split.test.mu1, result.tau_hat, fraction=0.3),
        "ate_hat": result.ate_hat,
        "runtime_sec": result.runtime_sec,
        **{f"meta_{key}": value for key, value in result.metadata.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACIC 2016 external context-selection validation.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "acic2016"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "acic2016_initial"))
    parser.add_argument("--conditions", default="1")
    parser.add_argument("--split-seeds", default="0,1,2")
    parser.add_argument("--context-sizes", default="16,64,256")
    parser.add_argument("--retrievers", default="random,knn,matched_pair,matched_pair_diverse,overlap_aware")
    parser.add_argument("--context-estimator", default="local_context_t_learner")
    parser.add_argument("--classical-estimators", default="global_t_learner,global_dr_learner")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=1600)
    parser.add_argument("--max-test-eval", type=int, default=128)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    conditions = tuple(int(item) for item in parse_csv(args.conditions))
    split_seeds = tuple(int(item) for item in parse_csv(args.split_seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    if not args.no_download:
        ensure_acic2016_files(data_dir, conditions)

    rows = []
    start = time.perf_counter()
    for condition in conditions:
        for split_seed in split_seeds:
            split = load_acic2016_split(
                data_dir,
                condition=condition,
                test_size=args.test_size,
                random_state=split_seed,
                max_samples=args.max_samples,
            )
            split = limit_test(split, args.max_test_eval, seed=split_seed)
            print(
                f"[experiment3] ACIC2016 condition={condition} split_seed={split_seed} "
                f"train={split.train.n_samples} test={split.test.n_samples}",
                flush=True,
            )
            for estimator_name in parse_csv(args.classical_estimators):
                estimator = make_estimator(estimator_name)
                result = estimator.estimate(split, retriever=None, context_size=split.train.n_samples, random_state=split_seed)
                rows.append(result_row(split, result, estimator_name, "global_all_data", split.train.n_samples, condition, split_seed))
            for context_size in [int(item) for item in parse_csv(args.context_sizes)]:
                for retriever_name in parse_csv(args.retrievers):
                    estimator = make_estimator(args.context_estimator)
                    retriever = make_retriever(retriever_name)
                    result = estimator.estimate(split, retriever=retriever, context_size=context_size, random_state=split_seed)
                    rows.append(result_row(split, result, args.context_estimator, retriever_name, context_size, condition, split_seed))

    write_rows(output_dir / "acic2016_results.csv", rows)
    write_json(
        output_dir / "environment.json",
        {
            **environment_info(),
            "elapsed_sec": time.perf_counter() - start,
            "source": ACIC2016_BASE_URL,
            "conditions": list(conditions),
            "split_seeds": list(split_seeds),
            "max_samples": args.max_samples,
            "max_test_eval": args.max_test_eval,
        },
    )
    print(f"[experiment3] wrote {len(rows)} rows to {output_dir / 'acic2016_results.csv'}")


if __name__ == "__main__":
    main()
