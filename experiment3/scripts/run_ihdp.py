from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccs.estimators import make_estimator  # noqa: E402
from ccs.metrics import ate_error, pehe, policy_value, uplift_at_fraction  # noqa: E402
from ccs.real_data import load_ihdp_split  # noqa: E402
from ccs.retrievers import make_retriever  # noqa: E402
from ccs.types import CausalDataset, CausalSplit  # noqa: E402
from ccs.utils import environment_info, write_json, write_rows  # noqa: E402


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


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


def result_row(split: CausalSplit, result, estimator: str, retriever: str, context_size: int, realization: int) -> dict[str, object]:
    value = policy_value(split.test.mu0, split.test.mu1, result.tau_hat)
    oracle_policy_value = float(np.mean(np.maximum(split.test.mu0, split.test.mu1)))
    return {
        "dataset": "ihdp",
        "realization": realization,
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
    parser = argparse.ArgumentParser(description="Run initial IHDP context-selection validation.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "ihdp"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "ihdp_initial"))
    parser.add_argument("--realizations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--context-sizes", default="16,32,64,128,256")
    parser.add_argument(
        "--retrievers",
        default="random,knn,treatment_balanced_random,propensity_nearest,matched_pair,matched_pair_diverse,overlap_aware",
    )
    parser.add_argument("--context-estimator", default="local_context_t_learner")
    parser.add_argument("--classical-estimators", default="global_t_learner,global_s_learner,global_x_learner,global_dr_learner")
    parser.add_argument("--max-test-eval", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for realization in [int(item) for item in parse_csv(args.realizations)]:
        split = limit_test(load_ihdp_split(args.data_dir, realization=realization), args.max_test_eval, seed=realization)
        print(
            f"[experiment3] IHDP realization={realization} train={split.train.n_samples} test={split.test.n_samples}",
            flush=True,
        )
        for estimator_name in parse_csv(args.classical_estimators):
            estimator = make_estimator(estimator_name)
            result = estimator.estimate(split, retriever=None, context_size=split.train.n_samples, random_state=realization)
            rows.append(result_row(split, result, estimator_name, "global_all_data", split.train.n_samples, realization))
        for context_size in [int(item) for item in parse_csv(args.context_sizes)]:
            for retriever_name in parse_csv(args.retrievers):
                estimator = make_estimator(args.context_estimator)
                retriever = make_retriever(retriever_name)
                result = estimator.estimate(split, retriever=retriever, context_size=context_size, random_state=realization)
                rows.append(result_row(split, result, args.context_estimator, retriever_name, context_size, realization))
    write_rows(output_dir / "ihdp_results.csv", rows)
    write_json(output_dir / "environment.json", {**environment_info(), "elapsed_sec": time.perf_counter() - start})
    print(f"[experiment3] wrote {len(rows)} rows to {output_dir / 'ihdp_results.csv'}")


if __name__ == "__main__":
    main()
