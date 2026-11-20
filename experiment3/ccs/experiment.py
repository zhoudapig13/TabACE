from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import fields
from itertools import product
from pathlib import Path

from .datasets import SyntheticSCMConfig, generate_synthetic_scm, split_dataset
from .estimators import make_estimator
from .metrics import ate_error, auuc_score, calibration_ece, pehe, policy_value, qini_score, uplift_at_fraction
from .retrievers import make_retriever
from .utils import environment_info, write_json, write_rows


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    random_states: tuple[int, ...] = (42,)
    test_size: float = 0.3
    context_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    retrievers: tuple[str, ...] = (
        "random",
        "knn",
        "treatment_balanced_random",
        "propensity_nearest",
        "matched_pair",
        "overlap_aware",
    )
    context_estimators: tuple[str, ...] = ("local_context_t_learner",)
    classical_estimators: tuple[str, ...] = ("global_t_learner", "global_s_learner")
    synthetic: SyntheticSCMConfig = SyntheticSCMConfig()
    confounding_strengths: tuple[float, ...] = (1.0,)
    treatment_imbalances: tuple[float, ...] = (0.0,)
    overlap_violations: tuple[float, ...] = (0.0,)
    max_test_eval: int = 120
    regimes: tuple[dict[str, object], ...] = ()


def run_experiment(config: ExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = 0

    synthetic_fields = {field.name for field in fields(SyntheticSCMConfig)}
    if config.regimes:
        scenario_iter = []
        for seed in config.random_states:
            for regime in config.regimes:
                scenario_iter.append((seed, regime))
    else:
        scenario_iter = [
            (
                seed,
                {
                    "name": f"conf{confounding}_imb{imbalance}_overlap{overlap}",
                    "confounding_strength": confounding,
                    "treatment_imbalance": imbalance,
                    "overlap_violation": overlap,
                },
            )
            for seed, confounding, imbalance, overlap in product(
                config.random_states,
                config.confounding_strengths,
                config.treatment_imbalances,
                config.overlap_violations,
            )
        ]

    for seed, regime in scenario_iter:
        scenario_id += 1
        regime_updates = {key: value for key, value in regime.items() if key in synthetic_fields}
        scm = replace(
            config.synthetic,
            random_state=seed,
            **regime_updates,
        )
        dataset = generate_synthetic_scm(scm)
        split = split_dataset(dataset, test_size=config.test_size, random_state=seed + 17)
        if config.max_test_eval > 0 and split.test.n_samples > config.max_test_eval:
            split = split_dataset(dataset, test_size=config.max_test_eval / dataset.n_samples, random_state=seed + 17)
        print(
            "[experiment3] scenario="
            f"{scenario_id}/{len(scenario_iter)} regime={regime.get('name', 'grid')} seed={seed} "
            f"train={split.train.n_samples} test={split.test.n_samples}",
            flush=True,
        )

        scenario = {
            "scenario_id": scenario_id,
            "regime": str(regime.get("name", "grid")),
            "dataset": dataset.name,
            "random_state": seed,
            "n_samples": dataset.n_samples,
            "n_train": split.train.n_samples,
            "n_test": split.test.n_samples,
            "n_features": dataset.n_features,
            "confounding_strength": scm.confounding_strength,
            "treatment_imbalance": scm.treatment_imbalance,
            "overlap_violation": scm.overlap_violation,
            "heterogeneity": scm.heterogeneity,
            "effect_nonlinearity": scm.effect_nonlinearity,
            "train_treated_rate": float(split.train.treatment.mean()),
            "test_treated_rate": float(split.test.treatment.mean()),
            "true_ate": split.test.ate,
        }

        for estimator_name in config.classical_estimators:
            estimator = make_estimator(estimator_name)
            result = estimator.estimate(split, retriever=None, context_size=split.train.n_samples, random_state=seed)
            rows.append(_row(scenario, estimator_name, "global_all_data", split.train.n_samples, result, split))

        for context_size in config.context_sizes:
            for retriever_name in config.retrievers:
                retriever = make_retriever(retriever_name)
                for estimator_name in config.context_estimators:
                    estimator = make_estimator(estimator_name)
                    result = estimator.estimate(split, retriever=retriever, context_size=context_size, random_state=seed)
                    rows.append(_row(scenario, estimator_name, retriever_name, context_size, result, split))

    write_rows(config.output_dir / "synthetic_results.csv", rows)
    write_json(config.output_dir / "environment.json", environment_info())
    return rows


def _row(scenario, estimator_name, retriever_name, context_size, result, split):
    value = policy_value(split.test.mu0, split.test.mu1, result.tau_hat)
    oracle_policy_value = float((split.test.mu1 * (split.test.tau > 0) + split.test.mu0 * (split.test.tau <= 0)).mean())
    return {
        **scenario,
        "estimator": estimator_name,
        "retriever": retriever_name,
        "context_size": int(context_size),
        "pehe": pehe(split.test.tau, result.tau_hat),
        "ate_error": ate_error(split.test.tau, result.ate_hat),
        "policy_value": value,
        "oracle_policy_value": oracle_policy_value,
        "policy_regret": oracle_policy_value - value,
        "uplift_top30": uplift_at_fraction(split.test.mu0, split.test.mu1, result.tau_hat, fraction=0.3),
        "auuc": auuc_score(split.test.outcome, split.test.treatment, result.tau_hat),
        "qini": qini_score(split.test.outcome, split.test.treatment, result.tau_hat),
        "tau_ece": calibration_ece(split.test.tau, result.tau_hat),
        "ate_hat": result.ate_hat,
        "runtime_sec": result.runtime_sec,
        **{f"meta_{key}": value for key, value in result.metadata.items()},
    }
