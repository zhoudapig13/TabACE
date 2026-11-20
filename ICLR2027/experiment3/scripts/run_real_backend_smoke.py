from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ccs.datasets import SyntheticSCMConfig, generate_synthetic_scm, split_dataset  # noqa: E402
from ccs.metrics import ate_error, pehe, policy_value  # noqa: E402
from ccs.real_data import load_ihdp_split  # noqa: E402
from ccs.retrievers import make_retriever  # noqa: E402
from ccs.types import CausalDataset, CausalSplit  # noqa: E402
from ccs.utils import write_json, write_rows  # noqa: E402


def subset_dataset(dataset: CausalDataset, indices: np.ndarray, suffix: str) -> CausalDataset:
    return CausalDataset(
        name=f"{dataset.name}_{suffix}",
        X=dataset.X[indices],
        treatment=dataset.treatment[indices],
        outcome=dataset.outcome[indices],
        mu0=dataset.mu0[indices],
        mu1=dataset.mu1[indices],
        tau=dataset.tau[indices],
        propensity=dataset.propensity[indices],
    )


def limit_test(split: CausalSplit, max_test_eval: int, seed: int) -> CausalSplit:
    if split.test.n_samples <= max_test_eval:
        return split
    rng = np.random.default_rng(seed)
    idx = rng.choice(split.test.n_samples, size=max_test_eval, replace=False)
    return CausalSplit(train=split.train, test=subset_dataset(split.test, idx, "tabicl_smoke"), random_state=seed)


def fit_predict_tabicl(X: np.ndarray, y: np.ndarray, x_query: np.ndarray, seed: int, device: str) -> float:
    from tabicl import TabICLRegressor

    if X.shape[0] < 2:
        return float(np.mean(y)) if y.size else 0.0
    model = TabICLRegressor(
        n_estimators=1,
        batch_size=8,
        device=device,
        random_state=seed,
        verbose=False,
        allow_auto_download=True,
    )
    model.fit(X, y)
    return float(model.predict(x_query.reshape(1, -1))[0])


def probe_tabpfn_regressor(split: CausalSplit, dataset_name: str, seed: int, device: str) -> dict[str, object]:
    """One tiny TabPFN model-level check; records license/download blockers instead of hiding them."""
    try:
        from tabpfn import TabPFNRegressor

        n_fit = min(16, split.train.n_samples)
        n_pred = min(2, split.test.n_samples)
        model = TabPFNRegressor(
            n_estimators=1,
            device=device,
            random_state=seed,
            show_progress_bar=False,
        )
        start = time.perf_counter()
        model.fit(split.train.X[:n_fit], split.train.outcome[:n_fit])
        pred = model.predict(split.test.X[:n_pred])
        return {
            "dataset": dataset_name,
            "backend": "tabpfn_regressor",
            "smoke_status": "fit_predict_ok",
            "n_fit": n_fit,
            "n_pred": n_pred,
            "prediction_mean": float(np.mean(pred)),
            "runtime_sec": time.perf_counter() - start,
        }
    except Exception as exc:
        message = str(exc).replace("\n", " ")
        return {
            "dataset": dataset_name,
            "backend": "tabpfn_regressor",
            "smoke_status": "fit_predict_blocked",
            "error_type": exc.__class__.__name__,
            "error_message": message[:500],
        }


def estimate_with_retriever(split: CausalSplit, retriever_name: str, context_size: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    retriever = make_retriever(retriever_name)
    tau_hat = np.zeros(split.test.n_samples)
    start = time.perf_counter()
    for query_index in range(split.test.n_samples):
        context = retriever.select(split, query_index, context_size, rng)
        treated = context.treatment > 0.5
        control = ~treated
        if np.sum(treated) < 2 or np.sum(control) < 2:
            treated = split.train.treatment > 0.5
            control = ~treated
            X_context = split.train.X
            y_context = split.train.outcome
        else:
            X_context = context.X
            y_context = context.outcome
        mu1 = fit_predict_tabicl(X_context[treated], y_context[treated], split.test.X[query_index], seed + query_index, device)
        mu0 = fit_predict_tabicl(X_context[control], y_context[control], split.test.X[query_index], seed + 1000 + query_index, device)
        tau_hat[query_index] = mu1 - mu0
    runtime = time.perf_counter() - start
    return tau_hat, runtime


def row(dataset_name: str, split: CausalSplit, retriever: str, context_size: int, tau_hat: np.ndarray, runtime: float) -> dict[str, object]:
    value = policy_value(split.test.mu0, split.test.mu1, tau_hat)
    oracle = float(np.mean(np.maximum(split.test.mu0, split.test.mu1)))
    return {
        "dataset": dataset_name,
        "backend": "tabicl_regressor",
        "retriever": retriever,
        "context_size": context_size,
        "n_train": split.train.n_samples,
        "n_test": split.test.n_samples,
        "pehe": pehe(split.test.tau, tau_hat),
        "ate_error": ate_error(split.test.tau, float(np.mean(tau_hat))),
        "policy_value": value,
        "policy_regret": oracle - value,
        "runtime_sec": runtime,
    }


def run_dataset(dataset_name: str, split: CausalSplit, retrievers: tuple[str, ...], context_size: int, seed: int, device: str) -> list[dict[str, object]]:
    rows = []
    for retriever in retrievers:
        print(f"[experiment3] real backend smoke dataset={dataset_name} retriever={retriever}", flush=True)
        tau_hat, runtime = estimate_with_retriever(split, retriever, context_size, seed, device)
        rows.append(row(dataset_name, split, retriever, context_size, tau_hat, runtime))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny real TabICL backend smoke for context retrieval.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "real_backend_smoke"))
    parser.add_argument("--ihdp-data-dir", default=str(ROOT / "data" / "ihdp"))
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--max-test-eval", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--retrievers", default="random,knn,matched_pair,matched_pair_diverse")
    parser.add_argument("--probe-tabpfn", action="store_true", help="Also attempt a tiny TabPFNRegressor fit/predict probe.")
    args = parser.parse_args()

    retrievers = tuple(item.strip() for item in args.retrievers.split(",") if item.strip())
    output_dir = Path(args.output_dir)
    synthetic = split_dataset(
        generate_synthetic_scm(
            SyntheticSCMConfig(
                n_samples=120,
                n_features=8,
                confounding_strength=1.5,
                overlap_violation=0.5,
                random_state=0,
            )
        ),
        test_size=0.25,
        random_state=7,
    )
    synthetic = limit_test(synthetic, args.max_test_eval, seed=0)
    rows = run_dataset("synthetic_scm", synthetic, retrievers, args.context_size, seed=0, device=args.device)
    probes = []
    if args.probe_tabpfn:
        probes.append(probe_tabpfn_regressor(synthetic, "synthetic_scm", seed=0, device=args.device))

    try:
        ihdp = limit_test(load_ihdp_split(args.ihdp_data_dir, realization=0), args.max_test_eval, seed=1)
        rows.extend(run_dataset("ihdp", ihdp, retrievers, args.context_size, seed=10, device=args.device))
        if args.probe_tabpfn:
            probes.append(probe_tabpfn_regressor(ihdp, "ihdp", seed=10, device=args.device))
    except Exception as exc:
        rows.append({"dataset": "ihdp", "backend": "tabicl_regressor", "error": repr(exc)})
        if args.probe_tabpfn:
            probes.append({"dataset": "ihdp", "backend": "tabpfn_regressor", "smoke_status": "data_load_blocked", "error": repr(exc)})

    write_rows(output_dir / "real_backend_smoke_results.csv", rows)
    if probes:
        write_rows(output_dir / "real_backend_model_probes.csv", probes)
    write_json(
        output_dir / "real_backend_smoke_environment.json",
        {
            "backend": "tabicl_regressor",
            "device": args.device,
            "context_size": args.context_size,
            "max_test_eval": args.max_test_eval,
            "retrievers": list(retrievers),
            "probe_tabpfn": args.probe_tabpfn,
        },
    )
    print(f"[experiment3] wrote {output_dir / 'real_backend_smoke_results.csv'}")


if __name__ == "__main__":
    main()
