"""Controlled weighting ablation using cached TabICL predictions.

All variants share exactly the same per-query alpha, Base prediction, and five
alternative predictions. Dataset-level outer folds protect evaluation labels.
This is a new reconstruction diagnostic, not a reproduction of manuscript scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

import fit_shared_reference as reference
from fit_combiners import sample_weights
from prepare_data import RUN, atomic_json
from run_tabicl import metrics, normalize

OUT = RUN / "results/fixed_gate_weighting_control_bv6f_v1"
REGULARIZATION = 1e-4
SEED = 0
JOBS = [(task, track) for task in ("classification", "regression")
        for track in ("five", "seven_reimplemented")]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.tobytes()).hexdigest()


def freeze_gate(cell, parameters, task):
    """Compute the reference gate once; never recompute it for a comparator."""
    p = np.asarray(parameters)
    if task == "classification":
        agreement = cell["S"]
        weights = softmax(p[:5] + cell["Q"] * p[5:10] + agreement * p[10:15], axis=1)
        alpha = p[-1] * cell["U"] * (weights * cell["Q"] * agreement).sum(axis=1)
    else:
        tau = np.exp(p[-2])
        agreement = np.exp(-cell["D"] / tau)
        weights = softmax(p[:5] + agreement * p[5:10], axis=1)
        aggregate = (weights * cell["M"]).sum(axis=1)
        alpha = p[-1] * (1 - np.exp(-np.abs(aggregate - cell["B"]) / tau)) * (weights * agreement).sum(axis=1)
    assert np.isfinite(alpha).all() and alpha.min() >= -1e-12 and alpha.max() <= 1 + 1e-12
    return alpha, agreement, weights


def design(cell, task):
    features = {}
    if task == "classification":
        features["Q"] = cell["Q"]
    features["S"] = cell["frozen_S"]
    return features


def pack(cells, task):
    result = {
        "X": np.concatenate([c["true_members"] if task == "classification" else c["M"] for c in cells]),
        "B": np.concatenate([c["true_base"] if task == "classification" else c["B"] for c in cells]),
        "alpha": np.concatenate([c["frozen_alpha"] for c in cells]),
        "weights": np.concatenate(sample_weights(cells)),
        "features": {name: np.concatenate([design(c, task)[name] for c in cells])
                     for name in design(cells[0], task)},
    }
    if task == "regression":
        result["Y"] = np.concatenate([c["Y"] for c in cells])
        result["weights"] *= np.concatenate([np.full(len(c["y"]), c["loss_scale"]) for c in cells])
    return result


def objective(parameters, data, names, task):
    prior = parameters[:5]
    logits = np.broadcast_to(prior, data["X"].shape).copy()
    for index, name in enumerate(names):
        logits += data["features"][name] * parameters[5 + index * 5:10 + index * 5]
    weights = softmax(logits, axis=1)
    aggregate = (weights * data["X"]).sum(axis=1)
    prediction = (1 - data["alpha"]) * data["B"] + data["alpha"] * aggregate
    if task == "classification":
        safe = np.clip(prediction, 1e-12, 1)
        loss = np.dot(data["weights"], -np.log(safe))
        derivative = -data["weights"] * data["alpha"] / safe
    else:
        error = prediction - data["Y"]
        loss = np.dot(data["weights"], error ** 2)
        derivative = 2 * data["weights"] * data["alpha"] * error
    logits_gradient = derivative[:, None] * weights * (data["X"] - aggregate[:, None])
    centered = prior - prior.mean()
    loss += REGULARIZATION * np.mean(centered ** 2)
    gradient = [logits_gradient.sum(axis=0) + 2 * REGULARIZATION * centered / 5]
    for index, name in enumerate(names):
        coefficients = parameters[5 + index * 5:10 + index * 5]
        loss += REGULARIZATION * np.mean(coefficients ** 2)
        gradient.append((logits_gradient * data["features"][name]).sum(axis=0)
                        + 2 * REGULARIZATION * coefficients / 5)
    return float(loss), np.concatenate(gradient)


def fit(data, names, task, initial):
    initial = np.asarray(initial, dtype=float)
    initial_loss = objective(initial, data, names, task)[0]
    result = minimize(objective, initial, args=(data, names, task), jac=True,
                      method="L-BFGS-B", bounds=[(-8, 8)] * 5 + [(0, 10)] * (5 * len(names)),
                      options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7})
    assert np.isfinite(result.x).all() and result.fun <= initial_loss + 1e-8
    return dict(parameters=result.x.tolist(), features=names, regularization=REGULARIZATION,
                success=bool(result.success), message=str(result.message), iterations=int(result.nit),
                training_objective=float(result.fun), initial_objective=initial_loss)


def branch_weights(cell, model, task):
    p = np.asarray(model["parameters"])
    logits = np.broadcast_to(p[:5], (len(cell["y"]), 5)).copy()
    for index, name in enumerate(model["features"]):
        logits += design(cell, task)[name] * p[5 + index * 5:10 + index * 5]
    return softmax(logits, axis=1)


def combine(cell, weights, task):
    a = cell["frozen_alpha"]
    if task == "classification":
        aggregate = np.einsum("nm,mnc->nc", weights, cell["bank"])
        return normalize((1 - a[:, None]) * cell["base"] + a[:, None] * aggregate)
    aggregate = (weights * cell["M"]).sum(axis=1)
    return ((1 - a) * cell["B"] + a * aggregate) * cell["source_scale"] + cell["source_mean"]


def worker(task, track):
    start = time.time()
    output = OUT / task / track
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", dict(status="RUNNING", started=stamp(), pid=os.getpid()))
    cells, fold_source = reference.load_cells("tabicl", track, task)
    assert cells
    rows, audits = [], []
    expected = {("classification", "five"): 1512, ("regression", "five"): 735,
                ("classification", "seven_reimplemented"): 1015,
                ("regression", "seven_reimplemented"): 957}[(task, track)]
    assert len(cells) == expected
    print("LOADED", task, track, "datasets", len({c["meta"]["data_id"] for c in cells}), "conditions", len(cells), flush=True)
    for fold in range(5):
        policy_path = RUN / "results/combiners" / track / task / f"fold_{fold}_tabace_tabicl_reimplemented.json"
        policy = json.loads(policy_path.read_text())
        train = [c for c in cells if c["fold"] != fold]
        test = [c for c in cells if c["fold"] == fold]
        train_ids = {int(c["meta"]["data_id"]) for c in train}
        test_ids = {int(c["meta"]["data_id"]) for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert train_ids == set(policy["train_data_ids"])
        assert test_ids == set(policy["test_data_ids"])
        parameters = policy["parameters"]
        for c in cells:
            assert c["bank"].shape[0] == 5
            c["frozen_alpha"], c["frozen_S"], c["reference_W"] = freeze_gate(c, parameters, task)
        data = pack(train, task)
        models = {"Static_learned": fit(data, [], task, np.zeros(5))}
        static = models["Static_learned"]["parameters"]
        if task == "classification":
            models["Confidence_only"] = fit(data, ["Q"], task, static + [0.] * 5)
            initial = models["Confidence_only"]["parameters"] + [0.] * 5
            names = ["Q", "S"]
        else:
            initial = static + [0.] * 5
            names = ["S"]
        models["Full_weighting"] = fit(data, names, task, initial)
        # All fitting ends before opening any held-out labels in metric computation.
        atomic_json(output / f"fold_{fold}_policies.json", dict(
            status="FROZEN_BEFORE_OUTER_TEST_METRICS", seed=SEED, models=models,
            train_data_ids=sorted(train_ids), test_data_ids=sorted(test_ids),
            anchor_policy=str(policy_path), anchor_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            gate="identical per-query alpha from outer-training-only reference policy",
            final_query_labels_used_in_fit=False, calibration_fold_source=fold_source))
        for c in test:
            gate_hash = digest(c["frozen_alpha"])
            original = reference.predict(c, np.asarray(parameters), task)
            reconstructed = combine(c, c["reference_W"], task)
            assert np.allclose(original, reconstructed, rtol=1e-10, atol=1e-10)
            predictions = {"Base": c["base"], "Original_full_reference": original}
            predictions.update({name: combine(c, branch_weights(c, model, task), task)
                                for name, model in models.items()})
            scale = max(float(c["y"].std()), 1e-12) if task == "regression" and track.startswith("seven") else c["meta"]["scale"]
            for name, prediction in predictions.items():
                assert gate_hash == digest(c["frozen_alpha"])
                rows.append(dict(task=task, track=track, data_id=c["meta"]["data_id"],
                    task_id=c["meta"]["task_id"], dataset=c["meta"]["dataset"],
                    fold=fold, family=c["meta"]["family"], rho=c["meta"]["rho"],
                    method=name, n_queries=len(c["y"]), gate_sha256=gate_hash,
                    **metrics(c["y"], prediction, task, scale)))
            audits.append(dict(cache=str(c["path"]), data_id=c["meta"]["data_id"], fold=fold,
                family=c["meta"]["family"], rho=c["meta"]["rho"], bank_sha256=digest(c["bank"]),
                base_sha256=digest(c["base"]), gate_sha256=gate_hash,
                context_roles=c["meta"].get("context_roles"), actual_context_rows=c["meta"].get("context_rows"),
                gate_reference_prediction_max_abs_error=float(np.max(np.abs(original - reconstructed)))))
        pd.DataFrame(rows).to_csv(output / "query_detail.csv", index=False)
        atomic_json(output / "progress.json", dict(completed_folds=fold + 1, total_folds=5,
                    metric_rows=len(rows), elapsed_seconds=time.time() - start))
        print("FOLD_DONE", task, track, fold, {k: round(v["training_objective"], 8) for k, v in models.items()}, flush=True)
    atomic_json(output / "audit.json", dict(status="PASS", cached_conditions=audits,
        exact_same_gate_for_all_methods=True, exact_same_bank_and_base=True,
        dataset_level_outer_train_test_disjoint=True, gate_uses_no_outer_test_labels=True,
        all_variants_calibrated_on_identical_outer_training_conditions=True,
        benchmark_modified=False, backbone_recomputed=False, gpu_count=0))
    atomic_json(output / "status.json", dict(status="COMPLETE", completed=stamp(),
        datasets=len({c["meta"]["data_id"] for c in cells}), conditions=len(cells),
        elapsed_seconds=time.time() - start))
    print("COMPLETE", task, track, round(time.time() - start, 2), flush=True)


def aggregate():
    frame = pd.concat([pd.read_csv(OUT / task / track / "query_detail.csv") for task, track in JOBS], ignore_index=True)
    frame.to_csv(OUT / "query_detail.csv", index=False)
    all_summaries = []
    for (task, track, method), group in frame.groupby(["task", "track", "method"], sort=False):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        for split, data in [("clean", group[group.rho == 0]), ("nonzero_strengths", group[group.rho > 0])]:
            summary = dict(task=task, track=track, split=split, method=method,
                datasets=int(data.data_id.nunique()), conditions=len(data), queries=int(data.n_queries.sum()))
            for key in keys:
                summary[key] = float(data.groupby("data_id")[key].mean().mean())
            all_summaries.append(summary)
    summary = pd.DataFrame(all_summaries)
    summary.to_csv(OUT / "averages.csv", index=False)
    differences = []
    rng = np.random.default_rng(SEED)
    for (task, track), part in frame[frame.rho > 0].groupby(["task", "track"]):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        comparators = ["Static_learned", "Confidence_only"] if task == "classification" else ["Static_learned"]
        means = part.groupby(["data_id", "method"])[keys].mean()
        for comparator in comparators:
            for key in keys:
                full = means.xs("Full_weighting", level="method")[key]
                other = means.xs(comparator, level="method")[key]
                delta = (full - other).to_numpy()
                bootstrap = delta[rng.integers(0, len(delta), size=(4000, len(delta)))].mean(axis=1)
                orientation = -1 if key == "nrmse" else 1
                signed = delta * orientation
                draws = rng.choice([-1, 1], size=(10000, len(delta)))
                pvalue = (1 + np.sum(np.abs((draws * delta).mean(axis=1)) >= abs(delta.mean()))) / 10001
                differences.append(dict(task=task, track=track, metric=key, comparator=comparator,
                    datasets=len(delta), full_minus_comparator=float(delta.mean()),
                    dataset_bootstrap_ci95_low=float(np.quantile(bootstrap, .025)),
                    dataset_bootstrap_ci95_high=float(np.quantile(bootstrap, .975)),
                    improved_datasets=int((signed > 1e-12).sum()), worse_datasets=int((signed < -1e-12).sum()),
                    paired_dataset_signflip_p=float(pvalue), seed=SEED,
                    uncertainty_unit="datasets, not random-seed repetitions"))
    comparison = pd.DataFrame(differences)
    pvalues = comparison.paired_dataset_signflip_p.to_numpy()
    order = np.argsort(pvalues)
    corrected = np.maximum.accumulate(np.minimum(1, (len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    restored = np.empty_like(corrected)
    restored[order] = corrected
    comparison["holm_corrected_p"] = restored
    comparison.to_csv(OUT / "paired_dataset_comparisons.csv", index=False)
    lines = ["# Fixed-gate weighting control — TabICL-v2", "",
        "New cached-bank reconstruction diagnostic; these are not manuscript reproduction scores.",
        "Five alternative contexts plus Base; all weighting variants use exactly identical per-query alpha.",
        "Five dataset-level outer folds; single seed 0; no new backbone inference or benchmark modification.",
        "Classification compares static learned, confidence-only, and full confidence/agreement weighting.",
        "Regression compares static learned and full agreement weighting; its paper formula has no standalone confidence term.",
        "The gate is an identical frozen reference gate and retains its evidence signals for every method.",
        "Only fusion weighting changes. Clean is evaluated separately and is included in calibration with the existing track weights.",
        "", "| Task | Track | Datasets | Method | Metric 1 | Metric 2 |", "|---|---|---:|---|---:|---:|"]
    for row in summary[summary.split == "nonzero_strengths"].to_dict("records"):
        keys = ("accuracy", "auc") if row["task"] == "classification" else ("r2", "nrmse")
        lines.append(f'| {row["task"]} | {row["track"]} | {row["datasets"]} | {row["method"]} | {row[keys[0]]:.6f} | {row[keys[1]]:.6f} |')
    lines += ["", "Classification: Acc / AUC. Regression: R² / nRMSE.",
              "Paired dataset bootstrap intervals and Holm-corrected sign-flip tests are in paired_dataset_comparisons.csv.",
              "No outcome ordering was enforced, and no final evaluation label was used to fit or select weights."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    atomic_json(OUT / "completion.json", dict(status="COMPLETE", completed=stamp(),
        result_markdown=str(OUT / "RESULTS.md"), result_csv=str(OUT / "averages.csv"),
        comparison_csv=str(OUT / "paired_dataset_comparisons.csv"), metric_rows=len(frame),
        main_rebuttal_documents_modified=False, gpu_count=0))
    print("ALL_COMPLETE", flush=True)
    print((OUT / "RESULTS.md").read_text(), flush=True)


def launch_all():
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "plan.json", dict(declared=stamp(), seed=SEED,
        task="isolate weighting with exact identical per-query alpha and alternative predictions",
        jobs=JOBS, backbone="TabICL-v2", bank="existing reconstructed Shared five-context bank",
        original_paper_reproduction=False, no_benchmark_modification=True,
        outer_fold_count=5, final_evaluation_labels_used_for_fit=False,
        gate_anchor="existing dataset-OOF reference policies, fixed independently of comparator weights",
        optimizer="L-BFGS-B", maximum_iterations=200, regularization=REGULARIZATION,
        classification_variants=["Static_learned", "Confidence_only", "Full_weighting"],
        regression_variants=["Static_learned", "Full_weighting"],
        no_performance_target_or_test_label_selection=True, gpu_count=0))
    children = []
    for task, track in JOBS:
        handle = (OUT / f"{task}_{track}.log").open("w")
        child = subprocess.Popen([sys.executable, "-u", __file__, "--task", task, "--track", track],
                                 stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
        children.append((task, track, child, handle))
    atomic_json(OUT / "workers.json", dict(supervisor_pid=os.getpid(),
        workers=[dict(task=t, track=k, pid=p.pid) for t, k, p, _ in children]))
    failed = []
    for task, track, child, handle in children:
        code = child.wait()
        handle.close()
        if code:
            failed.append(dict(task=task, track=track, returncode=code))
    if failed:
        atomic_json(OUT / "completion.json", dict(status="FAILED", failed=failed))
        raise RuntimeError(f"Workers failed: {failed}")
    aggregate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--track", choices=["five", "seven_reimplemented"])
    args = parser.parse_args()
    if args.task:
        worker(args.task, args.track)
    else:
        launch_all()
