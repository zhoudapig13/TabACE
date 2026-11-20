"""Single-seed reconstruction of the manuscript rule on frozen 256-row banks.

No auxiliary predictor, output calibration, enlarged gate, or target-score
selection is used. The upstream benchmark was developed using query feedback
and contains semisynthetic regression tasks; this is not an independent test
or an exact recovery of the original paper experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import torch

from fit_combiners import fold_mapping
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import metrics
import tabace_context_reproduction_v28 as banks
import tabace_fusion_v28 as rule

OUT = RUN / "results/tabace_paper_rule_reconstruction_v77"
BLOCKS = ("clean", "five", "seven")
REGULARIZATIONS = (0.0001, 0.001, 0.01)
EXPECTED = {
    "classification": {"clean": 72, "five": 1440, "seven": 980},
    "regression": {"clean": 35, "five": 700, "seven": 924},
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    if (OUT / "plan.json").exists():
        return
    snapshot = OUT / "implementation_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    code = {}
    for name in [Path(__file__).name, "tabace_fusion_v28.py", "tabace_context_reproduction_v28.py",
                 "shared_reference.py", "specific_reference.py", "fit_combiners.py", "run_tabicl.py"]:
        original = Path(__file__).with_name(name)
        destination = snapshot / name
        destination.write_bytes(original.read_bytes())
        code[name] = file_sha(destination)
    manifest = []
    for task in EXPECTED:
        mapping, origin = fold_mapping(task, "five")
        atomic_json(OUT / "folds" / f"{task}.json", {"mapping": mapping, "origin": origin})
        for role in ("validation", "query"):
            for block, expected in EXPECTED[task].items():
                files = sorted((banks.OUT / "banks" / role / "256" / task / block).glob("*/*.npz"))
                assert len(files) == expected, (role, task, block, len(files), expected)
                for path in files:
                    meta = json.loads(path.with_suffix(".json").read_text())
                    assert meta["actual_context"] <= 256
                    assert len(meta["shared_roles"]) == 5 and len(meta["specific_additional_roles"]) == 5
                    assert max(meta["context_rows"]) <= 256
                    manifest.append({"path": str(path), "sha256": file_sha(path),
                                     "metadata_sha256": file_sha(path.with_suffix(".json"))})
    atomic_json(OUT / "cache_manifest.json", manifest)
    atomic_json(OUT / "plan.json", {
        "status": "FROZEN_BEFORE_POLICY_FITTING", "seed": 0, "context_cap": 256,
        "n_estimators_per_branch": 1, "Shared_branches": 5,
        "Specific_total_branches": 10, "Specific_extra_branches": 5,
        "Specific_registry": "reconstructed five registered family-directed candidates plus Shared; not full uncapped historical registry",
        "fusion": "manuscript confidence/agreement weighting and Base gate; gain=1; alpha_max in [0,1]",
        "Shared_policy_scope": "one pooled policy per task and outer dataset fold across Clean/five/seven",
        "Specific_policy_scope": "one policy per declared family and outer dataset fold; clean falls back to Shared",
        "calibration": "held-out TRAIN pseudo-target banks on other datasets only",
        "outer_folds": 5, "fold_mapping": "historical five-track data_id map, shared across all blocks",
        "inner_selection": "smallest remaining fold; macro source CE or scaled MSE; includes Base/Shared fallback",
        "regularization_grid": list(REGULARIZATIONS), "gate_gain": 1,
        "classification_gate": "normalized Base entropy",
        "auxiliary_supervised_predictor": False, "posthoc_output_calibration": False,
        "backbone_weight_finetuning": False, "new_query_score_hyperparameter_selection": False,
        "paper_scores_used_for_selection": False, "benchmark_modified_this_run": False,
        "Base_predictions_modified": False, "upstream_development_benchmark": True,
        "upstream_query_feedback_used_for_benchmark_design": True,
        "semisynthetic_regression_blocks": ["clean", "five"],
        "original_paper_replication": False, "independent_test_claim": False,
        "upstream_numerical_replay_caveat": "cached member probabilities reused; no claim of a new full backbone replay",
        "code_sha256": code, "cache_manifest_sha256": file_sha(OUT / "cache_manifest.json"),
        "expected_conditions": EXPECTED, "Markdown_written": False,
    })


def load(role, task):
    mapping = {int(k): int(v) for k, v in json.loads((OUT / "folds" / f"{task}.json").read_text())["mapping"].items()}
    all_cells = []
    for block in BLOCKS:
        cells = rule.load(role, 256, task, block)
        for cell in cells:
            data_id = int(cell["meta"]["data_id"])
            cell["fold"] = mapping[data_id]
            cell["meta"]["block"] = block
            cell["meta"]["data_id"] = data_id
        all_cells.extend(cells)
    return all_cells


def source_loss(cells, policy, task):
    assert cells and all(c["role"] == "validation" for c in cells)
    count = policy.get("m", 5)
    data = rule.training_tensors(cells, count)
    with torch.no_grad():
        if policy["kind"] == "base":
            prediction = data["true_base"] if task == "classification" else data["B"]
        else:
            prediction = rule.mapping(torch.tensor(policy["parameters"], dtype=torch.float64),
                                      data, task, count, 1.)
        losses = -prediction.clamp(1e-12, 1).log() if task == "classification" else (prediction - data["Y"]).square()
        return float((losses * data["weights"]).sum())


def select(cells, task, count, inner_fold, fallback_inner, fallback_final):
    train = [c for c in cells if c["fold"] != inner_fold]
    validation = [c for c in cells if c["fold"] == inner_fold]
    assert train and validation
    train_ids = {c["meta"]["data_id"] for c in train}
    validation_ids = {c["meta"]["data_id"] for c in validation}
    assert train_ids.isdisjoint(validation_ids)
    best_loss = source_loss(validation, fallback_inner, task)
    best_regularization, best_inner = None, fallback_inner
    trials = [{"kind": "fallback", "source_validation_loss": best_loss}]
    for regularization in REGULARIZATIONS:
        candidate = rule.fit(train, task, count, 1., regularization)
        loss = source_loss(validation, candidate, task)
        trials.append({"regularization": regularization, "source_validation_loss": loss,
                       "optimization": candidate["optimization"]})
        if loss < best_loss - 1e-10:
            best_loss, best_regularization, best_inner = loss, regularization, candidate
    final = fallback_final if best_regularization is None else rule.fit(cells, task, count, 1., best_regularization)
    return final, best_inner, {"inner_fold": inner_fold, "inner_train_data_ids": sorted(train_ids),
        "inner_validation_data_ids": sorted(validation_ids), "trials": trials,
        "selected_regularization": best_regularization, "selected_source_loss": best_loss}


def worker(task, outer_fold):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    rule.GATE_VARIANT = "entropy"
    destination = OUT / "fold_runs" / task / f"fold_{outer_fold}"
    if (destination / "completion.json").exists():
        print("CACHED", task, outer_fold, flush=True)
        return
    started = time.time()
    source = load("validation", task)
    training = [c for c in source if c["fold"] != outer_fold]
    excluded_ids = {c["meta"]["data_id"] for c in source if c["fold"] == outer_fold}
    training_ids = {c["meta"]["data_id"] for c in training}
    assert excluded_ids.isdisjoint(training_ids)
    inner_fold = min(c["fold"] for c in training)
    if (destination / "frozen_policy.json").exists():
        saved = json.loads((destination / "frozen_policy.json").read_text())
        shared, specific = saved["Shared"], saved["Specific"]
    else:
        shared, shared_inner, shared_selection = select(training, task, 5, inner_fold,
                                                      {"kind": "base"}, {"kind": "base"})
        print("SHARED_FROZEN", task, outer_fold, shared_selection["selected_regularization"], flush=True)
        specific, details = {}, {}
        for block in ("five", "seven"):
            families = sorted({c["meta"]["family"] for c in training if c["meta"]["block"] == block})
            for family in families:
                subset = [c for c in training if c["meta"]["block"] == block and c["meta"]["family"] == family]
                key = block + "/" + family
                specific[key], _, details[key] = select(subset, task, 10, inner_fold, shared_inner, shared)
                print("SPECIFIC_FROZEN", task, outer_fold, key, details[key]["selected_regularization"], flush=True)
        atomic_json(destination / "frozen_policy.json", {
            "status": "FROZEN_BEFORE_THIS_RUN_QUERY_EVALUATION", "task": task,
            "outer_fold": outer_fold, "seed": 0, "training_data_ids": sorted(training_ids),
            "excluded_query_data_ids": sorted(excluded_ids), "Shared": shared,
            "Specific": specific, "Shared_selection": shared_selection,
            "Specific_selection": details, "source_calibration_only": True,
            "query_labels_used_for_fitting": False, "time_unix": time.time(),
            "plan_sha256": file_sha(OUT / "plan.json"),
        })
    # No query arrays are opened by this worker before all policies are fixed.
    queries = [c for c in load("query", task) if c["fold"] == outer_fold]
    assert {c["meta"]["data_id"] for c in queries}.isdisjoint(training_ids)
    rows = []
    base_max_difference = 0.
    for cell in queries:
        meta = cell["meta"]
        block, family = meta["block"], meta["family"]
        shared_prediction = rule.predict(cell, shared)
        specific_prediction = shared_prediction.copy() if block == "clean" else rule.predict(cell, specific[block + "/" + family])
        predictions = {"Base": cell["base"], "Shared": shared_prediction, "Specific": specific_prediction}
        output = OUT / "predictions/256" / task / block / str(meta["task_id"]) / cell["path"].name
        banks.scaling.npwrite(output, **predictions, y_target=cell["y"])
        with np.load(output) as saved:
            assert np.array_equal(saved["Base"], cell["base"])
        summary = {key: meta[key] for key in ("task_id", "data_id", "dataset", "family", "rho", "scale", "actual_context")}
        summary.update(task=task, block=block, context=256, seed=0, outer_fold=outer_fold,
            bank_path=str(cell["path"]), bank_sha256=file_sha(cell["path"]),
            source_input_hashes=meta["source_input_hashes"],
            policy_path=str(destination / "frozen_policy.json"),
            policy_sha256=file_sha(destination / "frozen_policy.json"),
            original_paper_replication=False, development_benchmark=True,
            methods={name: metrics(cell["y"], prediction, task, meta["scale"])
                     for name, prediction in predictions.items()})
        atomic_json(output.with_suffix(".json"), summary)
        for method, scores in summary["methods"].items():
            rows.append({k: summary[k] for k in ("task", "block", "context", "seed", "outer_fold", "task_id", "data_id", "dataset", "family", "rho")}
                        | {"method": method} | scores)
    pd.DataFrame(rows).to_csv(destination / "query_detail.csv", index=False)
    atomic_json(destination / "completion.json", {"status": "COMPLETE", "task": task,
        "outer_fold": outer_fold, "conditions": len(queries), "method_records": len(rows),
        "Base_max_difference": base_max_difference, "source_and_test_datasets_disjoint": True,
        "elapsed_seconds": time.time() - started})
    print("FOLD_COMPLETE", task, outer_fold, len(queries), flush=True)


def report():
    files = sorted((OUT / "fold_runs").glob("*/fold_*/query_detail.csv"))
    if not files:
        return
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    data.to_csv(OUT / "query_detail.csv", index=False)
    averages = []
    for (task, block, method), group in data.groupby(["task", "block", "method"], sort=False):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        # Equal dataset, family, and strength weighting, retaining every cohort.
        scores = group.groupby("data_id")[keys].mean().mean()
        averages.append({"task": task, "block": block, "method": method,
                         "datasets": group.data_id.nunique(), "conditions": len(group),
                         **scores.to_dict()})
    frame = pd.DataFrame(averages)
    frame.to_csv(OUT / "averages.csv", index=False)
    finished = list((OUT / "fold_runs").glob("*/fold_*/completion.json"))
    atomic_json(OUT / "progress.json", {"status": "COMPLETE_PENDING_AUDIT" if len(finished) == 10 else "RUNNING",
        "completed_folds": len(finished), "total_folds": 10, "method_records": len(data),
        "conditions": len(data) // 3, "original_paper_replication": False,
        "independent_test_claim": False})
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["initialize", "worker", "report"])
    parser.add_argument("--task", choices=list(EXPECTED))
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.phase == "initialize":
        initialize()
    elif args.phase == "worker":
        assert args.task is not None and args.fold is not None
        worker(args.task, args.fold)
    else:
        report()
