"""Validation-selected supervised adapters, explicitly distinct from TabACE.

Reuse existing TRAIN-only checkpoints. New residual strengths minimize source
validation CE/MSE, never distance to paper numbers or a requested test ranking.
The benchmark is a previously tuned development benchmark, not a new test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import joblib
import numpy as np
import pandas as pd
import torch

from prepare_data import RUN, atomic_json, protocol
from run_tabicl import normalize, metrics
import paper_rule_reconstruction_v77 as original

OUT = RUN / "results/tabace_validated_adaptation_v79"
BLENDS = [0., .1, .25, .5, .75, 1.]
JOBS = [(task, block) for task in ["classification", "regression"] for block in ["clean", "five", "seven"]]


def write_case(task, block, cell, predictions, provenance):
    meta = cell["meta"]
    tid, family, rho = meta["task_id"], meta["family"], meta["rho"]
    out = OUT / "predictions/256" / task / block / str(tid) / f"{family}_{rho}.npz"
    assert np.array_equal(predictions["Base"], cell["base"])
    original.banks.scaling.npwrite(out, **predictions, y_target=cell["y"])
    values = {name: metrics(cell["y"], p, task, meta["scale"]) for name, p in predictions.items()}
    record = {k: meta[k] for k in ["task_id", "dataset", "family", "rho", "scale"]}
    record.update(task=task, block=block, seed=0, context=256, methods=values,
        source_bank=str(cell["path"]), source_bank_sha256=original.file_sha(cell["path"]),
        extra_clean_TRAIN_supervision=True, checkpoint_finetuned=False,
        supervised_hybrid=True, original_TabACE_reproduction=False,
        prior_query_feedback_used_for_development=True, independent_test_claim=False,
        new_test_score_selection=False, matched_inference_budget_claim=False, **provenance)
    atomic_json(out.with_suffix(".json"), record)


def reg_loss(cells, predictions):
    return float(np.mean([np.mean((predictions[c["key"]] - c["y"]) ** 2) / c["meta"]["scale"] ** 2 for c in cells]))


def reg_select(cells, anchor, candidate):
    trials, best, minimum = [], 0., float("inf")
    for blend in BLENDS:
        p = {key: (1. - blend) * anchor[key] + blend * candidate[key] for key in anchor}
        loss = reg_loss(cells, p)
        trials.append({"blend": blend, "validation_normalized_MSE": loss})
        if loss < minimum - 1e-12:
            minimum, best = loss, blend
    predictions = {key: (1. - best) * anchor[key] + best * candidate[key] for key in anchor}
    return {"blend": best, "source_loss": minimum, "trials": trials}, predictions


def regression_residual(block):
    import moderate_adaptation_v75 as parent
    cells, base, strong_shared, strong_specific, origin = parent.regression_source(256, block)
    shared, shared_source = reg_select(cells, base, strong_shared)
    specific, _ = reg_select(cells, shared_source, strong_specific)
    folder = OUT / "policies" / "regression" / block
    freeze = folder / "frozen_policy.json"
    atomic_json(freeze, {"status": "FROZEN_BEFORE_QUERY", "Shared": shared, "Specific": specific,
        "objective": "minimum source normalized MSE; no requested score range or ranking",
        "candidate_origin": str(origin), "extra_clean_TRAIN_supervision": True,
        "source_conditions": len(cells), "query_labels_used_for_selection": False})
    for c in cells:
        output = OUT / "source/256" / c["key"]
        original.banks.scaling.npwrite(output, base=base[c["key"]], strong_shared=strong_shared[c["key"]],
            strong_specific=strong_specific[c["key"]], y_validation=c["y"])
    for c in parent.expert.h.risk.load(parent.expert.h.source.OUT, "query", 256, block):
        old = origin / "predictions/256" / c["key"]
        with np.load(old) as z:
            assert np.array_equal(z["Base"], c["base"]) and np.array_equal(z["y_target"], c["y"])
            a = shared["blend"]; sh = (1. - a) * c["base"] + a * z["Shared"]
            a = specific["blend"]; sp = (1. - a) * sh + a * z["Specific"]
        write_case("regression", block, c, {"Base": c["base"], "Shared": sh, "Specific": sp},
            {"policy_path": str(freeze), "candidate_path": str(old),
             "candidate_sha256": original.file_sha(old), "adaptation": "validation_MSE_selected_residual"})


def replay_regression_clean():
    import source_expert_v53 as parent
    policy_path = parent.OUT / "policies/256/regression/clean/all_policies_frozen.json"
    policy = json.loads(policy_path.read_text()); cache = {}
    atomic_json(OUT / "policies/regression/clean/frozen_policy.json", {
        "status": "REUSE_PREVIOUS_SOURCE_SELECTED_CHECKPOINTS", "parent_policy": str(policy_path),
        "parent_policy_sha256": original.file_sha(policy_path), "policy": policy})
    for c in parent.h.risk.load(parent.h.source.OUT, "query", 256, "clean"):
        meta = c["meta"]; tid = str(meta["task_id"])
        data, _ = parent.h.source.adaptation.raw.scaling.bank("regression", "clean", meta, "original", 0)
        info = json.loads((parent.OUT / "experts/256/clean" / f"{tid}.json").read_text())
        assert protocol.array_sha256(data["X_target"]) == meta["source_input_hashes"]["X_target"]
        def apply(recipe, anchor):
            choice = recipe["chosen"]
            if choice is None: return anchor
            path = info["query_model_paths"][choice["profile_index"]]
            if path not in cache: cache[path] = joblib.load(path)
            value = parent.h.head_predict(data["X_target"], cache[path]); a = choice["blend"]
            return (1. - a) * anchor + a * value
        sh = apply(policy["Shared"], c["base"])
        sp = apply(policy["Specific"]["original"], sh)
        write_case("regression", "clean", c, {"Base": c["base"], "Shared": sh, "Specific": sp},
                   {"policy_path": str(policy_path), "adaptation": "clean_TRAIN_auxiliary_regression_models"})


def replay_classification_clean():
    import weighted_gate_v62 as parent
    parent.configure()
    path = parent.OUT / "policies/256/classification/clean/all_policies_frozen.json"
    policy = json.loads(path.read_text())
    atomic_json(OUT / "policies/classification/clean/frozen_policy.json", {
        "status": "REUSE_PREVIOUS_SOURCE_SELECTED_CHECKPOINTS", "parent_policy": str(path),
        "parent_policy_sha256": original.file_sha(path), "policy": policy})
    for c in parent.load("query", 256):
        sh = parent.apply(c, policy["Shared_raw"], policy["Shared"], c["base"])
        sp = parent.apply(c, policy["Specific_raw"], policy["Specific"], sh)
        write_case("classification", "clean", c, {"Base": c["base"], "Shared": sh, "Specific": sp},
                   {"policy_path": str(path), "adaptation": "source_prior_weighted_gate_and_probability_calibration"})


def replay_classification_five():
    import probability_head_v51 as parent
    root = parent.OUT / "policies/256/classification/five"
    shared = json.loads((root / "Shared/policy.json").read_text())["model"]
    families = parent.source.adaptation.raw.protocol.FAMILIES
    specific = {f: json.loads((root / f"Specific_{f}/policy.json").read_text())["model"] for f in families}
    atomic_json(OUT / "policies/classification/five/frozen_policy.json", {
        "status": "REUSE_PREVIOUS_SOURCE_OOF_SELECTED_CHECKPOINTS", "Shared": shared,
        "Specific": specific, "source_policy_root": str(root), "original_row_group_folds": 4})
    cache = {}
    for c in parent.load("query", 256, "five"):
        sh = parent.predict_cell(c, shared, c["base"], "cpu", cache)
        sp = parent.predict_cell(c, specific[c["meta"]["family"]], sh, "cpu", cache)
        write_case("classification", "five", c, {"Base": c["base"], "Shared": sh, "Specific": sp},
                   {"policy_path": str(OUT / "policies/classification/five/frozen_policy.json"),
                    "adaptation": "TRAIN_group_OOF_supervised_probability_residual_heads"})


def classification_seven():
    import specific_family_selection_v76 as saved
    parent = saved.parent
    cells, Y, B, _ = saved.source(256)
    def loss(P, indices=None):
        if indices is not None: P, truth = P[indices], Y[indices]
        else: truth = Y
        return float(-np.log(np.clip(np.take_along_axis(P, truth[:, :, None], axis=2)[:, :, 0], 1e-12, 1.)).mean())
    best_shared = None; shared_loss = loss(B); Sh = B.copy(); sh_trials = []
    train = np.asarray([c["meta"]["priors"]["train_shared"] for c in cells])
    target = np.asarray([c["meta"]["priors"]["target_shared"] for c in cells])
    raw = np.asarray([c["shared"][0] for c in cells])
    for power, temperature in product(parent.POWERS, parent.TEMPERATURES):
        value = parent.correction(raw, train, target, power, temperature)
        for blend in BLENDS:
            p = normalize((1. - blend) * B + blend * value); nll = loss(p)
            choice = dict(power=power, temperature=temperature, blend=blend, profile_index=0)
            sh_trials.append(choice | {"source_CE": nll})
            if nll < shared_loss - 1e-12: shared_loss, best_shared, Sh = nll, choice, p
    descriptions = []; candidates = []
    for recipe, power, temperature in product(saved.RECIPES, parent.POWERS, parent.TEMPERATURES):
        descriptions.append(dict(recipe=recipe, power=power, temperature=temperature))
        candidates.append(saved.candidate(cells, recipe, power, temperature))
    data_ids = np.asarray([c["data_id"] for c in cells]); datasets = sorted(set(data_ids.tolist()))
    mapping = {d: i % 4 for i, d in enumerate(datasets)}
    folds = np.asarray([mapping[d] for d in data_ids])
    families = np.asarray([c["meta"]["family"] for c in cells]); cv = Sh.copy(); recipes = {}; fold_records = []
    def choose(indices):
        minimum = loss(Sh, indices); best = None
        for index, p in enumerate(candidates):
            for blend in BLENDS[1:]:
                value = normalize((1. - blend) * Sh[indices] + blend * p[indices])
                nll = loss(value, None) if len(indices) == len(Y) else float(-np.log(np.clip(
                    np.take_along_axis(value, Y[indices, :, None], 2)[:, :, 0], 1e-12, 1.)).mean())
                if nll < minimum - 1e-12: minimum, best = nll, dict(index=index, local_blend=blend)
        return best
    for family in parent.parent.expert.FAMILIES:
        group = np.flatnonzero(families == family)
        for fold in range(4):
            train_ids = group[folds[group] != fold]; valid_ids = group[folds[group] == fold]
            assert not set(data_ids[train_ids]) & set(data_ids[valid_ids])
            choice = choose(train_ids)
            if choice is not None:
                a = choice["local_blend"]; cv[valid_ids] = normalize((1. - a) * Sh[valid_ids] + a * candidates[choice["index"]][valid_ids])
            fold_records.append(dict(family=family, fold=fold, choice=choice,
                training_data_ids=sorted(set(data_ids[train_ids].tolist())), excluded_data_ids=sorted(set(data_ids[valid_ids].tolist()))))
        choice = choose(group)
        recipes[family] = None if choice is None else descriptions[choice["index"]] | {"local_blend": choice["local_blend"]}
    best_blend = 0.; minimum = loss(Sh); trials = []
    for blend in BLENDS:
        value = normalize((1. - blend) * Sh + blend * cv); nll = loss(value)
        trials.append(dict(blend=blend, source_dataset_OOF_CE=nll))
        if nll < minimum - 1e-12: minimum, best_blend = nll, blend
    freeze = OUT / "policies/classification/seven/frozen_policy.json"
    atomic_json(freeze, {"status": "FROZEN_BEFORE_QUERY", "Shared": best_shared, "Shared_trials": sh_trials,
        "Specific_blend": best_blend, "Specific_trials": trials, "Specific_recipes": recipes,
        "family_dataset_crossfits": fold_records, "objective": "source CE; no desired numerical gain or ordering constraint",
        "extra_clean_TRAIN_supervision": True, "known_generator_augmentation": True})
    for meta in parent.parent.RAW.frozen.cohort("classification", "seven"):
        tid = str(meta["task_id"]); pool = parent.parent.RAW.scaling.prepare("classification", "seven", meta)
        split = parent.parent.partition(meta); descriptor = parent.parent.priors(meta, pool, split, 256, "query"); cache = {}
        def infer(path, X):
            if str(path) not in cache: cache[str(path)] = joblib.load(path)
            return parent.parent.expert.predict(X, cache[str(path)], 2)
        for family, rho in parent.parent.RAW.case_grid("seven"):
            bank_path = parent.parent.expert.nn.source.OUT / "banks/query/256/classification/seven" / tid / f"{family}_{rho}.npz"
            record = json.loads(bank_path.with_suffix(".json").read_text())
            with np.load(bank_path) as z: b, truth = z["base"], z["y_target"]
            data, _ = parent.parent.RAW.scaling.bank("classification", "seven", meta, family, rho); X = data["X_target"]
            assert protocol.array_sha256(X) == record["source_input_hashes"]["X_target"]
            sh = b.copy()
            if best_shared is not None:
                path = parent.parent.expert.OUT / "experts/256/seven" / tid / "Shared/query_0.joblib"
                value = parent.correction(infer(path, X)[None], np.array([descriptor["train_shared"]]),
                    np.array([descriptor["target_shared"]]), best_shared["power"], best_shared["temperature"])[0]
                a = best_shared["blend"]; sh = normalize((1. - a) * b + a * value)
            sp = sh.copy(); choice = recipes[family]
            if choice is not None and best_blend:
                recipe, power, temperature = choice["recipe"], choice["power"], choice["temperature"]
                target = descriptor["target"][family]
                if recipe["kind"] == "legacy":
                    path = parent.parent.expert.OUT / "experts/256/seven" / tid / f"Specific_{family}" / f"query_{recipe['index']}.joblib"
                    value = parent.correction(infer(path, X)[None], np.array([descriptor["train"][family]]), np.array([target]), power, temperature)[0]
                else:
                    directory = saved.OUT / "experts" / tid / family
                    registry = json.loads((directory / "models_complete.json").read_text()); values = []
                    for member in range(recipe["members"]):
                        p = infer(directory / f"query_{member}.joblib", X)
                        values.append(parent.correction(p[None], np.array([registry["query_priors"][member]]), np.array([target]), power, 1.)[0])
                    p = np.mean(values, axis=0); logits = np.log(np.clip(p, 1e-12, 1.)) / temperature
                    value = normalize(np.exp(logits - logits.max(1, keepdims=True)))
                a = choice["local_blend"]; candidate = normalize((1. - a) * sh + a * value)
                sp = normalize((1. - best_blend) * sh + best_blend * candidate)
            cell = dict(meta=record, path=bank_path, base=b, y=truth)
            write_case("classification", "seven", cell, {"Base": b, "Shared": sh, "Specific": sp},
                {"policy_path": str(freeze), "adaptation": "TRAIN_augmented_auxiliary_experts_and_source_CE_selection"})


def worker(task, block):
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    if task == "classification":
        {"clean": replay_classification_clean, "five": replay_classification_five, "seven": classification_seven}[block]()
    elif block == "clean": replay_regression_clean()
    else: regression_residual(block)
    atomic_json(OUT / "completion" / f"{task}_{block}.json", {"status": "COMPLETE", "task": task, "block": block})
    print("COMPLETE", task, block, flush=True)


def report():
    rows = []
    for task, block in JOBS:
        for p in (OUT / "predictions/256" / task / block).glob("*/*.json"):
            meta = json.loads(p.read_text())
            for method, scores in meta["methods"].items():
                rows.append({k: meta[k] for k in ["task", "block", "task_id", "family", "rho"]} | {"method": method} | scores)
    if not rows: return
    data = pd.DataFrame(rows); data.to_csv(OUT / "detail.csv", index=False); averages = []
    for (task, block, method), group in data.groupby(["task", "block", "method"]):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        scores = group.groupby("task_id")[keys].mean().mean()
        averages.append(dict(task=task, block=block, method=method, datasets=group.task_id.nunique(), conditions=len(group), **scores.to_dict()))
    pd.DataFrame(averages).to_csv(OUT / "averages.csv", index=False)


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "plan.json", {"status": "FROZEN_BEFORE_NEW_SOURCE_SELECTION", "jobs": JOBS, "seed": 0,
        "context_cap": 256, "Base_registry": str(RUN / "results/tabicl_base_context_recovery_frozen_v43/parameters.json"),
        "Base_and_benchmark_unchanged": True, "blends": BLENDS,
        "new_selection_objective": "source CE for classification-seven; source normalized MSE for regression-five/seven",
        "reuse_source_selected_checkpoints": ["classification clean/five", "regression clean"],
        "desired_test_ranking_is_selection_constraint": False, "paper_numbers_are_selection_objective": False,
        "extra_clean_TRAIN_supervision": True, "supervised_hybrid": True,
        "original_paper_reproduction": False, "independent_test_claim": False,
        "upstream_query_feedback_used_for_development": True,
        "semisynthetic_regression_blocks": ["clean", "five"],
        "upstream_classification_five_perturbs_queries": True,
        "TFM_weights_finetuned": False, "existing_TRAIN_only_checkpoints_reused": True,
        "code_sha256": original.file_sha(__file__), "Markdown_written": False})
    records = []
    def execute(task, block):
        log = OUT / "logs" / f"{task}_{block}.log"; log.parent.mkdir(exist_ok=True)
        with log.open("a") as handle:
            result = subprocess.run([sys.executable, __file__, "worker", "--task", task, "--block", block], stdout=handle, stderr=subprocess.STDOUT)
        return dict(task=task, block=block, returncode=result.returncode, log=str(log))
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(execute, task, block) for task, block in JOBS]):
            records.append(future.result()); atomic_json(OUT / "execution.json", {"status": "RUNNING", "jobs": records}); print(records[-1], flush=True)
    report()
    atomic_json(OUT / "execution.json", {"status": "COMPLETE" if all(r["returncode"] == 0 for r in records) else "FAILED", "jobs": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["launch", "worker", "report"])
    parser.add_argument("--task", choices=["classification", "regression"]); parser.add_argument("--block", choices=["clean", "five", "seven"])
    args = parser.parse_args()
    if args.phase == "launch": launch()
    elif args.phase == "worker": worker(args.task, args.block)
    else: report()
