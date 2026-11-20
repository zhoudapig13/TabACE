"""Five-seed natural-shift repetitions with frozen data splits and hyperparameters.

Every outer seed reaches context sampling, TabICL random_state, torch/NumPy and
CatBoost training. Source fusion parameters are re-fitted with fixed selected
regularization. Model architecture, iterations, output calibration choice and
blend strength are frozen from the selected seed-0 configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from natural_components import CONTINUOUS, SHARED_NAMES, apply_policy, fit_policy
from nested_source_adapter import fit_head, head
from prepare_acs import RUN, save_json, sha
from row_budget_views import SPECIFIC_EXTRA_NAMES, prepare_views
from run_natural import CHECKPOINT, Predictor, digest, load
from single_specific_income_v12 import query_frame
from specific_convex_router_v10 import route

DEST = RUN / "results/multi_seed_natural_v13"
PLAN = RUN / "configs/protocol_v13_multi_seed_natural.json"
SCENARIOS = ["geographic", "temporal"]
TASKS = ["Employment", "Income", "TravelTime"]
ROLES = ["fit", "select", "ID", "OOD"]
KINDS = SHARED_NAMES+SPECIFIC_EXTRA_NAMES


def case_key(scenario, task):
    return scenario+"/"+task


def make_plan():
    cases = {}
    for scenario in SCENARIOS:
        for task in TASKS:
            special = scenario == "geographic" and task == "Income"
            core_path = RUN / "results/strict_source_nested_512_v8/policies" / f"Shared_{scenario}_{task}.json"
            core = json.loads(core_path.read_text())
            spec_path = RUN / "results/specific_convex_router_v10/policies" / f"{scenario}_{task}.json"
            spec = json.loads(spec_path.read_text())
            info = json.loads((RUN / "results/specific_query_adapter_v9" / scenario / task / (spec["config"]["id"]+".json")).read_text())
            case = {"scenario": scenario, "task": task,
                "source_domain": "2018_CA" if scenario == "geographic" else "2014_CA",
                "target_domain": "2018_MI" if scenario == "geographic" else "2018_CA",
                "Shared_fusion_regularization": core["fusion_selection"]["selected"]["regularization"],
                "Shared_head_regularization": core["head_selection"]["selected"]["regularization"],
                "Shared_core_reference": str(core_path), "Shared_core_reference_sha256": sha(core_path),
                "Shared_supervised_router": None,
                "Specific_router": {"kind": "meta_eight", "depth": spec["config"]["depth"],
                    "l2_leaf_reg": spec["config"]["l2"], "iterations": info["selected_iterations"],
                    "learning_rate": .05, "fit_rows": 2048, "beta": spec["selected"]["beta"]},
                "seed0_prediction_reference": str(RUN / "results/specific_convex_router_v10/predictions" / scenario / task)}
            if special:
                spath = RUN / "results/single_shared_geo_income_v11/frozen_policy.json"
                shared = json.loads(spath.read_text())
                sinfo = json.loads((spath.parent / (shared["model"]["id"]+".json")).read_text())
                ppath = RUN / "results/single_specific_geo_income_v12/frozen_policy.json"
                specific = json.loads(ppath.read_text())
                sparams = specific["router"]["parameters"]
                case["Shared_supervised_router"] = {"kind": "meta_five",
                    "depth": shared["model"]["depth"], "l2_leaf_reg": shared["model"]["l2"],
                    "iterations": sinfo["iterations"], "learning_rate": .05,
                    "fit_rows": 2048, "beta": shared["selected"]["beta"]}
                case["Specific_router"] = {"kind": "query_only", "depth": sparams["depth"],
                    "l2_leaf_reg": sparams["l2_leaf_reg"], "iterations": sparams["iterations"],
                    "learning_rate": sparams["learning_rate"], "fit_rows": specific["router"]["fit_rows"],
                    "beta": specific["selected"]["beta"]}
                case["seed0_prediction_reference"] = str(ppath.parent)
            cases[case_key(scenario, task)] = case
    datasets = {}
    for prefix in sorted((RUN / "data/splits").glob("*.json")):
        if not prefix.stem.startswith(("2014_CA_", "2018_CA_", "2018_MI_")):
            continue
        for suffix in ["source", "calibration", "test_features", "test_labels"]:
            path = prefix.with_name(prefix.stem+"_"+suffix+".npz")
            datasets[path.name] = sha(path)
    value = {"protocol_id": "multi_seed_natural_v13", "seeds": [0, 1, 2, 3, 4],
        "scenarios": SCENARIOS, "tasks": TASKS, "methods": ["Base", "Shared", "Specific"],
        "context_row_cap": 512, "Base_context_rows": 256, "Shared_context_types": 5,
        "Specific_context_types": 8, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
        "fixed_data_partitions": True, "fixed_source_pools": True,
        "fixed_source_fit_and_selection_partitions": True,
        "randomized_components": ["context row selection and bootstrap", "TabICL random_state",
                                  "torch/NumPy seed", "CatBoost random_seed"],
        "fixed_hyperparameters": True, "per_seed_hyperparameter_search": False,
        "all_seeds_retained_without_accuracy_filtering": True,
        "source_only_fitting": True, "target_labels_used_for_fitting": False,
        "statistics": {"mean": "arithmetic mean over exactly five seeds", "variance_ddof": 1,
                       "variance_units": "Accuracy squared", "fixed_split_conditional_variance": True},
        "additional_supervised_adaptation": True, "original_paper_reproduction": False,
        "dataset_sha256": datasets, "cases": cases,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "torch", "catboost", "scipy", "scikit-learn"]},
        "code_sha256": {name: sha(Path(__file__).with_name(name)) for name in ["multi_seed_natural_v13.py", "row_budget_views.py", "natural_components.py"]}}
    if PLAN.exists():
        previous = json.loads(PLAN.read_text())
        if previous != value:
            raise RuntimeError("Existing multi-seed plan differs; do not overwrite a frozen protocol")
    else:
        save_json(PLAN, value)
    print(PLAN, flush=True)


def data(scenario, task):
    source_domain = "2018_CA" if scenario == "geographic" else "2014_CA"
    target_domain = "2018_MI" if scenario == "geographic" else "2018_CA"
    prefix = RUN / "data/splits" / f"{source_domain}_{task}"
    source = load(prefix, "source")
    cal = load(prefix, "calibration")
    meta = json.loads(prefix.with_suffix(".json").read_text())
    query = {"fit": cal["X"][:2048], "select": cal["X"][2048:],
             "ID": load(prefix, "test_features")["X"],
             "OOD": load(RUN / "data/splits" / f"{target_domain}_{task}", "test_features")["X"]}
    return source, cal, meta, query


def probability_frame(X, feature_names, base, shared_core, members):
    post = np.clip(np.column_stack([shared_core[:, 1], base[:, 1]]+[m[:, 1] for m in members]), 1e-6, 1.-1e-6)
    numeric = np.column_stack([np.log(post/(1.-post)), post.std(1), post.max(1)-post.min(1),
                              -(shared_core*np.log(np.maximum(shared_core, 1e-10))).sum(1)])
    frame = pd.DataFrame(numeric, columns=[f"prediction_feature_{j}" for j in range(numeric.shape[1])])
    cats = []
    for j, name in enumerate(feature_names):
        label = "query_"+name
        if name in CONTINUOUS:
            frame[label] = X[:, j].astype(float)
        else:
            frame[label] = np.rint(X[:, j]).astype(np.int64).astype(str)
            cats.append(label)
    return frame, cats


def new_model(config, seed):
    return CatBoostClassifier(iterations=config["iterations"], learning_rate=config["learning_rate"],
        depth=config["depth"], l2_leaf_reg=config["l2_leaf_reg"], loss_function="Logloss",
        eval_metric="Logloss", random_seed=seed, thread_count=2, verbose=False,
        allow_writing_files=False, max_ctr_complexity=1)


def run(seed, scenario, task):
    plan = json.loads(PLAN.read_text())
    assert seed in plan["seeds"]
    case = plan["cases"][case_key(scenario, task)]
    output = DEST / f"seed_{seed}" / scenario / task
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        old = json.loads((output / "completion.json").read_text())
        assert old["plan_sha256"] == sha(PLAN)
        print("CACHED COMPLETE", seed, scenario, task, flush=True)
        return
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    started = time.time()
    source, calibration, metadata, queries = data(scenario, task)
    predictor = Predictor(); predictor.model.random_state = seed
    base_ids = np.random.default_rng(seed).permutation(len(source["y"]))[:256]
    banks, base = {}, {}
    for role, Q in queries.items():
        folder = output / "banks"; folder.mkdir(exist_ok=True)
        specs = [("Base", source["X"][base_ids], source["y"][base_ids], Q,
                  {"source_row_ids": base_ids})]
        specs += list(prepare_views(source["X"], source["y"], Q, metadata["features"], seed=seed)(512))
        values = {}
        for name, A, y, V, extra in specs:
            target = folder / f"{role}__{name}.npz"
            fingerprint = digest(A, y, V)
            if target.exists():
                info = json.loads(target.with_suffix(".json").read_text())
                assert info["input_sha256"] == fingerprint and info["TabICL_random_state"] == seed
                with np.load(target) as f:
                    prediction = f["prediction"]
            else:
                prediction = predictor.predict(A, y, V)
                indices = extra.pop("source_row_ids")
                np.savez_compressed(target, prediction=prediction, source_pool_indices=indices)
                save_json(target.with_suffix(".json"), {"seed": seed, "role": role, "kind": name,
                    "context_rows": len(y), "input_sha256": fingerprint, "constructor": extra,
                    "TabICL_random_state": int(predictor.model.random_state),
                    "test_labels_accessed": False})
            assert prediction.shape == (len(Q), 2) and np.all(np.isfinite(prediction))
            values[name] = prediction
        base[role] = values["Base"]
        banks[role] = np.stack([values[name] for name in KINDS])
        print(seed, scenario, task, "BANK", role, flush=True)
    fit_cell = {"base": base["fit"], "bank": banks["fit"][:5], "y": calibration["y"][:2048]}
    fusion, fit_info = fit_policy([fit_cell], case["Shared_fusion_regularization"])
    shared_core = {role: apply_policy(base[role], banks[role][:5], fusion)[0] for role in ROLES}
    if case["Shared_head_regularization"] is None:
        head_parameters, head_info = np.zeros(2), {"kind": "identity"}
    else:
        head_parameters, head_info = fit_head(shared_core["fit"], calibration["y"][:2048], case["Shared_head_regularization"])
    shared_core = {role: head(value, head_parameters) for role, value in shared_core.items()}
    shared = dict(shared_core)
    model_records = {}
    if case["Shared_supervised_router"] is not None:
        config = case["Shared_supervised_router"]
        train_frame, cats = probability_frame(queries["fit"], metadata["features"], base["fit"], shared_core["fit"], banks["fit"][:5])
        model = new_model(config, seed); model.fit(train_frame, calibration["y"][:2048], cat_features=cats)
        model_path = output / "Shared_router.cbm"; model.save_model(str(model_path))
        model_records["Shared"] = {"config": config, "model_path": str(model_path), "sha256": sha(model_path),
                                   "actual_random_seed": model.get_all_params()["random_seed"]}
        for role in ROLES:
            frame, _ = probability_frame(queries[role], metadata["features"], base[role], shared_core[role], banks[role][:5])
            p = model.predict_proba(frame)
            expert_bank = np.stack([shared_core[role]]+list(banks[role][:5]), axis=1)
            shared[role], _ = route(expert_bank, p, config["beta"])
    config = case["Specific_router"]
    if config["kind"] == "query_only":
        train_X = np.concatenate([source["X"], queries["fit"]]) if config["fit_rows"] == 4096 else queries["fit"]
        train_y = np.concatenate([source["y"], calibration["y"][:2048]]) if config["fit_rows"] == 4096 else calibration["y"][:2048]
        frame, cats = query_frame(train_X, metadata["features"])
    else:
        train_y = calibration["y"][:2048]
        frame, cats = probability_frame(queries["fit"], metadata["features"], base["fit"], shared_core["fit"], banks["fit"])
    model = new_model(config, seed); model.fit(frame, train_y, cat_features=cats)
    model_path = output / "Specific_router.cbm"; model.save_model(str(model_path))
    model_records["Specific"] = {"config": config, "model_path": str(model_path), "sha256": sha(model_path),
                                 "actual_random_seed": model.get_all_params()["random_seed"]}
    save_json(output / "frozen_before_test.json", {"status": "FROZEN", "seed": seed,
        "scenario": scenario, "task": task, "plan_sha256": sha(PLAN), "case_configuration": case,
        "Shared_fusion_parameters": fusion.tolist(), "Shared_fusion_fit": fit_info,
        "Shared_head_parameters": head_parameters.tolist(), "Shared_head_fit": head_info,
        "calibration_domains": [case["source_domain"]], "models": model_records,
        "per_seed_hyperparameter_search": False, "test_labels_accessed": False,
        "base_source_pool_indices": base_ids.tolist()})
    rows, concordance = [], []
    for role in ["ID", "OOD"]:
        if config["kind"] == "query_only":
            frame, _ = query_frame(queries[role], metadata["features"])
        else:
            frame, _ = probability_frame(queries[role], metadata["features"], base[role], shared_core[role], banks[role])
        router_p = model.predict_proba(frame)
        expert_bank = np.stack([shared[role]]+list(banks[role]), axis=1)
        specific, weights = route(expert_bank, router_p, config["beta"])
        domain = case["source_domain"] if role == "ID" else case["target_domain"]
        with np.load(RUN / "data/splits" / f"{domain}_{task}_test_labels.npz") as f:
            truth, row_ids = f["y"], f["row_ids"]
        predictions = {"Base": base[role], "Shared": shared[role], "Specific": specific}
        for method, p in predictions.items():
            rows.append({"seed": seed, "scenario": scenario, "task": task, "domain": role,
                "method": method, "accuracy": float(accuracy_score(truth, p.argmax(1))),
                "auc": float(roc_auc_score(truth, p[:, 1])),
                "log_loss": float(log_loss(truth, p, labels=[0, 1])), "rows": len(truth),
                "supervised_adapter": method == "Specific" or (method == "Shared" and case["Shared_supervised_router"] is not None)})
        np.savez_compressed(output / f"{role}_predictions.npz", Base=base[role], Shared=shared[role], Specific=specific,
            Shared_core=shared_core[role], router_prediction=router_p, router_weights=weights,
            y=truth, row_ids=row_ids)
        if seed == 0:
            reference = Path(case["seed0_prediction_reference"])
            if scenario == "geographic" and task == "Income":
                with np.load(reference / f"{role}_predictions.npz") as f:
                    old_shared, old_specific = f["shared"], f["specific"]
            else:
                with np.load(reference / f"{role}.npz") as f:
                    old_shared, old_specific = f["shared"], f["specific"]
            concordance.append({"domain": role,
                "Shared_max_probability_difference": float(np.max(np.abs(old_shared-shared[role]))),
                "Specific_max_probability_difference": float(np.max(np.abs(old_specific-specific))),
                "Shared_hard_disagreements": int(np.sum(old_shared.argmax(1) != shared[role].argmax(1))),
                "Specific_hard_disagreements": int(np.sum(old_specific.argmax(1) != specific.argmax(1)))})
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    save_json(output / "completion.json", {"status": "COMPLETE", "seed": seed, "scenario": scenario,
        "task": task, "plan_sha256": sha(PLAN), "seconds": time.time()-started,
        "seed0_reference_checks": concordance, "metrics_rows": len(rows)})
    print("COMPLETE", seed, scenario, task, json.dumps(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-plan", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    if args.make_plan:
        make_plan()
    else:
        if args.seed is None or args.scenario is None or args.task is None:
            parser.error("Worker requires --seed --scenario --task")
        run(args.seed, args.scenario, args.task)
