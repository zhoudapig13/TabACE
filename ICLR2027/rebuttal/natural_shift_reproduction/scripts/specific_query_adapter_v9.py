"""Source-only supervised query correction with frozen Shared and 512-row views.

This is an explicitly labelled additional supervised adapter, not the original
paper's fusion rule. Training and selection never open target test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fit_and_report import SCENARIOS, TASKS, paper_reference, prediction_for_member
from natural_components import CONTINUOUS, SHARED_NAMES
from nested_source_adapter import blend, head
from prepare_acs import RUN, save_json, sha
from row_budget_views import SPECIFIC_EXTRA_NAMES
from strict_nested_source import raw

DEST = RUN / "results/specific_query_adapter_v9"
REFERENCE = RUN / "results/strict_source_nested_512_v8"
BETAS = [0., .25, .5, .75, 1.]
GRID = [
    {"id": "logistic_C001", "kind": "logistic", "C": .01},
    {"id": "logistic_C01", "kind": "logistic", "C": .1},
    {"id": "logistic_C1", "kind": "logistic", "C": 1.},
    {"id": "catboost_d4_l3", "kind": "catboost", "depth": 4, "l2": 3.},
    {"id": "catboost_d6_l3", "kind": "catboost", "depth": 6, "l2": 3.},
    {"id": "catboost_d6_l10", "kind": "catboost", "depth": 6, "l2": 10.},
    {"id": "catboost_d8_l10", "kind": "catboost", "depth": 8, "l2": 10.},
]


def measures(y, p):
    return {"accuracy": float(accuracy_score(y, p.argmax(1))),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def domain_for(scenario, role):
    source = "2018_CA" if scenario == "geographic" else "2014_CA"
    return source if role != "OOD" else "2018_MI" if scenario == "geographic" else "2018_CA"


def data_for(scenario, task, role):
    prefix = RUN / "data/splits" / f"{domain_for(scenario, role)}_{task}"
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if role in ["fit", "select"]:
        with np.load(str(prefix)+"_calibration.npz", allow_pickle=False) as data:
            section = slice(0, 2048) if role == "fit" else slice(2048, None)
            values = {name: data[name][section] for name in ["X", "y", "households", "row_ids"]}
    else:
        with np.load(str(prefix)+"_test_features.npz", allow_pickle=False) as data:
            values = {name: data[name] for name in ["X", "households", "row_ids"]}
    return values, metadata


def shared_prediction(scenario, task, role):
    policy = json.loads((REFERENCE / "policies" / f"Shared_{scenario}_{task}.json").read_text())
    assert policy["calibration_domains"] == [domain_for(scenario, "fit")]
    return head(raw(scenario, task, role, policy["source_policy"]), policy["head_parameters"])


def feature_frame(scenario, task, role):
    values, metadata = data_for(scenario, task, role)
    X = values["X"]
    shared = shared_prediction(scenario, task, role)
    base = prediction_for_member(scenario, task, role, "Base")
    kinds = SHARED_NAMES+SPECIFIC_EXTRA_NAMES
    bank = np.stack([prediction_for_member(scenario, task, role, f"rows_512::{kind}") for kind in kinds])
    posterior = np.column_stack([shared[:, 1], base[:, 1]]+[p[:, 1] for p in bank])
    posterior = np.clip(posterior, 1e-6, 1.-1e-6)
    logits = np.log(posterior/(1.-posterior))
    prediction_features = np.column_stack([
        logits, posterior.std(1), posterior.max(1)-posterior.min(1),
        -(shared*np.log(np.maximum(shared, 1e-10))).sum(1)])
    frame = pd.DataFrame(prediction_features, columns=[f"prediction_feature_{j}" for j in range(prediction_features.shape[1])])
    categorical = []
    for j, name in enumerate(metadata["features"]):
        column = "query_"+name
        if name in CONTINUOUS:
            frame[column] = X[:, j].astype(float)
        else:
            frame[column] = np.rint(X[:, j]).astype(np.int64).astype(str)
            categorical.append(column)
    return frame, prediction_features, categorical, values, shared


def score_key(record):
    return record["accuracy"], -record["log_loss"], -record["beta"]


def fit_case(scenario, task):
    output = DEST / scenario / task
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frozen_policy.json").exists():
        print("CACHED FROZEN", scenario, task, flush=True)
        return
    frame, prediction_features, cats, fit, _ = feature_frame(scenario, task, "fit")
    val_frame, val_prediction_features, _, select, shared_val = feature_frame(scenario, task, "select")
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    train_ids, inner_ids = next(splitter.split(frame, fit["y"], fit["households"]))
    assert set(fit["households"][train_ids]).isdisjoint(fit["households"][inner_ids])
    shared_metrics = measures(select["y"], shared_val)
    trials = [{"candidate": "Shared_fallback", "beta": 0., **shared_metrics}]
    for config in GRID:
        if config["kind"] == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=config["C"], max_iter=1000, random_state=0))
            model.fit(prediction_features, fit["y"])
            p = model.predict_proba(val_prediction_features)
            artifact = output / (config["id"]+".joblib")
            joblib.dump(model, artifact)
            detail = {"fit_rows": len(fit["y"])}
        else:
            params = dict(iterations=600, learning_rate=.05, depth=config["depth"],
                          l2_leaf_reg=config["l2"], loss_function="Logloss", eval_metric="Logloss",
                          random_seed=0, thread_count=2, verbose=False,
                          allow_writing_files=False, max_ctr_complexity=1)
            preliminary = CatBoostClassifier(**params)
            preliminary.fit(frame.iloc[train_ids], fit["y"][train_ids], cat_features=cats,
                            eval_set=(frame.iloc[inner_ids], fit["y"][inner_ids]),
                            early_stopping_rounds=60, use_best_model=True)
            rounds = max(25, int(preliminary.get_best_iteration())+1)
            params["iterations"] = rounds
            model = CatBoostClassifier(**params)
            model.fit(frame, fit["y"], cat_features=cats)
            p = model.predict_proba(val_frame)
            artifact = output / (config["id"]+".cbm")
            model.save_model(str(artifact))
            detail = {"fit_rows": len(fit["y"]), "inner_train_rows": len(train_ids),
                      "inner_validation_rows": len(inner_ids), "selected_iterations": rounds}
        rows = [{"candidate": config["id"], "beta": beta,
                 **measures(select["y"], blend(shared_val, p, beta))} for beta in BETAS]
        trials.extend(rows)
        save_json(output / (config["id"]+".json"), {"config": config, "model_path": str(artifact),
                  "sha256": sha(artifact), "selection_trials": rows, **detail})
        print(scenario, task, config["id"], "validation", max(rows, key=score_key), flush=True)
    best = max(trials, key=score_key)
    selected_config = next((config for config in GRID if config["id"] == best["candidate"]), None)
    model_path = None
    if selected_config is not None:
        model_path = str(output / (best["candidate"]+(".joblib" if selected_config["kind"] == "logistic" else ".cbm")))
    pd.DataFrame(trials).to_csv(output / "source_validation_trials.csv", index=False)
    save_json(output / "frozen_policy.json", {
        "status": "FROZEN", "scenario": scenario, "task": task, "seed": 0,
        "model": selected_config, "model_path": model_path,
        "model_sha256": sha(Path(model_path)) if model_path else None, "selected": best,
        "shared_source_validation": shared_metrics, "calibration_domains": [domain_for(scenario, "fit")],
        "source_fit_rows": len(fit["y"]), "source_selection_rows": len(select["y"]),
        "fit_row_ids_sha256": hashlib.sha256(fit["row_ids"].tobytes()).hexdigest(),
        "select_row_ids_sha256": hashlib.sha256(select["row_ids"].tobytes()).hexdigest(),
        "shared_policy_sha256": sha(REFERENCE / "policies" / f"Shared_{scenario}_{task}.json"),
        "context_row_cap": 512, "context_types": 8,
        "additional_supervised_query_adapter": True, "original_paper_implementation": False,
        "target_domain_calibration_labels_used": False, "test_labels_accessed": False})
    print("FROZEN", scenario, task, best, flush=True)


def evaluate():
    policies = {}
    for scenario in SCENARIOS:
        for task in TASKS:
            path = DEST / scenario / task / "frozen_policy.json"
            policy = json.loads(path.read_text())
            assert policy["status"] == "FROZEN" and policy["test_labels_accessed"] is False
            assert policy["shared_policy_sha256"] == sha(REFERENCE / "policies" / f"Shared_{scenario}_{task}.json")
            policies[(scenario, task)] = policy
    save_json(DEST / "all_policies_frozen_before_test.json", {"status": "FROZEN",
        "policies": {str(key): value for key, value in policies.items()}})
    paper, rows, gains = paper_reference(), [], []
    for (scenario, task), policy in policies.items():
        model = None
        if policy["model"] is not None:
            assert sha(Path(policy["model_path"])) == policy["model_sha256"]
            if policy["model"]["kind"] == "logistic":
                model = joblib.load(policy["model_path"])
            else:
                model = CatBoostClassifier(); model.load_model(policy["model_path"])
        for role in ["ID", "OOD"]:
            frame, prediction_features, _, data, shared = feature_frame(scenario, task, role)
            with np.load(REFERENCE / "predictions" / scenario / task / f"{role}_Shared.npz") as old:
                assert np.array_equal(old["prediction"], shared)
            expert = shared if model is None else model.predict_proba(prediction_features if policy["model"]["kind"] == "logistic" else frame)
            specific = blend(shared, expert, policy["selected"]["beta"])
            prefix = RUN / "data/splits" / f"{domain_for(scenario, role)}_{task}"
            with np.load(str(prefix)+"_test_labels.npz", allow_pickle=False) as labels:
                y, row_ids = labels["y"], labels["row_ids"]
            assert np.array_equal(row_ids, data["row_ids"])
            for method, p in [("Shared", shared), ("Specific_supervised", specific)]:
                values = measures(y, p)
                reference = paper[(scenario, "Shared" if method == "Shared" else "Specific", role, task)]
                rows.append({"scenario": scenario, "task": task, "domain": role, "method": method, **values,
                    "auc": float(roc_auc_score(y, p[:, 1])), "paper_accuracy": reference,
                    "difference_from_paper": values["accuracy"]-reference, "seed": 0,
                    "additional_supervised_query_adapter": method != "Shared", "rows": len(y)})
            gain = float(np.mean(specific.argmax(1)==y)-np.mean(shared.argmax(1)==y))
            gains.append({"scenario": scenario, "task": task, "domain": role,
                          "shared_accuracy": float(np.mean(shared.argmax(1)==y)),
                          "specific_accuracy": float(np.mean(specific.argmax(1)==y)),
                          "accuracy_gain": gain, "required_gain": .005, "meets_threshold": gain+1e-12 >= .005})
            directory = DEST / "predictions" / scenario / task
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"{role}.npz", shared=shared, specific=specific,
                                y=y, row_ids=row_ids, households=data["households"])
    pd.DataFrame(rows).to_csv(DEST / "comparison_long.csv", index=False)
    gain_frame = pd.DataFrame(gains)
    gain_frame.to_csv(DEST / "gains_vs_fixed_shared.csv", index=False)
    save_json(DEST / "completion.json", {"status": "COMPLETE", "seed": 0,
        "required_minimum_gain_per_cell": .005, "cells": 12,
        "cells_meeting_threshold": int(gain_frame.meets_threshold.sum()),
        "all_12_meet_threshold": bool(gain_frame.meets_threshold.all()),
        "Shared_unchanged": True, "target_test_labels_used_for_selection": False,
        "additional_supervised_adapter_not_original_paper_method": True})
    print(gain_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        if not args.scenario or not args.task:
            parser.error("Training requires --scenario and --task")
        fit_case(args.scenario, args.task)
