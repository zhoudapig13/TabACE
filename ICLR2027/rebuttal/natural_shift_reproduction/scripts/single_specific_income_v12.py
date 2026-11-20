"""Single-case Specific adaptation with v11 Shared frozen.

Source-only query routers are compared at two source-label budgets. 4096 rows
means existing source-pool 2048 plus calibration-fit 2048; the independent source
selection 2048 remains unchanged. Every TFM context is still at most 512 rows.
This is supervised adaptation, not reproduction of the paper's original rule.
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
from sklearn.model_selection import GroupShuffleSplit

from fit_and_report import prediction_for_member
from natural_components import CONTINUOUS, SHARED_NAMES
from prepare_acs import RUN, save_json, sha
from row_budget_views import SPECIFIC_EXTRA_NAMES
import single_shared_income_v11 as shared_v11
from specific_query_adapter_v9 import (BETAS, GRID, data_for, feature_frame,
                                       measures, score_key)

DEST = RUN / "results/single_specific_geo_income_v12"
SHARED = RUN / "results/single_shared_geo_income_v11"
OLD_MODELS = RUN / "results/specific_query_adapter_v9/geographic/Income"
SCENARIO, TASK = "geographic", "Income"
QUERY_GRID = [{"depth": depth, "l2": penalty} for depth in [4, 6] for penalty in [3., 10.]]


def query_frame(X, features):
    columns, categorical = {}, []
    for j, name in enumerate(features):
        if name in CONTINUOUS:
            columns[name] = X[:, j].astype(float)
        else:
            columns[name] = np.rint(X[:, j]).astype(np.int64).astype(str)
            categorical.append(name)
    return pd.DataFrame(columns), categorical


def shared_policy():
    policy = json.loads((SHARED / "frozen_policy.json").read_text())
    assert sha(Path(policy["model_path"])) == policy["model_sha256"]
    return policy


def current_shared(role):
    policy = shared_policy()
    frame, numeric, _, values, bank = shared_v11.inputs(role)
    model = shared_v11.load_model(policy["model"], policy["model_path"])
    router = model.predict_proba(numeric if policy["model"]["kind"] == "logistic" else frame)
    return shared_v11.route(bank, router, policy["selected"]["beta"])[0]


def bank_for(role, shared):
    return np.stack([shared]+[prediction_for_member(SCENARIO, TASK, role, f"rows_512::{name}")
                             for name in SHARED_NAMES+SPECIFIC_EXTRA_NAMES], axis=1)


def route(bank, router, beta):
    n = len(router)
    scores = bank[:, :, 1]
    lower_id, upper_id = scores.argmin(1), scores.argmax(1)
    lower, upper = scores[np.arange(n), lower_id], scores[np.arange(n), upper_id]
    fraction = np.clip((router[:, 1]-lower)/np.maximum(upper-lower, 1e-12), 0., 1.)
    weights = np.zeros(scores.shape)
    weights[:, 0] = 1.-beta
    weights[np.arange(n), lower_id] += beta*(1.-fraction)
    weights[np.arange(n), upper_id] += beta*fraction
    result = np.einsum("nr,nrc->nc", weights, bank)
    if beta == 0.:
        result = bank[:, 0].copy()
    return result, weights


def models_for_validation():
    """Only source data are accessed during model fitting/selection."""
    fit, metadata = data_for(SCENARIO, TASK, "fit")
    selection, _ = data_for(SCENARIO, TASK, "select")
    source_path = RUN / "data/splits/2018_CA_Income_source.npz"
    with np.load(source_path) as arrays:
        source = {key: arrays[key] for key in ["X", "y", "row_ids", "households"]}
    assert set(fit["row_ids"]).isdisjoint(source["row_ids"])
    assert set(fit["households"]).isdisjoint(source["households"])
    assert set(fit["row_ids"]).isdisjoint(selection["row_ids"])
    select_query, _ = query_frame(selection["X"], metadata["features"])
    # Keep the prior calibrated eight-view models in the source-selected space.
    old_frame, old_numeric, _, _, _ = feature_frame(SCENARIO, TASK, "select")
    for config in GRID:
        info = json.loads((OLD_MODELS / (config["id"]+".json")).read_text())
        model = shared_v11.load_model(config, info["model_path"])
        assert sha(Path(info["model_path"])) == info["sha256"]
        p = model.predict_proba(old_numeric if config["kind"] == "logistic" else old_frame)
        yield {"id": "existing_"+config["id"], "kind": "existing_meta", "old_config": config,
               "model_path": info["model_path"], "model_sha256": info["sha256"],
               "fit_rows": 2048, "calibration_domains": ["2018_CA"]}, p
    for rows in [2048, 4096]:
        training = fit if rows == 2048 else {key: np.concatenate([source[key], fit[key]]) for key in fit}
        train_frame, cats = query_frame(training["X"], metadata["features"])
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
        learn, early = next(splitter.split(train_frame, training["y"], training["households"]))
        assert set(training["households"][learn]).isdisjoint(training["households"][early])
        for config in QUERY_GRID:
            label = f"query_n{rows}_d{config['depth']}_l{config['l2']:g}"
            params = dict(iterations=900, learning_rate=.05, depth=config["depth"],
                l2_leaf_reg=config["l2"], random_seed=0, thread_count=2, verbose=False,
                allow_writing_files=False, loss_function="Logloss", eval_metric="Logloss",
                max_ctr_complexity=1)
            preliminary = CatBoostClassifier(**params)
            preliminary.fit(train_frame.iloc[learn], training["y"][learn], cat_features=cats,
                eval_set=(train_frame.iloc[early], training["y"][early]),
                early_stopping_rounds=60, use_best_model=True)
            rounds = max(25, int(preliminary.get_best_iteration())+1)
            params["iterations"] = rounds
            model = CatBoostClassifier(**params)
            model.fit(train_frame, training["y"], cat_features=cats)
            model_path = DEST / (label+".cbm")
            model.save_model(str(model_path))
            record = {"id": label, "kind": "query_only", "model_path": str(model_path),
                "model_sha256": sha(model_path), "fit_rows": rows, "iterations": rounds,
                "parameters": params, "calibration_domains": ["2018_CA"],
                "fit_row_ids_sha256": hashlib.sha256(training["row_ids"].tobytes()).hexdigest(),
                "source_pool_labels_used_for_router": rows == 4096,
                "validation_row_ids_sha256": hashlib.sha256(selection["row_ids"].tobytes()).hexdigest(),
                "early_stopping_household_disjoint": True}
            save_json(DEST / (label+".json"), record)
            yield record, model.predict_proba(select_query)


def fit():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / "frozen_policy.json").exists():
        raise RuntimeError("Existing frozen run must not be retuned in place")
    values, _ = data_for(SCENARIO, TASK, "select")
    shared = current_shared("select")
    bank = bank_for("select", shared)
    candidates = {"Shared_fallback": None}
    trials = [{"candidate": "Shared_fallback", "beta": 0., **measures(values["y"], shared)}]
    for record, router in models_for_validation():
        candidates[record["id"]] = record
        rows = [{"candidate": record["id"], "beta": beta,
                 **measures(values["y"], route(bank, router, beta)[0])} for beta in BETAS]
        trials.extend(rows)
        print(record["id"], max(rows, key=score_key), flush=True)
    best = max(trials, key=score_key)
    pd.DataFrame(trials).to_csv(DEST / "source_validation_trials.csv", index=False)
    save_json(DEST / "frozen_policy.json", {"status": "FROZEN", "scenario": SCENARIO,
        "task": TASK, "seed": 0, "selected": best, "router": candidates[best["candidate"]],
        "calibration_domains": ["2018_CA"], "source_selection_rows": 2048,
        "Shared_policy_path": str(SHARED / "frozen_policy.json"),
        "Shared_policy_sha256": sha(SHARED / "frozen_policy.json"),
        "Shared_router_model_sha256": shared_policy()["model_sha256"],
        "context_types": SHARED_NAMES+SPECIFIC_EXTRA_NAMES, "context_row_cap": 512,
        "complete_updated_Shared_fallback": True, "additional_supervised_router": True,
        "test_labels_accessed": False, "target_domain_calibration_labels_used": False,
        "required_OOD_gain_over_Shared": .005, "other_cases_changed": False})
    print("FROZEN", best, flush=True)


def router_prediction(policy, role):
    record = policy["router"]
    if record is None:
        return current_shared(role)
    assert sha(Path(record["model_path"])) == record["model_sha256"]
    values, metadata = data_for(SCENARIO, TASK, role)
    if record["kind"] == "query_only":
        model = CatBoostClassifier(); model.load_model(record["model_path"])
        frame, _ = query_frame(values["X"], metadata["features"])
    else:
        config = record["old_config"]
        model = shared_v11.load_model(config, record["model_path"])
        combined, numeric, _, _, _ = feature_frame(SCENARIO, TASK, role)
        frame = numeric if config["kind"] == "logistic" else combined
    return model.predict_proba(frame)


def evaluate():
    policy = json.loads((DEST / "frozen_policy.json").read_text())
    assert policy["status"] == "FROZEN" and not policy["test_labels_accessed"]
    assert sha(SHARED / "frozen_policy.json") == policy["Shared_policy_sha256"]
    records = []
    for role in ["ID", "OOD"]:
        shared = current_shared(role)
        bank = bank_for(role, shared)
        router = router_prediction(policy, role)
        specific, weights = route(bank, router, policy["selected"]["beta"])
        domain = "2018_CA" if role == "ID" else "2018_MI"
        with np.load(RUN / "data/splits" / f"{domain}_Income_test_labels.npz") as labels:
            y, row_ids = labels["y"], labels["row_ids"]
        with np.load(SHARED / f"{role}_predictions.npz") as previous:
            assert np.array_equal(previous["shared_new"], shared)
            old_specific = previous["specific_unchanged"]
        sm, pm = measures(y, shared), measures(y, specific)
        records.append({"scenario": SCENARIO, "task": TASK, "domain": role,
            "Shared_accuracy": sm["accuracy"], "Specific_accuracy": pm["accuracy"],
            "Specific_minus_Shared": pm["accuracy"]-sm["accuracy"],
            "Specific_old_accuracy": measures(y, old_specific)["accuracy"],
            "meets_minimum_gain_0_005": pm["accuracy"]-sm["accuracy"]+1e-12 >= .005,
            "Specific_log_loss": pm["log_loss"], "rows": len(y), "seed": 0})
        np.savez_compressed(DEST / f"{role}_predictions.npz", shared=shared, specific=specific,
            old_specific=old_specific, router_prediction=router, weights=weights,
            y=y, row_ids=row_ids)
    pd.DataFrame(records).to_csv(DEST / "single_case_comparison.csv", index=False)
    ood = next(row for row in records if row["domain"] == "OOD")
    save_json(DEST / "completion.json", {"status": "COMPLETE", **ood,
        "Shared_unchanged": True, "other_cases_changed": False,
        "target_test_labels_used_for_selection": False, "additional_source_supervised_router": True})
    print(pd.DataFrame(records).to_string(index=False), flush=True)


def audit():
    policy = json.loads((DEST / "frozen_policy.json").read_text())
    assert policy["calibration_domains"] == ["2018_CA"] and policy["seed"] == 0
    assert policy["context_row_cap"] == 512 and len(policy["context_types"]) == 8
    assert not policy["test_labels_accessed"] and not policy["target_domain_calibration_labels_used"]
    assert sha(SHARED / "frozen_policy.json") == policy["Shared_policy_sha256"]
    trials = pd.read_csv(DEST / "source_validation_trials.csv")
    best = trials.sort_values(["accuracy", "log_loss", "beta"], ascending=[False, True, True], kind="stable").iloc[0]
    assert best.candidate == policy["selected"]["candidate"] and best.beta == policy["selected"]["beta"]
    results = pd.read_csv(DEST / "single_case_comparison.csv")
    max_error = 0.
    fit, _ = data_for(SCENARIO, TASK, "fit")
    selection, _ = data_for(SCENARIO, TASK, "select")
    if policy["router"] and policy["router"]["kind"] == "query_only":
        record = policy["router"]
        with np.load(RUN / "data/splits/2018_CA_Income_source.npz") as data:
            training_ids = np.concatenate([data["row_ids"], fit["row_ids"]]) if record["fit_rows"] == 4096 else fit["row_ids"]
        assert hashlib.sha256(training_ids.tobytes()).hexdigest() == record["fit_row_ids_sha256"]
        assert set(training_ids).isdisjoint(selection["row_ids"])
    for role in ["ID", "OOD"]:
        with np.load(DEST / f"{role}_predictions.npz") as f:
            saved = {name: f[name] for name in f.files}
        shared = current_shared(role)
        assert np.array_equal(shared, saved["shared"])
        with np.load(SHARED / f"{role}_predictions.npz") as previous:
            assert np.array_equal(previous["shared_new"], saved["shared"])
        bank = bank_for(role, shared)
        weights = saved["weights"]
        assert weights.shape == (len(shared), 9) and np.all(weights >= 0.)
        assert np.max(np.abs(weights.sum(1)-1.)) < 1e-12
        reconstructed = (bank*weights[:, :, None]).sum(1)
        beta = policy["selected"]["beta"]
        projected = np.clip(saved["router_prediction"][:, 1], bank[:, :, 1].min(1), bank[:, :, 1].max(1))
        expected_positive = (1.-beta)*shared[:, 1]+beta*projected
        assert np.max(np.abs(expected_positive-saved["specific"][:, 1])) < 1e-12
        error = float(np.max(np.abs(reconstructed-saved["specific"])))
        assert error < 1e-12
        max_error = max(max_error, error)
        assert np.max(np.abs(router_prediction(policy, role)-saved["router_prediction"])) < 1e-12
        for kind in policy["context_types"]:
            with np.load(RUN / "results/row_banks/geographic/Income" / f"n512_{role}__{kind}.npz") as ctx:
                assert len(ctx["context_source_pool_indices"]) <= 512
        domain = "2018_CA" if role == "ID" else "2018_MI"
        with np.load(RUN / "data/splits" / f"{domain}_Income_test_labels.npz") as labels:
            assert np.array_equal(labels["y"], saved["y"])
            assert np.array_equal(labels["row_ids"], saved["row_ids"])
        row = results[results.domain == role].iloc[0]
        assert abs(float(np.mean(saved["specific"].argmax(1)==saved["y"]))-row.Specific_accuracy) < 1e-12
        assert abs(float(np.mean(saved["shared"].argmax(1)==saved["y"]))-row.Shared_accuracy) < 1e-12
    ood = results[results.domain == "OOD"].iloc[0]
    save_json(DEST / "audit.json", {"status": "PASS", "scope": "geographic Income only",
        "Shared_accuracy": ood.Shared_accuracy, "Specific_accuracy": ood.Specific_accuracy,
        "accuracy_gain": ood.Specific_minus_Shared,
        "meets_minimum_gain_0_005": bool(ood.Specific_minus_Shared+1e-12 >= .005),
        "Shared_bitwise_unchanged": True, "source_only_selection": True,
        "context_row_cap": 512, "TFM_context_types": 8, "seed": 0,
        "router_training_rows": policy["router"]["fit_rows"] if policy["router"] else 0,
        "source_selection_rows": 2048, "additional_supervised_adapter": True,
        "max_prediction_reconstruction_error": max_error,
        "MD_summary_still_empty": (RUN.parents[1]/"REBUTTAL_EXPERIMENTS.md").stat().st_size == 0})
    print("AUDIT PASS", "OOD Shared", ood.Shared_accuracy, "Specific", ood.Specific_accuracy,
          "gain", ood.Specific_minus_Shared, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.evaluate:
        evaluate()
    else:
        fit()
