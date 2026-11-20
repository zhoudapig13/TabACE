"""One-case Shared adaptation: CA2018 -> MI2018 Income, at most 512 rows.

Five Shared views only: none of the three Specific-only views may be read.
The supervised source-only router changes the method and is explicitly marked.
All choices are frozen on CA2018 calibration before MI test labels are opened.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fit_and_report import prediction_for_member
from natural_components import CONTINUOUS, SHARED_NAMES
from nested_source_adapter import head
from prepare_acs import RUN, save_json, sha
from specific_query_adapter_v9 import BETAS, GRID, data_for, measures, score_key
from strict_nested_source import raw

SCENARIO, TASK = "geographic", "Income"
REFERENCE = RUN / "results/strict_source_nested_512_v8"
SPECIFIC_REFERENCE = RUN / "results/specific_convex_router_v10"
DEST = RUN / "results/single_shared_geo_income_v11"


def inputs(role):
    values, metadata = data_for(SCENARIO, TASK, role)
    policy_path = REFERENCE / "policies/Shared_geographic_Income.json"
    policy = json.loads(policy_path.read_text())
    assert policy["calibration_domains"] == ["2018_CA"]
    shared = head(raw(SCENARIO, TASK, role, policy["source_policy"]), policy["head_parameters"])
    members = [prediction_for_member(SCENARIO, TASK, role, f"rows_512::{kind}") for kind in SHARED_NAMES]
    bank = np.stack([shared]+members, axis=1)
    base = prediction_for_member(SCENARIO, TASK, role, "Base")
    post = np.clip(np.column_stack([shared[:, 1], base[:, 1]]+[p[:, 1] for p in members]), 1e-6, 1.-1e-6)
    numeric = np.column_stack([np.log(post/(1.-post)), post.std(1), post.max(1)-post.min(1),
                               -(shared*np.log(np.maximum(shared, 1e-10))).sum(1)])
    frame = pd.DataFrame(numeric, columns=[f"prediction_feature_{j}" for j in range(numeric.shape[1])])
    cats = []
    for j, name in enumerate(metadata["features"]):
        column = "query_"+name
        if name in CONTINUOUS:
            frame[column] = values["X"][:, j].astype(float)
        else:
            frame[column] = np.rint(values["X"][:, j]).astype(np.int64).astype(str)
            cats.append(column)
    return frame, numeric, cats, values, bank


def route(bank, router, beta):
    n = len(router)
    values = bank[:, :, 1]
    lo, hi = values.argmin(1), values.argmax(1)
    lower, upper = values[np.arange(n), lo], values[np.arange(n), hi]
    part = np.clip((router[:, 1]-lower)/np.maximum(upper-lower, 1e-12), 0., 1.)
    weights = np.zeros(values.shape)
    weights[:, 0] = 1.-beta
    weights[np.arange(n), lo] += beta*(1.-part)
    weights[np.arange(n), hi] += beta*part
    result = np.einsum("nr,nrc->nc", weights, bank)
    if beta == 0.:
        result = bank[:, 0].copy()
    return result, weights


def load_model(config, path):
    if config["kind"] == "logistic":
        return joblib.load(path)
    model = CatBoostClassifier(); model.load_model(str(path))
    return model


def fit():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / "frozen_policy.json").exists():
        raise RuntimeError("Frozen run already exists; evaluate or audit without changing its policy")
    frame, numeric, cats, source, _ = inputs("fit")
    val_frame, val_numeric, _, validation, val_bank = inputs("select")
    inner = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    train, early = next(inner.split(frame, source["y"], source["households"]))
    assert set(source["households"][train]).isdisjoint(source["households"][early])
    trials = [{"candidate": "Shared_fallback", "beta": 0., **measures(validation["y"], val_bank[:, 0])}]
    for config in GRID:
        if config["kind"] == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=config["C"], max_iter=1000, random_state=0))
            model.fit(numeric, source["y"])
            router = model.predict_proba(val_numeric)
            path = DEST / (config["id"]+".joblib")
            joblib.dump(model, path)
            rounds = None
        else:
            params = dict(iterations=600, learning_rate=.05, depth=config["depth"],
                l2_leaf_reg=config["l2"], random_seed=0, thread_count=2, verbose=False,
                loss_function="Logloss", eval_metric="Logloss", allow_writing_files=False,
                max_ctr_complexity=1)
            preliminary = CatBoostClassifier(**params)
            preliminary.fit(frame.iloc[train], source["y"][train], cat_features=cats,
                eval_set=(frame.iloc[early], source["y"][early]),
                early_stopping_rounds=60, use_best_model=True)
            rounds = max(25, int(preliminary.get_best_iteration())+1)
            params["iterations"] = rounds
            model = CatBoostClassifier(**params)
            model.fit(frame, source["y"], cat_features=cats)
            router = model.predict_proba(val_frame)
            path = DEST / (config["id"]+".cbm")
            model.save_model(str(path))
        rows = [{"candidate": config["id"], "beta": beta,
                 **measures(validation["y"], route(val_bank, router, beta)[0])} for beta in BETAS]
        trials.extend(rows)
        save_json(DEST / (config["id"]+".json"), {"config": config,
            "model_path": str(path), "model_sha256": sha(path), "iterations": rounds,
            "source_validation_trials": rows, "source_fit_rows": 2048,
            "early_stopping_household_disjoint": True})
        print(config["id"], max(rows, key=score_key), flush=True)
    best = max(trials, key=score_key)
    config = next((c for c in GRID if c["id"] == best["candidate"]), None)
    model_path = None
    if config:
        model_path = str(DEST / (best["candidate"]+(".joblib" if config["kind"] == "logistic" else ".cbm")))
    pd.DataFrame(trials).to_csv(DEST / "source_validation_trials.csv", index=False)
    save_json(DEST / "frozen_policy.json", {"status": "FROZEN", "scenario": SCENARIO,
        "task": TASK, "seed": 0, "selected": best, "model": config, "model_path": model_path,
        "model_sha256": sha(Path(model_path)) if model_path else None,
        "calibration_domains": ["2018_CA"], "source_fit_rows": 2048, "source_selection_rows": 2048,
        "source_fit_row_ids_sha256": hashlib.sha256(source["row_ids"].tobytes()).hexdigest(),
        "source_selection_row_ids_sha256": hashlib.sha256(validation["row_ids"].tobytes()).hexdigest(),
        "Shared_reference_policy_sha256": sha(REFERENCE / "policies/Shared_geographic_Income.json"),
        "context_types": SHARED_NAMES, "per_context_row_cap": 512,
        "Specific_only_contexts_used": False, "additional_supervised_router": True,
        "test_labels_accessed": False, "target_domain_calibration_labels_used": False,
        "numeric_test_target": "OOD Accuracy > 0.766; reported after source-only selection",
        "Specific_predictions_changed": False})
    print("FROZEN", best, flush=True)


def evaluate():
    policy = json.loads((DEST / "frozen_policy.json").read_text())
    assert policy["status"] == "FROZEN" and not policy["test_labels_accessed"]
    assert policy["Shared_reference_policy_sha256"] == sha(REFERENCE / "policies/Shared_geographic_Income.json")
    model = None
    if policy["model"]:
        assert sha(Path(policy["model_path"])) == policy["model_sha256"]
        model = load_model(policy["model"], policy["model_path"])
    records = []
    for role in ["ID", "OOD"]:
        frame, numeric, _, values, bank = inputs(role)
        router = bank[:, 0] if model is None else model.predict_proba(numeric if policy["model"]["kind"] == "logistic" else frame)
        prediction, weights = route(bank, router, policy["selected"]["beta"])
        domain = "2018_CA" if role == "ID" else "2018_MI"
        with np.load(RUN / "data/splits" / f"{domain}_Income_test_labels.npz") as labels:
            y, row_ids = labels["y"], labels["row_ids"]
        assert np.array_equal(row_ids, values["row_ids"])
        with np.load(SPECIFIC_REFERENCE / "predictions/geographic/Income" / f"{role}.npz") as old:
            previous_shared, unchanged_specific = old["shared"], old["specific"]
        assert np.array_equal(previous_shared, bank[:, 0])
        new_value = measures(y, prediction)
        records.append({"scenario": SCENARIO, "task": TASK, "domain": role,
            "old_shared_accuracy": measures(y, previous_shared)["accuracy"],
            "new_shared_accuracy": new_value["accuracy"], "new_shared_log_loss": new_value["log_loss"],
            "new_shared_auc": float(roc_auc_score(y, prediction[:, 1])),
            "Specific_unchanged_accuracy": measures(y, unchanged_specific)["accuracy"],
            "Specific_minus_new_Shared": measures(y, unchanged_specific)["accuracy"]-new_value["accuracy"],
            "rows": len(y), "seed": 0, "Shared_additional_supervised_router": True})
        np.savez_compressed(DEST / f"{role}_predictions.npz", shared_old=previous_shared,
            shared_new=prediction, specific_unchanged=unchanged_specific,
            router_prediction=router, weights=weights, y=y, row_ids=row_ids)
    frame = pd.DataFrame(records)
    frame.to_csv(DEST / "single_case_comparison.csv", index=False)
    ood = next(row for row in records if row["domain"] == "OOD")
    save_json(DEST / "completion.json", {"status": "COMPLETE", "OOD_accuracy": ood["new_shared_accuracy"],
        "required_strict_lower_bound": .766, "meets_requested_threshold": ood["new_shared_accuracy"] > .766,
        "Specific_unchanged": True, "other_cases_changed": False,
        "data_partition_unchanged": True, "test_labels_used_for_selection": False,
        "additional_supervised_Shared_adaptation": True})
    print(frame.to_string(index=False), flush=True)


def audit():
    policy = json.loads((DEST / "frozen_policy.json").read_text())
    assert policy["context_types"] == SHARED_NAMES and len(policy["context_types"]) == 5
    assert policy["per_context_row_cap"] == 512 and policy["seed"] == 0
    assert policy["calibration_domains"] == ["2018_CA"]
    assert not policy["test_labels_accessed"] and not policy["target_domain_calibration_labels_used"]
    candidates = pd.read_csv(DEST / "source_validation_trials.csv")
    best = candidates.sort_values(["accuracy", "log_loss", "beta"], ascending=[False, True, True], kind="stable").iloc[0]
    assert best.candidate == policy["selected"]["candidate"] and best.beta == policy["selected"]["beta"]
    assert abs(best.accuracy-policy["selected"]["accuracy"]) < 1e-12
    table = pd.read_csv(DEST / "single_case_comparison.csv")
    errors = []
    model = load_model(policy["model"], policy["model_path"]) if policy["model"] else None
    if model is not None:
        assert sha(Path(policy["model_path"])) == policy["model_sha256"]
        if policy["model"]["kind"] == "catboost":
            assert model.get_all_params()["random_seed"] == 0
    for role in ["ID", "OOD"]:
        with np.load(DEST / f"{role}_predictions.npz") as f:
            saved = {key: f[key] for key in f.files}
        frame, numeric, _, values, bank = inputs(role)
        fresh_router = bank[:, 0] if model is None else model.predict_proba(numeric if policy["model"]["kind"] == "logistic" else frame)
        assert np.max(np.abs(fresh_router-saved["router_prediction"])) < 1e-12
        weights = saved["weights"]
        assert weights.shape == (len(saved["y"]), 6) and np.all(weights >= 0.)
        assert np.max(np.abs(weights.sum(1)-1.)) < 1e-12
        reconstructed = (weights[:, :, None]*bank).sum(1)
        errors.append(float(np.max(np.abs(reconstructed-saved["shared_new"]))))
        assert errors[-1] < 1e-12
        assert np.array_equal(values["row_ids"], saved["row_ids"])
        with np.load(SPECIFIC_REFERENCE / "predictions/geographic/Income" / f"{role}.npz") as previous:
            assert np.array_equal(previous["shared"], saved["shared_old"])
            assert np.array_equal(previous["specific"], saved["specific_unchanged"])
        for kind in SHARED_NAMES:
            with np.load(RUN / "results/row_banks/geographic/Income" / f"n512_{role}__{kind}.npz") as f:
                assert len(f["context_source_pool_indices"]) <= 512
        own_domain = "2018_CA" if role == "ID" else "2018_MI"
        with np.load(RUN / "data/splits" / f"{own_domain}_Income_test_labels.npz") as labels:
            assert np.array_equal(labels["y"], saved["y"])
            assert np.array_equal(labels["row_ids"], saved["row_ids"])
        row = table[table.domain == role].iloc[0]
        assert abs(float(np.mean(saved["shared_new"].argmax(1)==saved["y"]))-row.new_shared_accuracy) < 1e-12
    ood = table[table.domain == "OOD"].iloc[0]
    save_json(DEST / "audit.json", {"status": "PASS", "scope": "geographic Income only",
        "OOD_accuracy": ood.new_shared_accuracy, "required_strict_lower_bound": .766,
        "meets_requested_threshold": bool(ood.new_shared_accuracy > .766),
        "max_prediction_reconstruction_error": max(errors), "source_only_selection": True,
        "Specific_bitwise_unchanged": True, "context_types": 5, "context_row_cap": 512,
        "seed": 0, "MD_summary_still_empty": (RUN.parents[1]/"REBUTTAL_EXPERIMENTS.md").stat().st_size == 0,
        "additional_supervised_router": True})
    print("AUDIT PASS", "OOD Shared", ood.new_shared_accuracy, "> .766:", bool(ood.new_shared_accuracy > .766), flush=True)


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
