"""Source-selected query router whose output is a convex TFM ensemble.

An auxiliary supervised router learns reliability from source rows. Its output
sets interpolation weights but cannot emit an unconstrained class probability.
The final output is a convex combination of full Shared and eight 512-row TFM
views. This remains an additional supervised adaptation, not the original rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fit_and_report import SCENARIOS, TASKS, paper_reference, prediction_for_member
from natural_components import SHARED_NAMES
from prepare_acs import RUN, save_json, sha
from row_budget_views import SPECIFIC_EXTRA_NAMES
from specific_query_adapter_v9 import (BETAS, GRID, REFERENCE, data_for, domain_for,
                                       feature_frame, measures, score_key)

MODELS = RUN / "results/specific_query_adapter_v9"
DEST = RUN / "results/specific_convex_router_v10"


def expert_bank(scenario, task, role, shared):
    return np.stack([shared]+[prediction_for_member(scenario, task, role, f"rows_512::{name}")
                              for name in SHARED_NAMES+SPECIFIC_EXTRA_NAMES], axis=1)


def route(bank, prediction, beta):
    n = len(prediction)
    scores = bank[:, :, 1]
    low, high = scores.argmin(1), scores.argmax(1)
    lower, upper = scores[np.arange(n), low], scores[np.arange(n), high]
    fraction = np.clip((prediction[:, 1]-lower)/np.maximum(upper-lower, 1e-12), 0., 1.)
    weights = np.zeros(scores.shape, dtype=float)
    weights[:, 0] = 1.-beta
    weights[np.arange(n), low] += beta*(1.-fraction)
    weights[np.arange(n), high] += beta*fraction
    result = np.einsum("nr,nrc->nc", weights, bank)
    if beta == 0.:
        result = bank[:, 0].copy()
    return result, weights


def load_model(config, path):
    if config["kind"] == "logistic":
        return joblib.load(path)
    model = CatBoostClassifier(); model.load_model(path)
    return model


def select_all():
    for scenario in SCENARIOS:
        for task in TASKS:
            frame, numeric, _, values, shared = feature_frame(scenario, task, "select")
            bank = expert_bank(scenario, task, "select", shared)
            trials = [{"candidate": "Shared_fallback", "beta": 0., **measures(values["y"], shared)}]
            for config in GRID:
                folder = MODELS / scenario / task
                info = json.loads((folder / (config["id"]+".json")).read_text())
                path = info["model_path"]
                assert sha(Path(path)) == info["sha256"]
                model = load_model(config, path)
                p = model.predict_proba(numeric if config["kind"] == "logistic" else frame)
                for beta in BETAS:
                    output, _ = route(bank, p, beta)
                    trials.append({"candidate": config["id"], "beta": beta, **measures(values["y"], output)})
            best = max(trials, key=score_key)
            config = next((c for c in GRID if c["id"] == best["candidate"]), None)
            path = None
            if config is not None:
                info = json.loads((MODELS / scenario / task / (config["id"]+".json")).read_text())
                path = info["model_path"]
            save_json(DEST / "policies" / f"{scenario}_{task}.json", {
                "status": "FROZEN", "scenario": scenario, "task": task, "seed": 0,
                "selected": best, "config": config, "model_path": path,
                "model_sha256": sha(Path(path)) if path else None,
                "calibration_domains": [domain_for(scenario, "fit")],
                "Shared_policy_sha256": sha(REFERENCE / "policies" / f"Shared_{scenario}_{task}.json"),
                "source_fit_rows": 2048, "source_selection_rows": 2048,
                "context_row_cap": 512, "TFM_context_types": 8,
                "full_Shared_fallback_available": True, "convex_output_constraint": True,
                "additional_supervised_router": True, "original_paper_rule": False,
                "test_labels_accessed": False, "target_domain_calibration_labels_used": False})
            (DEST / "policies").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trials).to_csv(DEST / "policies" / f"{scenario}_{task}_validation.csv", index=False)
            print("FROZEN", scenario, task, best, flush=True)


def evaluate():
    policies = {}
    for scenario in SCENARIOS:
        for task in TASKS:
            p = json.loads((DEST / "policies" / f"{scenario}_{task}.json").read_text())
            assert p["status"] == "FROZEN" and not p["test_labels_accessed"]
            assert p["Shared_policy_sha256"] == sha(REFERENCE / "policies" / f"Shared_{scenario}_{task}.json")
            policies[(scenario, task)] = p
    save_json(DEST / "all_policies_frozen_before_test.json", {"status": "FROZEN",
              "policies": {str(key): value for key, value in policies.items()}})
    gains, records, paper = [], [], paper_reference()
    for (scenario, task), policy in policies.items():
        if policy["model_path"]:
            assert sha(Path(policy["model_path"])) == policy["model_sha256"]
        model = load_model(policy["config"], policy["model_path"]) if policy["config"] else None
        for role in ["ID", "OOD"]:
            frame, numeric, _, values, shared = feature_frame(scenario, task, role)
            with np.load(REFERENCE / "predictions" / scenario / task / f"{role}_Shared.npz") as previous:
                assert np.array_equal(previous["prediction"], shared)
            gate = shared if model is None else model.predict_proba(numeric if policy["config"]["kind"] == "logistic" else frame)
            bank = expert_bank(scenario, task, role, shared)
            specific, weights = route(bank, gate, policy["selected"]["beta"])
            prefix = RUN / "data/splits" / f"{domain_for(scenario, role)}_{task}"
            with np.load(str(prefix)+"_test_labels.npz") as labels:
                y, row_ids = labels["y"], labels["row_ids"]
            assert np.array_equal(row_ids, values["row_ids"])
            delta = float(np.mean(specific.argmax(1)==y)-np.mean(shared.argmax(1)==y))
            gains.append({"scenario": scenario, "task": task, "domain": role,
                "shared_accuracy": float(np.mean(shared.argmax(1)==y)),
                "specific_accuracy": float(np.mean(specific.argmax(1)==y)), "accuracy_gain": delta,
                "required_gain": .005, "meets_threshold": delta+1e-12 >= .005})
            for method, p in [("Shared", shared), ("Specific_supervised_router", specific)]:
                values_metric = measures(y, p)
                p_ref = paper[(scenario, "Shared" if method == "Shared" else "Specific", role, task)]
                records.append({"scenario": scenario, "task": task, "domain": role, "method": method,
                    **values_metric, "auc": float(roc_auc_score(y, p[:, 1])), "paper_accuracy": p_ref,
                    "difference_from_paper": values_metric["accuracy"]-p_ref, "rows": len(y), "seed": 0,
                    "additional_supervised_router": method != "Shared"})
            folder = DEST / "predictions" / scenario / task
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{role}.npz", shared=shared, specific=specific,
                router_weights=weights, router_prediction=gate, y=y, row_ids=row_ids,
                households=values["households"])
    table = pd.DataFrame(gains)
    table.to_csv(DEST / "gains_vs_fixed_shared.csv", index=False)
    pd.DataFrame(records).to_csv(DEST / "comparison_long.csv", index=False)
    save_json(DEST / "completion.json", {"status": "COMPLETE", "seed": 0, "cells": 12,
        "required_minimum_accuracy_gain": .005, "cells_meeting_threshold": int(table.meets_threshold.sum()),
        "all_12_meet_threshold": bool(table.meets_threshold.all()), "Shared_unchanged": True,
        "context_row_cap": 512, "TFM_context_types": 8, "additional_supervised_router": True,
        "original_paper_implementation": False, "test_labels_used_for_selection": False})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        select_all()
