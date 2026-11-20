"""Audit actual model artifacts, frozen Shared, all 12 gains and convex weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from fit_and_report import SCENARIOS, TASKS, prediction_for_member
from natural_components import SHARED_NAMES
from prepare_acs import RUN, save_json, sha
from row_budget_views import SPECIFIC_EXTRA_NAMES
from specific_query_adapter_v9 import REFERENCE, data_for, domain_for, feature_frame


def audit(convex=False):
    destination = RUN / "results" / ("specific_convex_router_v10" if convex else "specific_query_adapter_v9")
    actual = []
    max_prediction_error = 0.
    max_weight_error = 0.
    assert json.loads((destination / "all_policies_frozen_before_test.json").read_text())["status"] == "FROZEN"
    summary = pd.read_csv(destination / "gains_vs_fixed_shared.csv")
    assert len(summary) == 12
    for scenario in SCENARIOS:
        for task in TASKS:
            if convex:
                path = destination / "policies" / f"{scenario}_{task}.json"
                policy = json.loads(path.read_text())
                config = policy["config"]
                shared_hash = policy["Shared_policy_sha256"]
                validation = pd.read_csv(destination / "policies" / f"{scenario}_{task}_validation.csv")
            else:
                path = destination / scenario / task / "frozen_policy.json"
                policy = json.loads(path.read_text())
                config = policy["model"]
                shared_hash = policy["shared_policy_sha256"]
                validation = pd.read_csv(destination / scenario / task / "source_validation_trials.csv")
            assert policy["calibration_domains"] == [domain_for(scenario, "fit")]
            assert policy["seed"] == 0 and policy["context_row_cap"] == 512
            assert policy["source_fit_rows"] == 2048 and policy["source_selection_rows"] == 2048
            assert not policy["test_labels_accessed"] and not policy["target_domain_calibration_labels_used"]
            assert shared_hash == sha(REFERENCE / "policies" / f"Shared_{scenario}_{task}.json")
            best = validation.sort_values(["accuracy", "log_loss", "beta"], ascending=[False, True, True], kind="stable").iloc[0]
            assert best.candidate == policy["selected"]["candidate"]
            assert best.beta == policy["selected"]["beta"]
            assert abs(best.accuracy-policy["selected"]["accuracy"]) < 1e-12
            source, _ = data_for(scenario, task, "fit")
            selection, _ = data_for(scenario, task, "select")
            training_record = policy
            if convex:
                training_record = json.loads((RUN / "results/specific_query_adapter_v9" / scenario / task / "frozen_policy.json").read_text())
            import hashlib
            assert training_record["fit_row_ids_sha256"] == hashlib.sha256(source["row_ids"].tobytes()).hexdigest()
            assert training_record["select_row_ids_sha256"] == hashlib.sha256(selection["row_ids"].tobytes()).hexdigest()
            assert set(source["row_ids"]).isdisjoint(selection["row_ids"])
            own_test, _ = data_for(scenario, task, "ID")
            assert set(source["households"]).isdisjoint(own_test["households"])
            assert set(selection["households"]).isdisjoint(own_test["households"])
            model = None
            if config:
                assert sha(Path(policy["model_path"])) == policy["model_sha256"]
                if config["kind"] == "logistic":
                    model = joblib.load(policy["model_path"])
                    assert model[-1].random_state == 0
                else:
                    model = CatBoostClassifier(); model.load_model(policy["model_path"])
                    assert model.get_all_params()["random_seed"] == 0
            for role in ["ID", "OOD"]:
                frame, numeric, _, values, shared = feature_frame(scenario, task, role)
                with np.load(REFERENCE / "predictions" / scenario / task / f"{role}_Shared.npz") as reference:
                    assert np.array_equal(reference["prediction"], shared)
                raw_router = shared if model is None else model.predict_proba(numeric if config["kind"] == "logistic" else frame)
                beta = policy["selected"]["beta"]
                with np.load(destination / "predictions" / scenario / task / f"{role}.npz") as file:
                    result = {key: file[key] for key in file.files}
                assert np.array_equal(result["shared"], shared)
                if convex:
                    bank = np.stack([shared]+[prediction_for_member(scenario, task, role, f"rows_512::{kind}")
                                              for kind in SHARED_NAMES+SPECIFIC_EXTRA_NAMES], axis=1)
                    weights = result["router_weights"]
                    assert weights.shape == (len(shared), 9) and np.all(weights >= 0.)
                    max_weight_error = max(max_weight_error, float(np.max(np.abs(weights.sum(1)-1.))))
                    assert max_weight_error < 1e-12
                    assert np.max(np.abs(raw_router-result["router_prediction"])) < 1e-12
                    mixture = (bank*weights[:, :, None]).sum(1)
                    # Independently verify the projection/beta mapping as well.
                    projected = np.clip(raw_router[:, 1], bank[:, :, 1].min(1), bank[:, :, 1].max(1))
                    expected_positive = (1.-beta)*shared[:, 1]+beta*projected
                    assert np.max(np.abs(mixture[:, 1]-expected_positive)) < 1e-12
                else:
                    mixture = (1.-beta)*shared+beta*raw_router
                diff = float(np.max(np.abs(mixture-result["specific"])))
                max_prediction_error = max(max_prediction_error, diff)
                assert diff < 1e-12
                assert np.all(np.isfinite(result["specific"]))
                assert np.all(result["specific"] >= 0.)
                assert np.max(np.abs(result["specific"].sum(1)-1.)) < 1e-10
                prefix = RUN / "data/splits" / f"{domain_for(scenario, role)}_{task}"
                with np.load(str(prefix)+"_test_labels.npz") as labels:
                    assert np.array_equal(labels["y"], result["y"])
                    assert np.array_equal(labels["row_ids"], result["row_ids"])
                assert np.array_equal(values["row_ids"], result["row_ids"])
                for kind in SHARED_NAMES+SPECIFIC_EXTRA_NAMES:
                    p = RUN / "results/row_banks" / scenario / task / f"n512_{role}__{kind}.npz"
                    with np.load(p) as cached:
                        ids = cached["context_source_pool_indices"]
                    assert len(ids) <= 512 and np.all((ids >= 0) & (ids < 2048))
                y = result["y"]
                shared_correct = result["shared"].argmax(1) == y
                specific_correct = result["specific"].argmax(1) == y
                gain = (int(specific_correct.sum())-int(shared_correct.sum()))/len(y)
                row = summary[(summary.scenario == scenario) & (summary.task == task) & (summary.domain == role)].iloc[0]
                assert abs(gain-row.accuracy_gain) < 1e-12
                assert abs(specific_correct.mean()-row.specific_accuracy) < 1e-12
                assert abs(shared_correct.mean()-row.shared_accuracy) < 1e-12
                actual.append({"scenario": scenario, "task": task, "domain": role,
                    "rows": len(y), "accuracy_gain": gain, "meets_required_gain": gain+1e-12 >= .005,
                    "newly_correct": int(np.sum(specific_correct & ~shared_correct)),
                    "newly_wrong": int(np.sum(shared_correct & ~specific_correct))})
    success = all(entry["meets_required_gain"] for entry in actual)
    record = {"status": "PASS", "run": destination.name, "cells_verified": len(actual),
        "required_accuracy_gain": .005, "all_12_meet_required_gain": success,
        "minimum_accuracy_gain": min(entry["accuracy_gain"] for entry in actual),
        "Shared_prediction_bitwise_unchanged": True, "seed": 0, "context_row_cap": 512,
        "TFM_context_types": 8, "convex_output_constraint": convex,
        "max_prediction_reconstruction_error": max_prediction_error,
        "max_convex_weight_sum_error": max_weight_error,
        "source_calibration_scope_verified": True, "target_labels_used_for_selection": False,
        "additional_supervised_router_or_corrector": True, "original_paper_Specific_reproduced": False,
        "cells": actual, "md_summary_still_empty": (RUN.parents[1] / "REBUTTAL_EXPERIMENTS.md").stat().st_size == 0,
        "code_sha256": {p.name: sha(p) for p in sorted((RUN / "scripts").glob("*.py"))}}
    save_json(destination / "audit.json", record)
    print(destination.name, "AUDIT PASS", "all 12 >= .005:", success,
          "minimum gain:", record["minimum_accuracy_gain"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--convex", action="store_true")
    args = parser.parse_args()
    audit(args.convex)
