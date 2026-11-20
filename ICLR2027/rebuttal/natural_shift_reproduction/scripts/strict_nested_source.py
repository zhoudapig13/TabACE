"""Strict per-scenario source-only calibration with exact Shared inheritance.

Temporal policies may use CA2014 labels only. Geographic policies may use CA2018
labels only. This removes pooled target-year supervision from earlier Shared
calibration. Every context remains capped at 512 rows; types remain 5/8.
"""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from fit_and_report import SCENARIOS, TASKS, choose, paper_reference, prediction_for_member
from fit_row_budgets import calibration
from natural_components import SHARED_NAMES, apply_policy
from nested_source_adapter import BETAS, blend, head, metric, select_head
from prepare_acs import RUN, save_json
from row_budget_views import SPECIFIC_EXTRA_NAMES

DEST = RUN / "results/strict_source_nested_512_v8"


def raw(scenario, task, role, policy):
    base = prediction_for_member(scenario, task, role, "Base")
    bank = np.stack([prediction_for_member(scenario, task, role, name) for name in policy["members"]])
    return apply_policy(base, bank, np.asarray(policy["parameters"]))[0]


def source(scenario, task, role, policy):
    domain = "2018_CA" if scenario == "geographic" else "2014_CA"
    with np.load(RUN / "data/splits" / f"{domain}_{task}_calibration.npz") as data:
        y = data["y"][:2048] if role == "fit" else data["y"][2048:]
    return {"p": raw(scenario, task, role, policy), "y": y}


def fit_all():
    policies = {}
    for scenario in SCENARIOS:
        domain = "2018_CA" if scenario == "geographic" else "2014_CA"
        for task in TASKS:
            names = [f"rows_512::{name}" for name in SHARED_NAMES]
            train, select = [calibration(scenario, task, role, names) for role in ["fit", "select"]]
            parameters, fusion_details = choose([train], [select])
            shared_mapping = {"members": names, "parameters": parameters.tolist(),
                              "calibration_domains": [domain], "selected_row_budget": 512,
                              "test_labels_accessed": False, "family_identifier_used": False,
                              "source_only_adaptation": True}
            shared_train = source(scenario, task, "fit", shared_mapping)
            shared_select = source(scenario, task, "select", shared_mapping)
            theta, head_details = select_head([shared_train], [shared_select])
            shared_record = {"method": "Shared", "scenario": scenario, "task": task,
                             "kind": "strict_source_nested_v8", "source_policy": shared_mapping,
                             "head_parameters": theta.tolist(), "fusion_selection": fusion_details,
                             "head_selection": head_details, "calibration_domains": [domain],
                             "context_type_count": 5, "per_context_row_cap": 512,
                             "source_only_adaptation": True, "test_labels_accessed": False,
                             "target_domain_calibration_labels_used": False}
            save_json(DEST / "policies" / f"Shared_{scenario}_{task}.json", shared_record)
            policies[("Shared", scenario, task)] = shared_record
            specific_names = [f"rows_512::{name}" for name in SHARED_NAMES+SPECIFIC_EXTRA_NAMES]
            sfit, sselect = [calibration(scenario, task, role, specific_names) for role in ["fit", "select"]]
            specific_parameters, specific_fusion_details = choose([sfit], [sselect])
            specific_mapping = {"members": specific_names, "parameters": specific_parameters.tolist(),
                                "calibration_domains": [domain], "selected_row_budget": 512,
                                "test_labels_accessed": False, "family_identifier_used": True,
                                "source_only_adaptation": True, "fusion_selection": specific_fusion_details}
            assert specific_mapping["selected_row_budget"] == 512 and len(specific_mapping["members"]) == 8
            specific_train = source(scenario, task, "fit", specific_mapping)
            specific_select = source(scenario, task, "select", specific_mapping)
            specific_theta, specific_head_details = select_head([specific_train], [specific_select])
            shared_p = head(shared_select["p"], theta)
            expert_p = head(specific_select["p"], specific_theta)
            trials = [{"beta": beta, **metric(shared_select["y"], blend(shared_p, expert_p, beta))} for beta in BETAS]
            best = max(trials, key=lambda row: (row["accuracy"], -row["log_loss"], -row["beta"]))
            assert best["accuracy"] >= trials[0]["accuracy"]
            specific_record = {"method": "Specific", "scenario": scenario, "task": task,
                "kind": "strict_source_nested_v8", "source_policy": specific_mapping,
                "head_parameters": specific_theta.tolist(), "shared_head_parameters": theta.tolist(),
                "calibration_domains": [domain], "context_type_count": 8, "per_context_row_cap": 512,
                "beta": best["beta"], "beta_selection_trials": trials, "selected": best,
                "shared_fallback_source_validation": trials[0], "head_selection": specific_head_details,
                "exact_shared_fallback_available": True, "strict_test_improvement_guaranteed": False,
                "source_only_adaptation": True, "target_domain_calibration_labels_used": False,
                "test_labels_accessed": False}
            save_json(DEST / "policies" / f"Specific_{scenario}_{task}.json", specific_record)
            policies[("Specific", scenario, task)] = specific_record
            print(scenario, task, "source", domain, "beta", best["beta"],
                  "source validation gain", best["accuracy"]-trials[0]["accuracy"], flush=True)
    manifest = {str(key): value for key, value in policies.items()}
    save_json(DEST / "policies/frozen_before_test.json", {
        "status": "FROZEN", "policies": manifest,
        "selection_data": "own source domain only; CA2014 for temporal, CA2018 for geographic",
        "sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()})
    return policies


def evaluate(policies):
    paper = paper_reference()
    rows = []
    for scenario in SCENARIOS:
        for task in TASKS:
            shared_policy = policies[("Shared", scenario, task)]
            specific_policy = policies[("Specific", scenario, task)]
            for role in ["ID", "OOD"]:
                base = prediction_for_member(scenario, task, role, "Base")
                shared = head(raw(scenario, task, role, shared_policy["source_policy"]), shared_policy["head_parameters"])
                expert = head(raw(scenario, task, role, specific_policy["source_policy"]), specific_policy["head_parameters"])
                specific = blend(shared, expert, specific_policy["beta"])
                assert np.array_equal(blend(shared, expert, 0.), shared)
                source_domain = "2018_CA" if scenario == "geographic" else "2014_CA"
                domain = source_domain if role == "ID" else "2018_MI" if scenario == "geographic" else "2018_CA"
                # This code runs only after all source-only policies are frozen.
                with np.load(RUN / "data/splits" / f"{domain}_{task}_test_labels.npz") as f:
                    y, row_ids = f["y"], f["row_ids"]
                for method, p in [("Base", base), ("Shared", shared), ("Specific", specific)]:
                    values = metric(y, p)
                    rows.append({"scenario": scenario, "method": method, "domain": role, "task": task,
                        **values, "auc": float(roc_auc_score(y, p[:, 1])), "rows": len(y), "seed": 0,
                        "paper_accuracy": paper[(scenario, method, role, task)],
                        "difference_from_paper": values["accuracy"]-paper[(scenario, method, role, task)],
                        "source_only_adaptation": method != "Base", "target_domain_calibration_labels_used": False,
                        "original_configuration_recovered": False})
                    directory = DEST / "predictions" / scenario / task
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory / f"{role}_{method}.npz", prediction=p, y=y, row_ids=row_ids)
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / "comparison_long.csv", index=False)
    wide = frame.pivot(index=["scenario", "method", "domain"], columns="task", values="accuracy").reindex(columns=TASKS)
    wide.to_csv(DEST / "accuracy_table.csv")
    save_json(DEST / "completion.json", {"status": "COMPLETE", "records": 36, "seed": 0,
        "exact_shared_fallback_available": True, "target_domain_calibration_labels_used": False,
        "strict_test_improvement_guaranteed": False, "source_selection_only": True,
        "test_labels_used_for_training_or_selection": False,
        "protocol": "exploratory reconstructed strict source-only nested adapter, not the original manuscript mapping"})
    print(wide.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    evaluate(fit_all())
