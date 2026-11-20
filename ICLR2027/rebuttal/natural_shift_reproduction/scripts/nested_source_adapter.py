"""Exploratory 512-row adaptation with exact Shared inheritance.

The old eight-view aggregator recomputes confidence/agreement after adding
members and therefore does not literally contain the old five-view mapping.
Here beta=0 returns the complete frozen Shared predictor bit-for-bit. Validation
may select that option; strict improvement on new test data is not guaranteed.

Shared pools CA2014 and CA2018 calibration rows, which includes labeled target
YEAR calibration for temporal evaluation, though no test rows. This historical
adapter is superseded by strict_nested_source.py. Paper accuracy values enter
only final reporting, never fitting or selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from fit_and_report import SCENARIOS, TASKS, paper_reference, prediction_for_member
from natural_components import apply_policy
from prepare_acs import RUN, save_json

OLD = RUN / "results/fixed_512_v6"
DEST = RUN / "results/nested_512_v7"
REGULARIZATIONS = [.001, .01, .1, 1.]
BETAS = [0., .125, .25, .5, .75, 1.]


def metric(y, p):
    return {"accuracy": float(accuracy_score(y, p.argmax(1))),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def head(p, theta):
    if np.array_equal(np.asarray(theta), np.zeros(2)):
        return p.copy()
    logit = np.log(np.clip(p[:, 1], 1e-10, 1.))-np.log(np.clip(p[:, 0], 1e-10, 1.))
    positive = expit(np.exp(theta[0])*logit+theta[1])
    return np.stack([1.-positive, positive], axis=1)


def fit_head(p, y, strength):
    logits = np.log(np.clip(p[:, 1], 1e-10, 1.))-np.log(np.clip(p[:, 0], 1e-10, 1.))
    def objective(theta):
        scale = np.exp(theta[0])
        value = scale*logits+theta[1]
        error = expit(value)-y
        loss = np.mean(np.logaddexp(0., value)-y*value)+strength*np.dot(theta, theta)
        gradient = np.array([np.mean(error*scale*logits), np.mean(error)])+2.*strength*theta
        return float(loss), gradient
    answer = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B",
                      bounds=[(-1.4, 1.4), (-2., 2.)], options={"maxiter": 150})
    assert answer.success and np.all(np.isfinite(answer.x))
    return answer.x, {"regularization": strength, "iterations": int(answer.nit), "success": bool(answer.success)}


def select_head(train, selection):
    train_p, train_y = np.concatenate([c["p"] for c in train]), np.concatenate([c["y"] for c in train])
    candidates = [(np.zeros(2), {"regularization": None, "kind": "identity"})]
    for strength in REGULARIZATIONS:
        theta, detail = fit_head(train_p, train_y, strength)
        candidates.append((theta, detail | {"kind": "source_logit_calibration"}))
    scored = []
    for theta, detail in candidates:
        values = [metric(cell["y"], head(cell["p"], theta)) for cell in selection]
        summary = {key: float(np.mean([entry[key] for entry in values])) for key in values[0]}
        scored.append((summary["accuracy"], -summary["log_loss"], theta, detail | summary))
    chosen = max(scored, key=lambda entry: (entry[0], entry[1]))
    return chosen[2], {"selected": chosen[3], "trials": [entry[3] | {"parameters": entry[2].tolist()} for entry in scored]}


def blend(shared, specific, beta):
    # Keep the fallback exact, rather than relying on floating-point algebra.
    if beta == 0.:
        return shared.copy()
    return (1.-beta)*shared+beta*specific


def old_policy(scenario, task, method):
    file = f"Shared_{task}.json" if method == "Shared" else f"Specific_{scenario}_{task}.json"
    record = json.loads((OLD / "policies" / file).read_text())
    assert record["selected_row_budget"] == 512
    assert len(record["members"]) == (5 if method == "Shared" else 8)
    return record


def raw_prediction(scenario, task, role, method):
    policy = old_policy(scenario, task, method)
    b = prediction_for_member(scenario, task, role, "Base")
    bank = np.stack([prediction_for_member(scenario, task, role, name) for name in policy["members"]])
    return apply_policy(b, bank, np.asarray(policy["parameters"]))[0]


def source_cell(scenario, task, role, method):
    domain = "2018_CA" if scenario == "geographic" else "2014_CA"
    with np.load(RUN / "data/splits" / f"{domain}_{task}_calibration.npz") as data:
        y = data["y"][:2048] if role == "fit" else data["y"][2048:]
    return {"p": raw_prediction(scenario, task, role, method), "y": y}


def fit_all():
    policies = {}
    for task in TASKS:
        train = [source_cell(s, task, "fit", "Shared") for s in SCENARIOS]
        selection = [source_cell(s, task, "select", "Shared") for s in SCENARIOS]
        theta, details = select_head(train, selection)
        shared_policy = {"head_parameters": theta.tolist(), "kind": "nested_source_adapter_v7",
                         "method": "Shared", "task": task, "test_labels_accessed": False,
                         "context_type_count": 5, "per_context_row_cap": 512,
                         "source_policy": old_policy("geographic", task, "Shared"),
                         "calibration_domains": ["2014_CA", "2018_CA"],
                         "target_year_calibration_labels_used_for_temporal": True, **details}
        save_json(DEST / "policies" / f"Shared_{task}.json", shared_policy)
        policies[("Shared", task)] = shared_policy
        print("Shared", task, "head", theta.tolist(), "source select", details["selected"], flush=True)
        for scenario in SCENARIOS:
            specific_train = [source_cell(scenario, task, "fit", "Specific")]
            specific_selection = [source_cell(scenario, task, "select", "Specific")]
            specific_theta, specific_details = select_head(specific_train, specific_selection)
            shared_cell = source_cell(scenario, task, "select", "Shared")
            shared_p = head(shared_cell["p"], theta)
            specific_p = head(specific_selection[0]["p"], specific_theta)
            y = shared_cell["y"]
            trials = [{"beta": beta, **metric(y, blend(shared_p, specific_p, beta))} for beta in BETAS]
            best = max(trials, key=lambda row: (row["accuracy"], -row["log_loss"], -row["beta"]))
            fallback = trials[0]
            assert best["accuracy"] >= fallback["accuracy"]
            assert np.array_equal(blend(shared_p, specific_p, 0.), shared_p)
            policy = {"method": "Specific", "scenario": scenario, "task": task,
                      "kind": "nested_source_adapter_v7", "head_parameters": specific_theta.tolist(),
                      "shared_head_parameters": theta.tolist(), "beta": best["beta"],
                      "source_policy": old_policy(scenario, task, "Specific"),
                      "context_type_count": 8, "per_context_row_cap": 512,
                      "head_selection": specific_details, "beta_selection_trials": trials,
                      "selected": best, "shared_fallback_source_validation": fallback,
                      "exact_shared_fallback_available": True,
                      "test_labels_accessed": False, "source_only_adaptation": scenario != "temporal",
                      "target_domain_calibration_labels_used": scenario == "temporal",
                      "strict_test_improvement_guaranteed": False}
            policies[("Specific", scenario, task)] = policy
            save_json(DEST / "policies" / f"Specific_{scenario}_{task}.json", policy)
            print("Specific", scenario, task, "beta", best["beta"], "validation delta", best["accuracy"]-fallback["accuracy"], flush=True)
    manifest = {str(key): value for key, value in policies.items()}
    save_json(DEST / "policies/frozen_before_test.json", {
        "status": "FROZEN", "selection_data": "pooled CA2014/CA2018 calibration fit/select; includes temporal target-year labels, never test labels",
        "policies": manifest, "sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()})
    return policies


def evaluate(policies):
    paper = paper_reference()
    rows = []
    checks = []
    for scenario in SCENARIOS:
        for task in TASKS:
            shared_policy = policies[("Shared", task)]
            specific_policy = policies[("Specific", scenario, task)]
            for role in ["ID", "OOD"]:
                base = prediction_for_member(scenario, task, role, "Base")
                shared = head(raw_prediction(scenario, task, role, "Shared"), shared_policy["head_parameters"])
                expert = head(raw_prediction(scenario, task, role, "Specific"), specific_policy["head_parameters"])
                specific = blend(shared, expert, specific_policy["beta"])
                assert np.array_equal(blend(shared, expert, 0.), shared)
                domain = ("2018_CA" if scenario == "geographic" else "2014_CA") if role == "ID" else ("2018_MI" if scenario == "geographic" else "2018_CA")
                # First test-label read after all nine policies have been frozen.
                with np.load(RUN / "data/splits" / f"{domain}_{task}_test_labels.npz") as f:
                    y, row_ids = f["y"], f["row_ids"]
                for method, p in [("Base", base), ("Shared", shared), ("Specific", specific)]:
                    summary = metric(y, p)
                    rows.append({"scenario": scenario, "method": method, "domain": role, "task": task,
                                 **summary, "auc": float(roc_auc_score(y, p[:, 1])), "rows": len(y), "seed": 0,
                                 "paper_accuracy": paper[(scenario, method, role, task)],
                                 "difference_from_paper": summary["accuracy"]-paper[(scenario, method, role, task)],
                                 "source_only_adaptation": method != "Base" and scenario != "temporal",
                                 "target_domain_calibration_labels_used": method != "Base" and scenario == "temporal",
                                 "calibration_supervision_domains": "none" if method == "Base" else "CA2014+CA2018",
                                 "strict_own_source_domain_calibration": False,
                                 "original_configuration_recovered": False})
                    path = DEST / "predictions" / scenario / task / f"{role}_{method}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, prediction=p, y=y, row_ids=row_ids)
                checks.append({"scenario": scenario, "task": task, "domain": role,
                               "specific_minus_shared_accuracy": float(np.mean(specific.argmax(1)==y)-np.mean(shared.argmax(1)==y)),
                               "beta": specific_policy["beta"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / "comparison_long.csv", index=False)
    wide = frame.pivot(index=["scenario", "method", "domain"], columns="task", values="accuracy").reindex(columns=TASKS)
    wide.to_csv(DEST / "accuracy_table.csv")
    save_json(DEST / "completion.json", {"status": "COMPLETE", "records": len(rows), "seed": 0,
        "protocol": "exploratory source-calibrated nested adapter, not original manuscript mapping",
        "exact_shared_fallback_available": True, "strict_test_improvement_guaranteed": False,
        "source_selection_only": False, "pooled_calendar_domain_calibration": True,
        "test_labels_used_for_training_or_selection": False,
        "test_comparisons": checks})
    print(wide.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    evaluate(fit_all())
