"""Source-only policy selection, followed by held-out test evaluation.

Reference paper values are used solely in the final comparison report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from natural_components import SHARED_NAMES, apply_policy, fit_policy
from prepare_acs import RUN, save_json

SCENARIOS = ["geographic", "temporal"]
TASKS = ["Employment", "Income", "TravelTime"]
LAMBDAS = [.0001, .001, .01, .1]


def branch(path):
    with np.load(path, allow_pickle=False) as data:
        return data["prediction"]


def prediction_for_member(scenario, task, role, name):
    if name.startswith("rows_"):
        prefix, name = name.split("::", 1)
        budget = int(prefix.removeprefix("rows_"))
        return branch(RUN / "results/row_banks" / scenario / task / f"n{budget}_{role}__{name}.npz")
    return branch(RUN / "results/banks" / scenario / task / f"{role}__{name}.npz")


def calibration(scenario, task, role, names):
    year = 2018 if scenario == "geographic" else 2014
    prefix = RUN / "data/splits" / f"{year}_CA_{task}"
    with np.load(str(prefix) + "_calibration.npz", allow_pickle=False) as data:
        y = data["y"][:2048] if role == "fit" else data["y"][2048:]
    root = RUN / "results/banks" / scenario / task
    return {"y": y, "base": branch(root / f"{role}__Base.npz"),
            "bank": np.stack([branch(root / f"{role}__{name}.npz") for name in names])}


def choose(fit, select):
    candidates = []
    for strength in LAMBDAS:
        parameters, details = fit_policy(fit, strength)
        loss = np.mean([log_loss(c["y"], apply_policy(c["base"], c["bank"], parameters)[0], labels=[0, 1])
                        for c in select])
        candidates.append((float(loss), parameters, details))
    loss, parameters, details = min(candidates, key=lambda item: item[0])
    return parameters, {"selected": details, "source_selection_loss": loss,
                         "all_candidates": [{"source_selection_loss": x[0], **x[2]} for x in candidates]}


def fit_all():
    policies = {}
    for scenario in SCENARIOS:
        for task in TASKS:
            marker = RUN / "results/banks" / scenario / task / "bank_complete.json"
            if not marker.exists():
                raise RuntimeError(f"Bank incomplete: {scenario} {task}")
    # A Shared policy sees neither geographic/temporal identifiers nor target
    # labels. Task-specific source adaptation is explicit; both scenarios share
    # exactly the same five-member policy for a given prediction task.
    for task in TASKS:
        train = [calibration(s, task, "fit", SHARED_NAMES) for s in SCENARIOS]
        select = [calibration(s, task, "select", SHARED_NAMES) for s in SCENARIOS]
        parameters, details = choose(train, select)
        policies[("Shared", task)] = (parameters, SHARED_NAMES)
        save_json(RUN / "results/policies" / f"Shared_{task}.json", {
            "method": "Shared", "task": task, "members": SHARED_NAMES,
            "parameters": parameters.tolist(), "calibration_domains": ["CA2014", "CA2018"],
            "family_identifier_used": False, "test_labels_accessed": False,
            "source_only_adaptation": True, **details})
        print(f"FITTED Shared {task}: source selection CE={details['source_selection_loss']:.5f}", flush=True)
        for scenario in SCENARIOS:
            marker = RUN / "results/banks" / scenario / task / "bank_complete.json"
            definitions = json.loads(marker.read_text())["definitions"]
            names = list(definitions)
            train = [calibration(scenario, task, "fit", names)]
            select = [calibration(scenario, task, "select", names)]
            parameters, details = choose(train, select)
            policies[("Specific", scenario, task)] = (parameters, names)
            save_json(RUN / "results/policies" / f"Specific_{scenario}_{task}.json", {
                "method": "Specific", "scenario": scenario, "task": task, "members": names,
                "parameters": parameters.tolist(), "family_identifier_used": True,
                "test_labels_accessed": False, "source_only_adaptation": True,
                "extra_source_pool_access": True, **details})
            print(f"FITTED Specific {scenario} {task}: members={len(names)} source selection CE={details['source_selection_loss']:.5f}", flush=True)
    manifest = {str(k): {"parameters": p.tolist(), "members": names}
                for k, (p, names) in policies.items()}
    save_json(RUN / "results/policies/frozen_before_test.json", {
        "status": "FROZEN", "selection_data": "disjoint source calibration labels only",
        "policies": manifest, "sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    })
    return policies


def paper_reference():
    rows = [
        ("geographic", "Base", "ID", [.7922, .7888, .5912]),
        ("geographic", "Base", "OOD", [.8043, .7655, .5472]),
        ("geographic", "Shared", "ID", [.7948, .7951, .5945]),
        ("geographic", "Shared", "OOD", [.8078, .7832, .5961]),
        ("geographic", "Specific", "ID", [.7979, .7955, .6002]),
        ("geographic", "Specific", "OOD", [.8089, .7840, .6034]),
        ("temporal", "Base", "ID", [.7704, .7926, .5951]),
        ("temporal", "Base", "OOD", [.7847, .7844, .5908]),
        ("temporal", "Shared", "ID", [.7751, .7968, .6032]),
        ("temporal", "Shared", "OOD", [.7900, .7937, .5975]),
        ("temporal", "Specific", "ID", [.7757, .7980, .6061]),
        ("temporal", "Specific", "OOD", [.7916, .7947, .5981]),
    ]
    return {(s, m, d, task): value for s, m, d, values in rows for task, value in zip(TASKS, values)}


def evaluate(policies, destination=None):
    destination = RUN / "results" if destination is None else destination
    destination.mkdir(parents=True, exist_ok=True)
    paper = paper_reference()
    records = []
    for scenario in SCENARIOS:
        for task in TASKS:
            root = RUN / "results/banks" / scenario / task
            for role in ["ID", "OOD"]:
                base = branch(root / f"{role}__Base.npz")
                predictions = {"Base": (base, np.zeros(len(base)))}
                for method in ["Shared", "Specific"]:
                    key = (method, task) if method == "Shared" else (method, scenario, task)
                    parameters, names = policies[key]
                    bank = np.stack([prediction_for_member(scenario, task, role, name) for name in names])
                    predictions[method] = apply_policy(base, bank, parameters)
                # No policy fitting or selection occurs after opening this file.
                source = "2018_CA" if scenario == "geographic" else "2014_CA"
                domain = source if role == "ID" else "2018_MI" if scenario == "geographic" else "2018_CA"
                with np.load(RUN / "data/splits" / f"{domain}_{task}_test_labels.npz", allow_pickle=False) as data:
                    y, row_ids = data["y"], data["row_ids"]
                for method, (p, alpha) in predictions.items():
                    value = accuracy_score(y, p.argmax(-1))
                    record = {
                        "scenario": scenario, "method": method, "domain": role, "task": task,
                        "accuracy": float(value), "auc": float(roc_auc_score(y, p[:, 1])),
                        "log_loss": float(log_loss(y, p, labels=[0, 1])), "rows": len(y),
                        "paper_accuracy": paper[(scenario, method, role, task)],
                        "difference_from_paper": float(value - paper[(scenario, method, role, task)]),
                        "delta_from_current_base": float(value - accuracy_score(y, base.argmax(-1))),
                        "mean_gate_alpha": float(alpha.mean()), "seed": 0,
                        "source_only_adaptation": method != "Base" and not (method == "Shared" and scenario == "temporal"),
                        "target_domain_calibration_labels_used": method == "Shared" and scenario == "temporal",
                        "calibration_supervision_domains": "none" if method == "Base" else "CA2014+CA2018" if method == "Shared" else source,
                        "strict_own_source_domain_calibration": method == "Specific",
                        "original_configuration_recovered": False,
                    }
                    records.append(record)
                    directory = destination / "predictions" / scenario / task
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory / f"{role}_{method}.npz", prediction=p,
                                        gate_alpha=alpha, y=y, row_ids=row_ids)
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "comparison_long.csv", index=False)
    wide = frame.pivot(index=["scenario", "method", "domain"], columns="task", values="accuracy")
    wide = wide.reindex(columns=TASKS)
    wide.to_csv(destination / "accuracy_table.csv")
    save_json(destination / "completion.json", {
        "status": "COMPLETE", "records": len(frame), "seed": 0,
        "mean_absolute_difference_from_paper": float(frame.difference_from_paper.abs().mean()),
        "max_absolute_difference_from_paper": float(frame.difference_from_paper.abs().max()),
        "test_labels_used_for_training_or_selection": False,
        "original_configuration_recovered": False,
    })
    print(wide.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    evaluate(fit_all())
