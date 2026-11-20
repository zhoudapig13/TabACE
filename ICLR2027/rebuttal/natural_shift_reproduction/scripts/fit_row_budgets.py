"""Select context row budgets, holding Shared/Specific type counts at 5/8."""
from __future__ import annotations

import hashlib
import json
import argparse
import numpy as np
import torch

from fit_and_report import SCENARIOS, TASKS, choose, evaluate, prediction_for_member
from natural_components import SHARED_NAMES
from prepare_acs import RUN, save_json
from row_budget_views import BUDGETS, SPECIFIC_EXTRA_NAMES

DEST = RUN / "results/row_budget_adapter_v4"


def calibration(scenario, task, role, names):
    domain = "2018_CA" if scenario == "geographic" else "2014_CA"
    with np.load(RUN / "data/splits" / f"{domain}_{task}_calibration.npz") as data:
        y = data["y"][:2048] if role == "fit" else data["y"][2048:]
    return {"y": y, "base": prediction_for_member(scenario, task, role, "Base"),
            "bank": np.stack([prediction_for_member(scenario, task, role, name) for name in names])}


def select_budget(task, scenarios, types, budgets=None):
    trials = []
    for budget in BUDGETS if budgets is None else budgets:
        names = [f"rows_{budget}::{name}" for name in types]
        fit = [calibration(scenario, task, "fit", names) for scenario in scenarios]
        select = [calibration(scenario, task, "select", names) for scenario in scenarios]
        parameters, details = choose(fit, select)
        trials.append((details["source_selection_loss"], budget, parameters, names, details))
    loss, budget, parameters, names, details = min(trials, key=lambda item: item[0])
    actual = {}
    for scenario in scenarios:
        actual[scenario] = {}
        for role in ["ID", "OOD"]:
            actual[scenario][role] = {}
            for name in types:
                p = RUN / "results/row_banks" / scenario / task / f"n{budget}_{role}__{name}.json"
                actual[scenario][role][name] = json.loads(p.read_text())["actual_context_rows"]
    return parameters, names, {
        "selected_row_budget": budget, "actual_member_count": len(types), "context_types": types,
        "actual_rows_by_role_and_type": actual, "selected": details,
        "source_selection_loss": loss,
        "row_budget_trials": [{"row_budget": trial[1], **trial[4]} for trial in trials],
        "source_pool_rows": 2048, "source_only_adaptation": True,
        "selection_data": "disjoint source calibration selection rows only",
        "test_labels_accessed": False,
    }


def main(matched=False, fixed_rows=None):
    global DEST
    if fixed_rows is not None:
        if fixed_rows not in BUDGETS:
            raise ValueError(f"Fixed budget must be one of {BUDGETS}")
        matched = True
    variant = f"fixed_{fixed_rows}_v6" if fixed_rows is not None else "matched_row_budget_v5" if matched else "row_budget_adapter_v4"
    DEST = RUN / "results" / variant
    torch.set_num_threads(1)
    for scenario in SCENARIOS:
        for task in TASKS:
            assert (RUN / "results/row_banks" / scenario / task / "complete.json").exists()
    policies = {}
    for task in TASKS:
        p, names, details = select_budget(task, SCENARIOS, SHARED_NAMES,
                                          [fixed_rows] if fixed_rows is not None else None)
        policies[("Shared", task)] = (p, names)
        save_json(DEST / "policies" / f"Shared_{task}.json", {
            "method": "Shared", "task": task, "members": names, "parameters": p.tolist(),
            "family_identifier_used": False, **details})
        print(f"Shared {task}: 5 types; max rows {details['selected_row_budget']}; source CE {details['source_selection_loss']:.6f}", flush=True)
        shared_budget = details["selected_row_budget"]
        for scenario in SCENARIOS:
            p, names, details = select_budget(task, [scenario], SHARED_NAMES+SPECIFIC_EXTRA_NAMES,
                                              [shared_budget] if matched else None)
            policies[("Specific", scenario, task)] = (p, names)
            save_json(DEST / "policies" / f"Specific_{scenario}_{task}.json", {
                "method": "Specific", "scenario": scenario, "task": task,
                "members": names, "parameters": p.tolist(), "family_identifier_used": True,
                "row_budget_matched_to_shared": matched, **details})
            print(f"Specific {scenario} {task}: 8 types; max rows {details['selected_row_budget']}; source CE {details['source_selection_loss']:.6f}", flush=True)
    manifest = {str(key): {"parameters": p.tolist(), "members": names}
                for key, (p, names) in policies.items()}
    save_json(DEST / "policies/frozen_before_test.json", {
        "status": "FROZEN", "selection_data": "source calibration only",
        "policies": manifest, "sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    })
    evaluate(policies, DEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--fixed-rows", type=int, default=None)
    args = parser.parse_args()
    main(args.matched, args.fixed_rows)
