"""Select canonical versus expanded source-pool adaptations on source data.

All candidate trials, including inferior choices, are saved. No target test
labels enter selection. Old results remain in the parent results directory.
"""
import hashlib
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import log_loss

from fit_and_report import SCENARIOS, TASKS, choose, calibration, evaluate
from natural_components import SHARED_NAMES
from prepare_acs import RUN, save_json

DEST = RUN / "results/expanded_adapter_v2"


def select_banks(task, variants, scenarios):
    tried = []
    for label, names in variants:
        fit = [calibration(s, task, "fit", names) for s in scenarios]
        select = [calibration(s, task, "select", names) for s in scenarios]
        parameters, details = choose(fit, select)
        tried.append((details["source_selection_loss"], parameters, names, label, details))
    best = min(tried, key=lambda entry: entry[0])
    loss, parameters, names, label, details = best
    return parameters, names, {
        "selected_bank": label, "source_selection_loss": loss, "selected": details,
        "bank_trials": [{"bank": x[3], "members": x[2], **x[4]} for x in tried],
        "selection_data": "source calibration select partition only",
        "test_labels_accessed": False, "source_only_adaptation": True,
    }


def main(coverage=False):
    global DEST
    prefix = "coverage" if coverage else "expanded"
    DEST = RUN / "results" / ("coverage_adapter_v3" if coverage else "expanded_adapter_v2")
    torch.set_num_threads(1)
    policies = {}
    for scenario in SCENARIOS:
        for task in TASKS:
            assert (RUN / "results/banks" / scenario / task / f"{prefix}_complete.json").exists()
    for task in TASKS:
        variants = [("canonical", SHARED_NAMES)] + [
            (prefix + "_" + mode, [f"{prefix}_{mode}_{name}" for name in SHARED_NAMES])
            for mode in ["rank", "strat"]]
        if coverage:
            # User explicitly permits moderate member expansion. These three
            # family-agnostic source resamples are cached for both scenarios.
            variants += [(label+"_8", names+[f"source_resample_{i}" for i in range(3)])
                         for label, names in list(variants)]
        p, names, details = select_banks(task, variants, SCENARIOS)
        policies[("Shared", task)] = (p, names)
        save_json(DEST / "policies" / f"Shared_{task}.json", {
            "method": "Shared", "task": task, "members": names, "parameters": p.tolist(),
            "family_identifier_used": False, "source_pool_rows": 2048,
            "per_member_context_cap": 256, "actual_member_count": len(names),
            "candidate_member_counts": [5, 8] if coverage else [5], **details})
        print(f"Shared {task}: {details['selected_bank']} validation CE {details['source_selection_loss']:.5f}", flush=True)
        for scenario in SCENARIOS:
            marker = RUN / "results/banks" / scenario / task / "bank_complete.json"
            original = list(json.loads(marker.read_text())["definitions"])
            expanded = original + [f"{prefix}_{mode}_{name}" for mode in ["rank", "strat"] for name in SHARED_NAMES]
            p, names, details = select_banks(task, [("original_specific", original), (prefix+"_specific", expanded)], [scenario])
            policies[("Specific", scenario, task)] = (p, names)
            save_json(DEST / "policies" / f"Specific_{scenario}_{task}.json", {
                "method": "Specific", "scenario": scenario, "task": task, "members": names,
                "parameters": p.tolist(), "family_identifier_used": True,
                "source_pool_rows": 2048, "per_member_context_cap": 256,
                "actual_member_count": len(names), **details})
            print(f"Specific {scenario} {task}: {details['selected_bank']} validation CE {details['source_selection_loss']:.5f}", flush=True)
    manifest = {str(k): {"parameters": p.tolist(), "members": names}
                for k, (p, names) in policies.items()}
    save_json(DEST / "policies/frozen_before_test.json", {
        "status": "FROZEN", "policies": manifest,
        "selection_data": "source calibration labels only",
        "sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    })
    evaluate(policies, DEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", action="store_true")
    main(parser.parse_args().coverage)
