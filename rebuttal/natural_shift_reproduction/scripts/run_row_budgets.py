"""Feature-only inference for the user-approved larger context row budgets."""
import argparse
import json
import time
import numpy as np
import torch

from prepare_acs import RUN, save_json
from row_budget_views import BUDGETS, prepare_views
from run_natural import Predictor, digest, load


def run(scenario, task, budgets=(512,)):
    torch.set_num_threads(1); torch.manual_seed(0)
    source_domain = "2018_CA" if scenario == "geographic" else "2014_CA"
    target_domain = "2018_MI" if scenario == "geographic" else "2018_CA"
    prefix = RUN / "data/splits" / f"{source_domain}_{task}"
    source = load(prefix, "source")
    calibration = load(prefix, "calibration")
    meta = json.loads(prefix.with_suffix(".json").read_text())
    groups = {"fit": calibration["X"][:2048], "select": calibration["X"][2048:],
              "ID": load(prefix, "test_features")["X"],
              "OOD": load(RUN / "data/splits" / f"{target_domain}_{task}", "test_features")["X"]}
    destination = RUN / "results/row_banks" / scenario / task
    destination.mkdir(parents=True, exist_ok=True)
    predictor = Predictor()
    for role, Q in groups.items():
        generator = prepare_views(source["X"], source["y"], Q, meta["features"])
        for budget in budgets:
            for name, X, y, V, configuration in generator(budget):
                path = destination / f"n{budget}_{role}__{name}.npz"
                signature = digest(X, y, V)
                if path.exists() and path.with_suffix(".json").exists():
                    assert json.loads(path.with_suffix(".json").read_text())["input_sha256"] == signature
                    continue
                start = time.time()
                p = predictor.predict(X, y, V)
                ids = configuration.pop("source_row_ids")
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, prediction=p, context_source_pool_indices=ids)
                temp.replace(path)
                save_json(path.with_suffix(".json"), {
                    "scenario": scenario, "task": task, "role": role, "branch": name,
                    "row_budget": budget, "actual_context_rows": len(y),
                    "unique_source_pool_rows": len(np.unique(ids)), "features": X.shape[1],
                    "input_sha256": signature, "constructor": configuration, "seed": 0,
                    "test_labels_accessed": False, "seconds": time.time()-start,
                })
                print(scenario, task, role, budget, name, "rows", len(y), "seconds", round(time.time()-start, 2), flush=True)
    save_json(destination / "complete.json", {"status": "COMPLETE", "requested_budgets": list(budgets),
                                               "scenario": scenario, "task": task})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--budgets", default="512")
    args = parser.parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]
    if any(value not in BUDGETS for value in budgets):
        raise ValueError(f"Supported budgets: {BUDGETS}")
    run(args.scenario, args.task, budgets)
