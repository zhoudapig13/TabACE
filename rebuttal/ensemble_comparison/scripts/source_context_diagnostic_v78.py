"""Source-side perturbation reconstruction, separately from v77 query shifts.

Three existing protocols are declared together, not chosen by test performance:
classification corrected source corruption; classification source-calibrated
corruption; regression original source corruption. Neither former protocol is
an exact recovery of the historical generator. All are reported, including
their own paired Base, with no forced ordering or target metric matching.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
import torch

from fit_shared_reference import load_cells
from prepare_data import RUN, atomic_json
from run_tabicl import metrics
import paper_rule_reconstruction_v77 as original
import tabace_fusion_v28 as rule

OUT = RUN / "results/tabace_source_context_diagnostic_v78"
SETTINGS = {"classification_source_calibrated": ("classification", "five_calibrated"),
            "classification_source_corrected": ("classification", "five_corrected"),
            "regression_source_original": ("regression", "five")}


def load(setting):
    task, track = SETTINGS[setting]
    common, _ = load_cells("tabicl", track, task)
    directed, _ = load_cells("tabicl", track, task, variant="specific")
    index = {(int(c["meta"]["task_id"]), c["meta"]["family"], int(c["meta"]["rho"])): c for c in common}
    result = []
    for extra in directed:
        meta = extra["meta"]
        key = (int(meta["task_id"]), meta["family"] if meta["rho"] else "original", int(meta["rho"]))
        shared = index[key]
        assert np.array_equal(shared["y"], extra["y"]), key
        assert np.array_equal(shared["base"], extra["base"]), key
        result.append({"base": shared["base"], "bank": np.concatenate([shared["bank"], extra["bank"]]),
            "y": shared["y"], "mu": shared["source_mean"], "sigma": shared["source_scale"],
            "meta": meta | {"block": "clean" if meta["rho"] == 0 else "five"},
            "fold": shared["fold"], "task": task, "role": "validation",
            "path": shared["path"], "specific_path": extra["path"]})
    expected = (72 if task == "classification" else 35) * 25
    assert len(result) == expected
    return result


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "FROZEN_BEFORE_FITTING", "settings": SETTINGS, "seed": 0,
        "outer_folds": 5, "Shared_branches": 5, "Specific_total_branches": 10,
        "gain": 1., "regularization_grid": list(original.REGULARIZATIONS),
        "calibration": "labeled episodes from other outer-training datasets; excluded evaluation data_id",
        "Shared_family_blind": True, "Specific_family_visible": True, "strength_is_policy_input": False,
        "query_labels_from_evaluated_dataset_used_for_fitting": False,
        "source_contexts_corrupted": True, "original_unmodified_queries_retained": True,
        "protocols_selected_by_test_performance_this_run": False,
        "all_three_declared_settings_reported": True, "Base_always_paired_to_own_protocol": True,
        "context_rows": "classification <=256; regression <=2048, following existing source-side caches",
        "new_auxiliary_supervised_model": False, "new_output_calibration": False,
        "original_paper_generator_recovered": False,
        "classification_source_calibrated_previously_used_development_feedback": True,
        "independent_test_claim_for_development_calibrated_protocol": False,
        "Markdown_written": False,
        "code_sha256": {name: original.file_sha(Path(__file__).with_name(name)) for name in
            [Path(__file__).name, "paper_rule_reconstruction_v77.py", "tabace_fusion_v28.py",
             "fit_shared_reference.py", "calibrated_five_protocol.py", "corrected_source_protocol.py"]}}
    if not (OUT / "plan.json").exists():
        atomic_json(OUT / "plan.json", plan)


def worker(setting, fold):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    rule.GATE_VARIANT = "entropy"
    task, track = SETTINGS[setting]
    folder = OUT / setting / f"fold_{fold}"
    if (folder / "completion.json").exists():
        return
    cells = load(setting)
    training = [c for c in cells if c["fold"] != fold]
    test = [c for c in cells if c["fold"] == fold]
    train_ids = {int(c["meta"]["data_id"]) for c in training}
    test_ids = {int(c["meta"]["data_id"]) for c in test}
    assert train_ids.isdisjoint(test_ids)
    inner = min(c["fold"] for c in training)
    shared, inner_shared, details = original.select(training, task, 5, inner,
                                                   {"kind": "base"}, {"kind": "base"})
    specific, selections = {}, {}
    for family in sorted({c["meta"]["family"] for c in training}):
        group = [c for c in training if c["meta"]["family"] == family]
        specific[family], _, selections[family] = original.select(group, task, 10, inner, inner_shared, shared)
    atomic_json(folder / "frozen_policy.json", {"Shared": shared, "Specific": specific,
        "Shared_selection": details, "Specific_selection": selections,
        "training_data_ids": sorted(train_ids), "excluded_data_ids": sorted(test_ids),
        "task": task, "track": track, "seed": 0, "outer_fold": fold,
        "plan_sha256": original.file_sha(OUT / "plan.json")})
    rows = []
    for cell in test:
        meta = cell["meta"]
        predictions = {"Base": cell["base"], "Shared": rule.predict(cell, shared),
                       "Specific": rule.predict(cell, specific[meta["family"]])}
        path = OUT / setting / "predictions" / str(meta["task_id"]) / f"{meta['family']}_{meta['rho']}.npz"
        original.banks.scaling.npwrite(path, **predictions, y_target=cell["y"])
        for method, prediction in predictions.items():
            values = metrics(cell["y"], prediction, task, meta["scale"])
            rows.append({"setting": setting, "task": task, "block": meta["block"], "method": method,
                "data_id": int(meta["data_id"]), "task_id": int(meta["task_id"]), "dataset": meta["dataset"],
                "family": meta["family"], "rho": int(meta["rho"]), "fold": fold,
                "shared_bank": str(cell["path"]), "specific_bank": str(cell["specific_path"]),
                "scale": meta["scale"], **values})
    pd.DataFrame(rows).to_csv(folder / "detail.csv", index=False)
    atomic_json(folder / "completion.json", {"status": "COMPLETE", "conditions": len(test),
        "setting": setting, "outer_fold": fold, "training_and_test_datasets_disjoint": True})
    print("COMPLETE", setting, fold, len(test), flush=True)


def report():
    files = sorted(OUT.glob("*/fold_*/detail.csv"))
    if not files:
        return
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    data.to_csv(OUT / "detail.csv", index=False)
    rows = []
    for (setting, task, block, method), group in data.groupby(["setting", "task", "block", "method"]):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        average = group.groupby("data_id")[keys].mean().mean()
        rows.append({"setting": setting, "task": task, "block": block, "method": method,
                     "datasets": group.data_id.nunique(), "conditions": len(group), **average.to_dict()})
    pd.DataFrame(rows).to_csv(OUT / "averages.csv", index=False)


def launch():
    initialize()
    records = []
    def execute(setting, fold):
        log = OUT / "logs" / f"{setting}_{fold}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as handle:
            status = subprocess.run([sys.executable, __file__, "worker", "--setting", setting, "--fold", str(fold)],
                                    stdout=handle, stderr=subprocess.STDOUT)
        return {"setting": setting, "fold": fold, "returncode": status.returncode, "log": str(log)}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(execute, setting, fold) for fold in range(5) for setting in SETTINGS]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            atomic_json(OUT / "execution.json", {"status": "RUNNING", "workers": records})
            report()
            print(record, flush=True)
    atomic_json(OUT / "execution.json", {"status": "COMPLETE" if all(r["returncode"] == 0 for r in records) else "FAILED", "workers": records})
    report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["launch", "worker", "report"])
    parser.add_argument("--setting", choices=list(SETTINGS))
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.phase == "launch":
        launch()
    elif args.phase == "worker":
        worker(args.setting, args.fold)
    else:
        report()
