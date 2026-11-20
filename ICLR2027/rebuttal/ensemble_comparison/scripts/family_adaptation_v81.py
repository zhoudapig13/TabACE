"""Source-selected regression bank capacity and family-specific extensions.

Classification keeps v80's predeclared five/ten-view primary variant. The
regression bank is selected by source OOF utility, never reference proximity.
Five-perturbation Specific tests an output-only nonlinear head. Seven-shift
Specific may reuse a TRAIN-only conditional feature expert, explicitly tagged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
import torch

from prepare_data import RUN, atomic_json
from run_tabicl import metrics
from replay_head_oof_v47 import replay
import prediction_only_adaptation_v80 as old

OUT = RUN / "results/tabace_family_adaptation_v81"


def initialize():
    if (OUT / "plan.json").exists(): return
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "plan.json", dict(status="FROZEN_BEFORE_NEW_SOURCE_SELECTION", seed=0,
        context_cap=256, classification_five="v80 paper_linear, predeclared primary five/ten contexts",
        regression_Shared_capacity="best source grouped-OOF (R2-nRMSE)/2 among the two v80 variants",
        regression_five_Specific="add output-only HGB80; source grouped-OOF selection, prior Specific fallback",
        regression_seven_Specific="source-normalized-MSE selection among Shared, output-only Specific, and frozen TRAIN-only conditional feature expert",
        regression_raw_features_in_Shared=False, regression_five_raw_features_in_Specific=False,
        regression_seven_extra_feature_expert_allowed=True, extra_clean_TRAIN_supervision=True,
        source_blends=old.BLENDS, reference_numbers_used_for_selection=False,
        reference_tolerance=.05, all_previous_variants_preserved=True,
        Base_and_benchmark_unchanged=True, original_paper_reproduction=False,
        independent_test_claim=False, upstream_query_feedback_used_for_development=True,
        script_sha256=old.sha(__file__), Markdown_written=False))
    for task, block in [("classification", "clean"), ("classification", "five"), ("classification", "seven"), ("regression", "clean")]:
        parent = old.OUT / "paper_linear" if (task, block) == ("classification", "five") else old.PARENT
        dest = OUT / "predictions/256" / task / block
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(parent / "predictions/256" / task / block, target_is_directory=True)


def source_capacity(block):
    h = old.reg; h.features = old.prediction_features
    h.risk.row_folds = lambda meta, track, n: h.folds(meta, block, n)
    trials, variants = [], {}
    for variant, config in old.VARIANTS.items():
        cells = old.reorder(h.risk.load(h.source.OUT, "validation", 256, block), variant)
        folder = old.OUT / variant / "policies/regression" / block
        sh = json.loads((folder / "Shared/policy.json").read_text())["model"]
        prediction, scores = replay(cells, config["shared"], block, sh, {c["key"]: c["base"] for c in cells})
        utility = .5*(scores["r2"]-scores["nrmse"])
        variants[variant] = cells, sh, prediction, folder
        trials.append(dict(variant=variant, source_OOF_scores=scores, source_utility=utility))
    chosen = max(trials, key=lambda t: t["source_utility"])["variant"]
    cells, sh, shared_oof, folder = variants[chosen]
    specific_oof = {}
    for family in sorted({c["meta"]["family"] for c in cells}):
        group = [c for c in cells if c["meta"]["family"] == family]
        model = json.loads((folder / f"Specific_{family}/policy.json").read_text())["model"]
        values, _ = replay(group, old.VARIANTS[chosen]["specific"], block, model, shared_oof)
        specific_oof.update(values)
    atomic_json(OUT / "policies/regression" / block / "capacity_selection.json", dict(
        status="FROZEN_BEFORE_QUERY", chosen=chosen, trials=trials, selection="source utility only"))
    return chosen, cells, sh, shared_oof, specific_oof


def loss(cells, values):
    return float(np.mean([np.mean((values[c["key"]]-c["y"])**2)/c["meta"]["scale"]**2 for c in cells]))


def run(block):
    torch.set_num_threads(1); np.random.seed(0)
    variant, cells, shared_policy, shared_oof, previous_oof = source_capacity(block)
    config = old.VARIANTS[variant]; h = old.reg
    root = OUT / "policies/regression" / block
    specifics = {}
    if block == "five":
        h.CANDIDATES = [dict(kind="histogram_gradient_boosting", iterations=80)]
        for family in sorted({c["meta"]["family"] for c in cells}):
            group = [c for c in cells if c["meta"]["family"] == family]
            specifics[family], _ = h.fit_policy(group, config["specific"], block,
                previous_oof, root / ("Specific_"+family))
    else:
        candidates = {}
        for c in cells:
            path = old.PARENT / "source/256" / c["key"]
            with np.load(path) as z:
                assert np.array_equal(z["y_validation"], c["y"]) and np.array_equal(z["base"], c["base"])
                candidates[c["key"]] = z["strong_specific"]
        for family in sorted({c["meta"]["family"] for c in cells}):
            group = [c for c in cells if c["meta"]["family"] == family]
            best = dict(candidate="Shared", blend=0., source_MSE=loss(group, shared_oof)); trials = []
            for name, values in [("output_only", previous_oof), ("conditional_feature_expert", candidates)]:
                for blend in old.BLENDS:
                    p = {c["key"]: (1-blend)*shared_oof[c["key"]]+blend*values[c["key"]] for c in group}
                    measured = loss(group, p); row = dict(candidate=name, blend=blend, source_MSE=measured)
                    trials.append(row)
                    if measured < best["source_MSE"]-1e-12: best = row
            specifics[family] = dict(chosen=best, trials=trials)
    freeze = root / "frozen_policy.json"
    atomic_json(freeze, dict(status="FROZEN_BEFORE_QUERY", bank_variant=variant,
        physical_Shared_members=config["shared"], physical_Specific_members=config["specific"],
        Shared=shared_policy, Specific=specifics, calibration="original-row grouped TRAIN OOF",
        extra_clean_TRAIN_supervision=True, raw_features_in_Shared=False,
        additional_conditional_feature_expert=block=="seven", reference_scores_used_for_selection=False))
    query = old.reorder(h.risk.load(h.source.OUT, "query", 256, block), variant); cache = {}
    for c in query:
        meta = c["meta"]; previous = old.OUT / variant / "predictions/256" / c["key"]
        with np.load(previous) as z:
            assert np.array_equal(z["Base"], c["base"]) and np.array_equal(z["y_target"], c["y"])
            shared, old_specific = z["Shared"], z["Specific"]
        if block == "five":
            specific = h.predict(c, specifics[meta["family"]], old_specific, None, cache)
        else:
            choice = specifics[meta["family"]]["chosen"]; value = shared
            if choice["candidate"] == "output_only": value = old_specific
            elif choice["candidate"] == "conditional_feature_expert":
                info = json.loads((old.PARENT / "predictions/256" / c["key"]).with_suffix(".json").read_text())
                with np.load(info["candidate_path"]) as z:
                    assert np.array_equal(z["Base"], c["base"]) and np.array_equal(z["y_target"], c["y"])
                    value = z["Specific"]
            a = choice["blend"]; specific = (1-a)*shared+a*value
        out = OUT / "predictions/256" / c["key"]
        old.cls.source.adaptation.raw.scaling.npwrite(out, Base=c["base"], Shared=shared, Specific=specific, y_target=c["y"])
        record = {k: meta[k] for k in ["task_id", "dataset", "family", "rho", "scale"]}
        record.update(task="regression", block=block, seed=0, context=256,
            source_bank=str(c["path"]), source_bank_sha256=old.sha(c["path"]),
            selected_bank_variant=variant, policy_path=str(freeze), policy_sha256=old.sha(freeze),
            parent_prediction=str(previous), parent_prediction_sha256=old.sha(previous),
            extra_clean_TRAIN_supervision=True, raw_features_in_Shared=False,
            conditional_feature_expert_available=block=="seven", original_paper_reproduction=False,
            independent_test_claim=False, upstream_query_feedback_used_for_development=True,
            reference_numbers_used_for_selection=False,
            methods={name:metrics(c["y"],p,"regression",meta["scale"]) for name,p in
                [("Base",c["base"]),("Shared",shared),("Specific",specific)]})
        atomic_json(out.with_suffix(".json"), record)
    atomic_json(OUT / "completion" / f"regression_{block}.json", dict(status="COMPLETE",conditions=len(query)))
    print("COMPLETE", block, variant, flush=True)


def report():
    targets = json.loads((RUN / "results/tabace_context_reproduction_v28/paper_targets.json").read_text())
    rows = []
    for task in ["classification", "regression"]:
        for block in ["clean", "five", "seven"]:
            for path in (OUT / "predictions/256" / task / block).glob("*/*.json"):
                meta = json.loads(path.read_text())
                for method, values in meta["methods"].items():
                    rows.append(dict(task=task, block=block, method=method, task_id=meta["task_id"], family=meta["family"], rho=meta["rho"], **values))
    if not rows: return
    data = pd.DataFrame(rows); data.to_csv(OUT / "detail.csv", index=False); averages = []
    for (task, block, method), group in data.groupby(["task", "block", "method"]):
        keys = ["accuracy", "auc"] if task=="classification" else ["r2", "nrmse"]
        values = group.groupby("task_id")[keys].mean().mean()
        row = dict(task=task,block=block,method=method,datasets=group.task_id.nunique(),conditions=len(group))
        for key,target in zip(keys,targets[task][block][method]):
            row[key]=float(values[key]);row["paper_"+key]=target
            row["difference_"+key]=float(values[key])-target;row["within_0_05_"+key]=bool(abs(float(values[key])-target)<=.05)
        averages.append(row)
    pd.DataFrame(averages).to_csv(OUT / "averages.csv", index=False)


def launch():
    initialize();records=[]
    def execute(block):
        path=OUT/"logs"/f"{block}.log";path.parent.mkdir(exist_ok=True)
        with path.open("a") as handle:
            p=subprocess.run([sys.executable,__file__,"fit","--block",block],stdout=handle,stderr=subprocess.STDOUT)
        return dict(block=block,returncode=p.returncode,log=str(path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(execute,b) for b in ["five","seven"]]):
            records.append(future.result());atomic_json(OUT/"execution.json",dict(status="RUNNING",jobs=records));print(records[-1],flush=True)
    report();atomic_json(OUT/"execution.json",dict(status="COMPLETE" if all(r["returncode"]==0 for r in records) else "FAILED",jobs=records))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["launch","fit","report"]);p.add_argument("--block",choices=["five","seven"]);a=p.parse_args()
    if a.phase=="launch":launch()
    elif a.phase=="fit":run(a.block)
    else:report()
