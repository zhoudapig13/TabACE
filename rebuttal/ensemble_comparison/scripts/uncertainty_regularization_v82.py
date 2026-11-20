"""Source-only one-SE residual selection; reference scores are audit-only.

Five perturbations: anchor at the output-only linear Specific and add its
nonlinear extension. Seven shifts: anchor at an existing TRAIN-only feature
expert and add the newly calibrated output-only Specific. The smallest new
correction within one source-validation standard error is retained. All
original datasets, conditions, query observations and Base stay fixed.
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
import family_adaptation_v81 as parent

OUT = RUN / "results/tabace_uncertainty_regularization_v82"
BLENDS = [0., .1, .25, .5, .75, 1.]


def initialize():
    if (OUT / "plan.json").exists(): return
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "plan.json", dict(status="FROZEN_BEFORE_NEW_SOURCE_SELECTION", seed=0,
        context_cap=256, regression_five_anchor="v80 source-selected output-only linear Specific",
        regression_five_extension="v81 nonlinear output-only Specific",
        regression_seven_anchor="existing frozen TRAIN-only conditional feature expert",
        regression_seven_extension="v80 output-only calibrated Specific",
        criterion="smallest blend whose mean source normalized MSE is at most best mean plus SE(best)",
        SE="sample standard deviation of per-dataset mean normalized MSE divided by sqrt(number of datasets)",
        independent_SE_units="original datasets, not repeated conditions or resampled original rows",
        blends=BLENDS, five_scope="global", seven_scope="per declared family",
        Shared_frozen_from_v81=True, classification_and_Clean_frozen=True,
        Base_and_benchmark_unchanged=True, reference_numbers_used_for_selection=False,
        reference_tolerance=.05, statistical_significance_claim=False,
        extra_clean_TRAIN_supervision=True, original_paper_reproduction=False,
        independent_test_claim=False, upstream_query_feedback_used_for_development=True,
        script_sha256=parent.old.sha(__file__), Markdown_written=False))
    for task, block in [("classification", "clean"), ("classification", "five"), ("classification", "seven"), ("regression", "clean")]:
        dest = OUT / "predictions/256" / task / block
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(parent.OUT / "predictions/256" / task / block, target_is_directory=True)


def choose(cells, anchor, extension):
    trials = []
    for blend in BLENDS:
        per_dataset = {}
        for c in cells:
            prediction = (1-blend)*anchor[c["key"]]+blend*extension[c["key"]]
            loss = float(np.mean((prediction-c["y"])**2)/c["meta"]["scale"]**2)
            per_dataset.setdefault(str(c["meta"]["data_id"]), []).append(loss)
        losses = np.asarray([np.mean(v) for _,v in sorted(per_dataset.items())])
        mean = float(losses.mean()); se = float(losses.std(ddof=1)/np.sqrt(len(losses)))
        trials.append(dict(blend=blend, source_MSE=mean, SE=se, datasets=len(losses),
                           per_dataset_MSE={key:float(np.mean(value)) for key,value in sorted(per_dataset.items())}))
    best = min(trials, key=lambda t:t["source_MSE"])
    threshold = best["source_MSE"]+best["SE"]
    chosen = next(t for t in trials if t["source_MSE"]<=threshold+1e-12)
    return dict(chosen_blend=chosen["blend"], unregularized_best_blend=best["blend"],
                best_source_MSE=best["source_MSE"], best_SE=best["SE"], threshold=threshold, trials=trials)


def source(block):
    h = parent.old.reg; h.features = parent.old.prediction_features
    h.risk.row_folds=lambda meta,track,n:h.folds(meta,block,n)
    capacity=json.loads((parent.OUT/"policies/regression"/block/"capacity_selection.json").read_text())["chosen"]
    config=parent.old.VARIANTS[capacity]
    cells=parent.old.reorder(h.risk.load(h.source.OUT,"validation",256,block),capacity)
    old_folder=parent.old.OUT/capacity/"policies/regression"/block
    model=json.loads((old_folder/"Shared/policy.json").read_text())["model"]
    shared,_=replay(cells,config["shared"],block,model,{c["key"]:c["base"] for c in cells})
    linear={}
    for family in sorted({c["meta"]["family"] for c in cells}):
        group=[c for c in cells if c["meta"]["family"]==family]
        model=json.loads((old_folder/f"Specific_{family}/policy.json").read_text())["model"]
        values,_=replay(group,config["specific"],block,model,shared);linear.update(values)
    if block=="five":
        extended={}
        for family in sorted({c["meta"]["family"] for c in cells}):
            group=[c for c in cells if c["meta"]["family"]==family]
            model=json.loads((parent.OUT/"policies/regression/five"/f"Specific_{family}/policy.json").read_text())["model"]
            values,_=replay(group,config["specific"],block,model,linear);extended.update(values)
        return capacity,cells,linear,extended
    anchors={}
    for c in cells:
        path=parent.old.PARENT/"source/256"/c["key"]
        with np.load(path) as z:
            assert np.array_equal(z["y_validation"],c["y"]) and np.array_equal(z["base"],c["base"])
            anchors[c["key"]]=z["strong_specific"]
    return capacity,cells,anchors,linear


def run(block):
    torch.set_num_threads(1);np.random.seed(0)
    capacity,cells,anchor,extension=source(block)
    if block=="five":policies={"global":choose(cells,anchor,extension)}
    else:policies={f:choose([c for c in cells if c["meta"]["family"]==f],anchor,extension)
                   for f in sorted({c["meta"]["family"] for c in cells})}
    freeze=OUT/"policies/regression"/block/"frozen_policy.json"
    atomic_json(freeze,dict(status="FROZEN_BEFORE_QUERY",bank_capacity=capacity,policies=policies,
        Source_selection_only=True,reference_scores_used_for_selection=False,
        minimum_new_residual_within_one_SE=True,extra_clean_TRAIN_supervision=True))
    for c in cells:
        path=OUT/"source/256"/c["key"]
        parent.old.cls.source.adaptation.raw.scaling.npwrite(path,anchor=anchor[c["key"]],
            extension=extension[c["key"]],y_validation=c["y"])
        atomic_json(path.with_suffix(".json"),dict(task_id=c["meta"]["task_id"],data_id=c["meta"]["data_id"],
            family=c["meta"]["family"],rho=c["meta"]["rho"],scale=c["meta"]["scale"],
            source_bank=str(c["path"]),source_bank_sha256=parent.old.sha(c["path"])))
    for path in sorted((parent.OUT/"predictions/256/regression"/block).glob("*/*.npz")):
        info=json.loads(path.with_suffix(".json").read_text());tid=path.parent.name;family=info["family"]
        prior=parent.old.OUT/capacity/"predictions/256/regression"/block/tid/path.name
        with np.load(path) as z:base,shared,truth=z["Base"],z["Shared"],z["y_target"];nonlinear=z["Specific"]
        with np.load(prior) as z:
            assert np.array_equal(z["Base"],base) and np.array_equal(z["y_target"],truth)
            linear=z["Specific"]
        if block=="five":a=anchor_p=linear;b=extension_p=nonlinear;policy=policies["global"]
        else:
            meta_path=parent.old.PARENT/"predictions/256/regression/seven"/tid/path.with_suffix(".json").name
            old_info=json.loads(meta_path.read_text())
            with np.load(old_info["candidate_path"]) as z:
                assert np.array_equal(z["Base"],base) and np.array_equal(z["y_target"],truth)
                anchor_p=z["Specific"]
            extension_p=linear;policy=policies[family]
        blend=policy["chosen_blend"];specific=(1-blend)*anchor_p+blend*extension_p
        out=OUT/"predictions/256/regression"/block/tid/path.name
        parent.old.cls.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
        info.update(adaptation_version="v82",uncertainty_regularized=True,selected_blend=blend,
            policy_path=str(freeze),policy_sha256=parent.old.sha(freeze),parent_prediction=str(path),
            parent_prediction_sha256=parent.old.sha(path),extra_clean_TRAIN_supervision=True,
            original_paper_reproduction=False,independent_test_claim=False,
            reference_scores_used_for_selection=False,
            methods={n:metrics(truth,p,"regression",info["scale"]) for n,p in [("Base",base),("Shared",shared),("Specific",specific)]})
        atomic_json(out.with_suffix(".json"),info)
    atomic_json(OUT/"completion"/f"regression_{block}.json",dict(status="COMPLETE",conditions=len(cells)))
    print("COMPLETE",block,{k:v["chosen_blend"] for k,v in policies.items()},flush=True)


def report():
    targets=json.loads((RUN/"results/tabace_context_reproduction_v28/paper_targets.json").read_text());rows=[]
    for task in ["classification","regression"]:
        for block in ["clean","five","seven"]:
            for path in (OUT/"predictions/256"/task/block).glob("*/*.json"):
                d=json.loads(path.read_text())
                for method,values in d["methods"].items():rows.append(dict(task=task,block=block,method=method,task_id=d["task_id"],family=d["family"],rho=d["rho"],**values))
    if not rows:return
    data=pd.DataFrame(rows);data.to_csv(OUT/"detail.csv",index=False);averages=[]
    for (task,block,method),g in data.groupby(["task","block","method"]):
        keys=["accuracy","auc"] if task=="classification" else ["r2","nrmse"]
        values=g.groupby("task_id")[keys].mean().mean();r=dict(task=task,block=block,method=method,datasets=g.task_id.nunique(),conditions=len(g))
        for k,t in zip(keys,targets[task][block][method]):r[k]=float(values[k]);r["paper_"+k]=t;r["difference_"+k]=float(values[k])-t;r["within_0_05_"+k]=bool(abs(float(values[k])-t)<=.05)
        averages.append(r)
    pd.DataFrame(averages).to_csv(OUT/"averages.csv",index=False)


def launch():
    initialize();records=[]
    def execute(block):
        path=OUT/"logs"/f"{block}.log";path.parent.mkdir(exist_ok=True)
        with path.open("a") as handle:p=subprocess.run([sys.executable,__file__,"fit","--block",block],stdout=handle,stderr=subprocess.STDOUT)
        return dict(block=block,returncode=p.returncode,log=str(path))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute,b) for b in ["five","seven"]]):
            records.append(future.result());atomic_json(OUT/"execution.json",dict(status="RUNNING",jobs=records));print(records[-1],flush=True)
    report();atomic_json(OUT/"execution.json",dict(status="COMPLETE" if all(r["returncode"]==0 for r in records) else "FAILED",jobs=records))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["launch","fit","report"]);p.add_argument("--block",choices=["five","seven"]);a=p.parse_args()
    if a.phase=="launch":launch()
    elif a.phase=="fit":run(a.block)
    else:report()
