"""Recompute the source selection, all 4,151 outcomes, and reference errors."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from prepare_data import atomic_json
from run_tabicl import metrics
import conservative_development_v83 as current


def audit():
    started=time.time();root=current.OUT;prior=current.prior;old=prior.parent.old
    plan=json.loads((root/"plan.json").read_text());assert old.sha(Path(current.__file__))==plan["script_sha256"]
    source_conditions=0
    for block in ["five","seven"]:
        p=root/"policies/regression"/block/"frozen_policy.json";policy=json.loads(p.read_text())
        assert old.sha(policy["source_policy"])==policy["source_policy_sha256"]
        packets=[]
        for path in sorted((prior.OUT/"source/256/regression"/block).glob("*/*.npz")):
            m=json.loads(path.with_suffix(".json").read_text())
            assert old.sha(m["source_bank"])==m["source_bank_sha256"]
            with np.load(path) as z:packets.append((m,{k:z[k] for k in z.files}))
        assert len(packets)==(700 if block=="five" else 924);source_conditions+=len(packets)
        for name,choice in policy["policies"].items():
            scope=packets if name=="global" else [(m,z) for m,z in packets if m["family"]==name]
            recomputed=[]
            for trial in choice["trials"]:
                by_dataset={};a=trial["blend"]
                for meta,z in scope:
                    prediction=(1-a)*z["anchor"]+a*z["extension"]
                    loss=float(np.mean((prediction-z["y_validation"])**2)/meta["scale"]**2)
                    by_dataset.setdefault(meta["data_id"],[]).append(loss)
                values=np.array([np.mean(v) for v in by_dataset.values()])
                mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
                assert abs(mean-trial["source_MSE"])<1e-12 and abs(se-trial["SE"])<1e-12
                recomputed.append(dict(blend=a,source_MSE=mean,SE=se))
            best=min(recomputed,key=lambda t:t["source_MSE"])
            threshold=best["source_MSE"]+current.MULTIPLIER*best["SE"]
            first=next(t for t in recomputed if t["source_MSE"]<=threshold+1e-12)
            assert first["blend"]==choice["chosen_blend"] and abs(threshold-choice["threshold"])<1e-12
    # Check every classification checkpoint and its original-row cross-fit IDs.
    model_hashes=0
    source_root=old.OUT/"paper_linear/classification_models"
    for meta_path in source_root.glob("*/*/query.json"):
        m=json.loads(meta_path.read_text());path=meta_path.with_suffix(".pt")
        assert old.sha(path)==m["model_sha256"]
        members=5 if meta_path.parent.name=="Shared" else 10;assert m["members"]==members
        assert m["profile"]["hidden"]==0
        train=set(m["training_original_ids"])
        split=old.cls.source.adaptation.raw.OUT/"validation_splits/classification/five"/f"{meta_path.parents[1].name}.npz"
        with np.load(split) as z:assert train.issubset(set(z["validation_original_ids"].tolist()))
        for record in m["original_row_crossfits"]:
            assert set(record["training_original_ids"]).isdisjoint(record["held_original_ids"])
            assert set(record["training_original_ids"])|set(record["held_original_ids"])==train
            pt=meta_path.parent/f"validation_{record['fold']}.pt"
            assert old.sha(pt)==record["model_sha256"];model_hashes+=1
        model_hashes+=1
    assert model_hashes==72*6*5
    rows=[];count=0;max_error=0.;counts={}
    base_root=old.PARENT/"predictions/256"
    for task in ["classification","regression"]:
        for block in ["clean","five","seven"]:
            files=sorted((root/"predictions/256"/task/block).glob("*/*.json"));counts[task+"/"+block]=len(files)
            expected={"classification":{"clean":72,"five":1440,"seven":980},"regression":{"clean":35,"five":700,"seven":924}}[task][block]
            assert len(files)==expected
            for path in files:
                meta=json.loads(path.read_text());baseline=base_root/task/block/path.parent.name/path.with_suffix(".npz").name
                with np.load(path.with_suffix(".npz")) as z,np.load(baseline) as b:
                    assert np.array_equal(z["Base"],b["Base"]) and np.array_equal(z["y_target"],b["y_target"])
                    for method in ["Base","Shared","Specific"]:
                        assert np.isfinite(z[method]).all()
                        if task=="classification":assert np.max(np.abs(z[method].sum(1)-1))<1e-12
                        values=metrics(z["y_target"],z[method],task,meta["scale"])
                        for key,value in values.items():
                            saved=meta["methods"][method][key]
                            assert (value is None and saved is None) or abs(value-saved)<1e-12
                        rows.append(dict(task=task,block=block,method=method,task_id=meta["task_id"],family=meta["family"],rho=meta["rho"],**values))
                    if task=="regression" and block in ["five","seven"]:
                        assert old.sha(meta["policy_path"])==meta["policy_sha256"]
                        assert old.sha(meta["anchor_prediction"])==meta["anchor_prediction_sha256"]
                        assert old.sha(meta["extension_prediction"])==meta["extension_prediction_sha256"]
                        with np.load(meta["anchor_prediction"]) as a,np.load(meta["extension_prediction"]) as e:
                            assert np.array_equal(a["Base"],z["Base"]) and np.array_equal(e["Base"],z["Base"])
                            prediction=(1-meta["selected_blend"])*a["Specific"]+meta["selected_blend"]*e["Specific"]
                            error=float(np.max(np.abs(prediction-z["Specific"])));max_error=max(max_error,error);assert error<1e-12
                        assert Path(meta["policy_path"]).stat().st_mtime<=path.stat().st_mtime
                count+=1
    data=pd.DataFrame(rows);data.to_csv(root/"audited_detail.csv",index=False)
    targets=json.loads((prior.RUN/"results/tabace_context_reproduction_v28/paper_targets.json").read_text())
    checks=[];maximum=0.
    for (task,block,method),group in data.groupby(["task","block","method"]):
        keys=["accuracy","auc"] if task=="classification" else ["r2","nrmse"]
        actual=group.groupby("task_id")[keys].mean().mean()
        for key,target in zip(keys,targets[task][block][method]):
            delta=float(actual[key])-target;maximum=max(maximum,abs(delta))
            checks.append(dict(task=task,block=block,method=method,metric=key,value=float(actual[key]),paper_value=target,difference=delta,within_0_05=bool(abs(delta)<=.05)))
    atomic_json(root/"reference_checks.json",checks)
    all_pass=all(c["within_0_05"] for c in checks)
    summary=dict(status="PASS",conditions=count,method_records=len(rows),counts=counts,
        source_conditions_recomputed=source_conditions,classification_checkpoint_hashes_and_crossfit_IDs_verified=model_hashes,
        Base_and_query_truth_bitwise_unchanged=True,all_metrics_recomputed=True,
        regression_mixture_max_error=max_error,source_coefficient_rule_recomputed=True,
        paper_reference_checks=len(checks),checks_within_0_05=sum(c["within_0_05"] for c in checks),
        all_paper_metrics_within_0_05=all_pass,maximum_absolute_paper_difference=maximum,
        original_paper_reproduction=False,independent_test_claim=False,prior_development_feedback=True,
        extra_clean_TRAIN_supervision=True,TFM_checkpoints_finetuned=False,full_fresh_TFM_inference=False,
        elapsed_seconds=time.time()-started,Markdown_bytes=(root.parents[3]/"REBUTTAL_EXPERIMENTS.md").stat().st_size)
    atomic_json(root/"audit.json",summary)
    atomic_json(root/"completion.json",dict(status="DEVELOPMENT_REFERENCE_TOLERANCE_MET" if all_pass else "COMPLETE_WITH_REFERENCE_GAPS",
        reference_tolerance=.05,all_metrics_within_tolerance=all_pass,audit="audit.json",
        original_paper_replication=False,independent_test_claim=False,seed=0))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":audit()
