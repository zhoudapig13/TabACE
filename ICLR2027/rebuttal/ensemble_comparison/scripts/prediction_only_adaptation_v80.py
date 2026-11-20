"""Capacity-controlled, output-only adapters; no reference-score selection.

The primary variant restores five Shared views and their five directed
Specific additions. An expanded-bank linear variant is retained as a control.
All hyperparameters and variants are declared before evaluating new queries.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import joblib
import numpy as np
import pandas as pd
import torch

from prepare_data import RUN, atomic_json, protocol
from run_tabicl import normalize, metrics
import probability_head_v51 as cls
import calibration_head_v45 as reg

OUT = RUN / "results/tabace_prediction_only_adaptation_v80"
VARIANTS = {"expanded_linear": dict(shared=10, specific=21, order=list(range(21))),
            "paper_linear": dict(shared=5, specific=10,
                order=list(range(5)) + list(range(10, 15)) + list(range(5, 10)) + list(range(15, 21)))}
PRIMARY = "paper_linear"
BLENDS = [0., .1, .25, .5, .75, 1.]
PROFILE = dict(hidden=0, weight_decay=1., epochs=120, learning_rate=.02)
PARENT = RUN / "results/tabace_validated_adaptation_v79"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    if (OUT / "plan.json").exists(): return
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / "implementation_snapshot"; snapshot.mkdir(exist_ok=True)
    code = {}
    for name in [Path(__file__).name, "probability_head_v51.py", "calibration_head_v45.py", "calibrated_risk_fusion_v37.py"]:
        src = Path(__file__).with_name(name); dst = snapshot / name
        dst.write_bytes(src.read_bytes()); code[name] = sha(dst)
    atomic_json(OUT / "plan.json", {
        "status": "FROZEN_BEFORE_FITS", "seed": 0, "context_cap": 256,
        "primary_variant": PRIMARY, "variants": VARIANTS,
        "classification_profile": PROFILE, "classification_blends": BLENDS,
        "regression_head_candidates": [dict(kind="ridge", alpha=a) for a in [.1, 1., 10., 100.]],
        "regression_head_inputs": "Base and TFM branch predictions ONLY; no original sample features",
        "classification_head_inputs": "Base/TFM log probabilities, their differences and entropies ONLY",
        "classification_training_anchor": "Base for both nested-capacity heads; Shared remains a deployment fallback",
        "source_supervision": "same authorized clean TRAIN calibration labels; original-row grouped four-fold OOF",
        "Shared_family_blind": True, "Specific_family_visible": True, "strength_is_model_input": False,
        "classification_selection": "minimum source OOF cross entropy; global Shared and per-family Specific residuals",
        "regression_selection": "existing source grouped-OOF reliability and Ridge validation utility",
        "reference_tolerance": .05, "reference_tolerance_used_for_selection": False,
        "test_scores_used_for_new_hyperparameters": False, "both_variants_reported": True,
        "Base_and_benchmark_unchanged": True, "frozen_blocks": ["classification clean/seven", "regression clean"],
        "original_paper_reproduction": False, "independent_test_claim": False,
        "upstream_query_feedback_used_for_development": True,
        "upstream_classification_five_is_query_side": True, "semisynthetic_regression_blocks": ["clean", "five"],
        "code_sha256": code, "Markdown_written": False})
    for variant in VARIANTS:
        for task, block in [("classification", "clean"), ("classification", "seven"), ("regression", "clean")]:
            dest = OUT / variant / "predictions/256" / task / block
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(PARENT / "predictions/256" / task / block, target_is_directory=True)


def reorder(cells, variant):
    order = VARIANTS[variant]["order"]
    for c in cells:
        assert c["bank"].shape[0] == 21
        c["bank"] = c["bank"][order]
    return cells


def cls_worker(shard, shards, device):
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    cells = cls.load("validation", 256, "five")
    tasks = sorted({str(c["meta"]["task_id"]) for c in cells}, key=int)
    for tid in tasks[shard::shards]:
        original = [c for c in cells if str(c["meta"]["task_id"]) == tid]
        for variant, config in VARIANTS.items():
            folder = OUT / variant / "classification_models" / tid
            if (folder / "completion.json").exists(): continue
            group = [c | {"bank": c["bank"][config["order"]]} for c in original]
            targets = [("Shared", group, config["shared"])] + [
                ("Specific_" + f, [c for c in group if c["meta"]["family"] == f], config["specific"])
                for f in sorted({c["meta"]["family"] for c in group})]
            source_predictions = {}
            for role, scope, members in targets:
                X = np.concatenate([cls.feature(c, members) for c in scope])
                Y = np.concatenate([c["y"] for c in scope]); B = np.concatenate([c["base"] for c in scope])
                ids = np.concatenate([c["ids"] for c in scope]); folds = np.concatenate([c["folds"] for c in scope])
                case_ids = np.concatenate([np.full(len(c["y"]), i, dtype=int) for i, c in enumerate(scope)])
                prediction = np.empty_like(B, dtype=float)
                lengths = [len(c["y"]) for c in scope]; starts = np.cumsum([0] + lengths[:-1])
                records = []
                for fold in range(4):
                    train, valid = folds != fold, folds == fold
                    assert train.any() and valid.any()
                    assert not np.intersect1d(ids[train], ids[valid]).size
                    fitted = cls.fit_net(X[train], Y[train], B[train], ids[train], PROFILE, device, case_ids[train])
                    prediction[valid] = cls.infer(X[valid], B[valid], fitted, device)
                    path = folder / role / f"validation_{fold}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(fitted, path)
                    records.append(dict(fold=fold, model_sha256=sha(path),
                        training_original_ids=np.unique(ids[train]).tolist(), held_original_ids=np.unique(ids[valid]).tolist()))
                fitted = cls.fit_net(X, Y, B, ids, PROFILE, device, case_ids)
                path = folder / role / "query.pt"; torch.save(fitted, path)
                atomic_json(path.with_suffix(".json"), dict(role=role, members=members,
                    profile=PROFILE, model_sha256=sha(path), unique_training_rows=len(np.unique(ids)),
                    training_original_ids=np.unique(ids).tolist(), input_features=X.shape[1],
                    input_sha256=protocol.array_sha256(X), target_sha256=protocol.array_sha256(Y),
                    original_row_crossfits=records, query_data_loaded=False))
                for c, start, length in zip(scope, starts, lengths):
                    key = c["key"]
                    source_predictions.setdefault(key, dict(base=c["base"], y_validation=c["y"],
                        original_ids=c["ids"], folds=c["folds"]))[role] = prediction[start:start+length]
            for c in group:
                role = "Specific_" + c["meta"]["family"]; values = source_predictions[c["key"]]
                out = OUT / variant / "source/256" / c["key"]
                cls.source.adaptation.raw.scaling.npwrite(out, base=values["base"], shared=values["Shared"],
                    specific=values[role], y_validation=values["y_validation"],
                    original_ids=values["original_ids"], folds=values["folds"])
                atomic_json(out.with_suffix(".json"), dict(source_bank=str(c["path"]),
                    source_bank_sha256=sha(c["path"]), task_id=c["meta"]["task_id"],
                    data_id=c["meta"]["data_id"], family=c["meta"]["family"], rho=c["meta"]["rho"],
                    source_only=True, raw_features_in_head=False))
            atomic_json(folder / "completion.json", dict(status="COMPLETE", dataset=tid, variant=variant,
                source_cases=len(group), original_query_data_loaded=False))
            print("CLASS_SOURCE_COMPLETE", shard, tid, variant, flush=True)


def cls_select_and_query(variant, device="cpu"):
    config = VARIANTS[variant]
    source = []
    for path in sorted((OUT / variant / "source/256/classification/five").glob("*/*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as z: source.append(dict(meta=meta, path=path, **{k: z[k] for k in z.files}))
    assert len(source) == 1440
    def choose(scope, baseline, field):
        trials = []; best = 0.; minimum = float("inf")
        for blend in BLENDS:
            losses = []
            for c in scope:
                p = normalize((1-blend)*baseline[str(c["path"])]+blend*c[field])
                losses.append(-np.log(np.clip(p[np.arange(len(c["y_validation"])), c["y_validation"]], 1e-12, 1.)).mean())
            loss = float(np.mean(losses)); trials.append(dict(blend=blend, source_OOF_CE=loss))
            if loss < minimum-1e-12: minimum, best = loss, blend
        return dict(blend=best, source_OOF_CE=minimum, trials=trials)
    base = {str(c["path"]): c["base"] for c in source}
    shared = choose(source, base, "shared")
    shared_oof = {str(c["path"]): normalize((1-shared["blend"])*c["base"]+shared["blend"]*c["shared"]) for c in source}
    specific = {}
    for family in sorted({c["meta"]["family"] for c in source}):
        specific[family] = choose([c for c in source if c["meta"]["family"] == family], shared_oof, "specific")
    freeze = OUT / variant / "policies/classification/five/frozen_policy.json"
    atomic_json(freeze, dict(status="FROZEN_BEFORE_QUERY", Shared=shared, Specific=specific,
        physical_Shared_members=config["shared"], physical_Specific_members=config["specific"],
        objective="minimum source grouped-OOF CE", query_labels_used_for_selection=False))
    query = reorder(cls.load("query", 256, "five"), variant); cache = {}
    for c in query:
        meta = c["meta"]; tid = str(meta["task_id"])
        def predict(role, members):
            path = OUT / variant / "classification_models" / tid / role / "query.pt"
            if str(path) not in cache: cache[str(path)] = torch.load(path, map_location="cpu", weights_only=True)
            return cls.infer(cls.feature(c, members), c["base"], cache[str(path)], device)
        a = shared["blend"]; sh = normalize((1-a)*c["base"]+a*predict("Shared", config["shared"]))
        a = specific[meta["family"]]["blend"]
        sp = normalize((1-a)*sh+a*predict("Specific_"+meta["family"], config["specific"]))
        write(variant, "classification", "five", c, sh, sp, freeze)
    atomic_json(OUT / variant / "completion/classification_five.json", dict(status="COMPLETE", conditions=1440))
    print("CLASS_QUERY_COMPLETE", variant, flush=True)


def prediction_features(unused_X, cell, members):
    return np.column_stack([cell["base"], cell["bank"][:members].T]).astype(float)


def regression(variant, block):
    torch.set_num_threads(1); np.random.seed(0)
    config = VARIANTS[variant]
    reg.features = prediction_features
    reg.CANDIDATES = [dict(kind="ridge", alpha=a) for a in [.1, 1., 10., 100.]]
    source = reorder(reg.risk.load(reg.source.OUT, "validation", 256, block), variant)
    root = OUT / variant / "policies/regression" / block
    shared, shared_oof = reg.fit_policy(source, config["shared"], block,
        {c["key"]: c["base"] for c in source}, root / "Shared")
    specific = {}
    for family in sorted({c["meta"]["family"] for c in source}):
        group = [c for c in source if c["meta"]["family"] == family]
        specific[family], _ = reg.fit_policy(group, config["specific"], block, shared_oof, root / ("Specific_"+family))
    freeze = root / "frozen_policy.json"
    atomic_json(freeze, dict(status="FROZEN_BEFORE_QUERY", Shared=shared, Specific=specific,
        physical_Shared_members=config["shared"], physical_Specific_members=config["specific"],
        raw_features_in_head=False, source_original_row_group_folds=4,
        query_labels_used_for_selection=False))
    query = reorder(reg.risk.load(reg.source.OUT, "query", 256, block), variant); cache = {}
    for c in query:
        sh = reg.predict(c, shared, c["base"], None, cache)
        sp = reg.predict(c, specific[c["meta"]["family"]], sh, None, cache)
        write(variant, "regression", block, c, sh, sp, freeze)
    atomic_json(OUT / variant / "completion" / f"regression_{block}.json", dict(status="COMPLETE", conditions=len(query)))
    print("REGRESSION_QUERY_COMPLETE", variant, block, len(query), flush=True)


def write(variant, task, block, cell, shared, specific, policy):
    meta = cell["meta"]
    out = OUT / variant / "predictions/256" / cell["key"]
    cls.source.adaptation.raw.scaling.npwrite(out, Base=cell["base"], Shared=shared, Specific=specific, y_target=cell["y"])
    scores = {name: metrics(cell["y"], p, task, meta["scale"]) for name, p in [("Base", cell["base"]), ("Shared", shared), ("Specific", specific)]}
    record = {k: meta[k] for k in ["task_id", "dataset", "family", "rho", "scale"]}
    record.update(task=task, block=block, seed=0, context=256, variant=variant, methods=scores,
        source_bank=str(cell["path"]), source_bank_sha256=sha(cell["path"]),
        policy_path=str(policy), policy_sha256=sha(policy),
        raw_features_in_calibration_head=False, extra_clean_TRAIN_supervision=True,
        Shared_members=VARIANTS[variant]["shared"], Specific_members=VARIANTS[variant]["specific"],
        Base_and_benchmark_unchanged=True, original_paper_reproduction=False, independent_test_claim=False,
        upstream_query_feedback_used_for_development=True, reference_scores_used_for_selection=False)
    atomic_json(out.with_suffix(".json"), record)


def report():
    targets = json.loads((RUN / "results/tabace_context_reproduction_v28/paper_targets.json").read_text())
    for variant in VARIANTS:
        rows = []
        for task in ["classification", "regression"]:
            for block in ["clean", "five", "seven"]:
                for path in (OUT / variant / "predictions/256" / task / block).glob("*/*.json"):
                    meta = json.loads(path.read_text())
                    for method, values in meta["methods"].items():
                        rows.append(dict(task=task, block=block, method=method, task_id=meta["task_id"], family=meta["family"], rho=meta["rho"], **values))
        if not rows: continue
        data = pd.DataFrame(rows); data.to_csv(OUT / variant / "detail.csv", index=False)
        averages = []
        for (task, block, method), group in data.groupby(["task", "block", "method"]):
            keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
            values = group.groupby("task_id")[keys].mean().mean()
            row = dict(variant=variant, task=task, block=block, method=method, datasets=group.task_id.nunique(), conditions=len(group))
            for key, target in zip(keys, targets[task][block][method]):
                row[key] = float(values[key]); row["paper_"+key] = target
                row["difference_"+key] = float(values[key])-target
                row["within_0_05_"+key] = bool(abs(float(values[key])-target) <= .05)
            averages.append(row)
        pd.DataFrame(averages).to_csv(OUT / variant / "averages.csv", index=False)


def launch():
    initialize(); jobs = []
    def execute(args, name, gpu=None):
        log = OUT / "logs" / f"{name}.log"; log.parent.mkdir(exist_ok=True)
        env = os.environ.copy()
        if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log.open("a") as handle:
            p = subprocess.run([sys.executable, __file__, *args], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return dict(name=name, returncode=p.returncode, log=str(log))
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(execute, ["class_source", "--shard", str(i), "--shards", "8"], f"class_source_{i}", i) for i in range(8)]
        futures += [executor.submit(execute, ["regression", "--variant", variant, "--block", block], f"regression_{variant}_{block}")
                    for variant in VARIANTS for block in ["five", "seven"]]
        for future in as_completed(futures):
            jobs.append(future.result()); atomic_json(OUT / "execution.json", dict(status="RUNNING", jobs=jobs)); print(jobs[-1], flush=True)
    if any(j["returncode"] for j in jobs):
        atomic_json(OUT / "execution.json", dict(status="FAILED", jobs=jobs)); return
    for variant in VARIANTS:
        jobs.append(execute(["class_query", "--variant", variant], f"class_query_{variant}"))
    report(); atomic_json(OUT / "execution.json", dict(status="COMPLETE" if not any(j["returncode"] for j in jobs) else "FAILED", jobs=jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["launch", "class_source", "class_query", "regression", "report"])
    parser.add_argument("--shard", type=int, default=0); parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--variant", choices=list(VARIANTS)); parser.add_argument("--block", choices=["five", "seven"])
    args = parser.parse_args()
    if args.phase == "launch": launch()
    elif args.phase == "class_source": cls_worker(args.shard, args.shards, "cuda:0")
    elif args.phase == "class_query": torch.set_num_threads(1); cls_select_and_query(args.variant)
    elif args.phase == "regression": regression(args.variant, args.block)
    else: report()
