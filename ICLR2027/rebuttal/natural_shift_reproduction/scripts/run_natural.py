"""Resumable feature-only inference. This worker never opens test labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from tabicl import TabICLClassifier

from natural_components import normalize, shared_views, specific_views
from prepare_acs import RUN, save_json

CHECKPOINT = RUN.parents[2] / "TabSwift/artifacts/checkpoints/tabicl-classifier-v2-20260212.ckpt"


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


class Predictor:
    def __init__(self):
        self.model = TabICLClassifier(
            n_estimators=1, random_state=0, device="cuda:0", n_jobs=1, batch_size=1,
            kv_cache=True, model_path=str(CHECKPOINT), allow_auto_download=False)

    def predict(self, X, y, Q):
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int))
        if not getattr(self.model, "_natural_cached_weights", False):
            self.model._load_model = lambda: None
            self.model._natural_cached_weights = True
        parts = []
        for start in range(0, len(Q), 1024):
            raw = self.model.predict_proba(np.asarray(Q[start:start+1024], dtype=np.float32))
            aligned = np.zeros((len(raw), 2))
            aligned[:, self.model.classes_.astype(int)] = raw
            parts.append(aligned)
        return normalize(np.concatenate(parts))


def load(prefix, role):
    with np.load(str(prefix) + f"_{role}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def run(scenario, task, base_only=False):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    year = 2018 if scenario == "geographic" else 2014
    target_year, target_state = (2018, "MI") if scenario == "geographic" else (2018, "CA")
    source_prefix = RUN / "data/splits" / f"{year}_CA_{task}"
    target_prefix = RUN / "data/splits" / f"{target_year}_{target_state}_{task}"
    source = load(source_prefix, "source")
    calibration = load(source_prefix, "calibration")
    source_test = load(source_prefix, "test_features")
    target_test = load(target_prefix, "test_features")
    meta = json.loads(source_prefix.with_suffix(".json").read_text())
    Xp, yp = source["X"], source["y"]
    ids = source["canonical"]
    X, y = Xp[ids], yp[ids]
    groups = {"fit": calibration["X"][:2048], "select": calibration["X"][2048:],
              "ID": source_test["X"], "OOD": target_test["X"]}
    predictor = Predictor()
    output = RUN / "results/banks" / scenario / task
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    definitions = {}
    for role, Q in groups.items():
        views = [("Base", X, y, Q, {"canonical_context": True})]
        if not base_only:
            views += shared_views(X, y, Q)
            views += specific_views(X, y, Q, Xp, yp, meta["features"], scenario)
        for name, A, b, queries, config in views:
            path = output / f"{role}__{name}.npz"
            signature = digest(A, b, queries)
            definitions[name] = config | {"rows": len(b), "features": A.shape[1]}
            if path.exists() and path.with_suffix(".json").exists():
                old = json.loads(path.with_suffix(".json").read_text())
                if old["input_sha256"] != signature:
                    raise RuntimeError(f"Cached inputs differ: {path}")
                continue
            t0 = time.time()
            p = predictor.predict(A, b, queries)
            assert p.shape == (len(Q), 2) and np.all(np.isfinite(p))
            assert np.max(np.abs(p.sum(-1)-1.)) < 1e-8
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, prediction=p)
            temp.replace(path)
            save_json(path.with_suffix(".json"), {
                "scenario": scenario, "task": task, "role": role, "branch": name,
                "input_sha256": signature, "context_rows": len(b), "features": A.shape[1],
                "query_rows": len(Q), "seconds": time.time()-t0, "constructor": config,
                "seed": 0, "checkpoint": str(CHECKPOINT), "test_labels_accessed": False,
            })
            print(json.dumps({"scenario": scenario, "task": task, "role": role,
                              "branch": name, "seconds": round(time.time()-t0, 2)}), flush=True)
    save_json(output / ("base_complete.json" if base_only else "bank_complete.json"), {
        "scenario": scenario, "task": task, "status": "COMPLETE", "seed": 0,
        "source_domain": f"CA{year}", "target_domain": f"{target_state}{target_year}",
        "definitions": definitions, "roles": {k: len(v) for k, v in groups.items()},
        "seconds": time.time()-started, "source_pool_rows": len(yp),
        "canonical_rows": len(y), "test_labels_accessed": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["geographic", "temporal"], required=True)
    parser.add_argument("--task", choices=["Employment", "Income", "TravelTime"], required=True)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    run(args.scenario, args.task, args.base_only)
