"""Explicit source-pool adaptation of the same five Shared roles.

Two candidate samplers are compared on source validation only. Neither changes
test data or per-branch row budget. The original canonical-bank results persist.
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from natural_components import SHARED_NAMES, geometry, distances
from prepare_acs import RUN, save_json
from run_natural import Predictor, digest, load


def take(y, score, budget, stratified, fraction=None):
    if fraction is not None:
        rng = np.random.default_rng(0)
        if stratified:
            npos = int(round(budget * np.mean(y)))
            chunks = []
            for label, count in [(0, budget-npos), (1, npos)]:
                ids = np.flatnonzero(y == label)
                retained = ids[np.argsort(score[ids], kind="stable")[:int(np.ceil(len(ids)*fraction))]]
                chunks.append(rng.choice(retained, count, replace=False))
            return np.concatenate(chunks)
        retained = np.argsort(score, kind="stable")[:int(np.ceil(len(y)*fraction))]
        # Ensure every class is available before random downsampling.
        mandatory = []
        for label in np.unique(y):
            available = retained[y[retained] == label]
            if not len(available):
                available = np.flatnonzero(y == label)
            mandatory.append(int(rng.choice(available)))
        remaining = np.setdiff1d(retained, mandatory)
        ids = np.r_[mandatory, rng.choice(remaining, budget-len(mandatory), replace=False)]
        rng.shuffle(ids)
        return ids
    if not stratified:
        ids = list(np.argsort(score, kind="stable")[:budget])
        for label in np.unique(y):
            if not np.any(y[ids] == label):
                ids[-1] = int(np.flatnonzero(y == label)[np.argmin(score[y == label])])
        return np.asarray(ids)
    # Preserve source class proportions, rather than exploiting target labels.
    npos = int(round(budget * np.mean(y)))
    result = []
    for label, count in [(0, budget-npos), (1, npos)]:
        ids = np.flatnonzero(y == label)
        result.extend(ids[np.argsort(score[ids], kind="stable")[:count]])
    return np.asarray(result, dtype=int)


def expanded_views(X, y, Q, canonical, mode, coverage=False):
    Z, V = geometry(X, Q)
    anchors = np.quantile(V, [.1, .3, .5, .7, .9], axis=0)
    near = pairwise_distances(Z, anchors, metric="sqeuclidean").min(1)
    target = distances(Z, V)
    outlying = (Z ** 2).mean(1)
    d = pairwise_distances(Z, metric="sqeuclidean")
    np.fill_diagonal(d, np.inf)
    neighbors = np.argpartition(d, 9, axis=1)[:, :10]
    disagreement = 1. - np.maximum((y[neighbors] == 0).mean(1), (y[neighbors] == 1).mean(1))
    scores = {"target_neighbor": near, "representative": target+.35*outlying,
              "locally_stable": disagreement+.10*target,
              "low_conflict": disagreement+.20*outlying}
    fractions = {"target_neighbor": .8, "representative": .7,
                 "locally_stable": .8, "low_conflict": .75}
    centered = X - X.mean(0)
    yc = y - y.mean()
    association = np.abs((centered * yc[:, None]).sum(0)) / np.maximum(
        np.sqrt((centered ** 2).sum(0) * (yc ** 2).sum()), 1e-10)
    columns = np.delete(np.arange(X.shape[1]), int(np.argmax(association)))
    for name in SHARED_NAMES:
        if name == "feature_subspace":
            yield name, X[canonical][:, columns], y[canonical], Q[:, columns]
        else:
            ids = take(y, scores[name], 256, mode == "strat",
                       fractions[name] if coverage else None)
            yield name, X[ids], y[ids], Q


def run(scenario, task, coverage=False):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source_domain = "2018_CA" if scenario == "geographic" else "2014_CA"
    target_domain = "2018_MI" if scenario == "geographic" else "2018_CA"
    src = RUN / "data/splits" / f"{source_domain}_{task}"
    tgt = RUN / "data/splits" / f"{target_domain}_{task}"
    source, calibration = load(src, "source"), load(src, "calibration")
    queries = {"fit": calibration["X"][:2048], "select": calibration["X"][2048:],
               "ID": load(src, "test_features")["X"], "OOD": load(tgt, "test_features")["X"]}
    root = RUN / "results/banks" / scenario / task
    predictor = Predictor()
    for role, Q in queries.items():
        for mode in ["rank", "strat"]:
            for name, X, y, V in expanded_views(source["X"], source["y"], Q, source["canonical"], mode, coverage):
                label = f"{'coverage' if coverage else 'expanded'}_{mode}_{name}"
                path = root / f"{role}__{label}.npz"
                signature = digest(X, y, V)
                if path.exists():
                    assert json.loads(path.with_suffix(".json").read_text())["input_sha256"] == signature
                    continue
                start = time.time()
                prediction = predictor.predict(X, y, V)
                np.savez_compressed(path, prediction=prediction)
                save_json(path.with_suffix(".json"), {
                    "branch": label, "scenario": scenario, "task": task, "role": role,
                    "input_sha256": signature, "seed": 0, "source_pool_rows": 2048,
                    "context_rows": len(y), "sampling_mode": mode, "source_only_adaptation": True,
                    "test_labels_accessed": False, "seconds": time.time()-start,
                    "preserve_registry_coverage_before_budget_sampling": coverage,
                })
                print(scenario, task, role, label, round(time.time()-start, 2), flush=True)
    marker = "coverage_complete.json" if coverage else "expanded_complete.json"
    save_json(root / marker, {"status": "COMPLETE", "scenario": scenario, "task": task})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    run(args.scenario, args.task, args.coverage)
