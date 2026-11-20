"""Frozen-checkpoint external baselines on the existing, unmodified ACS splits.

Inference saves probabilities before the separate scoring function reads labels.
DistPFN follows official eval_TABPFN_shift_X.py lines 81--85 exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "results/raw_acs_external_v14"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load(prefix, role):
    with np.load(str(prefix) + f"_{role}.npz", allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prefixes(scenario, task):
    year = 2018 if scenario == "geographic" else 2014
    target_state = "MI" if scenario == "geographic" else "CA"
    root = RUN / "data/splits"
    return (root / f"{year}_CA_{task}", root / f"2018_{target_state}_{task}", year)


def score(stem, scenario, task):
    """Called only once immutable predictions and inference metadata exist."""
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    source, target, _ = prefixes(scenario, task)
    meta = json.loads(stem.with_suffix(".json").read_text())
    rows = []
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as bank:
        for domain, prefix in [("ID", source), ("OOD", target)]:
            labels = load(prefix, "test_labels")
            features = load(prefix, "test_features")
            assert np.array_equal(labels["row_ids"], features["row_ids"])
            for name in meta["methods"]:
                p = bank[f"{domain}__{name}"]
                assert p.shape == (len(labels["y"]), 2)
                rows.append({
                    "backend": meta["backend"], "method": name, "scenario": scenario,
                    "task": task, "domain": domain, "seed": meta["seed"],
                    "n": len(p), "accuracy": accuracy_score(labels["y"], p.argmax(1)),
                    "auc": roc_auc_score(labels["y"], p[:, 1]),
                    "log_loss": log_loss(labels["y"], p, labels=[0, 1]),
                })
    save_json(stem.with_name(stem.name + "_metrics.json"), rows)
    print(json.dumps({"status": "COMPLETE", "metrics": rows}), flush=True)


def run(backend, scenario, task, seed):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    stem = OUT / f"{backend}__{scenario}__{task}__seed{seed}"
    if stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists():
        score(stem, scenario, task)
        return
    source_prefix, target_prefix, year = prefixes(scenario, task)
    pool = load(source_prefix, "source")
    context_idx = np.random.default_rng(seed).permutation(len(pool["y"]))[:512]
    X, y = pool["X"][context_idx].astype(np.float32), pool["y"][context_idx]
    assert len(y) == 512 and np.array_equal(np.unique(y), [0, 1])
    groups = {"ID": load(source_prefix, "test_features"),
              "OOD": load(target_prefix, "test_features")}
    meta = json.loads(source_prefix.with_suffix(".json").read_text())
    info = {
        "backend": backend, "scenario": scenario, "task": task, "seed": seed,
        "context_rows": len(y), "source_pool_rows": len(pool["y"]),
        "features": meta["features"], "source_year": year,
        "target_year": 2018, "original_paper_split_recovered": False,
        "test_labels_accessed_during_inference": False,
        "calibration_labels_used": False, "finetuning": False,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "gpu": torch.cuda.get_device_name(0), "context_index_sha256":
            hashlib.sha256(context_idx.tobytes()).hexdigest(),
        "source_file_sha256": sha(Path(str(source_prefix) + "_source.npz")),
        "test_feature_file_sha256": {
            domain: sha(Path(str(prefix) + "_test_features.npz"))
            for domain, prefix in [("ID", source_prefix), ("OOD", target_prefix)]},
    }
    arrays = {"context_indices": context_idx, "context_row_ids": pool["row_ids"][context_idx]}
    started = time.time()
    if backend == "distpfn":
        from run_natural import Predictor, CHECKPOINT

        predictor = Predictor()
        predictor.model.random_state = seed
        info.update({
            "methods": ["TabICL-v2-512", "DistPFN-TabICL-v2"],
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
            "n_estimators": 1, "inference_chunk_rows": 1024,
            "official_repo": "https://github.com/seunghan96/DistPFN",
            "official_commit": "6b375c8a7624e614e84072c419c4d127229c77b0",
            "correction_source": "src/evals/eval_TABPFN_shift_X.py:81-85",
            "temperature_variant": False,
            "target_marginal": "computed separately over the entire ID/OOD query group",
        })
        prior = np.bincount(y, minlength=2) / len(y)
        for domain, features in groups.items():
            p = predictor.predict(X, y, features["X"])
            adjusted = p * p.mean(axis=0) / (prior + 1e-8)
            corrected = adjusted / adjusted.sum(axis=1, keepdims=True)
            arrays[f"{domain}__TabICL-v2-512"] = p
            arrays[f"{domain}__DistPFN-TabICL-v2"] = corrected
            arrays[f"{domain}__row_ids"] = features["row_ids"]
            print(json.dumps({"status": "PREDICTED", "backend": backend,
                              "scenario": scenario, "task": task, "seed": seed,
                              "domain": domain, "rows": len(p),
                              "seconds": time.time() - started}), flush=True)
    else:
        import sys
        import sklearn

        repo = RUN / "external/Drift-Resilient_TabPFN"
        sys.path.insert(0, str(repo))
        from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig

        checkpoint = repo / "tabpfn/model_cache/tabpfn_dist_model_1.cpkt"
        model = get_best_tabpfn(
            task_type="dist_shift_multiclass", model_type="best_dist",
            paths_config=TabPFNModelPathsConfig(paths=[str(checkpoint)],
                                               task_type="dist_shift_multiclass"),
            device="cuda")
        model.show_progress = False
        model.seed = seed
        categorical = [i for i, name in enumerate(meta["features"])
                       if name not in {"AGEP", "WKHP", "POVPIP"}]
        model.set_categorical_features(categorical)
        info.update({
            "methods": ["Drift-Resilient"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "official_repo": "https://github.com/automl/Drift-Resilient_TabPFN",
            "official_commit": "a6e75afb82d13e7abb46deade3669c4106b3d636",
            "n_estimators": model.N_ensemble_configurations,
            "official_factory": "get_best_tabpfn(best_dist)",
            "categorical_feature_indices": categorical,
            "domain_encoding": "actual ACS survey year; no row-order timestamps",
            "source_contains_single_year": True,
            "inference_chunk_rows": None,
            "sklearn_version": sklearn.__version__,
            "compatibility_deviation": "PyTorch 2.8 for RTX 5090 instead of original 2.1",
        })
        model.fit(X, y, additional_x={"dist_shift_domain": torch.full((len(y),), float(year))})
        for domain, features in groups.items():
            Q = features["X"].astype(np.float32)
            query_year = year if domain == "ID" else 2018
            p = model.predict_proba(Q, additional_x={
                "dist_shift_domain": torch.full((len(Q),), float(query_year))})
            arrays[f"{domain}__Drift-Resilient"] = np.asarray(p)
            arrays[f"{domain}__row_ids"] = features["row_ids"]
            print(json.dumps({"status": "PREDICTED", "backend": backend,
                              "scenario": scenario, "task": task, "seed": seed,
                              "domain": domain, "rows": len(p),
                              "seconds": time.time() - started}), flush=True)
    for key, value in arrays.items():
        if "__" in key and not key.endswith("row_ids"):
            assert np.isfinite(value).all() and (value >= 0).all(), key
            assert np.allclose(value.sum(axis=1), 1, atol=1e-5), key
    OUT.mkdir(parents=True, exist_ok=True)
    temp = stem.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(stem.with_suffix(".npz"))
    info["seconds"] = time.time() - started
    info["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    save_json(stem.with_suffix(".json"), info)
    score(stem, scenario, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["distpfn", "drift"], required=True)
    parser.add_argument("--scenario", choices=["geographic", "temporal"], required=True)
    parser.add_argument("--task", choices=["Employment", "Income", "TravelTime"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.backend, args.scenario, args.task, args.seed)
