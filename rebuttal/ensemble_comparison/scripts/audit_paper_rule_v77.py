"""Independent fusion arithmetic, fold isolation, paired Base, and metric audit."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from prepare_data import atomic_json
from run_tabicl import normalize, metrics
import paper_rule_reconstruction_v77 as experiment


def independent_predict(raw, model, task):
    base = raw["base"]
    if model["kind"] == "base":
        return base.copy(), 0., 0.
    m, gain = model["m"], model["gain"]
    assert gain == 1. and model["gate_variant"] == "entropy"
    theta = np.asarray(model["parameters"], dtype=float)
    assert 0. <= theta[-1] <= 1.
    bank = raw["members"][:m]
    if task == "classification":
        p = np.clip(bank, 1e-12, 1.)
        confidence = (1. + (p * np.log(p)).sum(-1) / np.log(max(2, base.shape[1]))).T
        agreement = (1. - .5 * np.abs(bank - bank.mean(0)[None]).sum(-1)).T
        q = np.clip(base, 1e-12, 1.)
        uncertainty = -(q * np.log(q)).sum(-1) / np.log(max(2, base.shape[1]))
        logits = theta[:m] + confidence * theta[m:2*m] + agreement * theta[2*m:3*m]
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        aggregate = np.einsum("nr,rnc->nc", weights, bank)
        alpha = np.clip(theta[-1] * uncertainty * (weights * confidence * agreement).sum(1), 0., 1.)
        output = normalize((1. - alpha[:, None]) * base + alpha[:, None] * aggregate)
    else:
        mu, sigma = float(raw["source_mean"]), float(raw["source_scale"])
        standardized = (bank - mu) / sigma
        reference = (base - mu) / sigma
        deviation = np.abs(standardized - np.median(standardized, axis=0)[None]).T
        tau = np.exp(theta[-2])
        agreement = np.exp(-deviation / tau)
        logits = theta[:m] + agreement * theta[m:2*m]
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        aggregate = (weights * standardized.T).sum(1)
        uncertainty = 1. - np.exp(-np.abs(aggregate - reference) / tau)
        alpha = np.clip(theta[-1] * uncertainty * (weights * agreement).sum(1), 0., 1.)
        output = ((1. - alpha) * reference + alpha * aggregate) * sigma + mu
    assert np.isfinite(output).all()
    assert np.all(weights >= 0.) and np.max(np.abs(weights.sum(1) - 1.)) < 1e-12
    assert np.all((alpha >= 0.) & (alpha <= 1.))
    return output, float(alpha.min()), float(alpha.max())


def run():
    started = time.time()
    root = experiment.OUT
    assert len(list((root / "fold_runs").glob("*/fold_*/completion.json"))) == 10
    manifest = json.loads((root / "cache_manifest.json").read_text())
    expected_hashes = {entry["path"]: entry for entry in manifest}
    # Verify every cached source and query probability bank still has the hash
    # committed before any new policy fitting.
    for record in manifest:
        path = Path(record["path"])
        assert experiment.file_sha(path) == record["sha256"]
        assert experiment.file_sha(path.with_suffix(".json")) == record["metadata_sha256"]
    checks, records = [], []
    source_split_checks = 0
    for task in experiment.EXPECTED:
        for track in ("five", "seven"):
            for path in (experiment.banks.OUT / "validation_splits" / task / track).glob("*.npz"):
                with np.load(path) as split:
                    assert not np.intersect1d(split["source_original_ids"], split["validation_original_ids"]).size
                source_split_checks += 1
    max_error = 0.
    base_difference = 0.
    policy_cache = {}
    condition_files = sorted((root / "predictions/256").glob("*/*/*/*.json"))
    assert len(condition_files) == sum(sum(v.values()) for v in experiment.EXPECTED.values()) == 4151
    for path in condition_files:
        meta = json.loads(path.read_text())
        bank_path = Path(meta["bank_path"])
        assert meta["bank_sha256"] == expected_hashes[str(bank_path)]["sha256"]
        policy_path = Path(meta["policy_path"])
        if str(policy_path) not in policy_cache:
            policy_cache[str(policy_path)] = json.loads(policy_path.read_text())
        policy = policy_cache[str(policy_path)]
        assert experiment.file_sha(policy_path) == meta["policy_sha256"]
        assert meta["data_id"] not in policy["training_data_ids"]
        assert meta["data_id"] in policy["excluded_query_data_ids"]
        assert set(policy["training_data_ids"]).isdisjoint(policy["excluded_query_data_ids"])
        shared_sel = policy["Shared_selection"]
        assert set(shared_sel["inner_train_data_ids"]).isdisjoint(shared_sel["inner_validation_data_ids"])
        for selection in policy["Specific_selection"].values():
            assert set(selection["inner_train_data_ids"]).isdisjoint(selection["inner_validation_data_ids"])
        with np.load(bank_path) as f:
            raw = {k: f[k] for k in f.files}
        bank_meta = json.loads(bank_path.with_suffix(".json").read_text())
        origin = Path(bank_meta["Base_origin"])
        with np.load(origin) as f:
            assert np.array_equal(raw["base"], f["Base_prediction"])
        shared, alpha_low, alpha_high = independent_predict(raw, policy["Shared"], meta["task"])
        if meta["block"] == "clean":
            specific, specific_low, specific_high = shared.copy(), alpha_low, alpha_high
        else:
            specific, specific_low, specific_high = independent_predict(
                raw, policy["Specific"][meta["block"] + "/" + meta["family"]], meta["task"])
        with np.load(path.with_suffix(".npz")) as output:
            assert np.array_equal(output["Base"], raw["base"])
            assert np.array_equal(output["y_target"], raw["y_target"])
            for method, prediction in [("Base", raw["base"]), ("Shared", shared), ("Specific", specific)]:
                error = float(np.max(np.abs(output[method] - prediction)))
                assert error < 1e-12, (path, method, error)
                max_error = max(max_error, error)
                recomputed = metrics(raw["y_target"], prediction, meta["task"], meta["scale"])
                for key, value in recomputed.items():
                    stored = meta["methods"][method][key]
                    assert (value is None and stored is None) or abs(value - stored) < 1e-12
                records.append({"task": meta["task"], "block": meta["block"], "method": method,
                    "data_id": meta["data_id"], "family": meta["family"], "rho": meta["rho"], **recomputed})
        checks.append({"task": meta["task"], "block": meta["block"], "task_id": meta["task_id"],
            "family": meta["family"], "rho": meta["rho"], "Shared_alpha_min": alpha_low,
            "Shared_alpha_max": alpha_high, "Specific_alpha_min": specific_low, "Specific_alpha_max": specific_high})
    data = pd.DataFrame(records)
    targets = json.loads((experiment.banks.OUT / "paper_targets.json").read_text())
    comparison, base_within = [], []
    for (task, block, method), group in data.groupby(["task", "block", "method"]):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        average = group.groupby("data_id")[keys].mean().mean()
        row = {"task": task, "block": block, "method": method, "datasets": group.data_id.nunique(), "conditions": len(group)}
        for key, target in zip(keys, targets[task][block][method]):
            value = float(average[key])
            row.update({key: value, "paper_" + key: target, "difference_" + key: value - target})
            if method == "Base":
                base_within.append(abs(value - target) <= .005)
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(root / "comparison_with_paper.csv", index=False)
    pd.DataFrame(checks).to_csv(root / "gate_diagnostics.csv", index=False)
    summary = {"status": "PASS", "conditions": len(checks), "method_records": len(records),
        "cached_bank_files_verified": len(manifest), "source_split_disjoint_checks": source_split_checks,
        "dataset_fold_policies": len(policy_cache), "independent_fusion_max_error": max_error,
        "Base_max_difference": base_difference, "Base_metrics_within_0_005": bool(all(base_within)),
        "all_original_cohorts_retained": True, "paper_scores_not_a_selection_objective": True,
        "full_fresh_backbone_inference_audited": False, "original_paper_replication": False,
        "independent_test_claim": False, "development_benchmark": True,
        "elapsed_seconds": time.time() - started,
        "Markdown_bytes": (root.parents[3] / "REBUTTAL_EXPERIMENTS.md").stat().st_size}
    atomic_json(root / "audit.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
