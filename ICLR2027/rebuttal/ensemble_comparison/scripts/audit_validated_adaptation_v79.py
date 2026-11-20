"""Audit paired observations, recomputed metrics, selection and model ledgers."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from prepare_data import atomic_json
from run_tabicl import metrics
import validated_adaptation_v79 as experiment


def audit():
    started = time.time(); root = experiment.OUT
    plan = json.loads((root / "plan.json").read_text())
    assert experiment.original.file_sha(Path(experiment.__file__)) == plan["code_sha256"]
    execution = json.loads((root / "execution.json").read_text())
    assert execution["status"] == "COMPLETE" and len(execution["jobs"]) == 6
    original_root = experiment.original.OUT
    counts, rows, maximum = {}, [], 0.
    for task, block in experiment.JOBS:
        files = sorted((root / "predictions/256" / task / block).glob("*/*.json"))
        assert len(files) == experiment.original.EXPECTED[task][block]
        counts[task + "/" + block] = len(files)
        for path in files:
            meta = json.loads(path.read_text())
            assert meta["extra_clean_TRAIN_supervision"] and not meta["original_TabACE_reproduction"]
            source = Path(meta["source_bank"])
            assert experiment.original.file_sha(source) == meta["source_bank_sha256"]
            paired = original_root / "predictions/256" / task / block / path.parent.name / path.with_suffix(".npz").name
            with np.load(path.with_suffix(".npz")) as output, np.load(paired) as reference:
                assert np.array_equal(output["Base"], reference["Base"])
                assert np.array_equal(output["y_target"], reference["y_target"])
                for method in ["Base", "Shared", "Specific"]:
                    assert np.isfinite(output[method]).all()
                    if task == "classification":
                        assert np.all(output[method] >= 0.)
                        assert np.max(np.abs(output[method].sum(1) - 1.)) < 1e-12
                    scores = metrics(output["y_target"], output[method], task, meta["scale"])
                    for key, value in scores.items():
                        stored = meta["methods"][method][key]
                        assert (value is None and stored is None) or abs(value - stored) < 1e-12
                    rows.append({"task": task, "block": block, "method": method,
                        "task_id": meta["task_id"], "family": meta["family"], "rho": meta["rho"], **scores})
                if task == "regression" and block in ["five", "seven"]:
                    policy = json.loads((root / "policies" / task / block / "frozen_policy.json").read_text())
                    assert experiment.original.file_sha(meta["candidate_path"]) == meta["candidate_sha256"]
                    with np.load(meta["candidate_path"]) as candidate:
                        a = policy["Shared"]["blend"]
                        sh = (1. - a) * output["Base"] + a * candidate["Shared"]
                        a = policy["Specific"]["blend"]
                        sp = (1. - a) * sh + a * candidate["Specific"]
                        error = max(float(np.max(np.abs(sh - output["Shared"]))), float(np.max(np.abs(sp - output["Specific"]))))
                        maximum = max(maximum, error); assert error < 1e-12
    # Recompute all regression source-validation grid losses; selected strengths
    # must minimize the declared objective, irrespective of query ranking.
    source_trials = 0
    for block in ["five", "seven"]:
        policy = json.loads((root / "policies/regression" / block / "frozen_policy.json").read_text())
        directory = root / "source/256/regression" / block
        cache = []
        for path in sorted(directory.glob("*/*.npz")):
            meta_path = experiment.RUN / "results/tabace_recovery_contexts_v44/banks/validation/256/regression" / block / path.parent.name / path.with_suffix(".json").name
            meta = json.loads(meta_path.read_text())
            with np.load(path) as z: cache.append(({k: z[k] for k in z.files}, meta["scale"]))
        assert len(cache) == experiment.original.EXPECTED["regression"][block]
        shared_blend = policy["Shared"]["blend"]
        for name in ["Shared", "Specific"]:
            losses = []
            for trial in policy[name]["trials"]:
                blend = trial["blend"]; values = []
                for c, scale in cache:
                    anchor = c["base"] if name == "Shared" else (1. - shared_blend) * c["base"] + shared_blend * c["strong_shared"]
                    teacher = c["strong_shared"] if name == "Shared" else c["strong_specific"]
                    p = (1. - blend) * anchor + blend * teacher
                    values.append(float(np.mean((p - c["y_validation"]) ** 2) / scale ** 2))
                loss = float(np.mean(values)); assert abs(loss - trial["validation_normalized_MSE"]) < 1e-12
                losses.append(loss); source_trials += 1
            chosen = next(t for t in policy[name]["trials"] if t["blend"] == policy[name]["blend"])
            assert abs(chosen["validation_normalized_MSE"] - min(losses)) < 1e-12
    policy = json.loads((root / "policies/classification/seven/frozen_policy.json").read_text())
    for record in policy["family_dataset_crossfits"]:
        assert set(record["training_data_ids"]).isdisjoint(record["excluded_data_ids"])
    selected = next(t for t in policy["Specific_trials"] if t["blend"] == policy["Specific_blend"])
    assert selected["source_dataset_OOF_CE"] <= min(t["source_dataset_OOF_CE"] for t in policy["Specific_trials"]) + 1e-12
    # Check every selected balanced learner's stored training ID boundaries and
    # its actual checkpoint checksum, including validation and final refits.
    import specific_family_selection_v76 as prior
    checkpoint_count = 0
    for meta in prior.parent.parent.RAW.frozen.cohort("classification", "seven"):
        tid = str(meta["task_id"])
        pool = prior.parent.parent.RAW.scaling.prepare("classification", "seven", meta)
        split = prior.parent.parent.partition(meta)
        for family, choice in policy["Specific_recipes"].items():
            if choice is None or choice["recipe"]["kind"] != "balanced_bags": continue
            directory = prior.OUT / "experts" / tid / family
            registry = json.loads((directory / "models_complete.json").read_text())
            for record in registry["records"]:
                if record["member"] >= choice["recipe"]["members"]: continue
                ids = np.asarray(record["original_ids"])
                assert len(np.unique(ids)) == record["unique_rows"] <= 256
                assert not np.intersect1d(ids, pool["query_ids"]).size
                if record["role"] == "validation":
                    assert not np.intersect1d(ids, split["validation_original_ids"]).size
                path = directory / f"{record['role']}_{record['member']}.joblib"
                assert experiment.original.file_sha(path) == record["model_sha256"]
                assert not record["query_features_or_labels_used_for_fit"]
                checkpoint_count += 1
    data = pd.DataFrame(rows); data.to_csv(root / "audited_detail.csv", index=False)
    comparisons, ranking = [], []
    targets = json.loads((experiment.original.banks.OUT / "paper_targets.json").read_text())
    for (task, block), group in data.groupby(["task", "block"]):
        keys = ["accuracy", "auc"] if task == "classification" else ["r2", "nrmse"]
        values = {}
        for method in ["Base", "Shared", "Specific"]:
            cells = group[group.method == method]
            scores = cells.groupby("task_id")[keys].mean().mean()
            values[method] = scores.to_dict()
            row = dict(task=task, block=block, method=method)
            for key, target in zip(keys, targets[task][block][method]):
                row[key] = float(scores[key]); row["paper_" + key] = target; row["difference_" + key] = float(scores[key]) - target
            comparisons.append(row)
        for key in keys:
            sign = -1 if key == "nrmse" else 1
            ranking.append(dict(task=task, block=block, metric=key,
                Shared_better_than_Base=sign * (values["Shared"][key] - values["Base"][key]) > 0.,
                Specific_better_than_Shared=sign * (values["Specific"][key] - values["Shared"][key]) > 0.))
    pd.DataFrame(comparisons).to_csv(root / "comparison_with_paper.csv", index=False)
    atomic_json(root / "observed_rankings.json", ranking)
    summary = dict(status="PASS", conditions=sum(counts.values()), method_records=len(rows),
        counts=counts, paired_Base_and_query_labels_unchanged=True, metrics_recomputed_from_saved_arrays=True,
        regression_residual_max_error=maximum, regression_source_selection_trials_recomputed=source_trials,
        Specific_selector_dataset_crossfits_verified=len(policy["family_dataset_crossfits"]),
        selected_balanced_checkpoint_hashes_and_training_IDs_verified=checkpoint_count,
        all_requested_strict_rankings_observed=all(r["Shared_better_than_Base"] and r["Specific_better_than_Shared"] for r in ranking),
        numerical_paper_replication_claim=False, original_TabACE_replication_claim=False,
        independent_test_claim=False, extra_clean_TRAIN_supervision=True, reused_checkpoints=True,
        full_fresh_TFM_inference=False, elapsed_seconds=time.time()-started,
        Markdown_bytes=(root.parents[3]/"REBUTTAL_EXPERIMENTS.md").stat().st_size)
    atomic_json(root / "audit.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__": audit()
