"""Independent verification and ddof=1 statistics for all 30 repetitions."""
from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from audit_natural import independent_fusion
from multi_seed_natural_v13 import (DEST, KINDS, PLAN, ROLES, SCENARIOS, TASKS,
                                   case_key, probability_frame)
from prepare_acs import RUN, save_json, sha


def load(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def calibrated(p, theta):
    if theta == [0., 0.]:
        return p.copy()
    logits = np.log(p[:, 1]/p[:, 0])
    positive = expit(np.exp(theta[0])*logits+theta[1])
    return np.column_stack([1.-positive, positive])


def case_audit(seed, scenario, task, plan):
    folder = DEST / f"seed_{seed}" / scenario / task
    complete = json.loads((folder / "completion.json").read_text())
    frozen = json.loads((folder / "frozen_before_test.json").read_text())
    assert complete["status"] == "COMPLETE" and complete["plan_sha256"] == sha(PLAN)
    assert frozen["status"] == "FROZEN" and frozen["plan_sha256"] == sha(PLAN)
    assert frozen["seed"] == seed and not frozen["test_labels_accessed"]
    assert not frozen["per_seed_hyperparameter_search"]
    configured = plan["cases"][case_key(scenario, task)]
    assert frozen["case_configuration"] == configured
    assert frozen["calibration_domains"] == [configured["source_domain"]]
    for method, record in frozen["models"].items():
        path = Path(record["model_path"])
        assert sha(path) == record["sha256"]
        model = CatBoostClassifier(); model.load_model(str(path))
        assert model.get_all_params()["random_seed"] == seed == record["actual_random_seed"]
        expected = configured["Specific_router"] if method == "Specific" else configured["Shared_supervised_router"]
        assert record["config"] == expected
        assert model.tree_count_ == expected["iterations"]
        assert model.get_all_params()["depth"] == expected["depth"]
    source = load(RUN / "data/splits" / f"{configured['source_domain']}_{task}_source.npz")
    expected_base = np.random.default_rng(seed).permutation(len(source["y"]))[:256]
    assert frozen["base_source_pool_indices"] == expected_base.tolist()
    assert frozen["Shared_fusion_fit"]["regularization"] == configured["Shared_fusion_regularization"]
    table = pd.read_csv(folder / "metrics.csv")
    assert len(table) == 6 and set(table.seed) == {seed}
    signatures, maximum_error = {}, 0.
    for role in ROLES:
        for kind in ["Base"]+KINDS:
            path = folder / "banks" / f"{role}__{kind}.npz"
            info = json.loads(path.with_suffix(".json").read_text())
            values = load(path)
            assert info["seed"] == seed == info["TabICL_random_state"]
            assert len(values["source_pool_indices"]) <= (256 if kind == "Base" else 512)
            assert np.all((values["source_pool_indices"] >= 0) & (values["source_pool_indices"] < 2048))
            if kind == "Base":
                assert np.array_equal(values["source_pool_indices"], expected_base)
            if kind == "feature_subspace":
                assert np.array_equal(values["source_pool_indices"], np.random.default_rng(seed).permutation(2048)[:512])
            if role == "ID":
                signatures[kind] = info["input_sha256"]
    for role in ["ID", "OOD"]:
        saved = load(folder / f"{role}_predictions.npz")
        domain = configured["source_domain"] if role == "ID" else configured["target_domain"]
        truth = load(RUN / "data/splits" / f"{domain}_{task}_test_labels.npz")
        assert np.array_equal(saved["y"], truth["y"])
        assert np.array_equal(saved["row_ids"], truth["row_ids"])
        for method in ["Base", "Shared", "Specific"]:
            p = saved[method]
            assert np.all(np.isfinite(p)) and np.all(p >= 0.)
            assert np.max(np.abs(p.sum(1)-1.)) < 1e-8
            record = table[(table.domain == role) & (table.method == method)].iloc[0]
            assert abs(accuracy_score(saved["y"], p.argmax(1))-record.accuracy) < 1e-12
            assert abs(roc_auc_score(saved["y"], p[:, 1])-record.auc) < 1e-12
            assert abs(log_loss(saved["y"], p, labels=[0, 1])-record.log_loss) < 1e-12
        bank = np.stack([load(folder / "banks" / f"{role}__{kind}.npz")["prediction"] for kind in KINDS])
        core, _ = independent_fusion(saved["Base"], bank[:5], np.asarray(frozen["Shared_fusion_parameters"]))
        core = calibrated(core, frozen["Shared_head_parameters"])
        error = float(np.max(np.abs(core-saved["Shared_core"])))
        assert error < 1e-10
        maximum_error = max(maximum_error, error)
        if configured["Shared_supervised_router"] is None:
            assert np.max(np.abs(saved["Shared"]-core)) < 1e-10
        else:
            model = CatBoostClassifier(); model.load_model(frozen["models"]["Shared"]["model_path"])
            features = load(RUN / "data/splits" / f"{domain}_{task}_test_features.npz")["X"]
            names = json.loads((RUN / "data/splits" / f"{configured['source_domain']}_{task}.json").read_text())["features"]
            frame, _ = probability_frame(features, names, saved["Base"], core, bank[:5])
            router = model.predict_proba(frame)
            candidates = np.stack([core]+list(bank[:5]), axis=1)
            positive = np.clip(router[:, 1], candidates[:, :, 1].min(1), candidates[:, :, 1].max(1))
            beta = configured["Shared_supervised_router"]["beta"]
            expected = (1.-beta)*core[:, 1]+beta*positive
            assert np.max(np.abs(expected-saved["Shared"][:, 1])) < 1e-10
        weights = saved["router_weights"]
        candidates = np.stack([saved["Shared"]]+list(bank), axis=1)
        assert weights.shape == (len(saved["y"]), 9) and np.all(weights >= 0.)
        assert np.max(np.abs(weights.sum(1)-1.)) < 1e-12
        predicted = (weights[:, :, None]*candidates).sum(1)
        error = float(np.max(np.abs(predicted-saved["Specific"])))
        assert error < 1e-12
        maximum_error = max(maximum_error, error)
        positive = np.clip(saved["router_prediction"][:, 1], candidates[:, :, 1].min(1), candidates[:, :, 1].max(1))
        beta = configured["Specific_router"]["beta"]
        assert np.max(np.abs((1.-beta)*saved["Shared"][:, 1]+beta*positive-saved["Specific"][:, 1])) < 1e-12
    return table, {"seed": seed, "scenario": scenario, "task": task,
                   "max_reconstruction_error": maximum_error, "ID_input_signatures": signatures,
                   "seed0_reference_checks": complete["seed0_reference_checks"]}


def main():
    plan = json.loads(PLAN.read_text())
    assert plan["seeds"] == [0, 1, 2, 3, 4]
    for filename, expected in plan["dataset_sha256"].items():
        assert sha(RUN / "data/splits" / filename) == expected
    assert sha(Path(plan["checkpoint"])) == plan["checkpoint_sha256"]
    tables, checks = [], []
    for seed in plan["seeds"]:
        for scenario in SCENARIOS:
            for task in TASKS:
                table, audit = case_audit(seed, scenario, task, plan)
                tables.append(table); checks.append(audit)
    frame = pd.concat(tables, ignore_index=True)
    identifiers = ["seed", "scenario", "task", "domain", "method"]
    assert len(frame) == 180 and len(frame.drop_duplicates(identifiers)) == 180
    frame.to_csv(DEST / "all_seed_metrics.csv", index=False)
    summary = []
    for key, group in frame.groupby(["scenario", "task", "domain", "method"], sort=False):
        assert sorted(group.seed.tolist()) == plan["seeds"]
        row = dict(zip(["scenario", "task", "domain", "method"], key))
        row["seeds"] = "0,1,2,3,4"; row["n_seeds"] = 5
        for metric in ["accuracy", "auc", "log_loss"]:
            values = group[metric].to_numpy(dtype=float)
            row[metric+"_mean"] = float(values.mean())
            row[metric+"_std"] = float(values.std(ddof=1))
            row[metric+"_variance"] = float(values.var(ddof=1))
            # Independently check the unbiased sample-variance formula.
            manual = float(sum((value-float(values.mean()))**2 for value in values)/(len(values)-1))
            assert abs(manual-row[metric+"_variance"]) < 1e-15
        summary.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(DEST / "summary_mean_std_variance.csv", index=False)
    accuracy = frame.pivot(index=["seed", "scenario", "task", "domain"], columns="method", values="accuracy").reset_index()
    accuracy["Specific_minus_Shared"] = accuracy.Specific-accuracy.Shared
    accuracy["Specific_gain_at_least_0005"] = accuracy.Specific_minus_Shared >= .005-1e-12
    accuracy.to_csv(DEST / "per_seed_accuracy_and_gain.csv", index=False)
    gain_rows = []
    for key, group in accuracy.groupby(["scenario", "task", "domain"], sort=False):
        values = group.Specific_minus_Shared.to_numpy()
        gain_rows.append(dict(zip(["scenario", "task", "domain"], key)) | {
            "gain_mean": float(values.mean()), "gain_std": float(values.std(ddof=1)),
            "gain_variance": float(values.var(ddof=1)), "gain_min": float(values.min()),
            "gain_max": float(values.max()), "positive_seed_count": int((values > 0.).sum()),
            "seeds_meeting_gain_0005": int((values >= .005-1e-12).sum()), "n_seeds": 5})
    pd.DataFrame(gain_rows).to_csv(DEST / "paired_gain_summary.csv", index=False)
    display = summary_frame.copy()
    display["mean_std_variance"] = display.apply(lambda row: f"{row.accuracy_mean:.4f} ± {row.accuracy_std:.4f} ({row.accuracy_variance:.3e})", axis=1)
    display_table = display.pivot(index=["scenario", "task", "domain"], columns="method", values="mean_std_variance")
    display_table = display_table.reindex(columns=["Base", "Shared", "Specific"])
    display_table.to_csv(DEST / "formatted_accuracy_table.csv")
    for scenario in SCENARIOS:
        for task in TASKS:
            rows = [item for item in checks if item["scenario"] == scenario and item["task"] == task]
            assert len({item["ID_input_signatures"]["Base"] for item in rows}) == 5
            assert len({item["ID_input_signatures"]["feature_subspace"] for item in rows}) == 5
    reference = [row for item in checks if item["seed"] == 0 for row in item["seed0_reference_checks"]]
    assert len(reference) == 12
    assert all(row["Shared_hard_disagreements"] == 0 and row["Specific_hard_disagreements"] == 0 for row in reference)
    audit = {"status": "PASS", "seeds": plan["seeds"], "full_repetitions_verified": len(checks),
        "individual_metric_rows": len(frame), "summary_cells": len(summary_frame),
        "variance_ddof": 1, "variance_units": "Accuracy squared",
        "fixed_data_splits_verified": True, "hyperparameters_unchanged_across_seeds": True,
        "all_seeds_retained": True, "context_sampling_actually_changes_across_seeds": True,
        "TabICL_random_state_propagated": True, "CatBoost_seed_verified_from_models": True,
        "seed0_hard_predictions_reproduce_selected_reference": True,
        "maximum_reconstruction_error": max(row["max_reconstruction_error"] for row in checks),
        "source_only_fitting": True, "test_labels_used_for_model_selection": False,
        "supervised_adaptation_not_original_paper_reproduction": True,
        "MD_summary_still_empty": (RUN.parents[1] / "REBUTTAL_EXPERIMENTS.md").stat().st_size == 0,
        "case_checks": checks, "plan_sha256": sha(PLAN), "report_script_sha256": sha(Path(__file__))}
    save_json(DEST / "audit.json", audit)
    save_json(DEST / "completion.json", {"status": "COMPLETE", "seeds": plan["seeds"],
        "repetitions": 30, "individual_metric_rows": 180, "summary_cells": 36,
        "sample_variance_ddof": 1, "audit": str(DEST / "audit.json")})
    print(display_table.to_string(), flush=True)
    print("AUDIT PASS: 30 repetitions, 180 metric rows, five seeds in every summary cell", flush=True)


if __name__ == "__main__":
    main()
