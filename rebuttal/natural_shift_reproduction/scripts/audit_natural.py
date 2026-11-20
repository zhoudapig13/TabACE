"""Independently verify splits, saved predictions, metrics and fusion equations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from prepare_acs import RUN, save_json, sha


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def independent_fusion(base, bank, p):
    # Independent NumPy implementation of manuscript equations, rather than a
    # call back into the fitting/prediction implementation.
    def entropy(v):
        return -(v * np.log(np.maximum(v, 1e-10))).sum(-1) / np.log(2.)
    quality = (1. - entropy(bank)).T
    agreement = (1. - np.abs(bank-bank.mean(0)).sum(-1)/2.).T
    n = len(bank)
    score = p[:n] + quality*p[n:2*n] + agreement*p[2*n:3*n]
    exp_score = np.exp(score-score.max(1, keepdims=True))
    weight = exp_score / exp_score.sum(1, keepdims=True)
    alternative = (bank.transpose(1, 0, 2)*weight[:, :, None]).sum(1)
    alpha = p[-1]*entropy(base)*(weight*quality*agreement).sum(1)
    output = (1-alpha[:, None])*base + alpha[:, None]*alternative
    return output / output.sum(1, keepdims=True), alpha


def audit(destination):
    source_audit = []
    for meta_path in sorted((RUN / "data/splits").glob("*.json")):
        prefix = meta_path.with_suffix("")
        source = load(Path(str(prefix)+"_source.npz"))
        calibration = load(Path(str(prefix)+"_calibration.npz"))
        test = load(Path(str(prefix)+"_test_features.npz"))
        labels = load(Path(str(prefix)+"_test_labels.npz"))
        assert "y" not in test
        assert np.array_equal(test["row_ids"], labels["row_ids"])
        roles = [source, calibration, test]
        for i in range(3):
            for j in range(i):
                assert set(roles[i]["row_ids"]).isdisjoint(roles[j]["row_ids"])
                assert set(roles[i]["households"]).isdisjoint(roles[j]["households"])
        source_audit.append({"domain_task": meta_path.stem, "household_role_overlap": 0,
                             "source_rows": len(source["y"]), "calibration_rows": len(calibration["y"]),
                             "test_rows": len(labels["y"]),
                             "fit_select_rows_disjoint": True,
                             "fit_select_shared_households_within_calibration": len(set(calibration["households"][:2048]) & set(calibration["households"][2048:]))})
    frame = pd.read_csv(destination / "comparison_long.csv")
    assert len(frame) == 36 and len(frame.drop_duplicates(["scenario", "method", "domain", "task"])) == 36
    errors = []
    max_error = 0.
    for row in frame.to_dict("records"):
        s, t, d, m = row["scenario"], row["task"], row["domain"], row["method"]
        root = RUN / "results/banks" / s / t
        base = load(root / f"{d}__Base.npz")["prediction"]
        saved = load(destination / "predictions" / s / t / f"{d}_{m}.npz")
        p, y = saved["prediction"], saved["y"]
        if m == "Base":
            reconstructed, alpha = base, np.zeros(len(base))
        else:
            policy_path = destination / "policies" / (f"Shared_{t}.json" if m == "Shared" else f"Specific_{s}_{t}.json")
            policy = json.loads(policy_path.read_text())
            assert policy["test_labels_accessed"] is False
            if destination.name == "fixed_512_v6":
                assert policy["selected_row_budget"] == 512
                assert len(policy["members"]) == (5 if m == "Shared" else 8)
            members = []
            for name in policy["members"]:
                if name.startswith("rows_"):
                    prefix, branch_name = name.split("::", 1)
                    budget = int(prefix.removeprefix("rows_"))
                    path = RUN / "results/row_banks" / s / t / f"n{budget}_{d}__{branch_name}.npz"
                    context = load(path)
                    ids = context["context_source_pool_indices"]
                    assert len(ids) <= budget and np.all((ids >= 0) & (ids < 2048))
                else:
                    path = root / f"{d}__{name}.npz"
                members.append(load(path)["prediction"])
            bank = np.stack(members)
            reconstructed, alpha = independent_fusion(base, bank, np.asarray(policy["parameters"]))
        max_error = max(max_error, float(np.max(np.abs(p-reconstructed))))
        assert np.allclose(p, reconstructed, atol=1e-10)
        assert np.allclose(alpha, saved["gate_alpha"], atol=1e-10)
        assert np.all((alpha >= 0.) & (alpha <= 1.))
        assert np.all(np.isfinite(p)) and np.all(p >= 0.)
        assert np.max(np.abs(p.sum(1)-1.)) < 1e-8
        assert abs(float(np.mean(p.argmax(1)==y))-row["accuracy"]) < 1e-12
        assert abs(log_loss(y, p, labels=[0, 1])-row["log_loss"]) < 1e-12
        assert row["seed"] == 0
        if row["delta_from_current_base"] < 0.:
            errors.append({key: row[key] for key in ["scenario", "task", "domain", "method", "delta_from_current_base"]})
    policy_freeze = json.loads((destination / "policies/frozen_before_test.json").read_text())
    assert policy_freeze["status"] == "FROZEN"
    audit_record = {
        "status": "PASS", "table_cells": len(frame), "seed": 0,
        "split_checks": source_audit, "fusion_max_absolute_error": max_error,
        "test_labels_used_for_policy_selection": False,
        "test_labels_used_for_context_selection": False,
        "original_configuration_recovered": False,
        "lower_than_current_base_cases": errors,
        "mean_absolute_paper_gap_by_method": frame.groupby("method").difference_from_paper.apply(lambda v: float(v.abs().mean())).to_dict(),
        "source_files_sha256": {p.name: sha(p) for p in sorted((RUN / "scripts").glob("*.py"))},
        "checkpoint_sha256": sha(RUN.parents[2] / "TabSwift/artifacts/checkpoints/tabicl-classifier-v2-20260212.ckpt"),
        "empty_user_summary_md_preserved": (RUN.parents[1] / "REBUTTAL_EXPERIMENTS.md").stat().st_size == 0,
    }
    save_json(destination / "audit.json", audit_record)
    return audit_record


def diagnose_shared():
    # Diagnose, without changing selection after seeing test results.
    root = RUN / "results/banks/temporal/Employment"
    policy = json.loads((RUN / "results/expanded_adapter_v2/policies/Shared_Employment.json").read_text())
    records = []
    with np.load(RUN / "data/splits/2014_CA_Employment_calibration.npz") as data:
        selection_y = data["y"][2048:]
    for role in ["select", "OOD"]:
        domain = "2014_CA" if role == "select" else "2018_CA"
        y = selection_y if role == "select" else load(RUN / "data/splits" / f"{domain}_Employment_test_labels.npz")["y"]
        for name in ["Base"] + policy["members"]:
            p = load(root / f"{role}__{name}.npz")["prediction"]
            records.append({"role": role, "branch": name,
                            "accuracy": float(np.mean(p.argmax(1)==y)),
                            "log_loss": float(log_loss(y, p, labels=[0, 1])),
                            "predicted_positive_fraction": float(np.mean(p.argmax(1))),
                            "observed_positive_fraction": float(y.mean())})
    save_json(RUN / "results/expanded_adapter_v2/shared_temporal_employment_diagnostic.json", {
        "purpose": "separate implementation error from domain-generalization failure",
        "observations": records, "policy": policy["selected_bank"],
        "conclusion": "Source-validation selection improved CE, but temporal OOD accuracy declined. Independent reconstruction verifies fusion and metrics. This remains an observed generalization failure, not a recovered original configuration.",
        "posthoc_test_diagnosis_used_to_reselect_policy": False,
    })
    return records


if __name__ == "__main__":
    destinations = [RUN / "results", RUN / "results/expanded_adapter_v2", RUN / "results/coverage_adapter_v3", RUN / "results/row_budget_adapter_v4", RUN / "results/matched_row_budget_v5", RUN / "results/fixed_512_v6"]
    for destination in destinations:
        if not (destination / "completion.json").exists():
            continue
        answer = audit(destination)
        print(destination.name, answer["status"], "36 cells", "fusion error", answer["fusion_max_absolute_error"])
    if (RUN / "results/expanded_adapter_v2/policies/Shared_Employment.json").exists():
        records = diagnose_shared()
        for row in records:
            print(row)
