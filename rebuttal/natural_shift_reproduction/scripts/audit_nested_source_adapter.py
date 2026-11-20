"""Independent checks of exact inheritance, source selection and saved metrics."""
import json
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss

from audit_natural import independent_fusion, load
from prepare_acs import RUN, save_json, sha

DEST = RUN / "results/nested_512_v7"


def raw(policy, scenario, task, role):
    b = load(RUN / "results/banks" / scenario / task / f"{role}__Base.npz")["prediction"]
    members = []
    for name in policy["members"]:
        prefix, kind = name.split("::", 1)
        budget = int(prefix.removeprefix("rows_"))
        assert budget == 512
        cached = load(RUN / "results/row_banks" / scenario / task / f"n{budget}_{role}__{kind}.npz")
        assert len(cached["context_source_pool_indices"]) <= 512
        members.append(cached["prediction"])
    output, _ = independent_fusion(b, np.stack(members), np.asarray(policy["parameters"]))
    return output


def calibrate(p, theta):
    if theta == [0., 0.]:
        return p.copy()
    log_odds = np.log(p[:, 1]/p[:, 0])
    positive = expit(np.exp(theta[0])*log_odds+theta[1])
    return np.column_stack([1.-positive, positive])


def main(strict=False):
    global DEST
    if strict:
        DEST = RUN / "results/strict_source_nested_512_v8"
    frame = pd.read_csv(DEST / "comparison_long.csv")
    assert len(frame) == 36
    frozen = json.loads((DEST / "policies/frozen_before_test.json").read_text())
    assert frozen["status"] == "FROZEN"
    max_error = 0.
    source_validation_checks = []
    for scenario in ["geographic", "temporal"]:
        for task in ["Employment", "Income", "TravelTime"]:
            shared_name = f"Shared_{scenario}_{task}.json" if strict else f"Shared_{task}.json"
            shared = json.loads((DEST / "policies" / shared_name).read_text())
            specific = json.loads((DEST / "policies" / f"Specific_{scenario}_{task}.json").read_text())
            assert not shared["test_labels_accessed"] and not specific["test_labels_accessed"]
            if strict:
                expected = ["2018_CA" if scenario == "geographic" else "2014_CA"]
                assert shared["calibration_domains"] == expected
                assert specific["calibration_domains"] == expected
                assert shared["source_policy"]["calibration_domains"] == expected
                assert shared["target_domain_calibration_labels_used"] is False
                assert specific["target_domain_calibration_labels_used"] is False
            assert shared["head_parameters"] == specific["shared_head_parameters"]
            assert shared["context_type_count"] == 5 and specific["context_type_count"] == 8
            assert specific["exact_shared_fallback_available"]
            assert 0. <= specific["beta"] <= 1.
            trials = specific["beta_selection_trials"]
            assert any(trial["beta"] == 0. for trial in trials)
            assert specific["selected"]["accuracy"] >= specific["shared_fallback_source_validation"]["accuracy"]
            assert specific["selected"] == max(trials, key=lambda row: (row["accuracy"], -row["log_loss"], -row["beta"]))
            source_validation_checks.append({"scenario": scenario, "task": task,
                "selected_beta": specific["beta"],
                "accuracy_gain_over_shared": specific["selected"]["accuracy"]-specific["shared_fallback_source_validation"]["accuracy"]})
            for role in ["fit", "select", "ID", "OOD"]:
                shared_p = calibrate(raw(shared["source_policy"], scenario, task, role), shared["head_parameters"])
                expert_p = calibrate(raw(specific["source_policy"], scenario, task, role), specific["head_parameters"])
                fallback = (1.-0.)*shared_p+0.*expert_p
                assert np.array_equal(fallback, shared_p)
                beta = specific["beta"]
                result = shared_p.copy() if beta == 0. else (1.-beta)*shared_p+beta*expert_p
                if role not in ["ID", "OOD"]:
                    continue
                for method, prediction in [("Shared", shared_p), ("Specific", result)]:
                    saved = load(DEST / "predictions" / scenario / task / f"{role}_{method}.npz")
                    diff = float(np.max(np.abs(saved["prediction"]-prediction)))
                    max_error = max(max_error, diff)
                    assert diff < 1e-10
                    assert np.all(np.isfinite(prediction)) and np.all(prediction >= 0.)
                    assert np.max(np.abs(prediction.sum(1)-1.)) < 1e-10
                    record = frame[(frame.scenario == scenario) & (frame.task == task) & (frame.domain == role) & (frame.method == method)].iloc[0]
                    assert abs(float(np.mean(saved["y"] == prediction.argmax(1)))-record.accuracy) < 1e-12
                    assert abs(log_loss(saved["y"], prediction, labels=[0, 1])-record.log_loss) < 1e-12
    pairs = frame.pivot(index=["scenario", "domain", "task"], columns="method", values="accuracy")
    ordering = pairs.Specific-pairs.Shared
    source_heads = frame[frame.method != "Base"]
    summary = {"status": "PASS", "cells": 36, "prediction_max_error": max_error,
        "exact_shared_fallback_checked_on": "fit, select, ID, OOD for all six scenario/task cases",
        "source_validation_checks": source_validation_checks,
        "test_specific_strictly_higher_than_shared": int((ordering > 1e-12).sum()),
        "test_specific_equal_to_shared": int((ordering.abs() <= 1e-12).sum()),
        "test_specific_lower_than_shared": int((ordering < -1e-12).sum()),
        "adapted_test_cells_at_least_paper_accuracy": int((source_heads.difference_from_paper >= -1e-12).sum()),
        "adapted_test_cells_below_paper_accuracy": int((source_heads.difference_from_paper < -1e-12).sum()),
        "all_requested_test_thresholds_met": False,
        "test_labels_used_for_training_or_selection": False,
        "strict_own_source_domain_calibration": strict,
        "interpretation": "Exact functional inheritance and source-validation non-regression do not guarantee strict test improvement.",
        "code_sha256": {p.name: sha(p) for p in sorted((RUN / "scripts").glob("*.py"))},
        "user_md_summary_still_empty": (RUN.parents[1] / "REBUTTAL_EXPERIMENTS.md").stat().st_size == 0}
    save_json(DEST / "audit.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ["code_sha256", "source_validation_checks"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-source", action="store_true")
    main(parser.parse_args().strict_source)
