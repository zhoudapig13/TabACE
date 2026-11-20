"""Recompute frozen development Base curves from every saved prediction.

This verifies the existing v43 result; it does not select runs, change query
conditions, modify predictions, or imply independent reproduction evidence.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "results/tabicl_context_small_rise_audit_v69"
BUDGETS = [256, 1024, 4096, 8192]


def audit():
    registry_path = RUN / "results/tabicl_base_context_recovery_frozen_v43/parameters.json"
    registry = json.loads(registry_path.read_text())
    for name, digest in registry["frozen_artifact_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    roots = {
        "five": RUN / "results/tabicl_offset_response_recovery_v41",
        "seven": RUN / "results/tabicl_half_coupled_recovery_v42",
    }
    results = []
    checks = []
    verified = 0
    maximum_error = 0.0
    for block, root in roots.items():
        parameter = root / f"parameters_{block}.json"
        parameter_hash = hashlib.sha256(parameter.read_bytes()).hexdigest()
        expected_datasets = 35 if block == "five" else 33
        expected_families = 5 if block == "five" else 7
        first_keys = None
        first_queries = {}
        curve = []
        for budget in BUDGETS:
            records = []
            keys = set()
            datasets = set()
            for path in sorted((root / "query" / str(budget) / block).glob("*/*.json")):
                row = json.loads(path.read_text())
                assert row["seed"] == 0 and row["n_estimators"] == 1
                assert row["parameter_sha256"] == parameter_hash
                assert 0 < row["actual_context"] <= budget
                assert row["requested_context"] == budget
                assert row["block"] == block and row["rho"] in [20, 50, 80, 100]
                key = row["task_id"], row["family"], row["rho"]
                assert key not in keys
                keys.add(key)
                datasets.add(row["task_id"])
                query_hashes = row["input_hashes"]["X_target"], row["input_hashes"]["y_target"]
                if budget == 256:
                    first_queries[key] = query_hashes
                else:
                    assert first_queries[key] == query_hashes
                with np.load(path.with_suffix(".npz")) as arrays:
                    prediction = arrays["prediction"]
                    truth = arrays["y_target"]
                    assert prediction.shape == truth.shape
                    assert np.isfinite(prediction).all() and np.isfinite(truth).all()
                    squared = np.square(prediction - truth)
                    measured = {
                        "r2": 1.0 - float(squared.sum()) / max(float(np.square(truth - truth.mean()).sum()), 1e-12),
                        "nrmse": float(np.sqrt(squared.mean())) / row["scale"],
                        "nmae": float(np.abs(prediction - truth).mean()) / row["scale"],
                    }
                    error = max(abs(measured[k] - row["metrics"][k]) for k in measured)
                    maximum_error = max(maximum_error, error)
                    assert error < 1e-12, (path, error)
                records.append((measured["r2"], measured["nrmse"], row["actual_context"]))
                verified += 1
            assert len(datasets) == expected_datasets
            assert len(keys) == expected_datasets * expected_families * 4
            if first_keys is None:
                first_keys = keys
            else:
                assert first_keys == keys
            means = np.asarray(records).mean(axis=0)
            curve.append(means[:2])
            results.append(dict(block=block, requested_context=budget, datasets=len(datasets),
                                conditions=len(keys), r2=float(means[0]), nrmse=float(means[1]),
                                actual_context_mean=float(means[2])))
        curve = np.asarray(curve)
        assert np.all(np.diff(curve[:, 0]) > 0)
        assert np.all(np.diff(curve[:, 1]) < 0)
        gain = float(curve[-1, 0] - curve[0, 0])
        assert 0 < gain <= 0.08
        checks.append(dict(block=block, strict_mean_r2_increase=True, strict_mean_nrmse_decrease=True,
                           final_r2_gain=gain, r2_steps=np.diff(curve[:, 0]).tolist(),
                           same_datasets_conditions_and_query_inputs=True))
    assert verified == 6496
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "regression_context_averages.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = dict(status="PASS", verified_saved_predictions=verified, maximum_metric_error=maximum_error,
                  registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                  seed=0, n_estimators=1, frozen_configuration_hashes_verified=True, checks=checks,
                  development_only=True, extra_clean_TRAIN_supervision=True,
                  query_feedback_used_for_protocol_design=True, independent_test_claim=False,
                  original_paper_replication=False, pure_context_scaling_claim=False,
                  unchanged_corrupted_core256=True, predictions_modified=False,
                  Markdown_written=False, time_unix=time.time())
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(audit=report, averages=results), indent=2), flush=True)


if __name__ == "__main__":
    audit()
