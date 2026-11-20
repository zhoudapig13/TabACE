"""Recompute metrics from saved probabilities and audit the official correction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from raw_acs_external_v14 import RUN, OUT, load, prefixes, sha, save_json


def main():
    records, checks = [], []
    for seed in range(5):
        for scenario in ["geographic", "temporal"]:
            for task in ["Employment", "Income", "TravelTime"]:
                source, target, year = prefixes(scenario, task)
                pool = load(source, "source")
                context = np.random.default_rng(seed).permutation(len(pool["y"]))[:512]
                for backend in ["distpfn", "drift"]:
                    stem = OUT / f"{backend}__{scenario}__{task}__seed{seed}"
                    meta = json.loads(stem.with_suffix(".json").read_text())
                    assert meta["seed"] == seed and meta["context_rows"] == 512
                    assert meta["source_year"] == year and meta["target_year"] == 2018
                    assert not meta["test_labels_accessed_during_inference"]
                    assert not meta["calibration_labels_used"] and not meta["finetuning"]
                    assert sha(Path(meta["checkpoint"])) == meta["checkpoint_sha256"]
                    expected_members = 1 if backend == "distpfn" else 32
                    assert meta["n_estimators"] == expected_members
                    old_rows = json.loads(stem.with_name(stem.name + "_metrics.json").read_text())
                    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as bank:
                        assert np.array_equal(context, bank["context_indices"])
                        assert np.array_equal(pool["row_ids"][context], bank["context_row_ids"])
                        max_correction_error = 0.0
                        for domain, prefix in [("ID", source), ("OOD", target)]:
                            truth = load(prefix, "test_labels")
                            features = load(prefix, "test_features")
                            assert np.array_equal(bank[f"{domain}__row_ids"], truth["row_ids"])
                            assert np.array_equal(features["row_ids"], truth["row_ids"])
                            assert not set(pool["households"][context]) & set(features["households"])
                            if backend == "distpfn":
                                base = bank[f"{domain}__TabICL-v2-512"]
                                train_prior = np.bincount(pool["y"][context], minlength=2) / 512
                                unnormalized = base * base.mean(0) / (train_prior + 1e-8)
                                expected = unnormalized / unnormalized.sum(1, keepdims=True)
                                err = np.max(np.abs(expected - bank[f"{domain}__DistPFN-TabICL-v2"]))
                                assert err < 1e-12
                                max_correction_error = max(max_correction_error, float(err))
                            for method in meta["methods"]:
                                p = bank[f"{domain}__{method}"]
                                assert p.shape == (len(truth["y"]), 2)
                                assert np.isfinite(p).all() and (p >= 0).all()
                                assert np.allclose(p.sum(1), 1, atol=1e-5)
                                computed = {
                                    "accuracy": accuracy_score(truth["y"], p.argmax(1)),
                                    "auc": roc_auc_score(truth["y"], p[:, 1]),
                                    "log_loss": log_loss(truth["y"], p, labels=[0, 1])}
                                old = next(r for r in old_rows if r["domain"] == domain and r["method"] == method)
                                for key, value in computed.items():
                                    assert abs(value - old[key]) < 1e-12
                                records.append({"seed": seed, "scenario": scenario,
                                                "task": task, "domain": domain,
                                                "method": method, "n": len(p), **computed})
                    checks.append({"seed": seed, "scenario": scenario, "task": task,
                                   "backend": backend, "max_correction_error": max_correction_error})
    frame = pd.DataFrame(records)
    assert len(frame) == 180 and not frame.duplicated(["seed", "scenario", "task", "domain", "method"]).any()
    frame.to_csv(OUT / "per_seed_metrics.csv", index=False)
    rows = []
    for identifiers, group in frame.groupby(["scenario", "task", "domain", "method"], sort=False):
        assert sorted(group.seed.tolist()) == list(range(5))
        row = dict(zip(["scenario", "task", "domain", "method"], identifiers))
        row["n_seeds"] = 5
        for metric in ["accuracy", "auc", "log_loss"]:
            row.update({f"{metric}_mean": group[metric].mean(),
                        f"{metric}_sd": group[metric].std(ddof=1),
                        f"{metric}_sample_variance": group[metric].var(ddof=1)})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    save_json(OUT / "audit.json", {
        "status": "PASS", "jobs_checked": len(checks), "metric_rows": len(frame),
        "summary_rows": len(summary), "ddof": 1, "fixed_evaluation_split": True,
        "seed_randomness": "context sampling and model inference random state",
        "original_paper_split_recovered": False, "checks": checks})
    for scenario in ["geographic", "temporal"]:
        print(scenario)
        for method in ["TabICL-v2-512", "DistPFN-TabICL-v2", "Drift-Resilient"]:
            for domain in ["ID", "OOD"]:
                pieces = []
                for task in ["Employment", "Income", "TravelTime"]:
                    row = summary[(summary.scenario == scenario) & (summary.method == method)
                                  & (summary.domain == domain) & (summary.task == task)].iloc[0]
                    pieces.append(f"{row.accuracy_mean:.4f} ± {row.accuracy_sd:.4f}")
                print(method, domain, " | ".join(pieces))
    print("AUDIT PASS: 60 jobs, 180 metric rows, 36 summaries; sample SD/variance ddof=1")


if __name__ == "__main__":
    main()
