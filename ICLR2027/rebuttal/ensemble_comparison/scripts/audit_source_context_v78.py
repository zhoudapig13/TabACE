"""Verify source-context diagnostics independently of their fitted metrics."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from audit_paper_rule_v77 import independent_predict
from prepare_data import atomic_json
from run_tabicl import metrics
from shared_reference import load_condition
import source_context_diagnostic_v78 as experiment


def run():
    started = time.time()
    root = experiment.OUT
    plan = json.loads((root / "plan.json").read_text())
    assert len(list(root.glob("*/fold_*/completion.json"))) == 15
    for name, checksum in plan["code_sha256"].items():
        assert experiment.original.file_sha(Path(__file__).with_name(name)) == checksum
    max_error, count, metric_count, input_checks = 0., 0, 0, []
    bank_manifest = {}
    for setting, (task, track) in experiment.SETTINGS.items():
        cells = experiment.load(setting)
        expected = (72 if task == "classification" else 35) * 25
        assert len(cells) == expected
        policies = {fold: json.loads((root / setting / f"fold_{fold}/frozen_policy.json").read_text()) for fold in range(5)}
        sample_meta = cells[0]["meta"]
        # Reconstruct source corruptions for every family at the highest level;
        # verify that query features and labels are not modified by this track.
        clean, _ = load_condition(sample_meta, "original", 0)
        for family in sorted({c["meta"]["family"] for c in cells}):
            changed, details = load_condition(sample_meta, family, 100)
            assert np.array_equal(clean["X_target"], changed["X_target"])
            assert np.array_equal(clean["y_target"], changed["y_target"])
            input_checks.append({"setting": setting, "task_id": sample_meta["task_id"], "family": family,
                "source_features_changed": not np.array_equal(clean["X_source"], changed["X_source"]),
                "source_labels_changed": not np.array_equal(clean["y_source"], changed["y_source"]),
                "query_features_unchanged": True, "query_labels_unchanged": True})
        for cell in cells:
            meta, fold = cell["meta"], cell["fold"]
            policy = policies[fold]
            assert int(meta["data_id"]) in policy["excluded_data_ids"]
            assert int(meta["data_id"]) not in policy["training_data_ids"]
            assert set(policy["training_data_ids"]).isdisjoint(policy["excluded_data_ids"])
            for selection in [policy["Shared_selection"], *policy["Specific_selection"].values()]:
                assert set(selection["inner_train_data_ids"]).isdisjoint(selection["inner_validation_data_ids"])
            raw = {"base": cell["base"], "members": cell["bank"],
                   "source_mean": cell["mu"], "source_scale": cell["sigma"]}
            shared, _, _ = independent_predict(raw, policy["Shared"], task)
            specific, _, _ = independent_predict(raw, policy["Specific"][meta["family"]], task)
            output = root / setting / "predictions" / str(meta["task_id"]) / f"{meta['family']}_{meta['rho']}.npz"
            with np.load(output) as f:
                assert np.array_equal(f["Base"], cell["base"])
                assert np.array_equal(f["y_target"], cell["y"])
                for method, prediction in [("Base", cell["base"]), ("Shared", shared), ("Specific", specific)]:
                    error = float(np.max(np.abs(f[method] - prediction)))
                    assert error < 1e-12, (setting, output, method, error)
                    max_error = max(max_error, error)
                    metric_count += 1
            for path in [cell["path"], cell["specific_path"]]:
                key = str(path)
                if key not in bank_manifest:
                    bank_manifest[key] = {"sha256": experiment.original.file_sha(path),
                        "metadata_sha256": experiment.original.file_sha(path.with_suffix(".json"))}
            count += 1
    # Recompute every reported metric from the saved probability / prediction
    # array, rather than accepting previously assembled CSV scores.
    detail = pd.read_csv(root / "detail.csv")
    assert len(detail) == metric_count
    indexed = detail.set_index(["setting", "task_id", "family", "rho", "method"])
    for setting, (task, track) in experiment.SETTINGS.items():
        for path in (root / setting / "predictions").glob("*/*.npz"):
            family, rho_text = path.stem.rsplit("_", 1)
            rho, tid = int(rho_text), int(path.parent.name)
            with np.load(path) as f:
                for method in ["Base", "Shared", "Specific"]:
                    row = indexed.loc[(setting, tid, family, rho, method)]
                    recomputed = metrics(f["y_target"], f[method], task, row["scale"])
                    for key, value in recomputed.items():
                        assert (value is None and pd.isna(row[key])) or abs(value - row[key]) < 1e-12
    atomic_json(root / "source_input_semantics.json", input_checks)
    atomic_json(root / "reused_bank_hashes.json", bank_manifest)
    summary = {"status": "PASS", "conditions": count, "method_records": metric_count,
        "cached_bank_files": len(bank_manifest), "independent_fusion_max_error": max_error,
        "paired_Base_unchanged": True, "source_vs_query_semantics_checks": len(input_checks),
        "outer_and_inner_dataset_folds_disjoint": True,
        "original_generator_recovered": False, "fresh_full_backbone_replay": False,
        "development_calibrated_protocol_not_independent_test": True,
        "all_declared_protocols_retained": True, "elapsed_seconds": time.time() - started,
        "Markdown_bytes": (root.parents[3] / "REBUTTAL_EXPERIMENTS.md").stat().st_size}
    atomic_json(root / "audit.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
