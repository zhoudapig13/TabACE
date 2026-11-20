"""Correct pooled calibration provenance without changing any numeric result.

CA2018 calibration labels are source labels for geographic evaluation, but are
target-year labels for temporal evaluation. Earlier Shared policies pooled them
with CA2014 labels. V7 Specific inherits that entire pooled Shared mapping.
"""
import csv
import io
import json

from prepare_acs import RUN, save_json, sha


def main():
    corrected = []
    folders = [RUN / "results"] + [RUN / "results" / name for name in [
        "expanded_adapter_v2", "coverage_adapter_v3", "row_budget_adapter_v4",
        "matched_row_budget_v5", "fixed_512_v6", "nested_512_v7"]]
    for folder in folders:
        path = folder / "comparison_long.csv"
        if not path.exists():
            continue
        backup = folder / "comparison_long_before_supervision_metadata_correction.csv"
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            names, rows = list(reader.fieldnames), list(reader)
        new_fields = ["target_domain_calibration_labels_used", "calibration_supervision_domains",
                      "strict_own_source_domain_calibration"]
        for name in new_fields:
            if name not in names:
                names.append(name)
        for row in rows:
            method, scenario = row["method"], row["scenario"]
            pooled = method == "Shared" or (folder.name == "nested_512_v7" and method == "Specific")
            uses_target_year = pooled and scenario == "temporal"
            row["target_domain_calibration_labels_used"] = str(uses_target_year)
            row["calibration_supervision_domains"] = "none" if method == "Base" else "CA2014+CA2018" if pooled else "2018_CA" if scenario == "geographic" else "2014_CA"
            row["strict_own_source_domain_calibration"] = str(method != "Base" and not pooled)
            row["source_only_adaptation"] = str(method != "Base" and not uses_target_year)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=names, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        path.write_text(buffer.getvalue())
        # CSV strings representing measurements must be preserved exactly.
        with backup.open(newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        for old, new in zip(old_rows, rows):
            for key in ["accuracy", "auc", "log_loss", "paper_accuracy", "difference_from_paper"]:
                assert old[key] == new[key]
        note = {"pooled_Shared_calibration_domains": ["CA2014", "CA2018"],
                "temporal_Shared_includes_labeled_target_year_calibration": True,
                "temporal_Specific_inherits_target_year_calibration": folder.name == "nested_512_v7",
                "test_labels_used_for_fitting_or_selection": False,
                "numeric_results_unchanged": True,
                "strict_source_only_replacement": str(RUN / "results/strict_source_nested_512_v8")}
        save_json(folder / "supervision_scope_correction.json", note)
        corrected.append({"directory": str(folder), "numeric_results_unchanged": True,
                          "original_csv_sha256": sha(backup), "corrected_csv_sha256": sha(path)})
    save_json(RUN / "configs/supervision_metadata_correction.json", {
        "reason": "Pooled source-year labels become target-year supervision in temporal evaluation.",
        "corrected_results": corrected, "test_labels_were_never_used_for_training_or_selection": True})
    print("Corrected calibration provenance for", len(corrected), "tables; all numeric strings unchanged.")


if __name__ == "__main__":
    main()
