from __future__ import annotations

import csv
import json
import math
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
METHODS = {"tabicl_base", "random_context_6", "shared_context_6", "specific_context_6"}


def test_component_profile_is_complete_and_finite():
    rows = list(csv.DictReader((RESULTS / "component_timing_detail.csv").open()))
    assert len(rows) == 2 * 3 * 7 * 4
    assert {row["method"] for row in rows} == METHODS
    for row in rows:
        expected = 1 if row["method"] == "tabicl_base" else 6
        assert int(row["logical_calls"]) == expected
        assert int(row["observed_backbone_calls"]) == expected
        values = [
            float(row["total_seconds"]),
            float(row["backbone_seconds"]),
            float(row["context_and_fusion_seconds"]),
            float(row["backbone_fraction"]),
        ]
        assert all(math.isfinite(value) and value >= 0.0 for value in values)
        assert float(row["total_seconds"]) >= float(row["backbone_seconds"])
        assert 0.0 <= float(row["backbone_fraction"]) <= 1.0


def test_metadata_records_the_timing_boundary():
    metadata = json.loads((RESULTS / "metadata.json").read_text())
    assert metadata["cuda_synchronized"] is True
    assert metadata["checkpoint_warmed_once_per_task"] is True
    assert metadata["shift_construction_included"] is False
    assert metadata["context_construction_included"] is True
    assert metadata["fusion_included"] is True


def test_classification_context_preparation_is_the_observed_bottleneck():
    rows = list(csv.DictReader((RESULTS / "component_timing_summary.csv").open()))
    selected = {
        row["method"]: row
        for row in rows
        if row["task"] == "classification"
    }
    for method in ("shared_context_6", "specific_context_6"):
        assert float(selected[method]["median_context_and_fusion_seconds"]) > float(
            selected[method]["median_backbone_seconds"]
        )


def test_propensity_microbenchmark_is_complete():
    rows = list(csv.DictReader((RESULTS / "propensity_timing.csv").open()))
    assert len(rows) == 3 * 7
    assert {row["shift"] for row in rows} == {
        "covariate", "label", "concept", "selection", "domain", "temporal", "support"
    }
    assert all(math.isfinite(float(row["seconds"])) and float(row["seconds"]) > 0 for row in rows)
