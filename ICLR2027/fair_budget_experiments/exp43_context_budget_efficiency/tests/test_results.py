#!/usr/bin/env python3
"""Validate the completed Context-budget timing artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_results() -> None:
    detail = _rows("timing_detail.csv")
    summary = _rows("timing_summary.csv")
    metadata = json.loads((RESULTS / "metadata.json").read_text(encoding="utf-8"))
    repeats = int(metadata["repeats_per_cell"])
    assert len(detail) == 2 * 3 * 7 * repeats * 6
    assert len(summary) == 2 * 6
    assert {row["task"] for row in detail} == {"classification", "regression"}
    assert {int(row["k"]) for row in detail} == set(range(6))
    assert all(float(row["total_seconds"]) > 0.0 for row in detail)
    assert all(float(row["backbone_seconds"]) > 0.0 for row in detail)
    assert all(float(row["context_and_fusion_seconds"]) >= 0.0 for row in detail)
    assert all(
        int(row["observed_backbone_calls"])
        == (1 if int(row["k"]) == 0 else int(row["k"]))
        for row in detail
    )
    assert metadata["cuda_synchronized"] is True
    assert metadata["k_counts_nonbase_context_predictions"] is True


if __name__ == "__main__":
    test_results()
