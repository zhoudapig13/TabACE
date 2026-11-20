from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from protocol import FAMILIES, manipulate_source_context  # noqa: E402
from protocol import DatasetSpec, FixedSplit  # noqa: E402
from run_benchmark import lock_run_signature, model_features  # noqa: E402


def test_manipulations_do_not_mutate_query_or_source() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    y = np.tile(np.arange(4), 10)
    X_before = X.copy()
    y_before = y.copy()
    for family in FAMILIES:
        result = manipulate_source_context(
            "classification",
            X,
            y,
            family=family,
            rho=0.5,
            seed=1779,
            shift_scale=6.0,
        )
        assert result.X_context.shape == X.shape
        assert result.y_context.shape == y.shape
        assert len(result.affected_rows) == 20
        assert result.metadata["query_argument_present"] is False
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_zero_strength_is_identity() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = rng.normal(size=30)
    result = manipulate_source_context(
        "regression",
        X,
        y,
        family="original",
        rho=0.0,
        seed=1729,
        shift_scale=10.0,
    )
    np.testing.assert_array_equal(result.X_context, X)
    np.testing.assert_array_equal(result.y_context, y)
    assert len(result.affected_rows) == 0


def test_high_dimensional_adapter_is_source_fit_and_deterministic() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(24, 40)).astype(np.float32)
    target = rng.normal(size=(8, 40)).astype(np.float32)
    split = FixedSplit(
        spec=DatasetSpec(1, 1, "synthetic", "target"),
        X_source=source,
        X_target=target,
        y_source=np.tile([0, 1], 12),
        y_target=np.tile([0, 1], 4),
        source_indices=np.arange(24),
        target_indices=np.arange(24, 32),
        categorical_features=(),
    )
    first_source, first_target, first_metadata = model_features(
        split, source, threshold=16, components=10
    )
    second_source, second_target, second_metadata = model_features(
        split, source, threshold=16, components=10
    )
    assert first_source.shape == (24, 10)
    assert first_target.shape == (8, 10)
    assert first_metadata["feature_adapter"] == "truncated_svd_source_fit"
    assert first_metadata["feature_adapter_uses_target_labels"] is False
    np.testing.assert_allclose(first_source, second_source)
    np.testing.assert_allclose(first_target, second_target)
    assert first_metadata == second_metadata


def test_run_signature_rejects_mixed_configs(tmp_path: Path) -> None:
    signature = {"checkpoint": "abc", "pipelines": 4}
    lock_run_signature(tmp_path, signature)
    lock_run_signature(tmp_path, signature)
    with pytest.raises(RuntimeError, match="signature differs"):
        lock_run_signature(tmp_path, {"checkpoint": "abc", "pipelines": 1})
