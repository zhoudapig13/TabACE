import inspect
import numpy as np
from source_native_reliability_v191 import reliability


def test_reliability_uses_only_source_outcomes_and_source_predictions():
    assert list(inspect.signature(reliability).parameters) == ['y_source', 'prediction1', 'prediction8', 'task']
    y = np.linspace(-1, 1, 50)
    values, info = reliability(y, y+.3, y+.1, 'regression')
    assert np.all(values[:3] > 0) and values[3] > 0
    assert info['scored_source_rows'] == 50


def test_unscored_rare_source_rows_do_not_fabricate_validation_outcomes():
    y = np.array([0, 1, 0, 1])
    a = np.array([[.6, .4], [.4, .6], [.6, .4], [np.nan, np.nan]])
    b = np.array([[.8, .2], [.2, .8], [.8, .2], [np.nan, np.nan]])
    values, info = reliability(y, a, b, 'classification')
    assert np.isfinite(values).all()
    assert info['scored_source_rows'] == 3 and info['unscored_rare_source_rows'] == 1


def test_no_valid_holdout_is_explicit_neutral_evidence():
    y = np.array([0, 1])
    values, info = reliability(y, np.full((2, 2), np.nan), np.full((2, 2), np.nan), 'classification')
    np.testing.assert_array_equal(values, np.zeros(4))
    assert info['scored_source_rows'] == 0 and info['source_primary_Native8'] is None
