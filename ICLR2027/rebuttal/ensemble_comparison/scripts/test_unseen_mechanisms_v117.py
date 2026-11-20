import inspect

import numpy as np
import pytest

import unseen_mechanisms_v117 as protocol


@pytest.fixture
def example():
    rng = np.random.default_rng(71)
    X = np.c_[rng.normal(size=(128, 4)), rng.integers(0, 3, size=128)].astype(np.float32)
    Q = np.c_[rng.normal(size=(64, 4)), rng.integers(0, 3, size=64)].astype(np.float32)
    y = rng.integers(0, 2, size=128)
    return X, y, Q, np.arange(4)


def test_no_target_labels_interface():
    assert 'y_target' not in inspect.signature(protocol.generate).parameters


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('family', list(protocol.FAMILIES))
def test_clean_identity_determinism_and_label_independence(example, task, family):
    X, y, Q, numeric = example
    a, b, c, _ = protocol.generate(X, y, Q, numeric, task, family, 0)
    assert np.array_equal(a, X) and np.array_equal(b, y) and np.array_equal(c, Q)
    first = protocol.generate(X, y, Q, numeric, task, family, 80)
    second = protocol.generate(X, y[::-1], Q, numeric, task, family, 80)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[2], second[2])
    repeated = protocol.generate(X, y, Q, numeric, task, family, 80)
    assert all(np.array_equal(first[i], repeated[i]) for i in range(3))
    assert np.isfinite(first[0]).all() and np.isfinite(first[2]).all()
    assert np.array_equal(first[0][:, -1].astype(int), first[0][:, -1])
    if family != 'interaction_sampling':
        assert np.array_equal(first[1], y)
    else:
        rows = first[3]['selected_source_positions']
        assert np.array_equal(first[0], X[rows]) and np.array_equal(first[1], y[rows])
    if protocol.FAMILIES[family]['group'] == 'perturbation':
        assert np.array_equal(first[2], Q)
    if family in ('nonlinear_sensor', 'cross_feature_rotation'):
        if task == 'classification':
            assert np.array_equal(first[0], X)
        else:
            assert np.array_equal(first[2], Q)
    if family not in ('column_desynchronization', 'interaction_sampling'):
        assert np.array_equal(first[0][:, -1], X[:, -1])
        assert np.array_equal(first[2][:, -1], Q[:, -1])


def test_column_marginals_really_preserved(example):
    X, y, Q, numeric = example
    for s in protocol.STRENGTHS:
        transformed = protocol.generate(X, y, Q, numeric, 'classification', 'column_desynchronization', s)[0]
        assert np.array_equal(np.sort(transformed, axis=0), np.sort(X, axis=0))


def test_rotation_preserves_standardized_pair_energy(example):
    X, y, Q, numeric = example
    _, _, observed, info = protocol.generate(X, y, Q, numeric, 'classification', 'cross_feature_rotation', 100)
    center, scale = protocol.source_statistics(X)
    old = (Q-center)/scale
    new = (observed-center)/scale
    for a, b in info['rotation_pairs']:
        assert np.allclose(old[:, a]**2+old[:, b]**2, new[:, a]**2+new[:, b]**2, rtol=1e-5, atol=1e-5)


def test_categorical_only_is_explicitly_ineligible_for_numeric_maps():
    X = np.tile(np.arange(4), (12, 1)).astype(np.float32)
    X[:, 0] = np.arange(12)%3
    assert not protocol.applicable(X, [], 'cell_heavy_tail')[0]
    assert not protocol.applicable(X, [], 'cross_feature_rotation')[0]
