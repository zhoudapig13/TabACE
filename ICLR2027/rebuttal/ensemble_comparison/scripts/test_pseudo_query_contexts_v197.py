import inspect
import numpy as np
from pseudo_query_contexts_v197 import contexts


def test_no_query_outcome_interface_and_no_own_pseudo_query_inclusion():
    assert list(inspect.signature(contexts).parameters) == ['X_source', 'y_source', 'X_query', 'predictions', 'task']
    rng = np.random.default_rng(197)
    X, Q = rng.normal(size=(64, 4)), rng.normal(size=(80, 4))
    y = np.arange(64) % 3
    P = rng.uniform(.1, 1., size=(6, 80, 3))
    plans, teacher = contexts(X, y, Q, P, 'classification')
    for branch in plans:
        assert sorted(np.concatenate([p['query_rows'] for p in branch])) == list(range(80))
        for p in branch:
            assert not set(p['pseudo_query_rows']) & set(p['query_rows'])
            assert len(p['X']) == len(p['y']) == 64
            np.testing.assert_array_equal(p['pseudo_labels'], teacher[p['pseudo_query_rows']].argmax(axis=1))
            assert set(y) <= set(p['y'])


def test_regression_pseudo_labels_are_bounded_and_inputs_unchanged():
    rng = np.random.default_rng(7)
    X, Q = rng.normal(size=(80, 2)), rng.normal(size=(60, 2))
    y = np.linspace(-1, 1, 80)
    P = rng.normal(scale=30., size=(6, 60))
    copies = [v.copy() for v in (X, y, Q, P)]
    plans, _ = contexts(X, y, Q, P, 'regression')
    for branch in plans:
        for p in branch:
            assert np.all((p['pseudo_labels'] >= -1) & (p['pseudo_labels'] <= 1))
    for actual, expected in zip((X, y, Q, P), copies):
        np.testing.assert_array_equal(actual, expected)


def test_singleton_source_classes_are_kept_without_fake_extra_labels():
    X, Q = np.arange(20).reshape(10, 2), np.arange(60).reshape(30, 2)
    y = np.arange(10)
    P = np.full((6, 30, 10), .1)
    plans, _ = contexts(X, y, Q, P, 'classification')
    for branch in plans:
        for p in branch:
            assert p['no_pseudo_fallback'] and len(p['pseudo_query_rows']) == 0
            np.testing.assert_array_equal(p['y'], y)
