from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from experiment6.scripts.apa_tabicl import (
    APAConfig,
    APATabICLClassifier,
    CounterfactualContextConfig,
    NoShiftFeatureConfig,
    adaptive_posterior_correction,
    apa_predict_proba,
    baseline_predict_proba,
    classification_metrics,
    context_prior,
    counterfactual_context_predict_proba,
    correct_existing_proba_with_context_prior,
    prior_sensitivity_debias,
    select_balanced_knn_contexts,
    select_prior_aware_contexts,
)
from experiment6.scripts.run_apa_tabicl import (
    _local_support_p_values,
    context_consensus_calibrated_v42_predict,
    scale_aware_boundary_bypass_combo_predict,
    validation_member_selector_v42_fallback_predict,
)


def test_context_prior_uses_additive_smoothing() -> None:
    prior = context_prior(np.array([0, 0, 1]), n_classes=3, alpha=1.0)
    np.testing.assert_allclose(prior, np.array([3 / 6, 2 / 6, 1 / 6]))


def test_balanced_knn_contains_minority_when_available() -> None:
    X_train = np.array([[0.0], [0.1], [0.2], [5.0], [5.1]])
    y_train = np.array([0, 0, 0, 1, 1])
    contexts = select_balanced_knn_contexts(X_train, y_train, np.array([[0.05]]), k=4)
    selected_labels = y_train[contexts[0]]
    assert 1 in selected_labels


def test_prior_aware_context_improves_class_coverage() -> None:
    X_train = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [1.5]])
    y_train = np.array([0, 0, 0, 0, 0, 1])
    contexts = select_prior_aware_contexts(
        X_train,
        y_train,
        np.array([[0.05]]),
        k=4,
        candidate_multiplier=2,
        lambda_balance=2.0,
    )
    assert 1 in y_train[contexts[0]]


def test_adaptive_correction_normalizes_and_downweights_context_majority() -> None:
    proba = np.array([[0.45, 0.55]])
    p_eff = np.array([[0.85, 0.15]])
    adjusted, _, _, gamma = adaptive_posterior_correction(proba, p_eff, query_mode="single", fixed_gamma=1.0)
    np.testing.assert_allclose(adjusted.sum(axis=1), np.ones(1))
    assert adjusted[0, 1] > proba[0, 1]
    assert gamma[0] == 1.0


def test_apa_predict_proba_with_sklearn_estimator_smoke() -> None:
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-0.5, scale=0.4, size=(20, 2))
    X1 = rng.normal(loc=0.5, scale=0.4, size=(8, 2))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * len(X0) + [1] * len(X1))
    X_test = np.vstack([rng.normal(loc=-0.4, scale=0.4, size=(4, 2)), rng.normal(loc=0.6, scale=0.4, size=(4, 2))])

    config = APAConfig(context_strategy="prior_aware_knn", correction="adaptive", k=10, lambda_balance=0.5)
    result = apa_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
        config=config,
    )
    assert result.proba_adj.shape == (8, 2)
    np.testing.assert_allclose(result.proba_adj.sum(axis=1), np.ones(8))
    assert len(result.context_indices) == len(X_test)


def test_posthoc_full_context_correction_smoke() -> None:
    X_train = np.array([[-1.0], [-0.8], [-0.6], [0.7], [0.9], [1.1]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[-0.7], [0.8]])
    proba = np.array([[0.55, 0.45], [0.45, 0.55]])
    config = APAConfig(
        context_strategy="prior_aware_knn",
        correction="adaptive",
        k=4,
        lambda_balance=0.5,
        query_mode="hybrid",
        eta=0.5,
        gamma_max=0.8,
        max_baseline_margin=0.5,
        max_baseline_confidence=0.8,
    )
    result = correct_existing_proba_with_context_prior(proba, X_train, y_train, X_test, np.array([0, 1]), config)
    assert result.proba_adj.shape == proba.shape
    np.testing.assert_allclose(result.proba_adj.sum(axis=1), np.ones(2))
    assert np.all(result.gamma >= 0.0)


def test_binary_only_posthoc_gate_leaves_multiclass_unchanged() -> None:
    X_train = np.array([[-1.0], [-0.8], [0.0], [0.2], [1.0], [1.2]])
    y_train = np.array([0, 0, 1, 1, 2, 2])
    X_test = np.array([[-0.9], [0.1], [1.1]])
    proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    config = APAConfig(
        context_strategy="prior_aware_knn",
        correction="adaptive",
        k=4,
        binary_only=True,
        gamma_max=1.0,
    )
    result = correct_existing_proba_with_context_prior(proba, X_train, y_train, X_test, np.array([0, 1, 2]), config)
    np.testing.assert_allclose(result.proba_adj, proba)
    np.testing.assert_allclose(result.gamma, np.zeros(3))


def test_apa_wrapper_predict_proba_and_predict_smoke() -> None:
    rng = np.random.default_rng(7)
    X0 = rng.normal(loc=-1.0, scale=0.3, size=(12, 2))
    X1 = rng.normal(loc=1.0, scale=0.3, size=(12, 2))
    X_train = np.vstack([X0, X1])
    y_train = np.array(["a"] * len(X0) + ["b"] * len(X1))
    X_test = np.vstack([rng.normal(loc=-1.0, scale=0.3, size=(3, 2)), rng.normal(loc=1.0, scale=0.3, size=(3, 2))])

    clf = APATabICLClassifier(
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
        method="apa_v7",
        k=8,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    pred = clf.predict(X_test)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6))
    assert set(pred).issubset(set(clf.classes_))
    assert hasattr(clf, "last_branch_")


def test_prior_sensitivity_debias_returns_normalized_probabilities() -> None:
    context_priors = np.array(
        [
            [
                [0.80, 0.20],
                [0.50, 0.50],
                [0.20, 0.80],
            ]
        ]
    )
    context_probas = np.array(
        [
            [
                [0.70, 0.30],
                [0.55, 0.45],
                [0.35, 0.65],
            ]
        ]
    )
    debiased, sensitivity = prior_sensitivity_debias(context_probas, context_priors)
    assert debiased.shape == (1, 2)
    np.testing.assert_allclose(debiased.sum(axis=1), np.ones(1))
    assert sensitivity.shape == (1,)
    assert sensitivity[0] > 0.0


def test_classification_metrics_includes_roc_auc() -> None:
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[0.90, 0.10], [0.70, 0.30], [0.35, 0.65], [0.20, 0.80]])
    metrics = classification_metrics(y_true, proba, np.array([0, 1]))
    assert metrics["roc_auc"] == 1.0


def test_local_support_p_value_decreases_with_stronger_support() -> None:
    train_prior = np.array([0.75, 0.25])
    candidate_pred = np.array([1, 1])
    p_values = _local_support_p_values(
        local_candidate_support=np.array([0.25, 0.75]),
        candidate_pred=candidate_pred,
        train_prior=train_prior,
        k=16,
    )
    assert p_values[1] < p_values[0]


def test_counterfactual_context_predict_proba_smoke() -> None:
    rng = np.random.default_rng(11)
    X0 = rng.normal(loc=-0.7, scale=0.35, size=(18, 2))
    X1 = rng.normal(loc=0.7, scale=0.35, size=(18, 2))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * len(X0) + [1] * len(X1))
    X_test = np.vstack([rng.normal(loc=-0.6, scale=0.35, size=(3, 2)), rng.normal(loc=0.6, scale=0.35, size=(3, 2))])
    baseline = np.full((len(X_test), 2), 0.5)

    result = counterfactual_context_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
        config=CounterfactualContextConfig(k=8, aggregation="stability_debias", sensitivity_gate=0.0, disagreement_gate=0.0),
        baseline_proba=baseline,
        baseline_classes=np.array([0, 1]),
    )
    assert result.proba_adj.shape == (6, 2)
    np.testing.assert_allclose(result.proba_adj.sum(axis=1), np.ones(6))
    assert result.gamma.shape == (6,)
    assert result.delta.shape == (6,)


def test_scale_aware_boundary_bypass_combo_smoke() -> None:
    rng = np.random.default_rng(13)
    X0 = rng.normal(loc=-0.8, scale=0.35, size=(16, 2))
    X1 = rng.normal(loc=0.8, scale=0.35, size=(16, 2))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * len(X0) + [1] * len(X1))
    X_test = np.vstack([
        rng.normal(loc=-0.7, scale=0.35, size=(3, 2)),
        rng.normal(loc=0.7, scale=0.35, size=(3, 2)),
    ])
    baseline = baseline_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
    )
    result, details = scale_aware_boundary_bypass_combo_predict(
        X_train,
        y_train,
        X_test,
        baseline,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
        config=NoShiftFeatureConfig(k=8, max_changed_fraction=0.25),
        seed=0,
    )
    assert result.proba_adj.shape == baseline.proba_adj.shape
    np.testing.assert_allclose(result.proba_adj.sum(axis=1), np.ones(len(X_test)))
    assert details["scale_combo_branch"] == "small_scale_v27"


def test_context_consensus_calibrated_v42_smoke() -> None:
    rng = np.random.default_rng(17)
    X0 = rng.normal(loc=-0.8, scale=0.35, size=(16, 2))
    X1 = rng.normal(loc=0.8, scale=0.35, size=(16, 2))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * len(X0) + [1] * len(X1))
    X_test = np.vstack([
        rng.normal(loc=-0.7, scale=0.35, size=(3, 2)),
        rng.normal(loc=0.7, scale=0.35, size=(3, 2)),
    ])
    baseline = baseline_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
    )
    result, details = context_consensus_calibrated_v42_predict(
        X_train,
        y_train,
        X_test,
        baseline,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
        config=NoShiftFeatureConfig(k=8, max_changed_fraction=0.25),
        seed=0,
    )
    assert result.proba_adj.shape == baseline.proba_adj.shape
    np.testing.assert_allclose(result.proba_adj.sum(axis=1), np.ones(len(X_test)))
    assert "calibration_v44_applied" in details


def test_validation_member_selector_v42_fallback_smoke() -> None:
    rng = np.random.default_rng(19)
    X0 = rng.normal(loc=-0.8, scale=0.35, size=(20, 2))
    X1 = rng.normal(loc=0.8, scale=0.35, size=(20, 2))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * len(X0) + [1] * len(X1))
    X_test = np.vstack([
        rng.normal(loc=-0.7, scale=0.35, size=(3, 2)),
        rng.normal(loc=0.7, scale=0.35, size=(3, 2)),
    ])
    baseline = baseline_predict_proba(
        X_train,
        y_train,
        X_test,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
    )
    result, details = validation_member_selector_v42_fallback_predict(
        X_train,
        y_train,
        X_test,
        baseline,
        estimator_factory=lambda: KNeighborsClassifier(n_neighbors=3),
        config=NoShiftFeatureConfig(k=8, max_changed_fraction=0.25),
        seed=0,
    )
    assert result.proba_adj.shape == baseline.proba_adj.shape
    np.testing.assert_allclose(result.proba_adj.sum(axis=1), np.ones(len(X_test)))
    assert "validation_member_selected" in details
