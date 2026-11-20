"""Compatibility helpers for evaluating fixed 10-class LimiX checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationPrediction:
    probabilities: np.ndarray
    classes: np.ndarray
    strategy: str
    model_calls: int


def predict_proba_native_or_ovr(
    predictor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    max_native_classes: int = 10,
) -> ClassificationPrediction:
    """Use native multiclass inference up to the checkpoint limit, then OvR.

    The OvR path is a transparent compatibility wrapper: each original class
    becomes one binary LimiX task, and the positive probabilities are row-wise
    normalized to form a valid multiclass distribution.  It does not alter the
    checkpoint or fit parameters on target labels.
    """

    classes = np.unique(np.asarray(y_train))
    if len(classes) < 2:
        raise ValueError("classification requires at least two training classes")

    if len(classes) <= max_native_classes:
        probabilities = np.asarray(
            predictor.predict(
                x_train, y_train, x_test, task_type="Classification"
            ),
            dtype=np.float64,
        )
        model_classes = np.asarray(predictor.classes)
        positions = {label: index for index, label in enumerate(model_classes.tolist())}
        probabilities = probabilities[:, [positions[label] for label in classes.tolist()]]
        probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1.0e-12, None)
        return ClassificationPrediction(probabilities, classes, "native", 1)

    positive_probabilities = []
    for label in classes:
        binary_target = (np.asarray(y_train) == label).astype(np.int64)
        binary_probability = np.asarray(
            predictor.predict(
                x_train,
                binary_target,
                x_test,
                task_type="Classification",
            ),
            dtype=np.float64,
        )
        binary_classes = np.asarray(predictor.classes)
        positive_position = int(np.flatnonzero(binary_classes == 1)[0])
        positive_probabilities.append(binary_probability[:, positive_position])

    probabilities = np.column_stack(positive_probabilities)
    probabilities = np.clip(probabilities, 1.0e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return ClassificationPrediction(
        probabilities,
        classes,
        "one_vs_rest",
        len(classes),
    )
