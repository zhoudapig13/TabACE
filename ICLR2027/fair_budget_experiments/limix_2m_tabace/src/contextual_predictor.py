"""Batched per-query Context inference for the official LimiX predictor.

The public :class:`LimiXPredictor` accepts one training Context for an entire
target batch.  TabACE instead supplies one source-row index set per query.  The
LimiX backbone already supports a batch of independent in-context sequences;
this adapter exposes that capability while preserving the official
preprocessing ensemble, class shuffles, temperature, and frozen checkpoint.

No target labels are accepted by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import random
from typing import Any, Literal

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


Task = Literal["classification", "regression"]


def _as_context_matrix(
    value: np.ndarray, *, n_target: int, n_source: int, name: str
) -> np.ndarray:
    indices = np.asarray(value, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != n_target:
        raise ValueError(
            f"Context {name!r} must have shape [n_target, k], got {indices.shape}"
        )
    if indices.shape[1] < 2:
        raise ValueError(f"Context {name!r} must contain at least two rows")
    if np.any(indices < 0) or np.any(indices >= n_source):
        raise IndexError(f"Context {name!r} contains an out-of-range source index")
    return indices


def _classification_permutations(
    *, seed: int, n_estimators: int, n_classes: int
) -> list[np.ndarray]:
    """Reproduce ``LimiXPredictor._predict_cls`` class permutations exactly."""

    random.seed(seed)
    rng = np.random.default_rng(seed)
    noise = rng.random((n_estimators * 3, n_classes))
    unique = np.unique(np.argsort(noise, axis=1), axis=0)
    balanced = n_estimators // len(unique)
    output = [row for row in unique for _ in range(balanced)]
    remainder = n_estimators % len(unique)
    if remainder:
        output.extend(unique[index] for index in rng.choice(len(unique), size=remainder))
    return output


def _preprocess(
    predictor: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_pipeline: np.ndarray,
    pipeline: list[Any],
    pipeline_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    X = np.concatenate([X_train, X_test], axis=0)
    X = predictor.convert_x_dtypes(X)
    X = predictor.convert_category2num(X)
    X = np.asarray(X, dtype=np.float32)
    categorical = predictor.get_categorical_features_indices(X)
    for step_index, step in enumerate(pipeline):
        class_name = type(step).__name__
        if class_name in {"InferenceAttentionMap", "SubSampleData"}:
            raise ValueError(
                "Batched explicit Context inference requires a no-retrieval "
                f"LimiX configuration, found {class_name}"
            )
        X, categorical = step.fit_transform(
            X,
            categorical,
            predictor.seeds[pipeline_index * predictor.preprocess_num + step_index],
            y=y_pipeline,
        )
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=predictor.device)
    y_tensor = torch.as_tensor(
        y_pipeline, dtype=torch.float32, device=predictor.device
    )
    return X_tensor, y_tensor


def _model_output(
    predictor: Any,
    X: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    *,
    n_source: int,
    task: Task,
    batch_size: int,
    query_indices: np.ndarray | None = None,
) -> torch.Tensor:
    model = predictor.model.to(predictor.device)
    chunks: list[torch.Tensor] = []
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=predictor.device)
    target = X[n_source:]
    if query_indices is None:
        query_indices = np.arange(len(target), dtype=np.int64)
    query_tensor = torch.as_tensor(
        query_indices, dtype=torch.long, device=predictor.device
    )
    if len(query_tensor) != len(index_tensor):
        raise ValueError("query_indices and Context rows must have the same length")
    context_length = int(index_tensor.shape[1])
    for start in range(0, len(index_tensor), batch_size):
        stop = min(start + batch_size, len(index_tensor))
        batch_indices = index_tensor[start:stop]
        X_context = X[:n_source][batch_indices]
        y_context = y[batch_indices]
        X_query = target[query_tensor[start:stop]].unsqueeze(1)
        X_batch = torch.cat([X_context, X_query], dim=1)
        with (
            torch.autocast(
                device_type=predictor.device.type,
                enabled=bool(predictor.mix_precision),
            ),
            torch.inference_mode(),
        ):
            output = model(
                x=X_batch,
                y=y_context,
                eval_pos=context_length,
                task_type="cls" if task == "classification" else "reg",
            )
        if isinstance(output, dict):
            output = output["cls_output" if task == "classification" else "reg_output"]
        raw_shape = tuple(output.shape)
        expected = stop - start
        if output.ndim == 3 and output.shape[0] == expected and output.shape[1] == 1:
            output = output[:, 0, :]
        elif output.ndim == 3 and output.shape[0] == 1 and output.shape[1] == expected:
            # The regression decoder in LimiX v1.1 keeps sequence first for a
            # batched one-query call, whereas the classification decoder keeps
            # batch first.  Normalize both layouts to [batch, outputs].
            output = output[0, :, :]
        elif output.ndim == 3:
            raise RuntimeError(
                f"unexpected LimiX batched output shape {tuple(output.shape)} "
                f"for batch size {expected}"
            )
        elif output.ndim == 1:
            output = output[:, None]
        if output.shape[0] != expected:
            raise RuntimeError(
                f"LimiX normalized output shape {tuple(output.shape)} from raw "
                f"shape {raw_shape}, expected leading batch {expected}"
            )
        chunks.append(output.float().detach().cpu())
    return torch.cat(chunks, dim=0)


def predict_context_bank_native(
    predictor: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    contexts: Mapping[str, np.ndarray],
    *,
    task: Task,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Predict a named bank of fixed-width per-query Contexts.

    Classification labels may contain at most ten classes, matching native
    LimiX.  Use :func:`predict_context_bank_classification` for automatic OvR.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must be compatible 2D matrices")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different row counts")
    checked = {
        str(name): _as_context_matrix(
            value, n_target=len(X_test), n_source=len(X_train), name=str(name)
        )
        for name, value in contexts.items()
    }
    if not checked:
        return {}

    predictor.validate_data(
        X_train,
        y_train,
        reset=True,
        validate_separately=False,
        accept_sparse=False,
        dtype=None,
        ensure_all_finite=False,
    )
    predictor.validate_data(
        X_test,
        reset=True,
        validate_separately=False,
        accept_sparse=False,
        dtype=None,
        ensure_all_finite=False,
    )

    if task == "classification":
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y_train)
        n_classes = len(encoder.classes_)
        if not 2 <= n_classes <= 10:
            raise ValueError(f"native LimiX classification requires 2..10 classes, got {n_classes}")
        permutations = _classification_permutations(
            seed=int(predictor.seed),
            n_estimators=int(predictor.n_estimators),
            n_classes=n_classes,
        )
    else:
        encoded = np.asarray(y_train, dtype=np.float64)
        n_classes = 0
        permutations = []

    collected: dict[str, list[np.ndarray]] = {name: [] for name in checked}
    torch.manual_seed(int(predictor.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(predictor.seed))

    for pipeline_index, pipeline in enumerate(predictor.preprocess_pipelines):
        if task == "classification":
            permutation = permutations[pipeline_index]
            y_pipeline = permutation[np.asarray(encoded, dtype=np.int64)]
        else:
            permutation = None
            y_pipeline = encoded
        X_tensor, y_tensor = _preprocess(
            predictor,
            X_train,
            X_test,
            np.asarray(y_pipeline),
            pipeline,
            pipeline_index,
        )
        grouped: dict[int, list[tuple[str, np.ndarray]]] = {}
        for name, indices in checked.items():
            grouped.setdefault(int(indices.shape[1]), []).append((name, indices))
        for group in grouped.values():
            flat_indices = np.concatenate([indices for _, indices in group], axis=0)
            query_indices = np.tile(np.arange(len(X_test), dtype=np.int64), len(group))
            output = _model_output(
                predictor,
                X_tensor,
                y_tensor,
                flat_indices,
                n_source=len(X_train),
                task=task,
                batch_size=batch_size,
                query_indices=query_indices,
            )
            if os.environ.get("LIMIX_CONTEXT_DEBUG") == "1":
                print(
                    "context_group",
                    [(name, tuple(indices.shape)) for name, indices in group],
                    "output",
                    tuple(output.shape),
                    flush=True,
                )
            for member_index, (name, _) in enumerate(group):
                member_output = output[
                    member_index * len(X_test) : (member_index + 1) * len(X_test)
                ]
                if task == "classification":
                    logits = member_output[:, :n_classes]
                    if float(predictor.softmax_temperature) != 1.0:
                        logits = logits / float(predictor.softmax_temperature)
                    logits = logits[..., permutation]
                    value = torch.softmax(logits, dim=1).numpy()
                else:
                    value = member_output.numpy().reshape(-1)
                collected[name].append(np.asarray(value, dtype=np.float64))

    return {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in collected.items()
    }


def predict_context_bank_classification(
    predictor: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    contexts: Mapping[str, np.ndarray],
    *,
    max_native_classes: int = 10,
    batch_size: int = 32,
) -> tuple[dict[str, np.ndarray], np.ndarray, str, int]:
    """Native multiclass inference with deterministic OvR above ten classes."""

    classes = np.unique(np.asarray(y_train))
    if len(classes) <= max_native_classes:
        return (
            predict_context_bank_native(
                predictor,
                X_train,
                y_train,
                X_test,
                contexts,
                task="classification",
                batch_size=batch_size,
            ),
            classes,
            "native",
            1,
        )

    positive_by_member: dict[str, list[np.ndarray]] = {
        str(name): [] for name in contexts
    }
    for label in classes:
        binary = (np.asarray(y_train) == label).astype(np.int64)
        bank = predict_context_bank_native(
            predictor,
            X_train,
            binary,
            X_test,
            contexts,
            task="classification",
            batch_size=batch_size,
        )
        for name, probability in bank.items():
            positive_by_member[name].append(probability[:, 1])
    output: dict[str, np.ndarray] = {}
    for name, columns in positive_by_member.items():
        probability = np.column_stack(columns)
        probability = np.clip(probability, 1.0e-12, None)
        output[name] = probability / probability.sum(axis=1, keepdims=True)
    return output, classes, "one_vs_rest", int(len(classes))
