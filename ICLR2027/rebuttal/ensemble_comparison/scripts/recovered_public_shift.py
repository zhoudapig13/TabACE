from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np

@dataclass(frozen=True)
class PublicShiftSpec:
    dataset: str
    kind: str
    rho: float
    beta: float

def parse_public_shift_name(name: str) -> PublicShiftSpec | None:
    match = re.match(
        r"^publicshift_(.+)_(shortcut|boundary|local|aucband|mixed)_rho(\d+)_beta(\d+)$",
        name,
    )
    if not match:
        return None
    return PublicShiftSpec(
        dataset=match.group(1),
        kind=match.group(2),
        rho=float(int(match.group(3)) / 100.0),
        beta=float(int(match.group(4)) / 100.0),
    )

def _public_shift_feature_scores(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    X_float = np.asarray(X, dtype=np.float64)
    if X_float.ndim != 2 or X_float.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    centered = X_float - np.mean(X_float, axis=0, keepdims=True)
    x_std = np.std(centered, axis=0) + 1e-12
    scores = np.zeros(X_float.shape[1], dtype=np.float64)
    for cls in classes:
        indicator = (y == cls).astype(np.float64)
        y_std = float(np.std(indicator))
        if y_std <= 1e-12:
            continue
        corr = np.abs(np.mean(centered * (indicator[:, None] - np.mean(indicator)), axis=0) / (x_std * y_std))
        scores = np.maximum(scores, corr)
    return scores

def _public_shift_class_means(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    X_float = np.asarray(X, dtype=np.float64)
    global_mean = np.mean(X_float, axis=0)
    return np.vstack(
        [
            np.mean(X_float[y == cls], axis=0) if np.any(y == cls) else global_mean
            for cls in classes
        ]
    )

def _public_shift_select_mask(
    y: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    priority: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros(len(y), dtype=bool)
    rho = float(np.clip(rho, 0.0, 1.0))
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        take = int(round(rho * len(idx)))
        if take <= 0:
            continue
        if priority is None:
            chosen = rng.choice(idx, size=min(take, len(idx)), replace=False)
        else:
            order = idx[np.argsort(priority[idx])[::-1]]
            chosen = order[: min(take, len(order))]
        mask[chosen] = True
    return mask

def _public_shift_opposite_positions(
    y: np.ndarray,
    classes: np.ndarray,
    class_means: np.ndarray,
    feature_idx: np.ndarray,
) -> np.ndarray:
    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    positions = np.asarray([class_to_pos[label] for label in y], dtype=int)
    if len(classes) == 2:
        return 1 - positions
    opposite = np.zeros_like(positions)
    for row, pos in enumerate(positions):
        distances = np.sum((class_means[:, feature_idx] - class_means[pos, feature_idx]) ** 2, axis=1)
        distances[pos] = -np.inf
        opposite[row] = int(np.argmax(distances))
    return opposite

def _public_shift_move_toward_opposite(
    X_mod: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    classes: np.ndarray,
    class_means: np.ndarray,
    feature_idx: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    noise_scale: float = 0.02,
) -> None:
    if not np.any(mask) or len(feature_idx) == 0:
        return
    class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
    own_pos = np.asarray([class_to_pos[label] for label in y], dtype=int)
    opposite_pos = _public_shift_opposite_positions(y, classes, class_means, feature_idx)
    for row in np.flatnonzero(mask):
        delta = class_means[opposite_pos[row], feature_idx] - class_means[own_pos[row], feature_idx]
        X_mod[row, feature_idx] += float(strength) * delta
        if noise_scale > 0.0:
            feature_std = np.std(X_mod[:, feature_idx], axis=0) + 1e-6
            X_mod[row, feature_idx] += rng.normal(0.0, noise_scale, size=len(feature_idx)) * feature_std

def apply_public_shift_to_split(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    shift_spec = parse_public_shift_name(dataset_name)
    if shift_spec is None:
        return X_test, {}
    rng = np.random.default_rng(seed + 94091)
    X_train_float = np.asarray(X_train, dtype=np.float64)
    X_orig = np.asarray(X_test, dtype=np.float64)
    X_mod = X_orig.copy()
    classes = np.unique(y_train)
    class_means = _public_shift_class_means(X_train_float, y_train, classes)
    scores = _public_shift_feature_scores(X_train_float, y_train, classes)
    n_features = X_train_float.shape[1]
    order = np.argsort(scores)[::-1]
    top1 = order[: max(1, min(1, n_features))]
    top2 = order[: max(1, min(2, n_features))]
    top3 = order[: max(1, min(3, n_features))]
    low_half = np.argsort(scores)[: max(1, n_features // 2)]

    train_mean = np.mean(X_train_float, axis=0)
    train_std = np.std(X_train_float, axis=0) + 1e-6
    X_test_std = (X_orig - train_mean) / train_std
    if len(classes) == 2:
        direction = (class_means[1] - class_means[0]) / train_std
        midpoint = ((class_means[1] + class_means[0]) * 0.5 - train_mean) / train_std
        score = np.sum((X_test_std - midpoint) * direction, axis=1)
        signed = np.where(y_test == classes[1], score, -score)
        boundary_priority = -signed
    else:
        boundary_priority = -np.max(np.abs(X_test_std[:, top3]), axis=1)

    base_mask = _public_shift_select_mask(y_test, shift_spec.rho, rng)
    touched = np.zeros(len(y_test), dtype=bool)
    touched_features: set[int] = set()

    def apply_kind(kind: str, mask: np.ndarray) -> None:
        nonlocal touched
        if not np.any(mask):
            return
        beta = shift_spec.beta
        if kind == "shortcut":
            features = top2
            _public_shift_move_toward_opposite(
                X_mod, y_test, mask, classes, class_means, features, 0.60 + 0.35 * beta, rng
            )
        elif kind == "boundary":
            corrupt = top3
            support = low_half[: max(1, min(2, len(low_half)))]
            _public_shift_move_toward_opposite(
                X_mod, y_test, mask, classes, class_means, corrupt, 0.45 + 0.28 * beta, rng
            )
            class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
            global_mean = np.mean(X_train_float, axis=0)
            for row in np.flatnonzero(mask):
                own = class_to_pos.get(y_test[row], 0)
                X_mod[row, support] += (0.22 + 0.10 * beta) * (
                    class_means[own, support] - global_mean[support]
                )
            features = np.unique(np.concatenate([corrupt, support]))
        elif kind == "aucband":
            features = top1
            _public_shift_move_toward_opposite(
                X_mod, y_test, mask, classes, class_means, features, 0.30 + 0.18 * beta, rng
            )
        elif kind == "local":
            corrupt = top2
            support = low_half[: max(1, min(3, len(low_half)))]
            _public_shift_move_toward_opposite(
                X_mod, y_test, mask, classes, class_means, corrupt, 0.55 + 0.30 * beta, rng
            )
            class_to_pos = {cls: pos for pos, cls in enumerate(classes)}
            global_mean = np.mean(X_train_float, axis=0)
            for row in np.flatnonzero(mask):
                own = class_to_pos.get(y_test[row], 0)
                X_mod[row, support] += (0.30 + 0.15 * beta) * (
                    class_means[own, support] - global_mean[support]
                )
            features = np.unique(np.concatenate([corrupt, support]))
        else:
            raise ValueError(f"Unknown public shift kind {kind!r}")
        touched |= mask
        touched_features.update(int(idx) for idx in features)

    if shift_spec.kind == "mixed":
        idx = np.flatnonzero(base_mask)
        rng.shuffle(idx)
        chunks = np.array_split(idx, 4)
        for kind, chunk in zip(["shortcut", "boundary", "local", "aucband"], chunks):
            mask = np.zeros(len(y_test), dtype=bool)
            mask[chunk] = True
            apply_kind(kind, mask)
    elif shift_spec.kind == "boundary":
        boundary_mask = _public_shift_select_mask(y_test, shift_spec.rho, rng, priority=boundary_priority)
        apply_kind("boundary", boundary_mask)
    else:
        apply_kind(shift_spec.kind, base_mask)

    edit = X_mod - X_orig
    feature_shift_after = np.abs((np.mean(X_mod, axis=0) - train_mean) / train_std)
    feature_shift_before = np.abs((np.mean(X_orig, axis=0) - train_mean) / train_std)
    details = {
        "public_shift_source_dataset": shift_spec.dataset,
        "public_shift_kind": shift_spec.kind,
        "public_shift_rho": shift_spec.rho,
        "public_shift_beta": shift_spec.beta,
        "public_shift_actual_fraction": float(np.mean(touched)) if len(touched) else 0.0,
        "public_shift_changed_feature_count": len(touched_features),
        "public_shift_changed_features": "|".join(str(idx) for idx in sorted(touched_features)),
        "public_shift_feature_shift_mean_before": float(np.mean(feature_shift_before)),
        "public_shift_feature_shift_max_before": float(np.max(feature_shift_before)),
        "public_shift_feature_shift_mean_after": float(np.mean(feature_shift_after)),
        "public_shift_feature_shift_max_after": float(np.max(feature_shift_after)),
        "public_shift_edit_l2_mean": float(np.mean(np.linalg.norm(edit, axis=1))),
        "public_shift_edit_linf_mean": float(np.mean(np.max(np.abs(edit), axis=1))),
        "public_shift_edit_l2_touched_mean": (
            float(np.mean(np.linalg.norm(edit[touched], axis=1))) if np.any(touched) else 0.0
        ),
    }
    return X_mod.astype(np.float32), details
