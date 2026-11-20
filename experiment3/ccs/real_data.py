from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from .types import CausalDataset, CausalSplit


SUPPORTED_DATASETS = ("ihdp", "acic", "lalonde", "criteo_uplift")


def load_real_dataset(name: str, data_dir: str | Path) -> CausalDataset:
    """Placeholder interface for real causal/uplift benchmarks.

    The synthetic SCM path is intentionally the first complete benchmark. Real
    datasets differ in licenses, download flows, and potential-outcome
    availability, so this loader fixes the expected return type without hiding
    data acquisition behind brittle side effects.
    """
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown real dataset '{name}'. Supported placeholders: {', '.join(SUPPORTED_DATASETS)}")
    raise NotImplementedError(
        f"{normalized} loader is a placeholder. Put processed arrays under {Path(data_dir)} and implement "
        "conversion to CausalDataset with X, treatment, outcome, and available ground-truth effects."
    )


def load_ihdp_split(data_dir: str | Path, realization: int = 0) -> CausalSplit:
    data_dir = Path(data_dir)
    train_path = _find_first(
        data_dir,
        [
            "ihdp_npci_1-100.train.npz",
            "ihdp_npci_1-1000.train.npz",
            "train.npz",
            "ihdp_train.npz",
        ],
    )
    test_path = _find_first(
        data_dir,
        [
            "ihdp_npci_1-100.test.npz",
            "ihdp_npci_1-1000.test.npz",
            "test.npz",
            "ihdp_test.npz",
        ],
    )
    if train_path and test_path:
        train = _load_ihdp_npz(train_path, realization=realization, split_name="train")
        test = _load_ihdp_npz(test_path, realization=realization, split_name="test")
        return CausalSplit(train=train, test=test, random_state=realization)

    csv_path = _find_first(data_dir, ["ihdp.csv", "ihdp_data.csv", "processed.csv"])
    if csv_path:
        dataset = _load_ihdp_csv(csv_path)
        rng = np.random.default_rng(realization)
        indices = rng.permutation(dataset.n_samples)
        n_test = max(1, int(round(dataset.n_samples * 0.3)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return CausalSplit(train=_subset(dataset, train_idx, "train"), test=_subset(dataset, test_idx, "test"), random_state=realization)

    raise FileNotFoundError(
        f"No IHDP files found in {data_dir}. Expected ihdp_npci_1-100.train.npz and "
        "ihdp_npci_1-100.test.npz, or a processed CSV with treatment/outcome columns."
    )


def load_acic2016_split(
    data_dir: str | Path,
    condition: int = 1,
    test_size: float = 0.25,
    random_state: int = 0,
    max_samples: int = 0,
) -> CausalSplit:
    dataset = load_acic2016_dataset(data_dir, condition=condition)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(dataset.n_samples)
    if max_samples > 0 and max_samples < indices.size:
        indices = indices[:max_samples]
    n_test = max(1, int(round(indices.size * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return CausalSplit(
        train=_subset(dataset, train_idx, "train"),
        test=_subset(dataset, test_idx, "test"),
        random_state=random_state,
    )


def load_acic2016_dataset(data_dir: str | Path, condition: int = 1) -> CausalDataset:
    data_dir = Path(data_dir)
    x_path = data_dir / "x.csv"
    zymu_path = data_dir / f"zymu_{condition}.csv"
    if not x_path.exists() or not zymu_path.exists():
        raise FileNotFoundError(
            f"Missing ACIC 2016 files in {data_dir}. Expected x.csv and zymu_{condition}.csv."
        )

    with x_path.open(newline="", encoding="utf-8") as f:
        x_rows = list(csv.DictReader(f))
    with zymu_path.open(newline="", encoding="utf-8") as f:
        zymu_rows = list(csv.DictReader(f))
    if len(x_rows) != len(zymu_rows):
        raise ValueError(f"ACIC row count mismatch: {len(x_rows)} covariate rows vs {len(zymu_rows)} outcome rows")

    X = _encode_mixed_rows(x_rows)
    treatment = np.asarray([float(row["z"]) for row in zymu_rows], dtype=float)
    y0 = np.asarray([float(row["y0"]) for row in zymu_rows], dtype=float)
    y1 = np.asarray([float(row["y1"]) for row in zymu_rows], dtype=float)
    mu0 = np.asarray([float(row["mu0"]) for row in zymu_rows], dtype=float)
    mu1 = np.asarray([float(row["mu1"]) for row in zymu_rows], dtype=float)
    outcome = np.where(treatment > 0.5, y1, y0)
    return CausalDataset(
        name=f"acic2016_condition_{condition}",
        X=X,
        treatment=treatment,
        outcome=outcome,
        mu0=mu0,
        mu1=mu1,
        tau=mu1 - mu0,
        propensity=_estimate_propensity(X, treatment),
    )


def _encode_mixed_rows(rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot encode empty row set")
    columns = list(rows[0].keys())
    encoded_columns = []
    for column in columns:
        values = [row[column] for row in rows]
        parsed = [_parse_float(value) for value in values]
        if all(value is not None for value in parsed):
            encoded_columns.append(np.asarray(parsed, dtype=float)[:, None])
            continue
        categories = sorted(set(values))
        encoded = np.zeros((len(rows), len(categories)), dtype=float)
        lookup = {category: idx for idx, category in enumerate(categories)}
        for row_idx, value in enumerate(values):
            encoded[row_idx, lookup[value]] = 1.0
        encoded_columns.append(encoded)
    return np.hstack(encoded_columns).astype(float)


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _find_first(data_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


def _select_realization(array: np.ndarray, realization: int, *, is_feature: bool = False) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3:
        idx = realization % array.shape[-1]
        return array[:, :, idx]
    if is_feature and array.ndim == 2:
        return array
    if array.ndim == 2 and array.shape[1] > 1 and array.shape[0] != array.shape[1]:
        idx = realization % array.shape[1]
        return array[:, idx]
    return array.reshape(-1) if array.ndim > 1 and 1 in array.shape else array


def _get_npz_key(payload, candidates: list[str]) -> np.ndarray:
    for key in candidates:
        if key in payload:
            return payload[key]
    raise KeyError(f"None of these IHDP keys were found: {', '.join(candidates)}. Available: {', '.join(payload.files)}")


def _load_ihdp_npz(path: Path, realization: int, split_name: str) -> CausalDataset:
    with np.load(path) as payload:
        X = _select_realization(_get_npz_key(payload, ["x", "X", "features"]), realization, is_feature=True)
        treatment = _select_realization(_get_npz_key(payload, ["t", "treatment", "w"]), realization).astype(float).reshape(-1)
        outcome = _select_realization(_get_npz_key(payload, ["yf", "y_factual", "outcome", "y"]), realization).astype(float).reshape(-1)
        mu0 = _select_realization(_get_npz_key(payload, ["mu0", "m0"]), realization).astype(float).reshape(-1)
        mu1 = _select_realization(_get_npz_key(payload, ["mu1", "m1"]), realization).astype(float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"IHDP features must be 2D after realization selection, got shape={X.shape} from {path}")
    propensity = _estimate_propensity(X, treatment)
    return CausalDataset(
        name=f"ihdp_{split_name}_realization_{realization}",
        X=X,
        treatment=treatment,
        outcome=outcome,
        mu0=mu0,
        mu1=mu1,
        tau=mu1 - mu0,
        propensity=propensity,
    )


def _load_ihdp_csv(path: Path) -> CausalDataset:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    fields = rows[0].keys()
    treatment_key = _first_key(fields, ["treatment", "t", "w"])
    outcome_key = _first_key(fields, ["y_factual", "yf", "outcome", "y"])
    mu0_key = _first_key(fields, ["mu0", "m0"])
    mu1_key = _first_key(fields, ["mu1", "m1"])
    feature_keys = [key for key in fields if key not in {treatment_key, outcome_key, mu0_key, mu1_key, "y_cfactual", "ycf"}]
    X = np.asarray([[float(row[key]) for key in feature_keys] for row in rows], dtype=float)
    treatment = np.asarray([float(row[treatment_key]) for row in rows], dtype=float)
    outcome = np.asarray([float(row[outcome_key]) for row in rows], dtype=float)
    mu0 = np.asarray([float(row[mu0_key]) for row in rows], dtype=float)
    mu1 = np.asarray([float(row[mu1_key]) for row in rows], dtype=float)
    return CausalDataset(
        name="ihdp_csv",
        X=X,
        treatment=treatment,
        outcome=outcome,
        mu0=mu0,
        mu1=mu1,
        tau=mu1 - mu0,
        propensity=_estimate_propensity(X, treatment),
    )


def _first_key(fields, candidates: list[str]) -> str:
    lowered = {field.lower(): field for field in fields}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"Could not find any of {candidates}")


def _subset(dataset: CausalDataset, indices: np.ndarray, split_name: str) -> CausalDataset:
    return CausalDataset(
        name=f"{dataset.name}_{split_name}",
        X=dataset.X[indices],
        treatment=dataset.treatment[indices],
        outcome=dataset.outcome[indices],
        mu0=dataset.mu0[indices],
        mu1=dataset.mu1[indices],
        tau=dataset.tau[indices],
        propensity=dataset.propensity[indices],
    )


def _estimate_propensity(X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    alpha = 1e-2
    reg = alpha * np.eye(X_aug.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(X_aug.T @ X_aug + reg) @ X_aug.T @ treatment
    logits = X_aug @ coef
    logits = (logits - np.mean(logits)) / (np.std(logits) + 1e-8)
    p = 1.0 / (1.0 + np.exp(-logits))
    base = float(np.mean(treatment))
    p = 0.5 * p + 0.5 * base
    return np.clip(p, 0.02, 0.98)
