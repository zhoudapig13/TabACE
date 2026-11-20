#!/usr/bin/env python3
"""Quick aggregate diagnostic for partially captured seven-shift banks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    args = parser.parse_args()
    rows: dict[tuple[str, float], list[tuple[float, float]]] = {}
    datasets = 0
    for path in sorted((args.banks / args.task).glob("dataset_*.npz")):
        datasets += 1
        with np.load(path, allow_pickle=False) as payload:
            names = ("base", *payload["member_names"].astype(str).tolist())
            base = np.asarray(payload["base"], dtype=np.float64)
            members = np.asarray(payload["members"], dtype=np.float64)
            target = np.asarray(payload["y_target"])
            strength = payload["strength"].astype(float)
            scale = payload["metric_scale"].astype(float)
        for candidate_index, name in enumerate(names):
            raw = base if name == "base" else members[..., candidate_index - 1]
            for alpha in ((0.0,) if name == "base" else (0.1, 0.25, 0.5, 0.75, 1.0)):
                prediction = base if name == "base" else (1.0 - alpha) * base + alpha * raw
                for index in np.flatnonzero(strength > 0):
                    if args.task == "classification":
                        probability = np.clip(prediction[index], 1.0e-12, None)
                        probability /= probability.sum(axis=1, keepdims=True)
                        first = float(np.mean(probability.argmax(axis=1) == target[index]))
                        second = float(roc_auc_score(target[index], probability[:, 1]))
                    else:
                        first = float(r2_score(target[index], prediction[index]))
                        second = float(np.sqrt(np.mean((target[index] - prediction[index]) ** 2)) / scale[index])
                    rows.setdefault((name, alpha), []).append((first, second))
    ranked = sorted(
        ((np.mean(value, axis=0)[0], np.mean(value, axis=0)[1], key) for key, value in rows.items()),
        reverse=True,
    )
    print(f"datasets={datasets}")
    base_primary, base_secondary = np.mean(rows[("base", 0.0)], axis=0)
    print(
        f"{'base':24s} alpha=0.00 primary={base_primary:.6f} "
        f"secondary={base_secondary:.6f}"
    )
    for first, second, (name, alpha) in ranked[:20]:
        print(f"{name:24s} alpha={alpha:.2f} primary={first:.6f} secondary={second:.6f}")


if __name__ == "__main__":
    main()
