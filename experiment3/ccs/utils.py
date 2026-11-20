from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


def load_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def environment_info() -> dict[str, object]:
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    repo_src = Path(__file__).resolve().parents[2] / "src"
    if repo_src.exists() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    for module_name in ["numpy", "yaml", "matplotlib", "torch", "tabicl", "tabpfn", "sklearn"]:
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "available")
        except Exception as exc:
            info[module_name] = f"unavailable: {exc.__class__.__name__}"
    return info


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std
