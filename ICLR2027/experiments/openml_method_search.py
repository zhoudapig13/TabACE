from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from context_selection_search import (
    ClassificationDataset,
    make_split,
    method_specs,
    run_method,
    winner_rows,
)
from utils import RESULTS_DIR, environment_info, select_device, write_json, write_rows


@dataclass(frozen=True)
class OpenMLSpec:
    data_id: int
    alias: str
    note: str


DEFAULT_OPENML_SPECS = [
    OpenMLSpec(31, "openml_credit_g", "Credit risk with mixed categorical/numeric features"),
    OpenMLSpec(37, "openml_diabetes_pima", "Pima diabetes binary medical classification"),
    OpenMLSpec(44, "openml_spambase", "Email spam detection with many continuous features"),
    OpenMLSpec(1464, "openml_blood_transfusion", "Small blood donation classification"),
    OpenMLSpec(1480, "openml_ilpd", "Indian liver patient disease classification"),
]


def parse_specs(raw: str) -> list[OpenMLSpec]:
    if raw == "default":
        return DEFAULT_OPENML_SPECS
    specs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            data_id, alias = part.split(":", 1)
        else:
            data_id, alias = part, f"openml_{part}"
        specs.append(OpenMLSpec(int(data_id), alias.strip(), "user supplied"))
    return specs


def as_numeric_matrix(X: np.ndarray) -> np.ndarray:
    X_obj = np.asarray(X, dtype=object)
    numeric = np.empty(X_obj.shape, dtype=np.float32)
    categorical_cols: list[int] = []
    for col in range(X_obj.shape[1]):
        values = X_obj[:, col]
        try:
            numeric[:, col] = values.astype(np.float32)
        except (TypeError, ValueError):
            categorical_cols.append(col)

    if categorical_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        numeric[:, categorical_cols] = enc.fit_transform(X_obj[:, categorical_cols]).astype(np.float32)
    numeric = SimpleImputer(strategy="median").fit_transform(numeric).astype(np.float32)
    return numeric


def load_openml_dataset(spec: OpenMLSpec, *, max_samples: int, random_state: int) -> ClassificationDataset:
    data = fetch_openml(data_id=spec.data_id, as_frame=False, parser="liac-arff")
    X = as_numeric_matrix(data.data)
    target = np.asarray(data.target)
    mask = target != None  # noqa: E711
    X = X[mask]
    target = target[mask]
    y = LabelEncoder().fit_transform(target)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{spec.alias} has fewer than two classes after preprocessing")
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        selected = []
        per_class = max(1, max_samples // len(np.unique(y)))
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            take = min(per_class, len(cls_idx))
            selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        remaining = max_samples - len(selected)
        if remaining > 0:
            used = np.zeros(X.shape[0], dtype=bool)
            used[selected] = True
            candidates = np.flatnonzero(~used)
            selected.extend(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False).tolist())
        selected = np.asarray(selected)
        rng.shuffle(selected)
        X = X[selected]
        y = y[selected]
    return ClassificationDataset(name=spec.alias, X=X, y=y, source=f"openml:{spec.data_id}")


def write_report(rows: list[dict[str, object]], winners: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# OpenML Method Search",
        "",
        "This run applies the selection-bias and causal-proxy TabICL method search to external OpenML datasets.",
        "",
        "## Positive Deltas",
        "",
    ]
    if winners:
        lines.extend(["| Dataset | Method | Accuracy | Baseline | Delta |", "| --- | --- | ---: | ---: | ---: |"])
        for row in winners[:30]:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {float(row['metric_value']):.4f} | "
                f"{float(row['baseline_metric_value']):.4f} | {float(row['delta_vs_baseline']):+.4f} |"
            )
    else:
        lines.append("_No positive deltas found._")

    lines.extend(["", "## Best Per Dataset", ""])
    lines.extend(["| Dataset | Best Method | Best Accuracy | Baseline |", "| --- | --- | ---: | ---: |"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("metric_value") not in {"", None}:
            grouped.setdefault(str(row["dataset"]), []).append(row)
    for dataset, group in sorted(grouped.items()):
        baseline = next(row for row in group if row["method"] == "baseline_n1")
        best = max(group, key=lambda row: float(row["metric_value"]))
        lines.append(
            f"| {dataset} | {best['method']} | {float(best['metric_value']):.4f} | "
            f"{float(baseline['metric_value']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- OpenML datasets are capped to a small stratified sample for fast iteration.",
            "- These are real external tabular datasets, so gains here are more important than synthetic-only gains.",
            "- Positive methods should be rechecked with more random seeds and larger samples before being treated as robust.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabICL method search on OpenML datasets.")
    parser.add_argument("--openml", default="default", help="default or comma list like 31:credit,44:spam")
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="openml_method_search_results.csv")
    parser.add_argument("--winners-output", default="openml_method_search_winners.csv")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = select_device(args.device)
    rows: list[dict[str, object]] = []
    datasets: list[ClassificationDataset] = []
    for spec in parse_specs(args.openml):
        print(f"Loading OpenML data_id={spec.data_id} alias={spec.alias}")
        try:
            dataset = load_openml_dataset(spec, max_samples=args.max_samples, random_state=args.random_state)
        except Exception as exc:
            rows.append(
                {
                    "dataset": spec.alias,
                    "source": f"openml:{spec.data_id}",
                    "method": "load_failed",
                    "metric_name": "accuracy",
                    "metric_value": "",
                    "error": repr(exc),
                    "random_state": args.random_state,
                    "device": device,
                }
            )
            print(f"  failed: {exc!r}")
            continue
        datasets.append(dataset)
        print(f"Dataset={dataset.name} n={dataset.X.shape[0]} p={dataset.X.shape[1]}")
        split = make_split(dataset, args.test_size, args.random_state)
        for method in method_specs():
            print(f"  method={method.name}")
            try:
                rows.append(run_method(split, dataset, method, device=device, random_state=args.random_state))
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset.name,
                        "source": dataset.source,
                        "method": method.name,
                        "description": method.description,
                        "metric_name": "accuracy",
                        "metric_value": "",
                        "error": repr(exc),
                        "n_samples": int(dataset.X.shape[0]),
                        "n_features_original": int(dataset.X.shape[1]),
                        "device": device,
                        "random_state": args.random_state,
                    }
                )
                print(f"    failed: {exc!r}")

    valid_rows = [row for row in rows if row.get("metric_value") not in {"", None} and row["method"] != "load_failed"]
    winners = winner_rows(valid_rows)
    write_rows(RESULTS_DIR / args.output, rows)
    if winners:
        write_rows(RESULTS_DIR / args.winners_output, winners)
    else:
        with (RESULTS_DIR / args.winners_output).open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["dataset", "method", "delta_vs_baseline"]).writeheader()
    write_report(valid_rows, winners, RESULTS_DIR / "openml_method_search.md")
    write_json(
        RESULTS_DIR / "openml_method_search_environment.json",
        {
            **environment_info(device),
            "max_samples": args.max_samples,
            "random_state": args.random_state,
            "openml_specs": [spec.__dict__ for spec in parse_specs(args.openml)],
            "loaded_datasets": [dataset.name for dataset in datasets],
        },
    )
    print(f"Wrote {RESULTS_DIR / args.output}")
    print(f"Wrote {RESULTS_DIR / args.winners_output}")
    print(f"Wrote {RESULTS_DIR / 'openml_method_search.md'}")


if __name__ == "__main__":
    main()
