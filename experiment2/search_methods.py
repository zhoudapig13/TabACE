from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

sys.path.insert(0, str(ROOT / "experiments"))

from context_selection_search import load_datasets, make_split, method_specs, run_method  # noqa: E402
from openml_broad_sweep import EXTRA_OPENML_SPECS, specs_from_arg as broad_specs_from_arg  # noqa: E402
from openml_method_search import OpenMLSpec, load_openml_dataset, parse_specs  # noqa: E402
from openml_propensity_variant_search import EXPANDED_OPENML_SPECS, variant_specs  # noqa: E402
from utils import environment_info, select_device, write_json, write_rows  # noqa: E402


WEAK_OPENML_SPECS = [
    OpenMLSpec(23, "openml_cmc", "Contraceptive method choice; weak TabICL baseline in prior runs"),
    OpenMLSpec(1480, "openml_ilpd", "Indian liver patient; imbalance and low balanced accuracy"),
    OpenMLSpec(31, "openml_credit_g", "Credit risk; mixed categorical/numeric"),
    OpenMLSpec(1464, "openml_blood_transfusion", "Blood donation; imbalanced binary"),
    OpenMLSpec(469, "openml_dmft", "Dental caries DMFT; difficult multiclass"),
    OpenMLSpec(1067, "openml_kc1", "Software defect; noisy binary"),
]

CC18_DATA_IDS = [
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50,
    54, 151, 182, 188, 38, 307, 300, 458, 469, 554, 1049, 1050, 1053, 1063,
    1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501, 1480, 1485, 1486,
    1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381,
    40499, 40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996,
    41027, 23517, 40923, 40927, 40978, 40670, 40701,
]


DEFAULT_METHODS = [
    "baseline_n1",
    "causal_scaled",
    "causal_kbest",
    "propensity_rank_feature",
    "propensity_interactions",
    "propensity_causal_kbest",
    "quantile_features",
    "quantile_domain_shift_feature",
    "domain_shift_feature",
    "domain_causal_interactions",
    "rare_class_oversample",
    "rare_boundary_domain_shift",
    "outlier_winsor_combo",
    "linear_probe_feature",
    "hard_probe_combo",
    "knn_label_feature",
    "local_density_feature",
    "centroid_distance_feature",
]


def method_lookup():
    methods = {}
    for method in [*method_specs(), *variant_specs()]:
        methods[method.name] = method
    return methods


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def cc18_specs(limit: int | None = None) -> list[OpenMLSpec]:
    ids = CC18_DATA_IDS
    try:
        with urllib.request.urlopen("https://www.openml.org/api/v1/json/study/99", timeout=20) as response:
            payload = json.load(response)
        ids = [int(item) for item in payload["study"]["data"]["data_id"]]
    except Exception:
        ids = CC18_DATA_IDS
    if limit is not None:
        ids = ids[:limit]
    return [
        OpenMLSpec(data_id, f"openml_cc18_{data_id}", "OpenML-CC18 curated classification benchmark suite")
        for data_id in ids
    ]


def resolve_openml_specs(raw: str) -> list[OpenMLSpec]:
    if raw == "none":
        return []
    if raw == "weak":
        return WEAK_OPENML_SPECS
    if raw == "cc18":
        return cc18_specs()
    if raw == "cc18_small":
        return cc18_specs(limit=16)
    if raw == "expanded":
        return EXPANDED_OPENML_SPECS
    if raw == "extra":
        return EXTRA_OPENML_SPECS
    if raw in {"broad", "wide"}:
        return broad_specs_from_arg(raw)
    return parse_specs(raw)


def load_all_datasets(args: argparse.Namespace):
    datasets = []
    loaded = []
    failed = []

    if args.sklearn:
        for dataset in load_datasets(args.synthetic_samples, args.random_states[0]):
            if dataset.name in set(parse_csv(args.sklearn)):
                datasets.append(dataset)
                loaded.append(dataset.name)

    openml_specs = resolve_openml_specs(args.openml)
    if args.max_datasets > 0:
        openml_specs = openml_specs[: args.max_datasets]

    for spec in openml_specs:
        print(f"[experiment2] Loading OpenML data_id={spec.data_id} alias={spec.alias}...")
        try:
            dataset = load_openml_dataset(spec, max_samples=args.max_samples, random_state=args.random_states[0])
        except Exception as exc:
            failed.append(f"{spec.alias}: {exc!r}")
            continue
        datasets.append(dataset)
        loaded.append(dataset.name)

    return datasets, loaded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen many TabICL improvement ideas across datasets and metrics.")
    parser.add_argument("--openml", default="weak", help="none, weak, cc18, cc18_small, expanded, extra, broad, wide, default, or id:alias list")
    parser.add_argument(
        "--sklearn",
        default="breast_cancer,wine,digits,synth_selection_hard_overlap,synth_rare_boundary",
        help="Comma-separated local datasets from context_selection_search.py, or empty string.",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--random-states", dest="random_states_raw", default="42")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--max-datasets", type=int, default=0, help="Optional cap on loaded OpenML datasets; 0 means no cap.")
    parser.add_argument("--synthetic-samples", type=int, default=600)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="search_results.csv")
    parser.add_argument("--environment-output", default="search_environment.json")
    args = parser.parse_args()
    args.random_states = [int(part.strip()) for part in args.random_states_raw.split(",") if part.strip()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    methods = method_lookup()
    method_names = parse_csv(args.methods)
    selected_methods = []
    for name in method_names:
        if name not in methods:
            raise ValueError(f"Unknown method '{name}'. Available: {', '.join(sorted(methods))}")
        selected_methods.append(methods[name])
    if "baseline_n1" not in {method.name for method in selected_methods}:
        selected_methods.insert(0, methods["baseline_n1"])

    datasets, loaded, failed = load_all_datasets(args)
    rows = []
    for random_state in args.random_states:
        for dataset in datasets:
            print(f"[experiment2] Dataset={dataset.name} seed={random_state}")
            try:
                split = make_split(dataset, test_size=args.test_size, random_state=random_state)
            except Exception as exc:
                failed.append(f"{dataset.name}@{random_state}: split_failed {exc!r}")
                continue
            for method in selected_methods:
                print(f"[experiment2]   method={method.name}")
                try:
                    row = run_method(split, dataset, method, device=device, random_state=random_state)
                except Exception as exc:
                    row = {
                        "dataset": dataset.name,
                        "source": dataset.source,
                        "method": method.name,
                        "description": method.description,
                        "metric_name": "accuracy",
                        "metric_value": "",
                        "accuracy": "",
                        "balanced_accuracy": "",
                        "macro_f1": "",
                        "weighted_f1": "",
                        "roc_auc": "",
                        "n_samples": int(dataset.X.shape[0]),
                        "n_features_original": int(dataset.X.shape[1]),
                        "random_state": random_state,
                        "device": device,
                        "error": repr(exc),
                    }
                row["max_samples"] = int(args.max_samples)
                rows.append(row)

    output_path = RESULTS_DIR / args.output
    write_rows(output_path, rows)
    write_json(
        RESULTS_DIR / args.environment_output,
        {
            **environment_info(device),
            "loaded_datasets": loaded,
            "failed_datasets": failed,
            "methods": [method.name for method in selected_methods],
            "random_states": args.random_states,
            "max_samples": args.max_samples,
            "max_datasets": args.max_datasets,
            "synthetic_samples": args.synthetic_samples,
        },
    )
    print(f"[experiment2] Wrote {output_path}")
    if failed:
        print("[experiment2] Failed datasets/items:")
        for item in failed:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
