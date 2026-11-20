#!/usr/bin/env python3
"""Run the audited five-fold TabACE selector on captured LimiX banks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_context_only_five_shift as context_only,
)


def _install_cc18_numeric_loader() -> None:
    """Use the LimiX OpenML encoder while preserving the canonical CC18 split."""

    import openml

    openml.config.set_root_cache_directory(
        Path("/root/autodl-tmp/limix_tabace/results/paper_run/openml_cache")
    )
    path = (
        ROOT
        / "fair_budget_experiments"
        / "limix_2m_ovr"
        / "src"
        / "protocol.py"
    )
    spec = importlib.util.spec_from_file_location("limix_five_oof_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    by_data_id = {
        int(value.data_id): value
        for value in module.suite_specs(module.CC18_SUITE_ID)
    }

    def compatible_loader(dataset, _protocol):
        split = module.load_cc18_split(by_data_id[int(dataset.data_id)])
        return split.X_source, split.X_target, split.y_source, split.y_target

    candidates.load_cc18_split = compatible_loader
from fair_budget_experiments.unified_context_ensemble.src import (  # noqa: E402
    build_five_shift_candidate as candidates,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("classification", "regression", "all"), default="all"
    )
    parser.add_argument("--force-candidates", action="store_true")
    return parser.parse_args()


def configure(results_root: Path) -> tuple[Path, Path, Path]:
    _install_cc18_numeric_loader()
    captured = results_root / "banks"
    source_candidate = results_root / "source_candidate"
    oof = results_root / "oof"
    candidates.RESULTS = source_candidate
    candidates.CLASS_BANKS = captured / "classification"
    candidates.REG_BANKS = captured / "regression"
    # These files are not used by the context-only builder, but retain valid
    # defaults for the optional legacy candidate entry point.
    candidates.CLASS_RAW = (
        ROOT
        / "experiment11/results/coce_supplement"
        / "no_shift_five_context_ablation_tabpfn_v3_v2/detail.csv"
    )
    candidates.REG_RAW = (
        ROOT
        / "experiment11/results/coce_supplement"
        / "regression_five_context_ablation_tabpfn_v3_v3/detail.csv"
    )
    return captured, source_candidate, oof


def build(
    task: str,
    *,
    captured: Path,
    source_candidate: Path,
    oof: Path,
    force_candidates: bool,
) -> dict[str, object]:
    expected = 72 if task == "classification" else 35
    found = len(tuple((captured / task).glob("dataset_*.npz")))
    if found != expected:
        raise RuntimeError(f"{task}: expected {expected} LimiX banks, found {found}")
    if task == "classification":
        candidates._construct_classification_banks(force=force_candidates)
    else:
        candidates._construct_regression_banks(force=force_candidates)
    audit = context_only.build(
        task,
        oof,
        backbone_display="LimiX",
        checkpoint="LimiX-2M.ckpt",
    )
    audit["source_prediction_banks"] = str(captured / task)
    audit["candidate_prediction_banks"] = str(source_candidate / "banks" / task)
    audit["source_prediction_bank_count"] = found
    audit["candidate_prediction_bank_count"] = len(
        tuple((source_candidate / "banks" / task).glob("dataset_*.npz"))
    )
    audit["paper_tier_mapping"] = {
        "LimiX": "base",
        "+ TabACE-shared": "shared",
        "+ TabACE-specific": "pro",
    }
    path = oof / task / "audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    args = arguments()
    results_root = args.results_root.resolve()
    captured, source_candidate, oof = configure(results_root)
    tasks = (
        ("classification", "regression") if args.task == "all" else (args.task,)
    )
    audits = {
        task: build(
            task,
            captured=captured,
            source_candidate=source_candidate,
            oof=oof,
            force_candidates=args.force_candidates,
        )
        for task in tasks
    }
    oof.mkdir(parents=True, exist_ok=True)
    (oof / "combined_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
