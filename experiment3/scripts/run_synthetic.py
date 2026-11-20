from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ccs.datasets import SyntheticSCMConfig  # noqa: E402
from ccs.experiment import ExperimentConfig, run_experiment  # noqa: E402
from ccs.utils import load_yaml  # noqa: E402


def as_tuple(config: dict[str, object], key: str, default):
    value = config.get(key, default)
    return tuple(value if isinstance(value, list) else [value])


def build_config(path: Path, output_dir: str | None = None) -> ExperimentConfig:
    raw = load_yaml(path)
    synthetic_raw = raw.get("synthetic", {}) or {}
    ablations = raw.get("ablations", {}) or {}
    synthetic = SyntheticSCMConfig(**synthetic_raw)
    regimes = tuple(dict(item) for item in raw.get("regimes", []) or ())
    return ExperimentConfig(
        output_dir=Path(output_dir or raw.get("output_dir", ROOT / "results" / "synthetic_default")),
        random_states=tuple(int(x) for x in as_tuple(raw, "random_states", [42])),
        test_size=float(raw.get("test_size", 0.3)),
        context_sizes=tuple(int(x) for x in as_tuple(raw, "context_sizes", [16, 32, 64, 128, 256])),
        retrievers=tuple(str(x) for x in as_tuple(raw, "retrievers", [])),
        context_estimators=tuple(str(x) for x in as_tuple(raw, "context_estimators", ["local_context_t_learner"])),
        classical_estimators=tuple(str(x) for x in as_tuple(raw, "classical_estimators", ["global_t_learner", "global_s_learner"])),
        synthetic=synthetic,
        confounding_strengths=tuple(float(x) for x in as_tuple(ablations, "confounding_strengths", [synthetic.confounding_strength])),
        treatment_imbalances=tuple(float(x) for x in as_tuple(ablations, "treatment_imbalances", [synthetic.treatment_imbalance])),
        overlap_violations=tuple(float(x) for x in as_tuple(ablations, "overlap_violations", [synthetic.overlap_violation])),
        max_test_eval=int(raw.get("max_test_eval", 120)),
        regimes=regimes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic SCM experiments for causal context selection.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Use the small smoke config.")
    args = parser.parse_args()

    config_path = ROOT / "configs" / "smoke.yaml" if args.smoke else Path(args.config)
    config = build_config(config_path, output_dir=args.output_dir)
    rows = run_experiment(config)
    print(f"[experiment3] wrote {len(rows)} rows to {config.output_dir / 'synthetic_results.csv'}")


if __name__ == "__main__":
    main()
