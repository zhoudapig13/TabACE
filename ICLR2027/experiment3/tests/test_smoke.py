from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccs.experiment import ExperimentConfig, run_experiment
from ccs.datasets import SyntheticSCMConfig


def test_synthetic_smoke(tmp_path: Path) -> None:
    config = ExperimentConfig(
        output_dir=tmp_path,
        random_states=(0,),
        context_sizes=(8,),
        retrievers=("random", "overlap_aware"),
        context_estimators=("local_context_t_learner",),
        classical_estimators=("global_t_learner",),
        synthetic=SyntheticSCMConfig(n_samples=80, n_features=6, overlap_violation=0.5, random_state=0),
        confounding_strengths=(1.0,),
        treatment_imbalances=(0.0,),
        overlap_violations=(0.5,),
        max_test_eval=12,
    )
    rows = run_experiment(config)
    assert len(rows) == 3
    assert (tmp_path / "synthetic_results.csv").exists()
    assert {row["retriever"] for row in rows} == {"global_all_data", "random", "overlap_aware"}
    assert all("pehe" in row and "ate_error" in row for row in rows)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_synthetic_smoke(Path(tmp))
    print("[experiment3] smoke test passed")
