# LimiX-2M + OvR benchmark adapter

This directory evaluates the official LimiX-2M checkpoint on the fixed
OpenML-CC18 / OpenML-CTR23 controlled-Context benchmark used by the paper.

The classification interface is deliberately named **LimiX-2M + OvR**:

- tasks with at most 10 classes use the checkpoint's native classifier;
- tasks with more than 10 classes use one binary LimiX call per class and
  row-normalize the positive-class probabilities;
- no target labels are used by either path.

The runner is resumable at the dataset-condition level. It writes one JSON
cell per completed condition before producing `detail.csv`, `summary_long.csv`,
and `audit.json`. The clean condition is evaluated once and repeated over the
five family names only in the summary, matching the table generator's
five-cell clean convention.

## Reproducible invocation

From the repository root, with the official LimiX `V1.1.0` checkout and
checkpoint already present:

```powershell
..\tmp\limix-venv\Scripts\python.exe `
  fair_budget_experiments\limix_2m_ovr\src\run_benchmark.py `
  --limix-root ..\tmp\LimiX-v1.1 `
  --model-path ..\tmp\LimiX-official\cache\LimiX-2M.ckpt `
  --task both --device cuda
```

Use `--dataset` for a comma-separated list of OpenML data IDs or dataset
names, and `--classification-pipelines 1 --regression-pipelines 1` for a fast
pipeline smoke test. Paper results must use all 4 classification and all 8
regression members.

On a CUDA machine the same command and completion audit are bundled in:

```powershell
fair_budget_experiments\limix_2m_ovr\scripts\run_full.ps1
```

Each output directory is locked by `run_signature.json`; changing the
checkpoint, device type, ensemble size, or high-dimensional policy requires a
new output directory, so partial runs with different settings cannot mix.

After the run, apply the fail-closed coverage audit:

```powershell
..\tmp\limix-venv\Scripts\python.exe `
  fair_budget_experiments\limix_2m_ovr\src\audit_five_completion.py `
  fair_budget_experiments\limix_2m_ovr\results\paper_run
```

The audit requires exactly 72 classification datasets and 35 regression
datasets, 21 unique cells per dataset, the official 4/8-member ensembles,
fixed target cohorts, five OvR datasets, correct affected-row counts, and no
failures. A partial run exits nonzero and cannot be mistaken for a paper row.

For feature counts above 1,024, the runner first performs the registered
perturbation in the original feature space and then applies a deterministic
source-fit TruncatedSVD (at most 256 components) before LimiX inference. This
path uses no target labels and is recorded per cell as
`truncated_svd_source_fit`. Disable it only for a separate compatibility probe
with `--high-dimensional-threshold 0`.

The suite-level dispatch manifest can be regenerated without running model
inference:

```powershell
..\tmp\limix-venv\Scripts\python.exe `
  fair_budget_experiments\limix_2m_ovr\src\audit_suite_compatibility.py `
  --output fair_budget_experiments\limix_2m_ovr\results\suite_audit
```

The seven-distribution-shift track is evaluated by `src/run_seven_shift.py`.
It imports the paper's frozen baseline materializer from `--protocol-root`
instead of implementing a nominally similar replacement.  The resulting
`run_signature.json` records SHA-256 hashes for the classification and
regression shift constructors.  The paper-completion gate requires all 35
binary classification datasets, all 33 regression datasets, seven families,
five strengths, the official 4/8-member ensembles, and zero failures.

```powershell
..\tmp\limix-venv\Scripts\python.exe `
  fair_budget_experiments\limix_2m_ovr\src\run_seven_shift.py `
  --limix-root ..\tmp\LimiX-v1.1 `
  --model-path ..\tmp\LimiX-official\cache\LimiX-2M.ckpt `
  --protocol-root . `
  --task both --device cuda `
  --classification-pipelines 4 --regression-pipelines 8 `
  --output fair_budget_experiments\limix_2m_ovr\results\paper_run_seven `
  --continue-on-error
```

High-dimensional seven-shift datasets use one clean-source-fit SVD per
dataset.  The projection is fitted before any condition is evaluated, reused
for every family and strength, and never receives target labels.
