# Experiment 1: TabICL Reproduction

This directory is a clean reproduction lane for the original TabICL/TabICLv2
algorithm in this repository. It does not change `src/tabicl`; it only calls the
official scikit-learn style estimators and writes isolated outputs under
`experiment1/results/`.

## Quick run

```bash
python experiment1/run_reproduction.py \
  --datasets breast_cancer,wine,diabetes,synthetic_classification,synthetic_regression \
  --n-estimators 1 \
  --device auto
```

The first real TabICL run may download pretrained checkpoints unless local model
paths are provided.

## Full pipeline reproduction

Use `full_pipeline.py` when you want to reproduce the whole official-style
process: synthetic prior generation, pre-training, checkpoint handoff between
stages, and downstream evaluation.

Start with a dry run to inspect commands:

```bash
python experiment1/full_pipeline.py --preset smoke --dry-run
```

Then run a smoke pipeline:

```bash
python experiment1/full_pipeline.py --preset smoke --device cuda
```

Available presets:

- `smoke`: verifies the entire pipeline with 2 prior batches and 2 training
  steps.
- `tiny`: 100-step local reproduction for quick debugging.
- `small`: 1000-step reduced reproduction with `mix_scm`.
- `stage_like`: 5000-step single-stage run using the official architecture.
- `official`: three-stage official curriculum: Stage1 `max_seq_len=1024`,
  Stage2 `1000-40000`, Stage3 `40000-60000` with column/row freezing.

The `official` preset is intentionally large. It follows the official script
settings and is meant for a machine with enough GPU memory, time, and disk.

## Outputs

- `experiment1/results/reproduction_results.csv`: dataset metrics, runtime, and
  memory.
- `experiment1/results/reproduction_environment.json`: Python, Torch, CUDA, and
  TabICL version metadata.
- `experiment1/runs/<run-name>/manifest.json`: full-pipeline configuration.
- `experiment1/runs/<run-name>/*_prior/`: generated prior batches.
- `experiment1/runs/<run-name>/*_ckpts/`: pre-training checkpoints.
- `experiment1/runs/<run-name>/eval_reproduction_results.csv`: downstream
  dataset evaluation for the final checkpoint.
- `experiment1/runs/<run-name>/scale_reproduction_results.csv`: downstream
  evaluation across data sizes.

You can also run the scale evaluator directly:

```bash
python experiment1/run_scale_reproduction.py \
  --datasets synthetic_classification \
  --scales 100,300,1000,3000 \
  --classifier-model-path experiment1/runs/<run-name>/<stage>_ckpts/step-<n>.ckpt
```

Fine-tuning reproduction is available as a separate step:

```bash
python experiment1/run_finetune_reproduction.py \
  --dataset synthetic_classification \
  --n-samples 1000 \
  --epochs 3 \
  --classifier-model-path experiment1/runs/<run-name>/<stage>_ckpts/step-<n>.ckpt
```

## Recommended baseline protocol

Use the same values when comparing with later innovation experiments:

- Same dataset list.
- Same `--random-state`.
- Same `--test-size`.
- Same `--n-estimators`.
- Same checkpoint path, if using local checkpoints.

For classification checkpoints trained with `full_pipeline.py`, pass the final
checkpoint to evaluation via `--classifier-model-path`. Regression evaluation
continues to use the official downloaded regressor unless you provide a separate
regression checkpoint.
