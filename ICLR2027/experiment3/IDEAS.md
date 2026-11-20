# Experiment 3 Ideas

Goal: improve the unchanged TabICL baseline with small, measurable candidate
changes before promoting any method into the main code path.

## Starting hypotheses

- Robust preprocessing may help noisy or heavy-tailed tabular datasets.
- Quantile-normalized marginals may make feature scales more compatible across
  heterogeneous datasets.
- Simple row-level statistics may expose distribution-shape cues that TabICL can
  use without changing model internals.
- Candidate methods should first beat `baseline` on smoke datasets, then show
  stable gains across several random seeds.

## Candidate backlog

- Add a train-split-only transformer runner for stricter leakage-free screening.
- Add per-task candidate defaults for classification vs regression.
- Add guarded method selection based on internal validation, using accuracy/AUC
  protection before optimizing secondary metrics.
- Promote only methods with positive mean delta and limited worst-case drop.
