# Method Description

## Problem

The experiment studies context selection for tabular in-context-style CATE
estimation. For a test unit `x`, a retriever selects a small training context.
A local T-learner then fits separate outcome models on treated and control
examples inside that context and estimates `tau_hat(x) = mu1_hat(x) -
mu0_hat(x)`.

This isolates retrieval quality from large-model training. The main evidence is
therefore retrieval-level evidence, not final TabICL/TabPFN model-level
performance.

## Baselines

- `random`: uniformly samples context rows.
- `knn`: selects nearest neighbors by standardized feature distance.
- `global_t_learner` / `global_dr_learner`: full-data classical reference
  estimators, not context-limited retrieval methods.

## Causal-Aware Retrievers

- `matched_pair`: selects treated-control pairs with similar propensity and
  feature neighborhoods around the query. This preserves both treatment-arm
  coverage and local similarity.
- `matched_pair_diverse`: starts from matched treated-control candidates and
  adds a diversity pressure to reduce redundant context rows. This is the
  strongest synthetic method.
- `overlap_aware`: filters toward common-support examples. It is retained as a
  negative control because overlap filtering alone often sacrifices locality.

## Metrics

- Primary: PEHE, lower is better.
- Secondary: ATE error, policy value, policy regret, uplift at top 30%.
- Diagnostics: context treatment rate, mean absolute SMD, propensity distance,
  feature distance, diversity, duplicate rate.

## Datasets

- Synthetic SCM: controlled regimes with known `mu0`, `mu1`, and CATE.
- IHDP: semi-synthetic external validation with `mu0/mu1`; scaled run uses 50
  realizations and the full 75-unit test split.
- ACIC 2016: condition 1 from causallib, with `mu0/mu1`; initial validation uses
  3 split seeds and a bounded sample for runtime.

## Backend Status

TabICL and TabPFN packages are importable in the current environment. A tiny
real `TabICLRegressor` smoke is included under `results/real_backend_smoke/`.
TabPFN model fitting requires one-time license acceptance and a `TABPFN_TOKEN`,
so only import-level availability is currently verified for TabPFN.
