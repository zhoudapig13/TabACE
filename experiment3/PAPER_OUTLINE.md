# Paper Outline

## Title

Causal-Aware Context Selection for Tabular In-Context CATE Estimation

## Abstract Sketch

Tabular in-context learning systems depend on which examples are placed in the
context. We test whether causal structure should guide context retrieval for
conditional treatment effect estimation. Across controlled synthetic regimes,
diverse treated-control matching substantially improves PEHE over random and
kNN retrieval. Scaled IHDP and ACIC 2016 checks support matched causal context
retrieval as a family, while showing that the simpler matched-pair retriever is
more robust than the diversity-augmented variant outside the synthetic setting.

## Introduction

- Tabular ICL models reuse context examples rather than fitting a conventional
  dataset-specific model.
- For CATE, naive random/kNN contexts can omit one treatment arm or create poor
  balance.
- The paper asks whether context retrieval itself should be causal-aware.

## Method

- Keep the estimator fixed: a local context T-learner.
- Compare retrieval strategies under matched test conditions.
- Promote treated-control matching and diversity only when paired PEHE deltas
  support them.

## Experiments

- Extended synthetic SCM: main controlled validation.
- IHDP scaled: 50 realizations, full 75-unit test split.
- ACIC 2016 initial: condition 1 with 3 split seeds.
- Tiny real TabICL smoke: backend availability only.

## Main Results

- Synthetic: `matched_pair_diverse` strongly beats random/kNN and original
  `matched_pair`.
- IHDP scaled: `matched_pair` strongly beats random/kNN; `matched_pair_diverse`
  is not robust and loses to `matched_pair`.
- ACIC 2016 initial: `matched_pair` has significant initial gains;
  `matched_pair_diverse` trends positive against random/kNN but is underpowered
  and not better than `matched_pair`.

## Discussion

- The core contribution is retrieval-level: causal matching improves context
  quality.
- The best synthetic retriever is not necessarily the best external retriever.
- The negative result for `overlap_aware` suggests overlap without locality is
  insufficient.

## Limitations

- Main evidence uses a local fallback estimator, not final TabICL/TabPFN model
  adaptation.
- ACIC coverage is initial, not exhaustive.
- ATE and policy metrics are secondary and less stable than PEHE.
