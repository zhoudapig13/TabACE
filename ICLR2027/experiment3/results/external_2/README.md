# External Dataset 2: ACIC 2016

This directory is the objective-level `external_2` result directory. It mirrors
the ACIC 2016 condition 1 initial validation stored under
`results/acic2016_initial/`.

## Dataset

- Source: causallib public copy of ACIC 2016 challenge data.
- Files used: `experiment3/data/acic2016/x.csv` and
  `experiment3/data/acic2016/zymu_1.csv`.
- Ground truth: `zymu_1.csv` includes `mu0` and `mu1`, so PEHE is valid.

## Command

```bash
python experiment3/scripts/run_acic2016.py \
  --output-dir experiment3/results/acic2016_initial \
  --conditions 1 \
  --split-seeds 0,1,2 \
  --context-sizes 16,64,256 \
  --retrievers random,knn,matched_pair,matched_pair_diverse,overlap_aware \
  --classical-estimators global_t_learner,global_dr_learner \
  --max-samples 800 \
  --max-test-eval 64
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/acic2016_initial/acic2016_results.csv \
  --output-dir experiment3/results/acic2016_initial
python experiment3/scripts/diagnose_retrieval.py \
  --input experiment3/results/acic2016_initial/acic2016_results.csv \
  --output-dir experiment3/results/acic2016_initial \
  --bootstrap 1000
```

## Key Result

- `matched_pair_diverse` vs random: PEHE delta `-1.6153`, CI
  `[-3.2573, +0.2554]`, 88.89% wins.
- `matched_pair_diverse` vs kNN: PEHE delta `-0.9744`, CI
  `[-2.4402, +0.5998]`, 66.67% wins.
- `matched_pair` vs random: PEHE delta `-1.9291`, CI
  `[-2.9483, -0.9297]`, 88.89% wins.
- `matched_pair` vs kNN: PEHE delta `-1.2883`, CI
  `[-2.7564, -0.1937]`, 88.89% wins.

## Interpretation

This second external dataset gives partial support for causal-aware matched
context retrieval. It does not prove that `matched_pair_diverse` is uniformly
best, because the PEHE intervals for `matched_pair_diverse` cross zero and the
simpler `matched_pair` is stronger in this initial ACIC run.
