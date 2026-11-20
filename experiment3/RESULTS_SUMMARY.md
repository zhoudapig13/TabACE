# Experiment 3 Results Summary

## Smoke Synthetic SCM

Command:

```bash
python experiment3/scripts/run_synthetic.py --smoke
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/smoke/synthetic_results.csv
python experiment3/tests/test_smoke.py
```

Artifacts:

- `experiment3/results/smoke/synthetic_results.csv`
- `experiment3/results/smoke/synthetic_summary.csv`
- `experiment3/results/smoke/synthetic_report.md`
- `experiment3/results/smoke/pehe_by_context_size.png`
- `experiment3/results/smoke/environment.json`

Status: smoke run completed and direct smoke test passed.

Notes:

- The smoke run compares 6 context retrievers at context sizes 16 and 32, plus
  global S/T learner baselines.
- The main experiment wrappers intentionally use the local ridge T-learner
  fallback so retrieval can be isolated. Backend availability is checked
  separately under `results/backend_check.json` and
  `results/real_backend_smoke/`.
- The smoke configuration is only a correctness check; use
  `configs/default.yaml` for the first meaningful ablation.

## Default Synthetic Ablation

Command:

```bash
python experiment3/scripts/run_synthetic.py \
  --config experiment3/configs/default.yaml
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/synthetic_default/synthetic_results.csv
```

Additional gain table:

- `experiment3/results/synthetic_default/retrieval_gain_vs_random_knn.csv`
- `experiment3/results/synthetic_default/retrieval_gain_report.md`

Key result:

- `matched_pair` is the clearest positive retrieval result: PEHE improves in
  58/60 runs vs random context with mean PEHE delta `-0.7258`, and in 55/60
  runs vs kNN context with mean PEHE delta `-0.5702`.
- `propensity_nearest` also improves PEHE on average: mean PEHE delta `-0.4026`
  vs random and `-0.2470` vs kNN, but policy value is weaker.
- `treatment_balanced_random` improves PEHE vs random in 48/60 runs, but ATE
  error is not consistently better.
- `overlap_aware` is not promoted yet; this first implementation is worse on
  average and needs redesign.

Important caveat: this run uses the local ridge fallback for the ICL wrapper.
The evidence supports a retrieval-level gain for context-limited ICL-style
estimation, not a final claim of TabICL model-level improvement.

## Multi-Regime Diagnostic Run

Command:

```bash
python experiment3/scripts/run_synthetic.py \
  --config experiment3/configs/diagnostic.yaml
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/synthetic_diagnostic/synthetic_results.csv
python experiment3/scripts/diagnose_retrieval.py \
  --input experiment3/results/synthetic_diagnostic/synthetic_results.csv
```

Design:

- 4 named regimes: `strong_confounding`, `poor_overlap`, `high_imbalance`,
  `nonlinear_heterogeneity`.
- 10 seeds per regime.
- Context sizes: 16, 32, 64, 128, 256.
- 3360 result rows, including original baselines, matched-pair variants, and
  full-data S/T/X/DR learner reference baselines.

Key result:

- `matched_pair_diverse` is the new best retrieval variant:
  - vs random: PEHE delta `-0.6304`, 162/200 wins, relative PEHE reduction
    about `+21.13%`, policy delta `+0.0328`.
  - vs kNN: PEHE delta `-0.6483`, 167/200 wins, relative PEHE reduction about
    `+23.60%`, ATE error delta `-0.2067`.
- Original `matched_pair` remains a simpler robust method:
  - vs random: PEHE delta `-0.3480`, 137/200 wins.
  - vs kNN: PEHE delta `-0.3660`, 150/200 wins, ATE error delta `-0.1410`.
- `overlap_aware` and `overlap_filtered_matched_pair` are negative results.
  Filtering to common support without preserving local similarity hurts PEHE.

Next best ablation:

- Promote `matched_pair_diverse`, original `matched_pair`, `matched_pair_propensity`,
  and `matched_pair_feature` into a smaller large-test evaluation.
- Drop plain `overlap_aware` from positive-method tables; keep it only as a
  negative/control result.

## Paper-Ready Extended Validation

Command:

```bash
python experiment3/scripts/run_synthetic.py \
  --config experiment3/configs/extended.yaml
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/synthetic_extended/synthetic_results.csv \
  --output-dir experiment3/results/synthetic_extended
python experiment3/scripts/diagnose_retrieval.py \
  --input experiment3/results/synthetic_extended/synthetic_results.csv \
  --output-dir experiment3/results/synthetic_extended
```

Design:

- 7 regimes, including weak/easy, strong confounding, poor overlap, extreme poor
  overlap, high imbalance, nonlinear heterogeneity, and high-noise/high-dim.
- 20 seeds per regime.
- Context sizes: 16, 32, 64, 128, 256.
- Focused methods only: random, kNN, treatment-balanced random,
  propensity-nearest, original matched pair, diverse matched pair, and
  overlap-aware negative control.

Key result:

- `matched_pair_diverse` vs random: PEHE delta `-0.6438`, 83.00% win rate,
  95% bootstrap CI `[-0.7082, -0.5766]`.
- `matched_pair_diverse` vs kNN: PEHE delta `-0.7429`, 84.14% win rate,
  95% bootstrap CI `[-0.8275, -0.6714]`.
- `matched_pair_diverse` vs original `matched_pair`: PEHE delta `-0.2337`,
  68.14% win rate, 95% bootstrap CI `[-0.2711, -0.1976]`.

Mechanism:

- Compared with random, diverse matching lowers SMD, lowers propensity distance,
  lowers feature distance, and slightly increases diversity.
- Compared with original matched pair, diverse matching lowers SMD while
  increasing feature diversity, supporting the balance/locality/diversity
  tradeoff explanation.

## IHDP Initial Validation

Command:

```bash
python experiment3/scripts/prepare_ihdp.py
python experiment3/scripts/run_ihdp.py --max-test-eval 64
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/ihdp_initial/ihdp_results.csv \
  --output-dir experiment3/results/ihdp_initial
python experiment3/scripts/diagnose_retrieval.py \
  --input experiment3/results/ihdp_initial/ihdp_results.csv \
  --output-dir experiment3/results/ihdp_initial
```

Status:

- IHDP files downloaded successfully under `experiment3/data/ihdp/`.
- 10 realizations were evaluated.
- `matched_pair_diverse` improves PEHE vs random by `-1.3152` with 80.00% wins.
- It also improves mean PEHE vs kNN and original matched pair, but those
  bootstrap intervals are wider and should be treated as preliminary external
  support rather than final external proof.

This 10-realization result has been superseded by the scaled IHDP validation
below.

## IHDP Scaled Validation

Command:

```bash
python experiment3/scripts/run_ihdp.py \
  --output-dir experiment3/results/ihdp_scaled \
  --realizations 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49 \
  --context-sizes 16,64,256 \
  --retrievers random,knn,matched_pair,matched_pair_diverse,overlap_aware \
  --classical-estimators global_t_learner,global_dr_learner \
  --max-test-eval 0
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/ihdp_scaled/ihdp_results.csv \
  --output-dir experiment3/results/ihdp_scaled
python experiment3/scripts/diagnose_retrieval.py \
  --input experiment3/results/ihdp_scaled/ihdp_results.csv \
  --output-dir experiment3/results/ihdp_scaled \
  --bootstrap 1000
```

Main artifacts:

- `experiment3/results/ihdp_scaled/ihdp_results.csv`
- `experiment3/results/ihdp_scaled/synthetic_summary.csv`
- `experiment3/results/ihdp_scaled/retrieval_gain_report.md`
- `experiment3/results/ihdp_scaled/diagnostic_report.md`
- `experiment3/results/ihdp_scaled/paired_bootstrap_tests.csv`
- `experiment3/results/ihdp_scaled/plots/`

Design:

- 50 IHDP realizations.
- Full test split per realization: 75 test units, not the old 64-unit subset.
- Context sizes: 16, 64, 256.
- 850 result rows.

Key result:

- `matched_pair_diverse` does not show a robust scaled-IHDP PEHE gain:
  - vs random: mean PEHE delta `-0.0524`, 66.00% wins, bootstrap CI
    `[-0.4368, +0.3776]`.
  - vs kNN: mean PEHE delta `+0.0052`, 59.33% wins, bootstrap CI
    `[-0.4635, +0.5648]`.
  - vs original `matched_pair`: mean PEHE delta `+0.4885`, bootstrap CI
    `[+0.2304, +0.7959]`, so the diverse variant is worse on this scaled IHDP
    setting.
- The simpler `matched_pair` is the robust IHDP external winner:
  - vs random: mean PEHE delta `-0.5409`, 71.33% wins, bootstrap CI
    `[-0.8427, -0.2742]`.
  - vs kNN: mean PEHE delta `-0.4833`, 68.00% wins, bootstrap CI
    `[-0.8308, -0.1599]`.

Interpretation: scaled IHDP supports causal-aware treated-control matching as a
retrieval family, but it is a negative/limiting result for promoting
`matched_pair_diverse` as uniformly best.

## ACIC 2016 Initial Validation

Command:

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

Main artifacts:

- `experiment3/results/acic2016_initial/acic2016_results.csv`
- `experiment3/results/acic2016_initial/retrieval_gain_report.md`
- `experiment3/results/acic2016_initial/diagnostic_report.md`
- `experiment3/results/external_2/README.md`
- `experiment3/results/external_2/external_2_results.csv`

Design:

- ACIC 2016 condition 1 from causallib (`x.csv` and `zymu_1.csv`).
- 3 train/test split seeds, 800 sampled units, 64 evaluated test units per
  split.
- 51 result rows.

Key result:

- `matched_pair_diverse` has better mean PEHE than random and kNN, but the
  PEHE intervals cross zero:
  - vs random: mean PEHE delta `-1.6153`, 88.89% wins, bootstrap CI
    `[-3.2573, +0.2554]`.
  - vs kNN: mean PEHE delta `-0.9744`, 66.67% wins, bootstrap CI
    `[-2.4402, +0.5998]`.
  - vs original `matched_pair`: mean PEHE delta `+0.3139`, bootstrap CI
    `[-0.4533, +1.3385]`.
- Original `matched_pair` is stronger in this small ACIC check:
  - vs random: mean PEHE delta `-1.9291`, bootstrap CI `[-2.9483, -0.9297]`.
  - vs kNN: mean PEHE delta `-1.2883`, bootstrap CI `[-2.7564, -0.1937]`.

Interpretation: ACIC gives partial external support for matched causal context
retrieval, but not enough to claim `matched_pair_diverse` is uniformly better
outside the synthetic benchmark.

## Real Backend Smoke

Command:

```bash
python experiment3/scripts/check_backends.py
TABPFN_DISABLE_TELEMETRY=1 python experiment3/scripts/run_real_backend_smoke.py \
  --max-test-eval 4 \
  --context-size 16 \
  --device cpu \
  --probe-tabpfn
```

Status:

- `backend_check.json`: TabICL import available = `true`; TabPFN import
  available = `true`, version `8.0.8`.
- Tiny real `TabICLRegressor` smoke completed on synthetic SCM and IHDP.
- Tiny TabPFN fit/predict probe is blocked by `TabPFNLicenseError` because
  model-weight download requires one-time license acceptance and
  `TABPFN_TOKEN` in this non-interactive environment.

Smoke result:

- Synthetic SCM tiny smoke: `matched_pair_diverse` PEHE `1.0607`, better than
  random `1.1454`, kNN `1.6133`, and `matched_pair` `1.6909`.
- IHDP tiny smoke: kNN PEHE `0.7670` is best; `matched_pair_diverse` PEHE
  `1.5349` improves over `matched_pair` `2.0580` but not over kNN/random.
- This is only a backend-callability check, not paper-scale TabICL evidence.

## Baseline Audit

Artifacts:

- `experiment3/results/baseline_audit.md`
- `experiment3/results/classical_dependency_check.json`

Status:

- Required retrieval baselines are retained: random, kNN, `matched_pair`, and
  `matched_pair_diverse`.
- Built-in full-data learners (`global_s_learner`, `global_t_learner`,
  `global_x_learner`, `global_dr_learner`) are reference baselines only.
- Optional packages for causal forest / econml DRLearner / econml XLearner are
  unavailable: `econml`, `causalml`, `sklift`, and `dowhy` all raise
  `ModuleNotFoundError` in the current environment.

## Paper-Ready Audit

See `experiment3/results/paper_ready_audit.md`.

The main claim remains retrieval-level only. Current backend check shows TabICL
and TabPFN packages are importable, but main experiment evidence still uses the
local ridge T-learner fallback to isolate retrieval. TabICL has only a tiny
smoke run; TabPFN model fitting is blocked until a license token is supplied.
