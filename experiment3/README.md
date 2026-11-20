# Experiment 3: Causal Context Selection for Tabular ICL

Research question: for tabular in-context learning models such as TabICL or
TabPFN, does the choice of in-context examples matter for causal effect
estimation? This prototype isolates context retrieval as the object of study.
It does not train a new causal PFN. The goal is to compare generic similarity
retrieval against causal-aware retrieval based on propensity overlap,
treated-control balance, and matched context construction.

## What This Is Not

This is not a claim to be the first causal PFN-style model. Related model-level
directions already exist, including CausalPFN, Do-PFN, and CausalFM. The
positioning here is narrower: existing tabular ICL models or compatible
wrappers may fail when random or kNN contexts break treatment-control balance,
so causal context selection should be tested as an independent retrieval
principle.

## Structure

- `ccs/datasets.py`: synthetic SCM with tunable sample size, feature dimension,
  treatment imbalance, confounding, overlap violation, and heterogeneous
  treatment effect.
- `ccs/retrievers.py`: random, kNN, treatment-balanced random, propensity
  nearest, treated-control matched-pair, and overlap-aware context retrieval.
- `ccs/estimators.py`: local context T-learner fallback plus TabICL/TabPFN
  wrapper interfaces and global S/T learner baselines.
- `ccs/metrics.py`: PEHE, ATE error, policy value, uplift-at-top-fraction, and
  CATE calibration error.
- `scripts/run_synthetic.py`: experiment runner.
- `scripts/run_ihdp.py`: IHDP semi-synthetic external validation.
- `scripts/run_acic2016.py`: ACIC 2016 external validation with automatic
  causallib data download.
- `scripts/run_real_backend_smoke.py`: tiny real TabICL backend smoke plus a
  TabPFN model-fit probe.
- `scripts/summarize_results.py`: result table and PEHE plot generation.
- `configs/default.yaml`: default synthetic ablation.
- `configs/smoke.yaml`: small reproducibility check.
- `ccs/real_data.py`: IHDP and ACIC 2016 loaders plus placeholder interfaces
  for Lalonde and Criteo Uplift.

## Quick Smoke Test

```bash
python experiment3/scripts/run_synthetic.py --smoke
python experiment3/scripts/summarize_results.py \
  --input experiment3/results/smoke/synthetic_results.csv
```

Expected outputs:

- `experiment3/results/smoke/synthetic_results.csv`
- `experiment3/results/smoke/synthetic_summary.csv`
- `experiment3/results/smoke/synthetic_report.md`
- `experiment3/results/smoke/pehe_by_context_size.png`
- `experiment3/results/smoke/environment.json`

## Default Synthetic Ablation

```bash
python experiment3/scripts/run_synthetic.py \
  --config experiment3/configs/default.yaml

python experiment3/scripts/summarize_results.py \
  --input experiment3/results/synthetic_default/synthetic_results.csv
```

The default config sweeps context size, confounding strength, treatment
imbalance, and overlap violation while comparing:

- random context;
- kNN context;
- treatment-balanced random context;
- propensity-score nearest context;
- treated-control matched-pair context;
- overlap-aware context;
- global S/T learner classical baselines.

## Current Experimental Findings

The strongest current claim is retrieval-level, not model-level:

> Causal-aware matched treated-control context selection improves PEHE for
> context-limited tabular ICL-style CATE estimation over random and kNN
> retrieval. The strongest synthetic variant is `matched_pair_diverse`, while
> scaled external checks show that the simpler `matched_pair` is more robust on
> IHDP and ACIC 2016. Full-data classical causal learners remain competitive,
> and ATE/policy gains are secondary.

The 4-regime, 10-seed diagnostic run is stored under
`experiment3/results/synthetic_diagnostic/`. Key artifacts:

- `retrieval_gain_report.md`: promoted/negative retrieval conclusions.
- `diagnostic_report.md`: paired bootstrap intervals and quality correlations.
- `summary.md`: paper-style claims, weak claims, negative results, and what not
  to claim.
- `plots/`: PEHE/context-size curve, PEHE delta heatmap, ATE/policy scenario
  plots.

Main diagnostic and validation results:

- `matched_pair_diverse` is currently the strongest variant. In the extended
  synthetic validation it improves PEHE by `-0.6438` vs random, `-0.7429` vs
  kNN, and `-0.2337` vs the simpler `matched_pair`, with paired bootstrap CIs
  below zero.
- On scaled IHDP validation with 50 realizations and full 75-sample test splits,
  `matched_pair_diverse` does not show a robust PEHE gain vs random or kNN and
  is worse than `matched_pair`; the simpler `matched_pair` improves PEHE vs
  random by `-0.5409` and vs kNN by `-0.4833`, both with bootstrap CIs below
  zero.
- On ACIC 2016 condition 1 initial validation, `matched_pair_diverse` has better
  mean PEHE than random/kNN but the PEHE intervals cross zero; `matched_pair`
  is again the stronger external method.
- `overlap_aware` and `overlap_filtered_matched_pair` are negative results:
  common-support filtering alone can discard local similarity and hurt PEHE.

Paper-ready audit:

- `experiment3/results/paper_ready_audit.md`
- `experiment3/results/synthetic_extended/retrieval_gain_report.md`
- `experiment3/results/ihdp_initial/retrieval_gain_report.md`
- `experiment3/results/ihdp_scaled/retrieval_gain_report.md`
- `experiment3/results/ihdp_scaled/diagnostic_report.md`
- `experiment3/results/acic2016_initial/diagnostic_report.md`
- `experiment3/results/external_2/README.md`
- `experiment3/results/baseline_audit.md`
- `experiment3/RESULTS_TABLES.md`

External data status:

- IHDP initial validation uses the common `ihdp_npci_1-100.train.npz` and
  `ihdp_npci_1-100.test.npz` files. The helper script can download them from
  `http://www.fredjo.com/files/ihdp_npci_1-100.train.npz` and
  `http://www.fredjo.com/files/ihdp_npci_1-100.test.npz`.
- IHDP supports PEHE because the npz files include `mu0` and `mu1`.
- IHDP scaled validation now uses 50 realizations with the full 75-sample test
  split for each realization. It is external support for causal-aware matching
  as a retrieval family, not for the diverse variant specifically.
- ACIC 2016 condition 1 initial validation uses causallib's public ACIC 2016
  files (`x.csv` and `zymu_1.csv`) with `mu0/mu1` ground truth.
- `experiment3/results/external_2/` mirrors the ACIC 2016 initial validation as
  the second external dataset result directory required for the paper audit.

Baseline audit:

- `experiment3/results/baseline_audit.md` records that random, kNN,
  `matched_pair`, and `matched_pair_diverse` are retained as retrieval
  baselines.
- Built-in `global_s_learner`, `global_t_learner`, `global_x_learner`, and
  `global_dr_learner` remain full-data reference baselines only.
- Optional external causal packages (`econml`, `causalml`, `sklift`, `dowhy`)
  are unavailable in this environment; the dependency blocker is recorded in
  `experiment3/results/classical_dependency_check.json`.

Backend boundary:

- `experiment3/results/backend_check.json` records that TabICL and TabPFN
  packages are importable in this environment.
- `experiment3/results/real_backend_smoke/` records a tiny real
  `TabICLRegressor` smoke on synthetic SCM and IHDP. It is a backend
  availability check, not a paper-scale result.
- TabPFN import works, but `TabPFNRegressor.fit` requires one-time license
  acceptance / `TABPFN_TOKEN` to download model weights in this non-interactive
  environment. The probe records `TabPFNLicenseError`.

## Extending To Real Datasets

The first complete benchmark is synthetic SCM because it provides ground-truth
individual treatment effects for PEHE and ATE error. IHDP and ACIC 2016 are now
implemented because both include `mu0/mu1`. For Lalonde or Criteo Uplift,
implement `ccs/real_data.py::load_real_dataset` to return a `CausalDataset`. If
individual ground truth is unavailable, report ATE error only when a credible
reference estimate is available, and prioritize policy value, uplift curves,
AUUC, or Qini-style metrics.

## Current Fallback Behavior

The main result tables still use the local ridge T-learner trained only on the
selected context. This isolates retrieval quality without retraining any large
model. Real TabICL is importable and passes a tiny `TabICLRegressor` smoke, but
the current paper evidence should still be described as retrieval-level rather
than final TabICL/TabPFN model-level performance.
