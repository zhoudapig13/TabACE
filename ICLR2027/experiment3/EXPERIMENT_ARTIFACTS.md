# Experiment 3 Artifacts

Experiment outputs should live under `experiment3/results/`.

Record important result groups here once runs are complete:

| Group | Raw results | Summary | Report | Notes |
| --- | --- | --- | --- | --- |
| Synthetic SCM smoke | `results/smoke/synthetic_results.csv` | `results/smoke/synthetic_summary.csv` | `results/smoke/synthetic_report.md` | Includes `pehe_by_context_size.png` and `environment.json`; direct smoke test passed. |
| Default synthetic ablation | `results/synthetic_default/synthetic_results.csv` | `results/synthetic_default/synthetic_summary.csv` | `results/synthetic_default/synthetic_report.md` | 384 rows; context sizes 16/32/64/128/256 across confounding, imbalance, and overlap sweeps. |
| Retrieval gain vs random/kNN | `results/synthetic_default/retrieval_gain_vs_random_knn.csv` | TBD | `results/synthetic_default/retrieval_gain_report.md` | `matched_pair` is the strongest positive retrieval method so far. |
| Multi-regime diagnostic | `results/synthetic_diagnostic/synthetic_results.csv` | `results/synthetic_diagnostic/synthetic_summary.csv` | `results/synthetic_diagnostic/retrieval_gain_report.md` | 3360 rows; 4 regimes x 10 seeds x 5 context sizes; `matched_pair_diverse` is best PEHE variant. |
| Diagnostic deltas and tests | `results/synthetic_diagnostic/scenario_level_deltas.csv` | `results/synthetic_diagnostic/diagnostic_overall.csv` | `results/synthetic_diagnostic/diagnostic_report.md` | Includes paired bootstrap tests, grouping by context/regime/confounding/imbalance/overlap, and retrieval-quality correlations. |
| Diagnostic plots | TBD | TBD | `results/synthetic_diagnostic/summary.md` | Plots saved in `results/synthetic_diagnostic/plots/` and mirrored to `results/synthetic_default/plots/`. |
| Paper-ready audit | TBD | TBD | `results/paper_ready_audit.md` | Consolidates strongest claim, weak claim, negative result, relative PEHE audit, IHDP status, and backend boundary. |
| Extended synthetic validation | `results/synthetic_extended/synthetic_results.csv` | `results/synthetic_extended/diagnostic_overall.csv` | `results/synthetic_extended/retrieval_gain_report.md` | 5460 rows; 7 regimes x 20 seeds x focused retrievers; `matched_pair_diverse` PEHE CI is below zero vs random/kNN/original matched pair. |
| Extended synthetic plots | TBD | TBD | `results/synthetic_extended/diagnostic_report.md` | Includes PEHE delta by context size, PEHE delta by regime, CI forest plot, and retrieval-quality scatter plots. |
| IHDP initial validation | `results/ihdp_initial/ihdp_results.csv` | `results/ihdp_initial/diagnostic_overall.csv` | `results/ihdp_initial/retrieval_gain_report.md` | 10 IHDP realizations using downloaded `ihdp_npci_1-100` npz files; preliminary external support. |
| IHDP scaled validation | `results/ihdp_scaled/ihdp_results.csv` | `results/ihdp_scaled/diagnostic_overall.csv` | `results/ihdp_scaled/retrieval_gain_report.md` | 850 rows; 50 realizations x full 75-unit test split x context sizes 16/64/256; supports `matched_pair` externally, not `matched_pair_diverse`; includes diagnostic report, bootstrap CI, and plots. |
| ACIC 2016 initial validation | `results/acic2016_initial/acic2016_results.csv` | `results/acic2016_initial/diagnostic_overall.csv` | `results/acic2016_initial/retrieval_gain_report.md` | 51 rows; ACIC 2016 condition 1 from causallib; `matched_pair` has significant initial PEHE gains vs random/kNN. |
| External dataset 2 | `results/external_2/external_2_results.csv` | `results/external_2/diagnostic_overall.csv` | `results/external_2/README.md` | Objective-level second external dataset directory; mirrors ACIC 2016 condition 1 initial validation with report, bootstrap CI, and plots. |
| ACIC 2016 data | `data/acic2016/x.csv`, `data/acic2016/zymu_1.csv` | TBD | `data/acic2016/README.md` | Public causallib copy of ACIC 2016 condition 1 inputs with license included. |
| Backend check | `results/backend_check.json` | TBD | TBD | TabICL and TabPFN packages import successfully; main evidence still uses local fallback estimator. |
| Real backend smoke | `results/real_backend_smoke/real_backend_smoke_results.csv` | TBD | TBD | Tiny real `TabICLRegressor` smoke only; TabPFN fit/predict probe is blocked by license-token requirement. |
| Baseline audit | `results/classical_dependency_check.json` | TBD | `results/baseline_audit.md` | Required retrieval baselines retained; built-in full-data learners are reference baselines; optional econml/causalml/sklift/dowhy dependencies unavailable. |
| Paper materials | `RESULTS_TABLES.md` | TBD | `PAPER_OUTLINE.md` | Includes `METHOD_DESCRIPTION.md` and `CLAIMS_AND_LIMITATIONS.md`; summarizes safe claims and limitations. |
