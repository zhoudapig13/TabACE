# Diagnostic Report

Negative PEHE/ATE/policy-regret deltas are better. Positive policy-value deltas are better.
Relative PEHE reduction in the main tables is computed from aggregate means; the per-run percentage is kept only as a diagnostic because it can be unstable when a baseline PEHE is very small.

## Paper Methods Overall

| Method | Baseline | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | `knn` | 50 | 33/50 | -0.1230 | +2.64% | 35/50 | -0.5844 | 32/50 | +0.3313 |
| `matched_pair_diverse` | `knn` | 50 | 39/50 | -0.3326 | +7.15% | 36/50 | -0.7432 | 36/50 | +0.3824 |
| `matched_pair_diverse` | `matched_pair` | 50 | 31/50 | -0.2096 | +4.63% | 28/50 | -0.1588 | 22/50 | +0.0511 |
| `matched_pair` | `random` | 50 | 37/50 | -1.1056 | +19.62% | 25/50 | -0.0304 | 33/50 | +0.2259 |
| `matched_pair_diverse` | `random` | 50 | 40/50 | -1.3152 | +23.34% | 30/50 | -0.1893 | 37/50 | +0.2770 |

## Bootstrap / Paired Tests

| Method | Baseline | Metric | Runs | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate | Significant? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair` | `knn` | `pehe` | 50 | -0.1230 | -0.3675 | [-0.6544, +0.4584] | 0.668 | 0.1844 | 66.00% | mixed/no |
| `matched_pair` | `knn` | `ate_error` | 50 | -0.5844 | -0.1929 | [-1.0253, -0.2232] | 0.004042 | 0.0004338 | 70.00% | yes |
| `matched_pair` | `knn` | `policy_value` | 50 | +0.3313 | +0.1193 | [+0.1597, +0.5373] | 0.0009161 | 0.0002715 | 64.00% | yes |
| `matched_pair` | `knn` | `policy_regret` | 50 | -0.3313 | -0.1193 | [-0.5440, -0.1705] | 0.0009161 | 0.0002715 | 64.00% | yes |
| `matched_pair` | `random` | `pehe` | 50 | -1.1056 | -0.4690 | [-1.9128, -0.4673] | 0.002883 | 0.0003233 | 74.00% | yes |
| `matched_pair` | `random` | `ate_error` | 50 | -0.0304 | +0.0026 | [-0.2956, +0.2902] | 0.8364 | 0.2447 | 50.00% | mixed/no |
| `matched_pair` | `random` | `policy_value` | 50 | +0.2259 | +0.0963 | [+0.0693, +0.4036] | 0.01095 | 0.006748 | 66.00% | yes |
| `matched_pair` | `random` | `policy_regret` | 50 | -0.2259 | -0.0963 | [-0.3931, -0.0521] | 0.01095 | 0.006748 | 66.00% | yes |
| `matched_pair_diverse` | `knn` | `pehe` | 50 | -0.3326 | -0.6295 | [-0.9945, +0.5815] | 0.4155 | 0.0001835 | 78.00% | mixed/no |
| `matched_pair_diverse` | `knn` | `ate_error` | 50 | -0.7432 | -0.3207 | [-1.0973, -0.4014] | 7.534e-05 | 3.351e-06 | 72.00% | yes |
| `matched_pair_diverse` | `knn` | `policy_value` | 50 | +0.3824 | +0.1410 | [+0.2234, +0.5763] | 4.065e-05 | 4.769e-06 | 72.00% | yes |
| `matched_pair_diverse` | `knn` | `policy_regret` | 50 | -0.3824 | -0.1410 | [-0.5803, -0.2221] | 4.065e-05 | 4.769e-06 | 72.00% | yes |
| `matched_pair_diverse` | `matched_pair` | `pehe` | 50 | -0.2096 | -0.1009 | [-0.6442, +0.3980] | 0.4252 | 0.0274 | 62.00% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | 50 | -0.1588 | -0.0382 | [-0.4171, +0.0174] | 0.161 | 0.3719 | 56.00% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | 50 | +0.0511 | +0.0000 | [-0.0073, +0.1125] | 0.08835 | 0.07168 | 44.00% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `policy_regret` | 50 | -0.0511 | +0.0000 | [-0.1100, +0.0093] | 0.08835 | 0.07168 | 44.00% | mixed/no |
| `matched_pair_diverse` | `random` | `pehe` | 50 | -1.3152 | -1.1180 | [-2.2834, -0.5166] | 0.002442 | 7.666e-06 | 80.00% | yes |
| `matched_pair_diverse` | `random` | `ate_error` | 50 | -0.1893 | -0.1017 | [-0.3633, -0.0316] | 0.04082 | 0.04024 | 60.00% | yes |
| `matched_pair_diverse` | `random` | `policy_value` | 50 | +0.2770 | +0.2565 | [+0.1215, +0.4559] | 0.0009074 | 0.0002614 | 74.00% | yes |
| `matched_pair_diverse` | `random` | `policy_regret` | 50 | -0.2770 | -0.2565 | [-0.4460, -0.1359] | 0.0009074 | 0.0002614 | 74.00% | yes |

## Retrieval Quality Correlations

| Method | Baseline | Quality Delta | Runs | Corr With PEHE Improvement |
| --- | --- | --- | ---: | ---: |
| `overlap_aware` | `knn` | `meta_mean_context_treated_rate` | 50 | +0.3909 |
| `propensity_nearest` | `random` | `meta_mean_context_abs_smd` | 50 | +0.3369 |
| `propensity_nearest` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 50 | +0.3369 |
| `propensity_nearest` | `matched_pair` | `meta_mean_context_diversity` | 50 | +0.3308 |
| `overlap_aware` | `random` | `meta_mean_context_propensity_distance` | 50 | +0.3254 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_treated_rate` | 50 | +0.3224 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_diversity` | 50 | -0.2831 |
| `propensity_nearest` | `matched_pair` | `meta_mean_context_feature_distance` | 50 | +0.2808 |
| `overlap_aware` | `random` | `meta_mean_context_feature_distance` | 50 | -0.2682 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_propensity_distance` | 50 | -0.2541 |
| `overlap_aware` | `knn` | `meta_mean_context_propensity_distance` | 50 | +0.2477 |
| `overlap_aware` | `random` | `meta_mean_context_diversity` | 50 | -0.2379 |
| `propensity_nearest` | `knn` | `meta_mean_context_diversity` | 50 | -0.2358 |
| `treatment_balanced_random` | `knn` | `meta_mean_context_treated_rate` | 50 | +0.2335 |
| `overlap_aware` | `knn` | `meta_mean_context_overlap` | 50 | +0.2280 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_feature_distance` | 50 | -0.2211 |
| `treatment_balanced_random` | `knn` | `meta_mean_context_overlap` | 50 | +0.2033 |
| `overlap_aware` | `random` | `meta_mean_context_treated_rate` | 50 | +0.2005 |
| `overlap_aware` | `knn` | `meta_mean_context_diversity` | 50 | -0.1837 |
| `matched_pair_diverse` | `random` | `meta_mean_context_feature_distance` | 50 | -0.1793 |

## Interpretation

- `matched_pair_diverse` is the promoted method when its PEHE bootstrap CI stays below zero against random, kNN, and the original matched pair.
- ATE/policy results should be treated as secondary unless their paired intervals exclude zero.
- `overlap_aware` should remain a negative/control result unless its redesigned overlap-filtered matching variant is consistently better.
