# Diagnostic Report

Negative PEHE/ATE/policy-regret deltas are better. Positive policy-value deltas are better.
Relative PEHE reduction in the main tables is computed from aggregate means; the per-run percentage is kept only as a diagnostic because it can be unstable when a baseline PEHE is very small.

## Paper Methods Overall

| Method | Baseline | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | `knn` | 9 | 8/9 | -1.2883 | +24.20% | 5/9 | +0.1590 | 8/9 | +0.3337 |
| `matched_pair_diverse` | `knn` | 9 | 6/9 | -0.9744 | +18.30% | 3/9 | +0.2481 | 8/9 | +0.4247 |
| `matched_pair_diverse` | `matched_pair` | 9 | 4/9 | +0.3139 | -7.78% | 3/9 | +0.0891 | 6/9 | +0.0910 |
| `matched_pair` | `random` | 9 | 8/9 | -1.9291 | +32.34% | 5/9 | -0.0515 | 7/9 | +0.2719 |
| `matched_pair_diverse` | `random` | 9 | 8/9 | -1.6153 | +27.08% | 4/9 | +0.0376 | 8/9 | +0.3629 |

## Bootstrap / Paired Tests

| Method | Baseline | Metric | Runs | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate | Significant? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair` | `knn` | `pehe` | 9 | -1.2883 | -0.4775 | [-2.7564, -0.1937] | 0.07414 | 0.03815 | 88.89% | mixed/no |
| `matched_pair` | `knn` | `ate_error` | 9 | +0.1590 | -0.0624 | [-0.2696, +0.6482] | 0.5296 | 0.9528 | 55.56% | mixed/no |
| `matched_pair` | `knn` | `policy_value` | 9 | +0.3337 | +0.1980 | [+0.1287, +0.5579] | 0.005874 | 0.01516 | 88.89% | yes |
| `matched_pair` | `knn` | `policy_regret` | 9 | -0.3337 | -0.1980 | [-0.5461, -0.1280] | 0.005874 | 0.01516 | 88.89% | yes |
| `matched_pair` | `random` | `pehe` | 9 | -1.9291 | -1.9860 | [-2.9483, -0.9297] | 0.0004509 | 0.01516 | 88.89% | yes |
| `matched_pair` | `random` | `ate_error` | 9 | -0.0515 | -0.0371 | [-0.5747, +0.5187] | 0.8581 | 0.9528 | 55.56% | mixed/no |
| `matched_pair` | `random` | `policy_value` | 9 | +0.2719 | +0.2796 | [+0.0793, +0.4427] | 0.005853 | 0.03815 | 77.78% | yes |
| `matched_pair` | `random` | `policy_regret` | 9 | -0.2719 | -0.2796 | [-0.4442, -0.0793] | 0.005853 | 0.03815 | 77.78% | yes |
| `matched_pair_diverse` | `knn` | `pehe` | 9 | -0.9744 | -1.3177 | [-2.4402, +0.5998] | 0.2431 | 0.1731 | 66.67% | mixed/no |
| `matched_pair_diverse` | `knn` | `ate_error` | 9 | +0.2481 | +0.4682 | [-0.2792, +0.8367] | 0.4009 | 0.4413 | 33.33% | mixed/no |
| `matched_pair_diverse` | `knn` | `policy_value` | 9 | +0.4247 | +0.5187 | [+0.1467, +0.6838] | 0.005159 | 0.0284 | 88.89% | yes |
| `matched_pair_diverse` | `knn` | `policy_regret` | 9 | -0.4247 | -0.5187 | [-0.6855, -0.1516] | 0.005159 | 0.0284 | 88.89% | yes |
| `matched_pair_diverse` | `matched_pair` | `pehe` | 9 | +0.3139 | +0.1347 | [-0.4533, +1.3385] | 0.5092 | 0.859 | 44.44% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | 9 | +0.0891 | +0.1158 | [-0.5280, +0.6512] | 0.7761 | 0.5147 | 33.33% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | 9 | +0.0910 | +0.1550 | [-0.1239, +0.2947] | 0.4198 | 0.4413 | 66.67% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `policy_regret` | 9 | -0.0910 | -0.1550 | [-0.2882, +0.1319] | 0.4198 | 0.4413 | 66.67% | mixed/no |
| `matched_pair_diverse` | `random` | `pehe` | 9 | -1.6153 | -1.9373 | [-3.2573, +0.2554] | 0.09276 | 0.08583 | 88.89% | mixed/no |
| `matched_pair_diverse` | `random` | `ate_error` | 9 | +0.0376 | +0.4166 | [-0.3666, +0.4413] | 0.8656 | 0.9528 | 44.44% | mixed/no |
| `matched_pair_diverse` | `random` | `policy_value` | 9 | +0.3629 | +0.3311 | [+0.0120, +0.6643] | 0.04007 | 0.06632 | 88.89% | mixed/no |
| `matched_pair_diverse` | `random` | `policy_regret` | 9 | -0.3629 | -0.3311 | [-0.6698, -0.0188] | 0.04007 | 0.06632 | 88.89% | mixed/no |

## Retrieval Quality Correlations

| Method | Baseline | Quality Delta | Runs | Corr With PEHE Improvement |
| --- | --- | --- | ---: | ---: |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_treated_rate` | 9 | +0.8398 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_overlap` | 9 | +0.7835 |
| `matched_pair_diverse` | `random` | `meta_mean_context_treated_rate` | 9 | +0.7642 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_diversity` | 9 | +0.7631 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_feature_distance` | 9 | +0.7401 |
| `overlap_aware` | `random` | `meta_mean_context_abs_smd` | 9 | -0.6318 |
| `overlap_aware` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 9 | -0.6318 |
| `matched_pair` | `random` | `meta_mean_context_treated_rate` | 9 | +0.6107 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_abs_smd` | 9 | +0.6004 |
| `matched_pair_diverse` | `matched_pair` | `meta_context_abs_smd_delta_vs_full_train` | 9 | +0.6004 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_propensity_distance` | 9 | +0.5866 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_overlap` | 9 | +0.5308 |
| `matched_pair_diverse` | `random` | `meta_mean_context_overlap` | 9 | +0.5266 |
| `matched_pair_diverse` | `knn` | `meta_mean_context_overlap` | 9 | +0.4911 |
| `matched_pair_diverse` | `random` | `meta_mean_context_propensity_distance` | 9 | -0.4784 |
| `matched_pair_diverse` | `random` | `meta_mean_context_diversity` | 9 | +0.4736 |
| `matched_pair` | `random` | `meta_mean_context_diversity` | 9 | -0.4673 |
| `matched_pair` | `random` | `meta_mean_context_overlap` | 9 | +0.4656 |
| `matched_pair` | `knn` | `meta_mean_context_diversity` | 9 | +0.4649 |
| `matched_pair` | `knn` | `meta_mean_context_treated_rate` | 9 | -0.4645 |

## Interpretation

- `matched_pair_diverse` is the promoted method when its PEHE bootstrap CI stays below zero against random, kNN, and the original matched pair.
- ATE/policy results should be treated as secondary unless their paired intervals exclude zero.
- `overlap_aware` should remain a negative/control result unless its redesigned overlap-filtered matching variant is consistently better.
