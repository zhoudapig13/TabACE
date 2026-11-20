# Diagnostic Report

Negative PEHE/ATE/policy-regret deltas are better. Positive policy-value deltas are better.
Relative PEHE reduction in the main tables is computed from aggregate means; the per-run percentage is kept only as a diagnostic because it can be unstable when a baseline PEHE is very small.

## Paper Methods Overall

| Method | Baseline | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | `knn` | 150 | 102/150 | -0.4833 | +10.40% | 105/150 | -0.4646 | 104/150 | +0.4677 |
| `matched_pair_diverse` | `knn` | 150 | 89/150 | +0.0052 | -0.11% | 104/150 | -0.4711 | 89/150 | +0.3965 |
| `matched_pair_diverse` | `matched_pair` | 150 | 68/150 | +0.4885 | -11.73% | 84/150 | -0.0065 | 51/150 | -0.0712 |
| `matched_pair` | `random` | 150 | 107/150 | -0.5409 | +11.49% | 79/150 | -0.1415 | 89/150 | +0.3235 |
| `matched_pair_diverse` | `random` | 150 | 99/150 | -0.0524 | +1.11% | 82/150 | -0.1480 | 79/150 | +0.2523 |

## Bootstrap / Paired Tests

| Method | Baseline | Metric | Runs | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate | Significant? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair` | `knn` | `pehe` | 150 | -0.4833 | -0.2571 | [-0.8308, -0.1599] | 0.004908 | 8.384e-06 | 68.00% | yes |
| `matched_pair` | `knn` | `ate_error` | 150 | -0.4646 | -0.2548 | [-0.6995, -0.2310] | 4.82e-05 | 1.335e-09 | 70.00% | yes |
| `matched_pair` | `knn` | `policy_value` | 150 | +0.4677 | +0.0660 | [+0.3161, +0.6329] | 1.822e-09 | 2.663e-10 | 69.33% | yes |
| `matched_pair` | `knn` | `policy_regret` | 150 | -0.4677 | -0.0660 | [-0.6410, -0.3301] | 1.822e-09 | 2.663e-10 | 69.33% | yes |
| `matched_pair` | `random` | `pehe` | 150 | -0.5409 | -0.3360 | [-0.8427, -0.2742] | 0.0004173 | 9.07e-06 | 71.33% | yes |
| `matched_pair` | `random` | `ate_error` | 150 | -0.1415 | -0.0262 | [-0.2562, -0.0317] | 0.01561 | 0.09815 | 52.67% | mixed/no |
| `matched_pair` | `random` | `policy_value` | 150 | +0.3235 | +0.0346 | [+0.1772, +0.5227] | 0.0003244 | 4.45e-07 | 59.33% | yes |
| `matched_pair` | `random` | `policy_regret` | 150 | -0.3235 | -0.0346 | [-0.5206, -0.1765] | 0.0003244 | 4.45e-07 | 59.33% | yes |
| `matched_pair_diverse` | `knn` | `pehe` | 150 | +0.0052 | -0.1758 | [-0.4635, +0.5648] | 0.9847 | 0.02781 | 59.33% | mixed/no |
| `matched_pair_diverse` | `knn` | `ate_error` | 150 | -0.4711 | -0.2240 | [-0.6548, -0.2950] | 7.401e-07 | 1.14e-08 | 69.33% | yes |
| `matched_pair_diverse` | `knn` | `policy_value` | 150 | +0.3965 | +0.0408 | [+0.2667, +0.5473] | 4.406e-08 | 3.84e-06 | 59.33% | yes |
| `matched_pair_diverse` | `knn` | `policy_regret` | 150 | -0.3965 | -0.0408 | [-0.5566, -0.2629] | 4.406e-08 | 3.84e-06 | 59.33% | yes |
| `matched_pair_diverse` | `matched_pair` | `pehe` | 150 | +0.4885 | +0.0223 | [+0.2304, +0.7959] | 0.001074 | 0.06472 | 45.33% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | 150 | -0.0065 | -0.0203 | [-0.1299, +0.1002] | 0.9101 | 0.5266 | 56.00% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | 150 | -0.0712 | +0.0000 | [-0.1264, -0.0195] | 0.008405 | 0.06179 | 34.00% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `policy_regret` | 150 | +0.0712 | +0.0000 | [+0.0243, +0.1266] | 0.008405 | 0.06179 | 34.00% | mixed/no |
| `matched_pair_diverse` | `random` | `pehe` | 150 | -0.0524 | -0.3295 | [-0.4368, +0.3776] | 0.8061 | 0.04519 | 66.00% | mixed/no |
| `matched_pair_diverse` | `random` | `ate_error` | 150 | -0.1480 | -0.0260 | [-0.3028, -0.0020] | 0.05248 | 0.03585 | 54.67% | mixed/no |
| `matched_pair_diverse` | `random` | `policy_value` | 150 | +0.2523 | +0.0117 | [+0.1045, +0.4461] | 0.005624 | 0.001871 | 52.67% | yes |
| `matched_pair_diverse` | `random` | `policy_regret` | 150 | -0.2523 | -0.0117 | [-0.4478, -0.1081] | 0.005624 | 0.001871 | 52.67% | yes |

## Retrieval Quality Correlations

| Method | Baseline | Quality Delta | Runs | Corr With PEHE Improvement |
| --- | --- | --- | ---: | ---: |
| `matched_pair` | `knn` | `meta_mean_context_diversity` | 150 | +0.5003 |
| `matched_pair_diverse` | `random` | `meta_mean_context_diversity` | 150 | -0.4361 |
| `matched_pair_diverse` | `random` | `meta_mean_context_feature_distance` | 150 | -0.4311 |
| `matched_pair` | `knn` | `meta_mean_context_feature_distance` | 150 | +0.3732 |
| `matched_pair` | `random` | `meta_mean_context_abs_smd` | 150 | -0.3590 |
| `matched_pair` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 150 | -0.3590 |
| `matched_pair` | `knn` | `meta_mean_context_treated_rate` | 150 | +0.3268 |
| `matched_pair` | `random` | `meta_mean_context_diversity` | 150 | -0.3137 |
| `matched_pair` | `random` | `meta_mean_context_feature_distance` | 150 | -0.3122 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_treated_rate` | 150 | +0.2844 |
| `overlap_aware` | `knn` | `meta_mean_context_treated_rate` | 150 | +0.2838 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_diversity` | 150 | -0.2692 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_feature_distance` | 150 | -0.2563 |
| `matched_pair` | `knn` | `meta_mean_context_propensity_distance` | 150 | -0.2451 |
| `matched_pair_diverse` | `knn` | `meta_mean_context_diversity` | 150 | +0.2260 |
| `matched_pair_diverse` | `random` | `meta_mean_context_abs_smd` | 150 | -0.2110 |
| `matched_pair_diverse` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 150 | -0.2110 |
| `overlap_aware` | `matched_pair` | `meta_context_abs_smd_delta_vs_full_train` | 150 | -0.1933 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_abs_smd` | 150 | -0.1933 |
| `overlap_aware` | `knn` | `meta_mean_context_overlap` | 150 | +0.1801 |

## Interpretation

- `matched_pair_diverse` is the promoted method when its PEHE bootstrap CI stays below zero against random, kNN, and the original matched pair.
- ATE/policy results should be treated as secondary unless their paired intervals exclude zero.
- `overlap_aware` should remain a negative/control result unless its redesigned overlap-filtered matching variant is consistently better.
