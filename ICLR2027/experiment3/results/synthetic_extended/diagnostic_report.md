# Diagnostic Report

Negative PEHE/ATE/policy-regret deltas are better. Positive policy-value deltas are better.
Relative PEHE reduction in the main tables is computed from aggregate means; the per-run percentage is kept only as a diagnostic because it can be unstable when a baseline PEHE is very small.

## Paper Methods Overall

| Method | Baseline | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | `knn` | 700 | 533/700 | -0.5092 | +20.16% | 438/700 | -0.1589 | 326/700 | -0.0021 |
| `matched_pair_diverse` | `knn` | 700 | 589/700 | -0.7429 | +29.42% | 457/700 | -0.2117 | 371/700 | +0.0353 |
| `matched_pair_diverse` | `matched_pair` | 700 | 477/700 | -0.2337 | +11.59% | 385/700 | -0.0528 | 370/700 | +0.0374 |
| `matched_pair` | `random` | 700 | 499/700 | -0.4100 | +16.90% | 356/700 | -0.0161 | 349/700 | +0.0138 |
| `matched_pair_diverse` | `random` | 700 | 581/700 | -0.6438 | +26.53% | 376/700 | -0.0689 | 390/700 | +0.0512 |

## Bootstrap / Paired Tests

| Method | Baseline | Metric | Runs | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate | Significant? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair` | `knn` | `pehe` | 700 | -0.5092 | -0.2424 | [-0.5946, -0.4307] | 1.51e-32 | 6.612e-47 | 76.14% | yes |
| `matched_pair` | `knn` | `ate_error` | 700 | -0.1589 | -0.0899 | [-0.2061, -0.1135] | 3.957e-12 | 2.615e-13 | 62.57% | yes |
| `matched_pair` | `knn` | `policy_value` | 700 | -0.0021 | +0.0000 | [-0.0141, +0.0111] | 0.7301 | 0.6201 | 46.57% | mixed/no |
| `matched_pair` | `knn` | `policy_regret` | 700 | +0.0021 | +0.0000 | [-0.0099, +0.0156] | 0.7301 | 0.6201 | 46.57% | mixed/no |
| `matched_pair` | `random` | `pehe` | 700 | -0.4100 | -0.1972 | [-0.4818, -0.3440] | 9.943e-31 | 3.354e-34 | 71.29% | yes |
| `matched_pair` | `random` | `ate_error` | 700 | -0.0161 | -0.0070 | [-0.0609, +0.0297] | 0.4983 | 0.5183 | 50.86% | mixed/no |
| `matched_pair` | `random` | `policy_value` | 700 | +0.0138 | +0.0000 | [+0.0026, +0.0251] | 0.01501 | 0.005174 | 49.86% | yes |
| `matched_pair` | `random` | `policy_regret` | 700 | -0.0138 | +0.0000 | [-0.0250, -0.0024] | 0.01501 | 0.005174 | 49.86% | yes |
| `matched_pair_diverse` | `knn` | `pehe` | 700 | -0.7429 | -0.4544 | [-0.8275, -0.6714] | 8.255e-70 | 6.497e-78 | 84.14% | yes |
| `matched_pair_diverse` | `knn` | `ate_error` | 700 | -0.2117 | -0.1263 | [-0.2547, -0.1690] | 3.472e-20 | 2.038e-21 | 65.29% | yes |
| `matched_pair_diverse` | `knn` | `policy_value` | 700 | +0.0353 | +0.0149 | [+0.0230, +0.0476] | 5.128e-09 | 6.673e-09 | 53.00% | yes |
| `matched_pair_diverse` | `knn` | `policy_regret` | 700 | -0.0353 | -0.0149 | [-0.0474, -0.0227] | 5.128e-09 | 6.673e-09 | 53.00% | yes |
| `matched_pair_diverse` | `matched_pair` | `pehe` | 700 | -0.2337 | -0.0934 | [-0.2711, -0.1976] | 1.881e-31 | 6.741e-35 | 68.14% | yes |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | 700 | -0.0528 | -0.0198 | [-0.0807, -0.0239] | 0.0002483 | 0.0001937 | 55.00% | yes |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | 700 | +0.0374 | +0.0125 | [+0.0292, +0.0457] | 6.726e-19 | 6.765e-18 | 52.86% | yes |
| `matched_pair_diverse` | `matched_pair` | `policy_regret` | 700 | -0.0374 | -0.0125 | [-0.0463, -0.0293] | 6.726e-19 | 6.765e-18 | 52.86% | yes |
| `matched_pair_diverse` | `random` | `pehe` | 700 | -0.6438 | -0.3221 | [-0.7082, -0.5766] | 7.907e-75 | 4.792e-75 | 83.00% | yes |
| `matched_pair_diverse` | `random` | `ate_error` | 700 | -0.0689 | -0.0180 | [-0.1111, -0.0268] | 0.001164 | 0.002552 | 53.71% | yes |
| `matched_pair_diverse` | `random` | `policy_value` | 700 | +0.0512 | +0.0269 | [+0.0407, +0.0622] | 2.612e-20 | 3.733e-18 | 55.71% | yes |
| `matched_pair_diverse` | `random` | `policy_regret` | 700 | -0.0512 | -0.0269 | [-0.0635, -0.0402] | 2.612e-20 | 3.733e-18 | 55.71% | yes |

## Retrieval Quality Correlations

| Method | Baseline | Quality Delta | Runs | Corr With PEHE Improvement |
| --- | --- | --- | ---: | ---: |
| `matched_pair_diverse` | `random` | `meta_mean_context_propensity_distance` | 700 | -0.4015 |
| `overlap_aware` | `random` | `meta_mean_context_duplicate_rate` | 700 | -0.3970 |
| `overlap_aware` | `random` | `meta_mean_context_abs_smd` | 693 | -0.3814 |
| `overlap_aware` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 693 | -0.3814 |
| `overlap_aware` | `knn` | `meta_mean_context_duplicate_rate` | 700 | -0.3793 |
| `matched_pair_diverse` | `random` | `meta_mean_context_feature_distance` | 700 | -0.3701 |
| `matched_pair_diverse` | `random` | `meta_mean_context_abs_smd` | 700 | -0.3649 |
| `matched_pair_diverse` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 700 | -0.3649 |
| `matched_pair_diverse` | `knn` | `meta_mean_context_feature_distance` | 700 | +0.3599 |
| `matched_pair` | `random` | `meta_mean_context_propensity_distance` | 700 | -0.3532 |
| `overlap_aware` | `knn` | `meta_mean_context_abs_smd` | 693 | -0.3523 |
| `overlap_aware` | `knn` | `meta_context_abs_smd_delta_vs_full_train` | 693 | -0.3523 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_abs_smd` | 700 | -0.3432 |
| `matched_pair_diverse` | `matched_pair` | `meta_context_abs_smd_delta_vs_full_train` | 700 | -0.3432 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_abs_smd` | 693 | -0.3419 |
| `overlap_aware` | `matched_pair` | `meta_context_abs_smd_delta_vs_full_train` | 693 | -0.3419 |
| `overlap_aware` | `matched_pair` | `meta_mean_context_duplicate_rate` | 700 | -0.3399 |
| `propensity_nearest` | `random` | `meta_mean_context_feature_distance` | 700 | -0.3375 |
| `propensity_nearest` | `random` | `meta_mean_context_diversity` | 700 | -0.3257 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_diversity` | 700 | +0.3246 |

## Interpretation

- `matched_pair_diverse` is the promoted method when its PEHE bootstrap CI stays below zero against random, kNN, and the original matched pair.
- ATE/policy results should be treated as secondary unless their paired intervals exclude zero.
- `overlap_aware` should remain a negative/control result unless its redesigned overlap-filtered matching variant is consistently better.
