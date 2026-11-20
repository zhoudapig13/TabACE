# Diagnostic Report

Negative PEHE/ATE/policy-regret deltas are better. Positive policy-value deltas are better.
Relative PEHE reduction in the main tables is computed from aggregate means; the per-run percentage is kept only as a diagnostic because it can be unstable when a baseline PEHE is very small.

## Paper Methods Overall

| Method | Baseline | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | `knn` | 200 | 150/200 | -0.3660 | +15.82% | 128/200 | -0.1410 | 96/200 | -0.0232 |
| `matched_pair_diverse` | `knn` | 200 | 167/200 | -0.6483 | +28.03% | 133/200 | -0.2067 | 99/200 | +0.0085 |
| `matched_pair_diverse` | `matched_pair` | 200 | 141/200 | -0.2823 | +14.50% | 110/200 | -0.0657 | 99/200 | +0.0317 |
| `matched_pair` | `random` | 200 | 137/200 | -0.3480 | +15.17% | 95/200 | +0.0042 | 97/200 | +0.0010 |
| `matched_pair_diverse` | `random` | 200 | 162/200 | -0.6304 | +27.47% | 96/200 | -0.0615 | 100/200 | +0.0328 |

## Bootstrap / Paired Tests

| Method | Baseline | Metric | Runs | Mean Delta | Median Delta | 95% Bootstrap CI | t-test p | Wilcoxon p | Win Rate | Significant? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair` | `knn` | `pehe` | 200 | -0.3660 | -0.2861 | [-0.5287, -0.2178] | 6.696e-06 | 1.491e-11 | 75.00% | yes |
| `matched_pair` | `knn` | `ate_error` | 200 | -0.1410 | -0.0990 | [-0.2213, -0.0581] | 0.0006254 | 1.37e-05 | 64.00% | yes |
| `matched_pair` | `knn` | `policy_value` | 200 | -0.0232 | +0.0000 | [-0.0477, +0.0004] | 0.05895 | 0.5013 | 48.00% | mixed/no |
| `matched_pair` | `knn` | `policy_regret` | 200 | +0.0232 | +0.0000 | [+0.0006, +0.0476] | 0.05895 | 0.5013 | 48.00% | mixed/no |
| `matched_pair` | `random` | `pehe` | 200 | -0.3480 | -0.2078 | [-0.4921, -0.1955] | 4.503e-06 | 1.508e-07 | 68.50% | yes |
| `matched_pair` | `random` | `ate_error` | 200 | +0.0042 | +0.0104 | [-0.0884, +0.1028] | 0.9327 | 0.849 | 47.50% | mixed/no |
| `matched_pair` | `random` | `policy_value` | 200 | +0.0010 | +0.0000 | [-0.0222, +0.0248] | 0.9318 | 0.54 | 48.50% | mixed/no |
| `matched_pair` | `random` | `policy_regret` | 200 | -0.0010 | +0.0000 | [-0.0252, +0.0244] | 0.9318 | 0.54 | 48.50% | mixed/no |
| `matched_pair_diverse` | `knn` | `pehe` | 200 | -0.6483 | -0.4355 | [-0.7940, -0.5091] | 2.908e-18 | 5.839e-21 | 83.50% | yes |
| `matched_pair_diverse` | `knn` | `ate_error` | 200 | -0.2067 | -0.1332 | [-0.2849, -0.1299] | 1.757e-07 | 1.163e-07 | 66.50% | yes |
| `matched_pair_diverse` | `knn` | `policy_value` | 200 | +0.0085 | +0.0000 | [-0.0163, +0.0335] | 0.4956 | 0.2604 | 49.50% | mixed/no |
| `matched_pair_diverse` | `knn` | `policy_regret` | 200 | -0.0085 | +0.0000 | [-0.0333, +0.0134] | 0.4956 | 0.2604 | 49.50% | mixed/no |
| `matched_pair_diverse` | `matched_pair` | `pehe` | 200 | -0.2823 | -0.1023 | [-0.3695, -0.2097] | 5.976e-12 | 7.382e-13 | 70.50% | yes |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | 200 | -0.0657 | -0.0293 | [-0.1200, -0.0090] | 0.02498 | 0.03457 | 55.00% | yes |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | 200 | +0.0317 | +0.0000 | [+0.0151, +0.0494] | 0.0002452 | 0.0003454 | 49.50% | yes |
| `matched_pair_diverse` | `matched_pair` | `policy_regret` | 200 | -0.0317 | +0.0000 | [-0.0486, -0.0168] | 0.0002452 | 0.0003454 | 49.50% | yes |
| `matched_pair_diverse` | `random` | `pehe` | 200 | -0.6304 | -0.3173 | [-0.7591, -0.5016] | 7.651e-21 | 6.406e-21 | 81.00% | yes |
| `matched_pair_diverse` | `random` | `ate_error` | 200 | -0.0615 | +0.0065 | [-0.1419, +0.0137] | 0.1198 | 0.4334 | 48.00% | mixed/no |
| `matched_pair_diverse` | `random` | `policy_value` | 200 | +0.0328 | +0.0005 | [+0.0114, +0.0556] | 0.003844 | 0.008143 | 50.00% | yes |
| `matched_pair_diverse` | `random` | `policy_regret` | 200 | -0.0328 | -0.0005 | [-0.0547, -0.0104] | 0.003844 | 0.008143 | 50.00% | yes |

## Retrieval Quality Correlations

| Method | Baseline | Quality Delta | Runs | Corr With PEHE Improvement |
| --- | --- | --- | ---: | ---: |
| `propensity_nearest` | `random` | `meta_mean_context_feature_distance` | 200 | -0.5348 |
| `matched_pair_diverse` | `random` | `meta_mean_context_propensity_distance` | 200 | -0.4698 |
| `matched_pair_diverse` | `random` | `meta_mean_context_feature_distance` | 200 | -0.4647 |
| `matched_pair_diverse` | `random` | `meta_mean_context_abs_smd` | 200 | -0.4630 |
| `matched_pair_diverse` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 200 | -0.4630 |
| `matched_pair_caliper_c0.2` | `random` | `meta_mean_context_feature_distance` | 200 | -0.4468 |
| `overlap_filtered_matched_pair` | `matched_pair` | `meta_mean_context_overlap` | 200 | -0.4448 |
| `overlap_filtered_matched_pair` | `matched_pair` | `meta_mean_context_feature_distance` | 200 | -0.4148 |
| `matched_pair_hybrid_w0.05` | `random` | `meta_mean_context_abs_smd` | 200 | -0.4030 |
| `matched_pair_hybrid_w0.05` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 200 | -0.4030 |
| `matched_pair_diverse` | `matched_pair` | `meta_mean_context_feature_distance` | 200 | +0.3851 |
| `propensity_nearest` | `matched_pair` | `meta_mean_context_feature_distance` | 200 | -0.3708 |
| `matched_pair_caliper_c0.2` | `random` | `meta_mean_context_treated_rate` | 200 | -0.3697 |
| `matched_pair_feature` | `random` | `meta_mean_context_abs_smd` | 200 | -0.3662 |
| `matched_pair_feature` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 200 | -0.3662 |
| `matched_pair` | `random` | `meta_mean_context_abs_smd` | 200 | -0.3644 |
| `matched_pair` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 200 | -0.3644 |
| `matched_pair_hybrid_w0.2` | `random` | `meta_mean_context_abs_smd` | 200 | -0.3644 |
| `matched_pair_hybrid_w0.2` | `random` | `meta_context_abs_smd_delta_vs_full_train` | 200 | -0.3644 |
| `matched_pair_diverse` | `matched_pair` | `meta_context_abs_smd_delta_vs_full_train` | 200 | -0.3634 |

## Interpretation

- `matched_pair_diverse` is the promoted method when its PEHE bootstrap CI stays below zero against random, kNN, and the original matched pair.
- ATE/policy results should be treated as secondary unless their paired intervals exclude zero.
- `overlap_aware` should remain a negative/control result unless its redesigned overlap-filtered matching variant is consistently better.
