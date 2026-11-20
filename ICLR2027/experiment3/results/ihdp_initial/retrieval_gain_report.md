# IHDP Initial Retrieval Gain Report

Relative PEHE reduction uses aggregate means: `(mean baseline PEHE - mean method PEHE) / mean baseline PEHE`. Negative PEHE/ATE deltas are improvements; positive policy-value deltas are improvements.

## Overall vs `random`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 50 | 40/50 | -1.3152 | +23.34% | -0.1893 | +0.2770 |
| `matched_pair` | 50 | 37/50 | -1.1056 | +19.62% | -0.0304 | +0.2259 |
| `treatment_balanced_random` | 50 | 35/50 | -0.6772 | +12.02% | -0.1374 | +0.1104 |
| `propensity_nearest` | 50 | 20/50 | -0.2482 | +4.41% | +0.5414 | -0.2936 |
| `overlap_aware` | 50 | 14/50 | +0.9480 | -16.82% | +0.3876 | -0.3317 |

## Overall vs `knn`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 50 | 39/50 | -0.3326 | +7.15% | -0.7432 | +0.3824 |
| `matched_pair` | 50 | 33/50 | -0.1230 | +2.64% | -0.5844 | +0.3313 |
| `treatment_balanced_random` | 50 | 26/50 | +0.3054 | -6.56% | -0.6914 | +0.2158 |
| `propensity_nearest` | 50 | 16/50 | +0.7343 | -15.79% | -0.0125 | -0.1882 |
| `overlap_aware` | 50 | 12/50 | +1.9305 | -41.50% | -0.1664 | -0.2263 |

## Overall vs `matched_pair`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 50 | 31/50 | -0.2096 | +4.63% | -0.1588 | +0.0511 |
| `treatment_balanced_random` | 50 | 18/50 | +0.4284 | -9.46% | -0.1070 | -0.1154 |
| `propensity_nearest` | 50 | 14/50 | +0.8573 | -18.93% | +0.5719 | -0.5194 |
| `overlap_aware` | 50 | 4/50 | +2.0535 | -45.34% | +0.4180 | -0.5576 |

## Paper Method Bootstrap Checks

| Method | Baseline | Metric | Mean Delta | 95% CI | Wilcoxon p |
| --- | --- | --- | ---: | ---: | ---: |
| `matched_pair_diverse` | `random` | `pehe` | -1.3152 | [-2.2834, -0.5166] | 7.666e-06 |
| `matched_pair_diverse` | `random` | `ate_error` | -0.1893 | [-0.3633, -0.0316] | 0.04024 |
| `matched_pair_diverse` | `random` | `policy_value` | +0.2770 | [+0.1215, +0.4559] | 0.0002614 |
| `matched_pair_diverse` | `knn` | `pehe` | -0.3326 | [-0.9945, +0.5815] | 0.0001835 |
| `matched_pair_diverse` | `knn` | `ate_error` | -0.7432 | [-1.0973, -0.4014] | 3.351e-06 |
| `matched_pair_diverse` | `knn` | `policy_value` | +0.3824 | [+0.2234, +0.5763] | 4.769e-06 |
| `matched_pair_diverse` | `matched_pair` | `pehe` | -0.2096 | [-0.6442, +0.3980] | 0.0274 |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | -0.1588 | [-0.4171, +0.0174] | 0.3719 |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | +0.0511 | [-0.0073, +0.1125] | 0.07168 |
| `matched_pair` | `random` | `pehe` | -1.1056 | [-1.9128, -0.4673] | 0.0003233 |
| `matched_pair` | `random` | `ate_error` | -0.0304 | [-0.2956, +0.2902] | 0.2447 |
| `matched_pair` | `random` | `policy_value` | +0.2259 | [+0.0693, +0.4036] | 0.006748 |
| `matched_pair` | `knn` | `pehe` | -0.1230 | [-0.6544, +0.4584] | 0.1844 |
| `matched_pair` | `knn` | `ate_error` | -0.5844 | [-1.0253, -0.2232] | 0.0004338 |
| `matched_pair` | `knn` | `policy_value` | +0.3313 | [+0.1597, +0.5373] | 0.0002715 |
