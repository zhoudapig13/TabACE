# Extended Synthetic Retrieval Gain Report

Relative PEHE reduction uses aggregate means: `(mean baseline PEHE - mean method PEHE) / mean baseline PEHE`. Negative PEHE/ATE deltas are improvements; positive policy-value deltas are improvements.

## Overall vs `random`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 700 | 581/700 | -0.6438 | +26.53% | -0.0689 | +0.0512 |
| `propensity_nearest` | 700 | 414/700 | -0.4879 | +20.11% | +0.0705 | -0.2119 |
| `matched_pair` | 700 | 499/700 | -0.4100 | +16.90% | -0.0161 | +0.0138 |
| `treatment_balanced_random` | 700 | 434/700 | -0.2682 | +11.05% | -0.0204 | +0.0093 |
| `overlap_aware` | 700 | 240/700 | +1.4894 | -61.39% | +0.7152 | -0.1046 |

## Overall vs `knn`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 700 | 589/700 | -0.7429 | +29.42% | -0.2117 | +0.0353 |
| `propensity_nearest` | 700 | 435/700 | -0.5871 | +23.25% | -0.0723 | -0.2278 |
| `matched_pair` | 700 | 533/700 | -0.5092 | +20.16% | -0.1589 | -0.0021 |
| `treatment_balanced_random` | 700 | 439/700 | -0.3674 | +14.55% | -0.1631 | -0.0067 |
| `overlap_aware` | 700 | 238/700 | +1.3902 | -55.05% | +0.5724 | -0.1206 |

## Overall vs `matched_pair`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 700 | 477/700 | -0.2337 | +11.59% | -0.0528 | +0.0374 |
| `propensity_nearest` | 700 | 290/700 | -0.0779 | +3.87% | +0.0866 | -0.2257 |
| `treatment_balanced_random` | 700 | 254/700 | +0.1418 | -7.03% | -0.0043 | -0.0045 |
| `overlap_aware` | 700 | 130/700 | +1.8994 | -94.21% | +0.7313 | -0.1184 |

## Paper Method Bootstrap Checks

| Method | Baseline | Metric | Mean Delta | 95% CI | Wilcoxon p |
| --- | --- | --- | ---: | ---: | ---: |
| `matched_pair_diverse` | `random` | `pehe` | -0.6438 | [-0.7082, -0.5766] | 4.792e-75 |
| `matched_pair_diverse` | `random` | `ate_error` | -0.0689 | [-0.1111, -0.0268] | 0.002552 |
| `matched_pair_diverse` | `random` | `policy_value` | +0.0512 | [+0.0407, +0.0622] | 3.733e-18 |
| `matched_pair_diverse` | `knn` | `pehe` | -0.7429 | [-0.8275, -0.6714] | 6.497e-78 |
| `matched_pair_diverse` | `knn` | `ate_error` | -0.2117 | [-0.2547, -0.1690] | 2.038e-21 |
| `matched_pair_diverse` | `knn` | `policy_value` | +0.0353 | [+0.0230, +0.0476] | 6.673e-09 |
| `matched_pair_diverse` | `matched_pair` | `pehe` | -0.2337 | [-0.2711, -0.1976] | 6.741e-35 |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | -0.0528 | [-0.0807, -0.0239] | 0.0001937 |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | +0.0374 | [+0.0292, +0.0457] | 6.765e-18 |
| `matched_pair` | `random` | `pehe` | -0.4100 | [-0.4818, -0.3440] | 3.354e-34 |
| `matched_pair` | `random` | `ate_error` | -0.0161 | [-0.0609, +0.0297] | 0.5183 |
| `matched_pair` | `random` | `policy_value` | +0.0138 | [+0.0026, +0.0251] | 0.005174 |
| `matched_pair` | `knn` | `pehe` | -0.5092 | [-0.5946, -0.4307] | 6.612e-47 |
| `matched_pair` | `knn` | `ate_error` | -0.1589 | [-0.2061, -0.1135] | 2.615e-13 |
| `matched_pair` | `knn` | `policy_value` | -0.0021 | [-0.0141, +0.0111] | 0.6201 |
