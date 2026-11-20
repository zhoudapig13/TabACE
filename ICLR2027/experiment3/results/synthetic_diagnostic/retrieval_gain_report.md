# Diagnostic Synthetic Retrieval Gain Report

Relative PEHE reduction uses aggregate means: `(mean baseline PEHE - mean method PEHE) / mean baseline PEHE`. Negative PEHE/ATE deltas are improvements; positive policy-value deltas are improvements.

## Overall vs `random`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 200 | 162/200 | -0.6304 | +27.47% | -0.0615 | +0.0328 |
| `propensity_nearest` | 200 | 108/200 | -0.4925 | +21.46% | +0.0932 | -0.2406 |
| `matched_pair_propensity` | 200 | 144/200 | -0.3673 | +16.01% | -0.0066 | -0.0160 |
| `matched_pair_hybrid_w0.05` | 200 | 138/200 | -0.3515 | +15.32% | -0.0019 | -0.0046 |
| `matched_pair` | 200 | 137/200 | -0.3480 | +15.17% | +0.0042 | +0.0010 |
| `matched_pair_hybrid_w0.2` | 200 | 137/200 | -0.3480 | +15.17% | +0.0042 | +0.0010 |
| `matched_pair_hybrid_w0.5` | 200 | 134/200 | -0.3247 | +14.15% | -0.0056 | +0.0046 |
| `matched_pair_caliper_c0.05` | 200 | 98/200 | -0.3020 | +13.16% | -0.0171 | -0.0480 |
| `matched_pair_feature` | 200 | 130/200 | -0.2946 | +12.84% | +0.0777 | +0.0443 |
| `treatment_balanced_random` | 200 | 127/200 | -0.2769 | +12.07% | -0.0121 | +0.0045 |
| `matched_pair_caliper_c0.2` | 200 | 88/200 | -0.1220 | +5.31% | +0.0138 | -0.0095 |
| `matched_pair_caliper_c0.1` | 200 | 86/200 | -0.1115 | +4.86% | +0.0202 | -0.0214 |

## Overall vs `knn`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 200 | 167/200 | -0.6483 | +28.03% | -0.2067 | +0.0085 |
| `propensity_nearest` | 200 | 119/200 | -0.5104 | +22.07% | -0.0520 | -0.2649 |
| `matched_pair_propensity` | 200 | 151/200 | -0.3853 | +16.66% | -0.1517 | -0.0403 |
| `matched_pair_hybrid_w0.05` | 200 | 150/200 | -0.3695 | +15.98% | -0.1471 | -0.0289 |
| `matched_pair` | 200 | 150/200 | -0.3660 | +15.82% | -0.1410 | -0.0232 |
| `matched_pair_hybrid_w0.2` | 200 | 150/200 | -0.3660 | +15.82% | -0.1410 | -0.0232 |
| `matched_pair_hybrid_w0.5` | 200 | 152/200 | -0.3427 | +14.82% | -0.1508 | -0.0197 |
| `matched_pair_caliper_c0.05` | 200 | 107/200 | -0.3200 | +13.84% | -0.1622 | -0.0723 |
| `matched_pair_feature` | 200 | 146/200 | -0.3126 | +13.51% | -0.0675 | +0.0201 |
| `treatment_balanced_random` | 200 | 124/200 | -0.2948 | +12.75% | -0.1572 | -0.0198 |
| `matched_pair_caliper_c0.2` | 200 | 99/200 | -0.1399 | +6.05% | -0.1314 | -0.0338 |
| `matched_pair_caliper_c0.1` | 200 | 100/200 | -0.1295 | +5.60% | -0.1249 | -0.0457 |

## Overall vs `matched_pair`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 200 | 141/200 | -0.2823 | +14.50% | -0.0657 | +0.0317 |
| `propensity_nearest` | 200 | 75/200 | -0.1445 | +7.42% | +0.0890 | -0.2417 |
| `matched_pair_propensity` | 200 | 95/200 | -0.0193 | +0.99% | -0.0107 | -0.0171 |
| `matched_pair_hybrid_w0.05` | 200 | 89/200 | -0.0035 | +0.18% | -0.0061 | -0.0056 |
| `matched_pair_hybrid_w0.2` | 200 | 0/200 | +0.0000 | +0.00% | +0.0000 | +0.0000 |
| `matched_pair_hybrid_w0.5` | 200 | 86/200 | +0.0233 | -1.20% | -0.0098 | +0.0035 |
| `matched_pair_caliper_c0.05` | 200 | 61/200 | +0.0460 | -2.36% | -0.0212 | -0.0490 |
| `matched_pair_feature` | 200 | 76/200 | +0.0534 | -2.74% | +0.0735 | +0.0433 |
| `treatment_balanced_random` | 200 | 73/200 | +0.0712 | -3.66% | -0.0162 | +0.0035 |
| `matched_pair_caliper_c0.2` | 200 | 54/200 | +0.2261 | -11.61% | +0.0096 | -0.0105 |
| `matched_pair_caliper_c0.1` | 200 | 58/200 | +0.2365 | -12.15% | +0.0161 | -0.0225 |
| `overlap_filtered_matched_pair` | 200 | 19/200 | +0.8342 | -42.85% | +0.3646 | -0.0211 |

## Paper Method Bootstrap Checks

| Method | Baseline | Metric | Mean Delta | 95% CI | Wilcoxon p |
| --- | --- | --- | ---: | ---: | ---: |
| `matched_pair_diverse` | `random` | `pehe` | -0.6304 | [-0.7591, -0.5016] | 6.406e-21 |
| `matched_pair_diverse` | `random` | `ate_error` | -0.0615 | [-0.1419, +0.0137] | 0.4334 |
| `matched_pair_diverse` | `random` | `policy_value` | +0.0328 | [+0.0114, +0.0556] | 0.008143 |
| `matched_pair_diverse` | `knn` | `pehe` | -0.6483 | [-0.7940, -0.5091] | 5.839e-21 |
| `matched_pair_diverse` | `knn` | `ate_error` | -0.2067 | [-0.2849, -0.1299] | 1.163e-07 |
| `matched_pair_diverse` | `knn` | `policy_value` | +0.0085 | [-0.0163, +0.0335] | 0.2604 |
| `matched_pair_diverse` | `matched_pair` | `pehe` | -0.2823 | [-0.3695, -0.2097] | 7.382e-13 |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | -0.0657 | [-0.1200, -0.0090] | 0.03457 |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | +0.0317 | [+0.0151, +0.0494] | 0.0003454 |
| `matched_pair` | `random` | `pehe` | -0.3480 | [-0.4921, -0.1955] | 1.508e-07 |
| `matched_pair` | `random` | `ate_error` | +0.0042 | [-0.0884, +0.1028] | 0.849 |
| `matched_pair` | `random` | `policy_value` | +0.0010 | [-0.0222, +0.0248] | 0.54 |
| `matched_pair` | `knn` | `pehe` | -0.3660 | [-0.5287, -0.2178] | 1.491e-11 |
| `matched_pair` | `knn` | `ate_error` | -0.1410 | [-0.2213, -0.0581] | 1.37e-05 |
| `matched_pair` | `knn` | `policy_value` | -0.0232 | [-0.0477, +0.0004] | 0.5013 |
