# IHDP Scaled Retrieval Gain Report

Relative PEHE reduction uses aggregate means: `(mean baseline PEHE - mean
method PEHE) / mean baseline PEHE`. Negative PEHE/ATE deltas are improvements;
positive policy-value deltas are improvements.

This scaled IHDP run uses 50 realizations, the full 75-unit test split for each
realization, and compressed context sizes 16/64/256.

## Overall vs `random`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | 150 | 107/150 | -0.5409 | +11.49% | -0.1415 | +0.3235 |
| `matched_pair_diverse` | 150 | 99/150 | -0.0524 | +1.11% | -0.1480 | +0.2523 |
| `overlap_aware` | 150 | 32/150 | +1.1798 | -25.07% | +0.2428 | -0.2351 |

## Overall vs `knn`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | 150 | 102/150 | -0.4833 | +10.40% | -0.4646 | +0.4677 |
| `matched_pair_diverse` | 150 | 89/150 | +0.0052 | -0.11% | -0.4711 | +0.3965 |
| `overlap_aware` | 150 | 44/150 | +1.2374 | -26.62% | -0.0803 | -0.0909 |

## Overall vs `matched_pair`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 150 | 68/150 | +0.4885 | -11.73% | -0.0065 | -0.0712 |
| `overlap_aware` | 150 | 5/150 | +1.7208 | -41.32% | +0.3843 | -0.5586 |

## Paper Method Bootstrap Checks

| Method | Baseline | Metric | Mean Delta | 95% CI | Wilcoxon p |
| --- | --- | --- | ---: | ---: | ---: |
| `matched_pair_diverse` | `random` | `pehe` | -0.0524 | [-0.4368, +0.3776] | 0.04519 |
| `matched_pair_diverse` | `random` | `ate_error` | -0.1480 | [-0.3028, -0.0020] | 0.03585 |
| `matched_pair_diverse` | `random` | `policy_value` | +0.2523 | [+0.1045, +0.4461] | 0.001871 |
| `matched_pair_diverse` | `knn` | `pehe` | +0.0052 | [-0.4635, +0.5648] | 0.02781 |
| `matched_pair_diverse` | `knn` | `ate_error` | -0.4711 | [-0.6548, -0.2950] | 1.140e-08 |
| `matched_pair_diverse` | `knn` | `policy_value` | +0.3965 | [+0.2667, +0.5473] | 3.840e-06 |
| `matched_pair_diverse` | `matched_pair` | `pehe` | +0.4885 | [+0.2304, +0.7959] | 0.06472 |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | -0.0065 | [-0.1299, +0.1002] | 0.5266 |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | -0.0712 | [-0.1264, -0.0195] | 0.06179 |
| `matched_pair` | `random` | `pehe` | -0.5409 | [-0.8427, -0.2742] | 9.070e-06 |
| `matched_pair` | `random` | `ate_error` | -0.1415 | [-0.2562, -0.0317] | 0.09815 |
| `matched_pair` | `random` | `policy_value` | +0.3235 | [+0.1772, +0.5227] | 4.450e-07 |
| `matched_pair` | `knn` | `pehe` | -0.4833 | [-0.8308, -0.1599] | 8.384e-06 |
| `matched_pair` | `knn` | `ate_error` | -0.4646 | [-0.6995, -0.2310] | 1.335e-09 |
| `matched_pair` | `knn` | `policy_value` | +0.4677 | [+0.3161, +0.6329] | 2.663e-10 |

## Interpretation

This scaled IHDP check is mixed for the fixed main method
`matched_pair_diverse`: its mean PEHE is slightly better than random but the
bootstrap interval crosses zero, it is essentially tied with kNN, and it is
worse than original `matched_pair`. The stronger external conclusion is that
treated-control matching is useful, with `matched_pair` currently the more
robust IHDP retriever.
