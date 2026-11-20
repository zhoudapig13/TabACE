# ACIC 2016 Initial Retrieval Gain Report

Relative PEHE reduction uses aggregate means: `(mean baseline PEHE - mean
method PEHE) / mean baseline PEHE`. Negative PEHE/ATE deltas are improvements;
positive policy-value deltas are improvements.

This initial ACIC check uses condition 1 from causallib, 3 split seeds, 800
sampled units, 64 evaluated test units per split, and context sizes 16/64/256.

## Overall vs `random`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | 9 | 8/9 | -1.9291 | +32.34% | -0.0515 | +0.2719 |
| `matched_pair_diverse` | 9 | 8/9 | -1.6153 | +27.08% | +0.0376 | +0.3629 |
| `overlap_aware` | 9 | 6/9 | -0.2593 | +4.35% | +0.3417 | -0.1562 |

## Overall vs `knn`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair` | 9 | 8/9 | -1.2883 | +24.20% | +0.1590 | +0.3337 |
| `matched_pair_diverse` | 9 | 6/9 | -0.9744 | +18.30% | +0.2481 | +0.4247 |
| `overlap_aware` | 9 | 3/9 | +0.3816 | -7.17% | +0.5522 | -0.0944 |

## Overall vs `matched_pair`

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Delta | Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 9 | 4/9 | +0.3139 | -7.78% | +0.0891 | +0.0910 |
| `overlap_aware` | 9 | 1/9 | +1.6699 | -41.37% | +0.3932 | -0.4281 |

## Paper Method Bootstrap Checks

| Method | Baseline | Metric | Mean Delta | 95% CI | Wilcoxon p |
| --- | --- | --- | ---: | ---: | ---: |
| `matched_pair_diverse` | `random` | `pehe` | -1.6153 | [-3.2573, +0.2554] | 0.08583 |
| `matched_pair_diverse` | `random` | `ate_error` | +0.0376 | [-0.3666, +0.4413] | 0.9528 |
| `matched_pair_diverse` | `random` | `policy_value` | +0.3629 | [+0.0120, +0.6643] | 0.06632 |
| `matched_pair_diverse` | `knn` | `pehe` | -0.9744 | [-2.4402, +0.5998] | 0.1731 |
| `matched_pair_diverse` | `knn` | `ate_error` | +0.2481 | [-0.2792, +0.8367] | 0.4413 |
| `matched_pair_diverse` | `knn` | `policy_value` | +0.4247 | [+0.1467, +0.6838] | 0.02840 |
| `matched_pair_diverse` | `matched_pair` | `pehe` | +0.3139 | [-0.4533, +1.3385] | 0.8590 |
| `matched_pair_diverse` | `matched_pair` | `ate_error` | +0.0891 | [-0.5280, +0.6512] | 0.5147 |
| `matched_pair_diverse` | `matched_pair` | `policy_value` | +0.0910 | [-0.1239, +0.2947] | 0.4413 |
| `matched_pair` | `random` | `pehe` | -1.9291 | [-2.9483, -0.9297] | 0.01516 |
| `matched_pair` | `random` | `ate_error` | -0.0515 | [-0.5747, +0.5187] | 0.9528 |
| `matched_pair` | `random` | `policy_value` | +0.2719 | [+0.0793, +0.4427] | 0.03815 |
| `matched_pair` | `knn` | `pehe` | -1.2883 | [-2.7564, -0.1937] | 0.03815 |
| `matched_pair` | `knn` | `ate_error` | +0.1590 | [-0.2696, +0.6482] | 0.9528 |
| `matched_pair` | `knn` | `policy_value` | +0.3337 | [+0.1287, +0.5579] | 0.01516 |

## Interpretation

ACIC 2016 initial validation is underpowered for `matched_pair_diverse`: its
mean PEHE improves over random and kNN, but PEHE bootstrap intervals cross zero
and it does not improve over original `matched_pair`. The stronger initial ACIC
evidence supports the simpler `matched_pair` retriever.
