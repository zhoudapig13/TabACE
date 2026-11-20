# Results Tables

## Headline PEHE Deltas

Negative PEHE deltas are better. Rows compare the method against the named
baseline using paired contexts at the same dataset condition and context size.

| Dataset | Method | Baseline | Runs | PEHE delta | 95% bootstrap CI | Win rate | Takeaway |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Extended synthetic | `matched_pair_diverse` | `random` | 700 | -0.6438 | [-0.7082, -0.5766] | 83.00% | Strong positive |
| Extended synthetic | `matched_pair_diverse` | `knn` | 700 | -0.7429 | [-0.8275, -0.6714] | 84.14% | Strong positive |
| Extended synthetic | `matched_pair_diverse` | `matched_pair` | 700 | -0.2337 | [-0.2711, -0.1976] | 68.14% | Diverse wins synthetically |
| IHDP scaled | `matched_pair_diverse` | `random` | 150 | -0.0524 | [-0.4368, +0.3776] | 66.00% | Not robust |
| IHDP scaled | `matched_pair_diverse` | `knn` | 150 | +0.0052 | [-0.4635, +0.5648] | 59.33% | Not robust |
| IHDP scaled | `matched_pair_diverse` | `matched_pair` | 150 | +0.4885 | [+0.2304, +0.7959] | 45.33% | Diverse loses |
| IHDP scaled | `matched_pair` | `random` | 150 | -0.5409 | [-0.8427, -0.2742] | 71.33% | Strong external support |
| IHDP scaled | `matched_pair` | `knn` | 150 | -0.4833 | [-0.8308, -0.1599] | 68.00% | Strong external support |
| ACIC 2016 initial | `matched_pair_diverse` | `random` | 9 | -1.6153 | [-3.2573, +0.2554] | 88.89% | Favorable but underpowered |
| ACIC 2016 initial | `matched_pair_diverse` | `knn` | 9 | -0.9744 | [-2.4402, +0.5998] | 66.67% | Favorable but underpowered |
| ACIC 2016 initial | `matched_pair` | `random` | 9 | -1.9291 | [-2.9483, -0.9297] | 88.89% | Strong initial support |
| ACIC 2016 initial | `matched_pair` | `knn` | 9 | -1.2883 | [-2.7564, -0.1937] | 88.89% | Strong initial support |

`results/external_2/` mirrors the ACIC 2016 initial validation as the second
external dataset result directory.

## Real Backend Smoke

This table is a backend-callability check only.

| Dataset | Backend | Retriever | PEHE | ATE error |
| --- | --- | --- | ---: | ---: |
| Synthetic SCM tiny smoke | TabICLRegressor | `random` | 1.1454 | 0.7145 |
| Synthetic SCM tiny smoke | TabICLRegressor | `knn` | 1.6133 | 1.3399 |
| Synthetic SCM tiny smoke | TabICLRegressor | `matched_pair` | 1.6909 | 1.1087 |
| Synthetic SCM tiny smoke | TabICLRegressor | `matched_pair_diverse` | 1.0607 | 0.4428 |
| IHDP tiny smoke | TabICLRegressor | `random` | 0.8883 | 0.2496 |
| IHDP tiny smoke | TabICLRegressor | `knn` | 0.7670 | 0.2506 |
| IHDP tiny smoke | TabICLRegressor | `matched_pair` | 2.0580 | 0.6015 |
| IHDP tiny smoke | TabICLRegressor | `matched_pair_diverse` | 1.5349 | 0.4661 |

TabPFN import is available, but tiny fit/predict is blocked by
`TabPFNLicenseError` until a `TABPFN_TOKEN` is supplied.

## Baseline Audit

| Baseline group | Status |
| --- | --- |
| Retrieval baselines | random, kNN, `matched_pair`, and `matched_pair_diverse` retained. |
| Built-in full-data references | `global_s_learner`, `global_t_learner`, `global_x_learner`, `global_dr_learner` available as references only. |
| Optional classical packages | `econml`, `causalml`, `sklift`, and `dowhy` unavailable; blocker recorded in `results/classical_dependency_check.json`. |
