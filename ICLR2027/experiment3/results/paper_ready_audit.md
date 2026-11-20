# Paper-Ready Audit

This audit is based on `experiment3` only. It does not use experiment2
accuracy/AUC policy artifacts.

## Main Answer

There are gains over baseline retrieval, but the strongest claim must be
qualified.

- Extended synthetic: `matched_pair_diverse` is a clear improvement over random,
  kNN, and original `matched_pair`.
- Scaled IHDP and ACIC 2016: causal-aware matching still helps, but the simpler
  `matched_pair` is more robust than `matched_pair_diverse`.
- Backend: TabICL and TabPFN packages are importable; only a tiny real TabICL
  smoke has been run. Main result tables remain retrieval-level fallback
  evidence, not final TabICL/TabPFN model-level evidence.

## Extended Synthetic Validation

7 regimes x 20 seeds x 5 context sizes; 700 paired comparisons for each
headline row.

| Comparison | Runs | PEHE delta | PEHE CI | Wilcoxon p | Win rate | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair_diverse` vs `random` | 700 | -0.6438 | [-0.7082, -0.5766] | 4.792e-75 | 83.00% | Strong positive |
| `matched_pair_diverse` vs `knn` | 700 | -0.7429 | [-0.8275, -0.6714] | 6.497e-78 | 84.14% | Strong positive |
| `matched_pair_diverse` vs `matched_pair` | 700 | -0.2337 | [-0.2711, -0.1976] | 6.741e-35 | 68.14% | Diverse variant wins synthetically |
| `matched_pair` vs `random` | 700 | -0.4100 | [-0.4818, -0.3440] | 3.354e-34 | 71.29% | Robust simpler baseline |
| `matched_pair` vs `knn` | 700 | -0.5092 | [-0.5946, -0.4307] | 6.612e-47 | 76.14% | Robust simpler baseline |

## IHDP Scaled Validation

50 realizations, full 75-unit test split per realization, context sizes
16/64/256; 150 paired comparisons per headline row.

| Comparison | Runs | PEHE delta | PEHE CI | Wilcoxon p | Win rate | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair_diverse` vs `random` | 150 | -0.0524 | [-0.4368, +0.3776] | 0.0452 | 66.00% | Mean is favorable, CI crosses zero |
| `matched_pair_diverse` vs `knn` | 150 | +0.0052 | [-0.4635, +0.5648] | 0.0278 | 59.33% | No robust PEHE gain |
| `matched_pair_diverse` vs `matched_pair` | 150 | +0.4885 | [+0.2304, +0.7959] | 0.0647 | 45.33% | Diverse variant is worse |
| `matched_pair` vs `random` | 150 | -0.5409 | [-0.8427, -0.2742] | 9.070e-06 | 71.33% | Strong external support |
| `matched_pair` vs `knn` | 150 | -0.4833 | [-0.8308, -0.1599] | 8.384e-06 | 68.00% | Strong external support |

## ACIC 2016 Initial Validation

Condition 1 from causallib, 3 split seeds, 800 sampled units, 64 evaluated test
units per split, context sizes 16/64/256; 9 paired comparisons per headline
row.

| Comparison | Runs | PEHE delta | PEHE CI | Wilcoxon p | Win rate | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `matched_pair_diverse` vs `random` | 9 | -1.6153 | [-3.2573, +0.2554] | 0.0858 | 88.89% | Favorable but underpowered |
| `matched_pair_diverse` vs `knn` | 9 | -0.9744 | [-2.4402, +0.5998] | 0.1731 | 66.67% | Favorable but underpowered |
| `matched_pair_diverse` vs `matched_pair` | 9 | +0.3139 | [-0.4533, +1.3385] | 0.8590 | 44.44% | Not better than simpler matching |
| `matched_pair` vs `random` | 9 | -1.9291 | [-2.9483, -0.9297] | 0.0152 | 88.89% | Strong initial external support |
| `matched_pair` vs `knn` | 9 | -1.2883 | [-2.7564, -0.1937] | 0.0382 | 88.89% | Strong initial external support |

The objective-level second external dataset directory is
`results/external_2/`; it mirrors this ACIC 2016 initial validation and includes
raw results, bootstrap tests, retrieval gain report, diagnostic report, and
plots.

## Backend Boundary

`results/backend_check.json` records TabICL available = `true` and TabPFN
available = `true` (`8.0.8`).

`results/real_backend_smoke/` records a tiny real `TabICLRegressor` smoke. It
shows the backend is callable, but the sample is too small to support a paper
claim. TabPFN import works, but model fitting is blocked by `TabPFNLicenseError`
until the user supplies a license token (`TABPFN_TOKEN`) for model-weight
download in this non-interactive environment.

## Baseline Audit

The required retrieval baselines are retained: random, kNN, `matched_pair`, and
`matched_pair_diverse`.

Built-in full-data classical learners (`global_s_learner`, `global_t_learner`,
`global_x_learner`, and `global_dr_learner`) are treated as reference baselines
only. Not beating full-data learners is not a failure of the context-limited
retrieval question.

Optional external classical causal packages are unavailable in this environment:
`econml`, `causalml`, `sklift`, and `dowhy` all raise `ModuleNotFoundError`.
The blocker is recorded in `results/classical_dependency_check.json`.

## Claim To Use

The safe claim is:

> Causal-aware treated-control context retrieval improves PEHE for
> context-limited tabular ICL-style CATE estimation over random/kNN retrieval.
> `matched_pair_diverse` is the strongest synthetic method, while scaled IHDP
> and ACIC 2016 indicate the simpler `matched_pair` is the more robust external
> choice.

## What Not To Claim

- Do not claim a new causal PFN.
- Do not claim final TabICL/TabPFN model-level improvement.
- Do not claim `matched_pair_diverse` is uniformly best across external data.
- Do not claim full-data classical causal learners are beaten overall.
- Do not claim ATE/policy gains are as robust as PEHE gains.
- Do not mix these conclusions with experiment2 accuracy/AUC policy results.
