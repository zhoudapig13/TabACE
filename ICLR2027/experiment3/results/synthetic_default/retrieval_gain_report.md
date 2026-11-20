# Retrieval Gain Report

Diagnostic synthetic run with 4 named regimes, 10 seeds each, and context sizes 16/32/64/128/256. Negative PEHE/ATE/policy-regret deltas are improvements; positive policy-value deltas are improvements.

## Overall vs `random` Context

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 200 | 162/200 | -0.6304 | +21.13% | 96/200 | -0.0615 | 100/200 | +0.0328 |
| `propensity_nearest` | 200 | 108/200 | -0.4925 | -2.52% | 76/200 | +0.0932 | 31/200 | -0.2406 |
| `matched_pair_propensity` | 200 | 144/200 | -0.3673 | +9.43% | 99/200 | -0.0066 | 93/200 | -0.0160 |
| `matched_pair_hybrid_w0.05` | 200 | 138/200 | -0.3515 | +8.85% | 100/200 | -0.0019 | 99/200 | -0.0046 |
| `matched_pair` | 200 | 137/200 | -0.3480 | +9.55% | 95/200 | +0.0042 | 97/200 | +0.0010 |
| `matched_pair_hybrid_w0.2` | 200 | 137/200 | -0.3480 | +9.55% | 95/200 | +0.0042 | 97/200 | +0.0010 |
| `matched_pair_hybrid_w0.5` | 200 | 134/200 | -0.3247 | +8.59% | 89/200 | -0.0056 | 97/200 | +0.0046 |
| `matched_pair_caliper_c0.05` | 200 | 98/200 | -0.3020 | -3.87% | 91/200 | -0.0171 | 75/200 | -0.0480 |
| `matched_pair_feature` | 200 | 130/200 | -0.2946 | +6.51% | 85/200 | +0.0777 | 112/200 | +0.0443 |
| `treatment_balanced_random` | 200 | 127/200 | -0.2769 | +6.35% | 98/200 | -0.0121 | 94/200 | +0.0045 |
| `matched_pair_caliper_c0.2` | 200 | 88/200 | -0.1220 | -19.90% | 86/200 | +0.0138 | 89/200 | -0.0095 |
| `matched_pair_caliper_c0.1` | 200 | 86/200 | -0.1115 | -16.31% | 81/200 | +0.0202 | 82/200 | -0.0214 |

## Overall vs `knn` Context

| Method | Runs | PEHE Wins | Mean PEHE Delta | Rel. PEHE Reduction | ATE Wins | Mean ATE Delta | Policy Wins | Mean Policy Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_pair_diverse` | 200 | 167/200 | -0.6483 | +23.60% | 133/200 | -0.2067 | 99/200 | +0.0085 |
| `propensity_nearest` | 200 | 119/200 | -0.5104 | +3.36% | 101/200 | -0.0520 | 28/200 | -0.2649 |
| `matched_pair_propensity` | 200 | 151/200 | -0.3853 | +13.68% | 123/200 | -0.1517 | 85/200 | -0.0403 |
| `matched_pair_hybrid_w0.05` | 200 | 150/200 | -0.3695 | +12.83% | 124/200 | -0.1471 | 87/200 | -0.0289 |
| `matched_pair` | 200 | 150/200 | -0.3660 | +13.24% | 128/200 | -0.1410 | 96/200 | -0.0232 |
| `matched_pair_hybrid_w0.2` | 200 | 150/200 | -0.3660 | +13.24% | 128/200 | -0.1410 | 96/200 | -0.0232 |
| `matched_pair_hybrid_w0.5` | 200 | 152/200 | -0.3427 | +12.25% | 125/200 | -0.1508 | 88/200 | -0.0197 |
| `matched_pair_caliper_c0.05` | 200 | 107/200 | -0.3200 | +0.60% | 124/200 | -0.1622 | 68/200 | -0.0723 |
| `matched_pair_feature` | 200 | 146/200 | -0.3126 | +10.42% | 115/200 | -0.0675 | 98/200 | +0.0201 |
| `treatment_balanced_random` | 200 | 124/200 | -0.2948 | +8.66% | 128/200 | -0.1572 | 89/200 | -0.0198 |
| `matched_pair_caliper_c0.2` | 200 | 99/200 | -0.1399 | -12.47% | 122/200 | -0.1314 | 80/200 | -0.0338 |
| `matched_pair_caliper_c0.1` | 200 | 100/200 | -0.1295 | -9.53% | 119/200 | -0.1249 | 77/200 | -0.0457 |

## Bootstrap Evidence

| Method | Baseline | Metric | Mean Delta | 95% Bootstrap CI | p approx |
| --- | --- | --- | ---: | ---: | ---: |
| `matched_pair` | `knn` | `pehe` | -0.3660 | [-0.5287, -0.2178] | 6.696e-06 |
| `matched_pair` | `knn` | `ate_error` | -0.1410 | [-0.2213, -0.0581] | 0.0006254 |
| `matched_pair` | `knn` | `policy_value` | -0.0232 | [-0.0477, +0.0004] | 0.05895 |
| `matched_pair` | `random` | `pehe` | -0.3480 | [-0.4921, -0.1955] | 4.503e-06 |
| `matched_pair` | `random` | `ate_error` | +0.0042 | [-0.0884, +0.1028] | 0.9327 |
| `matched_pair` | `random` | `policy_value` | +0.0010 | [-0.0222, +0.0248] | 0.9318 |
| `matched_pair_diverse` | `knn` | `pehe` | -0.6483 | [-0.7944, -0.5165] | 2.908e-18 |
| `matched_pair_diverse` | `knn` | `ate_error` | -0.2067 | [-0.2886, -0.1302] | 1.757e-07 |
| `matched_pair_diverse` | `knn` | `policy_value` | +0.0085 | [-0.0160, +0.0338] | 0.4956 |
| `matched_pair_diverse` | `random` | `pehe` | -0.6304 | [-0.7613, -0.4974] | 7.651e-21 |
| `matched_pair_diverse` | `random` | `ate_error` | -0.0615 | [-0.1359, +0.0070] | 0.1198 |
| `matched_pair_diverse` | `random` | `policy_value` | +0.0328 | [+0.0110, +0.0550] | 0.003844 |
| `overlap_aware` | `knn` | `pehe` | +1.0627 | [+0.6835, +1.5793] | 1.748e-06 |
| `overlap_aware` | `knn` | `ate_error` | +0.5346 | [+0.3318, +0.7364] | 8.65e-08 |
| `overlap_aware` | `knn` | `policy_value` | -0.1039 | [-0.1402, -0.0695] | 6.515e-09 |
| `overlap_aware` | `random` | `pehe` | +1.0806 | [+0.6724, +1.5616] | 7.137e-07 |
| `overlap_aware` | `random` | `ate_error` | +0.6798 | [+0.4799, +0.8682] | 5.66e-12 |
| `overlap_aware` | `random` | `policy_value` | -0.0796 | [-0.1172, -0.0438] | 1.704e-05 |
| `overlap_filtered_matched_pair` | `knn` | `pehe` | +0.4682 | [+0.1779, +0.7793] | 0.002097 |
| `overlap_filtered_matched_pair` | `knn` | `ate_error` | +0.2236 | [+0.0768, +0.3825] | 0.004263 |
| `overlap_filtered_matched_pair` | `knn` | `policy_value` | -0.0443 | [-0.0683, -0.0196] | 0.0004835 |
| `overlap_filtered_matched_pair` | `random` | `pehe` | +0.4861 | [+0.2319, +0.7753] | 0.000371 |
| `overlap_filtered_matched_pair` | `random` | `ate_error` | +0.3687 | [+0.1947, +0.5567] | 1.905e-05 |
| `overlap_filtered_matched_pair` | `random` | `policy_value` | -0.0200 | [-0.0439, +0.0046] | 0.1138 |

## Main Takeaways

- `matched_pair_diverse` is the new best retrieval variant: it improves PEHE by about 21.1% vs random and 23.6% vs kNN on average, with PEHE wins in 162/200 and 167/200 paired comparisons.
- The original `matched_pair` remains a strong, simpler method: PEHE wins in 137/200 vs random and 150/200 vs kNN, with bootstrap CIs below zero.
- ATE improves clearly vs kNN for `matched_pair` and `matched_pair_diverse`, but remains unstable vs random. This supports PEHE as the main claim and ATE as a secondary, scenario-dependent result.
- `overlap_aware` and `overlap_filtered_matched_pair` are negative results. Common-support filtering without preserving local similarity hurts PEHE in this setup.
- Full-data global causal learners remain upper/reference baselines, not retrieval competitors. The retrieval claim is for context-limited ICL-style estimation.

## Best Aggregate Rows

| Estimator | Retriever | Context | PEHE | ATE Error | Policy Value | Policy Regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `local_context_t_learner` | `matched_pair_diverse` | 256 | 0.8285 | 0.2586 | 2.7313 | 0.0392 |
| `local_context_t_learner` | `matched_pair_propensity` | 256 | 0.8566 | 0.2686 | 2.7306 | 0.0399 |
| `local_context_t_learner` | `matched_pair` | 256 | 0.8569 | 0.2986 | 2.7281 | 0.0424 |
| `local_context_t_learner` | `matched_pair_hybrid_w0.2` | 256 | 0.8569 | 0.2986 | 2.7281 | 0.0424 |
| `local_context_t_learner` | `matched_pair_hybrid_w0.05` | 256 | 0.8633 | 0.2883 | 2.7290 | 0.0416 |
| `local_context_t_learner` | `matched_pair_hybrid_w0.5` | 256 | 0.8652 | 0.3131 | 2.7294 | 0.0411 |
| `global_s_learner` | `global_all_data` | 388 | 0.8767 | 0.2349 | 2.7510 | 0.0195 |
| `global_x_learner` | `global_all_data` | 388 | 0.8767 | 0.2350 | 2.7510 | 0.0195 |
| `global_t_learner` | `global_all_data` | 388 | 0.8768 | 0.2350 | 2.7510 | 0.0195 |
| `local_context_t_learner` | `overlap_filtered_matched_pair` | 256 | 0.8836 | 0.3035 | 2.7267 | 0.0438 |
| `local_context_t_learner` | `matched_pair_feature` | 256 | 0.9121 | 0.3357 | 2.7351 | 0.0354 |
| `global_dr_learner` | `global_all_data` | 388 | 0.9239 | 0.2860 | 2.7229 | 0.0476 |
| `local_context_t_learner` | `matched_pair_diverse` | 128 | 0.9291 | 0.2910 | 2.6929 | 0.0777 |
| `local_context_t_learner` | `treatment_balanced_random` | 256 | 0.9332 | 0.2334 | 2.7413 | 0.0292 |
| `local_context_t_learner` | `random` | 256 | 0.9380 | 0.2465 | 2.7451 | 0.0254 |

Note: this report was refreshed from `results/synthetic_diagnostic/`, the current multi-regime 10-seed diagnostic run.
