# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_s_learner | global_all_data | 520 | 12 | 0.5946 | 0.1957 | 2.2861 | 0.0005 |
| global_t_learner | global_all_data | 520 | 12 | 0.5948 | 0.1960 | 2.2861 | 0.0005 |
| local_context_t_learner | matched_pair | 256 | 12 | 0.6303 | 0.2110 | 2.2861 | 0.0352 |
| local_context_t_learner | knn | 256 | 12 | 0.6557 | 0.1586 | 2.2716 | 0.0299 |
| local_context_t_learner | treatment_balanced_random | 256 | 12 | 0.6559 | 0.1909 | 2.2830 | 0.0249 |
| local_context_t_learner | random | 256 | 12 | 0.6922 | 0.1849 | 2.2800 | 0.0221 |
| local_context_t_learner | overlap_aware | 256 | 12 | 0.6949 | 0.2336 | 2.2530 | 0.0487 |
| local_context_t_learner | matched_pair | 128 | 12 | 0.6978 | 0.2113 | 2.2714 | 0.0330 |
| local_context_t_learner | treatment_balanced_random | 128 | 12 | 0.8499 | 0.2099 | 2.2669 | 0.0218 |
| local_context_t_learner | propensity_nearest | 256 | 12 | 0.9424 | 0.2376 | 2.2462 | 0.0342 |
| local_context_t_learner | matched_pair | 64 | 12 | 0.9889 | 0.2491 | 2.2227 | 0.0317 |
| local_context_t_learner | random | 128 | 12 | 0.9890 | 0.2913 | 2.2385 | 0.0194 |
| local_context_t_learner | knn | 128 | 12 | 1.0196 | 0.2030 | 2.2229 | 0.0274 |
| local_context_t_learner | overlap_aware | 128 | 12 | 1.2041 | 0.1861 | 2.1423 | 0.0454 |
| local_context_t_learner | treatment_balanced_random | 64 | 12 | 1.2427 | 0.2191 | 2.1936 | 0.0202 |
| local_context_t_learner | propensity_nearest | 128 | 12 | 1.3875 | 0.1440 | 2.1848 | 0.0317 |
| local_context_t_learner | random | 64 | 12 | 1.4934 | 0.2509 | 2.1384 | 0.0178 |
| local_context_t_learner | propensity_nearest | 64 | 12 | 1.7548 | 0.2034 | 2.0725 | 0.0303 |
| local_context_t_learner | knn | 64 | 12 | 1.9557 | 0.2203 | 2.0886 | 0.0260 |
| local_context_t_learner | propensity_nearest | 32 | 12 | 1.9748 | 0.2971 | 1.9628 | 0.0295 |
| local_context_t_learner | matched_pair | 32 | 12 | 2.0764 | 0.3754 | 2.0486 | 0.0310 |
| local_context_t_learner | propensity_nearest | 16 | 12 | 2.4320 | 0.5953 | 1.7523 | 0.0298 |
| local_context_t_learner | treatment_balanced_random | 32 | 12 | 2.4444 | 0.3166 | 1.9653 | 0.0192 |
| local_context_t_learner | matched_pair | 16 | 12 | 2.4821 | 0.3599 | 2.0040 | 0.0302 |
| local_context_t_learner | knn | 32 | 12 | 2.8569 | 0.5267 | 2.0014 | 0.0251 |
| local_context_t_learner | random | 32 | 12 | 3.1598 | 0.3514 | 1.9492 | 0.0169 |
| local_context_t_learner | knn | 16 | 12 | 3.2386 | 0.8294 | 1.9915 | 0.0249 |
| local_context_t_learner | overlap_aware | 64 | 12 | 3.2590 | 1.1597 | 1.9667 | 0.0440 |
| local_context_t_learner | treatment_balanced_random | 16 | 12 | 3.3198 | 0.8223 | 1.9711 | 0.0182 |
| local_context_t_learner | overlap_aware | 32 | 12 | 3.6046 | 1.3567 | 2.0041 | 0.0429 |
| local_context_t_learner | random | 16 | 12 | 4.1702 | 0.6410 | 1.9195 | 0.0172 |
| local_context_t_learner | overlap_aware | 16 | 12 | 4.5771 | 1.3600 | 1.8186 | 0.0424 |
