# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_s_learner | global_all_data | 156 | 1 | 0.4523 | 0.1530 | 2.6979 | 0.0004 |
| global_t_learner | global_all_data | 156 | 1 | 0.4526 | 0.1538 | 2.6979 | 0.0008 |
| local_context_t_learner | treatment_balanced_random | 32 | 1 | 1.6327 | 0.3202 | 2.3466 | 0.0075 |
| local_context_t_learner | propensity_nearest | 32 | 1 | 2.0217 | 0.4589 | 2.4940 | 0.0096 |
| local_context_t_learner | matched_pair | 32 | 1 | 2.0402 | 0.1631 | 2.2918 | 0.0098 |
| local_context_t_learner | random | 32 | 1 | 2.1256 | 0.2845 | 2.2281 | 0.0069 |
| local_context_t_learner | propensity_nearest | 16 | 1 | 2.1548 | 0.5234 | 2.1690 | 0.0154 |
| local_context_t_learner | knn | 16 | 1 | 3.1673 | 1.8137 | 2.6335 | 0.0079 |
| local_context_t_learner | knn | 32 | 1 | 3.1952 | 0.3701 | 2.4292 | 0.0081 |
| local_context_t_learner | matched_pair | 16 | 1 | 3.5104 | 0.3660 | 2.2088 | 0.0097 |
| local_context_t_learner | overlap_aware | 16 | 1 | 3.7592 | 0.0813 | 2.0768 | 0.0159 |
| local_context_t_learner | overlap_aware | 32 | 1 | 4.0143 | 1.0991 | 2.5114 | 0.0157 |
| local_context_t_learner | random | 16 | 1 | 4.8474 | 2.0224 | 2.3283 | 0.0067 |
| local_context_t_learner | treatment_balanced_random | 16 | 1 | 5.8851 | 1.3093 | 2.1229 | 0.0071 |
