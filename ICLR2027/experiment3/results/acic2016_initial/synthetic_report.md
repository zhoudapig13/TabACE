# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_t_learner | global_all_data | 600 | 3 | 2.6486 | 0.3379 | 6.1510 | 0.0033 |
| global_dr_learner | global_all_data | 600 | 3 | 2.6506 | 0.3351 | 6.1510 | 0.0049 |
| local_context_t_learner | matched_pair | 256 | 3 | 3.1818 | 0.3769 | 6.0524 | 2.9989 |
| local_context_t_learner | matched_pair_diverse | 64 | 3 | 3.7982 | 0.5663 | 5.8486 | 2.3028 |
| local_context_t_learner | random | 256 | 3 | 3.8146 | 0.3770 | 5.9938 | 2.9820 |
| local_context_t_learner | overlap_aware | 256 | 3 | 4.0922 | 1.0229 | 5.9138 | 3.0406 |
| local_context_t_learner | matched_pair_diverse | 16 | 3 | 4.2465 | 1.3663 | 5.7095 | 0.2514 |
| local_context_t_learner | matched_pair | 16 | 3 | 4.2781 | 0.5278 | 5.3325 | 0.1531 |
| local_context_t_learner | knn | 16 | 3 | 4.3930 | 0.2839 | 5.1211 | 0.1436 |
| local_context_t_learner | matched_pair | 64 | 3 | 4.6488 | 1.2996 | 5.7698 | 0.2456 |
| local_context_t_learner | overlap_aware | 16 | 3 | 4.8620 | 0.4035 | 4.8247 | 0.1425 |
| local_context_t_learner | matched_pair_diverse | 256 | 3 | 5.0056 | 0.5391 | 5.8697 | 64.1891 |
| local_context_t_learner | knn | 64 | 3 | 5.6482 | 0.7391 | 5.4118 | 0.2335 |
| local_context_t_learner | random | 16 | 3 | 5.9120 | 0.8158 | 4.8810 | 0.1144 |
| local_context_t_learner | knn | 256 | 3 | 5.9324 | 0.7042 | 5.6207 | 2.9833 |
| local_context_t_learner | overlap_aware | 64 | 3 | 8.1642 | 1.9577 | 5.1320 | 0.2218 |
| local_context_t_learner | random | 64 | 3 | 8.1695 | 1.1661 | 5.4642 | 0.2090 |
