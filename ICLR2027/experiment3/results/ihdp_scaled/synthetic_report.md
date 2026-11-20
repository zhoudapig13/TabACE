# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local_context_t_learner | knn | 256 | 50 | 2.6414 | 0.6774 | 17.9661 | 0.5619 |
| global_dr_learner | global_all_data | 672 | 50 | 2.7291 | 0.2779 | 17.9665 | 0.0013 |
| global_t_learner | global_all_data | 672 | 50 | 2.7498 | 0.2886 | 17.9740 | 0.0011 |
| local_context_t_learner | matched_pair | 256 | 50 | 2.7883 | 0.3877 | 17.9815 | 0.5659 |
| local_context_t_learner | matched_pair_diverse | 256 | 50 | 2.8506 | 0.4011 | 17.9825 | 28.8443 |
| local_context_t_learner | random | 256 | 50 | 3.1957 | 0.2910 | 17.9462 | 0.5495 |
| local_context_t_learner | overlap_aware | 256 | 50 | 3.4934 | 0.5393 | 17.8977 | 0.5721 |
| local_context_t_learner | knn | 64 | 50 | 4.0552 | 0.6599 | 17.7273 | 0.0973 |
| local_context_t_learner | random | 64 | 50 | 4.1209 | 0.4624 | 17.7707 | 0.0752 |
| local_context_t_learner | matched_pair | 64 | 50 | 4.6852 | 0.4951 | 17.7289 | 0.1060 |
| local_context_t_learner | matched_pair_diverse | 16 | 50 | 4.9599 | 0.4547 | 17.5126 | 0.1470 |
| local_context_t_learner | matched_pair | 16 | 50 | 5.0213 | 0.6180 | 17.5706 | 0.0701 |
| local_context_t_learner | matched_pair_diverse | 64 | 50 | 6.1498 | 0.6257 | 17.5722 | 1.1714 |
| local_context_t_learner | random | 16 | 50 | 6.8009 | 1.1720 | 16.5935 | 0.0369 |
| local_context_t_learner | overlap_aware | 64 | 50 | 6.8314 | 1.0362 | 17.4116 | 0.0999 |
| local_context_t_learner | knn | 16 | 50 | 7.2482 | 1.5575 | 16.1844 | 0.0615 |
| local_context_t_learner | overlap_aware | 16 | 50 | 7.3324 | 1.0782 | 16.2958 | 0.0603 |
