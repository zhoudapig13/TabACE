# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local_context_t_learner | matched_pair | 256 | 10 | 2.7025 | 0.2746 | 15.1742 | 0.5039 |
| local_context_t_learner | matched_pair_diverse | 256 | 10 | 2.8322 | 0.2724 | 15.1446 | 24.8095 |
| global_dr_learner | global_all_data | 672 | 10 | 2.9399 | 0.1921 | 15.1541 | 0.0013 |
| local_context_t_learner | knn | 256 | 10 | 2.9627 | 0.6587 | 15.1001 | 0.4997 |
| global_t_learner | global_all_data | 672 | 10 | 2.9671 | 0.1927 | 15.1683 | 0.0011 |
| global_x_learner | global_all_data | 672 | 10 | 2.9671 | 0.1927 | 15.1683 | 0.0016 |
| global_s_learner | global_all_data | 672 | 10 | 2.9673 | 0.1925 | 15.1683 | 0.0017 |
| local_context_t_learner | treatment_balanced_random | 256 | 10 | 3.0963 | 0.2758 | 15.0920 | 0.4772 |
| local_context_t_learner | matched_pair_diverse | 128 | 10 | 3.1497 | 0.2705 | 15.1217 | 5.4290 |
| local_context_t_learner | matched_pair | 128 | 10 | 3.1564 | 0.1554 | 15.0927 | 0.1752 |
| local_context_t_learner | random | 256 | 10 | 3.4237 | 0.1794 | 15.1007 | 0.4863 |
| local_context_t_learner | overlap_aware | 256 | 10 | 3.5009 | 0.3090 | 15.0822 | 0.5043 |
| local_context_t_learner | propensity_nearest | 256 | 10 | 3.6147 | 0.1774 | 15.0793 | 0.5126 |
| local_context_t_learner | treatment_balanced_random | 128 | 10 | 3.7496 | 0.1778 | 14.9911 | 0.1534 |
| local_context_t_learner | propensity_nearest | 128 | 10 | 3.9199 | 0.1935 | 15.0485 | 0.1736 |
| local_context_t_learner | knn | 64 | 10 | 4.1756 | 0.7289 | 14.8050 | 0.0840 |
| local_context_t_learner | knn | 128 | 10 | 4.2147 | 0.8578 | 14.8826 | 0.1693 |
| local_context_t_learner | random | 64 | 10 | 4.3589 | 0.5345 | 15.0600 | 0.0653 |
| local_context_t_learner | propensity_nearest | 64 | 10 | 4.5012 | 0.4666 | 14.9040 | 0.0892 |
| local_context_t_learner | matched_pair_diverse | 16 | 10 | 4.5352 | 0.2565 | 14.8849 | 0.1288 |
| local_context_t_learner | matched_pair | 64 | 10 | 4.7582 | 0.4947 | 14.8140 | 0.0915 |
| local_context_t_learner | matched_pair | 16 | 10 | 4.8865 | 0.2571 | 14.9004 | 0.0607 |
| local_context_t_learner | knn | 32 | 10 | 4.9066 | 0.9548 | 14.4927 | 0.0606 |
| local_context_t_learner | overlap_aware | 128 | 10 | 4.9886 | 0.5601 | 14.8503 | 0.1638 |
| local_context_t_learner | matched_pair_diverse | 32 | 10 | 5.1552 | 0.3781 | 14.7919 | 0.2739 |
| local_context_t_learner | random | 128 | 10 | 5.3441 | 0.3124 | 14.8501 | 0.1529 |
| local_context_t_learner | treatment_balanced_random | 64 | 10 | 5.5694 | 0.5595 | 14.6248 | 0.0693 |
| local_context_t_learner | treatment_balanced_random | 16 | 10 | 5.8299 | 0.5892 | 14.4864 | 0.0347 |
| local_context_t_learner | matched_pair_diverse | 64 | 10 | 5.9246 | 0.3225 | 14.7781 | 1.0084 |
| local_context_t_learner | random | 16 | 10 | 6.0838 | 0.8513 | 14.1383 | 0.0340 |
| local_context_t_learner | treatment_balanced_random | 32 | 10 | 6.5416 | 0.1570 | 14.6941 | 0.0417 |
| local_context_t_learner | overlap_aware | 64 | 10 | 6.7967 | 0.7194 | 14.6275 | 0.0837 |
| local_context_t_learner | propensity_nearest | 32 | 10 | 6.9444 | 1.8664 | 13.8984 | 0.0656 |
| local_context_t_learner | knn | 16 | 10 | 7.0003 | 2.0158 | 13.5289 | 0.0547 |
| local_context_t_learner | overlap_aware | 16 | 10 | 7.0150 | 1.5638 | 13.2802 | 0.0534 |
| local_context_t_learner | matched_pair | 32 | 10 | 7.1413 | 1.1123 | 14.4844 | 0.0680 |
| local_context_t_learner | propensity_nearest | 16 | 10 | 7.9513 | 2.4494 | 12.9382 | 0.0589 |
| local_context_t_learner | random | 32 | 10 | 8.9622 | 0.5688 | 14.1872 | 0.0402 |
| local_context_t_learner | overlap_aware | 32 | 10 | 10.6113 | 1.2320 | 13.8377 | 0.0618 |
