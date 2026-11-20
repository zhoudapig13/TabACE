# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_s_learner | global_all_data | 388 | 120 | 0.7982 | 0.1928 | 2.6658 | 0.0004 |
| global_x_learner | global_all_data | 388 | 120 | 0.7983 | 0.1929 | 2.6658 | 0.0007 |
| global_t_learner | global_all_data | 388 | 120 | 0.7984 | 0.1929 | 2.6658 | 0.0004 |
| global_dr_learner | global_all_data | 388 | 120 | 0.8375 | 0.2161 | 2.6460 | 0.0005 |
| local_context_t_learner | matched_pair_diverse | 256 | 140 | 0.8906 | 0.2272 | 2.6118 | 3.5112 |
| local_context_t_learner | matched_pair | 256 | 140 | 0.9126 | 0.2540 | 2.6068 | 0.1861 |
| local_context_t_learner | treatment_balanced_random | 256 | 140 | 0.9675 | 0.2134 | 2.6150 | 0.1833 |
| local_context_t_learner | knn | 256 | 140 | 0.9823 | 0.3050 | 2.6074 | 0.1834 |
| local_context_t_learner | random | 256 | 140 | 0.9889 | 0.2246 | 2.6149 | 0.1832 |
| local_context_t_learner | propensity_nearest | 256 | 140 | 1.0503 | 0.3128 | 2.5752 | 0.1856 |
| local_context_t_learner | matched_pair_diverse | 128 | 140 | 1.0880 | 0.2836 | 2.5799 | 1.2721 |
| local_context_t_learner | matched_pair | 128 | 140 | 1.1522 | 0.3083 | 2.5654 | 0.0552 |
| local_context_t_learner | treatment_balanced_random | 128 | 140 | 1.2511 | 0.2399 | 2.5568 | 0.0514 |
| local_context_t_learner | random | 128 | 140 | 1.2913 | 0.2559 | 2.5497 | 0.0505 |
| local_context_t_learner | knn | 128 | 140 | 1.4195 | 0.3880 | 2.5236 | 0.0525 |
| global_s_learner | global_all_data | 328 | 20 | 1.5153 | 0.3328 | 2.3938 | 0.0012 |
| global_x_learner | global_all_data | 328 | 20 | 1.5154 | 0.3329 | 2.3938 | 0.0011 |
| global_t_learner | global_all_data | 328 | 20 | 1.5155 | 0.3329 | 2.3938 | 0.0006 |
| local_context_t_learner | matched_pair_diverse | 16 | 140 | 1.8802 | 0.5831 | 2.4911 | 0.0420 |
| local_context_t_learner | matched_pair_diverse | 64 | 140 | 1.8856 | 0.4126 | 2.4878 | 0.3876 |
| local_context_t_learner | propensity_nearest | 128 | 140 | 1.8886 | 0.5028 | 2.4136 | 0.0543 |
| local_context_t_learner | propensity_nearest | 64 | 140 | 1.8940 | 0.5697 | 2.2606 | 0.0263 |
| global_dr_learner | global_all_data | 328 | 20 | 1.9450 | 0.3768 | 2.3034 | 0.0009 |
| local_context_t_learner | matched_pair | 64 | 140 | 2.1014 | 0.5669 | 2.4470 | 0.0274 |
| local_context_t_learner | treatment_balanced_random | 64 | 140 | 2.1330 | 0.3267 | 2.4277 | 0.0237 |
| local_context_t_learner | matched_pair | 16 | 140 | 2.2135 | 0.6155 | 2.4235 | 0.0176 |
| local_context_t_learner | overlap_aware | 256 | 140 | 2.2768 | 0.5805 | 2.4834 | 0.1923 |
| local_context_t_learner | propensity_nearest | 16 | 140 | 2.3361 | 0.7944 | 1.8809 | 0.0165 |
| local_context_t_learner | propensity_nearest | 32 | 140 | 2.5221 | 0.7504 | 2.0122 | 0.0186 |
| local_context_t_learner | random | 64 | 140 | 2.6346 | 0.3834 | 2.3784 | 0.0225 |
| local_context_t_learner | treatment_balanced_random | 16 | 140 | 2.7501 | 1.1304 | 2.4557 | 0.0137 |
| local_context_t_learner | overlap_aware | 128 | 140 | 2.8373 | 0.7286 | 2.4225 | 0.0603 |
| local_context_t_learner | knn | 16 | 140 | 2.9055 | 0.9924 | 2.4326 | 0.0152 |
| local_context_t_learner | matched_pair_diverse | 32 | 140 | 3.1677 | 0.7267 | 2.2875 | 0.0948 |
| local_context_t_learner | random | 16 | 140 | 3.1789 | 1.0276 | 2.4075 | 0.0132 |
| local_context_t_learner | knn | 64 | 140 | 3.2928 | 0.6374 | 2.3966 | 0.0249 |
| local_context_t_learner | treatment_balanced_random | 32 | 140 | 3.6881 | 0.5654 | 2.1932 | 0.0159 |
| local_context_t_learner | matched_pair | 32 | 140 | 3.7011 | 0.7523 | 2.2285 | 0.0197 |
| local_context_t_learner | knn | 32 | 140 | 4.0265 | 0.9686 | 2.3216 | 0.0173 |
| local_context_t_learner | random | 32 | 140 | 4.0372 | 0.6862 | 2.2514 | 0.0148 |
