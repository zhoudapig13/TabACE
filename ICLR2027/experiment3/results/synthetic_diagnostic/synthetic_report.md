# Experiment 3 Synthetic Summary

Lower PEHE and ATE error are better. Higher policy value and uplift are better.

| Estimator | Retriever | Context | Runs | PEHE | ATE Error | Policy Value | Runtime Sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local_context_t_learner | matched_pair_diverse | 256 | 40 | 0.8285 | 0.2586 | 2.7313 | 3.3144 |
| local_context_t_learner | matched_pair_propensity | 256 | 40 | 0.8566 | 0.2686 | 2.7306 | 0.0178 |
| local_context_t_learner | matched_pair | 256 | 40 | 0.8569 | 0.2986 | 2.7281 | 0.0174 |
| local_context_t_learner | matched_pair_hybrid_w0.2 | 256 | 40 | 0.8569 | 0.2986 | 2.7281 | 0.0177 |
| local_context_t_learner | matched_pair_hybrid_w0.05 | 256 | 40 | 0.8633 | 0.2883 | 2.7290 | 0.0178 |
| local_context_t_learner | matched_pair_hybrid_w0.5 | 256 | 40 | 0.8652 | 0.3131 | 2.7294 | 0.0177 |
| global_s_learner | global_all_data | 388 | 40 | 0.8767 | 0.2349 | 2.7510 | 0.0004 |
| global_x_learner | global_all_data | 388 | 40 | 0.8767 | 0.2350 | 2.7510 | 0.0007 |
| global_t_learner | global_all_data | 388 | 40 | 0.8768 | 0.2350 | 2.7510 | 0.0004 |
| local_context_t_learner | overlap_filtered_matched_pair | 256 | 40 | 0.8836 | 0.3035 | 2.7267 | 0.0272 |
| local_context_t_learner | matched_pair_feature | 256 | 40 | 0.9121 | 0.3357 | 2.7351 | 0.0177 |
| global_dr_learner | global_all_data | 388 | 40 | 0.9239 | 0.2860 | 2.7229 | 0.0005 |
| local_context_t_learner | matched_pair_diverse | 128 | 40 | 0.9291 | 0.2910 | 2.6929 | 1.1964 |
| local_context_t_learner | treatment_balanced_random | 256 | 40 | 0.9332 | 0.2334 | 2.7413 | 0.0141 |
| local_context_t_learner | random | 256 | 40 | 0.9380 | 0.2465 | 2.7451 | 0.0131 |
| local_context_t_learner | knn | 256 | 40 | 0.9527 | 0.3690 | 2.7303 | 0.0152 |
| local_context_t_learner | matched_pair | 128 | 40 | 1.0069 | 0.3128 | 2.6836 | 0.0155 |
| local_context_t_learner | matched_pair_hybrid_w0.2 | 128 | 40 | 1.0069 | 0.3128 | 2.6836 | 0.0159 |
| local_context_t_learner | matched_pair_propensity | 128 | 40 | 1.0206 | 0.3448 | 2.6666 | 0.0159 |
| local_context_t_learner | matched_pair_hybrid_w0.5 | 128 | 40 | 1.0253 | 0.3285 | 2.6702 | 0.0158 |
| local_context_t_learner | matched_pair_hybrid_w0.05 | 128 | 40 | 1.0313 | 0.3423 | 2.6696 | 0.0158 |
| local_context_t_learner | matched_pair_caliper_c0.05 | 256 | 40 | 1.0588 | 0.2997 | 2.6965 | 0.0184 |
| local_context_t_learner | propensity_nearest | 256 | 40 | 1.0982 | 0.4224 | 2.6711 | 0.0167 |
| local_context_t_learner | treatment_balanced_random | 128 | 40 | 1.1190 | 0.2569 | 2.6833 | 0.0122 |
| local_context_t_learner | matched_pair_feature | 128 | 40 | 1.1215 | 0.3953 | 2.7027 | 0.0158 |
| local_context_t_learner | matched_pair_caliper_c0.1 | 256 | 40 | 1.1337 | 0.3000 | 2.6712 | 0.0186 |
| local_context_t_learner | random | 128 | 40 | 1.1725 | 0.2593 | 2.6878 | 0.0113 |
| local_context_t_learner | matched_pair_caliper_c0.2 | 256 | 40 | 1.1795 | 0.2795 | 2.6559 | 0.0193 |
| local_context_t_learner | overlap_filtered_matched_pair | 128 | 40 | 1.2325 | 0.3397 | 2.6817 | 0.0259 |
| local_context_t_learner | knn | 128 | 40 | 1.3017 | 0.4274 | 2.6771 | 0.0135 |
| local_context_t_learner | matched_pair_diverse | 64 | 40 | 1.3245 | 0.3506 | 2.6229 | 0.3564 |
| local_context_t_learner | overlap_aware | 256 | 40 | 1.3863 | 0.3996 | 2.6493 | 0.0229 |
| local_context_t_learner | matched_pair_caliper_c0.05 | 128 | 40 | 1.4861 | 0.4035 | 2.6240 | 0.0166 |
| local_context_t_learner | treatment_balanced_random | 64 | 40 | 1.5562 | 0.2821 | 2.5660 | 0.0112 |
| local_context_t_learner | matched_pair | 64 | 40 | 1.5939 | 0.5244 | 2.5574 | 0.0147 |
| local_context_t_learner | matched_pair_hybrid_w0.2 | 64 | 40 | 1.5939 | 0.5244 | 2.5574 | 0.0150 |
| local_context_t_learner | matched_pair_hybrid_w0.5 | 64 | 40 | 1.6041 | 0.4718 | 2.5631 | 0.0150 |
| local_context_t_learner | matched_pair_feature | 64 | 40 | 1.6277 | 0.4266 | 2.6145 | 0.0150 |
| local_context_t_learner | matched_pair_hybrid_w0.05 | 64 | 40 | 1.6305 | 0.4973 | 2.5862 | 0.0150 |
| local_context_t_learner | matched_pair_propensity | 64 | 40 | 1.6488 | 0.4884 | 2.5634 | 0.0150 |
