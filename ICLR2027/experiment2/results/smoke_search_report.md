# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| causal_kbest | synth_selection_hard_overlap | 1 | 0.8667 -> 0.8800 (+0.0133) | 0.8659 -> 0.8794 (+0.0135) | 0.8661 -> 0.8797 (+0.0136) | 0.9253 -> 0.9523 (+0.0270) | acc 1/1, bacc 1/1, f1 1/1 | Keep top causal-proxy features |
| causal_kbest | synth_rare_boundary | 1 | 0.8933 -> 0.9067 (+0.0133) | 0.7835 -> 0.7916 (+0.0081) | 0.8016 -> 0.8203 (+0.0187) | 0.8958 -> 0.9094 (+0.0136) | acc 1/1, bacc 1/1, f1 1/1 | Keep top causal-proxy features |
| linear_probe_feature | breast_cancer | 1 | 0.9790 -> 0.9860 (+0.0070) | 0.9756 -> 0.9850 (+0.0094) | 0.9774 -> 0.9850 (+0.0076) | 0.9973 -> 0.9977 (+0.0004) | acc 1/1, bacc 1/1, f1 1/1 | Append balanced linear probe uncertainty and class-probability features |
| quantile_features | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9795 (+0.0039) | 0.9774 -> 0.9776 (+0.0002) | 0.9973 -> 0.9977 (+0.0004) | acc 0/1, bacc 1/1, f1 1/1 | Quantile-normal feature transform before TabICL |
| rare_class_oversample | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Oversample low-frequency classes in the training context |
| knn_label_feature | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9979 (+0.0006) | acc 0/1, bacc 0/1, f1 0/1 | Append leave-one-out KNN label distribution features |
| rare_class_oversample | synth_selection_hard_overlap | 1 | 0.8667 -> 0.8667 (+0.0000) | 0.8659 -> 0.8659 (+0.0000) | 0.8661 -> 0.8661 (+0.0000) | 0.9253 -> 0.9253 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Oversample low-frequency classes in the training context |
| knn_label_feature | synth_rare_boundary | 1 | 0.8933 -> 0.8933 (+0.0000) | 0.7835 -> 0.7835 (+0.0000) | 0.8016 -> 0.8016 (+0.0000) | 0.8958 -> 0.8908 (-0.0050) | acc 0/1, bacc 0/1, f1 0/1 | Append leave-one-out KNN label distribution features |
| quantile_features | synth_selection_hard_overlap | 1 | 0.8667 -> 0.8533 (-0.0133) | 0.8659 -> 0.8531 (-0.0128) | 0.8661 -> 0.8532 (-0.0128) | 0.9253 -> 0.9161 (-0.0092) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normal feature transform before TabICL |
| quantile_features | synth_rare_boundary | 1 | 0.8933 -> 0.8800 (-0.0133) | 0.7835 -> 0.7754 (-0.0081) | 0.8016 -> 0.7840 (-0.0176) | 0.8958 -> 0.8908 (-0.0050) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normal feature transform before TabICL |
| causal_kbest | breast_cancer | 1 | 0.9790 -> 0.9580 (-0.0210) | 0.9756 -> 0.9473 (-0.0283) | 0.9774 -> 0.9543 (-0.0231) | 0.9973 -> 0.9954 (-0.0019) | acc 0/1, bacc 0/1, f1 0/1 | Keep top causal-proxy features |
| rare_class_oversample | synth_rare_boundary | 1 | 0.8933 -> 0.8667 (-0.0267) | 0.7835 -> 0.6762 (-0.1073) | 0.8016 -> 0.7115 (-0.0900) | 0.8958 -> 0.9032 (+0.0074) | acc 0/1, bacc 0/1, f1 0/1 | Oversample low-frequency classes in the training context |
| linear_probe_feature | synth_rare_boundary | 1 | 0.8933 -> 0.8400 (-0.0533) | 0.7835 -> 0.7512 (-0.0323) | 0.8016 -> 0.7365 (-0.0651) | 0.8958 -> 0.8400 (-0.0558) | acc 0/1, bacc 0/1, f1 0/1 | Append balanced linear probe uncertainty and class-probability features |
| knn_label_feature | synth_selection_hard_overlap | 1 | 0.8667 -> 0.7600 (-0.1067) | 0.8659 -> 0.7589 (-0.1070) | 0.8661 -> 0.7579 (-0.1082) | 0.9253 -> 0.8990 (-0.0263) | acc 0/1, bacc 0/1, f1 0/1 | Append leave-one-out KNN label distribution features |
| linear_probe_feature | synth_selection_hard_overlap | 1 | 0.8667 -> 0.7467 (-0.1200) | 0.8659 -> 0.7454 (-0.1206) | 0.8661 -> 0.7438 (-0.1223) | 0.9253 -> 0.8514 (-0.0740) | acc 0/1, bacc 0/1, f1 0/1 | Append balanced linear probe uncertainty and class-probability features |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| synth_selection_hard_overlap | 42 | causal_kbest | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8667 -> 0.8800 (+0.0133) | 0.8659 -> 0.8794 (+0.0135) | 0.8661 -> 0.8797 (+0.0136) | 0.9253 -> 0.9523 (+0.0270) |
| synth_rare_boundary | 42 | causal_kbest | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8933 -> 0.9067 (+0.0133) | 0.7835 -> 0.7916 (+0.0081) | 0.8016 -> 0.8203 (+0.0187) | 0.8958 -> 0.9094 (+0.0136) |
| breast_cancer | 42 | linear_probe_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9790 -> 0.9860 (+0.0070) | 0.9756 -> 0.9850 (+0.0094) | 0.9774 -> 0.9850 (+0.0076) | 0.9973 -> 0.9977 (+0.0004) |
| synth_rare_boundary | 42 | rare_class_oversample | roc_auc | 0.8933 -> 0.8667 (-0.0267) | 0.7835 -> 0.6762 (-0.1073) | 0.8016 -> 0.7115 (-0.0900) | 0.8958 -> 0.9032 (+0.0074) |
| breast_cancer | 42 | quantile_features | balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9795 (+0.0039) | 0.9774 -> 0.9776 (+0.0002) | 0.9973 -> 0.9977 (+0.0004) |
| breast_cancer | 42 | knn_label_feature | roc_auc | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9979 (+0.0006) |
