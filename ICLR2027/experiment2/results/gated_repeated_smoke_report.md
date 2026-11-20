# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| validation_gated_feature | openml_cmc | 1 | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6294 (+0.0357) | 0.7861 -> 0.7835 (-0.0026) | acc 1/1, bacc 1/1, f1 1/1 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_repeated_feature | openml_cmc | 1 | 0.5933 -> 0.6000 (+0.0067) | 0.5933 -> 0.6000 (+0.0067) | 0.5937 -> 0.6020 (+0.0083) | 0.7861 -> 0.7777 (-0.0084) | acc 1/1, bacc 1/1, f1 1/1 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_feature | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_repeated_feature | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_feature | openml_kc1 | 1 | 0.7267 -> 0.7067 (-0.0200) | 0.7267 -> 0.7067 (-0.0200) | 0.7239 -> 0.7041 (-0.0198) | 0.8145 -> 0.8101 (-0.0044) | acc 0/1, bacc 0/1, f1 0/1 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_repeated_feature | openml_kc1 | 1 | 0.7267 -> 0.7000 (-0.0267) | 0.7267 -> 0.7000 (-0.0267) | 0.7239 -> 0.6961 (-0.0278) | 0.8145 -> 0.7901 (-0.0244) | acc 0/1, bacc 0/1, f1 0/1 | Choose a feature transform by averaging three internal validation splits |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_cmc | 42 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6294 (+0.0357) | 0.7861 -> 0.7835 (-0.0026) |
| openml_cmc | 42 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6000 (+0.0067) | 0.5933 -> 0.6000 (+0.0067) | 0.5937 -> 0.6020 (+0.0083) | 0.7861 -> 0.7777 (-0.0084) |
