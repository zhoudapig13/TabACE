# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| propensity_quantile_domain_feature | openml_ilpd | 1 | 0.6712 -> 0.7055 (+0.0342) | 0.5634 -> 0.6087 (+0.0453) | 0.5656 -> 0.6143 (+0.0487) | 0.7337 -> 0.7299 (-0.0039) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-normalize features and append domain-shift plus propensity features |
| domain_shift_feature | openml_ilpd | 1 | 0.6712 -> 0.7055 (+0.0342) | 0.5634 -> 0.6158 (+0.0524) | 0.5656 -> 0.6208 (+0.0552) | 0.7337 -> 0.7138 (-0.0199) | acc 1/1, bacc 1/1, f1 1/1 | Append train/test domain-shift probability features |
| validation_gated_feature | openml_cmc | 1 | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6294 (+0.0357) | 0.7861 -> 0.7835 (-0.0026) | acc 1/1, bacc 1/1, f1 1/1 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_feature | openml_ilpd | 1 | 0.6712 -> 0.6986 (+0.0274) | 0.5634 -> 0.6110 (+0.0476) | 0.5656 -> 0.6151 (+0.0496) | 0.7337 -> 0.7283 (-0.0055) | acc 1/1, bacc 1/1, f1 1/1 | Choose identity or a candidate feature transform using an internal validation split |
| quantile_domain_probe_feature | openml_ilpd | 1 | 0.6712 -> 0.6918 (+0.0205) | 0.5634 -> 0.5991 (+0.0357) | 0.5656 -> 0.6031 (+0.0376) | 0.7337 -> 0.7319 (-0.0018) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-normalize features and append domain-shift plus linear-probe features |
| quantile_domain_probe_feature | openml_cmc | 1 | 0.5933 -> 0.6067 (+0.0133) | 0.5933 -> 0.6067 (+0.0133) | 0.5937 -> 0.6087 (+0.0151) | 0.7861 -> 0.7834 (-0.0027) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-normalize features and append domain-shift plus linear-probe features |
| domain_shift_feature | openml_cmc | 1 | 0.5933 -> 0.6067 (+0.0133) | 0.5933 -> 0.6067 (+0.0133) | 0.5937 -> 0.6081 (+0.0144) | 0.7861 -> 0.7918 (+0.0057) | acc 1/1, bacc 1/1, f1 1/1 | Append train/test domain-shift probability features |
| propensity_quantile_domain_feature | openml_cmc | 1 | 0.5933 -> 0.6000 (+0.0067) | 0.5933 -> 0.6000 (+0.0067) | 0.5937 -> 0.6020 (+0.0083) | 0.7861 -> 0.7777 (-0.0084) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-normalize features and append domain-shift plus propensity features |
| validation_gated_feature | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Choose identity or a candidate feature transform using an internal validation split |
| domain_shift_feature | breast_cancer | 1 | 0.9790 -> 0.9720 (-0.0070) | 0.9756 -> 0.9700 (-0.0056) | 0.9774 -> 0.9700 (-0.0074) | 0.9973 -> 0.9966 (-0.0006) | acc 0/1, bacc 0/1, f1 0/1 | Append train/test domain-shift probability features |
| quantile_domain_probe_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9683 (-0.0072) | 0.9774 -> 0.9629 (-0.0145) | 0.9973 -> 0.9964 (-0.0008) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append domain-shift plus linear-probe features |
| propensity_quantile_domain_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9645 (-0.0111) | 0.9774 -> 0.9627 (-0.0148) | 0.9973 -> 0.9971 (-0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append domain-shift plus propensity features |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_ilpd | 42 | domain_shift_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6712 -> 0.7055 (+0.0342) | 0.5634 -> 0.6158 (+0.0524) | 0.5656 -> 0.6208 (+0.0552) | 0.7337 -> 0.7138 (-0.0199) |
| openml_ilpd | 42 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6712 -> 0.6986 (+0.0274) | 0.5634 -> 0.6110 (+0.0476) | 0.5656 -> 0.6151 (+0.0496) | 0.7337 -> 0.7283 (-0.0055) |
| openml_ilpd | 42 | propensity_quantile_domain_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6712 -> 0.7055 (+0.0342) | 0.5634 -> 0.6087 (+0.0453) | 0.5656 -> 0.6143 (+0.0487) | 0.7337 -> 0.7299 (-0.0039) |
| openml_ilpd | 42 | quantile_domain_probe_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6712 -> 0.6918 (+0.0205) | 0.5634 -> 0.5991 (+0.0357) | 0.5656 -> 0.6031 (+0.0376) | 0.7337 -> 0.7319 (-0.0018) |
| openml_cmc | 42 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6294 (+0.0357) | 0.7861 -> 0.7835 (-0.0026) |
| openml_cmc | 42 | quantile_domain_probe_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6067 (+0.0133) | 0.5933 -> 0.6067 (+0.0133) | 0.5937 -> 0.6087 (+0.0151) | 0.7861 -> 0.7834 (-0.0027) |
| openml_cmc | 42 | domain_shift_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5933 -> 0.6067 (+0.0133) | 0.5933 -> 0.6067 (+0.0133) | 0.5937 -> 0.6081 (+0.0144) | 0.7861 -> 0.7918 (+0.0057) |
| openml_cmc | 42 | propensity_quantile_domain_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6000 (+0.0067) | 0.5933 -> 0.6000 (+0.0067) | 0.5937 -> 0.6020 (+0.0083) | 0.7861 -> 0.7777 (-0.0084) |
