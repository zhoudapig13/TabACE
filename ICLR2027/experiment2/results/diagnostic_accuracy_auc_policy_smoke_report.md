# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| diagnostic_accuracy_auc_policy | 4 | 4 | +0.0033 | 2/4 | +0.0025 | 2/4 | Use training diagnostics to enable strong-weight sample attention or low-feature binary ensembling, otherwise baseline |
| ensemble_n4 | 4 | 4 | -0.0067 | 2/4 | +0.0011 | 2/4 | Increase TabICL ensemble size to 4 |
| strong_weight_sample_attention_policy | 4 | 4 | +0.0017 | 1/4 | +0.0021 | 1/4 | Enable sample-level causal attention only for very concentrated causal-proxy weights |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| ensemble_n4 | openml_credit_g | 1 | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7264 (+0.0072) | 0.8028 -> 0.8085 (+0.0057) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| strong_weight_sample_attention_policy | openml_credit_g | 1 | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7261 (+0.0069) | 0.8028 -> 0.8114 (+0.0085) | acc 1/1, bacc 1/1, f1 1/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| diagnostic_accuracy_auc_policy | openml_credit_g | 1 | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7261 (+0.0069) | 0.8028 -> 0.8114 (+0.0085) | acc 1/1, bacc 1/1, f1 1/1 | Use training diagnostics to enable strong-weight sample attention or low-feature binary ensembling, otherwise baseline |
| ensemble_n4 | openml_blood_transfusion | 1 | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7877 (+0.0015) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| diagnostic_accuracy_auc_policy | openml_blood_transfusion | 1 | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7877 (+0.0015) | acc 1/1, bacc 1/1, f1 1/1 | Use training diagnostics to enable strong-weight sample attention or low-feature binary ensembling, otherwise baseline |
| strong_weight_sample_attention_policy | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| diagnostic_accuracy_auc_policy | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Use training diagnostics to enable strong-weight sample attention or low-feature binary ensembling, otherwise baseline |
| strong_weight_sample_attention_policy | openml_blood_transfusion | 1 | 0.7800 -> 0.7800 (+0.0000) | 0.6715 -> 0.6715 (+0.0000) | 0.6894 -> 0.6894 (+0.0000) | 0.7862 -> 0.7862 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| strong_weight_sample_attention_policy | openml_kc1 | 1 | 0.7267 -> 0.7267 (+0.0000) | 0.7267 -> 0.7267 (+0.0000) | 0.7239 -> 0.7239 (+0.0000) | 0.8145 -> 0.8145 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| diagnostic_accuracy_auc_policy | openml_kc1 | 1 | 0.7267 -> 0.7267 (+0.0000) | 0.7267 -> 0.7267 (+0.0000) | 0.7239 -> 0.7239 (+0.0000) | 0.8145 -> 0.8145 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Use training diagnostics to enable strong-weight sample attention or low-feature binary ensembling, otherwise baseline |
| ensemble_n4 | breast_cancer | 1 | 0.9790 -> 0.9720 (-0.0070) | 0.9756 -> 0.9700 (-0.0056) | 0.9774 -> 0.9700 (-0.0074) | 0.9973 -> 0.9969 (-0.0004) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4 | openml_kc1 | 1 | 0.7267 -> 0.6933 (-0.0333) | 0.7267 -> 0.6933 (-0.0333) | 0.7239 -> 0.6914 (-0.0325) | 0.8145 -> 0.8120 (-0.0025) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_blood_transfusion | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7877 (+0.0015) |
| openml_blood_transfusion | 42 | diagnostic_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7877 (+0.0015) |
| openml_credit_g | 42 | strong_weight_sample_attention_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7261 (+0.0069) | 0.8028 -> 0.8114 (+0.0085) |
| openml_credit_g | 42 | diagnostic_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7261 (+0.0069) | 0.8028 -> 0.8114 (+0.0085) |
| openml_credit_g | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7264 (+0.0072) | 0.8028 -> 0.8085 (+0.0057) |
