# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| outlier_winsor_ensemble_n4 | 3 | 3 | -0.0057 | 1/3 | +0.0009 | 2/3 | Outlier-winsor features with four TabICL estimators |
| accuracy_auc_strict_repeated_effective_gate | 3 | 3 | -0.0153 | 0/3 | +0.0056 | 2/3 | Use repeated accuracy/AUC validation with strict no-drop guards to choose among effective candidates or baseline |
| propensity_quantile_domain_feature | 3 | 3 | -0.0127 | 0/3 | +0.0003 | 1/3 | Quantile-normalize features and append domain-shift plus propensity features |
| strong_weight_sample_attention_policy | 3 | 3 | +0.0000 | 0/3 | -0.0023 | 0/3 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| ensemble_n4 | 3 | 3 | -0.0157 | 0/3 | -0.0016 | 0/3 | Increase TabICL ensemble size to 4 |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| outlier_winsor_ensemble_n4 | breast_cancer | 1 | 0.9790 -> 0.9860 (+0.0070) | 0.9756 -> 0.9850 (+0.0094) | 0.9774 -> 0.9850 (+0.0076) | 0.9973 -> 0.9981 (+0.0008) | acc 1/1, bacc 1/1, f1 1/1 | Outlier-winsor features with four TabICL estimators |
| strong_weight_sample_attention_policy | openml_credit_g | 1 | 0.7440 -> 0.7440 (+0.0000) | 0.7439 -> 0.7440 (+0.0001) | 0.7439 -> 0.7440 (+0.0001) | 0.7852 -> 0.7783 (-0.0069) | acc 0/1, bacc 1/1, f1 1/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| propensity_quantile_domain_feature | openml_kc1 | 1 | 0.7760 -> 0.7760 (+0.0000) | 0.7768 -> 0.7769 (+0.0001) | 0.7743 -> 0.7736 (-0.0007) | 0.8391 -> 0.8258 (-0.0133) | acc 0/1, bacc 1/1, f1 0/1 | Quantile-normalize features and append domain-shift plus propensity features |
| strong_weight_sample_attention_policy | breast_cancer | 1 | 0.9790 -> 0.9790 (+0.0000) | 0.9756 -> 0.9756 (+0.0000) | 0.9774 -> 0.9774 (+0.0000) | 0.9973 -> 0.9973 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| strong_weight_sample_attention_policy | openml_kc1 | 1 | 0.7760 -> 0.7760 (+0.0000) | 0.7768 -> 0.7768 (+0.0000) | 0.7743 -> 0.7743 (+0.0000) | 0.8391 -> 0.8391 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Enable sample-level causal attention only for very concentrated causal-proxy weights |
| ensemble_n4 | breast_cancer | 1 | 0.9790 -> 0.9720 (-0.0070) | 0.9756 -> 0.9700 (-0.0056) | 0.9774 -> 0.9700 (-0.0074) | 0.9973 -> 0.9969 (-0.0004) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4 | openml_kc1 | 1 | 0.7760 -> 0.7680 (-0.0080) | 0.7768 -> 0.7687 (-0.0081) | 0.7743 -> 0.7665 (-0.0077) | 0.8391 -> 0.8368 (-0.0023) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| outlier_winsor_ensemble_n4 | openml_kc1 | 1 | 0.7760 -> 0.7680 (-0.0080) | 0.7768 -> 0.7687 (-0.0081) | 0.7743 -> 0.7665 (-0.0077) | 0.8391 -> 0.8417 (+0.0026) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| accuracy_auc_strict_repeated_effective_gate | openml_kc1 | 1 | 0.7760 -> 0.7680 (-0.0080) | 0.7768 -> 0.7687 (-0.0081) | 0.7743 -> 0.7665 (-0.0077) | 0.8391 -> 0.8417 (+0.0026) | acc 0/1, bacc 0/1, f1 0/1 | Use repeated accuracy/AUC validation with strict no-drop guards to choose among effective candidates or baseline |
| propensity_quantile_domain_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9645 (-0.0111) | 0.9774 -> 0.9627 (-0.0148) | 0.9973 -> 0.9971 (-0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append domain-shift plus propensity features |
| accuracy_auc_strict_repeated_effective_gate | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9645 (-0.0111) | 0.9774 -> 0.9627 (-0.0148) | 0.9973 -> 0.9971 (-0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Use repeated accuracy/AUC validation with strict no-drop guards to choose among effective candidates or baseline |
| outlier_winsor_ensemble_n4 | openml_credit_g | 1 | 0.7440 -> 0.7280 (-0.0160) | 0.7439 -> 0.7279 (-0.0160) | 0.7439 -> 0.7278 (-0.0160) | 0.7852 -> 0.7844 (-0.0008) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| propensity_quantile_domain_feature | openml_credit_g | 1 | 0.7440 -> 0.7200 (-0.0240) | 0.7439 -> 0.7198 (-0.0241) | 0.7439 -> 0.7197 (-0.0241) | 0.7852 -> 0.7995 (+0.0143) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append domain-shift plus propensity features |
| accuracy_auc_strict_repeated_effective_gate | openml_credit_g | 1 | 0.7440 -> 0.7200 (-0.0240) | 0.7439 -> 0.7198 (-0.0241) | 0.7439 -> 0.7197 (-0.0241) | 0.7852 -> 0.7995 (+0.0143) | acc 0/1, bacc 0/1, f1 0/1 | Use repeated accuracy/AUC validation with strict no-drop guards to choose among effective candidates or baseline |
| ensemble_n4 | openml_credit_g | 1 | 0.7440 -> 0.7120 (-0.0320) | 0.7439 -> 0.7117 (-0.0321) | 0.7439 -> 0.7115 (-0.0323) | 0.7852 -> 0.7832 (-0.0020) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_credit_g | 42 | propensity_quantile_domain_feature | roc_auc | 0.7440 -> 0.7200 (-0.0240) | 0.7439 -> 0.7198 (-0.0241) | 0.7439 -> 0.7197 (-0.0241) | 0.7852 -> 0.7995 (+0.0143) |
| openml_credit_g | 42 | accuracy_auc_strict_repeated_effective_gate | roc_auc | 0.7440 -> 0.7200 (-0.0240) | 0.7439 -> 0.7198 (-0.0241) | 0.7439 -> 0.7197 (-0.0241) | 0.7852 -> 0.7995 (+0.0143) |
| breast_cancer | 42 | outlier_winsor_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9790 -> 0.9860 (+0.0070) | 0.9756 -> 0.9850 (+0.0094) | 0.9774 -> 0.9850 (+0.0076) | 0.9973 -> 0.9981 (+0.0008) |
| openml_kc1 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.7760 -> 0.7680 (-0.0080) | 0.7768 -> 0.7687 (-0.0081) | 0.7743 -> 0.7665 (-0.0077) | 0.8391 -> 0.8417 (+0.0026) |
| openml_kc1 | 42 | accuracy_auc_strict_repeated_effective_gate | roc_auc | 0.7760 -> 0.7680 (-0.0080) | 0.7768 -> 0.7687 (-0.0081) | 0.7743 -> 0.7665 (-0.0077) | 0.8391 -> 0.8417 (+0.0026) |
| openml_credit_g | 42 | strong_weight_sample_attention_policy | balanced_accuracy, macro_f1, weighted_f1 | 0.7440 -> 0.7440 (+0.0000) | 0.7439 -> 0.7440 (+0.0001) | 0.7439 -> 0.7440 (+0.0001) | 0.7852 -> 0.7783 (-0.0069) |
| openml_kc1 | 42 | propensity_quantile_domain_feature | balanced_accuracy | 0.7760 -> 0.7760 (+0.0000) | 0.7768 -> 0.7769 (+0.0001) | 0.7743 -> 0.7736 (-0.0007) | 0.8391 -> 0.8258 (-0.0133) |
