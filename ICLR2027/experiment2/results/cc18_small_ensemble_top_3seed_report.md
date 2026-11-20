# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ensemble_n4 | 16 | 48 | -0.0008 | 6/16 | +0.0007 | 9/16 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | 16 | 48 | -0.0012 | 5/16 | +0.0006 | 8/16 | Average probabilities instead of logits with four estimators |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| ensemble_n4 | openml_cc18_31 | 3 | 0.6933 -> 0.6960 (+0.0027) | 0.6933 -> 0.6960 (+0.0027) | 0.6933 -> 0.6957 (+0.0024) | 0.7555 -> 0.7595 (+0.0040) | acc 2/3, bacc 2/3, f1 2/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_31 | 3 | 0.6933 -> 0.6960 (+0.0027) | 0.6933 -> 0.6960 (+0.0027) | 0.6933 -> 0.6957 (+0.0024) | 0.7555 -> 0.7598 (+0.0043) | acc 2/3, bacc 2/3, f1 2/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_32 | 3 | 0.9760 -> 0.9787 (+0.0027) | 0.9761 -> 0.9786 (+0.0026) | 0.9759 -> 0.9786 (+0.0027) | 0.9996 -> 0.9995 (-0.0001) | acc 2/3, bacc 2/3, f1 2/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_32 | 3 | 0.9760 -> 0.9787 (+0.0027) | 0.9761 -> 0.9786 (+0.0026) | 0.9759 -> 0.9786 (+0.0027) | 0.9996 -> 0.9995 (-0.0001) | acc 2/3, bacc 2/3, f1 2/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_6 | 3 | 0.8267 -> 0.8240 (-0.0027) | 0.8250 -> 0.8231 (-0.0019) | 0.8203 -> 0.8192 (-0.0011) | 0.9862 -> 0.9875 (+0.0013) | acc 2/3, bacc 2/3, f1 2/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_6 | 3 | 0.8267 -> 0.8240 (-0.0027) | 0.8250 -> 0.8231 (-0.0019) | 0.8203 -> 0.8190 (-0.0014) | 0.9862 -> 0.9869 (+0.0007) | acc 2/3, bacc 2/3, f1 2/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_11 | 3 | 0.9760 -> 0.9733 (-0.0027) | 0.9677 -> 0.9730 (+0.0053) | 0.9533 -> 0.9503 (-0.0031) | 0.9974 -> 0.9983 (+0.0009) | acc 1/3, bacc 2/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_11 | 3 | 0.9760 -> 0.9733 (-0.0027) | 0.9677 -> 0.9730 (+0.0053) | 0.9533 -> 0.9503 (-0.0031) | 0.9974 -> 0.9984 (+0.0009) | acc 1/3, bacc 2/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_23 | 3 | 0.6107 -> 0.6160 (+0.0053) | 0.6116 -> 0.6169 (+0.0053) | 0.6089 -> 0.6147 (+0.0058) | 0.7710 -> 0.7743 (+0.0033) | acc 1/3, bacc 1/3, f1 2/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_23 | 3 | 0.6107 -> 0.6160 (+0.0053) | 0.6116 -> 0.6169 (+0.0053) | 0.6089 -> 0.6147 (+0.0058) | 0.7710 -> 0.7743 (+0.0033) | acc 1/3, bacc 1/3, f1 2/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_12 | 3 | 0.9707 -> 0.9733 (+0.0027) | 0.9703 -> 0.9731 (+0.0028) | 0.9703 -> 0.9730 (+0.0027) | 0.9994 -> 0.9995 (+0.0001) | acc 1/3, bacc 1/3, f1 2/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4 | openml_cc18_3 | 3 | 0.9813 -> 0.9840 (+0.0027) | 0.9814 -> 0.9840 (+0.0026) | 0.9813 -> 0.9840 (+0.0027) | 0.9980 -> 0.9977 (-0.0003) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_3 | 3 | 0.9813 -> 0.9840 (+0.0027) | 0.9814 -> 0.9840 (+0.0026) | 0.9813 -> 0.9840 (+0.0027) | 0.9980 -> 0.9979 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_15 | 3 | 0.9733 -> 0.9760 (+0.0027) | 0.9735 -> 0.9763 (+0.0028) | 0.9733 -> 0.9760 (+0.0027) | 0.9985 -> 0.9984 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_15 | 3 | 0.9733 -> 0.9760 (+0.0027) | 0.9735 -> 0.9763 (+0.0028) | 0.9733 -> 0.9760 (+0.0027) | 0.9985 -> 0.9984 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_18 | 3 | 0.7547 -> 0.7547 (+0.0000) | 0.7575 -> 0.7573 (-0.0002) | 0.7493 -> 0.7490 (-0.0003) | 0.9678 -> 0.9679 (+0.0002) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_18 | 3 | 0.7547 -> 0.7547 (+0.0000) | 0.7575 -> 0.7573 (-0.0002) | 0.7493 -> 0.7490 (-0.0003) | 0.9678 -> 0.9679 (+0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_28 | 3 | 0.9787 -> 0.9787 (+0.0000) | 0.9793 -> 0.9791 (-0.0002) | 0.9790 -> 0.9787 (-0.0003) | 0.9995 -> 0.9994 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_28 | 3 | 0.9787 -> 0.9787 (+0.0000) | 0.9793 -> 0.9791 (-0.0002) | 0.9790 -> 0.9787 (-0.0003) | 0.9995 -> 0.9994 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_44 | 3 | 0.9440 -> 0.9440 (-0.0000) | 0.9440 -> 0.9440 (+0.0000) | 0.9440 -> 0.9440 (-0.0000) | 0.9843 -> 0.9847 (+0.0004) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_44 | 3 | 0.9440 -> 0.9440 (-0.0000) | 0.9440 -> 0.9440 (+0.0000) | 0.9440 -> 0.9440 (-0.0000) | 0.9843 -> 0.9842 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_22 | 3 | 0.8507 -> 0.8480 (-0.0027) | 0.8511 -> 0.8485 (-0.0026) | 0.8398 -> 0.8404 (+0.0006) | 0.9861 -> 0.9856 (-0.0005) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_22 | 3 | 0.8507 -> 0.8480 (-0.0027) | 0.8511 -> 0.8485 (-0.0026) | 0.8398 -> 0.8404 (+0.0006) | 0.9861 -> 0.9854 (-0.0007) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_16 | 3 | 0.9547 -> 0.9493 (-0.0053) | 0.9534 -> 0.9483 (-0.0051) | 0.9534 -> 0.9482 (-0.0053) | 0.9987 -> 0.9986 (-0.0000) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4 | openml_cc18_14 | 3 | 0.8640 -> 0.8587 (-0.0053) | 0.8650 -> 0.8590 (-0.0060) | 0.8639 -> 0.8588 (-0.0051) | 0.9869 -> 0.9875 (+0.0005) | acc 1/3, bacc 1/3, f1 1/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_14 | 3 | 0.8640 -> 0.8587 (-0.0053) | 0.8650 -> 0.8590 (-0.0060) | 0.8639 -> 0.8588 (-0.0051) | 0.9869 -> 0.9875 (+0.0006) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4_avg_probs | openml_cc18_16 | 3 | 0.9547 -> 0.9467 (-0.0080) | 0.9534 -> 0.9455 (-0.0079) | 0.9534 -> 0.9456 (-0.0078) | 0.9987 -> 0.9986 (-0.0001) | acc 1/3, bacc 1/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4_avg_probs | openml_cc18_12 | 3 | 0.9707 -> 0.9707 (+0.0000) | 0.9703 -> 0.9703 (+0.0000) | 0.9703 -> 0.9702 (-0.0001) | 0.9994 -> 0.9995 (+0.0001) | acc 0/3, bacc 0/3, f1 1/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_29 | 3 | 0.8453 -> 0.8427 (-0.0027) | 0.8448 -> 0.8421 (-0.0027) | 0.8442 -> 0.8414 (-0.0028) | 0.9331 -> 0.9345 (+0.0015) | acc 0/3, bacc 0/3, f1 0/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_29 | 3 | 0.8453 -> 0.8427 (-0.0027) | 0.8448 -> 0.8421 (-0.0027) | 0.8442 -> 0.8414 (-0.0028) | 0.9331 -> 0.9345 (+0.0014) | acc 0/3, bacc 0/3, f1 0/3 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_37 | 3 | 0.7600 -> 0.7493 (-0.0107) | 0.7602 -> 0.7496 (-0.0106) | 0.7593 -> 0.7484 (-0.0109) | 0.8261 -> 0.8258 (-0.0003) | acc 0/3, bacc 0/3, f1 0/3 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | openml_cc18_37 | 3 | 0.7600 -> 0.7493 (-0.0107) | 0.7602 -> 0.7496 (-0.0106) | 0.7593 -> 0.7484 (-0.0109) | 0.8261 -> 0.8260 (-0.0001) | acc 0/3, bacc 0/3, f1 0/3 | Average probabilities instead of logits with four estimators |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_cc18_31 | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7280 -> 0.7520 (+0.0240) | 0.7280 -> 0.7520 (+0.0241) | 0.7280 -> 0.7520 (+0.0240) | 0.8080 -> 0.8182 (+0.0102) |
| openml_cc18_31 | 43 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7280 -> 0.7520 (+0.0240) | 0.7280 -> 0.7520 (+0.0241) | 0.7280 -> 0.7520 (+0.0240) | 0.8080 -> 0.8182 (+0.0102) |
| openml_cc18_6 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8160 -> 0.8400 (+0.0240) | 0.8173 -> 0.8404 (+0.0231) | 0.8145 -> 0.8341 (+0.0196) | 0.9862 -> 0.9889 (+0.0027) |
| openml_cc18_14 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9945 (+0.0020) |
| openml_cc18_14 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9948 (+0.0023) |
| openml_cc18_6 | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8240 -> 0.8400 (+0.0160) | 0.8231 -> 0.8385 (+0.0154) | 0.8132 -> 0.8295 (+0.0163) | 0.9880 -> 0.9893 (+0.0013) |
| openml_cc18_23 | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5920 -> 0.6080 (+0.0160) | 0.5933 -> 0.6092 (+0.0159) | 0.5875 -> 0.6044 (+0.0169) | 0.7707 -> 0.7746 (+0.0039) |
| openml_cc18_23 | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5920 -> 0.6080 (+0.0160) | 0.5933 -> 0.6092 (+0.0159) | 0.5875 -> 0.6044 (+0.0169) | 0.7707 -> 0.7747 (+0.0040) |
| openml_cc18_18 | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7280 -> 0.7440 (+0.0160) | 0.7321 -> 0.7481 (+0.0160) | 0.7220 -> 0.7386 (+0.0166) | 0.9647 -> 0.9649 (+0.0002) |
| openml_cc18_18 | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7280 -> 0.7440 (+0.0160) | 0.7321 -> 0.7481 (+0.0160) | 0.7220 -> 0.7386 (+0.0166) | 0.9647 -> 0.9649 (+0.0002) |
| openml_cc18_31 | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6080 -> 0.6240 (+0.0160) | 0.6080 -> 0.6243 (+0.0163) | 0.6080 -> 0.6236 (+0.0156) | 0.6733 -> 0.6772 (+0.0038) |
| openml_cc18_31 | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6080 -> 0.6240 (+0.0160) | 0.6080 -> 0.6243 (+0.0163) | 0.6080 -> 0.6236 (+0.0156) | 0.6733 -> 0.6774 (+0.0041) |
| openml_cc18_28 | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9680 -> 0.9840 (+0.0160) | 0.9686 -> 0.9846 (+0.0160) | 0.9691 -> 0.9846 (+0.0155) | 0.9992 -> 0.9993 (+0.0001) |
| openml_cc18_28 | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9680 -> 0.9840 (+0.0160) | 0.9686 -> 0.9846 (+0.0160) | 0.9691 -> 0.9846 (+0.0155) | 0.9992 -> 0.9993 (+0.0001) |
| openml_cc18_44 | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9360 -> 0.9520 (+0.0160) | 0.9360 -> 0.9520 (+0.0160) | 0.9360 -> 0.9520 (+0.0160) | 0.9764 -> 0.9762 (-0.0003) |
| openml_cc18_44 | 43 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9360 -> 0.9520 (+0.0160) | 0.9360 -> 0.9520 (+0.0160) | 0.9360 -> 0.9520 (+0.0160) | 0.9764 -> 0.9757 (-0.0008) |
| openml_cc18_6 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8160 -> 0.8320 (+0.0160) | 0.8173 -> 0.8327 (+0.0154) | 0.8145 -> 0.8267 (+0.0122) | 0.9862 -> 0.9893 (+0.0031) |
| openml_cc18_11 | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9704 -> 0.9764 (+0.0060) | 0.9274 -> 0.9404 (+0.0130) | 0.9973 -> 0.9988 (+0.0016) |
| openml_cc18_11 | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9704 -> 0.9764 (+0.0060) | 0.9274 -> 0.9404 (+0.0130) | 0.9973 -> 0.9991 (+0.0018) |
| openml_cc18_11 | 43 | ensemble_n4 | balanced_accuracy, roc_auc | 0.9680 -> 0.9520 (-0.0160) | 0.9326 -> 0.9427 (+0.0100) | 0.9326 -> 0.9105 (-0.0221) | 0.9950 -> 0.9960 (+0.0010) |
| openml_cc18_11 | 43 | ensemble_n4_avg_probs | balanced_accuracy, roc_auc | 0.9680 -> 0.9520 (-0.0160) | 0.9326 -> 0.9427 (+0.0100) | 0.9326 -> 0.9105 (-0.0221) | 0.9950 -> 0.9960 (+0.0010) |
| openml_cc18_6 | 43 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8240 -> 0.8320 (+0.0080) | 0.8231 -> 0.8308 (+0.0077) | 0.8132 -> 0.8212 (+0.0080) | 0.9880 -> 0.9884 (+0.0004) |
| openml_cc18_22 | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.8800 -> 0.8880 (+0.0080) | 0.8808 -> 0.8885 (+0.0077) | 0.8777 -> 0.8865 (+0.0088) | 0.9900 -> 0.9883 (-0.0017) |
| openml_cc18_22 | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.8800 -> 0.8880 (+0.0080) | 0.8808 -> 0.8885 (+0.0077) | 0.8777 -> 0.8865 (+0.0088) | 0.9900 -> 0.9882 (-0.0018) |
| openml_cc18_16 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) |
| openml_cc18_16 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) |
| openml_cc18_15 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) |
| openml_cc18_15 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) |
| openml_cc18_32 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) |
| openml_cc18_32 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) |
| openml_cc18_12 | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9596 -> 0.9679 (+0.0083) | 0.9592 -> 0.9672 (+0.0080) | 0.9988 -> 0.9991 (+0.0003) |
| openml_cc18_3 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) |
| openml_cc18_3 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) |
| openml_cc18_32 | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9840 -> 0.9920 (+0.0080) | 0.9840 -> 0.9917 (+0.0077) | 0.9840 -> 0.9919 (+0.0080) | 0.9993 -> 0.9993 (+0.0000) |
| openml_cc18_32 | 43 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9840 -> 0.9920 (+0.0080) | 0.9840 -> 0.9917 (+0.0077) | 0.9840 -> 0.9919 (+0.0080) | 0.9993 -> 0.9993 (+0.0000) |
| openml_cc18_23 | 42 | ensemble_n4 | macro_f1, weighted_f1, roc_auc | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7701 (+0.0059) |
| openml_cc18_23 | 42 | ensemble_n4_avg_probs | macro_f1, weighted_f1, roc_auc | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7700 (+0.0058) |
| openml_cc18_29 | 42 | ensemble_n4 | roc_auc | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9304 (+0.0051) |
| openml_cc18_29 | 42 | ensemble_n4_avg_probs | roc_auc | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9301 (+0.0049) |
| openml_cc18_14 | 43 | ensemble_n4_avg_probs | roc_auc | 0.8400 -> 0.8240 (-0.0160) | 0.8417 -> 0.8256 (-0.0160) | 0.8407 -> 0.8252 (-0.0156) | 0.9810 -> 0.9836 (+0.0026) |
| openml_cc18_37 | 43 | ensemble_n4 | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7602 -> 0.7523 (-0.0079) | 0.7599 -> 0.7517 (-0.0081) | 0.8390 -> 0.8415 (+0.0026) |
| openml_cc18_37 | 43 | ensemble_n4_avg_probs | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7602 -> 0.7523 (-0.0079) | 0.7599 -> 0.7517 (-0.0081) | 0.8390 -> 0.8415 (+0.0026) |
| openml_cc18_14 | 43 | ensemble_n4 | roc_auc | 0.8400 -> 0.8240 (-0.0160) | 0.8417 -> 0.8256 (-0.0160) | 0.8407 -> 0.8252 (-0.0156) | 0.9810 -> 0.9835 (+0.0025) |
| openml_cc18_44 | 44 | ensemble_n4 | roc_auc | 0.9360 -> 0.9200 (-0.0160) | 0.9359 -> 0.9199 (-0.0160) | 0.9360 -> 0.9200 (-0.0160) | 0.9844 -> 0.9852 (+0.0008) |
| openml_cc18_44 | 42 | ensemble_n4 | roc_auc | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9928 (+0.0008) |
| openml_cc18_18 | 42 | ensemble_n4 | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9650 (+0.0007) |
| openml_cc18_18 | 42 | ensemble_n4_avg_probs | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9649 (+0.0006) |
| openml_cc18_44 | 44 | ensemble_n4_avg_probs | roc_auc | 0.9360 -> 0.9200 (-0.0160) | 0.9359 -> 0.9199 (-0.0160) | 0.9360 -> 0.9200 (-0.0160) | 0.9844 -> 0.9849 (+0.0005) |
| openml_cc18_29 | 44 | ensemble_n4 | roc_auc | 0.8720 -> 0.8720 (+0.0000) | 0.8717 -> 0.8717 (+0.0000) | 0.8718 -> 0.8718 (+0.0000) | 0.9549 -> 0.9555 (+0.0005) |
| openml_cc18_12 | 44 | ensemble_n4 | macro_f1, weighted_f1 | 0.9680 -> 0.9680 (+0.0000) | 0.9679 -> 0.9679 (+0.0000) | 0.9679 -> 0.9679 (+0.0000) | 0.9999 -> 0.9996 (-0.0002) |
| openml_cc18_12 | 44 | ensemble_n4_avg_probs | macro_f1, weighted_f1 | 0.9680 -> 0.9680 (+0.0000) | 0.9679 -> 0.9679 (+0.0000) | 0.9679 -> 0.9679 (+0.0000) | 0.9999 -> 0.9996 (-0.0002) |
| openml_cc18_12 | 43 | ensemble_n4_avg_probs | weighted_f1, roc_auc | 0.9600 -> 0.9600 (+0.0000) | 0.9596 -> 0.9596 (+0.0000) | 0.9592 -> 0.9589 (-0.0002) | 0.9988 -> 0.9991 (+0.0003) |
| openml_cc18_29 | 44 | ensemble_n4_avg_probs | roc_auc | 0.8720 -> 0.8720 (+0.0000) | 0.8717 -> 0.8717 (+0.0000) | 0.8718 -> 0.8718 (+0.0000) | 0.9549 -> 0.9552 (+0.0003) |
| openml_cc18_12 | 42 | ensemble_n4 | roc_auc | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) |
| openml_cc18_12 | 42 | ensemble_n4_avg_probs | roc_auc | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) |
| openml_cc18_22 | 42 | ensemble_n4 | roc_auc | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9873 (+0.0002) |
| openml_cc18_23 | 43 | ensemble_n4 | roc_auc | 0.6160 -> 0.6160 (+0.0000) | 0.6169 -> 0.6169 (+0.0000) | 0.6164 -> 0.6164 (+0.0000) | 0.7780 -> 0.7781 (+0.0001) |
| openml_cc18_23 | 43 | ensemble_n4_avg_probs | roc_auc | 0.6160 -> 0.6160 (+0.0000) | 0.6169 -> 0.6169 (+0.0000) | 0.6164 -> 0.6164 (+0.0000) | 0.7780 -> 0.7781 (+0.0001) |
| openml_cc18_22 | 43 | ensemble_n4 | roc_auc | 0.8160 -> 0.8080 (-0.0080) | 0.8218 -> 0.8147 (-0.0071) | 0.8026 -> 0.8011 (-0.0015) | 0.9812 -> 0.9812 (+0.0000) |
| openml_cc18_16 | 43 | ensemble_n4 | roc_auc | 0.9520 -> 0.9360 (-0.0160) | 0.9513 -> 0.9346 (-0.0167) | 0.9504 -> 0.9339 (-0.0164) | 0.9984 -> 0.9984 (+0.0000) |
| openml_cc18_16 | 43 | ensemble_n4_avg_probs | roc_auc | 0.9520 -> 0.9360 (-0.0160) | 0.9513 -> 0.9346 (-0.0167) | 0.9504 -> 0.9339 (-0.0164) | 0.9984 -> 0.9984 (+0.0000) |
| openml_cc18_22 | 42 | ensemble_n4_avg_probs | roc_auc | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9871 (+0.0000) |
