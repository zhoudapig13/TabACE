# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ensemble_n4_temp07 | 16 | 16 | -0.0005 | 6/16 | +0.0009 | 10/16 | Sharper softmax temperature with four estimators |
| ensemble_n4 | 16 | 16 | -0.0005 | 6/16 | +0.0009 | 10/16 | Increase TabICL ensemble size to 4 |
| ensemble_n4_avg_probs | 16 | 16 | +0.0000 | 6/16 | +0.0009 | 9/16 | Average probabilities instead of logits with four estimators |
| quantile_domain_probe_ensemble_n4 | 16 | 16 | -0.0210 | 4/16 | -0.0057 | 4/16 | Quantile-domain-probe features with four TabICL estimators |
| outlier_winsor_ensemble_n4 | 16 | 16 | -0.0055 | 2/16 | +0.0010 | 10/16 | Outlier-winsor features with four TabICL estimators |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| quantile_domain_probe_ensemble_n4 | openml_cc18_37 | 1 | 0.7200 -> 0.7520 (+0.0320) | 0.7197 -> 0.7519 (+0.0323) | 0.7194 -> 0.7519 (+0.0326) | 0.7913 -> 0.8062 (+0.0148) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-domain-probe features with four TabICL estimators |
| ensemble_n4_avg_probs | openml_cc18_6 | 1 | 0.8160 -> 0.8400 (+0.0240) | 0.8173 -> 0.8404 (+0.0231) | 0.8145 -> 0.8341 (+0.0196) | 0.9862 -> 0.9889 (+0.0027) | acc 1/1, bacc 1/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_6 | 1 | 0.8160 -> 0.8320 (+0.0160) | 0.8173 -> 0.8327 (+0.0154) | 0.8145 -> 0.8267 (+0.0122) | 0.9862 -> 0.9893 (+0.0031) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_6 | 1 | 0.8160 -> 0.8320 (+0.0160) | 0.8173 -> 0.8327 (+0.0154) | 0.8145 -> 0.8267 (+0.0122) | 0.9862 -> 0.9893 (+0.0031) | acc 1/1, bacc 1/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4 | openml_cc18_14 | 1 | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9945 (+0.0020) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_14 | 1 | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9945 (+0.0020) | acc 1/1, bacc 1/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_14 | 1 | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9948 (+0.0023) | acc 1/1, bacc 1/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_3 | 1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_3 | 1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) | acc 1/1, bacc 1/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_3 | 1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) | acc 1/1, bacc 1/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_6 | 1 | 0.8160 -> 0.8240 (+0.0080) | 0.8173 -> 0.8231 (+0.0058) | 0.8145 -> 0.8198 (+0.0053) | 0.9862 -> 0.9895 (+0.0033) | acc 1/1, bacc 1/1, f1 1/1 | Outlier-winsor features with four TabICL estimators |
| quantile_domain_probe_ensemble_n4 | openml_cc18_14 | 1 | 0.8960 -> 0.9040 (+0.0080) | 0.8949 -> 0.9019 (+0.0071) | 0.8928 -> 0.9028 (+0.0100) | 0.9925 -> 0.9923 (-0.0002) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-domain-probe features with four TabICL estimators |
| ensemble_n4 | openml_cc18_15 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_15 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) | acc 1/1, bacc 1/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_15 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) | acc 1/1, bacc 1/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| quantile_domain_probe_ensemble_n4 | openml_cc18_15 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9956 (-0.0028) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-domain-probe features with four TabICL estimators |
| ensemble_n4 | openml_cc18_16 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_16 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) | acc 1/1, bacc 1/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_16 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) | acc 1/1, bacc 1/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| quantile_domain_probe_ensemble_n4 | openml_cc18_16 | 1 | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9993 (+0.0001) | acc 1/1, bacc 1/1, f1 1/1 | Quantile-domain-probe features with four TabICL estimators |
| ensemble_n4 | openml_cc18_32 | 1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) | acc 1/1, bacc 1/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_32 | 1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) | acc 1/1, bacc 1/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_32 | 1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) | acc 1/1, bacc 1/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_32 | 1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0080) | 0.9997 -> 0.9998 (+0.0001) | acc 1/1, bacc 1/1, f1 1/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_23 | 1 | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7701 (+0.0059) | acc 0/1, bacc 0/1, f1 1/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_23 | 1 | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7701 (+0.0059) | acc 0/1, bacc 0/1, f1 1/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_23 | 1 | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7700 (+0.0058) | acc 0/1, bacc 0/1, f1 1/1 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cc18_11 | 1 | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_11 | 1 | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_11 | 1 | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_11 | 1 | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_12 | 1 | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_12 | 1 | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_12 | 1 | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |
| quantile_domain_probe_ensemble_n4 | openml_cc18_12 | 1 | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9995 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-domain-probe features with four TabICL estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_12 | 1 | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9999 (+0.0004) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_15 | 1 | 0.9600 -> 0.9600 (+0.0000) | 0.9590 -> 0.9590 (+0.0000) | 0.9598 -> 0.9598 (+0.0000) | 0.9985 -> 0.9987 (+0.0003) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_44 | 1 | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9928 (+0.0008) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_44 | 1 | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9928 (+0.0008) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_44 | 1 | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9921 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_44 | 1 | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9941 (+0.0020) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| quantile_domain_probe_ensemble_n4 | openml_cc18_3 | 1 | 0.9920 -> 0.9840 (-0.0080) | 0.9921 -> 0.9840 (-0.0081) | 0.9920 -> 0.9840 (-0.0080) | 1.0000 -> 0.9985 (-0.0015) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-domain-probe features with four TabICL estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_3 | 1 | 0.9920 -> 0.9840 (-0.0080) | 0.9921 -> 0.9839 (-0.0082) | 0.9920 -> 0.9840 (-0.0080) | 1.0000 -> 0.9995 (-0.0005) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_16 | 1 | 0.9600 -> 0.9520 (-0.0080) | 0.9583 -> 0.9500 (-0.0083) | 0.9584 -> 0.9506 (-0.0078) | 0.9992 -> 0.9996 (+0.0004) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_18 | 1 | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9650 (+0.0007) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_18 | 1 | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9650 (+0.0007) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_18 | 1 | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9649 (+0.0006) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_18 | 1 | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7487 (-0.0096) | 0.7519 -> 0.7380 (-0.0139) | 0.9644 -> 0.9645 (+0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_22 | 1 | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9873 (+0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_22 | 1 | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9876 (+0.0005) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_22 | 1 | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9871 (+0.0000) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_22 | 1 | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8324 (-0.0067) | 0.9871 -> 0.9865 (-0.0006) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_28 | 1 | 0.9840 -> 0.9760 (-0.0080) | 0.9846 -> 0.9763 (-0.0083) | 0.9840 -> 0.9756 (-0.0084) | 0.9995 -> 0.9992 (-0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_28 | 1 | 0.9840 -> 0.9760 (-0.0080) | 0.9846 -> 0.9763 (-0.0083) | 0.9840 -> 0.9756 (-0.0084) | 0.9995 -> 0.9992 (-0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_28 | 1 | 0.9840 -> 0.9760 (-0.0080) | 0.9846 -> 0.9763 (-0.0083) | 0.9840 -> 0.9756 (-0.0084) | 0.9995 -> 0.9992 (-0.0003) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |
| quantile_domain_probe_ensemble_n4 | openml_cc18_28 | 1 | 0.9840 -> 0.9760 (-0.0080) | 0.9846 -> 0.9769 (-0.0077) | 0.9840 -> 0.9763 (-0.0077) | 0.9995 -> 0.9988 (-0.0006) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-domain-probe features with four TabICL estimators |
| outlier_winsor_ensemble_n4 | openml_cc18_28 | 1 | 0.9840 -> 0.9760 (-0.0080) | 0.9846 -> 0.9763 (-0.0083) | 0.9840 -> 0.9756 (-0.0084) | 0.9995 -> 0.9991 (-0.0003) | acc 0/1, bacc 0/1, f1 0/1 | Outlier-winsor features with four TabICL estimators |
| ensemble_n4 | openml_cc18_29 | 1 | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9304 (+0.0051) | acc 0/1, bacc 0/1, f1 0/1 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cc18_29 | 1 | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9304 (+0.0051) | acc 0/1, bacc 0/1, f1 0/1 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cc18_29 | 1 | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9301 (+0.0049) | acc 0/1, bacc 0/1, f1 0/1 | Average probabilities instead of logits with four estimators |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_cc18_37 | 42 | quantile_domain_probe_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7520 (+0.0320) | 0.7197 -> 0.7519 (+0.0323) | 0.7194 -> 0.7519 (+0.0326) | 0.7913 -> 0.8062 (+0.0148) |
| openml_cc18_6 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8160 -> 0.8400 (+0.0240) | 0.8173 -> 0.8404 (+0.0231) | 0.8145 -> 0.8341 (+0.0196) | 0.9862 -> 0.9889 (+0.0027) |
| openml_cc18_14 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9945 (+0.0020) |
| openml_cc18_14 | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9945 (+0.0020) |
| openml_cc18_14 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8960 -> 0.9120 (+0.0160) | 0.8949 -> 0.9103 (+0.0154) | 0.8928 -> 0.9101 (+0.0172) | 0.9925 -> 0.9948 (+0.0023) |
| openml_cc18_6 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8160 -> 0.8320 (+0.0160) | 0.8173 -> 0.8327 (+0.0154) | 0.8145 -> 0.8267 (+0.0122) | 0.9862 -> 0.9893 (+0.0031) |
| openml_cc18_6 | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8160 -> 0.8320 (+0.0160) | 0.8173 -> 0.8327 (+0.0154) | 0.8145 -> 0.8267 (+0.0122) | 0.9862 -> 0.9893 (+0.0031) |
| openml_cc18_14 | 42 | quantile_domain_probe_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.8960 -> 0.9040 (+0.0080) | 0.8949 -> 0.9019 (+0.0071) | 0.8928 -> 0.9028 (+0.0100) | 0.9925 -> 0.9923 (-0.0002) |
| openml_cc18_16 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) |
| openml_cc18_16 | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) |
| openml_cc18_16 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9996 (+0.0004) |
| openml_cc18_16 | 42 | quantile_domain_probe_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9583 -> 0.9667 (+0.0083) | 0.9584 -> 0.9668 (+0.0084) | 0.9992 -> 0.9993 (+0.0001) |
| openml_cc18_15 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) |
| openml_cc18_15 | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) |
| openml_cc18_15 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9990 (+0.0005) |
| openml_cc18_15 | 42 | quantile_domain_probe_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9600 -> 0.9680 (+0.0080) | 0.9590 -> 0.9673 (+0.0083) | 0.9598 -> 0.9679 (+0.0081) | 0.9985 -> 0.9956 (-0.0028) |
| openml_cc18_32 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) |
| openml_cc18_32 | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) |
| openml_cc18_32 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0081) | 0.9997 -> 0.9997 (+0.0000) |
| openml_cc18_32 | 42 | outlier_winsor_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.9680 -> 0.9760 (+0.0080) | 0.9673 -> 0.9756 (+0.0083) | 0.9679 -> 0.9759 (+0.0080) | 0.9997 -> 0.9998 (+0.0001) |
| openml_cc18_3 | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) |
| openml_cc18_3 | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) |
| openml_cc18_3 | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.9920 -> 1.0000 (+0.0080) | 0.9921 -> 1.0000 (+0.0079) | 0.9920 -> 1.0000 (+0.0080) | 1.0000 -> 1.0000 (+0.0000) |
| openml_cc18_6 | 42 | outlier_winsor_ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.8160 -> 0.8240 (+0.0080) | 0.8173 -> 0.8231 (+0.0058) | 0.8145 -> 0.8198 (+0.0053) | 0.9862 -> 0.9895 (+0.0033) |
| openml_cc18_37 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.7200 -> 0.7120 (-0.0080) | 0.7197 -> 0.7117 (-0.0079) | 0.7194 -> 0.7115 (-0.0078) | 0.7913 -> 0.7993 (+0.0079) |
| openml_cc18_23 | 42 | ensemble_n4_temp07 | macro_f1, weighted_f1, roc_auc | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7701 (+0.0059) |
| openml_cc18_23 | 42 | ensemble_n4 | macro_f1, weighted_f1, roc_auc | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7701 (+0.0059) |
| openml_cc18_23 | 42 | ensemble_n4_avg_probs | macro_f1, weighted_f1, roc_auc | 0.6240 -> 0.6240 (+0.0000) | 0.6245 -> 0.6245 (+0.0000) | 0.6227 -> 0.6233 (+0.0006) | 0.7642 -> 0.7700 (+0.0058) |
| openml_cc18_29 | 42 | ensemble_n4 | roc_auc | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9304 (+0.0051) |
| openml_cc18_29 | 42 | ensemble_n4_temp07 | roc_auc | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9304 (+0.0051) |
| openml_cc18_29 | 42 | ensemble_n4_avg_probs | roc_auc | 0.8320 -> 0.8240 (-0.0080) | 0.8312 -> 0.8231 (-0.0081) | 0.8299 -> 0.8214 (-0.0084) | 0.9252 -> 0.9301 (+0.0049) |
| openml_cc18_44 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9941 (+0.0020) |
| openml_cc18_23 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.6240 -> 0.6080 (-0.0160) | 0.6245 -> 0.6086 (-0.0159) | 0.6227 -> 0.6065 (-0.0162) | 0.7642 -> 0.7662 (+0.0019) |
| openml_cc18_14 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.8960 -> 0.8800 (-0.0160) | 0.8949 -> 0.8782 (-0.0167) | 0.8928 -> 0.8776 (-0.0152) | 0.9925 -> 0.9944 (+0.0019) |
| openml_cc18_23 | 42 | quantile_domain_probe_ensemble_n4 | roc_auc | 0.6240 -> 0.5680 (-0.0560) | 0.6245 -> 0.5683 (-0.0561) | 0.6227 -> 0.5716 (-0.0511) | 0.7642 -> 0.7658 (+0.0016) |
| openml_cc18_44 | 42 | ensemble_n4 | roc_auc | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9928 (+0.0008) |
| openml_cc18_44 | 42 | ensemble_n4_temp07 | roc_auc | 0.9600 -> 0.9600 (+0.0000) | 0.9602 -> 0.9602 (+0.0000) | 0.9600 -> 0.9600 (+0.0000) | 0.9921 -> 0.9928 (+0.0008) |
| openml_cc18_18 | 42 | ensemble_n4_temp07 | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9650 (+0.0007) |
| openml_cc18_18 | 42 | ensemble_n4 | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9650 (+0.0007) |
| openml_cc18_18 | 42 | ensemble_n4_avg_probs | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7500 (-0.0083) | 0.7519 -> 0.7415 (-0.0104) | 0.9644 -> 0.9649 (+0.0006) |
| openml_cc18_22 | 42 | ensemble_n4_temp07 | roc_auc | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9876 (+0.0005) |
| openml_cc18_12 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9999 (+0.0004) |
| openml_cc18_16 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.9600 -> 0.9520 (-0.0080) | 0.9583 -> 0.9500 (-0.0083) | 0.9584 -> 0.9506 (-0.0078) | 0.9992 -> 0.9996 (+0.0004) |
| openml_cc18_15 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.9600 -> 0.9600 (+0.0000) | 0.9590 -> 0.9590 (+0.0000) | 0.9598 -> 0.9598 (+0.0000) | 0.9985 -> 0.9987 (+0.0003) |
| openml_cc18_29 | 42 | quantile_domain_probe_ensemble_n4 | roc_auc | 0.8320 -> 0.8160 (-0.0160) | 0.8312 -> 0.8149 (-0.0163) | 0.8299 -> 0.8121 (-0.0178) | 0.9252 -> 0.9255 (+0.0003) |
| openml_cc18_12 | 42 | ensemble_n4 | roc_auc | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) |
| openml_cc18_12 | 42 | ensemble_n4_temp07 | roc_auc | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) |
| openml_cc18_12 | 42 | ensemble_n4_avg_probs | roc_auc | 0.9840 -> 0.9840 (+0.0000) | 0.9833 -> 0.9833 (+0.0000) | 0.9839 -> 0.9839 (+0.0000) | 0.9995 -> 0.9997 (+0.0002) |
| openml_cc18_22 | 42 | ensemble_n4 | roc_auc | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9873 (+0.0002) |
| openml_cc18_18 | 42 | outlier_winsor_ensemble_n4 | roc_auc | 0.7600 -> 0.7520 (-0.0080) | 0.7583 -> 0.7487 (-0.0096) | 0.7519 -> 0.7380 (-0.0139) | 0.9644 -> 0.9645 (+0.0002) |
| openml_cc18_22 | 42 | ensemble_n4_avg_probs | roc_auc | 0.8560 -> 0.8480 (-0.0080) | 0.8506 -> 0.8423 (-0.0083) | 0.8391 -> 0.8337 (-0.0054) | 0.9871 -> 0.9871 (+0.0000) |
