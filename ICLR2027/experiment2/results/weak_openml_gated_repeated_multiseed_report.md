# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| validation_gated_feature | openml_cmc | 5 | 0.5427 -> 0.5733 (+0.0307) | 0.5427 -> 0.5733 (+0.0307) | 0.5443 -> 0.5750 (+0.0307) | 0.7522 -> 0.7485 (-0.0037) | acc 5/5, bacc 5/5, f1 5/5 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_repeated_feature | openml_cmc | 5 | 0.5427 -> 0.5547 (+0.0120) | 0.5427 -> 0.5547 (+0.0120) | 0.5443 -> 0.5562 (+0.0119) | 0.7522 -> 0.7399 (-0.0123) | acc 4/5, bacc 4/5, f1 4/5 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_feature | openml_credit_g | 5 | 0.6653 -> 0.6787 (+0.0133) | 0.6653 -> 0.6787 (+0.0133) | 0.6648 -> 0.6783 (+0.0136) | 0.7612 -> 0.7640 (+0.0028) | acc 3/5, bacc 3/5, f1 4/5 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_repeated_feature | openml_ilpd | 5 | 0.7014 -> 0.7151 (+0.0137) | 0.5718 -> 0.5715 (-0.0003) | 0.5697 -> 0.5579 (-0.0118) | 0.7685 -> 0.7566 (-0.0118) | acc 3/5, bacc 3/5, f1 3/5 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_feature | openml_ilpd | 5 | 0.7014 -> 0.6959 (-0.0055) | 0.5718 -> 0.5523 (-0.0195) | 0.5697 -> 0.5331 (-0.0367) | 0.7685 -> 0.7526 (-0.0158) | acc 3/5, bacc 3/5, f1 3/5 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_feature | openml_blood_transfusion | 5 | 0.7560 -> 0.7560 (+0.0000) | 0.6440 -> 0.6452 (+0.0012) | 0.6562 -> 0.6577 (+0.0015) | 0.7540 -> 0.7503 (-0.0037) | acc 2/5, bacc 3/5, f1 3/5 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_feature | openml_dmft | 5 | 0.2427 -> 0.2587 (+0.0160) | 0.2427 -> 0.2587 (+0.0160) | 0.2243 -> 0.2376 (+0.0132) | 0.6048 -> 0.5984 (-0.0064) | acc 2/5, bacc 2/5, f1 2/5 | Choose identity or a candidate feature transform using an internal validation split |
| validation_gated_repeated_feature | openml_credit_g | 5 | 0.6653 -> 0.6720 (+0.0067) | 0.6653 -> 0.6720 (+0.0067) | 0.6648 -> 0.6715 (+0.0067) | 0.7612 -> 0.7656 (+0.0044) | acc 1/5, bacc 1/5, f1 2/5 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_repeated_feature | openml_dmft | 5 | 0.2427 -> 0.2507 (+0.0080) | 0.2427 -> 0.2507 (+0.0080) | 0.2243 -> 0.2309 (+0.0066) | 0.6048 -> 0.6048 (-0.0000) | acc 1/5, bacc 1/5, f1 1/5 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_repeated_feature | openml_blood_transfusion | 5 | 0.7560 -> 0.7533 (-0.0027) | 0.6440 -> 0.6433 (-0.0007) | 0.6562 -> 0.6554 (-0.0008) | 0.7540 -> 0.7469 (-0.0072) | acc 1/5, bacc 1/5, f1 1/5 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_repeated_feature | openml_kc1 | 5 | 0.7400 -> 0.7293 (-0.0107) | 0.7400 -> 0.7293 (-0.0107) | 0.7382 -> 0.7272 (-0.0110) | 0.8262 -> 0.8110 (-0.0152) | acc 1/5, bacc 1/5, f1 1/5 | Choose a feature transform by averaging three internal validation splits |
| validation_gated_feature | openml_kc1 | 5 | 0.7400 -> 0.7280 (-0.0120) | 0.7400 -> 0.7280 (-0.0120) | 0.7382 -> 0.7257 (-0.0125) | 0.8262 -> 0.8127 (-0.0135) | acc 0/5, bacc 0/5, f1 0/5 | Choose identity or a candidate feature transform using an internal validation split |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_ilpd | 43 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7055 -> 0.7671 (+0.0616) | 0.5662 -> 0.6378 (+0.0717) | 0.5648 -> 0.6519 (+0.0870) | 0.7818 -> 0.7784 (-0.0034) |
| openml_ilpd | 46 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6575 -> 0.6986 (+0.0411) | 0.5183 -> 0.5330 (+0.0147) | 0.5106 -> 0.5139 (+0.0033) | 0.7507 -> 0.8297 (+0.0790) |
| openml_ilpd | 46 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6575 -> 0.7192 (+0.0616) | 0.5183 -> 0.5616 (+0.0433) | 0.5106 -> 0.5531 (+0.0425) | 0.7507 -> 0.8239 (+0.0733) |
| openml_cmc | 44 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5067 -> 0.5667 (+0.0600) | 0.5067 -> 0.5667 (+0.0600) | 0.5080 -> 0.5674 (+0.0594) | 0.7430 -> 0.7426 (-0.0004) |
| openml_cmc | 44 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5067 -> 0.5667 (+0.0600) | 0.5067 -> 0.5667 (+0.0600) | 0.5080 -> 0.5674 (+0.0594) | 0.7430 -> 0.7426 (-0.0004) |
| openml_dmft | 45 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2067 -> 0.2600 (+0.0533) | 0.2067 -> 0.2600 (+0.0533) | 0.1844 -> 0.2435 (+0.0591) | 0.6031 -> 0.6022 (-0.0009) |
| openml_credit_g | 46 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6600 (+0.0467) | 0.6133 -> 0.6600 (+0.0467) | 0.6131 -> 0.6596 (+0.0466) | 0.6999 -> 0.7499 (+0.0500) |
| openml_credit_g | 46 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6600 (+0.0467) | 0.6133 -> 0.6600 (+0.0467) | 0.6131 -> 0.6596 (+0.0466) | 0.6999 -> 0.7499 (+0.0500) |
| openml_ilpd | 42 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6712 -> 0.6986 (+0.0274) | 0.5634 -> 0.6110 (+0.0476) | 0.5656 -> 0.6151 (+0.0496) | 0.7337 -> 0.7283 (-0.0055) |
| openml_ilpd | 42 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6712 -> 0.7055 (+0.0342) | 0.5634 -> 0.6087 (+0.0453) | 0.5656 -> 0.6143 (+0.0487) | 0.7337 -> 0.7299 (-0.0039) |
| openml_dmft | 45 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2067 -> 0.2467 (+0.0400) | 0.2067 -> 0.2467 (+0.0400) | 0.1844 -> 0.2175 (+0.0331) | 0.6031 -> 0.6030 (-0.0001) |
| openml_cmc | 42 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6294 (+0.0357) | 0.7861 -> 0.7835 (-0.0026) |
| openml_dmft | 46 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2200 -> 0.2467 (+0.0267) | 0.2200 -> 0.2467 (+0.0267) | 0.2147 -> 0.2423 (+0.0276) | 0.5851 -> 0.5683 (-0.0167) |
| openml_credit_g | 43 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7000 -> 0.7267 (+0.0267) | 0.7000 -> 0.7267 (+0.0267) | 0.6993 -> 0.7266 (+0.0272) | 0.8048 -> 0.7989 (-0.0059) |
| openml_blood_transfusion | 46 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7533 -> 0.7667 (+0.0133) | 0.6270 -> 0.6492 (+0.0222) | 0.6367 -> 0.6637 (+0.0270) | 0.7582 -> 0.7483 (-0.0099) |
| openml_cmc | 45 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5600 -> 0.5867 (+0.0267) | 0.5600 -> 0.5867 (+0.0267) | 0.5628 -> 0.5890 (+0.0262) | 0.7563 -> 0.7670 (+0.0107) |
| openml_ilpd | 43 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7055 -> 0.7260 (+0.0205) | 0.5662 -> 0.5735 (+0.0073) | 0.5648 -> 0.5697 (+0.0049) | 0.7818 -> 0.7864 (+0.0046) |
| openml_cmc | 43 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5467 (+0.0200) | 0.5267 -> 0.5467 (+0.0200) | 0.5271 -> 0.5463 (+0.0193) | 0.7481 -> 0.7475 (-0.0006) |
| openml_cmc | 46 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5467 (+0.0200) | 0.5267 -> 0.5467 (+0.0200) | 0.5299 -> 0.5488 (+0.0189) | 0.7278 -> 0.7179 (-0.0099) |
| openml_blood_transfusion | 45 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6651 (+0.0159) | 0.6593 -> 0.6768 (+0.0175) | 0.7164 -> 0.7211 (+0.0047) |
| openml_blood_transfusion | 46 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7533 -> 0.7600 (+0.0067) | 0.6270 -> 0.6381 (+0.0111) | 0.6367 -> 0.6503 (+0.0137) | 0.7582 -> 0.7404 (-0.0178) |
| openml_cmc | 43 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5400 (+0.0133) | 0.5267 -> 0.5400 (+0.0133) | 0.5271 -> 0.5402 (+0.0131) | 0.7481 -> 0.7316 (-0.0165) |
| openml_blood_transfusion | 42 | validation_gated_feature | roc_auc | 0.7800 -> 0.7733 (-0.0067) | 0.6715 -> 0.6602 (-0.0114) | 0.6894 -> 0.6767 (-0.0127) | 0.7862 -> 0.7961 (+0.0099) |
| openml_cmc | 42 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5933 -> 0.6000 (+0.0067) | 0.5933 -> 0.6000 (+0.0067) | 0.5937 -> 0.6020 (+0.0083) | 0.7861 -> 0.7777 (-0.0084) |
| openml_blood_transfusion | 43 | validation_gated_feature | balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7467 -> 0.7467 (+0.0000) | 0.6214 -> 0.6280 (+0.0066) | 0.6309 -> 0.6387 (+0.0078) | 0.7340 -> 0.7387 (+0.0047) |
| openml_kc1 | 45 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7133 -> 0.7200 (+0.0067) | 0.7133 -> 0.7200 (+0.0067) | 0.7112 -> 0.7188 (+0.0076) | 0.8075 -> 0.7996 (-0.0079) |
| openml_credit_g | 44 | validation_gated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6000 -> 0.6067 (+0.0067) | 0.6000 -> 0.6067 (+0.0067) | 0.5994 -> 0.6062 (+0.0069) | 0.7170 -> 0.7148 (-0.0021) |
| openml_cmc | 45 | validation_gated_repeated_feature | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5600 -> 0.5667 (+0.0067) | 0.5600 -> 0.5667 (+0.0067) | 0.5628 -> 0.5694 (+0.0066) | 0.7563 -> 0.7373 (-0.0190) |
| openml_credit_g | 45 | validation_gated_feature | macro_f1, weighted_f1 | 0.6933 -> 0.6933 (+0.0000) | 0.6933 -> 0.6933 (+0.0000) | 0.6928 -> 0.6933 (+0.0005) | 0.7813 -> 0.7655 (-0.0158) |
| openml_credit_g | 45 | validation_gated_repeated_feature | macro_f1, weighted_f1 | 0.6933 -> 0.6933 (+0.0000) | 0.6933 -> 0.6933 (+0.0000) | 0.6928 -> 0.6933 (+0.0005) | 0.7813 -> 0.7655 (-0.0158) |
