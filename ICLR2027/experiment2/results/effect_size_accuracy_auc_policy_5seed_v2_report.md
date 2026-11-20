# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| effect_size_accuracy_auc_policy | 7 | 35 | +0.0090 | 5/7 | +0.0047 | 5/7 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| broad_accuracy_auc_policy | 7 | 35 | +0.0067 | 5/7 | +0.0048 | 5/7 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| effect_size_accuracy_auc_policy | openml_blood_transfusion | 5 | 0.7560 -> 0.7667 (+0.0107) | 0.6440 -> 0.6568 (+0.0128) | 0.6562 -> 0.6717 (+0.0155) | 0.7540 -> 0.7543 (+0.0003) | acc 5/5, bacc 5/5, f1 5/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| effect_size_accuracy_auc_policy | openml_cmc | 5 | 0.5427 -> 0.5640 (+0.0213) | 0.5427 -> 0.5640 (+0.0213) | 0.5443 -> 0.5657 (+0.0214) | 0.7522 -> 0.7587 (+0.0065) | acc 4/5, bacc 4/5, f1 4/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| broad_accuracy_auc_policy | openml_blood_transfusion | 5 | 0.7560 -> 0.7640 (+0.0080) | 0.6440 -> 0.6523 (+0.0083) | 0.6562 -> 0.6655 (+0.0093) | 0.7540 -> 0.7577 (+0.0037) | acc 4/5, bacc 4/5, f1 4/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| broad_accuracy_auc_policy | openml_credit_g | 5 | 0.6653 -> 0.6813 (+0.0160) | 0.6653 -> 0.6813 (+0.0160) | 0.6648 -> 0.6810 (+0.0162) | 0.7612 -> 0.7675 (+0.0064) | acc 3/5, bacc 3/5, f1 4/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| effect_size_accuracy_auc_policy | openml_credit_g | 5 | 0.6653 -> 0.6813 (+0.0160) | 0.6653 -> 0.6813 (+0.0160) | 0.6648 -> 0.6810 (+0.0162) | 0.7612 -> 0.7675 (+0.0064) | acc 3/5, bacc 3/5, f1 4/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| broad_accuracy_auc_policy | openml_cmc | 5 | 0.5427 -> 0.5507 (+0.0080) | 0.5427 -> 0.5507 (+0.0080) | 0.5443 -> 0.5526 (+0.0083) | 0.7522 -> 0.7561 (+0.0039) | acc 3/5, bacc 3/5, f1 3/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| broad_accuracy_auc_policy | openml_ilpd | 5 | 0.7014 -> 0.7123 (+0.0110) | 0.5718 -> 0.5823 (+0.0105) | 0.5697 -> 0.5781 (+0.0084) | 0.7685 -> 0.7862 (+0.0177) | acc 2/5, bacc 2/5, f1 2/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| effect_size_accuracy_auc_policy | openml_ilpd | 5 | 0.7014 -> 0.7123 (+0.0110) | 0.5718 -> 0.5823 (+0.0105) | 0.5697 -> 0.5781 (+0.0084) | 0.7685 -> 0.7862 (+0.0177) | acc 2/5, bacc 2/5, f1 2/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| broad_accuracy_auc_policy | openml_kc1 | 5 | 0.7400 -> 0.7440 (+0.0040) | 0.7400 -> 0.7440 (+0.0040) | 0.7382 -> 0.7427 (+0.0044) | 0.8262 -> 0.8281 (+0.0019) | acc 2/5, bacc 2/5, f1 2/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| effect_size_accuracy_auc_policy | openml_kc1 | 5 | 0.7400 -> 0.7440 (+0.0040) | 0.7400 -> 0.7440 (+0.0040) | 0.7382 -> 0.7427 (+0.0044) | 0.8262 -> 0.8281 (+0.0019) | acc 2/5, bacc 2/5, f1 2/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| broad_accuracy_auc_policy | breast_cancer | 5 | 0.9720 -> 0.9720 (+0.0000) | 0.9700 -> 0.9700 (+0.0000) | 0.9700 -> 0.9700 (+0.0000) | 0.9953 -> 0.9953 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| effect_size_accuracy_auc_policy | breast_cancer | 5 | 0.9720 -> 0.9720 (+0.0000) | 0.9700 -> 0.9700 (+0.0000) | 0.9700 -> 0.9700 (+0.0000) | 0.9953 -> 0.9953 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |
| broad_accuracy_auc_policy | openml_dmft | 5 | 0.2427 -> 0.2427 (+0.0000) | 0.2427 -> 0.2427 (+0.0000) | 0.2243 -> 0.2243 (+0.0000) | 0.6048 -> 0.6048 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use training diagnostics to route among prior accuracy/AUC-positive methods, otherwise baseline |
| effect_size_accuracy_auc_policy | openml_dmft | 5 | 0.2427 -> 0.2427 (+0.0000) | 0.2427 -> 0.2427 (+0.0000) | 0.2243 -> 0.2243 (+0.0000) | 0.6048 -> 0.6048 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use stronger diagnostic routing to target >=0.01 accuracy gains while guarding AUC |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_ilpd | 46 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6575 -> 0.6986 (+0.0411) | 0.5183 -> 0.5330 (+0.0147) | 0.5106 -> 0.5139 (+0.0033) | 0.7507 -> 0.8297 (+0.0790) |
| openml_ilpd | 46 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6575 -> 0.6986 (+0.0411) | 0.5183 -> 0.5330 (+0.0147) | 0.5106 -> 0.5139 (+0.0033) | 0.7507 -> 0.8297 (+0.0790) |
| openml_cmc | 44 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5067 -> 0.5600 (+0.0533) | 0.5067 -> 0.5600 (+0.0533) | 0.5080 -> 0.5622 (+0.0542) | 0.7430 -> 0.7635 (+0.0205) |
| openml_credit_g | 44 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6000 -> 0.6400 (+0.0400) | 0.6000 -> 0.6400 (+0.0400) | 0.5994 -> 0.6397 (+0.0404) | 0.7170 -> 0.7147 (-0.0023) |
| openml_credit_g | 44 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6000 -> 0.6400 (+0.0400) | 0.6000 -> 0.6400 (+0.0400) | 0.5994 -> 0.6397 (+0.0404) | 0.7170 -> 0.7147 (-0.0023) |
| openml_ilpd | 42 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6712 -> 0.6849 (+0.0137) | 0.5634 -> 0.6014 (+0.0380) | 0.5656 -> 0.6040 (+0.0384) | 0.7337 -> 0.7434 (+0.0096) |
| openml_ilpd | 42 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6712 -> 0.6849 (+0.0137) | 0.5634 -> 0.6014 (+0.0380) | 0.5656 -> 0.6040 (+0.0384) | 0.7337 -> 0.7434 (+0.0096) |
| openml_credit_g | 46 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6467 (+0.0333) | 0.6133 -> 0.6467 (+0.0333) | 0.6131 -> 0.6467 (+0.0336) | 0.6999 -> 0.7287 (+0.0288) |
| openml_credit_g | 46 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6467 (+0.0333) | 0.6133 -> 0.6467 (+0.0333) | 0.6131 -> 0.6467 (+0.0336) | 0.6999 -> 0.7287 (+0.0288) |
| openml_cmc | 42 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6261 (+0.0324) | 0.7861 -> 0.7927 (+0.0066) |
| openml_blood_transfusion | 46 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7533 -> 0.7733 (+0.0200) | 0.6270 -> 0.6540 (+0.0270) | 0.6367 -> 0.6698 (+0.0331) | 0.7582 -> 0.7512 (-0.0070) |
| openml_blood_transfusion | 44 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7600 -> 0.7733 (+0.0133) | 0.6507 -> 0.6735 (+0.0227) | 0.6646 -> 0.6893 (+0.0247) | 0.7753 -> 0.7747 (-0.0006) |
| openml_kc1 | 45 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7133 -> 0.7333 (+0.0200) | 0.7133 -> 0.7333 (+0.0200) | 0.7112 -> 0.7329 (+0.0217) | 0.8075 -> 0.8078 (+0.0004) |
| openml_kc1 | 45 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7133 -> 0.7333 (+0.0200) | 0.7133 -> 0.7333 (+0.0200) | 0.7112 -> 0.7329 (+0.0217) | 0.8075 -> 0.8078 (+0.0004) |
| openml_cmc | 44 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5067 -> 0.5267 (+0.0200) | 0.5067 -> 0.5267 (+0.0200) | 0.5080 -> 0.5280 (+0.0200) | 0.7430 -> 0.7410 (-0.0020) |
| openml_cmc | 45 | broad_accuracy_auc_policy | roc_auc | 0.5600 -> 0.5600 (+0.0000) | 0.5600 -> 0.5600 (+0.0000) | 0.5628 -> 0.5610 (-0.0017) | 0.7563 -> 0.7724 (+0.0161) |
| openml_cmc | 42 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5933 -> 0.6067 (+0.0133) | 0.5933 -> 0.6067 (+0.0133) | 0.5937 -> 0.6081 (+0.0144) | 0.7861 -> 0.7918 (+0.0057) |
| openml_blood_transfusion | 43 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7467 -> 0.7533 (+0.0067) | 0.6214 -> 0.6327 (+0.0114) | 0.6309 -> 0.6445 (+0.0136) | 0.7340 -> 0.7409 (+0.0069) |
| openml_cmc | 45 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5600 -> 0.5733 (+0.0133) | 0.5600 -> 0.5733 (+0.0133) | 0.5628 -> 0.5746 (+0.0118) | 0.7563 -> 0.7587 (+0.0025) |
| openml_blood_transfusion | 45 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6587 (+0.0095) | 0.6593 -> 0.6711 (+0.0118) | 0.7164 -> 0.7228 (+0.0063) |
| openml_blood_transfusion | 45 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6524 (+0.0032) | 0.6593 -> 0.6651 (+0.0058) | 0.7164 -> 0.7181 (+0.0017) |
| openml_blood_transfusion | 44 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7600 -> 0.7667 (+0.0067) | 0.6507 -> 0.6621 (+0.0114) | 0.6646 -> 0.6771 (+0.0125) | 0.7753 -> 0.7768 (+0.0015) |
| openml_blood_transfusion | 42 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7867 (+0.0004) |
| openml_blood_transfusion | 42 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7845 (-0.0017) |
| openml_blood_transfusion | 43 | broad_accuracy_auc_policy | roc_auc | 0.7467 -> 0.7467 (+0.0000) | 0.6214 -> 0.6147 (-0.0066) | 0.6309 -> 0.6226 (-0.0083) | 0.7340 -> 0.7432 (+0.0092) |
| openml_cmc | 43 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5271 -> 0.5359 (+0.0088) | 0.7481 -> 0.7476 (-0.0005) |
| openml_credit_g | 42 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7261 (+0.0069) | 0.8028 -> 0.8114 (+0.0085) |
| openml_credit_g | 42 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7261 (+0.0069) | 0.8028 -> 0.8114 (+0.0085) |
| openml_cmc | 43 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5271 -> 0.5354 (+0.0083) | 0.7481 -> 0.7507 (+0.0027) |
| openml_kc1 | 44 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7733 -> 0.7800 (+0.0067) | 0.7733 -> 0.7800 (+0.0067) | 0.7727 -> 0.7795 (+0.0068) | 0.8645 -> 0.8697 (+0.0052) |
| openml_kc1 | 44 | effect_size_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7733 -> 0.7800 (+0.0067) | 0.7733 -> 0.7800 (+0.0067) | 0.7727 -> 0.7795 (+0.0068) | 0.8645 -> 0.8697 (+0.0052) |
| openml_blood_transfusion | 46 | broad_accuracy_auc_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7533 -> 0.7600 (+0.0067) | 0.6270 -> 0.6317 (+0.0048) | 0.6367 -> 0.6425 (+0.0058) | 0.7582 -> 0.7614 (+0.0032) |
| openml_kc1 | 42 | broad_accuracy_auc_policy | roc_auc | 0.7267 -> 0.7200 (-0.0067) | 0.7267 -> 0.7200 (-0.0067) | 0.7239 -> 0.7175 (-0.0064) | 0.8145 -> 0.8211 (+0.0066) |
| openml_kc1 | 42 | effect_size_accuracy_auc_policy | roc_auc | 0.7267 -> 0.7200 (-0.0067) | 0.7267 -> 0.7200 (-0.0067) | 0.7239 -> 0.7175 (-0.0064) | 0.8145 -> 0.8211 (+0.0066) |
| openml_credit_g | 45 | broad_accuracy_auc_policy | macro_f1, weighted_f1, roc_auc | 0.6933 -> 0.6933 (+0.0000) | 0.6933 -> 0.6933 (+0.0000) | 0.6928 -> 0.6931 (+0.0003) | 0.7813 -> 0.7844 (+0.0030) |
| openml_credit_g | 45 | effect_size_accuracy_auc_policy | macro_f1, weighted_f1, roc_auc | 0.6933 -> 0.6933 (+0.0000) | 0.6933 -> 0.6933 (+0.0000) | 0.6928 -> 0.6931 (+0.0003) | 0.7813 -> 0.7844 (+0.0030) |
