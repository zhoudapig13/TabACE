# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| compact_unified_defense_policy | 11 | 55 | +0.0072 | 4/11 | +0.0020 | 4/11 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| compact_unified_defense_policy | openml_cmc | 5 | 0.5427 -> 0.5680 (+0.0253) | 0.5427 -> 0.5680 (+0.0253) | 0.5443 -> 0.5692 (+0.0249) | 0.7522 -> 0.7580 (+0.0057) | acc 5/5, bacc 5/5, f1 5/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_blood_transfusion | 5 | 0.7560 -> 0.7667 (+0.0107) | 0.6440 -> 0.6568 (+0.0128) | 0.6562 -> 0.6717 (+0.0155) | 0.7540 -> 0.7543 (+0.0003) | acc 5/5, bacc 5/5, f1 5/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_credit_g | 5 | 0.6653 -> 0.6880 (+0.0227) | 0.6653 -> 0.6880 (+0.0227) | 0.6648 -> 0.6866 (+0.0218) | 0.7612 -> 0.7675 (+0.0064) | acc 3/5, bacc 3/5, f1 4/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_ilpd | 5 | 0.7014 -> 0.7219 (+0.0205) | 0.5718 -> 0.5990 (+0.0272) | 0.5697 -> 0.6009 (+0.0312) | 0.7685 -> 0.7782 (+0.0098) | acc 3/5, bacc 3/5, f1 3/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | breast_cancer | 5 | 0.9720 -> 0.9720 (+0.0000) | 0.9700 -> 0.9700 (+0.0000) | 0.9700 -> 0.9700 (+0.0000) | 0.9953 -> 0.9953 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_diabetes_pima | 5 | 0.7573 -> 0.7573 (+0.0000) | 0.7496 -> 0.7496 (+0.0000) | 0.7516 -> 0.7516 (+0.0000) | 0.8409 -> 0.8409 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_spambase | 5 | 0.9267 -> 0.9267 (+0.0000) | 0.9267 -> 0.9267 (+0.0000) | 0.9266 -> 0.9266 (+0.0000) | 0.9801 -> 0.9801 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_balance_scale | 5 | 0.9800 -> 0.9800 (+0.0000) | 0.9810 -> 0.9810 (+0.0000) | 0.9569 -> 0.9569 (+0.0000) | 0.9993 -> 0.9993 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_credit_approval | 5 | 0.8787 -> 0.8787 (+0.0000) | 0.8787 -> 0.8787 (+0.0000) | 0.8786 -> 0.8786 (+0.0000) | 0.9385 -> 0.9385 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_tic_tac_toe | 5 | 0.9893 -> 0.9893 (+0.0000) | 0.9893 -> 0.9893 (+0.0000) | 0.9893 -> 0.9893 (+0.0000) | 0.9998 -> 0.9998 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_vehicle | 5 | 0.8760 -> 0.8760 (+0.0000) | 0.8758 -> 0.8758 (+0.0000) | 0.8755 -> 0.8755 (+0.0000) | 0.9773 -> 0.9773 (+0.0000) | acc 0/5, bacc 0/5, f1 0/5 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_ilpd | 46 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6575 -> 0.6986 (+0.0411) | 0.5183 -> 0.5826 (+0.0643) | 0.5106 -> 0.5866 (+0.0760) | 0.7507 -> 0.7825 (+0.0318) |
| openml_credit_g | 44 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6000 -> 0.6667 (+0.0667) | 0.6000 -> 0.6667 (+0.0667) | 0.5994 -> 0.6666 (+0.0672) | 0.7170 -> 0.7147 (-0.0023) |
| openml_cmc | 44 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5067 -> 0.5600 (+0.0533) | 0.5067 -> 0.5600 (+0.0533) | 0.5080 -> 0.5622 (+0.0542) | 0.7430 -> 0.7635 (+0.0205) |
| openml_ilpd | 42 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6712 -> 0.7055 (+0.0342) | 0.5634 -> 0.6087 (+0.0453) | 0.5656 -> 0.6143 (+0.0487) | 0.7337 -> 0.7370 (+0.0032) |
| openml_credit_g | 46 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6533 (+0.0400) | 0.6133 -> 0.6533 (+0.0400) | 0.6131 -> 0.6523 (+0.0393) | 0.6999 -> 0.7287 (+0.0288) |
| openml_cmc | 42 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6261 (+0.0324) | 0.7861 -> 0.7927 (+0.0066) |
| openml_blood_transfusion | 46 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7533 -> 0.7733 (+0.0200) | 0.6270 -> 0.6540 (+0.0270) | 0.6367 -> 0.6698 (+0.0331) | 0.7582 -> 0.7512 (-0.0070) |
| openml_ilpd | 43 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7055 -> 0.7329 (+0.0274) | 0.5662 -> 0.5925 (+0.0263) | 0.5648 -> 0.5959 (+0.0311) | 0.7818 -> 0.7956 (+0.0137) |
| openml_cmc | 46 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5467 (+0.0200) | 0.5267 -> 0.5467 (+0.0200) | 0.5299 -> 0.5476 (+0.0177) | 0.7278 -> 0.7241 (-0.0037) |
| openml_blood_transfusion | 43 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7467 -> 0.7533 (+0.0067) | 0.6214 -> 0.6327 (+0.0114) | 0.6309 -> 0.6445 (+0.0136) | 0.7340 -> 0.7409 (+0.0069) |
| openml_blood_transfusion | 45 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6524 (+0.0032) | 0.6593 -> 0.6651 (+0.0058) | 0.7164 -> 0.7181 (+0.0017) |
| openml_cmc | 45 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5600 -> 0.5733 (+0.0133) | 0.5600 -> 0.5733 (+0.0133) | 0.5628 -> 0.5746 (+0.0118) | 0.7563 -> 0.7587 (+0.0025) |
| openml_blood_transfusion | 44 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7600 -> 0.7667 (+0.0067) | 0.6507 -> 0.6621 (+0.0114) | 0.6646 -> 0.6771 (+0.0125) | 0.7753 -> 0.7768 (+0.0015) |
| openml_blood_transfusion | 42 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7845 (-0.0017) |
| openml_credit_g | 42 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7212 (+0.0020) | 0.8028 -> 0.8114 (+0.0085) |
| openml_cmc | 43 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5271 -> 0.5354 (+0.0083) | 0.7481 -> 0.7507 (+0.0027) |
| openml_credit_g | 45 | compact_unified_defense_policy | macro_f1, weighted_f1, roc_auc | 0.6933 -> 0.6933 (+0.0000) | 0.6933 -> 0.6933 (+0.0000) | 0.6928 -> 0.6933 (+0.0004) | 0.7813 -> 0.7844 (+0.0030) |
