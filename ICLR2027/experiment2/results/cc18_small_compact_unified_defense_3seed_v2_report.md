# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Overall Method Stability

| Method | Datasets | Runs | Acc Mean Delta | Acc Positive Datasets | AUC Mean Delta | AUC Positive Datasets | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| compact_unified_defense_policy | 16 | 48 | +0.0035 | 2/16 | +0.0006 | 1/16 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| compact_unified_defense_policy | openml_cc18_23 | 3 | 0.5422 -> 0.5733 (+0.0311) | 0.5422 -> 0.5733 (+0.0311) | 0.5429 -> 0.5746 (+0.0317) | 0.7590 -> 0.7690 (+0.0099) | acc 3/3, bacc 3/3, f1 3/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_31 | 3 | 0.6733 -> 0.6978 (+0.0244) | 0.6733 -> 0.6978 (+0.0244) | 0.6726 -> 0.6957 (+0.0231) | 0.7749 -> 0.7749 (+0.0000) | acc 2/3, bacc 2/3, f1 2/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_3 | 3 | 0.9800 -> 0.9800 (+0.0000) | 0.9800 -> 0.9800 (+0.0000) | 0.9800 -> 0.9800 (+0.0000) | 0.9995 -> 0.9995 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_6 | 3 | 0.8267 -> 0.8267 (+0.0000) | 0.8269 -> 0.8269 (+0.0000) | 0.8234 -> 0.8234 (+0.0000) | 0.9842 -> 0.9842 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_11 | 3 | 0.9800 -> 0.9800 (+0.0000) | 0.9779 -> 0.9779 (+0.0000) | 0.9553 -> 0.9553 (+0.0000) | 0.9991 -> 0.9991 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_12 | 3 | 0.9644 -> 0.9644 (+0.0000) | 0.9644 -> 0.9644 (+0.0000) | 0.9639 -> 0.9639 (+0.0000) | 0.9984 -> 0.9984 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_14 | 3 | 0.8356 -> 0.8356 (+0.0000) | 0.8356 -> 0.8356 (+0.0000) | 0.8346 -> 0.8346 (+0.0000) | 0.9860 -> 0.9860 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_15 | 3 | 0.9644 -> 0.9644 (+0.0000) | 0.9611 -> 0.9611 (+0.0000) | 0.9628 -> 0.9628 (+0.0000) | 0.9962 -> 0.9962 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_16 | 3 | 0.9644 -> 0.9644 (+0.0000) | 0.9644 -> 0.9644 (+0.0000) | 0.9642 -> 0.9642 (+0.0000) | 0.9987 -> 0.9987 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_18 | 3 | 0.7067 -> 0.7067 (+0.0000) | 0.7067 -> 0.7067 (+0.0000) | 0.6939 -> 0.6939 (+0.0000) | 0.9514 -> 0.9514 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_22 | 3 | 0.8600 -> 0.8600 (+0.0000) | 0.8600 -> 0.8600 (+0.0000) | 0.8584 -> 0.8584 (+0.0000) | 0.9881 -> 0.9881 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_28 | 3 | 0.9911 -> 0.9911 (+0.0000) | 0.9911 -> 0.9911 (+0.0000) | 0.9911 -> 0.9911 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_29 | 3 | 0.8733 -> 0.8733 (+0.0000) | 0.8733 -> 0.8733 (+0.0000) | 0.8732 -> 0.8732 (+0.0000) | 0.9319 -> 0.9319 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_32 | 3 | 0.9867 -> 0.9867 (+0.0000) | 0.9867 -> 0.9867 (+0.0000) | 0.9867 -> 0.9867 (+0.0000) | 1.0000 -> 1.0000 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_37 | 3 | 0.7711 -> 0.7711 (+0.0000) | 0.7625 -> 0.7625 (+0.0000) | 0.7649 -> 0.7649 (+0.0000) | 0.8582 -> 0.8582 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |
| compact_unified_defense_policy | openml_cc18_44 | 3 | 0.9267 -> 0.9267 (+0.0000) | 0.9267 -> 0.9267 (+0.0000) | 0.9266 -> 0.9266 (+0.0000) | 0.9772 -> 0.9772 (+0.0000) | acc 0/3, bacc 0/3, f1 0/3 | Use a compact unified defense line with binary boundary calibration, multiclass domain robustness, and baseline fallback |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_cc18_31 | 44 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6000 -> 0.6667 (+0.0667) | 0.6000 -> 0.6667 (+0.0667) | 0.5994 -> 0.6666 (+0.0672) | 0.7170 -> 0.7147 (-0.0023) |
| openml_cc18_23 | 44 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5067 -> 0.5600 (+0.0533) | 0.5067 -> 0.5600 (+0.0533) | 0.5080 -> 0.5622 (+0.0542) | 0.7430 -> 0.7635 (+0.0205) |
| openml_cc18_23 | 42 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5933 -> 0.6267 (+0.0333) | 0.5933 -> 0.6267 (+0.0333) | 0.5937 -> 0.6261 (+0.0324) | 0.7861 -> 0.7927 (+0.0066) |
| openml_cc18_31 | 42 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7212 (+0.0020) | 0.8028 -> 0.8114 (+0.0085) |
| openml_cc18_23 | 43 | compact_unified_defense_policy | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5271 -> 0.5354 (+0.0083) | 0.7481 -> 0.7507 (+0.0027) |
