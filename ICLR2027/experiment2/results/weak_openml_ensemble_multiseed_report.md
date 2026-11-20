# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| ensemble_n4 | openml_credit_g | 5 | 0.6653 -> 0.6813 (+0.0160) | 0.6653 -> 0.6813 (+0.0160) | 0.6648 -> 0.6808 (+0.0160) | 0.7612 -> 0.7645 (+0.0033) | acc 5/5, bacc 5/5, f1 5/5 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_credit_g | 5 | 0.6653 -> 0.6813 (+0.0160) | 0.6653 -> 0.6813 (+0.0160) | 0.6648 -> 0.6808 (+0.0160) | 0.7612 -> 0.7645 (+0.0033) | acc 5/5, bacc 5/5, f1 5/5 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_credit_g | 5 | 0.6653 -> 0.6813 (+0.0160) | 0.6653 -> 0.6813 (+0.0160) | 0.6648 -> 0.6808 (+0.0160) | 0.7612 -> 0.7645 (+0.0033) | acc 5/5, bacc 5/5, f1 5/5 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_blood_transfusion | 5 | 0.7560 -> 0.7627 (+0.0067) | 0.6440 -> 0.6527 (+0.0087) | 0.6562 -> 0.6656 (+0.0095) | 0.7540 -> 0.7570 (+0.0030) | acc 4/5, bacc 4/5, f1 4/5 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_blood_transfusion | 5 | 0.7560 -> 0.7627 (+0.0067) | 0.6440 -> 0.6527 (+0.0087) | 0.6562 -> 0.6656 (+0.0095) | 0.7540 -> 0.7570 (+0.0030) | acc 4/5, bacc 4/5, f1 4/5 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_blood_transfusion | 5 | 0.7560 -> 0.7627 (+0.0067) | 0.6440 -> 0.6527 (+0.0087) | 0.6562 -> 0.6656 (+0.0095) | 0.7540 -> 0.7569 (+0.0029) | acc 4/5, bacc 4/5, f1 4/5 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_cmc | 5 | 0.5427 -> 0.5453 (+0.0027) | 0.5427 -> 0.5453 (+0.0027) | 0.5443 -> 0.5465 (+0.0021) | 0.7522 -> 0.7496 (-0.0027) | acc 3/5, bacc 3/5, f1 3/5 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_cmc | 5 | 0.5427 -> 0.5453 (+0.0027) | 0.5427 -> 0.5453 (+0.0027) | 0.5443 -> 0.5465 (+0.0021) | 0.7522 -> 0.7493 (-0.0029) | acc 3/5, bacc 3/5, f1 3/5 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_cmc | 5 | 0.5427 -> 0.5453 (+0.0027) | 0.5427 -> 0.5453 (+0.0027) | 0.5443 -> 0.5465 (+0.0021) | 0.7522 -> 0.7495 (-0.0027) | acc 3/5, bacc 3/5, f1 3/5 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_dmft | 5 | 0.2427 -> 0.2413 (-0.0013) | 0.2427 -> 0.2413 (-0.0013) | 0.2243 -> 0.2248 (+0.0005) | 0.6048 -> 0.5987 (-0.0060) | acc 2/5, bacc 2/5, f1 2/5 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_dmft | 5 | 0.2427 -> 0.2413 (-0.0013) | 0.2427 -> 0.2413 (-0.0013) | 0.2243 -> 0.2248 (+0.0005) | 0.6048 -> 0.5988 (-0.0059) | acc 2/5, bacc 2/5, f1 2/5 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_dmft | 5 | 0.2427 -> 0.2413 (-0.0013) | 0.2427 -> 0.2413 (-0.0013) | 0.2243 -> 0.2248 (+0.0005) | 0.6048 -> 0.5987 (-0.0061) | acc 2/5, bacc 2/5, f1 2/5 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_ilpd | 5 | 0.7014 -> 0.6945 (-0.0068) | 0.5718 -> 0.5656 (-0.0062) | 0.5697 -> 0.5603 (-0.0095) | 0.7685 -> 0.7614 (-0.0071) | acc 2/5, bacc 2/5, f1 2/5 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_ilpd | 5 | 0.7014 -> 0.6945 (-0.0068) | 0.5718 -> 0.5656 (-0.0062) | 0.5697 -> 0.5603 (-0.0095) | 0.7685 -> 0.7614 (-0.0071) | acc 2/5, bacc 2/5, f1 2/5 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_ilpd | 5 | 0.7014 -> 0.6945 (-0.0068) | 0.5718 -> 0.5656 (-0.0062) | 0.5697 -> 0.5603 (-0.0095) | 0.7685 -> 0.7612 (-0.0072) | acc 2/5, bacc 2/5, f1 2/5 | Average probabilities instead of logits with four estimators |
| ensemble_n4 | openml_kc1 | 5 | 0.7400 -> 0.7333 (-0.0067) | 0.7400 -> 0.7333 (-0.0067) | 0.7382 -> 0.7319 (-0.0063) | 0.8262 -> 0.8246 (-0.0016) | acc 1/5, bacc 1/5, f1 1/5 | Increase TabICL ensemble size to 4 |
| ensemble_n4_temp07 | openml_kc1 | 5 | 0.7400 -> 0.7333 (-0.0067) | 0.7400 -> 0.7333 (-0.0067) | 0.7382 -> 0.7319 (-0.0063) | 0.8262 -> 0.8246 (-0.0016) | acc 1/5, bacc 1/5, f1 1/5 | Sharper softmax temperature with four estimators |
| ensemble_n4_avg_probs | openml_kc1 | 5 | 0.7400 -> 0.7333 (-0.0067) | 0.7400 -> 0.7333 (-0.0067) | 0.7382 -> 0.7319 (-0.0063) | 0.8262 -> 0.8245 (-0.0017) | acc 1/5, bacc 1/5, f1 1/5 | Average probabilities instead of logits with four estimators |

## Top Single-Run Metric Improvements

| Dataset | Seed | Method | Improved Metrics | Accuracy | BAcc | Macro F1 | AUC |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| openml_credit_g | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6000 -> 0.6400 (+0.0400) | 0.6000 -> 0.6400 (+0.0400) | 0.5994 -> 0.6399 (+0.0406) | 0.7170 -> 0.7196 (+0.0027) |
| openml_credit_g | 44 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6000 -> 0.6400 (+0.0400) | 0.6000 -> 0.6400 (+0.0400) | 0.5994 -> 0.6399 (+0.0406) | 0.7170 -> 0.7196 (+0.0027) |
| openml_credit_g | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6000 -> 0.6400 (+0.0400) | 0.6000 -> 0.6400 (+0.0400) | 0.5994 -> 0.6399 (+0.0406) | 0.7170 -> 0.7191 (+0.0021) |
| openml_dmft | 46 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2200 -> 0.2467 (+0.0267) | 0.2200 -> 0.2467 (+0.0267) | 0.2147 -> 0.2523 (+0.0376) | 0.5851 -> 0.5831 (-0.0020) |
| openml_dmft | 46 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2200 -> 0.2467 (+0.0267) | 0.2200 -> 0.2467 (+0.0267) | 0.2147 -> 0.2523 (+0.0376) | 0.5851 -> 0.5836 (-0.0015) |
| openml_dmft | 46 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2200 -> 0.2467 (+0.0267) | 0.2200 -> 0.2467 (+0.0267) | 0.2147 -> 0.2523 (+0.0376) | 0.5851 -> 0.5835 (-0.0015) |
| openml_blood_transfusion | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7600 -> 0.7800 (+0.0200) | 0.6507 -> 0.6848 (+0.0341) | 0.6646 -> 0.7013 (+0.0367) | 0.7753 -> 0.7734 (-0.0019) |
| openml_blood_transfusion | 44 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7600 -> 0.7800 (+0.0200) | 0.6507 -> 0.6848 (+0.0341) | 0.6646 -> 0.7013 (+0.0367) | 0.7753 -> 0.7734 (-0.0019) |
| openml_blood_transfusion | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7600 -> 0.7800 (+0.0200) | 0.6507 -> 0.6848 (+0.0341) | 0.6646 -> 0.7013 (+0.0367) | 0.7753 -> 0.7734 (-0.0019) |
| openml_ilpd | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6712 -> 0.6849 (+0.0137) | 0.5634 -> 0.5943 (+0.0309) | 0.5656 -> 0.5977 (+0.0321) | 0.7337 -> 0.7349 (+0.0011) |
| openml_ilpd | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6712 -> 0.6849 (+0.0137) | 0.5634 -> 0.5943 (+0.0309) | 0.5656 -> 0.5977 (+0.0321) | 0.7337 -> 0.7349 (+0.0011) |
| openml_ilpd | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6712 -> 0.6849 (+0.0137) | 0.5634 -> 0.5943 (+0.0309) | 0.5656 -> 0.5977 (+0.0321) | 0.7337 -> 0.7340 (+0.0002) |
| openml_credit_g | 46 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6333 (+0.0200) | 0.6133 -> 0.6333 (+0.0200) | 0.6131 -> 0.6320 (+0.0190) | 0.6999 -> 0.7095 (+0.0096) |
| openml_credit_g | 46 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6333 (+0.0200) | 0.6133 -> 0.6333 (+0.0200) | 0.6131 -> 0.6320 (+0.0190) | 0.6999 -> 0.7095 (+0.0096) |
| openml_credit_g | 46 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.6133 -> 0.6333 (+0.0200) | 0.6133 -> 0.6333 (+0.0200) | 0.6131 -> 0.6320 (+0.0190) | 0.6999 -> 0.7095 (+0.0096) |
| openml_dmft | 45 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2067 -> 0.2200 (+0.0133) | 0.2067 -> 0.2200 (+0.0133) | 0.1844 -> 0.2040 (+0.0196) | 0.6031 -> 0.5974 (-0.0057) |
| openml_dmft | 45 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2067 -> 0.2200 (+0.0133) | 0.2067 -> 0.2200 (+0.0133) | 0.1844 -> 0.2040 (+0.0196) | 0.6031 -> 0.5972 (-0.0059) |
| openml_dmft | 45 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.2067 -> 0.2200 (+0.0133) | 0.2067 -> 0.2200 (+0.0133) | 0.1844 -> 0.2040 (+0.0196) | 0.6031 -> 0.5975 (-0.0056) |
| openml_cmc | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5267 -> 0.5400 (+0.0133) | 0.5267 -> 0.5400 (+0.0133) | 0.5271 -> 0.5410 (+0.0139) | 0.7481 -> 0.7494 (+0.0013) |
| openml_cmc | 43 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5267 -> 0.5400 (+0.0133) | 0.5267 -> 0.5400 (+0.0133) | 0.5271 -> 0.5410 (+0.0139) | 0.7481 -> 0.7491 (+0.0010) |
| openml_cmc | 43 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.5267 -> 0.5400 (+0.0133) | 0.5267 -> 0.5400 (+0.0133) | 0.5271 -> 0.5410 (+0.0139) | 0.7481 -> 0.7493 (+0.0013) |
| openml_cmc | 45 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5600 -> 0.5733 (+0.0133) | 0.5600 -> 0.5733 (+0.0133) | 0.5628 -> 0.5751 (+0.0124) | 0.7563 -> 0.7547 (-0.0016) |
| openml_cmc | 45 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5600 -> 0.5733 (+0.0133) | 0.5600 -> 0.5733 (+0.0133) | 0.5628 -> 0.5751 (+0.0124) | 0.7563 -> 0.7557 (-0.0006) |
| openml_cmc | 45 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5600 -> 0.5733 (+0.0133) | 0.5600 -> 0.5733 (+0.0133) | 0.5628 -> 0.5751 (+0.0124) | 0.7563 -> 0.7545 (-0.0018) |
| openml_blood_transfusion | 45 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6587 (+0.0095) | 0.6593 -> 0.6711 (+0.0118) | 0.7164 -> 0.7230 (+0.0066) |
| openml_blood_transfusion | 45 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6587 (+0.0095) | 0.6593 -> 0.6711 (+0.0118) | 0.7164 -> 0.7230 (+0.0066) |
| openml_blood_transfusion | 45 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7400 -> 0.7533 (+0.0133) | 0.6492 -> 0.6587 (+0.0095) | 0.6593 -> 0.6711 (+0.0118) | 0.7164 -> 0.7230 (+0.0066) |
| openml_blood_transfusion | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7877 (+0.0015) |
| openml_blood_transfusion | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7877 (+0.0015) |
| openml_blood_transfusion | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7800 -> 0.7867 (+0.0067) | 0.6715 -> 0.6829 (+0.0114) | 0.6894 -> 0.7019 (+0.0124) | 0.7862 -> 0.7875 (+0.0013) |
| openml_kc1 | 45 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7133 -> 0.7200 (+0.0067) | 0.7133 -> 0.7200 (+0.0067) | 0.7112 -> 0.7192 (+0.0080) | 0.8075 -> 0.7996 (-0.0078) |
| openml_kc1 | 45 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7133 -> 0.7200 (+0.0067) | 0.7133 -> 0.7200 (+0.0067) | 0.7112 -> 0.7192 (+0.0080) | 0.8075 -> 0.7996 (-0.0078) |
| openml_kc1 | 45 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7133 -> 0.7200 (+0.0067) | 0.7133 -> 0.7200 (+0.0067) | 0.7112 -> 0.7192 (+0.0080) | 0.8075 -> 0.7998 (-0.0076) |
| openml_credit_g | 42 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7264 (+0.0072) | 0.8028 -> 0.8085 (+0.0057) |
| openml_credit_g | 42 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7264 (+0.0072) | 0.8028 -> 0.8085 (+0.0057) |
| openml_credit_g | 42 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7200 -> 0.7267 (+0.0067) | 0.7200 -> 0.7267 (+0.0067) | 0.7192 -> 0.7264 (+0.0072) | 0.8028 -> 0.8085 (+0.0057) |
| openml_ilpd | 44 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7260 -> 0.7329 (+0.0068) | 0.5593 -> 0.5641 (+0.0048) | 0.5456 -> 0.5505 (+0.0049) | 0.8088 -> 0.7933 (-0.0156) |
| openml_ilpd | 44 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7260 -> 0.7329 (+0.0068) | 0.5593 -> 0.5641 (+0.0048) | 0.5456 -> 0.5505 (+0.0049) | 0.8088 -> 0.7933 (-0.0156) |
| openml_ilpd | 44 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.7260 -> 0.7329 (+0.0068) | 0.5593 -> 0.5641 (+0.0048) | 0.5456 -> 0.5505 (+0.0049) | 0.8088 -> 0.7935 (-0.0153) |
| openml_credit_g | 45 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6933 -> 0.7000 (+0.0067) | 0.6933 -> 0.7000 (+0.0067) | 0.6928 -> 0.6997 (+0.0068) | 0.7813 -> 0.7780 (-0.0034) |
| openml_credit_g | 45 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6933 -> 0.7000 (+0.0067) | 0.6933 -> 0.7000 (+0.0067) | 0.6928 -> 0.6997 (+0.0068) | 0.7813 -> 0.7780 (-0.0034) |
| openml_credit_g | 45 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.6933 -> 0.7000 (+0.0067) | 0.6933 -> 0.7000 (+0.0067) | 0.6928 -> 0.6997 (+0.0068) | 0.7813 -> 0.7783 (-0.0030) |
| openml_credit_g | 43 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7000 -> 0.7067 (+0.0067) | 0.7000 -> 0.7067 (+0.0067) | 0.6993 -> 0.7058 (+0.0065) | 0.8048 -> 0.8069 (+0.0021) |
| openml_credit_g | 43 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7000 -> 0.7067 (+0.0067) | 0.7000 -> 0.7067 (+0.0067) | 0.6993 -> 0.7058 (+0.0065) | 0.8048 -> 0.8069 (+0.0021) |
| openml_credit_g | 43 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7000 -> 0.7067 (+0.0067) | 0.7000 -> 0.7067 (+0.0067) | 0.6993 -> 0.7058 (+0.0065) | 0.8048 -> 0.8071 (+0.0023) |
| openml_cmc | 46 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5299 -> 0.5346 (+0.0047) | 0.7278 -> 0.7233 (-0.0045) |
| openml_cmc | 46 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5299 -> 0.5346 (+0.0047) | 0.7278 -> 0.7212 (-0.0066) |
| openml_cmc | 46 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1 | 0.5267 -> 0.5333 (+0.0067) | 0.5267 -> 0.5333 (+0.0067) | 0.5299 -> 0.5346 (+0.0047) | 0.7278 -> 0.7233 (-0.0045) |
| openml_blood_transfusion | 46 | ensemble_n4 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7533 -> 0.7600 (+0.0067) | 0.6270 -> 0.6317 (+0.0048) | 0.6367 -> 0.6425 (+0.0058) | 0.7582 -> 0.7620 (+0.0038) |
| openml_blood_transfusion | 46 | ensemble_n4_temp07 | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7533 -> 0.7600 (+0.0067) | 0.6270 -> 0.6317 (+0.0048) | 0.6367 -> 0.6425 (+0.0058) | 0.7582 -> 0.7620 (+0.0038) |
| openml_blood_transfusion | 46 | ensemble_n4_avg_probs | accuracy, balanced_accuracy, macro_f1, weighted_f1, roc_auc | 0.7533 -> 0.7600 (+0.0067) | 0.6270 -> 0.6317 (+0.0048) | 0.6367 -> 0.6425 (+0.0058) | 0.7582 -> 0.7618 (+0.0036) |
| openml_blood_transfusion | 43 | ensemble_n4 | roc_auc | 0.7467 -> 0.7333 (-0.0133) | 0.6214 -> 0.6053 (-0.0161) | 0.6309 -> 0.6115 (-0.0194) | 0.7340 -> 0.7390 (+0.0049) |
| openml_blood_transfusion | 43 | ensemble_n4_temp07 | roc_auc | 0.7467 -> 0.7333 (-0.0133) | 0.6214 -> 0.6053 (-0.0161) | 0.6309 -> 0.6115 (-0.0194) | 0.7340 -> 0.7390 (+0.0049) |
| openml_blood_transfusion | 43 | ensemble_n4_avg_probs | roc_auc | 0.7467 -> 0.7333 (-0.0133) | 0.6214 -> 0.6053 (-0.0161) | 0.6309 -> 0.6115 (-0.0194) | 0.7340 -> 0.7390 (+0.0049) |
| openml_kc1 | 44 | ensemble_n4 | roc_auc | 0.7733 -> 0.7733 (+0.0000) | 0.7733 -> 0.7733 (+0.0000) | 0.7727 -> 0.7723 (-0.0004) | 0.8645 -> 0.8676 (+0.0030) |
| openml_kc1 | 44 | ensemble_n4_temp07 | roc_auc | 0.7733 -> 0.7733 (+0.0000) | 0.7733 -> 0.7733 (+0.0000) | 0.7727 -> 0.7723 (-0.0004) | 0.8645 -> 0.8676 (+0.0030) |
| openml_kc1 | 44 | ensemble_n4_avg_probs | roc_auc | 0.7733 -> 0.7733 (+0.0000) | 0.7733 -> 0.7733 (+0.0000) | 0.7727 -> 0.7723 (-0.0004) | 0.8645 -> 0.8676 (+0.0030) |
| openml_ilpd | 43 | ensemble_n4_avg_probs | roc_auc | 0.7055 -> 0.6986 (-0.0068) | 0.5662 -> 0.5401 (-0.0261) | 0.5648 -> 0.5267 (-0.0382) | 0.7818 -> 0.7823 (+0.0005) |
