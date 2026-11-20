# Fixed-gate weighting control — TabICL-v2

New cached-bank reconstruction diagnostic; these are not manuscript reproduction scores.
Five alternative contexts plus Base; all weighting variants use exactly identical per-query alpha.
Five dataset-level outer folds; single seed 0; no new backbone inference or benchmark modification.
Classification compares static learned, confidence-only, and full confidence/agreement weighting.
Regression compares static learned and full agreement weighting; its paper formula has no standalone confidence term.
The gate is an identical frozen reference gate and retains its evidence signals for every method.
Only fusion weighting changes. Clean is evaluated separately and is included in calibration with the existing track weights.

| Task | Track | Datasets | Method | Metric 1 | Metric 2 |
|---|---|---:|---|---:|---:|
| classification | five | 72 | Base | 0.667632 | 0.824631 |
| classification | five | 72 | Original_full_reference | 0.675532 | 0.832237 |
| classification | five | 72 | Static_learned | 0.675532 | 0.832647 |
| classification | five | 72 | Confidence_only | 0.675559 | 0.832651 |
| classification | five | 72 | Full_weighting | 0.675548 | 0.832658 |
| classification | seven_reimplemented | 35 | Base | 0.752782 | 0.812457 |
| classification | seven_reimplemented | 35 | Original_full_reference | 0.753101 | 0.813702 |
| classification | seven_reimplemented | 35 | Static_learned | 0.753021 | 0.813697 |
| classification | seven_reimplemented | 35 | Confidence_only | 0.753021 | 0.813697 |
| classification | seven_reimplemented | 35 | Full_weighting | 0.753021 | 0.813654 |
| regression | five | 35 | Base | -1.107163 | 1.297429 |
| regression | five | 35 | Original_full_reference | -0.907185 | 1.263509 |
| regression | five | 35 | Static_learned | -0.871822 | 1.269296 |
| regression | five | 35 | Full_weighting | -0.872535 | 1.269410 |
| regression | seven_reimplemented | 33 | Base | -0.614747 | 0.944044 |
| regression | seven_reimplemented | 33 | Original_full_reference | 0.661905 | 0.529576 |
| regression | seven_reimplemented | 33 | Static_learned | 0.685260 | 0.506285 |
| regression | seven_reimplemented | 33 | Full_weighting | 0.685210 | 0.506331 |

Classification: Acc / AUC. Regression: R² / nRMSE.
Paired dataset bootstrap intervals and Holm-corrected sign-flip tests are in paired_dataset_comparisons.csv.
No outcome ordering was enforced, and no final evaluation label was used to fit or select weights.
