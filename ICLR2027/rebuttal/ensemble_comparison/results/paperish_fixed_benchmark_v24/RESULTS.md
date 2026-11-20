# Frozen-benchmark Shared / Specific replay

Benchmark inputs and Base predictions are held fixed. Seed = 0. Evaluation strengths are {0.2, 0.5, 0.8, 1.0}. Classification uses 72 datasets for Clean/five corruptions and 35 for seven shifts; regression uses 35 and 33, respectively.

Observed-only uses no clean reference rows. Reference005 and reference020 replace at most 5% and 20% of the rows within each Shared view using clean FT TRAIN examples disjoint from calibration and final evaluation rows. This is additional clean-reference supervision for the frozen predictor. All context sizes remain at most 256. Shared has five alternative views; Specific inherits them and adds five recovered family-directed views. The backbone stays frozen and the original fusion gate is used.

Fusion policies are fitted with five-fold dataset-level OOF calibration (available calibration strengths: clean, 0.5 and 1.0). The reference fractions are fixed conditions, not selected using final query scores. All completed conditions are reported.

This is a controlled reconstruction on the frozen Table 28 development benchmark, rather than exact archived main-table replication. Regression Clean/five remain the existing semi-synthetic tasks. The original complete deployed Specific lists have not been recovered; this run uses ten reconstructed alternative views. Prior adapted Table 28 and rebuttal/PDF files are preserved.

## observed_only

| Task | Method | Clean | Five corruptions | Seven shifts |
|---|---|---|---|---|
| classification | Base | 0.8033 / 0.9007 | 0.4973 / 0.6729 | 0.7634 / 0.8246 |
| classification | Shared | 0.8028 / 0.9007 | 0.4967 / 0.6723 | 0.7633 / 0.8248 |
| classification | Specific | 0.8039 / 0.9006 | 0.4973 / 0.6731 | 0.7676 / 0.8261 |
| regression | Base | 0.7425 / 0.4803 | -1.0763 / 1.2685 | 0.2746 / 0.7288 |
| regression | Shared | 0.7356 / 0.4876 | -0.9255 / 1.2288 | 0.2601 / 0.7433 |
| regression | Specific | 0.7425 / 0.4803 | -0.3820 / 1.0562 | 0.3260 / 0.7118 |

## reference005

| Task | Method | Clean | Five corruptions | Seven shifts |
|---|---|---|---|---|
| classification | Base | 0.8033 / 0.9007 | 0.4973 / 0.6729 | 0.7634 / 0.8246 |
| classification | Shared | 0.8034 / 0.9011 | 0.5001 / 0.6807 | 0.7634 / 0.8260 |
| classification | Specific | 0.8039 / 0.9008 | 0.4988 / 0.6800 | 0.7680 / 0.8262 |
| regression | Base | 0.7425 / 0.4803 | -1.0763 / 1.2685 | 0.2746 / 0.7288 |
| regression | Shared | 0.7241 / 0.4978 | -0.4678 / 1.0770 | 0.4895 / 0.6572 |
| regression | Specific | 0.7425 / 0.4803 | -0.2607 / 1.0170 | 0.4066 / 0.6838 |

## reference020

| Task | Method | Clean | Five corruptions | Seven shifts |
|---|---|---|---|---|
| classification | Base | 0.8033 / 0.9007 | 0.4973 / 0.6729 | 0.7634 / 0.8246 |
| classification | Shared | 0.8046 / 0.9017 | 0.5082 / 0.6929 | 0.7652 / 0.8277 |
| classification | Specific | 0.8038 / 0.9013 | 0.5053 / 0.6918 | 0.7683 / 0.8267 |
| regression | Base | 0.7425 / 0.4803 | -1.0763 / 1.2685 | 0.2746 / 0.7288 |
| regression | Shared | 0.7288 / 0.4933 | -0.0191 / 0.9046 | 0.5880 / 0.5898 |
| regression | Specific | 0.7424 / 0.4804 | -0.1583 / 0.9751 | 0.4410 / 0.6663 |
