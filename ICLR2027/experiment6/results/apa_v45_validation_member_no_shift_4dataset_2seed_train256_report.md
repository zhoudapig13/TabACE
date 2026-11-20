# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_1498_sa_heart --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_validation_member_selector_router --output-prefix apa_v45_validation_member_no_shift_4dataset_2seed_train256
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_boundary_bypass_combo_router | 8 | +0.0098 | +0.0087 | +0.0102 | +0.0063 | +0.0023 | +0.0052 |
| apa_cf_no_shift_validation_member_selector_router | 8 | +0.0078 | +0.0055 | +0.0072 | -0.0001 | +0.0022 | +0.0071 |
| tabicl_baseline | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0156, bacc +0.0109, macro-F1 +0.0138, minority recall +0.0000, ROC-AUC +0.0024, ECE -0.0041.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0156, bacc +0.0109, macro-F1 +0.0138, minority recall +0.0000, ROC-AUC +0.0024, ECE -0.0041.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc -0.0156, bacc -0.0256, macro-F1 -0.0237, minority recall -0.0513, ROC-AUC -0.0012, ECE +0.0154.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0078, bacc +0.0050, macro-F1 +0.0081, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0077.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0078, bacc +0.0050, macro-F1 +0.0081, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0077.
- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0078, bacc +0.0058, macro-F1 +0.0062, minority recall +0.0000, ROC-AUC +0.0008, ECE -0.0079.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0078, bacc +0.0058, macro-F1 +0.0062, minority recall +0.0000, ROC-AUC +0.0008, ECE -0.0079.
- `openml_1498_sa_heart` seed 1 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0312, bacc +0.0297, macro-F1 +0.0339, minority recall +0.0256, ROC-AUC +0.0104, ECE +0.0307.
- `openml_1498_sa_heart` seed 1 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0312, bacc +0.0297, macro-F1 +0.0339, minority recall +0.0256, ROC-AUC +0.0104, ECE +0.0307.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0156, bacc +0.0182, macro-F1 +0.0193, minority recall +0.0250, ROC-AUC +0.0051, ECE +0.0150.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_validation_member_selector_router` (apa_v45): shift , acc +0.0156, bacc +0.0182, macro-F1 +0.0193, minority recall +0.0250, ROC-AUC +0.0051, ECE +0.0150.
