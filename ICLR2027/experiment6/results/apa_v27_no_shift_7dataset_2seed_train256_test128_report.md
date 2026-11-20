# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman,openml_1498_sa_heart,openml_1464_blood_transfusion,openml_1050_pc3 --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_guarded_boundary_union_router --output-prefix apa_v27_no_shift_7dataset_2seed_train256_test128
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_guarded_boundary_union_router | 14 | +0.0010 | -0.0042 | -0.0082 | -0.0174 | +0.0010 | +0.0007 |
| tabicl_baseline | 14 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc -0.0144, macro-F1 -0.0103, minority recall -0.0513, ROC-AUC +0.0012, ECE +0.0011.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0156, bacc +0.0109, macro-F1 +0.0138, minority recall +0.0000, ROC-AUC +0.0024, ECE -0.0041.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc -0.0078, bacc -0.0128, macro-F1 -0.0117, minority recall -0.0256, ROC-AUC -0.0012, ECE +0.0083.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc -0.0156, bacc -0.0229, macro-F1 -0.0220, minority recall -0.0357, ROC-AUC -0.0039, ECE -0.0085.
- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc -0.0129, macro-F1 -0.0159, minority recall -0.0488, ROC-AUC +0.0011, ECE +0.0010.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0156, bacc +0.0116, macro-F1 +0.0124, minority recall +0.0000, ROC-AUC +0.0028, ECE -0.0153.
- `openml_43_haberman` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc -0.0200, bacc -0.0333, macro-F1 -0.0674, minority recall -0.0667, ROC-AUC -0.0019, ECE -0.0092.
- `openml_43_haberman` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc -0.0200, bacc -0.0333, macro-F1 -0.0674, minority recall -0.0667, ROC-AUC -0.0019, ECE -0.0092.
- `openml_1498_sa_heart` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0312, bacc +0.0297, macro-F1 +0.0339, minority recall +0.0256, ROC-AUC +0.0104, ECE +0.0307.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0156, bacc +0.0182, macro-F1 +0.0193, minority recall +0.0250, ROC-AUC +0.0051, ECE +0.0150.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
