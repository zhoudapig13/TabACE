# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1050_pc3,openml_1498_sa_heart --split-mode ordered --seeds 2,3 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_guarded_boundary_union_router,apa_cf_no_shift_pvalue_guarded_boundary_router --output-prefix apa_v33_pvalue_guard_smoke
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_guarded_boundary_union_router | 8 | +0.0078 | +0.0074 | +0.0074 | +0.0054 | +0.0020 | -0.0019 |
| apa_cf_no_shift_pvalue_guarded_boundary_router | 8 | +0.0059 | +0.0058 | +0.0057 | +0.0054 | +0.0017 | +0.0000 |
| tabicl_baseline | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0312, bacc +0.0250, macro-F1 +0.0235, minority recall +0.0000, ROC-AUC +0.0104, ECE +0.0141.
- `openml_31_credit_g` seed 3 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0312, bacc +0.0250, macro-F1 +0.0235, minority recall +0.0000, ROC-AUC +0.0104, ECE +0.0141.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0312, bacc +0.0339, macro-F1 +0.0359, minority recall +0.0435, ROC-AUC +0.0053, ECE -0.0296.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0156, bacc +0.0217, macro-F1 +0.0220, minority recall +0.0435, ROC-AUC +0.0032, ECE -0.0140.
- `openml_37_diabetes` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 3 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 2 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 3 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 3 `apa_cf_no_shift_pvalue_guarded_boundary_router` (apa_v33): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
