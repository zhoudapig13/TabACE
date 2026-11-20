# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_1480_ilpd --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_cross_model_tree_rescue_router,apa_cf_no_shift_prp_tree_rescue_router --output-prefix apa_v56_prp_tree_rescue_ilpd_2seed_train128
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_boundary_bypass_combo_router | 2 | +0.0156 | +0.0173 | +0.0200 | +0.0227 | +0.0033 | -0.0165 |
| apa_cf_no_shift_cross_model_tree_rescue_router | 2 | +0.0156 | +0.0227 | +0.0270 | +0.0455 | +0.0028 | -0.0169 |
| apa_cf_no_shift_prp_tree_rescue_router | 2 | +0.0156 | +0.0227 | +0.0270 | +0.0455 | +0.0028 | -0.0169 |
| tabicl_baseline | 2 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0012, ECE -0.0008.
- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_cross_model_tree_rescue_router` (apa_v50): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0012, ECE -0.0008.
- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_prp_tree_rescue_router` (apa_v56): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0012, ECE -0.0008.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_boundary_bypass_combo_router` (apa_v42): shift , acc +0.0312, bacc +0.0346, macro-F1 +0.0401, minority recall +0.0455, ROC-AUC +0.0054, ECE -0.0323.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_cross_model_tree_rescue_router` (apa_v50): shift , acc +0.0312, bacc +0.0455, macro-F1 +0.0541, minority recall +0.0909, ROC-AUC +0.0043, ECE -0.0331.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_prp_tree_rescue_router` (apa_v56): shift , acc +0.0312, bacc +0.0455, macro-F1 +0.0541, minority recall +0.0909, ROC-AUC +0.0043, ECE -0.0331.
