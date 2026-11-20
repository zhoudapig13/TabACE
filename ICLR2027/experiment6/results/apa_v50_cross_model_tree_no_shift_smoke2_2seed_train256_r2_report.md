# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_cross_model_tree_rescue_router | 4 | +0.0098 | +0.0106 | +0.0129 | +0.0134 | +0.0011 | +0.0050 |
| tabicl_baseline | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_no_shift_cross_model_tree_rescue_router` (apa_v50): shift , acc +0.0078, bacc +0.0128, macro-F1 +0.0116, minority recall +0.0256, ROC-AUC +0.0006, ECE +0.0081.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_cross_model_tree_rescue_router` (apa_v50): shift , acc +0.0234, bacc +0.0248, macro-F1 +0.0321, minority recall +0.0278, ROC-AUC +0.0039, ECE +0.0042.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_cross_model_tree_rescue_router` (apa_v50): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_cross_model_tree_rescue_router` (apa_v50): shift , acc +0.0078, bacc +0.0050, macro-F1 +0.0081, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0077.
