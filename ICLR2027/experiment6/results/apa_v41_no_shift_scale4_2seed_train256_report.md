# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_scale_combo_router --output-prefix apa_v41_no_shift_scale4_2seed_train256
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_scale_combo_router | 8 | +0.0020 | +0.0014 | +0.0018 | +0.0000 | +0.0001 | -0.0000 |
| tabicl_baseline | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0078, bacc +0.0050, macro-F1 +0.0081, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0077.
- `openml_1480_ilpd` seed 1 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0078, bacc +0.0058, macro-F1 +0.0062, minority recall +0.0000, ROC-AUC +0.0008, ECE -0.0079.
- `openml_43_haberman` seed 1 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 2 `apa_cf_no_shift_scale_combo_router` (apa_v41): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
