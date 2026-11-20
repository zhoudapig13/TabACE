# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,breast_cancer,wine,openml_blood_transfusion,openml_diabetes_pima --seeds 0,1 --n-samples 140 --train-size 70 --test-size 24 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_directional_safety
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_directional_router | 10 | +0.0000 | +0.0000 | -0.0015 | +0.0083 | +0.0003 |
| apa_dual_shift_router_v7 | 10 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| tabicl_baseline | 10 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_imbalanced` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 0 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 1 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 0 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 1 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 0 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_directional_router` (apa_v8): shift , acc -0.0417, bacc -0.0417, macro-F1 -0.0478, minority recall +0.0000, ECE -0.0405.
- `openml_blood_transfusion` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_directional_router` (apa_v8): shift , acc +0.0417, bacc +0.0417, macro-F1 +0.0324, minority recall +0.0833, ECE +0.0430.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
