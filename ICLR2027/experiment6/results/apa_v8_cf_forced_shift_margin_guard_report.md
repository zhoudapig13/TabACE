# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_credit_g,openml_kc1,openml_diabetes_pima --force-label-shift --label-shift-strengths 0.3,0.6,0.9 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_forced_shift_margin_guard
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_directional_router | 18 | +0.0538 | +0.0283 | +0.0336 | +0.0000 | -0.0095 |
| apa_dual_shift_router_v7 | 18 | +0.0035 | +0.0018 | +0.0020 | +0.0000 | -0.0002 |
| tabicl_baseline | 18 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0042.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0030.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0218, minority recall +0.0000, ECE -0.0398.
- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0018.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0275.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0018.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0275.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0009.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0938, bacc +0.0517, macro-F1 +0.0678, minority recall +0.0000, ECE -0.0208.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0011.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0005.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0306.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0306.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0228.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0487, minority recall +0.0000, ECE -0.0358.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0044.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0453, minority recall +0.0000, ECE -0.0503.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0034.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0225.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0034.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0225.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
