# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_credit_g,openml_diabetes_pima,openml_kc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_directional_router,apa_cf_strict_margin_router,apa_cf_batch_margin_router --output-prefix apa_v10_cf_batch_margin_probe
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_batch_margin_router | 48 | +0.0117 | +0.0061 | +0.0071 | +0.0000 | -0.0007 |
| apa_cf_directional_router | 48 | +0.0326 | -0.0064 | +0.0159 | -0.0473 | -0.0044 |
| apa_cf_strict_margin_router | 48 | +0.0059 | +0.0031 | +0.0037 | +0.0000 | -0.0036 |
| tabicl_baseline | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0218, minority recall +0.0000, ECE -0.0398.
- `openml_credit_g` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0218, minority recall +0.0000, ECE +0.0064.
- `openml_credit_g` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc -0.1494, macro-F1 -0.0423, minority recall -0.3333, ECE +0.0590.
- `openml_credit_g` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0204, minority recall +0.0000, ECE +0.0065.
- `openml_credit_g` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0008.
- `openml_credit_g` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc -0.0312, bacc -0.0312, macro-F1 -0.0367, minority recall +0.0625, ECE +0.0100.
- `openml_diabetes_pima` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0487, minority recall +0.0000, ECE -0.0358.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0239, minority recall +0.0000, ECE -0.0299.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0487, minority recall +0.0000, ECE -0.0358.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0453, minority recall +0.0000, ECE -0.0503.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0015.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0408, minority recall +0.0000, ECE -0.0221.
- `openml_diabetes_pima` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0460, minority recall +0.0000, ECE -0.0362.
- `openml_diabetes_pima` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0228, minority recall +0.0000, ECE -0.0049.
- `openml_diabetes_pima` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0177, minority recall +0.0000, ECE -0.0303.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc -0.0312, bacc -0.5000, macro-F1 -0.0659, minority recall -1.0000, ECE -0.0038.
- `openml_diabetes_pima` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0522, minority recall +0.0000, ECE +0.0170.
- `openml_diabetes_pima` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0296.
- `openml_diabetes_pima` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0177, minority recall +0.0000, ECE -0.0303.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc -0.0312, bacc -0.5000, macro-F1 -0.0659, minority recall -1.0000, ECE -0.0038.
- `openml_diabetes_pima` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0522, minority recall +0.0000, ECE +0.0170.
- `openml_diabetes_pima` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0296.
- `openml_diabetes_pima` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0938, bacc +0.0517, macro-F1 +0.0678, minority recall +0.0000, ECE -0.0208.
- `openml_kc1` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0005.
- `openml_kc1` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.1250, bacc +0.0690, macro-F1 +0.0703, minority recall +0.0000, ECE -0.0033.
- `openml_kc1` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0304.
- `openml_kc1` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0409, minority recall +0.0000, ECE -0.0677.
- `openml_kc1` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0409, minority recall +0.0000, ECE -0.0677.
- `openml_kc1` seed 2 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
