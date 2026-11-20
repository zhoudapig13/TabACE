# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_diabetes_pima --force-label-shift --label-shift-strengths 0.0,0.6,0.9 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_dual_shift_router_v7,apa_cf_directional_router,apa_cf_strict_margin_router,apa_cf_batch_margin_router,apa_cf_tempered_rescue_router,apa_cf_hybrid_tempered_router,apa_cf_hybrid_calibrated_router,apa_cf_hybrid_stability_router --output-prefix apa_v1_v16_fixed11_selected_benchmark_by_dataset_openml_diabetes_pima
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_batch_margin_router | 6 | +0.0625 | +0.0323 | +0.0365 | +0.0000 | +0.0046 |
| apa_cf_directional_router | 6 | +0.0729 | +0.0376 | +0.0426 | +0.0000 | +0.0012 |
| apa_cf_hybrid_calibrated_router | 6 | +0.0521 | +0.0269 | +0.0306 | +0.0000 | -0.0066 |
| apa_cf_hybrid_stability_router | 6 | +0.0417 | +0.0215 | +0.0240 | +0.0000 | +0.0032 |
| apa_cf_hybrid_tempered_router | 6 | +0.0625 | +0.0323 | +0.0365 | +0.0000 | +0.0046 |
| apa_cf_strict_margin_router | 6 | +0.0104 | +0.0054 | +0.0059 | +0.0000 | -0.0101 |
| apa_cf_tempered_rescue_router | 6 | +0.0208 | +0.0108 | +0.0117 | +0.0000 | -0.0050 |
| apa_dual_shift_router_v7 | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0064 |
| apa_full_prior_aware_tiny | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0002 |
| tabicl_baseline | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_diabetes_pima` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_tempered_rescue_router` (apa_v13): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_tempered_rescue_router` (apa_v13): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0001.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0034.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0177, minority recall +0.0000, ECE -0.0303.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_tempered_rescue_router` (apa_v13): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0177, minority recall +0.0000, ECE -0.0304.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0390.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0546, minority recall +0.0000, ECE -0.0056.
- `openml_diabetes_pima` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0006.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0225.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_tempered_rescue_router` (apa_v13): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0152.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0192.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0152.
- `openml_diabetes_pima` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0001.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0034.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0177, minority recall +0.0000, ECE -0.0303.
- `openml_diabetes_pima` seed 0 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_tempered_rescue_router` (apa_v13): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0177, minority recall +0.0000, ECE -0.0304.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0390.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0546, minority recall +0.0000, ECE -0.0056.
- `openml_diabetes_pima` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0006.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0225.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_diabetes_pima` seed 1 `apa_cf_strict_margin_router` (apa_v9): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_batch_margin_router` (apa_v10): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_tempered_rescue_router` (apa_v13): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0152.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0192.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0152.
