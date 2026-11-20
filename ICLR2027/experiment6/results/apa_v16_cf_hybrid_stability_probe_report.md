# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_balanced,openml_kc1,openml_diabetes_pima,openml_credit_g --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_tempered_router,apa_cf_hybrid_stability_router --output-prefix apa_v16_cf_hybrid_stability_probe
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_hybrid_stability_router | 48 | +0.0247 | +0.0130 | +0.0187 | +0.0000 | +0.0013 |
| apa_cf_hybrid_tempered_router | 48 | +0.0267 | +0.0109 | +0.0184 | -0.0069 | -0.0013 |
| tabicl_baseline | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_balanced` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0826, minority recall +0.0000, ECE -0.0096.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0826, minority recall +0.0000, ECE -0.0096.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc -0.1494, macro-F1 -0.0467, minority recall -0.3333, ECE -0.0054.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0511, minority recall +0.0000, ECE +0.0295.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0588, minority recall +0.0000, ECE +0.0313.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0588, minority recall +0.0000, ECE +0.0313.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0424, minority recall +0.0000, ECE +0.0336.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0424, minority recall +0.0000, ECE +0.0336.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0588, minority recall +0.0000, ECE +0.0313.
- `synthetic_balanced` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0588, minority recall +0.0000, ECE +0.0313.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0424, minority recall +0.0000, ECE +0.0336.
- `synthetic_balanced` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0424, minority recall +0.0000, ECE +0.0336.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0938, bacc +0.0517, macro-F1 +0.0678, minority recall +0.0000, ECE -0.0208.
- `openml_kc1` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0938, bacc +0.0517, macro-F1 +0.0678, minority recall +0.0000, ECE -0.0208.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0005.
- `openml_kc1` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0005.
- `openml_kc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0355, minority recall +0.0000, ECE -0.0427.
- `openml_kc1` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0938, bacc +0.0517, macro-F1 +0.0529, minority recall +0.0000, ECE -0.0098.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0278, minority recall +0.0000, ECE -0.0611.
- `openml_kc1` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0278, minority recall +0.0000, ECE -0.0611.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0320.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE +0.0298.
- `openml_kc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0278, minority recall +0.0000, ECE -0.0611.
- `openml_kc1` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0278, minority recall +0.0000, ECE -0.0611.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0487, minority recall +0.0000, ECE -0.0358.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0239, minority recall +0.0000, ECE -0.0059.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0134.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0134.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0546, minority recall +0.0000, ECE -0.0056.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0152.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0546, minority recall +0.0000, ECE -0.0056.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE +0.0152.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0352.
- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 2 `apa_cf_hybrid_stability_router` (apa_v16): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
