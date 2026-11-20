# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_credit_g,openml_diabetes_pima,openml_kc1 --force-label-shift --label-shift-strengths 0.0,0.6,0.9 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_tempered_router,apa_cf_dual_mode_v35,apa_cf_dual_mode_v36 --output-prefix apa_v36_dual_mode_forced_shift_smoke
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_dual_mode_v35 | 18 | +0.0365 | +0.0197 | +0.0221 | +0.0035 | +0.0002 | +0.0011 |
| apa_cf_dual_mode_v36 | 18 | +0.0365 | +0.0197 | +0.0221 | +0.0035 | +0.0002 | +0.0011 |
| apa_cf_hybrid_tempered_router | 18 | +0.0347 | +0.0179 | +0.0201 | +0.0000 | +0.0000 | -0.0002 |
| tabicl_baseline | 18 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.0000, acc +0.0312, bacc +0.0312, macro-F1 +0.0356, minority recall +0.0625, ROC-AUC +0.0039, ECE +0.0236.
- `openml_credit_g` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.0000, acc +0.0312, bacc +0.0312, macro-F1 +0.0356, minority recall +0.0625, ROC-AUC +0.0039, ECE +0.0236.
- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0129.
- `openml_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0129.
- `openml_credit_g` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0129.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0496.
- `openml_diabetes_pima` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0351, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0496.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0320.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0298.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v35` (apa_v35): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0320.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0320.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v35` (apa_v35): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0298.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0298.
