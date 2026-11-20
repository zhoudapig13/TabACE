# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1498_sa_heart --force-label-shift --label-shift-strengths 0.6,0.9 --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_dual_mode_v38,apa_cf_mixed_v38_v50_router --output-prefix apa_v57_mixed_v38_v50_forced_shift_target3_2seed
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_dual_mode_v38 | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | +0.0000 | -0.0065 |
| apa_cf_mixed_v38_v50_router | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | +0.0000 | -0.0065 |
| tabicl_baseline | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_dual_mode_v38` (apa_v38): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_dual_mode_v38` (apa_v38): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_31_credit_g` seed 2 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_31_credit_g` seed 1 `apa_cf_dual_mode_v38` (apa_v38): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_dual_mode_v38` (apa_v38): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_31_credit_g` seed 2 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_37_diabetes` seed 1 `apa_cf_dual_mode_v38` (apa_v38): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 1 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 2 `apa_cf_dual_mode_v38` (apa_v38): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_37_diabetes` seed 2 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_37_diabetes` seed 1 `apa_cf_dual_mode_v38` (apa_v38): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 1 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 2 `apa_cf_dual_mode_v38` (apa_v38): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_37_diabetes` seed 2 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_1498_sa_heart` seed 1 `apa_cf_dual_mode_v38` (apa_v38): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 1 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 2 `apa_cf_dual_mode_v38` (apa_v38): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0411, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0570.
- `openml_1498_sa_heart` seed 2 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0411, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0570.
- `openml_1498_sa_heart` seed 1 `apa_cf_dual_mode_v38` (apa_v38): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 1 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 2 `apa_cf_dual_mode_v38` (apa_v38): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0411, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0570.
- `openml_1498_sa_heart` seed 2 `apa_cf_mixed_v38_v50_router` (apa_v57): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0411, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0570.
