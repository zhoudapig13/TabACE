# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_881_mv,openml_37_diabetes,openml_151_electricity --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_diverse_router,apa_cf_no_shift_soft_diverse_router,apa_cf_dual_mode_router_v17 --output-prefix apa_v22_soft_diverse_diagnostic
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_dual_mode_router_v17 | 8 | +0.0020 | +0.0023 | +0.0024 | +0.0046 | +0.0020 |
| apa_cf_no_shift_diverse_router | 8 | +0.0020 | +0.0023 | +0.0024 | +0.0046 | +0.0020 |
| apa_cf_no_shift_soft_diverse_router | 8 | +0.0039 | +0.0042 | +0.0045 | +0.0054 | -0.0033 |
| tabicl_baseline | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0030.
- `openml_31_credit_g` seed 2 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 1 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 1 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 1 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 2 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 2 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 2 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0156, bacc +0.0185, macro-F1 +0.0193, minority recall +0.0370, ECE +0.0162.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0008.
- `openml_37_diabetes` seed 1 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0156, bacc +0.0185, macro-F1 +0.0193, minority recall +0.0370, ECE +0.0162.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0312, bacc +0.0339, macro-F1 +0.0359, minority recall +0.0435, ECE -0.0302.
- `openml_37_diabetes` seed 2 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_151_electricity` seed 1 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 1 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 1 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 2 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 2 `apa_cf_no_shift_soft_diverse_router` (apa_v22): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 2 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
