# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_37_diabetes,openml_151_electricity --split-mode ordered --seeds 2,1 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_diverse_router,apa_cf_no_shift_boundary_router,apa_cf_dual_mode_router_v17 --output-prefix apa_v21_feature_evidence_negative_probe
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_dual_mode_router_v17 | 4 | +0.0039 | +0.0046 | +0.0048 | +0.0093 | +0.0041 |
| apa_cf_no_shift_boundary_router | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0004 |
| apa_cf_no_shift_diverse_router | 4 | +0.0039 | +0.0046 | +0.0048 | +0.0093 | +0.0041 |
| tabicl_baseline | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_37_diabetes` seed 2 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `openml_37_diabetes` seed 2 `apa_cf_no_shift_boundary_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0017.
- `openml_37_diabetes` seed 2 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0156, bacc +0.0185, macro-F1 +0.0193, minority recall +0.0370, ECE +0.0162.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_boundary_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 1 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0156, bacc +0.0185, macro-F1 +0.0193, minority recall +0.0370, ECE +0.0162.
- `openml_151_electricity` seed 2 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 2 `apa_cf_no_shift_boundary_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 2 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 1 `apa_cf_no_shift_diverse_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 1 `apa_cf_no_shift_boundary_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 1 `apa_cf_dual_mode_router_v17` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
