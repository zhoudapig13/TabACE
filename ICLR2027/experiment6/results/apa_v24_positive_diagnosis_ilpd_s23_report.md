# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_1480_ilpd --split-mode ordered --seeds 2,3 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_router,apa_cf_no_shift_union_router,apa_cf_no_shift_boundary_union_router --output-prefix apa_v24_positive_diagnosis_ilpd_s23
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_boundary_router | 2 | +0.0078 | +0.0053 | +0.0059 | +0.0000 | +0.0072 |
| apa_cf_no_shift_boundary_union_router | 2 | +0.0234 | +0.0226 | +0.0259 | +0.0227 | -0.0089 |
| apa_cf_no_shift_union_router | 2 | +0.0234 | +0.0226 | +0.0259 | +0.0227 | -0.0091 |
| tabicl_baseline | 2 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_boundary_router` (apa_v21): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_union_router` (apa_v23): shift , acc +0.0312, bacc +0.0346, macro-F1 +0.0401, minority recall +0.0455, ECE -0.0323.
- `openml_1480_ilpd` seed 2 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0312, bacc +0.0346, macro-F1 +0.0401, minority recall +0.0455, ECE -0.0323.
- `openml_1480_ilpd` seed 3 `apa_cf_no_shift_boundary_router` (apa_v21): shift , acc +0.0156, bacc +0.0106, macro-F1 +0.0118, minority recall +0.0000, ECE +0.0144.
- `openml_1480_ilpd` seed 3 `apa_cf_no_shift_union_router` (apa_v23): shift , acc +0.0156, bacc +0.0106, macro-F1 +0.0118, minority recall +0.0000, ECE +0.0142.
- `openml_1480_ilpd` seed 3 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0156, bacc +0.0106, macro-F1 +0.0118, minority recall +0.0000, ECE +0.0144.
