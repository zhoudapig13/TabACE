# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_37_diabetes --force-label-shift --label-shift-strengths 0,0.6,0.9 --seeds 0 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_diverse_router,apa_cf_dual_mode_router_v17,apa_cf_directional_router --output-prefix apa_v21_dual_shift_safety_smoke
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_directional_router | 3 | +0.0208 | +0.0106 | +0.0110 | +0.0000 | -0.0108 |
| apa_cf_dual_mode_router_v17 | 3 | +0.0260 | +0.0158 | +0.0160 | +0.0104 | -0.0167 |
| apa_cf_no_shift_diverse_router | 3 | +0.0052 | +0.0052 | +0.0051 | +0.0104 | -0.0059 |
| tabicl_baseline | 3 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_37_diabetes` seed 0 `apa_cf_no_shift_diverse_router` (apa_v21): shift 0.0000, acc +0.0156, bacc +0.0156, macro-F1 +0.0152, minority recall +0.0312, ECE -0.0177.
- `openml_37_diabetes` seed 0 `apa_cf_dual_mode_router_v17` (apa_v21): shift 0.0000, acc +0.0156, bacc +0.0156, macro-F1 +0.0152, minority recall +0.0312, ECE -0.0177.
- `openml_37_diabetes` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 0 `apa_cf_no_shift_diverse_router` (apa_v21): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 0 `apa_cf_dual_mode_router_v17` (apa_v21): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 0 `apa_cf_no_shift_diverse_router` (apa_v21): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 0 `apa_cf_dual_mode_router_v17` (apa_v21): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ECE -0.0162.
