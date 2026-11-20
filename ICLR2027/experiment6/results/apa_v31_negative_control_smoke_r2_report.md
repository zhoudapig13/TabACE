# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_1050_pc3,openml_1498_sa_heart --split-mode ordered --seeds 2,3 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_negative_control_stability_router --output-prefix apa_v31_negative_control_smoke_r2
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_negative_control_stability_router | 6 | +0.0000 | -0.0014 | -0.0029 | -0.0069 | -0.0012 |
| tabicl_baseline | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 2 `apa_cf_no_shift_negative_control_stability_router` (apa_v31): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 3 `apa_cf_no_shift_negative_control_stability_router` (apa_v31): shift , acc +0.0000, bacc -0.0083, macro-F1 -0.0173, minority recall -0.0417, ECE -0.0074.
- `openml_1050_pc3` seed 2 `apa_cf_no_shift_negative_control_stability_router` (apa_v31): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 3 `apa_cf_no_shift_negative_control_stability_router` (apa_v31): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_negative_control_stability_router` (apa_v31): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 3 `apa_cf_no_shift_negative_control_stability_router` (apa_v31): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
