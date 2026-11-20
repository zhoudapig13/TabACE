# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard,synthetic_label_shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1 --n-samples 140 --train-size 70 --test-size 24 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_stability_router,apa_cf_directional_router --output-prefix apa_v8_cf_directional_round3
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_directional_router | 16 | +0.0469 | +0.0245 | +0.0321 | +0.0000 | -0.0123 |
| apa_cf_stability_router | 16 | +0.0286 | -0.0760 | +0.0010 | -0.2031 | +0.0128 |
| apa_dual_shift_router_v7 | 16 | +0.0156 | +0.0082 | +0.0104 | +0.0000 | -0.0089 |
| tabicl_baseline | 16 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.0000, acc -0.0833, bacc -0.0833, macro-F1 -0.0864, minority recall -0.1667, ECE +0.0221.
- `synthetic_label_shift_hard` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.0000, acc +0.0417, bacc +0.0417, macro-F1 +0.0409, minority recall +0.0000, ECE +0.0032.
- `synthetic_label_shift_hard` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE -0.0043.
- `synthetic_label_shift_hard` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.3000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE +0.0357.
- `synthetic_label_shift_hard` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE -0.0109.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0004.
- `synthetic_label_shift_hard` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.3000, acc +0.0000, bacc -0.4783, macro-F1 -0.0947, minority recall -1.0000, ECE +0.0584.
- `synthetic_label_shift_hard` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0131.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE -0.0043.
- `synthetic_label_shift_hard` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.6000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE +0.0357.
- `synthetic_label_shift_hard` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE -0.0109.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0004.
- `synthetic_label_shift_hard` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.6000, acc +0.0000, bacc -0.4783, macro-F1 -0.0947, minority recall -1.0000, ECE +0.0584.
- `synthetic_label_shift_hard` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0131.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE -0.0043.
- `synthetic_label_shift_hard` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.9000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE +0.0357.
- `synthetic_label_shift_hard` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0417, bacc +0.0217, macro-F1 +0.0286, minority recall +0.0000, ECE -0.0109.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0004.
- `synthetic_label_shift_hard` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.9000, acc +0.0000, bacc -0.4783, macro-F1 -0.0947, minority recall -1.0000, ECE +0.0584.
- `synthetic_label_shift_hard` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0131.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.0000, acc +0.0833, bacc +0.0833, macro-F1 +0.0833, minority recall +0.0833, ECE +0.0943.
- `synthetic_label_shift` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.0000, acc -0.0833, bacc -0.0833, macro-F1 -0.0845, minority recall -0.1667, ECE -0.0532.
- `synthetic_label_shift` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0015.
- `synthetic_label_shift` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.3000, acc +0.0417, bacc +0.0217, macro-F1 +0.0319, minority recall +0.0000, ECE +0.0051.
- `synthetic_label_shift` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0417, bacc +0.0217, macro-F1 +0.0319, minority recall +0.0000, ECE +0.0438.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0417, bacc +0.0217, macro-F1 +0.0266, minority recall +0.0000, ECE -0.0419.
- `synthetic_label_shift` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.3000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0533.
- `synthetic_label_shift` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0852.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0015.
- `synthetic_label_shift` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.6000, acc +0.0417, bacc +0.0217, macro-F1 +0.0319, minority recall +0.0000, ECE +0.0051.
- `synthetic_label_shift` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0417, bacc +0.0217, macro-F1 +0.0319, minority recall +0.0000, ECE +0.0438.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0417, bacc +0.0217, macro-F1 +0.0266, minority recall +0.0000, ECE -0.0419.
- `synthetic_label_shift` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.6000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0533.
- `synthetic_label_shift` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0852.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0015.
- `synthetic_label_shift` seed 0 `apa_cf_stability_router` (apa_v8): shift 0.9000, acc +0.0417, bacc +0.0217, macro-F1 +0.0319, minority recall +0.0000, ECE +0.0051.
- `synthetic_label_shift` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0417, bacc +0.0217, macro-F1 +0.0319, minority recall +0.0000, ECE +0.0438.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0417, bacc +0.0217, macro-F1 +0.0266, minority recall +0.0000, ECE -0.0419.
- `synthetic_label_shift` seed 1 `apa_cf_stability_router` (apa_v8): shift 0.9000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0533.
- `synthetic_label_shift` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0833, bacc +0.0435, macro-F1 +0.0553, minority recall +0.0000, ECE -0.0852.
