# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_hard_shift_router,apa_dual_shift_router --output-prefix apa_v6_hard_label_shift_5seed
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_dual_shift_router | 20 | +0.0062 | +0.0033 | +0.0040 | +0.0000 | -0.0001 |
| apa_hard_shift_router | 20 | +0.0062 | +0.0033 | +0.0040 | +0.0000 | -0.0001 |
| apa_shift_margin_gate | 20 | +0.0025 | +0.0013 | +0.0014 | +0.0000 | -0.0025 |
| tabicl_baseline | 20 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 3 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 3 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 3 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `synthetic_label_shift_hard` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0022.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0022.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0159.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0159.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0202, minority recall +0.0000, ECE +0.0064.
- `synthetic_label_shift_hard` seed 2 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0202, minority recall +0.0000, ECE +0.0064.
- `synthetic_label_shift_hard` seed 3 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 3 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0051.
- `synthetic_label_shift_hard` seed 3 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0051.
- `synthetic_label_shift_hard` seed 4 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0028.
- `synthetic_label_shift_hard` seed 4 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0028.
- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0253.
- `synthetic_label_shift_hard` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0228.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0228.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0159, minority recall +0.0000, ECE +0.0259.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0159, minority recall +0.0000, ECE +0.0259.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `synthetic_label_shift_hard` seed 2 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `synthetic_label_shift_hard` seed 3 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 3 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0026.
- `synthetic_label_shift_hard` seed 3 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0026.
- `synthetic_label_shift_hard` seed 4 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0253.
- `synthetic_label_shift_hard` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0228.
- `synthetic_label_shift_hard` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0228.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0159, minority recall +0.0000, ECE +0.0259.
- `synthetic_label_shift_hard` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0159, minority recall +0.0000, ECE +0.0259.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_hard_shift_router` (apa_v5): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `synthetic_label_shift_hard` seed 2 `apa_dual_shift_router` (apa_v6): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `synthetic_label_shift_hard` seed 3 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 3 `apa_hard_shift_router` (apa_v5): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0026.
- `synthetic_label_shift_hard` seed 3 `apa_dual_shift_router` (apa_v6): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0026.
- `synthetic_label_shift_hard` seed 4 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_hard_shift_router` (apa_v5): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 4 `apa_dual_shift_router` (apa_v6): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
