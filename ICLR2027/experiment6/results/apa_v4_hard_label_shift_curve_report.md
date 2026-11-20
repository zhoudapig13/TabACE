# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_aggressive_margin_push,apa_aggressive_batch_push,apa_validation_selector --output-prefix apa_v4_hard_label_shift_curve
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_aggressive_batch_push | 12 | +0.0083 | +0.0047 | +0.0058 | +0.0000 | +0.0003 |
| apa_aggressive_margin_push | 12 | +0.0104 | +0.0055 | +0.0067 | +0.0000 | +0.0035 |
| apa_shift_margin_gate | 12 | +0.0042 | +0.0021 | +0.0023 | +0.0000 | -0.0042 |
| apa_validation_selector | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| tabicl_baseline | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_margin_push` (apa_v4): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_batch_push` (apa_v4): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0003.
- `synthetic_label_shift_hard` seed 0 `apa_validation_selector` (apa_v3): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_margin_push` (apa_v4): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_batch_push` (apa_v4): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_validation_selector` (apa_v3): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_margin_push` (apa_v4): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0398.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_batch_push` (apa_v4): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0356.
- `synthetic_label_shift_hard` seed 2 `apa_validation_selector` (apa_v3): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_margin_push` (apa_v4): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0022.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_batch_push` (apa_v4): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0008.
- `synthetic_label_shift_hard` seed 0 `apa_validation_selector` (apa_v3): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_margin_push` (apa_v4): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0159.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_batch_push` (apa_v4): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0210, minority recall +0.0000, ECE +0.0004.
- `synthetic_label_shift_hard` seed 1 `apa_validation_selector` (apa_v3): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_margin_push` (apa_v4): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0202, minority recall +0.0000, ECE +0.0064.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_batch_push` (apa_v4): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0202, minority recall +0.0000, ECE +0.0246.
- `synthetic_label_shift_hard` seed 2 `apa_validation_selector` (apa_v3): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0253.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_margin_push` (apa_v4): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0228.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_batch_push` (apa_v4): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0263.
- `synthetic_label_shift_hard` seed 0 `apa_validation_selector` (apa_v3): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_margin_push` (apa_v4): shift 0.6000, acc +0.0250, bacc +0.0128, macro-F1 +0.0159, minority recall +0.0000, ECE +0.0259.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_batch_push` (apa_v4): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0006.
- `synthetic_label_shift_hard` seed 1 `apa_validation_selector` (apa_v3): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_margin_push` (apa_v4): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_batch_push` (apa_v4): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0012.
- `synthetic_label_shift_hard` seed 2 `apa_validation_selector` (apa_v3): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0253.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_margin_push` (apa_v4): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0228.
- `synthetic_label_shift_hard` seed 0 `apa_aggressive_batch_push` (apa_v4): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0140, minority recall +0.0000, ECE -0.0263.
- `synthetic_label_shift_hard` seed 0 `apa_validation_selector` (apa_v3): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_margin_push` (apa_v4): shift 0.9000, acc +0.0250, bacc +0.0128, macro-F1 +0.0159, minority recall +0.0000, ECE +0.0259.
- `synthetic_label_shift_hard` seed 1 `apa_aggressive_batch_push` (apa_v4): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0006.
- `synthetic_label_shift_hard` seed 1 `apa_validation_selector` (apa_v3): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_margin_push` (apa_v4): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `synthetic_label_shift_hard` seed 2 `apa_aggressive_batch_push` (apa_v4): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0012.
- `synthetic_label_shift_hard` seed 2 `apa_validation_selector` (apa_v3): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
