# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_shift_margin_gate,apa_shift_margin_gate_soft --output-prefix apa_v2_label_shift_5seed
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_full_prior_aware_tiny | 5 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0113 |
| apa_shift_margin_gate | 5 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0163 |
| apa_shift_margin_gate_soft | 5 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0097 |
| tabicl_baseline | 5 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_label_shift` seed 0 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0220.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0431.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0318.
- `synthetic_label_shift` seed 1 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0321.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0512.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0316.
- `synthetic_label_shift` seed 2 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0007.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0164.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0169.
- `synthetic_label_shift` seed 3 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift` seed 3 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 3 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 4 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0017.
- `synthetic_label_shift` seed 4 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0036.
- `synthetic_label_shift` seed 4 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0023.
