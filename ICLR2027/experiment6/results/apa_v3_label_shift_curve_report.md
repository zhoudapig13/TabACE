# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_shift_margin_gate,apa_shift_margin_gate_soft,apa_validation_selector --output-prefix apa_v3_label_shift_curve
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_full_prior_aware_tiny | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0101 |
| apa_shift_margin_gate | 12 | +0.0021 | +0.0013 | +0.0026 | +0.0000 | -0.0145 |
| apa_shift_margin_gate_soft | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0078 |
| apa_validation_selector | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| tabicl_baseline | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_label_shift` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate_soft` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_validation_selector` (apa_v3): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate_soft` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_validation_selector` (apa_v3): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 2 `apa_full_prior_aware_tiny` (apa_v1): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate_soft` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 2 `apa_validation_selector` (apa_v3): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0004.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0173.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate_soft` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0004.
- `synthetic_label_shift` seed 0 `apa_validation_selector` (apa_v3): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0007.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0313, minority recall +0.0000, ECE -0.0003.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate_soft` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0005.
- `synthetic_label_shift` seed 1 `apa_validation_selector` (apa_v3): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 2 `apa_full_prior_aware_tiny` (apa_v1): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0101.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0004.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate_soft` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0003.
- `synthetic_label_shift` seed 2 `apa_validation_selector` (apa_v3): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0220.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0431.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate_soft` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0318.
- `synthetic_label_shift` seed 0 `apa_validation_selector` (apa_v3): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0321.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0512.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate_soft` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0316.
- `synthetic_label_shift` seed 1 `apa_validation_selector` (apa_v3): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 2 `apa_full_prior_aware_tiny` (apa_v1): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0007.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0164.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate_soft` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0169.
- `synthetic_label_shift` seed 2 `apa_validation_selector` (apa_v3): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_full_prior_aware_tiny` (apa_v1): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0220.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0431.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate_soft` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0318.
- `synthetic_label_shift` seed 0 `apa_validation_selector` (apa_v3): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_full_prior_aware_tiny` (apa_v1): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0321.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0512.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate_soft` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0316.
- `synthetic_label_shift` seed 1 `apa_validation_selector` (apa_v3): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 2 `apa_full_prior_aware_tiny` (apa_v1): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0007.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0164.
- `synthetic_label_shift` seed 2 `apa_shift_margin_gate_soft` (apa_v2): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0169.
- `synthetic_label_shift` seed 2 `apa_validation_selector` (apa_v3): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
