# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,synthetic_label_shift,wine,breast_cancer --seeds 0,1 --n-samples 160 --train-size 80 --test-size 30 --k 24 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_shift_margin_gate,apa_shift_margin_gate_soft --output-prefix apa_v2_small_tabicl
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_full_prior_aware_tiny | 8 | +0.0057 | +0.0057 | +0.0019 | +0.0000 | -0.0012 |
| apa_shift_margin_gate | 8 | +0.0057 | +0.0057 | +0.0019 | +0.0000 | -0.0004 |
| apa_shift_margin_gate_soft | 8 | +0.0057 | +0.0057 | +0.0019 | +0.0000 | -0.0005 |
| tabicl_baseline | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_imbalanced` seed 0 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `synthetic_imbalanced` seed 0 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 0 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `synthetic_imbalanced` seed 1 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0069.
- `synthetic_imbalanced` seed 1 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 1 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0013.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0010.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0031.
- `synthetic_label_shift` seed 1 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0455, bacc +0.0455, macro-F1 +0.0148, minority recall +0.0000, ECE -0.0017.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): acc +0.0455, bacc +0.0455, macro-F1 +0.0148, minority recall +0.0000, ECE -0.0023.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0455, bacc +0.0455, macro-F1 +0.0148, minority recall +0.0000, ECE -0.0017.
- `wine` seed 0 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `wine` seed 0 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 0 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 0 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `breast_cancer` seed 0 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 0 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 1 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0003.
- `breast_cancer` seed 1 `apa_shift_margin_gate` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 1 `apa_shift_margin_gate_soft` (apa_v2): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0003.
