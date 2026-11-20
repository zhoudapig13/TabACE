# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,synthetic_label_shift,wine,breast_cancer,synthetic_multiclass_imbalance --label-shift-strengths 0.0,0.3,0.6 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_hard_shift_router,apa_dual_shift_router --output-prefix apa_v6_mixed_safety
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_dual_shift_router | 14 | +0.0018 | +0.0011 | +0.0022 | +0.0000 | -0.0080 |
| apa_hard_shift_router | 14 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0054 |
| apa_shift_margin_gate | 14 | +0.0018 | +0.0011 | +0.0022 | +0.0000 | -0.0080 |
| tabicl_baseline | 14 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_imbalanced` seed 0 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 0 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 0 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 1 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 1 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 1 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0173.
- `synthetic_label_shift` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0113.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0173.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0313, minority recall +0.0000, ECE -0.0003.
- `synthetic_label_shift` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0176.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.3000, acc +0.0250, bacc +0.0152, macro-F1 +0.0313, minority recall +0.0000, ECE -0.0003.
- `synthetic_label_shift` seed 0 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0431.
- `synthetic_label_shift` seed 0 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0097.
- `synthetic_label_shift` seed 0 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0431.
- `synthetic_label_shift` seed 1 `apa_shift_margin_gate` (apa_v2): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0512.
- `synthetic_label_shift` seed 1 `apa_hard_shift_router` (apa_v5): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0559.
- `synthetic_label_shift` seed 1 `apa_dual_shift_router` (apa_v6): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0512.
- `wine` seed 0 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 0 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 0 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `wine` seed 1 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 0 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `breast_cancer` seed 0 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `breast_cancer` seed 0 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0000.
- `breast_cancer` seed 1 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 1 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `breast_cancer` seed 1 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_multiclass_imbalance` seed 0 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_multiclass_imbalance` seed 0 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_multiclass_imbalance` seed 0 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_multiclass_imbalance` seed 1 `apa_shift_margin_gate` (apa_v2): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `synthetic_multiclass_imbalance` seed 1 `apa_hard_shift_router` (apa_v5): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_multiclass_imbalance` seed 1 `apa_dual_shift_router` (apa_v6): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
