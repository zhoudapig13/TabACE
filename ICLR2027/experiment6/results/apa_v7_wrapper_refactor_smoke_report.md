# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --label-shift-strengths 0.3 --seeds 1 --n-samples 120 --train-size 60 --test-size 25 --k 20 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7 --output-prefix apa_v7_wrapper_refactor_smoke
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_dual_shift_router_v7 | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0507 |
| tabicl_baseline | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_label_shift` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0507.
