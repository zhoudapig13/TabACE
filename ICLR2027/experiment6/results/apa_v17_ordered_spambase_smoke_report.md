# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_44_spambase --split-mode ordered --seeds 0 --n-samples 0 --train-size 128 --test-size 64 --k 8 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_natural_guard_router --output-prefix apa_v17_ordered_spambase_smoke
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_natural_guard_router | 1 | +0.0156 | +0.0156 | +0.0042 | +0.0000 | +0.0139 |
| tabicl_baseline | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_44_spambase` seed 0 `apa_cf_natural_guard_router` (apa_v17): shift , acc +0.0156, bacc +0.0156, macro-F1 +0.0042, minority recall +0.0000, ECE +0.0139.
