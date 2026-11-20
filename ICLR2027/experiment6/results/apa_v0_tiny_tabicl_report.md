# APA-TabICL Experiment 6 Report

Algorithm version: `apa_v0`.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced --seeds 0 --n-samples 80 --train-size 40 --test-size 12 --k 12 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_knn_context,apa_prior_aware_adaptive --output-prefix apa_v0_tiny_tabicl
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_knn_context | 1 | -0.2500 | -0.1667 | -0.2487 | +0.0000 | +0.1438 |
| apa_prior_aware_adaptive | 1 | -0.0833 | -0.0556 | -0.0959 | +0.0000 | +0.0748 |
| tabicl_baseline | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_imbalanced` seed 0 `apa_knn_context`: acc -0.2500, bacc -0.1667, macro-F1 -0.2487, minority recall +0.0000, ECE +0.1438.
- `synthetic_imbalanced` seed 0 `apa_prior_aware_adaptive`: acc -0.0833, bacc -0.0556, macro-F1 -0.0959, minority recall +0.0000, ECE +0.0748.
