# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced --seeds 0 --n-samples 80 --train-size 40 --test-size 12 --k 12 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_prior_aware_adaptive,apa_full_prior_knn_adaptive,apa_full_prior_aware_adaptive,apa_full_prior_aware_tiny --output-prefix apa_v1_tiny_tabicl
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_full_prior_aware_adaptive | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0001 |
| apa_full_prior_aware_tiny | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0001 |
| apa_full_prior_knn_adaptive | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| apa_prior_aware_adaptive | 1 | -0.0833 | -0.0556 | -0.0959 | +0.0000 | +0.0748 |
| tabicl_baseline | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `synthetic_imbalanced` seed 0 `apa_prior_aware_adaptive` (apa_v0): acc -0.0833, bacc -0.0556, macro-F1 -0.0959, minority recall +0.0000, ECE +0.0748.
- `synthetic_imbalanced` seed 0 `apa_full_prior_knn_adaptive` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `synthetic_imbalanced` seed 0 `apa_full_prior_aware_adaptive` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0001.
- `synthetic_imbalanced` seed 0 `apa_full_prior_aware_tiny` (apa_v1): acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0001.
