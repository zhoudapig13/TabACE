# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_43_haberman,openml_1498_sa_heart,openml_1050_pc3 --split-mode ordered --seeds 1,2,3 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_guarded_boundary_union_router --output-prefix apa_v27_guarded_boundary_key4_r3
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_guarded_boundary_union_router | 12 | +0.0052 | +0.0027 | +0.0033 | -0.0042 | -0.0026 |
| tabicl_baseline | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0156, bacc +0.0116, macro-F1 +0.0137, minority recall +0.0000, ECE +0.0049.
- `openml_31_credit_g` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0312, bacc +0.0250, macro-F1 +0.0235, minority recall +0.0000, ECE +0.0141.
- `openml_43_haberman` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0156, bacc +0.0104, macro-F1 +0.0128, minority recall +0.0000, ECE -0.0159.
- `openml_43_haberman` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0156, bacc +0.0109, macro-F1 +0.0118, minority recall +0.0000, ECE -0.0164.
- `openml_43_haberman` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0014.
- `openml_1498_sa_heart` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc -0.0156, bacc -0.0250, macro-F1 -0.0224, minority recall -0.0500, ECE -0.0164.
- `openml_1498_sa_heart` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 1 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 2 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 3 `apa_cf_no_shift_guarded_boundary_union_router` (apa_v27): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
