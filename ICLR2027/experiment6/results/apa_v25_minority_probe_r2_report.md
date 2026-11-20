# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_1050_pc3,openml_1490_planning_relax,openml_1464_blood_transfusion,openml_31_credit_g,openml_37_diabetes,openml_43_haberman --split-mode ordered --seeds 1 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_minority_router,apa_cf_no_shift_boundary_union_router --output-prefix apa_v25_minority_probe_r2
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_no_shift_boundary_union_router | 6 | +0.0078 | +0.0068 | +0.0076 | +0.0062 | +0.0009 |
| apa_cf_no_shift_minority_router | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| tabicl_baseline | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_1050_pc3` seed 1 `apa_cf_no_shift_minority_router` (apa_v25): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 1 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1490_planning_relax` seed 1 `apa_cf_no_shift_minority_router` (apa_v25): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1490_planning_relax` seed 1 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_no_shift_minority_router` (apa_v25): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_no_shift_minority_router` (apa_v25): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0156, bacc +0.0116, macro-F1 +0.0137, minority recall +0.0000, ECE +0.0049.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_minority_router` (apa_v25): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 1 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0156, bacc +0.0185, macro-F1 +0.0193, minority recall +0.0370, ECE +0.0162.
- `openml_43_haberman` seed 1 `apa_cf_no_shift_minority_router` (apa_v25): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 1 `apa_cf_no_shift_boundary_union_router` (apa_v24): shift , acc +0.0156, bacc +0.0104, macro-F1 +0.0128, minority recall +0.0000, ECE -0.0159.
