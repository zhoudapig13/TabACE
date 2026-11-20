# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_1464_blood_transfusion,openml_31_credit_g,openml_881_mv,openml_37_diabetes,openml_1600_spectf,openml_151_electricity,openml_179_adult,openml_851_tecator --split-mode ordered --seeds 0,1,2,3,4 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_local_shift_val_router --output-prefix apa_v19_natural_ordered_candidates
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_local_shift_val_router | 40 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| tabicl_baseline | 40 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_1464_blood_transfusion` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_881_mv` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_37_diabetes` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1600_spectf` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1600_spectf` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1600_spectf` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1600_spectf` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_1600_spectf` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_151_electricity` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_179_adult` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_179_adult` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_179_adult` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_179_adult` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_179_adult` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_851_tecator` seed 0 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_851_tecator` seed 1 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_851_tecator` seed 2 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_851_tecator` seed 3 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_851_tecator` seed 4 `apa_cf_local_shift_val_router` (apa_v19): shift , acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
