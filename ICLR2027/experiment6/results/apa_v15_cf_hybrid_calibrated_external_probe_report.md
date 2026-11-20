# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_blood_transfusion,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_tempered_router,apa_cf_hybrid_calibrated_router --output-prefix apa_v15_cf_hybrid_calibrated_external_probe
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_hybrid_calibrated_router | 48 | +0.0130 | +0.0069 | +0.0086 | +0.0000 | -0.0009 |
| apa_cf_hybrid_tempered_router | 48 | +0.0137 | +0.0072 | +0.0095 | +0.0000 | +0.0002 |
| tabicl_baseline | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_ilpd` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0188, minority recall +0.0000, ECE -0.0289.
- `openml_ilpd` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0188, minority recall +0.0000, ECE -0.0295.
- `openml_ilpd` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0316, minority recall +0.0000, ECE -0.0001.
- `openml_ilpd` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0316, minority recall +0.0000, ECE -0.0026.
- `openml_ilpd` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0316, minority recall +0.0000, ECE -0.0001.
- `openml_ilpd` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0316, minority recall +0.0000, ECE -0.0026.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0203, minority recall +0.0000, ECE -0.0579.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0203, minority recall +0.0000, ECE -0.0573.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0453, minority recall +0.0000, ECE -0.0447.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0453, minority recall +0.0000, ECE -0.0595.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0176, minority recall +0.0000, ECE -0.0290.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0176, minority recall +0.0000, ECE -0.0296.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0284.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0286.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0173, minority recall +0.0000, ECE +0.0300.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0173, minority recall +0.0000, ECE +0.0305.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0284.
- `openml_blood_transfusion` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0286.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0173, minority recall +0.0000, ECE +0.0300.
- `openml_blood_transfusion` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0173, minority recall +0.0000, ECE +0.0305.
- `openml_pc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0352.
- `openml_pc1` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0358.
- `openml_pc1` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0805, minority recall +0.0000, ECE +0.0688.
- `openml_pc1` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0374, minority recall +0.0000, ECE +0.0373.
- `openml_pc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0045.
- `openml_pc1` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0041.
- `openml_pc1` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0515, minority recall +0.0000, ECE +0.0718.
- `openml_pc1` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0515, minority recall +0.0000, ECE +0.0711.
- `openml_pc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0045.
- `openml_pc1` seed 2 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0041.
- `openml_pc1` seed 3 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0515, minority recall +0.0000, ECE +0.0718.
- `openml_pc1` seed 3 `apa_cf_hybrid_calibrated_router` (apa_v15): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0515, minority recall +0.0000, ECE +0.0711.
