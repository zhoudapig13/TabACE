# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_openml_forced_shift_seed23
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_directional_router | 48 | +0.0260 | -0.0098 | +0.0128 | -0.0473 | +0.0001 |
| apa_dual_shift_router_v7 | 48 | +0.0033 | +0.0017 | +0.0019 | +0.0000 | +0.0015 |
| tabicl_baseline | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_ilpd` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0173.
- `openml_ilpd` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0188, minority recall +0.0000, ECE -0.0289.
- `openml_ilpd` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0048.
- `openml_ilpd` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0316, minority recall +0.0000, ECE -0.0001.
- `openml_ilpd` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0048.
- `openml_ilpd` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0316, minority recall +0.0000, ECE -0.0001.
- `openml_credit_g` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc -0.0312, bacc -0.0172, macro-F1 -0.0203, minority recall +0.0000, ECE +0.0308.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc -0.1494, macro-F1 -0.0423, minority recall -0.3333, ECE +0.0590.
- `openml_credit_g` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0136.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0008.
- `openml_credit_g` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0326.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0326.
- `openml_credit_g` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0142, minority recall +0.0000, ECE +0.0313.
- `openml_credit_g` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0003.
- `openml_blood_transfusion` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0453, minority recall +0.0000, ECE -0.0447.
- `openml_blood_transfusion` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0020.
- `openml_blood_transfusion` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0350, minority recall +0.0000, ECE -0.0544.
- `openml_blood_transfusion` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0010.
- `openml_blood_transfusion` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0284.
- `openml_blood_transfusion` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0002.
- `openml_blood_transfusion` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0173, minority recall +0.0000, ECE +0.0300.
- `openml_blood_transfusion` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0010.
- `openml_blood_transfusion` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0284.
- `openml_blood_transfusion` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0002.
- `openml_blood_transfusion` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0173, minority recall +0.0000, ECE +0.0300.
- `openml_kc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0014.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.1250, bacc +0.0690, macro-F1 +0.0703, minority recall +0.0000, ECE -0.0033.
- `openml_kc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0028.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0409, minority recall +0.0000, ECE -0.0677.
- `openml_kc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0028.
- `openml_kc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0409, minority recall +0.0000, ECE -0.0677.
- `openml_kc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0009.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc -0.0312, bacc -0.0312, macro-F1 -0.0367, minority recall +0.0625, ECE +0.0100.
- `openml_diabetes_pima` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0005.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0408, minority recall +0.0000, ECE -0.0221.
- `openml_diabetes_pima` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0189.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0460, minority recall +0.0000, ECE -0.0362.
- `openml_diabetes_pima` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0335.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc -0.0312, bacc -0.5000, macro-F1 -0.0659, minority recall -1.0000, ECE -0.0038.
- `openml_diabetes_pima` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0174.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0522, minority recall +0.0000, ECE +0.0170.
- `openml_diabetes_pima` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0335.
- `openml_diabetes_pima` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc -0.0312, bacc -0.5000, macro-F1 -0.0659, minority recall -1.0000, ECE -0.0038.
- `openml_diabetes_pima` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0174.
- `openml_diabetes_pima` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0522, minority recall +0.0000, ECE +0.0170.
- `openml_pc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0012.
- `openml_pc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0352.
- `openml_pc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0033.
- `openml_pc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0805, minority recall +0.0000, ECE +0.0688.
- `openml_pc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0036.
- `openml_pc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0045.
- `openml_pc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0240, minority recall +0.0000, ECE +0.0192.
- `openml_pc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0515, minority recall +0.0000, ECE +0.0718.
- `openml_pc1` seed 2 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0036.
- `openml_pc1` seed 2 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0174, minority recall +0.0000, ECE -0.0045.
- `openml_pc1` seed 3 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0240, minority recall +0.0000, ECE +0.0192.
- `openml_pc1` seed 3 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0515, minority recall +0.0000, ECE +0.0718.
