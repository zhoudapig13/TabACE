# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_openml_forced_shift_round1
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_directional_router | 48 | +0.0286 | -0.0082 | +0.0149 | -0.0486 | -0.0189 |
| apa_dual_shift_router_v7 | 48 | +0.0020 | +0.0010 | +0.0012 | +0.0000 | -0.0008 |
| tabicl_baseline | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_ilpd` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0105.
- `openml_ilpd` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0091.
- `openml_ilpd` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_ilpd` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0105.
- `openml_ilpd` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0091.
- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0042.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc -0.1494, macro-F1 -0.0277, minority recall -0.3333, ECE -0.0071.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0030.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0218, minority recall +0.0000, ECE -0.0398.
- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0018.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc -0.0312, bacc -0.5000, macro-F1 -0.0594, minority recall -1.0000, ECE -0.0109.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0275.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_credit_g` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0018.
- `openml_credit_g` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc -0.0312, bacc -0.5000, macro-F1 -0.0594, minority recall -1.0000, ECE -0.0109.
- `openml_credit_g` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0275.
- `openml_credit_g` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0358, minority recall +0.0000, ECE -0.0129.
- `openml_blood_transfusion` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_blood_transfusion` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0013.
- `openml_blood_transfusion` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_blood_transfusion` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0203, minority recall +0.0000, ECE -0.0316.
- `openml_blood_transfusion` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0408, minority recall +0.0000, ECE -0.0963.
- `openml_blood_transfusion` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0011.
- `openml_blood_transfusion` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0025.
- `openml_blood_transfusion` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0003.
- `openml_blood_transfusion` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0278, minority recall +0.0000, ECE -0.1119.
- `openml_blood_transfusion` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0011.
- `openml_blood_transfusion` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0025.
- `openml_blood_transfusion` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0003.
- `openml_blood_transfusion` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0278, minority recall +0.0000, ECE -0.1119.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0009.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0938, bacc +0.0517, macro-F1 +0.0678, minority recall +0.0000, ECE -0.0208.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0011.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0225, minority recall +0.0000, ECE -0.0005.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0306.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0625, bacc +0.0323, macro-F1 +0.0355, minority recall +0.0000, ECE -0.0659.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE -0.0354.
- `openml_kc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0179, minority recall +0.0000, ECE -0.0306.
- `openml_kc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0625, bacc +0.0323, macro-F1 +0.0355, minority recall +0.0000, ECE -0.0659.
- `openml_kc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0002.
- `openml_kc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE -0.0354.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0228.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0487, minority recall +0.0000, ECE -0.0358.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0044.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0625, bacc +0.0345, macro-F1 +0.0453, minority recall +0.0000, ECE -0.0503.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0034.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0225.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_diabetes_pima` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0034.
- `openml_diabetes_pima` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.1250, bacc +0.0645, macro-F1 +0.0745, minority recall +0.0000, ECE -0.0359.
- `openml_diabetes_pima` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE -0.0225.
- `openml_diabetes_pima` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0938, bacc +0.0484, macro-F1 +0.0532, minority recall +0.0000, ECE +0.0396.
- `openml_pc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.0000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0000.
- `openml_pc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.3000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0025.
- `openml_pc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.3000, acc +0.0312, bacc +0.0172, macro-F1 +0.0203, minority recall +0.0000, ECE -0.0153.
- `openml_pc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0031.
- `openml_pc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0460.
- `openml_pc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0025.
- `openml_pc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.6000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE -0.0295.
- `openml_pc1` seed 0 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0031.
- `openml_pc1` seed 0 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0183, minority recall +0.0000, ECE -0.0460.
- `openml_pc1` seed 1 `apa_dual_shift_router_v7` (apa_v7): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ECE +0.0025.
- `openml_pc1` seed 1 `apa_cf_directional_router` (apa_v8): shift 0.9000, acc +0.0312, bacc +0.0161, macro-F1 +0.0176, minority recall +0.0000, ECE -0.0295.
