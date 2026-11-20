# APA-TabICL Experiment 6 Report

Algorithm versions: `apa_v0` reruns TabICL with local contexts; `apa_v1` keeps full-context TabICL output and uses local contexts only for posterior correction; `apa_v2` adds batch-shift and margin/confidence gates; `apa_v3` uses train-internal validation selection; `apa_v4` tests more aggressive margin-gated correction for hard shift cases; `apa_v5` raises the hard-shift router threshold using observed no-shift diagnostics.

This run compares original TabICL, KNN context, balanced KNN context, prior-aware KNN context, local correction, and prior-aware adaptive correction. All artifacts are stored in `experiment6`.

## Command

```text
experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman,openml_1498_sa_heart,openml_1464_blood_transfusion,openml_1050_pc3,openml_kc1 --force-label-shift --label-shift-strengths 0.6,0.9 --seeds 0,1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_tempered_router,apa_cf_dual_mode_v36 --output-prefix apa_v36_dual_mode_forced_shift_8dataset_3seed
```

## Method Summary

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ROC-AUC delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apa_cf_dual_mode_v36 | 48 | +0.0205 | +0.0104 | +0.0111 | +0.0000 | +0.0007 | -0.0046 |
| apa_cf_hybrid_tempered_router | 48 | +0.0306 | +0.0155 | +0.0177 | +0.0000 | +0.0000 | -0.0084 |
| tabicl_baseline | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Dataset Detail

- `openml_31_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0430, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0582.
- `openml_31_credit_g` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc -0.0156, bacc -0.0079, macro-F1 -0.0106, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0167.
- `openml_31_credit_g` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_31_credit_g` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_31_credit_g` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0430, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0582.
- `openml_31_credit_g` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_31_credit_g` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc -0.0156, bacc -0.0079, macro-F1 -0.0106, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0167.
- `openml_31_credit_g` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_31_credit_g` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0079, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0035.
- `openml_37_diabetes` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_37_diabetes` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_37_diabetes` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0164, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0162.
- `openml_37_diabetes` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0094, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0163.
- `openml_37_diabetes` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_37_diabetes` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0302, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0309.
- `openml_1480_ilpd` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0149, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0166.
- `openml_1480_ilpd` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0938, bacc +0.0476, macro-F1 +0.0591, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0878.
- `openml_1480_ilpd` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0938, bacc +0.0476, macro-F1 +0.0591, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0878.
- `openml_1480_ilpd` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1480_ilpd` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0149, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0166.
- `openml_1480_ilpd` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0938, bacc +0.0476, macro-F1 +0.0591, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0878.
- `openml_1480_ilpd` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0938, bacc +0.0476, macro-F1 +0.0591, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0878.
- `openml_43_haberman` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc -0.0156, bacc -0.0091, macro-F1 -0.0175, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0164.
- `openml_43_haberman` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc -0.0156, bacc -0.0091, macro-F1 -0.0179, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0113.
- `openml_43_haberman` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc -0.0156, bacc -0.0079, macro-F1 -0.0145, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0162.
- `openml_43_haberman` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_43_haberman` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0469, bacc +0.0238, macro-F1 +0.0275, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0276.
- `openml_1498_sa_heart` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0469, bacc +0.0238, macro-F1 +0.0275, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0276.
- `openml_1498_sa_heart` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc -0.0156, bacc -0.0079, macro-F1 -0.0095, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0161.
- `openml_1498_sa_heart` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0411, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0570.
- `openml_1498_sa_heart` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0469, bacc +0.0238, macro-F1 +0.0275, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0276.
- `openml_1498_sa_heart` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0469, bacc +0.0238, macro-F1 +0.0275, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0276.
- `openml_1498_sa_heart` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1498_sa_heart` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc -0.0156, bacc -0.0079, macro-F1 -0.0095, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0161.
- `openml_1498_sa_heart` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0411, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0570.
- `openml_1498_sa_heart` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0781, bacc +0.0397, macro-F1 +0.0509, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0065.
- `openml_1464_blood_transfusion` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0210, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0151.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0156, bacc +0.0079, macro-F1 +0.0126, minority recall +0.0000, ROC-AUC +0.0159, ECE -0.0159.
- `openml_1464_blood_transfusion` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0469, bacc +0.0238, macro-F1 +0.0270, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0192.
- `openml_1464_blood_transfusion` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0781, bacc +0.0397, macro-F1 +0.0509, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0065.
- `openml_1464_blood_transfusion` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0210, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0151.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1464_blood_transfusion` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0156, bacc +0.0079, macro-F1 +0.0126, minority recall +0.0000, ROC-AUC +0.0159, ECE -0.0159.
- `openml_1464_blood_transfusion` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0469, bacc +0.0238, macro-F1 +0.0270, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0192.
- `openml_1464_blood_transfusion` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0313, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0001.
- `openml_1050_pc3` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0625, bacc +0.0317, macro-F1 +0.0313, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0001.
- `openml_1050_pc3` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0469, bacc +0.0238, macro-F1 +0.0226, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0056.
- `openml_1050_pc3` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0469, bacc +0.0238, macro-F1 +0.0226, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0056.
- `openml_1050_pc3` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0313, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0001.
- `openml_1050_pc3` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0625, bacc +0.0317, macro-F1 +0.0313, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0001.
- `openml_1050_pc3` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_1050_pc3` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0469, bacc +0.0238, macro-F1 +0.0226, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0056.
- `openml_1050_pc3` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0469, bacc +0.0238, macro-F1 +0.0226, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0056.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0172, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0219.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0312, bacc +0.0159, macro-F1 +0.0172, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0219.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0781, bacc +0.0397, macro-F1 +0.0403, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0554.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0781, bacc +0.0397, macro-F1 +0.0403, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0554.
- `openml_kc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.6000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 0 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0172, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0219.
- `openml_kc1` seed 0 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0312, bacc +0.0159, macro-F1 +0.0172, minority recall +0.0000, ROC-AUC +0.0000, ECE -0.0219.
- `openml_kc1` seed 1 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0781, bacc +0.0397, macro-F1 +0.0403, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0554.
- `openml_kc1` seed 1 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0781, bacc +0.0397, macro-F1 +0.0403, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0554.
- `openml_kc1` seed 2 `apa_cf_hybrid_tempered_router` (apa_v14): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
- `openml_kc1` seed 2 `apa_cf_dual_mode_v36` (apa_v36): shift 0.9000, acc +0.0000, bacc +0.0000, macro-F1 +0.0000, minority recall +0.0000, ROC-AUC +0.0000, ECE +0.0000.
