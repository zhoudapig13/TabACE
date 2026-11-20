# V50/V56 Train256 Deepening Notes

Date: 2026-07-06

## Motivation

V50/V56 were positive on train128 no-shift tests, but the final target is
stable no-shift improvement at larger data scales. This note deepens the
evaluation on train256/test128.

V50 is the simpler prediction version:

- V42 boundary-bypass primary.
- ExtraTrees/RandomForest consensus as sparse heterogeneous rescue.
- Local support, binomial p-value, tree confidence/margin, prior drift, and
  minority-risk guards.

V56 is the explainable diagnostic version:

- Same V50 rescue structure.
- Lazy PRP family diagnostics only near tree-rescue candidates.

## Commands

ILPD train256, two seeds:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1480_ilpd --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_cross_model_tree_rescue_router,apa_cf_no_shift_prp_tree_rescue_router --output-prefix apa_v56_v50_ilpd_2seed_train256
```

Credit-g train256 V56 check:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_prp_tree_rescue_router --output-prefix apa_v56_prp_tree_rescue_credit_2seed_train256
```

Existing matched V50 credit/diabetes train256 file:

```text
experiment6/results/apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2_detail.csv
```

## ILPD Result

On `openml_1480_ilpd`, train256/test128, seeds 1 and 2:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V42 boundary-bypass | 2 | +0.0039 | +0.0029 | +0.0031 | +0.0004 | -0.0040 | 1 / 0 / 1 |
| V50 cross-model tree rescue | 2 | +0.0039 | +0.0029 | +0.0031 | +0.0004 | -0.0040 | 1 / 0 / 1 |
| V56 PRP-aware tree rescue | 2 | +0.0039 | +0.0029 | +0.0031 | +0.0004 | -0.0040 | 1 / 0 / 1 |

Interpretation:

- The train128 tree rescue on ILPD does not reproduce at train256.
- V50/V56 safely fall back to V42 behavior.
- No new negative split appears.

## Credit-G V56 Check

On `openml_31_credit_g`, train256/test128, seeds 1 and 2:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V56 PRP-aware tree rescue | 2 | +0.0156 | +0.0188 | +0.0218 | +0.0023 | +0.0061 | 2 / 0 / 0 |

PRP diagnostics:

- seed 1: tree-selected fraction 0.0078125, PRP evaluated, rejected V50 fraction 0.0.
- seed 2: tree-selected fraction 0.0078125, PRP evaluated, rejected V50 fraction 0.0.

Interpretation:

- V56 preserves the V50 train256 credit-g rescues.
- PRP is a useful diagnostic layer here, not a performance-improving selector.

## Matched Train256 Views

V50 on available train256 three-dataset view:

Datasets:

- `openml_31_credit_g`
- `openml_37_diabetes`
- `openml_1480_ilpd`

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V42 matched | 6 | +0.0052 | +0.0036 | +0.0047 | +0.0005 | -0.0007 | 3 / 0 / 3 |
| V50 matched | 6 | +0.0078 | +0.0081 | +0.0097 | +0.0009 | +0.0020 | 4 / 0 / 2 |
| V50 - V42 | 6 | +0.0026 | +0.0045 | +0.0050 | +0.0003 | +0.0027 | no extra negatives |

V56 on currently available train256 two-dataset view:

Datasets:

- `openml_31_credit_g`
- `openml_1480_ilpd`

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V42 matched | 4 | +0.0059 | +0.0042 | +0.0050 | +0.0008 | -0.0030 | 2 / 0 / 2 |
| V56 matched | 4 | +0.0098 | +0.0108 | +0.0125 | +0.0013 | +0.0011 | 3 / 0 / 1 |
| V56 - V42 | 4 | +0.0039 | +0.0067 | +0.0075 | +0.0005 | +0.0041 | no extra negatives |

## Full Seven-Dataset V50 Train256 View

After the initial train256 deepening, the remaining four V50 gaps were filled:

- `openml_1498_sa_heart`
- `openml_43_haberman`
- `openml_1464_blood_transfusion`
- `openml_1050_pc3`

Commands:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1498_sa_heart,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_gap2_2seed_train256
```

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1464_blood_transfusion,openml_1050_pc3 --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_gap2b_2seed_train256
```

Full seven-dataset train256/test128 result:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V42 boundary-bypass | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0013 | +0.0030 | 5 / 0 / 9 |
| V50 cross-model tree rescue | 14 | +0.0067 | +0.0079 | +0.0085 | +0.0016 | +0.0042 | 6 / 0 / 8 |
| V50 - V42 | 14 | +0.0011 | +0.0029 | +0.0027 | +0.0003 | +0.0012 | no extra negatives |

Per-dataset V50 - V42 deltas:

| Dataset | Acc | BAcc | Macro-F1 | ROC-AUC | ECE | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `openml_31_credit_g` | +0.0078 | +0.0134 | +0.0149 | +0.0010 | +0.0082 | main tree-rescue gain |
| `openml_1498_sa_heart` | +0.0000 | +0.0070 | +0.0041 | +0.0009 | +0.0005 | improves BAcc/F1 without changing Acc |
| `openml_1480_ilpd` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | falls back to V42 |
| `openml_37_diabetes` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | falls back to V42 |
| `openml_43_haberman` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | no extra risk |
| `openml_1464_blood_transfusion` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | no extra risk |
| `openml_1050_pc3` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | no extra risk |

## Conclusion

The train256 evidence strengthens the prediction-line conclusion:

- V50 has positive hard-metric lift over V42 on the full seven-dataset
  train256 view.
- Gains are driven by credit-g; ILPD train128 rescue does not reproduce at
  train256. The full view also shows a BAcc/F1 improvement on sa_heart.
- No hard-metric negative split is introduced in the full train256 view.
- ECE is mixed/worse on train256, so V50/V56 should be reported primarily as
  accuracy/BAcc/Macro-F1 improvements, not calibration improvements.
- V56 is useful as an explainable diagnostic variant that preserves V50
  rescues, but it has not produced additional prediction gains.

Current algorithm line:

**Sparse heterogeneous rescue for no-shift tabular foundation models**:
V42 primary, rare ET/RF consensus rescue, local p-value and prior-risk guards,
optional PRP diagnostics.
