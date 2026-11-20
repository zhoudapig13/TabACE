# Fixed 11-Dataset Selected Benchmark

Settings: 11 datasets, label-shift strengths `0.0,0.6,0.9`, seeds `0,1`, 66 groups per method. Deltas are relative to `tabicl_baseline`.

## Overall Method Ranking

| method | groups | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc | max/min bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_directional_router` | 66 | +0.0161 | +0.0085 | +0.0128 | -0.0011 | 20 / 0 | +0.0645 / +0.0000 |
| `apa_cf_hybrid_tempered_router` | 66 | +0.0128 | +0.0066 | +0.0110 | +0.0012 | 15 / 0 | +0.0645 / +0.0000 |
| `apa_cf_batch_margin_router` | 66 | +0.0109 | +0.0056 | +0.0099 | +0.0013 | 11 / 0 | +0.0645 / +0.0000 |
| `apa_cf_hybrid_calibrated_router` | 66 | +0.0109 | +0.0056 | +0.0099 | -0.0001 | 15 / 0 | +0.0645 / +0.0000 |
| `apa_cf_hybrid_stability_router` | 66 | +0.0099 | +0.0051 | +0.0093 | +0.0006 | 15 / 0 | +0.0484 / +0.0000 |
| `apa_cf_tempered_rescue_router` | 66 | +0.0043 | +0.0022 | +0.0048 | -0.0008 | 9 / 0 | +0.0161 / +0.0000 |
| `apa_dual_shift_router_v7` | 66 | +0.0028 | +0.0015 | +0.0026 | -0.0002 | 6 / 0 | +0.0161 / +0.0000 |
| `apa_cf_strict_margin_router` | 66 | +0.0019 | +0.0010 | +0.0011 | -0.0019 | 4 / 0 | +0.0161 / +0.0000 |
| `apa_full_prior_aware_tiny` | 66 | +0.0009 | +0.0005 | +0.0003 | -0.0001 | 2 / 0 | +0.0161 / +0.0000 |
| `tabicl_baseline` | 66 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 | +0.0000 / +0.0000 |

## Dataset x Method Matrix

Each cell is `accuracy delta / balanced-accuracy delta`. Full macro-F1, ECE, positive/negative counts are in `apa_v1_v16_fixed11_selected_benchmark_dataset_method.csv`.

| dataset | v1 tiny prior | v7 dual router | v8 directional | v9 strict | v10 batch | v13 rescue | v14 hybrid | v15 calibrated | v16 stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `breast_cancer` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 |
| `openml_blood_transfusion` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 |
| `openml_credit_g` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0208/+0.0108 | +0.0000/+0.0000 | +0.0208/+0.0108 | +0.0104/+0.0054 | +0.0208/+0.0108 | +0.0208/+0.0108 | +0.0208/+0.0108 |
| `openml_diabetes_pima` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0729/+0.0376 | +0.0104/+0.0054 | +0.0625/+0.0323 | +0.0208/+0.0108 | +0.0625/+0.0323 | +0.0521/+0.0269 | +0.0417/+0.0215 |
| `openml_ilpd` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 |
| `openml_kc1` | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0208/+0.0108 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0208/+0.0108 | +0.0208/+0.0108 | +0.0208/+0.0108 |
| `openml_pc1` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 |
| `synthetic_balanced` | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0104/+0.0054 | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0000/+0.0000 | +0.0104/+0.0054 | +0.0104/+0.0054 | +0.0104/+0.0054 |
| `synthetic_imbalanced` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0052/+0.0027 | +0.0000/+0.0000 | +0.0052/+0.0027 | +0.0052/+0.0027 | +0.0052/+0.0027 | +0.0052/+0.0027 | +0.0052/+0.0027 |
| `synthetic_label_shift` | +0.0104/+0.0054 | +0.0104/+0.0054 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0000/+0.0000 |
| `synthetic_label_shift_hard` | +0.0000/+0.0000 | +0.0000/+0.0000 | +0.0260/+0.0160 | +0.0104/+0.0054 | +0.0208/+0.0108 | +0.0000/+0.0000 | +0.0208/+0.0108 | +0.0104/+0.0054 | +0.0104/+0.0054 |

## Takeaways

- `apa_cf_directional_router` has the largest average gain on this 66-group benchmark: accuracy `+0.0161`, balanced accuracy `+0.0085`, macro-F1 `+0.0128`, with `20 / 0` positive/negative bacc groups.
- Among the later safer candidates, `apa_cf_hybrid_tempered_router` is strongest: accuracy `+0.0128`, balanced accuracy `+0.0066`, macro-F1 `+0.0110`, with `15 / 0` positive/negative bacc groups.
- `apa_cf_hybrid_stability_router` is more conservative than v14: it keeps zero negative bacc groups, but drops average accuracy gain to `+0.0099`.
- Early probability-style correction is weak on this benchmark: `apa_full_prior_aware_tiny` only reaches accuracy `+0.0009`, balanced accuracy `+0.0005`.
- `openml_kc1`, `openml_credit_g`, `openml_diabetes_pima`, `synthetic_balanced`, `synthetic_imbalanced`, and `synthetic_label_shift_hard` show clear context-intervention gains. `openml_ilpd` and `breast_cancer` are mostly no-op safety cases.
