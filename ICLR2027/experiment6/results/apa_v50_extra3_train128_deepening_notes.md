# V50 train128 extra-dataset deepening

## Goal

After broad method screening, V50 cross-model tree rescue is the strongest new
direction. This run extends V50 beyond the original 4-dataset smoke by adding
the remaining datasets from the current V42 large no-shift benchmark:

- `openml_1498_sa_heart`
- `openml_1464_blood_transfusion`
- `openml_1050_pc3`

This gives a 7-dataset x 2-seed train128/test64 view when combined with the
previous V50 smoke4 result.

## Command

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1498_sa_heart,openml_1464_blood_transfusion,openml_1050_pc3 --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_extra3_2seed_train128
```

## Extra3 result

On the three added datasets, both V42 and V50 were all-zero relative to the
baseline. This is useful: V50 did not introduce extra negative splits on these
additional datasets, but it also did not find new tree-rescue positives.

## Combined 7-dataset train128/test64 result

Combined sources:

- first 4 datasets: `apa_v50_cross_model_tree_no_shift_smoke4_2seed_train128_r2_detail.csv`
- V42 comparison for first 4: `apa_v46_diversity_greedy_no_shift_smoke4_2seed_train128_detail.csv`
- extra 3 datasets: `apa_v50_cross_model_tree_no_shift_extra3_2seed_train128_detail.csv`

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass | 14 | +0.0089 | +0.0086 | +0.0095 | +0.0018 | -0.0054 | 6/0/8 |
| V50 cross-model tree rescue | 14 | +0.0089 | +0.0093 | +0.0105 | +0.0017 | -0.0054 | 6/0/8 |
| V50 - V42 | 14 | +0.0000 | +0.0008 | +0.0010 | -0.0001 | -0.0001 | no extra negatives |

The V50 gain over V42 comes from one tree-rescue split:

- `openml_1480_ilpd seed=2`

## Decision

V50 remains the best new method to deepen:

- It preserves V42's safety on this broader 7-dataset train128 view.
- It adds a small BAcc/Macro-F1 improvement without adding hard-metric negative
  splits.
- The effect is sparse, so the next useful deepening step is a larger-scale
  validation, not more threshold tweaking.

V50 is still not promoted to current-best because full train256 7-dataset
validation is incomplete and expensive.

## Artifacts

- `apa_v50_cross_model_tree_no_shift_extra3_2seed_train128_detail.csv`
- `apa_v50_cross_model_tree_no_shift_extra3_2seed_train128_summary.csv`
