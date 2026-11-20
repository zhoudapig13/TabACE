# V50 cross-model tree rescue ensemble

## Goal

This round tests the reference idea of model-member diversity: keep the current
best no-shift TabICL line as the primary member, then add a heterogeneous tree
member only as a guarded rescue.

The primary member is V42 `apa_cf_no_shift_boundary_bypass_combo_router`.

## Method

V50 r2 uses a precision-first cross-model selector:

1. Run V42 and use it as the primary prediction.
2. If V42's batch prior JS is above `0.0015`, return V42 directly and do not
   train the tree member. This avoids stacking tree rescue on less-clean
   no-shift batches.
3. Train two lightweight tree members on the same training set:
   `ExtraTreesClassifier` and `RandomForestClassifier`, both with balanced
   class weights and 128 trees.
4. Align tree probabilities to TabICL class order and average the two tree
   distributions.
5. Allow a tree rescue only when all of these sample-level gates pass:
   V42 did not already change the sample, ExtraTrees and RandomForest agree,
   tree batch JS is not large, primary margin is low, tree margin/confidence are
   acceptable, local KNN label support favors the tree class, local binomial
   p-value is small, local purity is acceptable, candidate prior drift is small,
   and minority-to-majority flips pass a stricter support gate.
6. Cap tree rescues at `3.125%` of the batch.

This keeps the unified line:

`V42 primary no-shift member -> optional cross-model rescue -> same local/p-value/prior risk guards`.

## r1 failure and r2 fix

The first version used local support and p-value guards but did not require an
extremely low primary batch prior JS. It found a real positive case on
`openml_1480_ilpd seed=2`, but introduced a negative case on
`openml_1480_ilpd seed=1`.

Probe results:

| Split | Acc delta | BAcc delta | Macro-F1 delta | Note |
|---|---:|---:|---:|---|
| `openml_1480_ilpd seed=1` | -0.0156 | -0.0109 | -0.0129 | bad tree rescue |
| `openml_1480_ilpd seed=2` | +0.0312 | +0.0455 | +0.0541 | good tree rescue |

The bad rescue had strong local support, so local p-value alone was not enough.
The useful separator was primary batch prior JS:

| Split | primary batch JS |
|---|---:|
| bad `ilpd seed=1` | 0.007279 |
| good `ilpd seed=2` | 0.000053 |

V50 r2 therefore adds `primary_js <= 0.0015` before training/applying tree
rescue.

## No-shift train128/test64 smoke

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_smoke4_2seed_train128_r2
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass, same suite | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |
| V50 cross-model tree rescue | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6/0/2 |
| V50 - V42 | 8 | +0.0000 | +0.0014 | +0.0018 | -0.0001 | -0.0001 | no extra negatives |

Only `openml_1480_ilpd seed=2` selected tree rescues in this suite. The tree
member improved BAcc and Macro-F1 there without changing the average accuracy
relative to V42.

## No-shift train256/test128 scale check

Full 4-dataset train256 was too slow for this pass, so this is a focused scale
check on `openml_31_credit_g` and `openml_37_diabetes`, seeds `1,2`.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 subset, same splits | 4 | +0.0059 | +0.0040 | +0.0055 | +0.0006 | +0.0009 | 2/0/2 |
| V50 cross-model tree rescue | 4 | +0.0098 | +0.0106 | +0.0129 | +0.0011 | +0.0050 | 3/0/1 |
| V50 - V42 subset | 4 | +0.0039 | +0.0067 | +0.0075 | +0.0005 | +0.0041 | no extra negatives |

The scale check is encouraging on hard metrics, but ECE is worse than V42 on
this subset.

## Decision

V50 is a promising new direction and is the first heterogeneous model-member
ensemble that improves over V42 on a matched smoke suite without adding
accuracy/BAcc/Macro-F1 negative splits.

Do not mark it as the new main result yet:

- It still needs the full large no-shift 7-dataset x 2-seed train256/test128
  evaluation.
- It is significantly slower than V42 because it runs V42 internally and may
  train tree members.
- Its calibration trade-off is mixed: train128 ECE is slightly better than V42,
  but the train256 subset has worse ECE.

Recommended next step: optimize/cap the V50 cost, then run the full V42 large
no-shift benchmark and compare hard metrics plus ROC-AUC/ECE before promoting
it.

## Artifacts

- `apa_v50_cross_model_tree_no_shift_smoke4_2seed_train128_r2_detail.csv`
- `apa_v50_cross_model_tree_no_shift_smoke4_2seed_train128_r2_summary.csv`
- `apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2_detail.csv`
- `apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2_summary.csv`
