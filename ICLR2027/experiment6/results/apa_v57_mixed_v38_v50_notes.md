# V57 Mixed V38/V50 Router Notes

Date: 2026-07-06

## Motivation

V50 became the strongest pure no-shift large-scale hard-metric candidate after
the full train256 seven-dataset check. V43 was the previous mixed router:

- no-shift large low-JS batches: V42 boundary-bypass scale combo;
- ambiguous/high-JS batches: V38 V14-priority label-shift correction or
  baseline.

V57 updates only the no-shift branch:

- no-shift large low-JS batches: V50 sparse cross-model tree rescue;
- ambiguous/high-JS batches: unchanged V38 label-shift branch.

This is a batch-condition router, not a dataset-specific splice.

## Method

The router first computes the baseline batch-prior JS against the train prior.

- If `batch_prior_js <= min(config.max_batch_prior_js, 0.004)` and
  `train_size >= 192 or test_size >= 96`, use the V50 no-shift member.
- Otherwise use V38:
  - try V14-priority counterfactual label-shift correction;
  - keep the correction only if it changes samples and
    `cf_mean_context_disagreement >= 0.012`;
  - otherwise return baseline.

V50 itself is:

- V42 primary;
- ExtraTrees/RandomForest consensus rescue;
- local support, binomial p-value, prior drift, tree confidence/margin, and
  minority-risk guards.

## Commands

No-shift target, train256/test128:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_1498_sa_heart --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_mixed_v38_v50_router --output-prefix apa_v57_mixed_v38_v50_no_shift_target2_2seed_train256
```

Forced-shift target:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1498_sa_heart --force-label-shift --label-shift-strengths 0.6,0.9 --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_dual_mode_v38,apa_cf_mixed_v38_v50_router --output-prefix apa_v57_mixed_v38_v50_forced_shift_target3_2seed
```

## Results

No-shift target, 2 datasets x 2 seeds:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V57 mixed V38/V50 | 4 | +0.0195 | +0.0249 | +0.0263 | +0.0054 | +0.0147 | 4 / 0 / 0 |

Branch statistics:

- `no_shift_v50_cross_model_tree_rescue`: 4 / 4.

Forced-shift target, 3 datasets x 2 seeds x 2 shift strengths:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V38 V14-priority | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | -0.0065 | 8 / 0 / 4 |
| V57 mixed V38/V50 | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | -0.0065 | 8 / 0 / 4 |

V57 vs V38 forced-shift row-wise check:

- mismatches: 0 / 12
- max absolute metric delta difference: 0.0

Forced-shift branch statistics:

- `label_shift_v38_priority`: 8 / 12
- `baseline_shift_guard`: 4 / 12

## Conclusion

V57 is the natural unified successor to V43:

- It preserves the V38 forced-shift behavior on the targeted forced-shift
  check.
- It routes large no-shift low-JS batches to V50, the current no-shift
  train256 hard-metric best.
- It has no observed Acc/BAcc/F1 negative split in the targeted checks.

However, V43 remains the more fully validated mixed router until V57 is run on
the full V43 forced-shift targeted suite and a broader no-shift confirmation.

