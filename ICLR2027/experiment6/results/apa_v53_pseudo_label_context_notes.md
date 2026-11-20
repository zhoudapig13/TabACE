# V53 pseudo-label context augmentation attempt

## Goal

This round tests the reference idea of soft/pseudo-label augmentation. Since
TabICL consumes hard labels in the current pipeline, V53 uses high-confidence
test-batch predictions as label-side pseudo-context for low-margin test rows.

This is a transductive inference-time augmentation:

`high-confidence test predictions -> pseudo-labeled context -> low-margin row reprediction -> local/p-value guard`

No test labels are used.

## Method

V53:

1. Run the TabICL baseline.
2. Select a pseudo-label pool from test rows where baseline confidence is at
   least `0.90` and top-2 margin is at least `0.48`.
3. Select at most `12.5%` low-margin target rows.
4. For each target row, augment the training context with pseudo-labeled test
   rows, excluding the target row itself.
5. Refit TabICL for that single target row.
6. Accept a changed prediction only if original-train local KNN support and
   binomial p-value agree with the candidate.

## No-shift train128/test64 smoke

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_pseudo_label_context_router --output-prefix apa_v53_pseudo_label_no_shift_smoke4_2seed_train128
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V53 pseudo-label context | 8 | +0.0020 | +0.0015 | +0.0017 | +0.0005 | -0.0019 | 1/0/7 |

The only positive split was `openml_37_diabetes seed=2`:

| Split | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta |
|---|---:|---:|---:|---:|---:|
| `openml_37_diabetes seed=2` | +0.0156 | +0.0122 | +0.0136 | +0.0042 | -0.0153 |

## No-shift train256/test128 scale check

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_pseudo_label_context_router --output-prefix apa_v53_pseudo_label_no_shift_smoke2_2seed_train256
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V53 pseudo-label context | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/0/4 |

## Decision

Do not deepen V53 now.

The method is safe and cheap enough, but the signal is weak: one positive
train128 split and no train256 gains. This is useful evidence that simple
pseudo-label context augmentation is less promising than cross-model tree
rescue or the V42 primary line.

If pseudo-label augmentation is revisited, it should probably use a stronger
teacher ensemble to produce pseudo-labels, not the baseline alone.

## Artifacts

- `apa_v53_pseudo_label_no_shift_smoke4_2seed_train128_detail.csv`
- `apa_v53_pseudo_label_no_shift_smoke4_2seed_train128_summary.csv`
- `apa_v53_pseudo_label_no_shift_smoke2_2seed_train256_detail.csv`
- `apa_v53_pseudo_label_no_shift_smoke2_2seed_train256_summary.csv`
