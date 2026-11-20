# V52 fast-primary tree rescue attempt

## Goal

V50 cross-model tree rescue was promising but expensive. V52 tests whether the
same tree-rescue idea can be made cheaper by replacing the V42 primary member
with the cheaper V27 guarded-boundary primary member.

This is an efficiency ablation, not a dataset-specific branch:

`fast V27 primary -> same ExtraTrees/RandomForest consensus rescue -> same local/p-value/prior guards`

## Method

V52 reuses the V50 cross-model rescue function with `primary_member="v27_fast"`:

- Primary member: V27 guarded boundary-union with V42-compatible settings.
- Rescue member: ExtraTrees + RandomForest consensus, same as V50.
- Rescue gates: same as V50, including low primary batch prior JS, local support,
  p-value, prior drift, model agreement, and change cap.

## No-shift train128/test64 smoke

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_fast_tree_rescue_router --output-prefix apa_v52_fast_tree_no_shift_smoke4_2seed_train128
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V52 fast tree rescue | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6/0/2 |

This matches V50 on this small-scale suite, because V42 also falls back to V27
on small batches.

## No-shift train256/test128 scale check

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 256 --test-size 128 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_fast_tree_rescue_router --output-prefix apa_v52_fast_tree_no_shift_smoke2_2seed_train256
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V50 same subset | 4 | +0.0098 | +0.0106 | +0.0129 | +0.0011 | +0.0050 | 3/0/1 | 146.4s |
| V52 fast tree rescue | 4 | +0.0020 | -0.0031 | +0.0001 | +0.0003 | +0.0033 | 1/3/0 | 60.5s |

V52 is much faster, but it introduces hard-metric negative splits:

- `openml_37_diabetes seed=1`: Acc/BAcc/Macro-F1 `-0.0078/-0.0128/-0.0117`
- `openml_37_diabetes seed=2`: Acc/BAcc/Macro-F1 `-0.0156/-0.0229/-0.0220`

## Decision

Do not expand V52.

The experiment shows that V50's expensive V42 primary member is not just a cost
overhead; it is also acting as a stabilizer on larger no-shift batches. The
cheaper V27 primary loses that protection and cannot replace V50/V42 at
train256 scale.

## Artifacts

- `apa_v52_fast_tree_no_shift_smoke4_2seed_train128_detail.csv`
- `apa_v52_fast_tree_no_shift_smoke4_2seed_train128_summary.csv`
- `apa_v52_fast_tree_no_shift_smoke2_2seed_train256_detail.csv`
- `apa_v52_fast_tree_no_shift_smoke2_2seed_train256_summary.csv`
