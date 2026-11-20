# V56 PRP-Aware Tree Rescue Notes

Date: 2026-07-06

## Motivation

After the broad V46-V55 screen, V50 was the strongest new line: keep V42 as
the primary no-shift router and allow a low-frequency ExtraTrees/RandomForest
consensus rescue only when local p-value, tree agreement, prior drift, and
minority-risk guards pass.

V54 showed that PRP / multi-context uncertainty diagnostics are useful for
explaining risk, but too conservative as a hard gate on all V42 edits. V56
therefore uses PRP only at the tree-rescue decision point.

## Method

V56 is a strict extension of V50:

1. Run V42 `boundary-bypass scale combo` as the primary member.
2. If primary batch-prior JS is too high, return V42 only.
3. Train ExtraTrees and RandomForest as heterogeneous rescue members.
4. Build candidate rescues only for rows where:
   - V42 does not change the TabICL baseline prediction;
   - ExtraTrees and RandomForest agree;
   - tree margin/confidence are sufficient;
   - local neighbor support and local binomial p-value support the tree class;
   - candidate prior drift and minority-to-majority risk are bounded.
5. Run PRP family diagnostics lazily only if there is a V50 candidate or a
   near-candidate. The diagnostic family contains boundary, diverse,
   prototype, clean, and minority-density contexts.
6. Keep/expand tree rescues when either:
   - the original tree evidence is very strong; or
   - PRP risk is low, family agreement is non-trivial, context disagreement is
     bounded, and local purity is adequate.

Important implementation point: the first eager-PRP smoke was interrupted after
about 9 minutes because it ran the PRP family diagnostics even when no tree
candidate existed. V56 was changed to lazy PRP before final measurement.

## Commands

Key ILPD probe:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1480_ilpd --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_cross_model_tree_rescue_router,apa_cf_no_shift_prp_tree_rescue_router --output-prefix apa_v56_prp_tree_rescue_ilpd_2seed_train128
```

Remaining smoke datasets:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_prp_tree_rescue_router --output-prefix apa_v56_prp_tree_rescue_no_shift_smoke3_2seed_train128
```

Extra 3 datasets from the train128 seven-dataset view:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1498_sa_heart,openml_1464_blood_transfusion,openml_1050_pc3 --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_prp_tree_rescue_router --output-prefix apa_v56_prp_tree_rescue_no_shift_extra3_2seed_train128
```

## Results

Four-dataset smoke, train128/test64, 2 seeds:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V42 boundary-bypass | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| V50 cross-model tree rescue | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |
| V56 PRP-aware tree rescue | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |

Seven-dataset train128/test64 view, 2 seeds:

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero | Mean runtime | Tree-selected fraction | PRP-eval fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V42 boundary-bypass | 14 | +0.0089 | +0.0086 | +0.0095 | +0.0018 | -0.0054 | 6 / 0 / 8 | 28.1s | 0.0000 | 0.000 |
| V50 cross-model tree rescue | 14 | +0.0089 | +0.0093 | +0.0105 | +0.0017 | -0.0054 | 6 / 0 / 8 | 26.7s | 0.0022 | 0.000 |
| V56 PRP-aware tree rescue | 14 | +0.0089 | +0.0093 | +0.0105 | +0.0017 | -0.0054 | 6 / 0 / 8 | 27.3s | 0.0022 | 0.100 |

V56 - V42 on the seven-dataset view:

- Acc: +0.0000
- BAcc: +0.0008
- Macro-F1: +0.0010
- ROC-AUC: -0.0001
- ECE: -0.0001

PRP was active only on `openml_1480_ilpd seed=2`, where it preserved the V50
tree rescue:

- selected fraction: 0.03125
- PRP risk among selected rows: 0.2856
- PRP family agreement among selected rows: 0.4000
- rejected V50 candidate fraction: 0.0

## Conclusion

V56 is safe but does not improve over V50 on current no-shift train128 tests.
The useful lesson is architectural:

- PRP diagnostics are too heavy and too conservative as a global hard gate.
- Lazy PRP is cheap enough when used only around rare rescue candidates.
- The best current paper line remains V50/V56 style:
  **V42 primary + sparse heterogeneous rescue + optional PRP reliability
  diagnostics**.

For prediction metrics, V50 remains the simpler preferred version. V56 is worth
keeping as the explainable/diagnostic variant of the same unified framework.

