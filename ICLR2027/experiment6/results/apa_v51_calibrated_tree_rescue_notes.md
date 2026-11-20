# V51 calibration-aware tree rescue attempt

## Goal

V50 improved hard metrics over V42 on matched no-shift smoke suites, but its
train256 scale check showed mixed calibration behavior. V51 tests whether the
accepted tree-rescue samples can be calibrated without changing V50's hard
selection logic.

## Method

V51 reuses V50 exactly up to the accepted tree-rescue mask:

`V42 primary -> tree consensus rescue eligibility -> local/p-value/prior guards`

The only change is probability calibration on rows where tree rescue is
accepted. For each accepted row, V51 increases the rescued class confidence
according to:

- local support advantage;
- tree margin;
- local support p-value.

The hard prediction is preserved. Non-tree-rescue rows are identical to V50 and
V42 primary behavior.

## Probe

The first probe used the only train128 smoke split where V50 selected tree
rescues: `openml_1480_ilpd seed=2`.

| Method | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta |
|---|---:|---:|---:|---:|---:|
| V50 | +0.0312 | +0.0455 | +0.0541 | +0.0043 | -0.0331 |
| V51 | +0.0312 | +0.0455 | +0.0541 | +0.0043 | -0.0173 |

V51 preserved hard metrics but weakened the ECE improvement. The mean
calibration boost on selected samples was `+0.2079`, which was too aggressive
for this split.

## No-shift train128/test64 smoke

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_calibrated_tree_rescue_router --output-prefix apa_v51_calibrated_tree_no_shift_smoke4_2seed_train128
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V50 cross-model tree rescue | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6/0/2 |
| V51 calibrated tree rescue | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0074 | 6/0/2 |

## Decision

Do not expand V51.

V51 is not harmful to hard metrics, but it is strictly worse than V50 on ECE in
the matched smoke. The result suggests that post-hoc confidence boosting on
rescued tree samples is not a good calibration strategy. If calibration is
revisited, it should be a route-level temperature/gate evaluated by NLL/Brier/ECE
rather than manually increasing confidence on selected rows.

## Artifacts

- `apa_v51_calibrated_tree_ilpd_s2_probe_detail.csv`
- `apa_v51_calibrated_tree_ilpd_s2_probe_summary.csv`
- `apa_v51_calibrated_tree_no_shift_smoke4_2seed_train128_detail.csv`
- `apa_v51_calibrated_tree_no_shift_smoke4_2seed_train128_summary.csv`
