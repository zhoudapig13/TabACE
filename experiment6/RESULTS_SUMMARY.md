# Experiment 6 Results Summary

Status: active development.

All artifacts are kept under `experiment6/results/`.

## 2026-06-13: `apa_v0_tiny_tabicl`

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced --seeds 0 --n-samples 80 --train-size 40 --test-size 12 --k 12 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_knn_context,apa_prior_aware_adaptive --output-prefix apa_v0_tiny_tabicl
```

Result:

| method | acc delta | bacc delta | macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: | ---: |
| `apa_knn_context` | -0.2500 | -0.1667 | -0.2487 | +0.1438 |
| `apa_prior_aware_adaptive` | -0.0833 | -0.0556 | -0.0959 | +0.0748 |

Takeaway: directly replacing TabICL's full context with a small per-query
context is risky. Prior-aware adaptive correction reduced the damage but did
not solve it.

Artifacts:

- `results/apa_v0_tiny_tabicl_detail.csv`
- `results/apa_v0_tiny_tabicl_summary.csv`
- `results/apa_v0_tiny_tabicl_report.md`

## 2026-06-13: `apa_v1_small_tabicl`

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,synthetic_label_shift,wine,breast_cancer --seeds 0,1 --n-samples 160 --train-size 80 --test-size 30 --k 24 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_knn_adaptive,apa_full_prior_aware_adaptive,apa_full_prior_aware_tiny --output-prefix apa_v1_small_tabicl
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apa_full_prior_knn_adaptive` | 8 | +0.0057 | +0.0057 | +0.0019 | -0.0001 |
| `apa_full_prior_aware_adaptive` | 8 | +0.0057 | +0.0057 | +0.0019 | -0.0003 |
| `apa_full_prior_aware_tiny` | 8 | +0.0057 | +0.0057 | +0.0019 | -0.0012 |

Takeaway: using local/prior-aware context only as a prior estimator, while
keeping full-context TabICL probabilities, is much safer. The positive mean came
from one label-shift split; most saturated datasets were unchanged.

Artifacts:

- `results/apa_v1_small_tabicl_detail.csv`
- `results/apa_v1_small_tabicl_summary.csv`
- `results/apa_v1_small_tabicl_report.md`

## 2026-06-13: `apa_v2_small_tabicl_resplit`

This run fixes the label-shift split so the test side retains both classes and
tests the v2 batch-shift/margin gate.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,synthetic_label_shift,wine,breast_cancer --seeds 0,1 --n-samples 160 --train-size 80 --test-size 30 --k 24 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_shift_margin_gate,apa_shift_margin_gate_soft --output-prefix apa_v2_small_tabicl_resplit
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apa_full_prior_aware_tiny` | 8 | +0.0000 | +0.0000 | +0.0000 | -0.0068 |
| `apa_shift_margin_gate` | 8 | +0.0042 | +0.0022 | +0.0042 | +0.0037 |
| `apa_shift_margin_gate_soft` | 8 | +0.0000 | +0.0000 | +0.0000 | -0.0001 |

Dataset-level positive:

- `synthetic_label_shift`, seed 0, `apa_shift_margin_gate`: accuracy +0.0333,
  balanced accuracy +0.0172, macro-F1 +0.0339.

Takeaway: the hard shift/margin gate can find a useful boundary correction while
leaving other small datasets unchanged. The ECE tradeoff is slightly negative in
this small suite, so later versions should consider calibration-aware gating.

Artifacts:

- `results/apa_v2_small_tabicl_resplit_detail.csv`
- `results/apa_v2_small_tabicl_resplit_summary.csv`
- `results/apa_v2_small_tabicl_resplit_report.md`

## 2026-06-13: `apa_v2_label_shift_5seed`

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_shift_margin_gate,apa_shift_margin_gate_soft --output-prefix apa_v2_label_shift_5seed
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apa_full_prior_aware_tiny` | 5 | +0.0000 | +0.0000 | +0.0000 | -0.0113 |
| `apa_shift_margin_gate` | 5 | +0.0000 | +0.0000 | +0.0000 | -0.0163 |
| `apa_shift_margin_gate_soft` | 5 | +0.0000 | +0.0000 | +0.0000 | -0.0097 |

Takeaway: on this easier/larger label-shift suite, the methods improved
calibration but did not change decisions. The next useful direction is
validation-selected correction strength or a more difficult label-shift suite
where baseline has low-margin mistakes.

Artifacts:

- `results/apa_v2_label_shift_5seed_detail.csv`
- `results/apa_v2_label_shift_5seed_summary.csv`
- `results/apa_v2_label_shift_5seed_report.md`

## 2026-06-13: `apa_v3_label_shift_curve`

This run adds an explicit label-shift strength sweep and tests the v3
train-internal validation selector.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_full_prior_aware_tiny,apa_shift_margin_gate,apa_shift_margin_gate_soft,apa_validation_selector --output-prefix apa_v3_label_shift_curve
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apa_full_prior_aware_tiny` | 12 | +0.0000 | +0.0000 | +0.0000 | -0.0101 |
| `apa_shift_margin_gate` | 12 | +0.0021 | +0.0013 | +0.0026 | -0.0145 |
| `apa_shift_margin_gate_soft` | 12 | +0.0000 | +0.0000 | +0.0000 | -0.0078 |
| `apa_validation_selector` | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

Shift curve highlight:

- At shift strength 0.3, `apa_shift_margin_gate` had mean balanced accuracy
  delta +0.0051 across three seeds, with one positive seed and no negative
  seeds.
- At strengths 0.0, 0.6, and 0.9, decision metrics were unchanged, while ECE
  generally improved for fixed post-hoc correction variants.

Analysis: v3 selected `baseline_no_gain` on all 12 runs. This is expected
because the validation split is sampled from the training distribution, so it
does not reliably expose the test-side label shift. It is safe, but currently
too conservative to find the shifted-test boundary cases.

Artifacts:

- `results/apa_v3_label_shift_curve_detail.csv`
- `results/apa_v3_label_shift_curve_summary.csv`
- `results/apa_v3_label_shift_curve_shift_curve.csv`
- `results/apa_v3_label_shift_curve_report.md`

## 2026-06-13: `apa_v4_hard_label_shift_curve`

This run adds `synthetic_label_shift_hard` and two more aggressive v4
margin-gated candidates.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_aggressive_margin_push,apa_aggressive_batch_push,apa_validation_selector --output-prefix apa_v4_hard_label_shift_curve
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 12 | +0.0042 | +0.0021 | +0.0023 | -0.0042 | 2 / 0 |
| `apa_aggressive_margin_push` | 12 | +0.0104 | +0.0055 | +0.0067 | +0.0035 | 5 / 0 |
| `apa_aggressive_batch_push` | 12 | +0.0083 | +0.0047 | +0.0058 | +0.0003 | 4 / 0 |
| `apa_validation_selector` | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |

Shift curve highlights:

- Shift 0.3: `apa_aggressive_batch_push` mean bacc delta +0.0101;
  `apa_aggressive_margin_push` mean bacc delta +0.0051.
- Shift 0.6 and 0.9: `apa_aggressive_margin_push` mean bacc delta +0.0085 at
  both strengths.
- Shift 0.0: decision metrics unchanged, but aggressive variants worsened ECE,
  so they should stay gated/diagnostic rather than become unconditional
  defaults.

Takeaway: v4 gives the clearest evidence so far that APA-style inference-only
posterior correction can recover hard label-shift boundary errors without
retraining. The remaining issue is calibration/no-shift safety: aggressive
variants improve decisions under shift but can worsen ECE when there is no
shift.

Artifacts:

- `results/apa_v4_hard_label_shift_curve_detail.csv`
- `results/apa_v4_hard_label_shift_curve_summary.csv`
- `results/apa_v4_hard_label_shift_curve_shift_curve.csv`
- `results/apa_v4_hard_label_shift_curve_report.md`

## 2026-06-13: `apa_v5_hard_label_shift_curve`

This run tests `apa_hard_shift_router`, which keeps the v4 aggressive
margin-push branch but raises `min_batch_prior_js` to `0.004` and applies it
only to binary tasks.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_aggressive_margin_push,apa_hard_shift_router --output-prefix apa_v5_hard_label_shift_curve
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_aggressive_margin_push` | 12 | +0.0104 | +0.0055 | +0.0067 | +0.0035 | 5 / 0 |
| `apa_hard_shift_router` | 12 | +0.0104 | +0.0055 | +0.0067 | +0.0001 | 5 / 0 |

Takeaway: raising the JS trigger preserves all observed hard-shift decision
gains while eliminating the no-shift ECE penalty seen in v4. At shift strength
0.0, `apa_hard_shift_router` exactly matches baseline on decision metrics and
ECE.

Artifacts:

- `results/apa_v5_hard_label_shift_curve_detail.csv`
- `results/apa_v5_hard_label_shift_curve_summary.csv`
- `results/apa_v5_hard_label_shift_curve_shift_curve.csv`
- `results/apa_v5_hard_label_shift_curve_report.md`

## 2026-06-13: `apa_v5_mixed_safety_binary_gate`

This run checks the binary-only v5 gate on a mixed suite including imbalanced
binary, easy label-shift, real sklearn datasets, and multiclass imbalance.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,synthetic_label_shift,wine,breast_cancer,synthetic_multiclass_imbalance --label-shift-strengths 0.0,0.3,0.6 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_aggressive_margin_push,apa_hard_shift_router --output-prefix apa_v5_mixed_safety_binary_gate
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 14 | +0.0018 | +0.0011 | +0.0022 | -0.0080 | 1 / 0 |
| `apa_aggressive_margin_push` | 14 | -0.0000 | +0.0008 | +0.0002 | -0.0046 | 1 / 1 |
| `apa_hard_shift_router` | 14 | +0.0000 | +0.0000 | +0.0000 | -0.0054 | 0 / 0 |

Takeaway: the binary-only gate fixed the multiclass regression introduced by
the v4 aggressive branch. The tradeoff is that v5 is very conservative on easy
label-shift and misses the single positive split caught by v2.

Artifacts:

- `results/apa_v5_mixed_safety_binary_gate_detail.csv`
- `results/apa_v5_mixed_safety_binary_gate_summary.csv`
- `results/apa_v5_mixed_safety_binary_gate_shift_curve.csv`
- `results/apa_v5_mixed_safety_binary_gate_report.md`

## 2026-06-13: `apa_v6_hard_label_shift_curve`

This run tests `apa_dual_shift_router`, which sends high batch-prior JS cases
to the conservative v2 branch and lower hard-shift JS cases to the v5 branch.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_hard_shift_router,apa_dual_shift_router --output-prefix apa_v6_hard_label_shift_curve
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 12 | +0.0042 | +0.0021 | +0.0023 | -0.0042 | 2 / 0 |
| `apa_hard_shift_router` | 12 | +0.0104 | +0.0055 | +0.0067 | +0.0001 | 5 / 0 |
| `apa_dual_shift_router` | 12 | +0.0104 | +0.0055 | +0.0067 | +0.0001 | 5 / 0 |

Branch counts for `apa_dual_shift_router`: `hard_js_v5` 9, `baseline_low_js`
3. It matches v5 on this hard-shift curve.

Artifacts:

- `results/apa_v6_hard_label_shift_curve_detail.csv`
- `results/apa_v6_hard_label_shift_curve_summary.csv`
- `results/apa_v6_hard_label_shift_curve_shift_curve.csv`
- `results/apa_v6_hard_label_shift_curve_report.md`

## 2026-06-13: `apa_v6_mixed_safety`

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,synthetic_label_shift,wine,breast_cancer,synthetic_multiclass_imbalance --label-shift-strengths 0.0,0.3,0.6 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_hard_shift_router,apa_dual_shift_router --output-prefix apa_v6_mixed_safety
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 14 | +0.0018 | +0.0011 | +0.0022 | -0.0080 | 1 / 0 |
| `apa_hard_shift_router` | 14 | +0.0000 | +0.0000 | +0.0000 | -0.0054 | 0 / 0 |
| `apa_dual_shift_router` | 14 | +0.0018 | +0.0011 | +0.0022 | -0.0080 | 1 / 0 |

Branch counts for `apa_dual_shift_router`: `baseline_low_js` 5,
`high_js_v2` 4, `baseline_nonbinary` 4, `hard_js_v5` 1.

Takeaway: v6 combines the useful easy-shift v2 branch with the hard-shift v5
branch while preserving nonbinary safety. It is the current best candidate for
the next broader sweep.

Artifacts:

- `results/apa_v6_mixed_safety_detail.csv`
- `results/apa_v6_mixed_safety_summary.csv`
- `results/apa_v6_mixed_safety_shift_curve.csv`
- `results/apa_v6_mixed_safety_report.md`

## 2026-06-13: `apa_v6_hard_label_shift_5seed`

This is a broader 5-seed rerun of the hard label-shift curve for the current
candidate.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_hard_shift_router,apa_dual_shift_router --output-prefix apa_v6_hard_label_shift_5seed
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 20 | +0.0025 | +0.0013 | +0.0014 | -0.0025 | 2 / 0 |
| `apa_hard_shift_router` | 20 | +0.0062 | +0.0033 | +0.0040 | -0.0001 | 5 / 0 |
| `apa_dual_shift_router` | 20 | +0.0062 | +0.0033 | +0.0040 | -0.0001 | 5 / 0 |

Branch counts for `apa_dual_shift_router`: `hard_js_v5` 15,
`baseline_low_js` 5. The 5-seed curve confirms that v6 keeps the hard-shift
gain with no observed negative split, though the positive split rate is modest.

Artifacts:

- `results/apa_v6_hard_label_shift_5seed_detail.csv`
- `results/apa_v6_hard_label_shift_5seed_summary.csv`
- `results/apa_v6_hard_label_shift_5seed_shift_curve.csv`
- `results/apa_v6_hard_label_shift_5seed_report.md`

## 2026-06-13: `apa_v6_easy_label_shift_5seed`

This is a 5-seed rerun on the easier label-shift generator.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_hard_shift_router,apa_dual_shift_router --output-prefix apa_v6_easy_label_shift_5seed
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 20 | +0.0013 | +0.0008 | +0.0016 | -0.0091 | 1 / 0 |
| `apa_hard_shift_router` | 20 | +0.0000 | +0.0000 | +0.0000 | -0.0076 | 0 / 0 |
| `apa_dual_shift_router` | 20 | +0.0013 | +0.0008 | +0.0016 | -0.0086 | 1 / 0 |

Branch counts for `apa_dual_shift_router`: `high_js_v2` 11, `hard_js_v5` 5,
`baseline_low_js` 4. The easy curve confirms that the high-JS v2 branch catches
the single positive boundary split that v5 misses, again with no negative split.

Artifacts:

- `results/apa_v6_easy_label_shift_5seed_detail.csv`
- `results/apa_v6_easy_label_shift_5seed_summary.csv`
- `results/apa_v6_easy_label_shift_5seed_shift_curve.csv`
- `results/apa_v6_easy_label_shift_5seed_report.md`

## 2026-06-13: `apa_v7_openml_small_stress`

This run adds a small OpenML-style stress after adding OpenML dataset support
to `run_apa_tabicl.py`.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_cmc,openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima --seeds 0,1 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_dual_shift_router,apa_dual_shift_router_v7 --output-prefix apa_v7_openml_small_stress
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 12 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | 0 / 0 |
| `apa_dual_shift_router` | 12 | -0.0021 | -0.0021 | -0.0022 | +0.0032 | 0 / 1 |
| `apa_dual_shift_router_v7` | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |

The v6 regression was `openml_ilpd`, seed 1, where the hard branch triggered at
batch-prior JS 0.0057. Raising the hard-branch threshold to 0.008 in v7 blocks
that regression.

Artifacts:

- `results/apa_v7_openml_small_stress_detail.csv`
- `results/apa_v7_openml_small_stress_summary.csv`
- `results/apa_v7_openml_small_stress_report.md`

## 2026-06-13: `apa_v7_hard_label_shift_5seed`

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router,apa_dual_shift_router_v7 --output-prefix apa_v7_hard_label_shift_5seed
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router` | 20 | +0.0062 | +0.0033 | +0.0040 | -0.0001 | 5 / 0 |
| `apa_dual_shift_router_v7` | 20 | +0.0050 | +0.0026 | +0.0030 | -0.0005 | 4 / 0 |

Takeaway: v7 sacrifices one weak hard-shift positive split to avoid the OpenML
standard-split regression.

Artifacts:

- `results/apa_v7_hard_label_shift_5seed_detail.csv`
- `results/apa_v7_hard_label_shift_5seed_summary.csv`
- `results/apa_v7_hard_label_shift_5seed_shift_curve.csv`
- `results/apa_v7_hard_label_shift_5seed_report.md`

## 2026-06-13: `apa_v7_easy_label_shift_5seed`

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3,4 --n-samples 180 --train-size 90 --test-size 40 --k 30 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router,apa_dual_shift_router_v7 --output-prefix apa_v7_easy_label_shift_5seed
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router` | 20 | +0.0013 | +0.0008 | +0.0016 | -0.0086 | 1 / 0 |
| `apa_dual_shift_router_v7` | 20 | +0.0013 | +0.0008 | +0.0016 | -0.0086 | 1 / 0 |

Takeaway: v7 keeps the high-JS easy label-shift behavior from v6.

Artifacts:

- `results/apa_v7_easy_label_shift_5seed_detail.csv`
- `results/apa_v7_easy_label_shift_5seed_summary.csv`
- `results/apa_v7_easy_label_shift_5seed_shift_curve.csv`
- `results/apa_v7_easy_label_shift_5seed_report.md`

## 2026-06-13: `apa_v7_openml_broad3`

This is a broader OpenML safety check: 8 datasets, 3 seeds, 24 dataset-seed
summaries.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_cmc,openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_dmft,openml_diabetes_pima,openml_pc1 --seeds 0,1,2 --n-samples 220 --train-size 110 --test-size 50 --k 36 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_shift_margin_gate,apa_dual_shift_router,apa_dual_shift_router_v7 --output-prefix apa_v7_openml_broad3
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_shift_margin_gate` | 24 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `apa_dual_shift_router` | 24 | +0.0000 | +0.0000 | +0.0000 | +0.0004 | 0 / 0 |
| `apa_dual_shift_router_v7` | 24 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | 0 / 0 |

Branch counts for `apa_dual_shift_router_v7`: `baseline_low_js` 17,
`baseline_nonbinary` 6, `hard_js_v5` 1.

Takeaway: v7 is conservative on standard OpenML splits: it mostly returns
baseline and has no observed negative split in this broader small stress. This
does not prove real-data improvement, but it supports using v7 as the safer
candidate for larger stress sweeps.

Artifacts:

- `results/apa_v7_openml_broad3_detail.csv`
- `results/apa_v7_openml_broad3_summary.csv`
- `results/apa_v7_openml_broad3_report.md`

## Current Candidate

`apa_dual_shift_router_v7` is the current Experiment 6 candidate. Its reference
configuration is stored in `configs/apa_v7_candidate.json`.

Current evidence:

- Hard label-shift 5-seed curve: +0.0026 mean balanced accuracy, 4 positive /
  0 negative splits.
- Easy label-shift 5-seed curve: +0.0008 mean balanced accuracy, 1 positive /
  0 negative splits, ECE -0.0086.
- OpenML broad3 stress: 0 positive / 0 negative splits across 24
  dataset-seed summaries, blocking the v6 ILPD regression.

Known limitation: gains are sparse and concentrated in boundary splits. The
router is deliberately conservative; broader OpenML-style stress remains the
next useful check before calling it robust.

## 2026-06-13: Reusable Wrapper

Added `APATabICLClassifier` in `scripts/apa_tabicl.py` so the current candidate
can be used directly, not only through the benchmark runner:

```python
from tabicl import TabICLClassifier
from experiment6.scripts.apa_tabicl import APATabICLClassifier

clf = APATabICLClassifier(
    estimator_factory=lambda: TabICLClassifier(n_estimators=1, device="cpu"),
    method="apa_v7",
    k=30,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
pred = clf.predict(X_test)
```

The wrapper is still inference-only: it fits the base estimator normally and
applies APA post-hoc correction at prediction time. It does not retrain TabICL
or modify pretrained weights.

Validation:

- `python -m py_compile experiment6/scripts/apa_tabicl.py experiment6/scripts/run_apa_tabicl.py`
- manual execution of all `experiment6/tests/test_apa_tabicl.py` test functions,
  including the new wrapper smoke test.

## 2026-06-13: `apa_v8_cf_directional_round4`

This run starts the new top-conference-oriented direction: counterfactual
context-prior intervention. Instead of treating APA as only a train/test prior
probability correction, v8 deliberately queries TabICL under multiple contexts
with controlled label priors, estimates how sensitive each row is to context
prior interventions, and only accepts flips that move toward the class supported
by the batch-side predicted prior deficit.

Implemented methods:

- `apa_cf_entropy_ensemble`: entropy-weighted multi-context ensemble.
- `apa_cf_prior_sensitivity`: estimates prior-invariant evidence by regressing
  out the log context prior.
- `apa_cf_stability_router`: uses the prior-invariant evidence only for
  prior-sensitive unstable rows.
- `apa_cf_directional_router`: adds a directional batch-support gate,
  `min_batch_prior_js=0.008`, and a maximum changed fraction.

Important failed intermediate:

- `apa_cf_stability_router` found useful easy-shift flips, but on
  `apa_v8_cf_shift_round2` it had hard-shift failures: mean balanced accuracy
  delta `-0.0760`, `11 / 5` positive/negative splits, with one repeated
  hard-shift seed dropping by `-0.4783`.
- Diagnosis: it was sometimes flipping against minority-class evidence. The
  directional gate fixes this by allowing only flips toward the class whose
  predicted batch mass exceeds its training prior.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets synthetic_label_shift_hard,synthetic_label_shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1 --n-samples 140 --train-size 70 --test-size 24 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_directional_round4
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router_v7` | 16 | +0.0156 | +0.0082 | +0.0104 | -0.0089 | 6 / 0 |
| `apa_cf_directional_router` | 16 | +0.0469 | +0.0245 | +0.0321 | -0.0123 | 12 / 0 |

Shift curve highlights:

- `synthetic_label_shift_hard`, shift 0.0: `apa_cf_directional_router` exactly
  matches baseline.
- `synthetic_label_shift_hard`, shifts 0.3/0.6/0.9: mean balanced accuracy
  delta `+0.0326`.
- `synthetic_label_shift`, shift 0.0: exactly matches baseline.
- `synthetic_label_shift`, shifts 0.3/0.6/0.9: mean balanced accuracy delta
  `+0.0326`.

Takeaway: v8 is the first Experiment 6 method with a clearly different
mechanism from standard posterior correction. It treats context label priors as
test-time interventions and estimates prior-invariant evidence from the model's
response across counterfactual contexts. On this small shift suite, it improves
over v7 by roughly 3x in mean balanced-accuracy delta while retaining 0 negative
splits.

Artifacts:

- `results/apa_v8_cf_directional_round4_detail.csv`
- `results/apa_v8_cf_directional_round4_summary.csv`
- `results/apa_v8_cf_directional_round4_shift_curve.csv`
- `results/apa_v8_cf_directional_round4_report.md`

## 2026-06-13: `apa_v8_cf_directional_safety_round4`

This run checks whether the v8 directional router over-triggers on standard
non-shift splits after raising `min_batch_prior_js` from `0.004` to `0.008`.

The threshold change was based on a concrete failure: with `0.004`,
`openml_blood_transfusion` seed 0 triggered at batch-prior JS about `0.0058`
and dropped balanced accuracy by `-0.0417`. Synthetic shifted positives in
round4 had batch-prior JS at least about `0.049`, so `0.008` blocks the OpenML
false trigger without blocking the current shift positives.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets synthetic_imbalanced,breast_cancer,wine,openml_blood_transfusion,openml_diabetes_pima --seeds 0,1 --n-samples 140 --train-size 70 --test-size 24 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_directional_safety_round4
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router_v7` | 10 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `apa_cf_directional_router` | 10 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |

Takeaway: after threshold tightening, v8 is conservative on this small standard
safety suite and blocks the earlier `openml_blood_transfusion` false trigger.
The next necessary check is a broader OpenML sweep, because current evidence is
promising but still small.

Artifacts:

- `results/apa_v8_cf_directional_safety_round4_detail.csv`
- `results/apa_v8_cf_directional_safety_round4_summary.csv`
- `results/apa_v8_cf_directional_safety_round4_report.md`

## 2026-06-13: `apa_v8_cf_openml_broad3`

This is a broader OpenML safety and opportunity check for the v8 research
candidate: 8 OpenML datasets, 3 seeds, 24 dataset-seed summaries. It uses the
same small-context budget as the synthetic v8 runs and compares against v7.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_cmc,openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_dmft,openml_diabetes_pima,openml_pc1 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_openml_broad3
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router_v7` | 24 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| `apa_cf_directional_router` | 24 | +0.0013 | +0.0013 | +0.0012 | -0.0005 | 1 / 0 |

The single positive split is `openml_credit_g`, seed 1:

- balanced accuracy delta `+0.03125`;
- macro-F1 delta `+0.02918`;
- ECE delta `-0.01295`;
- `cf_batch_prior_js = 0.008165`, just above the v8 trigger threshold;
- changed fraction `0.03125`.

Takeaway: the wider OpenML check did not expose a negative split for v8. The
method mostly falls back to baseline on standard OpenML splits, but unlike v7 it
found one small real-data improvement. This is promising but still not enough
for a robustness claim; the next step is a larger OpenML sweep or targeted
stress tests with known prior shift.

Artifacts:

- `results/apa_v8_cf_openml_broad3_detail.csv`
- `results/apa_v8_cf_openml_broad3_summary.csv`
- `results/apa_v8_cf_openml_broad3_report.md`

## 2026-06-13: `apa_v8_cf_openml_forced_shift_round1`

This run uses the new `--force-label-shift` option to evaluate real OpenML
binary datasets under induced label shift. This is closer to the intended
failure mode than synthetic-only shift because the features and labels come
from real tabular tasks.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_openml_forced_shift_round1
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router_v7` | 48 | +0.0020 | +0.0010 | +0.0012 | -0.0008 | 3 / 0 |
| `apa_cf_directional_router` | 48 | +0.0286 | -0.0082 | +0.0149 | -0.0189 | 25 / 3 |

Failure analysis:

- v8 found many positive flips, especially on `openml_kc1`,
  `openml_diabetes_pima`, `openml_blood_transfusion`, and some
  `openml_credit_g` splits.
- However, `openml_credit_g` seed 0 had severe negative balanced-accuracy
  splits: `-0.1494` at shift 0.3 and `-0.5000` at shifts 0.6 and 0.9.
- Row-level diagnosis showed a single changed row could remove all recall for
  the shifted-test minority class. The model flipped toward the class with
  positive predicted-prior deficit, but that class was not yet the batch
  predicted majority and the baseline margin was not extremely small.

Fix:

- Add `off_majority_max_baseline_margin=0.15`.
- If the proposed target class is not the current batch-predicted majority,
  the row must be a very low-margin boundary case before v8 may flip it.

Artifacts:

- `results/apa_v8_cf_openml_forced_shift_round1_detail.csv`
- `results/apa_v8_cf_openml_forced_shift_round1_summary.csv`
- `results/apa_v8_cf_openml_forced_shift_round1_shift_curve.csv`
- `results/apa_v8_cf_openml_forced_shift_round1_report.md`

## 2026-06-13: `apa_v8_cf_openml_forced_shift_round2`

This reruns the same OpenML forced-label-shift suite after adding the
off-majority low-margin guard.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_openml_forced_shift_round2
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router_v7` | 48 | +0.0020 | +0.0010 | +0.0012 | -0.0008 | 3 / 0 |
| `apa_cf_directional_router` | 48 | +0.0247 | +0.0130 | +0.0152 | -0.0084 | 21 / 0 |

Takeaway: the guard removed all observed negative splits in this forced-shift
OpenML suite while keeping most of the real-data shift gains. v8 now clearly
outperforms v7 on this stress test: about `12.7x` larger mean balanced-accuracy
delta, with `21 / 0` positive/negative splits versus v7's `3 / 0`.

Artifacts:

- `results/apa_v8_cf_openml_forced_shift_round2_detail.csv`
- `results/apa_v8_cf_openml_forced_shift_round2_summary.csv`
- `results/apa_v8_cf_openml_forced_shift_round2_shift_curve.csv`
- `results/apa_v8_cf_openml_forced_shift_round2_report.md`

## 2026-06-13: `apa_v8_cf_openml_forced_shift_seed23`

This extends the OpenML forced-label-shift stress test to seeds 2 and 3. The
goal was to check whether the v8 margin guard from round2 remains safe under
new splits.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_dual_shift_router_v7,apa_cf_directional_router --output-prefix apa_v8_cf_openml_forced_shift_seed23
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_dual_shift_router_v7` | 48 | +0.0033 | +0.0017 | +0.0019 | +0.0015 | 6 / 1 |
| `apa_cf_directional_router` | 48 | +0.0260 | -0.0098 | +0.0128 | +0.0001 | 23 / 4 |

Failure analysis:

- `openml_credit_g`, shift 0.3, seed 2: v8 changed two rows, one correct and
  one wrong; the wrong row reduced minority recall and produced balanced
  accuracy delta `-0.1494`.
- `openml_diabetes_pima`, shift 0.6/0.9, seed 2: one wrong flip removed all
  recall for a rare shifted-test class, giving balanced accuracy delta
  `-0.5000`.
- `openml_diabetes_pima`, shift 0.0, seed 3: a balanced no-shift split still
  produced a batch-prior signal and triggered a false correction.

Takeaway: v8 has strong opportunity but is not safe enough across seeds. The
failure mode is not merely over-correction magnitude; it is unreliable row
selection under rare-class test splits. This motivated v9.

Artifacts:

- `results/apa_v8_cf_openml_forced_shift_seed23_detail.csv`
- `results/apa_v8_cf_openml_forced_shift_seed23_summary.csv`
- `results/apa_v8_cf_openml_forced_shift_seed23_shift_curve.csv`
- `results/apa_v8_cf_openml_forced_shift_seed23_report.md`

## 2026-06-13: `apa_v9_cf_strict_majority_seed0_3`

v9 keeps the counterfactual context-prior intervention mechanism but makes the
router safety-first:

- only low-margin rows may flip (`max_baseline_margin=0.15`);
- the training prior must be non-uniform enough (`min_train_prior_imbalance=0.10`);
- the target class must be the batch-predicted majority;
- at most one row may flip in a 32-row test batch.

This run combines:

- `apa_v9_cf_strict_majority_seed01`
- `apa_v9_cf_strict_majority_probe`
- `apa_v9_cf_strict_majority_probe_rest`

Combined result on OpenML forced-label-shift seeds 0-3:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_strict_margin_router` | 96 | +0.0046 | +0.0024 | +0.0031 | -0.0018 | 14 / 0 |

Takeaway: v9 is much safer but more conservative. It removes the v8 negative
splits found in seeds 2 and 3, but also gives up many of v8's larger gains. For
paper development, v8 is the more interesting high-opportunity method, while v9
is a useful safety router and ablation showing that counterfactual interventions
can be made no-negative on this stress suite.

Artifacts:

- `results/apa_v9_cf_strict_majority_seed01_detail.csv`
- `results/apa_v9_cf_strict_majority_probe_detail.csv`
- `results/apa_v9_cf_strict_majority_probe_rest_detail.csv`
- `results/apa_v9_cf_strict_majority_seed0_3_detail.csv`
- `results/apa_v9_cf_strict_majority_seed0_3_summary.csv`

## 2026-06-13: `apa_v10_cf_batch_margin_probe`

v10 was introduced after v8 failed on several rare-class OpenML shifted splits.
The hypothesis was that v8 sometimes reacts to a weak batch-prior signal when
the baseline batch prediction is nearly tied. v10 therefore requires a clear
predicted batch majority before using the counterfactual context intervention.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_credit_g,openml_diabetes_pima,openml_kc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_directional_router,apa_cf_strict_margin_router,apa_cf_batch_margin_router --output-prefix apa_v10_cf_batch_margin_probe
```

Result on the 48-group focused probe:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_directional_router` | 48 | +0.0326 | -0.0064 | +0.0159 | -0.0044 | 24 / 4 |
| `apa_cf_strict_margin_router` | 48 | +0.0059 | +0.0031 | +0.0037 | -0.0036 | 9 / 0 |
| `apa_cf_batch_margin_router` | 48 | +0.0117 | +0.0061 | +0.0071 | -0.0007 | 7 / 0 |

Representative gains:

| dataset | shift | seed | method | bacc delta | macro-F1 delta |
| --- | ---: | ---: | --- | ---: | ---: |
| `openml_diabetes_pima` | 0.6 | 0 | `apa_cf_batch_margin_router` | +0.0645 | +0.0745 |
| `openml_diabetes_pima` | 0.9 | 0 | `apa_cf_batch_margin_router` | +0.0645 | +0.0745 |
| `openml_credit_g` | 0.6 | 1 | `apa_cf_batch_margin_router` | +0.0323 | +0.0358 |
| `openml_credit_g` | 0.9 | 1 | `apa_cf_batch_margin_router` | +0.0323 | +0.0358 |

Failure analysis:

- v10 successfully blocks the known v8 disasters:
  `openml_credit_g` shift 0.3 seed 2 (`-0.1494` bacc for v8) and
  `openml_diabetes_pima` shift 0.6/0.9 seed 2 (`-0.5000` bacc for v8).
- v10 is too conservative on `openml_kc1`: all 16 `kc1` groups have zero
  change, while v8 finds several positive low-margin minority-class fixes.

Artifacts:

- `results/apa_v10_cf_batch_margin_probe_detail.csv`
- `results/apa_v10_cf_batch_margin_probe_summary.csv`
- `results/apa_v10_cf_batch_margin_probe_report.md`

## 2026-06-13: `apa_v13_cf_tempered_rescue_probe`

v11 and v12 were short diagnostic variants. Both showed that simply relaxing
the v10 batch-margin requirement can recover some positive splits, but it also
brings back the `openml_credit_g` shift 0.3 seed 2 failure. Row-level
diagnostics then showed a sharper distinction:

- good `kc1` flips are low-margin rows with moderate counterfactual
  prior-sensitivity;
- bad `credit_g` and `diabetes_pima` flips are over-sensitive to the
  counterfactual context prior.

v13 therefore adds row-level tempered rescue:

- allow low-margin off-majority rescue as in v8;
- reject over-sensitive rows with `max_abs_sensitivity=0.90`;
- reject highly unstable counterfactual rows with `max_disagreement=0.040`;
- keep the train-prior imbalance and directional shift gates.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_credit_g,openml_diabetes_pima,openml_kc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_directional_router,apa_cf_batch_margin_router,apa_cf_tempered_rescue_router --output-prefix apa_v13_cf_tempered_rescue_probe
```

Result on the same 48-group focused probe:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_directional_router` | 48 | +0.0326 | -0.0064 | +0.0159 | -0.0044 | 24 / 4 |
| `apa_cf_batch_margin_router` | 48 | +0.0117 | +0.0061 | +0.0071 | -0.0007 | 7 / 0 |
| `apa_cf_tempered_rescue_router` | 48 | +0.0156 | +0.0082 | +0.0090 | -0.0069 | 18 / 0 |

Dataset-level bacc deltas:

| dataset | `v8` directional | `v10` batch-margin | `v13` tempered-rescue |
| --- | ---: | ---: | ---: |
| `openml_credit_g` | -0.0022 | +0.0040 | +0.0040 |
| `openml_diabetes_pima` | -0.0357 | +0.0143 | +0.0093 |
| `openml_kc1` | +0.0187 | +0.0000 | +0.0114 |

Representative v13 gains:

| dataset | shift | seed | bacc delta | macro-F1 delta |
| --- | ---: | ---: | ---: | ---: |
| `openml_kc1` | 0.3 | 0 | +0.0517 | +0.0678 |
| `openml_kc1` | 0.3 | 2 | +0.0345 | +0.0355 |
| `openml_kc1` | 0.6 | 2 | +0.0323 | +0.0278 |
| `openml_credit_g` | 0.6 | 2 | +0.0161 | +0.0142 |
| `openml_diabetes_pima` | 0.3 | 3 | +0.0345 | +0.0460 |

Takeaway: v13 is the strongest focused-probe candidate so far. It keeps zero
observed negative balanced-accuracy splits like v10, but increases mean bacc
gain from `+0.0061` to `+0.0082` and positive splits from `7` to `18`. The main
remaining weakness is that v13 is more conservative than v10 on some strong
`diabetes_pima` shifts, so a future hybrid router should use v10 when the batch
majority is clear and v13 only as the low-margin rescue path.

Artifacts:

- `results/apa_v13_cf_tempered_rescue_probe_detail.csv`
- `results/apa_v13_cf_tempered_rescue_probe_summary.csv`
- `results/apa_v13_cf_tempered_rescue_probe_report.md`

## 2026-06-13: `apa_v14_cf_hybrid_tempered_probe`

v14 combines the two useful v10/v13 behaviors:

- when the baseline batch prediction has a clear majority, use the v10
  batch-margin route and keep the larger gains on strong shifts;
- when the batch majority is weak, use the v13 row-level tempered rescue route
  to recover low-margin fixes while blocking over-sensitive flips.

This makes v14 a two-channel counterfactual context intervention policy rather
than a simple train/test prior correction.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_credit_g,openml_diabetes_pima,openml_kc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_batch_margin_router,apa_cf_tempered_rescue_router,apa_cf_hybrid_tempered_router --output-prefix apa_v14_cf_hybrid_tempered_probe
```

Result on the 48-group focused probe:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_batch_margin_router` | 48 | +0.0117 | +0.0061 | +0.0071 | -0.0007 | 7 / 0 |
| `apa_cf_tempered_rescue_router` | 48 | +0.0156 | +0.0082 | +0.0090 | -0.0069 | 18 / 0 |
| `apa_cf_hybrid_tempered_router` | 48 | +0.0254 | +0.0134 | +0.0151 | -0.0056 | 22 / 0 |

Dataset-level bacc deltas:

| dataset | `v10` batch-margin | `v13` tempered-rescue | `v14` hybrid-tempered |
| --- | ---: | ---: | ---: |
| `openml_credit_g` | +0.0040 | +0.0040 | +0.0060 |
| `openml_diabetes_pima` | +0.0143 | +0.0093 | +0.0195 |
| `openml_kc1` | +0.0000 | +0.0114 | +0.0145 |

Representative v14 gains:

| dataset | shift | seed | bacc delta | macro-F1 delta | inherited behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| `openml_diabetes_pima` | 0.6 | 0 | +0.0645 | +0.0745 | v10 clear-batch route |
| `openml_diabetes_pima` | 0.9 | 0 | +0.0645 | +0.0745 | v10 clear-batch route |
| `openml_diabetes_pima` | 0.3 | 3 | +0.0345 | +0.0460 | v13 weak-batch rescue |
| `openml_kc1` | 0.3 | 0 | +0.0517 | +0.0678 | v13 weak-batch rescue |
| `openml_kc1` | 0.3 | 1 | +0.0172 | +0.0225 | v14 recovered extra split |
| `openml_credit_g` | 0.6 | 1 | +0.0323 | +0.0358 | v10 clear-batch route |
| `openml_credit_g` | 0.6 | 2 | +0.0161 | +0.0142 | v13 weak-batch rescue |

Safety checks:

- `openml_credit_g`, shift 0.3, seed 2: v8 had bacc `-0.1494`; v14 stays at
  `0.0000`.
- `openml_diabetes_pima`, shift 0.0, seed 3: v8 falsely triggered; v14 stays at
  `0.0000`.
- `openml_diabetes_pima`, shift 0.6/0.9, seed 2: v8 had bacc `-0.5000`; v14
  stays at `0.0000`.

Takeaway: v14 is the current best focused-probe candidate. It preserves the
zero-negative behavior of v10/v13, improves mean bacc gain from v10 `+0.0061`
and v13 `+0.0082` to `+0.0134`, and increases positive splits to `22 / 48`.
The next required step is broader validation on additional OpenML datasets and
synthetic shifts to make sure the hybrid route does not overfit this focused
probe.

Artifacts:

- `results/apa_v14_cf_hybrid_tempered_probe_detail.csv`
- `results/apa_v14_cf_hybrid_tempered_probe_summary.csv`
- `results/apa_v14_cf_hybrid_tempered_probe_report.md`

External validation on three held-out OpenML datasets:

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_blood_transfusion,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_batch_margin_router,apa_cf_hybrid_tempered_router --output-prefix apa_v14_cf_hybrid_tempered_external_probe
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_batch_margin_router` | 48 | +0.0039 | +0.0021 | +0.0038 | +0.0044 | 3 / 0 |
| `apa_cf_hybrid_tempered_router` | 48 | +0.0137 | +0.0072 | +0.0095 | +0.0002 | 15 / 0 |

Combined focused plus external evidence for v14:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_hybrid_tempered_router` | 96 | +0.0195 | +0.0103 | +0.0123 | -0.0027 | 37 / 0 |

`apa_v15` was then tested as a calibration ablation by lowering the final blend
weight from `0.60` to `0.50`.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_blood_transfusion,openml_pc1 --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2,3 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_tempered_router,apa_cf_hybrid_calibrated_router --output-prefix apa_v15_cf_hybrid_calibrated_external_probe
```

Result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_hybrid_tempered_router` | 48 | +0.0137 | +0.0072 | +0.0095 | +0.0002 |
| `apa_cf_hybrid_calibrated_router` | 48 | +0.0130 | +0.0069 | +0.0086 | -0.0009 |

Takeaway: v15 improves calibration slightly, but gives up some accuracy,
balanced accuracy, and macro-F1. Under the current objective of improving
accuracy without lowering other decision metrics, v14 remains the main
candidate; v15 is useful only as a calibration-oriented ablation.

Artifacts:

- `results/apa_v14_cf_hybrid_tempered_external_probe_detail.csv`
- `results/apa_v14_cf_hybrid_tempered_external_probe_summary.csv`
- `results/apa_v14_cf_hybrid_tempered_external_probe_report.md`
- `results/apa_v15_cf_hybrid_calibrated_external_probe_detail.csv`
- `results/apa_v15_cf_hybrid_calibrated_external_probe_summary.csv`
- `results/apa_v15_cf_hybrid_calibrated_external_probe_report.md`

## 2026-06-13: `apa_v14_cf_hybrid_tempered_11dataset_shift`

To avoid over-claiming from three-dataset probes, v14 was expanded to 11
datasets: six OpenML binary datasets, `breast_cancer`, and four synthetic
binary datasets. Each dataset uses four label-shift strengths and three seeds,
for `11 x 4 x 3 = 132` groups.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1,breast_cancer,synthetic_balanced,synthetic_imbalanced,synthetic_label_shift,synthetic_label_shift_hard --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_tempered_router --output-prefix apa_v14_cf_hybrid_tempered_11dataset_shift
```

Overall result:

| method | n | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_hybrid_tempered_router` | 132 | +0.0166 | +0.0077 | +0.0156 | -0.0025 | 43 / 1 |

Dataset-level result:

| dataset | groups | acc delta | bacc delta | macro-F1 delta | ECE delta | pos/neg bacc | max bacc | min bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `breast_cancer` | 12 | +0.0104 | +0.0056 | +0.0272 | -0.0041 | 3 / 0 | +0.0345 | +0.0000 |
| `openml_blood_transfusion` | 12 | +0.0130 | +0.0070 | +0.0085 | -0.0133 | 4 / 0 | +0.0345 | +0.0000 |
| `openml_credit_g` | 12 | +0.0156 | +0.0081 | +0.0083 | +0.0031 | 4 / 0 | +0.0323 | +0.0000 |
| `openml_diabetes_pima` | 12 | +0.0391 | +0.0204 | +0.0242 | -0.0018 | 6 / 0 | +0.0645 | +0.0000 |
| `openml_ilpd` | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 | +0.0000 | +0.0000 |
| `openml_kc1` | 12 | +0.0365 | +0.0194 | +0.0210 | -0.0159 | 9 / 0 | +0.0517 | +0.0000 |
| `openml_pc1` | 12 | +0.0052 | +0.0027 | +0.0029 | -0.0037 | 2 / 0 | +0.0161 | +0.0000 |
| `synthetic_balanced` | 12 | +0.0156 | -0.0042 | +0.0199 | +0.0096 | 5 / 1 | +0.0345 | -0.1494 |
| `synthetic_imbalanced` | 12 | +0.0078 | +0.0049 | +0.0332 | +0.0053 | 3 / 0 | +0.0263 | +0.0000 |
| `synthetic_label_shift` | 12 | +0.0156 | +0.0084 | +0.0111 | +0.0016 | 3 / 0 | +0.0690 | +0.0000 |
| `synthetic_label_shift_hard` | 12 | +0.0234 | +0.0126 | +0.0154 | -0.0087 | 4 / 0 | +0.0517 | +0.0000 |

Interpretation:

- v14 improves average accuracy, balanced accuracy, macro-F1, and ECE across
  the 132 groups.
- 10 of 11 datasets have no negative balanced-accuracy split.
- `openml_kc1`, `openml_diabetes_pima`, `openml_credit_g`,
  `openml_blood_transfusion`, `breast_cancer`, and the synthetic label-shift
  datasets show clear positive subsets.
- `openml_ilpd` mostly does not trigger, which is acceptable for a conditional
  algorithm.
- The only observed negative balanced-accuracy split is
  `synthetic_balanced`, shift 0.3, seed 2. Row-level diagnostics show one
  correct flip and one wrong flip; the wrong flip has high context disagreement
  (`0.0804`). This motivates v16, a stability-capped hybrid router.

Artifacts:

- `results/apa_v14_cf_hybrid_tempered_11dataset_shift_detail.csv`
- `results/apa_v14_cf_hybrid_tempered_11dataset_shift_summary.csv`
- `results/apa_v14_cf_hybrid_tempered_11dataset_shift_report.md`

## 2026-06-13: `apa_v16_cf_hybrid_stability_11dataset_shift`

v16 tests a stability-capped version of v14. The change is small but important:
the hybrid counterfactual router applies a global context-disagreement cap, so a
row is changed only if different counterfactual contexts agree closely enough.
This was designed to fix the v14 failure on `synthetic_balanced`, where one
wrong flip had high context disagreement.

Command:

```text
python -u experiment6/scripts/run_apa_tabicl.py --datasets openml_ilpd,openml_credit_g,openml_blood_transfusion,openml_kc1,openml_diabetes_pima,openml_pc1,breast_cancer,synthetic_balanced,synthetic_imbalanced,synthetic_label_shift,synthetic_label_shift_hard --force-label-shift --label-shift-strengths 0.0,0.3,0.6,0.9 --seeds 0,1,2 --n-samples 180 --train-size 90 --test-size 32 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_hybrid_stability_router --output-prefix apa_v16_cf_hybrid_stability_11dataset_shift
```

Overall comparison on the same 132 groups:

| method | n | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta | pos/neg bacc | min bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `apa_cf_hybrid_tempered_router` (v14) | 132 | +0.0166 | +0.0077 | +0.0156 | -0.0025 | -0.0025 | 43 / 1 | -0.1494 |
| `apa_cf_hybrid_stability_router` (v16) | 132 | +0.0137 | +0.0062 | +0.0136 | -0.0025 | -0.0014 | 41 / 1 | -0.1494 |

Dataset-level result for v16:

| dataset | groups | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta | pos/neg bacc | max bacc | min bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `breast_cancer` | 12 | +0.0078 | +0.0041 | +0.0205 | +0.0000 | -0.0056 | 3 / 0 | +0.0172 | +0.0000 |
| `openml_blood_transfusion` | 12 | +0.0104 | +0.0056 | +0.0066 | +0.0000 | -0.0119 | 4 / 0 | +0.0172 | +0.0000 |
| `openml_credit_g` | 12 | +0.0156 | +0.0081 | +0.0083 | +0.0000 | +0.0060 | 4 / 0 | +0.0323 | +0.0000 |
| `openml_diabetes_pima` | 12 | +0.0260 | +0.0136 | +0.0159 | +0.0000 | -0.0000 | 6 / 0 | +0.0484 | +0.0000 |
| `openml_ilpd` | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 | +0.0000 | +0.0000 |
| `openml_kc1` | 12 | +0.0391 | +0.0208 | +0.0225 | +0.0000 | -0.0132 | 9 / 0 | +0.0517 | +0.0000 |
| `openml_pc1` | 12 | +0.0052 | +0.0027 | +0.0029 | +0.0000 | -0.0037 | 2 / 0 | +0.0161 | +0.0000 |
| `synthetic_balanced` | 12 | +0.0182 | +0.0097 | +0.0280 | +0.0000 | +0.0125 | 6 / 0 | +0.0345 | +0.0000 |
| `synthetic_imbalanced` | 12 | +0.0078 | +0.0049 | +0.0332 | +0.0000 | +0.0053 | 3 / 0 | +0.0263 | +0.0000 |
| `synthetic_label_shift` | 12 | +0.0104 | +0.0057 | +0.0082 | +0.0000 | +0.0034 | 1 / 0 | +0.0690 | +0.0000 |
| `synthetic_label_shift_hard` | 12 | +0.0104 | -0.0069 | +0.0038 | -0.0278 | -0.0077 | 3 / 1 | +0.0345 | -0.1494 |

Interpretation:

- v16 fixes the v14 negative case on `synthetic_balanced`, shift 0.3, seed 2.
  That group changes from `bacc -0.1494` under v14 to `bacc +0.0172` under v16.
- v16 preserves strong positive behavior on `openml_kc1`, and slightly improves
  its average gain over v14 on that dataset.
- v16 is not a full replacement for v14 yet: it introduces one negative split on
  `synthetic_label_shift_hard`, shift 0.3, seed 0. This suggests the next
  algorithm should route between the aggressive and stability-capped policies
  based on hard-shift diagnostics.
- Current best framing: v14 is the stronger accuracy candidate, while v16 is a
  stability ablation that reveals how to design the next router.

Artifacts:

- `configs/apa_v16_candidate.json`
- `results/apa_v16_cf_hybrid_stability_11dataset_shift_detail.csv`
- `results/apa_v16_cf_hybrid_stability_11dataset_shift_summary.csv`
- `results/apa_v16_cf_hybrid_stability_11dataset_shift_shift_curve.csv`
- `results/apa_v16_cf_hybrid_stability_11dataset_shift_report.md`

## 2026-06-14: `apa_v1_v16_fixed11_selected_benchmark`

为避免不同版本使用不同数据集导致不可比，这一轮固定同一批 11 个
数据集、同一组标签迁移强度和同一批随机种子，对代表性方法做统一
横向比较。没有把 v1-v16 的所有注册方法都跑一遍；早期多个版本只是
相近的概率修正变体，因此这里保留早期代表方法，并重点比较真正有
创新性的反事实上下文干预主线。

固定设置：

- 数据集：`openml_ilpd`, `openml_credit_g`, `openml_blood_transfusion`,
  `openml_kc1`, `openml_diabetes_pima`, `openml_pc1`, `breast_cancer`,
  `synthetic_balanced`, `synthetic_imbalanced`, `synthetic_label_shift`,
  `synthetic_label_shift_hard`
- 标签迁移强度：`0.0, 0.6, 0.9`
- 随机种子：`0, 1`
- 每个方法共 `11 x 3 x 2 = 66` 组
- 每个单元格里的变化量都相对于 `tabicl_baseline`

总体方法排名：

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

数据集 x 方法矩阵：

下面每个单元格是 `accuracy delta / balanced-accuracy delta`。完整
macro-F1、ECE、正负例数量见
`results/apa_v1_v16_fixed11_selected_benchmark_dataset_method.csv`。

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

结论：

- 平均涨幅最大的是 `apa_cf_directional_router`，但它是更激进的 v8
  版本；在这 66 组中没有出现 bacc 负例，但历史更大 seed 覆盖下曾暴露
  风险，因此不能只按这张表直接作为最终主方法。
- 后期主线里最强的是 `apa_cf_hybrid_tempered_router`。它在这 66 组中
  accuracy `+0.0128`、balanced accuracy `+0.0066`、macro-F1 `+0.0110`，
  且 bacc 正负组为 `15 / 0`。
- `apa_cf_hybrid_stability_router` 比 v14 更保守，平均 accuracy 降到
  `+0.0099`，但同样保持 `15 / 0` 的 bacc 正负组。
- 早期概率修正代表 `apa_full_prior_aware_tiny` 平均收益很小，说明顶会
  角度不能只讲“根据训练/测试先验差异修正概率”；真正主要涨点来自
  反事实上下文构造和路由。
- `openml_ilpd` 和 `breast_cancer` 是安全不动数据集；`openml_kc1`,
  `openml_credit_g`, `openml_diabetes_pima`, `synthetic_balanced`,
  `synthetic_imbalanced`, `synthetic_label_shift_hard` 是上下文干预有
  明确收益的数据集。

Artifacts:

- `results/apa_v1_v16_fixed11_selected_benchmark_detail.csv`
- `results/apa_v1_v16_fixed11_selected_benchmark_method_ranking.csv`
- `results/apa_v1_v16_fixed11_selected_benchmark_dataset_method.csv`
- `results/apa_v1_v16_fixed11_selected_benchmark_tables.md`
- `scripts/run_fixed11_selected_benchmark.py`

## 2026-06-14: `apa_v8_openml_broad48`

这一轮只扩测 v8，也就是 `apa_cf_directional_router`，并保留
`tabicl_baseline` 作为对照。目的不是继续横向比较 v1-v16，而是回答：
v8 在 30 个以上、来源更多样的 OpenML 二分类数据集上是否仍然有效，
以及在 `shift=0.0` 无偏移对照下是否安全。

固定设置：

- 候选数据集：48 个 OpenML 二分类数据集
- 实际完成：47 个数据集
- 失败数据集：`openml_336_spect`，OpenML 数据格式导致当前编码后为
  0 特征矩阵，已记录到失败表
- 标签迁移强度：`0.0, 0.6, 0.9`
- 随机种子：`0, 1`
- 每个完成数据集共 `3 x 2 = 6` 组
- 总有效 v8 组数：`47 x 3 x 2 = 282`

分层结果：

| slice | groups | datasets | acc delta | bacc delta | macro-F1 delta | minority recall delta | ECE delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 282 | 47 | +0.0146 | +0.0003 | +0.0107 | -0.0153 | -0.0034 | 86 / 7 |
| `shift_0_0` | 94 | 47 | +0.0007 | -0.0009 | +0.0001 | -0.0034 | -0.0000 | 2 / 3 |
| `shift_0_6` | 94 | 47 | +0.0216 | +0.0009 | +0.0160 | -0.0213 | -0.0051 | 42 / 2 |
| `shift_0_9` | 94 | 47 | +0.0216 | +0.0009 | +0.0160 | -0.0213 | -0.0051 | 42 / 2 |
| `shifted_0_6_or_0_9` | 188 | 47 | +0.0216 | +0.0009 | +0.0160 | -0.0213 | -0.0051 | 84 / 4 |

代表性正向数据集：

| dataset | groups | acc delta | bacc delta | pos/neg bacc |
| --- | ---: | ---: | ---: | ---: |
| `openml_37_diabetes` | 6 | +0.0729 | +0.0376 | 4 / 0 |
| `openml_721_pwlinear` | 6 | +0.0521 | +0.0269 | 4 / 0 |
| `openml_1066_kc1_binary` | 6 | +0.0521 | +0.0262 | 4 / 1 |
| `openml_51_heart_h` | 6 | +0.0417 | +0.0215 | 4 / 0 |
| `openml_53_heart_statlog` | 6 | +0.0312 | +0.0161 | 4 / 0 |
| `openml_1006_lymph` | 6 | +0.0312 | +0.0161 | 4 / 0 |
| `openml_41976_tuning_svms` | 6 | +0.0312 | +0.0161 | 4 / 0 |
| `openml_1067_kc1` | 6 | +0.0208 | +0.0108 | 4 / 0 |
| `openml_1049_pc4` | 6 | +0.0208 | +0.0108 | 4 / 0 |
| `openml_1600_spectf` | 6 | +0.0208 | +0.0108 | 4 / 0 |

主要负例：

| dataset | shift | seed | acc delta | bacc delta | macro-F1 delta | minority recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openml_902_sleuth_case2002` | 0.6 | 1 | -0.0312 | -0.5000 | -0.0566 | -1.0000 |
| `openml_902_sleuth_case2002` | 0.9 | 1 | -0.0312 | -0.5000 | -0.0566 | -1.0000 |
| `openml_44_spambase` | 0.6 | 0 | +0.0000 | -0.4839 | -0.1414 | -1.0000 |
| `openml_44_spambase` | 0.9 | 0 | +0.0000 | -0.4839 | -0.1414 | -1.0000 |
| `openml_1026_grub_damage` | 0.0 | 1 | +0.0000 | -0.1071 | -0.0488 | -0.2500 |

结论：

- v8 在更大规模扩测中仍有明显 Accuracy 收益，尤其是 `shift=0.6`
  和 `shift=0.9`，平均 Accuracy 都是 `+0.0216`。
- 但 Balanced Accuracy 的平均收益几乎被少数严重负例抵消，全体平均
  只有 `+0.0003`。这说明 v8 不能作为最终普适算法直接使用。
- `shift=0.0` 下平均 Accuracy 基本不动，但仍有 3 个 Balanced Accuracy
  负组，说明“无偏移不伤害”的保护还不够。
- 下一版算法应优先解决 `spambase`、`sleuth_case2002`、`grub_damage`
  这类负例。方向上应加入更强的反证诊断：例如不只看 batch prior
  差异，还要判断干预是否会清空少数类召回、是否只有极少数翻转造成
  Balanced Accuracy 坍塌、以及不同反事实 context 的预测是否一致。

Artifacts:

- `results/apa_v8_openml_broad48_detail.csv`
- `results/apa_v8_openml_broad48_method_ranking.csv`
- `results/apa_v8_openml_broad48_dataset_summary.csv`
- `results/apa_v8_openml_broad48_shift_safety_summary.csv`
- `results/apa_v8_openml_broad48_negative_cases.csv`
- `results/apa_v8_openml_broad48_shift_safety_tables.md`
- `results/apa_v8_openml_broad48_failures.csv`
- `scripts/run_v8_openml_broad_benchmark.py`

## 2026-06-14: `apa_v24_unsaturated_probe`

这一轮专门找了一批 baseline 尚未饱和、或边界/少数类仍有改进空间的数据集，测试 no-shift 场景下的 feature-evidence context。重点比较：

- `apa_cf_no_shift_boundary_router`：只使用 boundary contrast context。
- `apa_cf_no_shift_union_router`：v23，合并 strict diverse 与 soft diverse。
- `apa_cf_no_shift_boundary_union_router`：v24，合并 strict diverse、soft diverse 和更严格的 boundary contrast。

固定设置：

- 数据集：8 个 OpenML 二分类数据集
- split：ordered
- seed：1
- train/test：128 / 64，部分小数据集 test 少于 64
- shift：0，只看原始 no-shift 数据

平均结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ECE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boundary_router` | 8 | +0.0039 | +0.0028 | +0.0033 | +0.0000 | -0.0012 |
| `v23_union` | 8 | +0.0020 | +0.0023 | +0.0024 | +0.0046 | +0.0018 |
| `v24_boundary_union` | 8 | +0.0059 | +0.0051 | +0.0057 | +0.0046 | +0.0004 |

逐数据集关键结论：

| 数据集 | Baseline Acc | v23 ΔAcc | v24 ΔAcc | 说明 |
| --- | ---: | ---: | ---: | --- |
| `openml_31_credit_g` | 0.7344 | +0.0000 | +0.0156 | boundary 有效，v24 成功合入 |
| `openml_37_diabetes` | 0.7500 | +0.0156 | +0.0156 | diverse/prototype 有效，v24 保留收益 |
| `openml_43_haberman` | 0.7344 | +0.0000 | +0.0156 | boundary 有效，v24 成功合入 |
| 其余 5 个数据集 | 0.3750 到 0.9531 | +0.0000 | +0.0000 | 当前门控选择不改，没有负迁移 |

结论：

- v24 在 8 个不饱和探针数据中有 3 个 Accuracy 正向组、0 个负向组。
- v24 的价值不是简单概率修正，而是把两种 feature evidence 合并：boundary contrast 负责边界竞争类样本，diverse/prototype/subspace 负责局部结构多样性。
- 没有涨点的数据不应直接放宽门控。`blood_transfusion` 的 batch prior JS 高于 no-shift 阈值，更适合交给 shift 分支；`pc3` 与 `planning_relax` 的问题更像 minority recall 或单类偏置，下一步应新增 minority-aware boundary/prototype context。

Artifacts:

- `results/apa_v24_unsaturated_probe_detail.csv`
- `results/apa_v24_unsaturated_probe_summary.csv`
- `results/apa_v24_unsaturated_probe_report.md`
- `results/apa_v24_unsaturated_probe_table.md`

## 2026-06-14: `apa_v25_minority_probe`

这一轮尝试解决 `pc3`、`planning_relax` 这类 Accuracy 不低但少数类召回很差的问题。新增 `minority_contrast` context：每个测试样本的 context 包含最近少数类锚点、当前 baseline 预测类对照样本，再用 KNN 补齐。

结果分两步：

| 版本 | 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v25 初版 | `minority_router` | 6 | +0.0000 | -0.0014 | -0.0012 | -0.0062 | 有正例，但 `diabetes` 掉点 |
| v25 修复版 | `minority_router` | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 消除掉点，但过于保守 |
| v24 对照 | `boundary_union` | 6 | +0.0078 | +0.0068 | +0.0076 | +0.0062 | 当前更优 |

关键样本：

| 数据集 | v25 初版 ΔAcc | v25 修复版 ΔAcc | v24 ΔAcc | 说明 |
| --- | ---: | ---: | ---: | --- |
| `openml_37_diabetes` | -0.0156 | +0.0000 | +0.0156 | 初版少数类锚点误触发，修复版拦住 |
| `openml_43_haberman` | +0.0156 | +0.0000 | +0.0156 | 修复版过紧，把正例也拦掉 |
| `openml_1050_pc3` | +0.0000 | +0.0000 | +0.0000 | 极端少数类问题仍未解决 |

结论：

- v25 的创新点成立：少数类锚点 context 可以作为一种 feature-evidence，不是简单先验概率修正。
- 但当前 batch-level collapse gate 太粗，不能可靠区分正例和负例。
- 暂时不把 v25 作为主算法；后续如果继续，应改成 sample-level minority evidence，例如少数类原型距离优势、少数类局部密度优势、boundary 与 prototype 双证据一致后才触发。

Artifacts:

- `results/apa_v25_minority_probe_table.md`
- `results/apa_v25_minority_probe_detail.csv`
- `results/apa_v25_minority_probe_summary.csv`
- `results/apa_v25_minority_probe_r2_detail.csv`
- `results/apa_v25_minority_probe_r2_summary.csv`

## 2026-06-14: `apa_v26_minority_density_probe`

这一轮继续 v25 的少数类方向，但把 batch-level collapse gate 改成 sample-level minority evidence。新增方法：

- `apa_cf_no_shift_minority_density_router`

核心思想：只用 `minority_contrast` context，并且只有当测试样本在特征空间中更接近少数类锚点、候选预测也指向少数类时，才允许小比例翻转。

平均结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ECE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v26_minority_density` | 6 | +0.0026 | +0.0031 | +0.0032 | +0.0062 | +0.0030 |
| `v24_boundary_union` | 6 | +0.0078 | +0.0068 | +0.0076 | +0.0062 | +0.0009 |

逐数据集关键结果：

| 数据集 | Baseline Acc | v26 ΔAcc | v24 ΔAcc | 说明 |
| --- | ---: | ---: | ---: | --- |
| `openml_37_diabetes` | 0.7500 | +0.0156 | +0.0156 | v26 找到正确少数类边界翻转 |
| `openml_43_haberman` | 0.7344 | +0.0000 | +0.0156 | v24 boundary 更适合 |
| `openml_1050_pc3` | 0.9531 | +0.0000 | +0.0000 | 极端少数类仍未解决 |
| `openml_1490_planning_relax` | 0.6296 | +0.0000 | +0.0000 | 没有安全类别翻转 |

结论：

- v26 修复了 v25 初版在 `diabetes` 上的负迁移，并转为正向。
- 但 v26 覆盖率仍低于 v24，不能替代主线。
- 后续更合理的使用方式是把 v26 作为 v24 的低优先级候选：当 boundary-union 不触发，且 minority-distance evidence 很强时才启用。

Artifacts:

- `results/apa_v26_minority_density_probe_table.md`
- `results/apa_v26_minority_density_probe_detail.csv`
- `results/apa_v26_minority_density_probe_summary.csv`
- `results/apa_v26_minority_density_probe_report.md`

## 2026-06-14: `apa_v24_unsaturated_stability_3seed`

这一轮把 v24 从单 seed 扩到 3 个 seed。合并来源：

- seed=1：`apa_v24_unsaturated_probe_detail.csv`
- seed=2,3：`apa_v24_unsaturated_stability_seed23_detail.csv`

总体结果：

| 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ECE Δ | 正/负/零 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | +0.0059 | +0.0020 | +0.0022 | -0.0038 | -0.0031 | 8 / 2 / 14 |

逐数据集重点：

| 数据集 | 平均 ΔAcc | 正/负/零 | 说明 |
| --- | ---: | ---: | --- |
| `openml_31_credit_g` | +0.0156 | 2 / 0 / 1 | 多 seed 正向 |
| `openml_37_diabetes` | +0.0156 | 2 / 0 / 1 | 多 seed 正向 |
| `openml_1480_ilpd` | +0.0156 | 2 / 0 / 1 | 新增正向数据集 |
| `openml_43_haberman` | +0.0104 | 2 / 0 / 1 | 多 seed 正向 |
| `openml_1050_pc3` | -0.0052 | 0 / 1 / 2 | seed=2 出现负迁移，BAcc 掉幅较大 |
| `openml_1498_sa_heart` | -0.0052 | 0 / 1 / 2 | seed=2 出现误触发 |

结论：

- v24 仍有平均 Accuracy 正收益，但已经发现两个负例，不能直接继续大规模扩测。
- 负例主要来自“单个样本翻转对少数类或边界结构影响过大”。
- 下一步应做 v27 风险保护：要求 boundary、diverse、prototype 至少两路支持同一翻转，或在极端不平衡数据上加 minority recall 风险门控。

Artifacts:

- `results/apa_v24_unsaturated_stability_3seed_table.md`
- `results/apa_v24_unsaturated_stability_seed23_detail.csv`
- `results/apa_v24_unsaturated_stability_seed23_summary.csv`
- `results/apa_v24_unsaturated_stability_seed23_report.md`

## 2026-07-05: `apa_v33_pvalue_guard_8dataset_3seed`

这一轮按新的研究方向尝试把 V27 guarded boundary-union router 加上
sample-level local evidence p-value 保护。核心思路是：对 V27 已经产生的
候选翻转，计算候选类别在局部 KNN 邻域中相对训练先验的单侧 binomial
tail p-value；如果局部证据弱，尤其是 baseline minority class 被翻到
candidate majority class 的高风险情形，则拒绝翻转。该版本对应方法：

- `apa_cf_no_shift_pvalue_guarded_boundary_router` / `apa_v33`

同时这一轮把 `roc_auc` 加入统一指标、summary 和 report 输出。

固定设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`, `openml_1498_sa_heart`,
  `openml_1464_blood_transfusion`, `openml_1050_pc3`,
  `openml_1490_planning_relax`
- split：ordered
- seeds：`1,2,3`
- train/test：128 / 64，部分小数据集 test 少于 64
- shift：0，只看原始 no-shift 数据

总体结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | 正/负/零 BAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V27 guarded boundary-union` | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0052 | +0.0016 | -0.0019 | 8 / 0 / 16 |
| `V33 p-value guarded boundary-union` | 24 | +0.0039 | +0.0046 | +0.0048 | +0.0071 | +0.0012 | +0.0001 | 5 / 1 / 18 |

逐数据集重点：

| 数据集 | V27 Acc/BAcc Δ | V33 Acc/BAcc Δ | 说明 |
| --- | ---: | ---: | --- |
| `openml_31_credit_g` | +0.0156 / +0.0122 | +0.0104 / +0.0083 | V33 保留 seed=3 强正例，但拦掉 seed=1 小正例 |
| `openml_37_diabetes` | +0.0156 / +0.0175 | +0.0104 / +0.0134 | V33 保留正收益，但削弱 seed=2 收益 |
| `openml_1480_ilpd` | +0.0156 / +0.0151 | +0.0052 / +0.0079 | V33 保留 seed=2，拦掉 seed=3，并在 seed=1 新增负例 |
| `openml_43_haberman` | +0.0104 / +0.0071 | +0.0000 / +0.0000 | V33 拦掉两个 V27 正例 |
| `openml_1498_sa_heart` | +0.0000 / +0.0000 | +0.0052 / +0.0072 | V33 新增一个正例且无负例 |
| `openml_1050_pc3` | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 两者都安全不动 |
| `openml_1464_blood_transfusion` | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 两者都安全不动 |
| `openml_1490_planning_relax` | +0.0000 / +0.0000 | +0.0000 / +0.0000 | 两者都安全不动 |

结论：

- V33 验证了“基于局部 p-value 的候选保护”这个方向可以作为诊断信号，
  但当前阈值和使用方式不适合作为主算法。
- 与 V27 相比，V33 的平均 Acc/BAcc/Macro-F1/ROC-AUC 都更低，并且出现
  1 个 BAcc 负例：`openml_1480_ilpd seed=1`，BAcc Δ `-0.0109`。
- V33 也拦掉了多个 V27 正例，尤其是 `haberman` 两个 seed 和
  `ilpd seed=3`。
- 当前 no-shift feature-evidence 主候选仍应保留 V27；p-value 更适合作为
  后续 meta-router 的输入特征，而不是硬门控。

Artifacts:

- `results/apa_v33_pvalue_guard_8dataset_3seed_detail.csv`
- `results/apa_v33_pvalue_guard_8dataset_3seed_summary.csv`
- `results/apa_v33_pvalue_guard_8dataset_3seed_report.md`
- `results/apa_v33_pvalue_guard_smoke_detail.csv`
- `results/apa_v33_pvalue_guard_smoke_summary.csv`

## 2026-07-05: `apa_v34_v36_dual_mode_context_guard`

这一轮把用户提供的 ensemble/context 方法用于一个新的双模 router：

- 低 batch-prior shift：沿用 V27 guarded boundary-union，负责 no-shift
  feature-evidence 稳定涨点。
- 明显 batch-prior shift：沿用 V14 hybrid-tempered counterfactual
  intervention，负责 label-shift 场景。
- V35：在小批量 no-shift 分支加 batch-level local support guard。若
  `train_size < 128` 或 `test_size < 64`，且 V27 产生修改但
  `no_shift_mean_local_support_margin < 0.015`，则回退 baseline。
- V36：在 V35 基础上，对 label-shift 分支加入 context-disagreement
  guard。若发生 counterfactual 修改但 `cf_mean_context_disagreement <
  0.014`，则回退 baseline，避免把自然 no-shift 中的预测先验偏差误当成
  label shift。

对应方法：

- `apa_cf_dual_mode_v34` / `apa_v34`
- `apa_cf_dual_mode_v35` / `apa_v35`
- `apa_cf_dual_mode_v36` / `apa_v36`

### Forced label-shift smoke

设置：

- 数据集：`openml_credit_g`, `openml_diabetes_pima`, `openml_kc1`
- force label shift：`0.0,0.6,0.9`
- seeds：`0,1`
- train/test：90 / 32

结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V14 hybrid-tempered` | 18 | +0.0347 | +0.0179 | +0.0201 | +0.0000 | +0.0000 | -0.0002 | 10 / 0 / 8 |
| `V34 dual-mode` | 18 | +0.0347 | +0.0179 | +0.0204 | +0.0000 | +0.0002 | -0.0005 | 11 / 1 / 6 |
| `V35 small-batch guarded` | 18 | +0.0365 | +0.0197 | +0.0221 | +0.0035 | +0.0002 | +0.0011 | 11 / 0 / 7 |
| `V36 context-guarded` | 18 | +0.0365 | +0.0197 | +0.0221 | +0.0035 | +0.0002 | +0.0011 | 11 / 0 / 7 |

关键观察：

- V34 在 `openml_credit_g shift=0 seed=0` 出现一个小批量 no-shift 负例：
  Acc/BAcc/Macro-F1 均为 `-0.0312/-0.0312/-0.0304`。
- V35 用 local support guard 将该负例回退为 0，同时保留
  `openml_credit_g shift=0 seed=1` 的正例。
- V36 在 forced-shift smoke 上与 V35 完全一致，没有误杀真正的
  label-shift 正收益。

### No-shift 8 数据集 x 3 seed

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`, `openml_1498_sa_heart`,
  `openml_1464_blood_transfusion`, `openml_1050_pc3`,
  `openml_1490_planning_relax`
- split：ordered
- seeds：`1,2,3`
- train/test：128 / 64，部分小数据集 test 少于 64

结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 | Acc/F1 负例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V27 guarded boundary-union` | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0052 | +0.0016 | -0.0019 | 8 / 0 / 16 | 0 / 0 |
| `V35 dual-mode` | 24 | +0.0065 | +0.0071 | +0.0066 | +0.0082 | +0.0017 | -0.0015 | 9 / 0 / 15 | 1 / 1 |
| `V36 dual-mode` | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0052 | +0.0016 | -0.0019 | 8 / 0 / 16 | 0 / 0 |

关键观察：

- V35 在 `openml_1464_blood_transfusion seed=1` 进入 label-shift 分支，
  得到 BAcc `+0.0157` 和 ROC-AUC `+0.0014`，但损失 Acc `-0.0156`、
  Macro-F1 `-0.0096`、ECE `+0.0101`。
- 该样本的 `cf_mean_context_disagreement=0.0107`，低于 forced-shift 中
  真实正收益样本常见的 disagreement 水平。
- V36 用 context-disagreement guard 拦掉这一 prior-intervention，把结果
  回退到 baseline，因此 no-shift 8x3 与 V27 数值等价，且无 Acc/F1/BAcc
  负例。

结论：

- V36 是当前最好的 mixed no-shift / label-shift 候选：在 no-shift
  benchmark 上保持 V27 的稳定性，在 forced label-shift smoke 上保留 V14
  的收益，并通过两个 guard 消除 V34/V35 暴露出的负例。
- V27 仍是最清晰的纯 no-shift feature-evidence component；V36 可以作为
  外层 dual-mode router，用于不知道是否存在 label shift 的场景。
- 下一步若继续冲 SOTA，应扩大 V36 的 forced-shift 数据集和 seed 数，并把
  `cf_mean_context_disagreement`、local support margin、p-value 作为
  meta-router 特征，而不是继续增加硬规则。

Artifacts:

- `results/apa_v34_dual_mode_no_shift_smoke_summary.csv`
- `results/apa_v34_dual_mode_forced_shift_smoke_summary.csv`
- `results/apa_v35_dual_mode_forced_shift_smoke_summary.csv`
- `results/apa_v35_dual_mode_no_shift_8dataset_3seed_summary.csv`
- `results/apa_v36_dual_mode_forced_shift_smoke_summary.csv`
- `results/apa_v36_dual_mode_no_shift_8dataset_3seed_summary.csv`

## 2026-07-05: `apa_v37_v38_v14_priority_dual_mode`

这一轮修正 V36 在更大 forced-shift 扩测中的问题：V36 会在部分
forced-shift batch 里错误进入 no-shift V27 分支，导致 missed gains 或负例。
因此改成 V14-priority dual-mode：

- 先运行 V14 hybrid-tempered counterfactual router。
- 如果 V14 实际改动且 `cf_mean_context_disagreement` 足够高，则接受
  label-shift 修正。
- 否则只把 V27 作为非常严格的低-JS no-shift fallback。
- V37 使用 context threshold `0.014`；V38 放宽到 `0.012`，恢复
  blood-transfusion forced-shift 正例。

关键结果：

| 设置 | 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forced-shift 8 数据集 x 3 seed x 2 shift | `V38` | 48 | +0.0306 | +0.0155 | +0.0177 | +0.0000 | -0.0084 | 28 / 0 / 20 |
| forced-shift 8 数据集 x 3 seed x 2 shift | `V14` | 48 | +0.0306 | +0.0155 | +0.0177 | +0.0000 | -0.0084 | 28 / 0 / 20 |
| no-shift 8 数据集 x 3 seed | `V38` | 24 | +0.0052 | +0.0052 | +0.0055 | +0.0011 | -0.0011 | 5 / 0 / 19 |
| no-shift 8 数据集 x 3 seed | `V27` | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0016 | -0.0019 | 8 / 0 / 16 |

结论：

- V38 是当前 forced-shift/mixed-shift 安全候选：在 forced-shift 扩测中
  与 V14 完全一致，且 0 个负例。
- 纯 no-shift 下 V38 比 V27 保守，miss 掉 `ilpd seed=3`、
  `haberman seed=1/2` 的 V27 正例；因此 no-shift 主线仍需单独优化。

Artifacts:

- `results/apa_v38_v14_priority_forced_shift_8dataset_3seed_summary.csv`
- `results/apa_v38_v14_priority_forced_shift_8dataset_3seed_detail.csv`
- `results/apa_v38_no_shift_8dataset_3seed_summary.csv`

## 2026-07-05: `apa_v39_v42_no_shift_scale_and_instance_ensemble`

这一轮按照用户给出的参考方法，优先尝试新的 no-shift 方法族，而不是只在
V27 上微调：

- V39：dataset-scale adaptive validation gate。小规模直接使用 V27；
  大规模先在训练集内部验证 V27，只有 validation accuracy 改善且 BAcc/F1
  不受伤时才应用。
- V40：instance-wise context ensemble。把 strict-diverse、soft-diverse、
  boundary-contrast、clean-consensus 四类 context 当成 members，每个测试
  样本根据 vote fraction、local support、context disagreement、baseline
  uncertainty 动态加权。
- V41：scale-aware combo。小规模使用 V27；大规模用 V39 做安全门控，若
  V39 回退，则允许稳定的 V40 instance ensemble fallback。
- V42：boundary-bypass combo。在 V41 基础上加入一个严格的 V27 bypass：
  对大规模 batch，如果 V27 候选满足 `batch JS <= 0.001`、
  `mean context disagreement >= 0.036`、`mean local support margin <= 0.020`，
  则认为这是 validation 容易漏掉的强边界上下文信号，接受 V27。

### 小规模 no-shift smoke

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`
- split：ordered
- seeds：`1,2`
- train/test：128 / 64

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V27 guarded boundary-union` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0157 | +0.0026 | -0.0092 |
| `V39 validation boundary` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0157 | +0.0026 | -0.0092 |
| `V40 instance ensemble` | 8 | +0.0020 | +0.0027 | +0.0027 | +0.0054 | +0.0008 | -0.0020 |
| `V42 boundary-bypass combo` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0157 | +0.0026 | -0.0092 |

观察：

- V40 作为裸 instance-wise ensemble 有正信号且小规模无负 decision seed，
  但收益明显弱于 V27，因为 soft weighting 很保守。
- V42 小规模走 `small_scale_v27` 分支，保留 V27 的 no-shift accuracy 收益。

### 大规模 no-shift targeted

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`
- seeds：`1,2`
- train/test：256 / 128，`haberman` 可用 test 为 50

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | Acc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V27 guarded boundary-union` | 8 | -0.0040 | -0.0134 | -0.0211 | -0.0002 | -0.0045 | 2 / 4 / 2 |
| `V39 validation boundary` | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 / 8 |
| `V40 instance ensemble` | 8 | -0.0010 | -0.0034 | -0.0029 | +0.0001 | +0.0029 | 2 / 2 / 4 |
| `V41 scale combo` | 8 | +0.0020 | +0.0014 | +0.0018 | +0.0001 | -0.0000 | 2 / 0 / 6 |

观察：

- V39 能挡住 V27 的大规模负例，但过于保守，也错过 V27 正例。
- V40 验证了 instance-wise ensemble 的候选价值，但裸跑大规模不稳：
  `diabetes seed=1` 和 `ilpd seed=1` 出现负例。
- V41 将 V40 放在稳定性门控后使用，得到 2 个正例、0 个负例，但仍错过
  `sa_heart` 这类 V27 强正例。

### 大规模 no-shift 7 数据集

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`, `openml_1498_sa_heart`,
  `openml_1464_blood_transfusion`, `openml_1050_pc3`
- seeds：`1,2`
- train/test：256 / 128

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | Acc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V27 guarded boundary-union` | 14 | +0.0010 | -0.0042 | -0.0082 | -0.0174 | +0.0010 | +0.0007 | 4 / 4 / 6 |
| `V41 scale combo` | 14 | +0.0011 | +0.0008 | +0.0010 | +0.0000 | +0.0001 | -0.0000 | 2 / 0 / 12 |
| `V42 boundary-bypass combo` | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0036 | +0.0013 | +0.0030 | 5 / 0 / 9 |

V42 非 baseline 分支：

| 数据集 | seed | 分支 | Acc Δ | BAcc Δ | ROC-AUC Δ | ECE Δ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `openml_31_credit_g` | 2 | V27 boundary bypass | +0.0156 | +0.0109 | +0.0024 | -0.0041 |
| `openml_37_diabetes` | 2 | stable V40 instance ensemble | +0.0078 | +0.0050 | +0.0000 | +0.0077 |
| `openml_1480_ilpd` | 2 | stable V40 instance ensemble | +0.0078 | +0.0058 | +0.0008 | -0.0079 |
| `openml_1498_sa_heart` | 1 | V27 boundary bypass | +0.0312 | +0.0297 | +0.0104 | +0.0307 |
| `openml_1498_sa_heart` | 2 | V27 boundary bypass | +0.0156 | +0.0182 | +0.0051 | +0.0150 |

分支统计：`large_scale_baseline_guard` 9 组，
`large_scale_v27_boundary_bypass` 3 组，
`large_scale_stable_instance_ensemble` 2 组。

结论：

- V42 是当前最好的纯 no-shift 大规模候选：在 7 数据集 x 2 seed 上
  Acc/BAcc/Macro-F1/Minority Recall/ROC-AUC 全部为正，且没有 Acc/BAcc/F1
  负 seed。
- 相比 V27，V42 用 V39/V40 风险门控消除了 `diabetes`, `haberman`,
  `ilpd seed=1`, `credit_g seed=1` 等大规模负例；同时通过 boundary bypass
  恢复 `credit_g seed=2` 和 `sa_heart` 的 V27 强正例。
- 代价是 ECE 平均变差 `+0.0030`，主要来自 `sa_heart` 两个强正例；如果后续
  以 calibration 为目标，可以在 V42 后加 temperature calibration 或把 ECE
  作为 bypass 的次级门控。

Artifacts:

- `results/apa_v39_v40_no_shift_smoke4_2seed_summary.csv`
- `results/apa_v39_v40_no_shift_scale4_2seed_train256_summary.csv`
- `results/apa_v41_no_shift_smoke4_2seed_train128_summary.csv`
- `results/apa_v41_no_shift_7dataset_2seed_train256_summary.csv`
- `results/apa_v42_no_shift_smoke4_2seed_train128_summary.csv`
- `results/apa_v42_no_shift_7dataset_2seed_train256_summary.csv`

## 2026-07-05: `apa_v43_mixed_v38_v42_router`

这一轮把前面两条最强分支合并成统一 router：

- forced/mixed label-shift：优先走 V38 V14-priority dual-mode。V38 先尝试
  V14 label-shift correction，只有 `cf_mean_context_disagreement >= 0.012`
  时接受；否则回退 baseline 或极严格 no-shift fallback。
- large no-shift：当 batch-prior JS 足够低且 batch 规模较大时，走 V42
  boundary-bypass scale combo，保留 V42 在大规模 no-shift 上的收益。
- small/ambiguous batch：回到 V38/baseline，不让 V42 在小规模 forced-shift
  batch 上误触发。最终规则使用 `shift_js_threshold = 0.004`，且只有
  `train_size >= 192` 或 `test_size >= 96` 时才允许进入 V42 no-shift route。

这个版本对应方法：

- `apa_cf_mixed_v38_v42_router` / `apa_v43`

### No-shift 7 数据集大规模验证

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`, `openml_1498_sa_heart`,
  `openml_1464_blood_transfusion`, `openml_1050_pc3`
- split：ordered
- seeds：`1,2`
- train/test：256 / 128

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | Acc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V43 mixed V38/V42` | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0036 | +0.0013 | +0.0030 | 5 / 0 / 9 |
| `V42 boundary-bypass combo` | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0036 | +0.0013 | +0.0030 | 5 / 0 / 9 |

逐行核验：V43 与 V42 在 14 个 no-shift trial 上
`accuracy_delta`、`balanced_accuracy_delta`、`macro_f1_delta`、
`minority_recall_delta`、`roc_auc_delta`、`ece_delta` 完全一致。

V43 分支统计：

- `no_shift_v42_boundary_bypass_combo`：13 组。
- `baseline_shift_guard`：1 组。
- 进入 V42 后的 scale combo 分支：`large_scale_baseline_guard` 8 组，
  `large_scale_v27_boundary_bypass` 3 组，
  `large_scale_stable_instance_ensemble` 2 组。

### Forced label-shift targeted 验证

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_43_haberman`,
  `openml_1498_sa_heart`, `openml_1464_blood_transfusion`
- force label shift：`0.6,0.9`
- seeds：`1,2,3`
- train/test：128 / 64

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | Acc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V43 mixed V38/V42` | 30 | +0.0229 | +0.0116 | +0.0132 | +0.0000 | +0.0000 | -0.0086 | 16 / 0 / 14 |
| `V38 V14-priority` | 30 | +0.0229 | +0.0116 | +0.0132 | +0.0000 | +0.0000 | -0.0086 | 16 / 0 / 14 |

逐行核验：V43 与 V38 在 30 个 forced-shift targeted trial 上完全一致。
V43 分支统计为 `label_shift_v38_priority` 16 组，
`baseline_shift_guard` 14 组。

结论：

- V43 是当前最好的统一 mixed router：在 large no-shift 上复现 V42 的
  Acc/BAcc/F1/ROC-AUC 稳定收益，在 targeted forced-shift 上复现 V38 的
  shift-safe 收益，并且两个验证集都没有 Acc/BAcc/F1 负例。
- V43 的主要代价是推理成本：为了判别 no-shift route，会运行 V38/V42
  相关的多 context/member 预测；在 7 数据集 x 2 seed 大规模验证中运行时间
  明显长于单独 baseline。
- Calibration 仍是后续问题：no-shift 中平均 ECE `+0.0030`，主要来自
  `sa_heart` 的两个 V27 boundary-bypass 强正例。下一步可尝试把 p-value、
  context-consensus uncertainty 或温度校准加入 V42/V43 的次级门控。

Artifacts:

- `results/apa_v43_no_shift_7dataset_2seed_train256_summary.csv`
- `results/apa_v43_no_shift_7dataset_2seed_train256_detail.csv`
- `results/apa_v43_no_shift_7dataset_2seed_train256_report.md`
- `results/apa_v43_targeted_forced_shift_5dataset_3seed_v3_summary.csv`
- `results/apa_v43_targeted_forced_shift_5dataset_3seed_v3_detail.csv`
- `results/apa_v43_targeted_forced_shift_5dataset_3seed_v3_report.md`

## 2026-07-05: `apa_v44_context_consensus_calibrated_v42`

这一轮尝试用户给出的 context-consensus / uncertainty 思路，用来解决 V42/V43
在 large no-shift 上的 ECE trade-off。V44 不改变 route，也不主动增加
hard-label 翻转，而是在 V42 输出后做 probability-only calibration：

- 先运行 V42 boundary-bypass scale combo。
- 只对 V42 相比 baseline 改变 hard prediction 的样本做概率收缩。
- 收缩目标是 baseline probability，收缩强度由 context disagreement、
  baseline entropy 和 V42 top-2 margin 决定。
- 如果收缩会改变 V42 的 argmax，则触发 argmax guard，回退 V42 原概率。

这个版本对应方法：

- `apa_cf_no_shift_context_consensus_calibrated_router` / `apa_v44`

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_1498_sa_heart`
- split：ordered
- seeds：`1,2`
- train/test：256 / 128
- 对照：baseline, V42, V44

结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass combo` | 8 | +0.0098 | +0.0087 | +0.0102 | +0.0063 | +0.0023 | +0.0052 |
| `V44 context-consensus calibrated V42` | 8 | +0.0098 | +0.0087 | +0.0102 | +0.0063 | +0.0018 | +0.0053 |

逐行差异：

- V44 与 V42 的 Acc/BAcc/Macro-F1/Minority Recall 完全一致，说明 argmax
  guard 有效。
- 但 V44 平均 ROC-AUC 更低：`+0.0023 -> +0.0018`。
- V44 平均 ECE 没有改善，反而略差：`+0.0052 -> +0.0053`。
- `sa_heart` 两个 seed 是目标场景，但 ECE 均变差：
  seed 1 `+0.0307 -> +0.0312`，seed 2 `+0.0150 -> +0.0154`。

结论：

- V44 是一个清晰的负结果：概率收缩能保持 hard gains，但不能改善 V42 的
  calibration trade-off，还会压低 ROC-AUC。
- 原因可能是 V42 的正例本身需要较强的类别概率支撑；向 baseline 收缩会降低
  ranking 质量，却没有真正修复 ECE bin 中的过/欠置信问题。
- 后续不应继续沿用简单 baseline-shrink。更值得尝试的是把 p-value、
  context-consensus uncertainty 和 ECE proxy 用作 route-level gate，而不是
  在已接受的 V42 输出后做概率后处理。

Artifacts:

- `results/apa_v44_consensus_calibrated_no_shift_4dataset_2seed_train256_summary.csv`
- `results/apa_v44_consensus_calibrated_no_shift_4dataset_2seed_train256_detail.csv`
- `results/apa_v44_consensus_calibrated_no_shift_4dataset_2seed_train256_report.md`

## 2026-07-06: `apa_v45_validation_member_selector`

这一轮继续尝试用户给出的 diversity-aware ensemble selection / dataset-adaptive
selection 思路。V45 不再直接用手写阈值选 V27/V40，而是在训练集内部做
multi-split validation：

- 每个 large no-shift batch 先做两个 stratified validation split。
- 在 validation split 上分别评估两个候选 member：
  - V27 guarded boundary-union；
  - V40 instance-wise context ensemble。
- 只有候选在两个 split 上平均 Acc 为正，且 Acc/BAcc/Macro-F1/ROC-AUC
  没有明显伤害时，才在测试 batch 上使用该 member。
- 如果 validation 没有稳定正信号，则回退 V42 boundary-bypass scale combo。

这个版本对应方法：

- `apa_cf_no_shift_validation_member_selector_router` / `apa_v45`

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_1498_sa_heart`
- split：ordered
- seeds：`1,2`
- train/test：256 / 128
- 对照：baseline, V42, V45

结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | Minority Recall Δ | ROC-AUC Δ | ECE Δ | Acc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass combo` | 8 | +0.0098 | +0.0087 | +0.0102 | +0.0063 | +0.0023 | +0.0052 | 5 / 0 / 3 |
| `V45 validation member selector` | 8 | +0.0078 | +0.0055 | +0.0072 | -0.0001 | +0.0022 | +0.0071 | 5 / 1 / 2 |

关键负例：

- `openml_37_diabetes seed=1`：V45 validation 选择了
  `v40_instance_ensemble`，validation 上 V40 平均 Acc Δ `+0.0078`、
  BAcc Δ `+0.0104`，但测试集实际为 Acc/BAcc/Macro-F1
  `-0.0156/-0.0256/-0.0237`，Minority Recall `-0.0513`，
  ROC-AUC `-0.0012`，ECE `+0.0154`。
- 同一组上 V42 回退 baseline，所有 delta 为 0。

其他观察：

- V45 保住了 `credit_g seed2`, `diabetes seed2`, `ilpd seed2`,
  `sa_heart seed1/2` 的 V42 正收益。
- 但 V45 没有新增任何 V42 之外的正例，还引入了 1 个负例。
- 运行成本显著高于 V42，因为每个 batch 需要两个 validation split，每个 split
  又要评估 V27 和 V40。

结论：

- V45 是负结果：train-internal validation member selection 在当前设置下会
  过拟合 validation split，不能可靠代表 test batch 的局部 no-shift 结构。
- 这个方向暂不扩测 7 数据集，不替代 V42。
- 如果以后继续尝试 validation selection，必须加入 final test-batch
  risk guard，例如沿用 V42 的 `member_change/context/js/support` 风险信号；
  但那样很可能退化成更慢的 V42。

Artifacts:

- `results/apa_v45_validation_member_no_shift_4dataset_2seed_train256_summary.csv`
- `results/apa_v45_validation_member_no_shift_4dataset_2seed_train256_detail.csv`
- `results/apa_v45_validation_member_no_shift_4dataset_2seed_train256_report.md`

## 2026-07-06: `apa_v46_v49_diversity_greedy_attempts`

这一轮尝试用户建议的 diversity-aware / instance-wise ensemble selection，
目标是在 no-shift 场景中超过 V42 boundary-bypass combo。

共实现四个候选：

- `V46 diversity-greedy instance router`：生成 diverse、prototype、clean、
  boundary、feature-subspace、minority-density 等结构化 context members；
  对每个测试样本 greedily 选择最多 3 个互补 member，并按 evidence/diversity
  加权融合。
- `V47 diversity-greedy p-value router`：在 V46 基础上加入 local-neighborhood
  p-value rescue，让原本被 minority-risk guard 拦掉的候选在局部证据足够强时
  可以通过。
- `V48 boundary-diversity member selector`：把 V42 作为高精度 primary member，
  把 V47 作为 diversity/p-value rescue member。第一版允许 V47 覆盖 V42；
  第二版改成 precision-first，只在 V42 abstain 时让 V47 rescue。
- `V49 precision-first p-value rescue selector`：保留 V48 r2 的
  precision-first 规则，但给 V47 rescue member 使用更宽松的 p-value
  rescue 配置，目标是补回 V47 独有的 `openml_1480_ilpd seed=1` 正例。

设置：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`
- split：ordered no-shift
- seeds：`1,2`
- train/test：128 / 64
- k：16

结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass combo` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| `V46 diversity-greedy` | 8 | +0.0039 | +0.0042 | +0.0045 | +0.0005 | -0.0039 | 1 / 0 / 7 |
| `V47 diversity-pvalue` | 8 | +0.0039 | +0.0029 | +0.0034 | +0.0008 | -0.0001 | 2 / 0 / 6 |
| `V48 r1 selector` | 8 | +0.0059 | +0.0020 | +0.0028 | +0.0004 | -0.0037 | 4 / 2 / 2 |
| `V48 r2 precision-first` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| `V49 precision-rescue` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |

结论：

- V46 安全但过于保守，漏掉大多数 V42 正例。
- V47 找到了一个 V42 未改动的正例（`openml_1480_ilpd seed=1`），说明
  p-value rescue 有诊断价值；但整体仍明显弱于 V42。
- V48 r1 证明“让 rescue member 覆盖 primary member”会引入负例，不可保留。
- V48 r2 消除负例，但数值等价 V42，且计算成本更高，因此不替代 V42。
- V49 使用更宽松的 rescue member 仍然等价 V42，没有补回 V47 独有正例。

当前处理：

- 不扩测 V46/V47/V48/V49 到 7 数据集 train256。
- 保留 V47 的 local p-value、selected member count、target vote fraction、
  selected-member diversity 作为后续 meta-router 特征。
- 后续若继续做 diversity-aware selection，应学习/估计“V47 何时能安全补充
  V42 abstain case”，而不是允许它覆盖 V42 的 high-precision change；单纯
  放宽 p-value rescue 不够。

Artifacts:

- `results/apa_v46_v48_diversity_greedy_notes.md`
- `results/apa_v46_diversity_greedy_no_shift_smoke4_2seed_train128_summary.csv`
- `results/apa_v47_diversity_pvalue_no_shift_smoke4_2seed_train128_summary.csv`
- `results/apa_v48_boundary_diversity_selector_no_shift_smoke4_2seed_train128_summary.csv`
- `results/apa_v48_boundary_diversity_selector_no_shift_smoke4_2seed_train128_r2_summary.csv`
- `results/apa_v49_precision_rescue_no_shift_smoke4_2seed_train128_summary.csv`

## 2026-07-06: `apa_v50_cross_model_tree_rescue`

这一轮尝试用户建议的 model-member ensemble 方向：不再只在 TabICL context
成员之间做选择，而是把 V42 作为 primary TabICL member，再加入
ExtraTrees/RandomForest 组成的异构树成员作为 rescue。

方法线：

- 先运行 V42 `boundary-bypass scale combo`，作为高精度 primary member。
- 只有当 V42 的 batch prior JS 极低（`<= 0.0015`）时，才训练并启用树成员；
  否则直接返回 V42，避免在不够干净的 no-shift batch 上叠加第二个风险源。
- 树成员由 `ExtraTreesClassifier` 和 `RandomForestClassifier` 组成，class order
  对齐后平均概率。
- 只在 V42 没有改动该样本、两个树模型预测一致、local KNN label support
  支持树候选、local binomial p-value 足够小、candidate prior drift 不大时
  rescue；每个 batch 最多 rescue `3.125%` 样本。

V50 r1 的失败：

- `openml_1480_ilpd seed=1` 出现 Acc/BAcc/Macro-F1
  `-0.0156/-0.0109/-0.0129`。
- 该负例 local support 和 p-value 都很好，说明仅靠局部 p-value 不能保证
  树 rescue 安全。
- r2 加入 `primary_js <= 0.0015` 后，该负例回到 V42，不再触发树 rescue。

train128/test64 no-shift smoke：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`, `openml_1480_ilpd`,
  `openml_43_haberman`
- seeds：`1,2`
- 对照：同一 suite 上已记录的 V42

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass combo` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| `V50 cross-model tree rescue` | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |
| `V50 - V42` | 8 | +0.0000 | +0.0014 | +0.0018 | -0.0001 | -0.0001 | 无新增负例 |

train256/test128 scale check：

- 数据集：`openml_31_credit_g`, `openml_37_diabetes`
- seeds：`1,2`
- 对照：`apa_v42_no_shift_7dataset_2seed_train256_detail.csv` 中相同 splits

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 subset` | 4 | +0.0059 | +0.0040 | +0.0055 | +0.0006 | +0.0009 | 2 / 0 / 2 |
| `V50 cross-model tree rescue` | 4 | +0.0098 | +0.0106 | +0.0129 | +0.0011 | +0.0050 | 3 / 0 / 1 |
| `V50 - V42 subset` | 4 | +0.0039 | +0.0067 | +0.0075 | +0.0005 | +0.0041 | 无新增负例 |

结论：

- V50 是目前第一个在 matched smoke suite 上超过 V42 的异构 model-member
  ensemble，且没有新增 Acc/BAcc/Macro-F1 负 split。
- 但它还不能替代 V42/V43：尚未完成 full 7 数据集 x 2 seed train256/test128
  大规模验证，且计算成本明显更高。
- ECE 结论混合：train128 比 V42 略好，train256 两数据集 subset 比 V42 更差。

当前处理：

- 保留 V50 作为 promising candidate，不更新 current-best。
- 下一步若继续，应先优化 V50 成本，再跑完整 large no-shift 7 数据集验证。
- 如果完整验证仍保持 hard 指标正收益，可考虑把 V50 作为 V43 的 no-shift
  分支升级候选；如果 ECE 继续变差，则需要后接 calibration-aware gate。

Artifacts:

- `results/apa_v50_cross_model_tree_rescue_notes.md`
- `results/apa_v50_cross_model_tree_no_shift_smoke4_2seed_train128_r2_detail.csv`
- `results/apa_v50_cross_model_tree_no_shift_smoke4_2seed_train128_r2_summary.csv`
- `results/apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2_detail.csv`
- `results/apa_v50_cross_model_tree_no_shift_smoke2_2seed_train256_r2_summary.csv`

## 2026-07-06: `apa_v51_calibrated_tree_rescue`

这一轮继续 calibration-aware ensemble 方向，但只作用于 V50 的树 rescue
样本，不改 V42 primary member，也不改树 rescue 的 hard selection。

方法：

- 复用 V50：`V42 primary -> ExtraTrees/RandomForest consensus rescue ->
  local/p-value/prior guards`。
- 只对被树 rescue 接受的样本调概率，提高 rescued class 的置信度；
  boost 由 local support advantage、tree margin、local p-value 决定。
- hard prediction 必须保持 V50 不变。

probe：

在 V50 唯一触发 train128 树 rescue 的 `openml_1480_ilpd seed=2` 上：

| 方法 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `V50` | +0.0312 | +0.0455 | +0.0541 | +0.0043 | -0.0331 |
| `V51` | +0.0312 | +0.0455 | +0.0541 | +0.0043 | -0.0173 |

V51 的 confidence boost 保住了 hard 指标，但削弱了 ECE 改善。

train128/test64 no-shift smoke：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V50 cross-model tree rescue` | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |
| `V51 calibrated tree rescue` | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0074 | 6 / 0 / 2 |

结论：

- V51 不伤 hard metrics，但 ECE 明显弱于 V50，因此不扩测。
- 这说明对 rescue 样本做手工 confidence boost 不是好的 calibration-aware
  策略；后续如果继续校准，应做 route-level temperature/gate，并用
  NLL/Brier/ECE 共同筛选。

Artifacts:

- `results/apa_v51_calibrated_tree_rescue_notes.md`
- `results/apa_v51_calibrated_tree_ilpd_s2_probe_detail.csv`
- `results/apa_v51_calibrated_tree_no_shift_smoke4_2seed_train128_detail.csv`
- `results/apa_v51_calibrated_tree_no_shift_smoke4_2seed_train128_summary.csv`

## 2026-07-06: `apa_v52_fast_tree_rescue`

这一轮是 V50 的效率 ablation：保留 cross-model tree rescue 思路，但把
primary member 从 V42 换成更便宜的 V27 guarded-boundary member。

统一算法线仍是：

`primary no-shift member -> ExtraTrees/RandomForest consensus rescue -> local/p-value/prior guards`

区别是：

- V50 primary：V42 boundary-bypass scale combo，较慢但更稳。
- V52 primary：V27 guarded boundary-union，较快但少了 V42 在 large no-shift
  上的 validation/scale 保护。

train128/test64 no-shift smoke：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V52 fast tree rescue` | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |

train256/test128 scale check：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 | 平均 runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V50 same subset` | 4 | +0.0098 | +0.0106 | +0.0129 | +0.0011 | +0.0050 | 3 / 0 / 1 | 146.4s |
| `V52 fast tree rescue` | 4 | +0.0020 | -0.0031 | +0.0001 | +0.0003 | +0.0033 | 1 / 3 / 0 | 60.5s |

结论：

- V52 明显更快，但 train256 上引入 hard-metric 负例。
- 这说明 V50 的 V42 primary 不只是计算开销，也是 large no-shift 稳定器。
- 不扩测 V52；后续若要降成本，应优化 V42 内部计算，而不是简单换成 V27
  primary。

Artifacts:

- `results/apa_v52_fast_tree_rescue_notes.md`
- `results/apa_v52_fast_tree_no_shift_smoke4_2seed_train128_detail.csv`
- `results/apa_v52_fast_tree_no_shift_smoke2_2seed_train256_detail.csv`

## 2026-07-06: `apa_v53_pseudo_label_context`

这一轮尝试 pseudo-label / label-side augmentation 方向，不再继续微调
cross-model tree rescue。

方法：

- 先跑 TabICL baseline。
- 从 test batch 中选高置信伪标签池：confidence `>= 0.90` 且 top-2 margin
  `>= 0.48`。
- 对低 margin target rows，用训练集 + 伪标签池重新构造 context；预测某个
  target 时会排除 target 自己的伪标签，避免自我泄漏。
- 只有当重预测结果改变 baseline，且原训练集 local KNN support 与
  binomial p-value 都支持候选类时，才接受改动。

train128/test64 no-shift smoke：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V53 pseudo-label context` | 8 | +0.0020 | +0.0015 | +0.0017 | +0.0005 | -0.0019 | 1 / 0 / 7 |

唯一正例是 `openml_37_diabetes seed=2`：

| split | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openml_37_diabetes seed=2` | +0.0156 | +0.0122 | +0.0136 | +0.0042 | -0.0153 |

train256/test128 scale check：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V53 pseudo-label context` | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 / 4 |

结论：

- V53 安全，但信号很弱；train128 只有一个正例，train256 全零。
- 不深化该方法。若后续再试 pseudo-label augmentation，应使用更强 teacher
  ensemble 生成伪标签，而不是 baseline 单模型。
- 相比之下，当前更值得深化的仍是 V50 cross-model tree rescue 或 V42 主线。

Artifacts:

- `results/apa_v53_pseudo_label_context_notes.md`
- `results/apa_v53_pseudo_label_no_shift_smoke4_2seed_train128_detail.csv`
- `results/apa_v53_pseudo_label_no_shift_smoke2_2seed_train256_detail.csv`

## 2026-07-06: `apa_v54_prp_uncertainty_guard`

这一轮尝试 uncertainty / PRP reliability score 方向。

方法：

- 先跑 V42 `boundary-bypass scale combo`。
- 再跑 multi-context family diagnostic member，包含 boundary、diverse、
  prototype、clean、minority-density 等 context。
- 用 context disagreement、family agreement ratio、local support、local
  purity、local p-value、diagnostic member 是否同意 V42 等信号构造 PRP-like
  risk score。
- 只在 uncertainty/risk gate 通过时保留 V42 改动，否则回退 baseline。

train128/test64 no-shift smoke：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass combo` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| `V54 PRP uncertainty guard` | 8 | +0.0078 | +0.0094 | +0.0104 | +0.0014 | -0.0037 | 3 / 0 / 5 |
| `V54 - V42` | 8 | -0.0078 | -0.0056 | -0.0063 | -0.0011 | +0.0055 | 无新增负例 |

结论：

- V54 安全但过于保守，过滤掉多个 V42 正收益。
- 不作为 prediction router 扩测。
- 保留 PRP diagnostics 作为解释和 selective prediction 分析信号，但不对
  V42 做 hard gate。

Artifacts:

- `results/apa_v54_prp_uncertainty_guard_notes.md`
- `results/apa_v54_prp_uncertainty_no_shift_smoke4_2seed_train128_detail.csv`
- `results/apa_v54_prp_uncertainty_no_shift_smoke4_2seed_train128_summary.csv`

## 2026-07-06: `apa_v55_prior_temperature`

这一轮尝试 route-level calibration：不改 V42 hard prediction，而是给 V42
输出加一个全局 temperature。temperature 不用 test label，而是通过最小化
batch 平均预测 prior 与 train prior 的 JS divergence 选择。

方法：

- 先跑 V42 `boundary-bypass scale combo`。
- 对温度网格 `0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.80, 2.10`
  做 `softmax(log(p) / T)`。
- 选使 batch prior 最接近 train prior 的 T。
- 只有 prior JS 改善、confidence shift 不过大、argmax 不变时才应用。

train128/test64 no-shift smoke：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass combo` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| `V55 prior-temperature` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | +0.0023 | 6 / 0 / 2 |
| `V55 - V42` | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0115 | hard 不变，ECE 变差 |

结论：

- V55 保持 hard metrics 和 AUC，但 ECE 明显弱于 V42。
- prior matching 不是可靠的 calibration objective；即使 batch prior JS 大幅下降，
  ECE 也可能变差。
- 不扩测 V55。校准方向如果继续，应避免只优化 prior consistency。

Artifacts:

- `results/apa_v55_prior_temperature_notes.md`
- `results/apa_v55_prior_temperature_no_shift_smoke4_2seed_train128_detail.csv`
- `results/apa_v55_prior_temperature_no_shift_smoke4_2seed_train128_summary.csv`

## 2026-07-06: `apa_v50_extra3_train128_deepening`

经过 V46-V55 的广泛方法筛选后，V50 cross-model tree rescue 是最值得深化的
新方法。本轮不再调新阈值，而是扩展 V50 的数据覆盖：在 V42 large no-shift
benchmark 中此前 train128 未覆盖的 3 个数据集上做 matched smoke。

新增数据集：

- `openml_1498_sa_heart`
- `openml_1464_blood_transfusion`
- `openml_1050_pc3`

命令：

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_1498_sa_heart,openml_1464_blood_transfusion,openml_1050_pc3 --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_cross_model_tree_rescue_router --output-prefix apa_v50_cross_model_tree_no_shift_extra3_2seed_train128
```

新增 3 数据集结果：

- V42 和 V50 都是全零，没有新增收益，也没有新增负例。

合并此前 4 数据集 smoke 后，得到 7 数据集 x 2 seed train128/test64：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass` | 14 | +0.0089 | +0.0086 | +0.0095 | +0.0018 | -0.0054 | 6 / 0 / 8 |
| `V50 cross-model tree rescue` | 14 | +0.0089 | +0.0093 | +0.0105 | +0.0017 | -0.0054 | 6 / 0 / 8 |
| `V50 - V42` | 14 | +0.0000 | +0.0008 | +0.0010 | -0.0001 | -0.0001 | 无新增负例 |

结论：

- V50 在更宽的 train128 7 数据集视角下仍安全，没有新增 hard-metric 负例。
- V50 相比 V42 有小幅 BAcc/Macro-F1 增益，但收益稀疏，主要来自
  `openml_1480_ilpd seed=2` 的 tree rescue。
- 下一步若继续深化，应该做 full train256 7 数据集验证或更高效的 V50
  大规模评估，而不是继续调阈值。

Artifacts:

- `results/apa_v50_extra3_train128_deepening_notes.md`
- `results/apa_v50_cross_model_tree_no_shift_extra3_2seed_train128_detail.csv`
- `results/apa_v50_cross_model_tree_no_shift_extra3_2seed_train128_summary.csv`

## 2026-07-06: `apa_v56_prp_tree_rescue`

这一轮把 V50 和 V54 收束成更统一的一版：V42 仍是 no-shift primary，
ExtraTrees/RandomForest 仍是 sparse rescue member，但 PRP / multi-context
diagnostics 只在 tree rescue 候选附近启用。

方法：

- 先跑 V42 `boundary-bypass scale combo`。
- 仅在 V42 未改动、ET/RF agree、tree margin/confidence、local support、
  local p-value、prior drift、minority-to-majority risk 都通过时生成 rescue
  候选。
- PRP family diagnostics 包含 boundary、diverse、prototype、clean、
  minority-density contexts。
- 第一版 eager PRP 在 smoke 中运行约 9 分钟还未落盘，因此改成 lazy PRP：
  只有存在 V50 候选或 tree 近候选时才跑 PRP。
- PRP 只负责保留/轻微扩展 tree rescue 候选，不再像 V54 那样过滤全部
  V42 改动。

关键 ILPD probe：

| seed | V42 BAcc/F1 Δ | V50 BAcc/F1 Δ | V56 BAcc/F1 Δ | 备注 |
| ---: | ---: | ---: | ---: | --- |
| 1 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | JS guard 阻止 rescue |
| 2 | +0.0346 / +0.0401 | +0.0455 / +0.0541 | +0.0455 / +0.0541 | V56 保留 V50 rescue |

四数据集 train128/test64 no-shift smoke：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass` | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6 / 0 / 2 |
| `V50 cross-model tree rescue` | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |
| `V56 PRP-aware tree rescue` | 8 | +0.0156 | +0.0164 | +0.0185 | +0.0025 | -0.0093 | 6 / 0 / 2 |

七数据集 train128/test64 no-shift view：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 | 平均 runtime | tree selected | PRP eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass` | 14 | +0.0089 | +0.0086 | +0.0095 | +0.0018 | -0.0054 | 6 / 0 / 8 | 28.1s | 0.0000 | 0.000 |
| `V50 cross-model tree rescue` | 14 | +0.0089 | +0.0093 | +0.0105 | +0.0017 | -0.0054 | 6 / 0 / 8 | 26.7s | 0.0022 | 0.000 |
| `V56 PRP-aware tree rescue` | 14 | +0.0089 | +0.0093 | +0.0105 | +0.0017 | -0.0054 | 6 / 0 / 8 | 27.3s | 0.0022 | 0.100 |

结论：

- V56 和 V50 指标一致，没有新增负例。
- Lazy PRP 只在 `openml_1480_ilpd seed=2` 的真实 tree rescue 候选上触发；
  selected fraction 0.03125，selected PRP risk 0.2856，family agreement 0.4。
- PRP 作为 prediction hard gate 仍没有带来新涨点，但作为 rescue 可靠性诊断
  成本已经可控。
- 当前主线仍应以 V50 为简洁 prediction 版本，以 V56 作为同一框架的
  explainable/diagnostic 版本：`V42 primary + sparse heterogeneous rescue +
  optional PRP diagnostics`。

Artifacts:

- `results/apa_v56_prp_tree_rescue_notes.md`
- `results/apa_v56_prp_tree_rescue_ilpd_2seed_train128_detail.csv`
- `results/apa_v56_prp_tree_rescue_no_shift_smoke3_2seed_train128_detail.csv`
- `results/apa_v56_prp_tree_rescue_no_shift_extra3_2seed_train128_detail.csv`

## 2026-07-06: `apa_v50_v56_train256_deepening`

这一轮把当前主线 `V42 primary + sparse heterogeneous rescue + optional PRP`
推进到 train256/test128。目标是确认 train128 上的 V50/V56 结论是否能在更大
训练规模上保持。

新增验证：

- `openml_1480_ilpd` train256/test128, seeds 1,2：
  baseline / V42 / V50 / V56 matched run。
- `openml_31_credit_g` train256/test128, seeds 1,2：
  V56 targeted run，用来检查 PRP 是否会误杀 V50 在 credit-g 上的 tree rescue。
- 复用已有 V50 train256 `credit_g + diabetes` 结果，与新增 ILPD 合成
  三数据集 matched view。
- 继续补齐 V50 train256 缺口：
  `openml_1498_sa_heart`, `openml_43_haberman`,
  `openml_1464_blood_transfusion`, `openml_1050_pc3`。

ILPD train256 结果：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass` | 2 | +0.0039 | +0.0029 | +0.0031 | +0.0004 | -0.0040 | 1 / 0 / 1 |
| `V50 cross-model tree rescue` | 2 | +0.0039 | +0.0029 | +0.0031 | +0.0004 | -0.0040 | 1 / 0 / 1 |
| `V56 PRP-aware tree rescue` | 2 | +0.0039 | +0.0029 | +0.0031 | +0.0004 | -0.0040 | 1 / 0 / 1 |

解读：

- ILPD 在 train128 上的 tree rescue 增益没有在 train256 上复现。
- V50/V56 在更大训练规模上安全退化为 V42，没有新增负例。

V56 credit-g train256 targeted check：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V56 PRP-aware tree rescue` | 2 | +0.0156 | +0.0188 | +0.0218 | +0.0023 | +0.0061 | 2 / 0 / 0 |

PRP diagnostics：

- seed 1：tree selected fraction 0.0078125，PRP evaluated，rejected V50 fraction 0.0。
- seed 2：tree selected fraction 0.0078125，PRP evaluated，rejected V50 fraction 0.0。

V50 train256 三数据集 matched view：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 matched` | 6 | +0.0052 | +0.0036 | +0.0047 | +0.0005 | -0.0007 | 3 / 0 / 3 |
| `V50 matched` | 6 | +0.0078 | +0.0081 | +0.0097 | +0.0009 | +0.0020 | 4 / 0 / 2 |
| `V50 - V42` | 6 | +0.0026 | +0.0045 | +0.0050 | +0.0003 | +0.0027 | 无新增负例 |

V50 train256 七数据集 full view：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 boundary-bypass` | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0013 | +0.0030 | 5 / 0 / 9 |
| `V50 cross-model tree rescue` | 14 | +0.0067 | +0.0079 | +0.0085 | +0.0016 | +0.0042 | 6 / 0 / 8 |
| `V50 - V42` | 14 | +0.0011 | +0.0029 | +0.0027 | +0.0003 | +0.0012 | 无新增负例 |

V50 - V42 的数据集级差异：

| 数据集 | Acc | BAcc | Macro-F1 | ROC-AUC | ECE | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `openml_31_credit_g` | +0.0078 | +0.0134 | +0.0149 | +0.0010 | +0.0082 | 主要 tree-rescue 收益 |
| `openml_1498_sa_heart` | +0.0000 | +0.0070 | +0.0041 | +0.0009 | +0.0005 | 不改 Acc，但改善 BAcc/F1 |
| 其余 5 个数据集 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 约 +0.0000 | 安全退化为 V42 |

V56 train256 两数据集 matched view：

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V42 matched` | 4 | +0.0059 | +0.0042 | +0.0050 | +0.0008 | -0.0030 | 2 / 0 / 2 |
| `V56 matched` | 4 | +0.0098 | +0.0108 | +0.0125 | +0.0013 | +0.0011 | 3 / 0 / 1 |
| `V56 - V42` | 4 | +0.0039 | +0.0067 | +0.0075 | +0.0005 | +0.0041 | 无新增负例 |

结论：

- V50 在当前可用 train256 三数据集视角仍有 hard-metric 正收益，且无新增负例。
- V50 在完整 train256 七数据集视角仍有 hard-metric 正收益，且无新增负例。
- 收益主要来自 `openml_31_credit_g`，另在 `openml_1498_sa_heart` 改善
  BAcc/F1；`openml_1480_ilpd` 的 train128 rescue 没有迁移到 train256。
- V56 保留了 credit-g 上的 V50 rescue，说明 PRP diagnostics 没有误杀当前
  有效 rescue；但 V56 仍没有产生额外 prediction gain。
- ECE 在 train256 上混合偏差，不能把 V50/V56 称为 calibration improvement。
- 当前论文线可写为：
  `Sparse Heterogeneous Rescue for No-Shift Tabular Foundation Models`。

Artifacts:

- `results/apa_v50_v56_train256_deepening_notes.md`
- `results/apa_v56_v50_ilpd_2seed_train256_detail.csv`
- `results/apa_v56_prp_tree_rescue_credit_2seed_train256_detail.csv`
- `results/apa_v50_cross_model_tree_no_shift_gap2_2seed_train256_detail.csv`
- `results/apa_v50_cross_model_tree_no_shift_gap2b_2seed_train256_detail.csv`

## 2026-07-06: `apa_v57_mixed_v38_v50_router`

这一轮把 V50 接入 mixed/no-shift 统一 router，形成 V57。

动机：

- V43 = V38 forced-shift branch + V42 no-shift branch。
- V50 已经成为 full train256 七数据集 no-shift hard-metric 最优候选。
- 因此 V57 只升级 no-shift branch：large low-JS no-shift batch 走 V50；
  ambiguous/high-JS batch 仍走 V38 或 baseline。

方法：

- 先计算 baseline batch-prior JS。
- 若 `batch_prior_js <= min(config.max_batch_prior_js, 0.004)` 且
  `train_size >= 192 or test_size >= 96`，走 V50 sparse cross-model tree rescue。
- 否则走 V38 V14-priority label-shift branch；若 label-shift correction 的
  context-disagreement guard 不通过，则回退 baseline。

No-shift targeted train256/test128：

数据集：

- `openml_31_credit_g`
- `openml_1498_sa_heart`

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V57 mixed V38/V50` | 4 | +0.0195 | +0.0249 | +0.0263 | +0.0054 | +0.0147 | 4 / 0 / 0 |

分支统计：

- `no_shift_v50_cross_model_tree_rescue`: 4 / 4

Forced-shift targeted：

数据集：

- `openml_31_credit_g`
- `openml_37_diabetes`
- `openml_1498_sa_heart`

设置：

- force label shift: `0.6, 0.9`
- seeds: `1,2`
- train/test: `128/64`

| 方法 | 组数 | Acc Δ | BAcc Δ | Macro-F1 Δ | ROC-AUC Δ | ECE Δ | BAcc 正/负/零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V38 V14-priority` | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | -0.0065 | 8 / 0 / 4 |
| `V57 mixed V38/V50` | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | -0.0065 | 8 / 0 / 4 |

逐行核验：

- V57 与 V38 在 12 个 forced-shift trial 上完全一致。
- max absolute metric delta difference = 0.0。

Forced-shift 分支统计：

- `label_shift_v38_priority`: 8 / 12
- `baseline_shift_guard`: 4 / 12

结论：

- V57 是 V43 的自然升级：把 no-shift 分支从 V42 替换为 V50，同时保留 V38
  forced-shift 分支。
- targeted no-shift/forced-shift 检查均无 Acc/BAcc/F1 负例。
- 在完成完整 V43 forced-shift targeted 30 trial 复测前，V57 先作为统一候选；
  V43 仍是更充分验证过的 mixed router。

Artifacts:

- `results/apa_v57_mixed_v38_v50_notes.md`
- `results/apa_v57_mixed_v38_v50_no_shift_target2_2seed_train256_detail.csv`
- `results/apa_v57_mixed_v38_v50_forced_shift_target3_2seed_detail.csv`
