# V46-V49 diversity / p-value ensemble attempts

## Goal

This round tests the reference idea of diversity-aware and instance-wise
ensembling on the current no-shift line. The baseline to beat is V42
`apa_cf_no_shift_boundary_bypass_combo_router`.

All runs below use ordered no-shift splits on:

`openml_31_credit_g, openml_37_diabetes, openml_1480_ilpd, openml_43_haberman`

with seeds `1,2`, train/test `128/64`, `k=16`, CPU, one TabICL estimator.

## V46: diversity-greedy instance router

Method:

- Generate structured context members: diverse, prototype, clean,
  boundary, feature-subspace, and minority-density.
- For each test sample, greedily select up to three members using:
  member evidence + small JS-diversity reward - cost penalty.
- Fuse selected members with per-sample weights.
- Accept a changed prediction only when local support, context agreement,
  minority-risk guard, and change budget pass.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_diversity_greedy_router --output-prefix apa_v46_diversity_greedy_no_shift_smoke4_2seed_train128
```

Result:

| method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |
| V46 diversity-greedy | 8 | +0.0039 | +0.0042 | +0.0045 | +0.0005 | -0.0039 | 1/0/7 |

Conclusion: safe but too conservative. It only recovered the
`openml_37_diabetes seed=2` positive case and missed most V42 gains.

## V47: diversity-greedy with local p-value rescue

Method:

- Same member pool and greedy selection as V46.
- Adds a local-neighborhood p-value rescue for minority-risk boundary changes.
- Allows a candidate only when the local support p-value and local purity are
  acceptable.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_bypass_combo_router,apa_cf_no_shift_diversity_pvalue_router --output-prefix apa_v47_diversity_pvalue_no_shift_smoke4_2seed_train128
```

Result:

| method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |
| V47 diversity-pvalue | 8 | +0.0039 | +0.0029 | +0.0034 | +0.0008 | -0.0001 | 2/0/6 |

V47 found one V42-missed positive case (`openml_1480_ilpd seed=1`) but still
missed many V42 positives and weakened the `openml_37_diabetes seed=2` gain.
It is a useful diagnostic member, not a replacement.

## V48: boundary-diversity member selector

V48 tests a higher-level member selector:

- V42 is treated as the high-precision primary boundary member.
- V47 is treated as a diversity/p-value rescue member.
- r1 allowed V47 to override V42; this caused negative diabetes splits.
- r2 is precision-first: keep V42 whenever it changes; let V47 act only when
  V42 abstains.

Commands:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_diversity_selector_router --output-prefix apa_v48_boundary_diversity_selector_no_shift_smoke4_2seed_train128

python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_boundary_diversity_selector_router --output-prefix apa_v48_boundary_diversity_selector_no_shift_smoke4_2seed_train128_r2
```

Result:

| method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V48 r1 selector | 8 | +0.0059 | +0.0020 | +0.0028 | +0.0004 | -0.0037 | 4/2/2 |
| V48 r2 precision-first | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |

Conclusion: r1 is rejected due to two BAcc negative splits. r2 is safe but
numerically equal to V42 on this smoke suite, while being much more expensive.

## V49: precision-first p-value rescue selector

V49 keeps the V48 r2 precision-first rule, but lets the V47 rescue member use a
more permissive p-value rescue configuration. The goal was to recover V47's
`openml_1480_ilpd seed=1` positive case while preserving V42's high-precision
changes.

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_precision_rescue_router --output-prefix apa_v49_precision_rescue_no_shift_smoke4_2seed_train128
```

Result:

| method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V49 precision-rescue | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |

Conclusion: V49 is safe but still numerically equal to V42. The permissive
rescue did not recover V47's extra `openml_1480_ilpd seed=1` positive case once
the precision-first selector gates were applied.

## Decision

Do not replace V42 with V46/V47/V48/V49.

Keep the useful pieces as future meta-router features:

- per-sample selected member count;
- local support p-value;
- selected-member diversity;
- target vote fraction;
- minority-risk rescue diagnostics.

The next promising direction is not another hard cascade, but a cheaper
meta-router that learns or scores when V47's extra candidate is safe, without
overriding V42's high-precision changes.
