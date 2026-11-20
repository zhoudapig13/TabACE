# V54 PRP uncertainty guarded V42

## Goal

This round tests the reference idea of uncertainty / PRP-style reliability
scoring. Instead of introducing another prediction member, V54 uses
multi-context prediction disagreement to decide whether V42 edits should be
kept.

## Method

V54:

1. Run V42 `boundary-bypass scale combo`.
2. Run a multi-context family diagnostic member over boundary, diverse,
   prototype, clean, and minority-density contexts.
3. Compute a PRP-like risk score from:
   context disagreement, family agreement ratio, local support, local purity,
   local p-value, and whether the diagnostic member agrees with V42.
4. Keep a V42 edit only when uncertainty/risk gates pass; otherwise return the
   baseline prediction for that row.

This tests whether uncertainty diagnostics can improve stability by filtering
unsafe edits.

## No-shift train128/test64 smoke

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_prp_uncertainty_guard_router --output-prefix apa_v54_prp_uncertainty_no_shift_smoke4_2seed_train128
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass, same suite | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |
| V54 PRP uncertainty guard | 8 | +0.0078 | +0.0094 | +0.0104 | +0.0014 | -0.0037 | 3/0/5 |
| V54 - V42 | 8 | -0.0078 | -0.0056 | -0.0063 | -0.0011 | +0.0055 | no extra negatives |

V54 rejected several V42-positive edits:

- `openml_31_credit_g seed=1`
- `openml_43_haberman seed=1`
- `openml_43_haberman seed=2`
- half of the V42 edits on `openml_1480_ilpd seed=2`

## Decision

Do not expand V54 as a prediction router.

The PRP uncertainty score is safe, but too conservative. It removes useful V42
edits and weakens accuracy, BAcc, Macro-F1, ROC-AUC, and ECE relative to V42.

Keep the PRP diagnostics as analysis features:

- high risk score explains why certain V42 edits are fragile;
- context disagreement/family agreement can be useful for reporting or
  selective prediction;
- but hard gating by this score is not competitive with V42.

## Artifacts

- `apa_v54_prp_uncertainty_no_shift_smoke4_2seed_train128_detail.csv`
- `apa_v54_prp_uncertainty_no_shift_smoke4_2seed_train128_summary.csv`
