# V55 prior-matched temperature calibration

## Goal

This round tests a route-level calibration method. Unlike V51, V55 does not
manually boost selected rows. It keeps V42 hard predictions fixed and chooses a
single temperature by matching the batch predicted class prior to the training
class prior.

No test labels are used.

## Method

V55:

1. Run V42 `boundary-bypass scale combo`.
2. For a grid of temperatures, rescale V42 probabilities:
   `p_T = softmax(log(p) / T)`.
3. Choose the temperature that minimizes JS divergence between mean predicted
   batch prior and the training prior.
4. Apply the temperature only if:
   the prior JS improves, confidence shift is not too large, and argmax labels
   are unchanged.

This is intended as a cheap calibration-aware route-level postprocessor.

## No-shift train128/test64 smoke

Command:

```text
python experiment6/scripts/run_apa_tabicl.py --datasets openml_31_credit_g,openml_37_diabetes,openml_1480_ilpd,openml_43_haberman --split-mode ordered --seeds 1,2 --n-samples 0 --train-size 128 --test-size 64 --k 16 --n-estimators 1 --device cpu --methods tabicl_baseline,apa_cf_no_shift_prior_temperature_router --output-prefix apa_v55_prior_temperature_no_shift_smoke4_2seed_train128
```

| Method | n | Acc delta | BAcc delta | Macro-F1 delta | ROC-AUC delta | ECE delta | BAcc pos/neg/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass, same suite | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | -0.0092 | 6/0/2 |
| V55 prior-temperature | 8 | +0.0156 | +0.0150 | +0.0167 | +0.0026 | +0.0023 | 6/0/2 |
| V55 - V42 | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0115 | hard metrics unchanged, ECE worse |

Temperatures were often applied and did reduce train-prior JS, but this did not
translate into better ECE. Example:

- `openml_1480_ilpd seed=1`: T = `2.1`, prior JS improved from `0.007279` to
  `0.000026`, but ECE worsened by `+0.0325`.
- `openml_1480_ilpd seed=2`: T = `0.9`, prior JS improved from `0.000053` to
  `0.000002`, but ECE worsened by `+0.0556`.

## Decision

Do not expand V55.

Matching the predicted batch prior to the train prior is not a reliable
calibration objective for this no-shift setting. It preserves hard metrics and
AUC, but it weakens ECE relative to V42.

Calibration should not be optimized only by prior consistency. If revisited, it
should use a validation-free objective closer to calibration itself, or use a
carefully guarded validation temperature with NLL/Brier/ECE checks.

## Artifacts

- `apa_v55_prior_temperature_no_shift_smoke4_2seed_train128_detail.csv`
- `apa_v55_prior_temperature_no_shift_smoke4_2seed_train128_summary.csv`
