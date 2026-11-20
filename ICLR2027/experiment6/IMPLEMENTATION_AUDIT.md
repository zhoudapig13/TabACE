# Experiment 6 Implementation Audit

This file maps the requested Experiment 6 requirements to concrete artifacts.
All code, configs, tests, and generated results for this line of work are kept
under `experiment6/`.

## Scope Guarantee

- Experiment files: `experiment6/scripts/`, `experiment6/configs/`,
  `experiment6/tests/`, and top-level Experiment 6 documentation.
- Result files: `experiment6/results/`.
- No source files outside `experiment6/` were edited for this experiment.
- No TabICL pretrained weights are modified. The methods are inference-time
  wrappers around the existing classifier interface.

## Requested Components

### Prior-Aware Context Construction

Implemented in `experiment6/scripts/apa_tabicl.py`:

- `select_knn_contexts`: nearest-neighbor local contexts.
- `select_balanced_knn_contexts`: class-balanced nearest-neighbor contexts.
- `select_prior_aware_contexts`: contexts selected with local neighborhood,
  class coverage, and batch-prior targets.
- `select_contexts`: dispatcher used by the experiment runner.

Runner ablations in `experiment6/scripts/run_apa_tabicl.py`:

- `apa_knn_context`
- `apa_balanced_knn`
- `apa_prior_aware_knn`

### Local and Effective Priors

Implemented in `experiment6/scripts/apa_tabicl.py`:

- `context_prior`: smoothed label prior from the selected context.
- `effective_priors`: supports context prior, optional query prior, and optional
  attention-weighted context prior.
- `batch_prior_js_score`: batch-level predicted-prior shift diagnostic.

Note: the reusable effective-prior function supports attention weights, but the
current TabICL sklearn inference path used here does not expose attention maps.
Therefore Experiment 6 evaluates context-prior and predicted-prior variants, not
a runtime attention-map ablation.

### Adaptive Posterior Correction

Implemented in `experiment6/scripts/apa_tabicl.py`:

- `query_distribution`: derives an uncertainty-aware query posterior proxy.
- `js_divergence`: shift score between prior estimates.
- `adaptive_posterior_correction`: applies gated prior-ratio posterior updates.
- `correct_existing_proba_with_context_prior`: posthoc correction for full-context
  TabICL predictions.
- `apa_v7_config_for_proba`: current candidate routing policy.
- `APATabICLClassifier`: reusable sklearn-style wrapper.

Current candidate:

- `apa_dual_shift_router_v7`
- Config: `experiment6/configs/apa_v7_candidate.json`

### Ablations

Required ablations are available in the method registry of
`experiment6/scripts/run_apa_tabicl.py`:

- Raw TabICL baseline: `tabicl_baseline`
- KNN context only: `apa_knn_context`
- Balanced KNN context: `apa_balanced_knn`
- Prior-aware KNN context: `apa_prior_aware_knn`
- Local correction: `apa_local_correction`
- Full prior-aware adaptive method: `apa_prior_aware_adaptive`

Additional iterations are preserved for comparison:

- `apa_full_prior_knn_adaptive`
- `apa_full_prior_aware_adaptive`
- `apa_full_prior_aware_tiny`
- `apa_shift_margin_gate`
- `apa_shift_margin_gate_soft`
- `apa_validation_selector`
- `apa_aggressive_margin_push`
- `apa_aggressive_batch_push`
- `apa_hard_shift_router`
- `apa_dual_shift_router`
- `apa_dual_shift_router_v7`

### Metrics

Implemented in `experiment6/scripts/apa_tabicl.py`:

- Accuracy
- Balanced accuracy
- Macro F1
- Negative log loss
- Expected calibration error
- Per-class recall
- Minority recall

Generated detail CSV files include per-split metrics and method metadata.
Generated summary CSV files include mean deltas versus the baseline.

### Label-Shift Curves

Implemented in `experiment6/scripts/run_apa_tabicl.py` via:

- `--label-shift-strengths`

Generated curve artifacts include:

- `experiment6/results/apa_v3_label_shift_curve_shift_curve.csv`
- `experiment6/results/apa_v4_hard_label_shift_curve_shift_curve.csv`
- `experiment6/results/apa_v5_hard_label_shift_curve_shift_curve.csv`
- `experiment6/results/apa_v5_mixed_safety_binary_gate_shift_curve.csv`
- `experiment6/results/apa_v6_hard_label_shift_5seed_shift_curve.csv`
- `experiment6/results/apa_v6_easy_label_shift_5seed_shift_curve.csv`
- `experiment6/results/apa_v7_hard_label_shift_5seed_shift_curve.csv`
- `experiment6/results/apa_v7_easy_label_shift_5seed_shift_curve.csv`

### Dataset Coverage

Runner support includes sklearn/synthetic datasets:

- `breast_cancer`
- `wine`
- `iris`
- `synthetic_balanced`
- `synthetic_imbalanced`
- `synthetic_label_shift`
- `synthetic_label_shift_hard`
- `synthetic_multiclass_imbalance`

Runner support also includes OpenML aliases:

- `openml_cmc`
- `openml_ilpd`
- `openml_credit_g`
- `openml_blood_transfusion`
- `openml_kc1`
- `openml_dmft`
- `openml_diabetes_pima`
- `openml_pc1`

### Current Result Summary

The current selected candidate is `apa_dual_shift_router_v7`.

Most relevant result artifacts:

- `experiment6/results/apa_v7_openml_small_stress_report.md`
- `experiment6/results/apa_v7_hard_label_shift_5seed_report.md`
- `experiment6/results/apa_v7_easy_label_shift_5seed_report.md`
- `experiment6/results/apa_v7_openml_broad3_report.md`

Key observed behavior from `experiment6/RESULTS_SUMMARY.md`:

- Hard synthetic label-shift, 5 seeds: mean balanced accuracy delta `+0.0026`,
  with `4 / 0` positive/negative balanced-accuracy splits.
- Easy synthetic label-shift, 5 seeds: mean balanced accuracy delta `+0.0008`,
  with `1 / 0` positive/negative balanced-accuracy splits and improved ECE.
- OpenML small stress suite: `0 / 0` positive/negative balanced-accuracy splits,
  blocking the v6 ILPD regression.
- OpenML broad 8-dataset x 3-seed suite: `0 / 0` positive/negative
  balanced-accuracy splits; v7 is intentionally conservative on real-data splits
  without a detected prior-shift signal.

## Validation

Latest validation command:

```bash
python -m json.tool experiment6/configs/apa_v7_candidate.json >/dev/null
python -m py_compile experiment6/scripts/apa_tabicl.py experiment6/scripts/run_apa_tabicl.py
python - <<'PY'
from experiment6.tests.test_apa_tabicl import *
for name, fn in sorted(globals().items()):
    if name.startswith('test_'):
        fn()
        print(f'{name}: ok')
PY
```

The manual test run covers 8 test functions in
`experiment6/tests/test_apa_tabicl.py`.

