# Experiment 6

Experiment 6 is the active workspace for the next round of experiments.

## Boundary

All upcoming experiment-specific content, files, scripts, configurations, test
helpers, run logs, and generated results for this line of work should live under
`experiment6/`.

Repository-level shared code may still be imported when needed, but new
experiment artifacts should not be written to the top-level `results/` or
`scripts/` directories unless explicitly requested.

## Structure

- `ALGORITHM.md`: method description, assumptions, and protocol.
- `configs/`: runnable experiment configurations.
- `scripts/`: experiment entry points and analysis utilities.
- `tests/`: smoke tests and regression checks.
- `results/`: generated outputs, summaries, figures, and reports.
- `RESULTS_SUMMARY.md`: running log of commands, artifacts, and findings.

## Current Candidate

The conservative reusable candidate is `apa_dual_shift_router_v7`. The current
research candidate is `apa_cf_directional_router`, which uses counterfactual
context-prior intervention rather than only a fixed probability correction.

For direct v7 use, import the wrapper from `experiment6/scripts/apa_tabicl.py`:

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

This wrapper is inference-only: it fits the base TabICL estimator in the normal
in-context-learning sense and applies APA post-hoc correction at prediction
time. It does not retrain TabICL or modify pretrained weights.

The v8 research candidate is currently exposed through
`experiment6/scripts/run_apa_tabicl.py` as method
`apa_cf_directional_router`.
