# Baseline Audit

This audit records which baselines are part of experiment3 and which optional
classical causal dependencies are available in the current environment.

## Retrieval Baselines

The required retrieval baselines are present and retained:

- `random`
- `knn`
- `matched_pair`
- `matched_pair_diverse`

The main method remains `matched_pair_diverse`; no new retriever is promoted in
this audit.

## Full-Data Classical References

The built-in full-data reference estimators are available through
`experiment3/ccs/estimators.py`:

- `global_s_learner`
- `global_t_learner`
- `global_x_learner`
- `global_dr_learner`

These are reference baselines only. Not beating a full-data learner is not a
failure for the context-limited retrieval question.

## Optional External Classical Packages

Checked in the current environment:

| Package / estimator | Status | Consequence |
| --- | --- | --- |
| `econml` | unavailable: `ModuleNotFoundError` | Cannot add `CausalForestDML`, `DRLearner`, or `XLearner` references yet. |
| `causalml` | unavailable: `ModuleNotFoundError` | Cannot add causalml uplift/meta-learner references yet. |
| `sklift` | unavailable: `ModuleNotFoundError` | Cannot add scikit-uplift AUUC/Qini references yet. |
| `dowhy` | unavailable: `ModuleNotFoundError` | No DoWhy reference path available. |

The current paper-ready tables therefore use the built-in global learners as
classical references and record optional classical-package support as a
dependency blocker.
