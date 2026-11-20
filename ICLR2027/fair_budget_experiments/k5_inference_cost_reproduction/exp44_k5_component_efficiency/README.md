# Experiment 11: component-level inference efficiency

This diagnostic profiles the same representative six-Context execution paths
used by Experiment 03 and separates synchronized TabICL fit-and-predict time
from Context construction and fusion time. It uses three datasets, seven shift
families, and strength 0.5 for each task.

The checkpoint is loaded once per task and reused. Shift construction and the
one-time checkpoint initialization are outside the recorded cells. Context
construction, repeated TabICL calls, reliability-independent aggregation, and
CUDA synchronization are inside the recorded cells.

These runs diagnose the mechanism's implementation cost. They are not a
replacement for the frozen-test accuracy experiment, and the six-Context paths
should not be presented as timing measurements of TabACE-Pro.

## Observed median breakdown

| Task | Path | Total (ms) | TabICL calls (ms) | Context + fusion (ms) | TabICL fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| Classification | TabICL Base | 20.8 | 20.7 | 0.1 | 99.6% |
| Classification | Random Context-6 | 121.5 | 120.1 | 1.0 | 99.2% |
| Classification | Shared Context-6 | 658.8 | 120.5 | 539.4 | 17.7% |
| Classification | Shift-specific Context-6 | 659.7 | 120.9 | 540.6 | 17.5% |
| Regression | TabICL Base | 16.7 | 16.7 | 0.1 | 99.6% |
| Regression | Random Context-6 | 100.5 | 99.9 | 0.6 | 99.4% |
| Regression | Shared Context-6 | 109.6 | 100.0 | 9.4 | 91.5% |
| Regression | Shift-specific Context-6 | 103.7 | 99.8 | 2.8 | 97.8% |

The classification propensity-weight calculation takes 546.9 ms at the
median over the same 21 cells, accounting for nearly all of the classification
Context-construction residual. The main engineering opportunity is therefore
to cache or amortize propensity estimation; reliability scoring and probability
fusion are not the observed bottleneck.
