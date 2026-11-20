# Experiment 2 Search Summary

This report compares each candidate method against `baseline_n1` on the same dataset split. Positive run counts show whether the gain appeared once or repeatedly across random seeds.

## Strongest Dataset-Method Signals

| Method | Dataset | Runs | Accuracy Change | BAcc Change | Macro F1 Change | AUC Change | Positive Runs | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| quantile_linear_probe_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9683 (-0.0072) | 0.9774 -> 0.9629 (-0.0145) | 0.9973 -> 0.9964 (-0.0008) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append balanced linear-probe features |
| quantile_domain_probe_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9683 (-0.0072) | 0.9774 -> 0.9629 (-0.0145) | 0.9973 -> 0.9964 (-0.0008) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append domain-shift plus linear-probe features |
| causal_quantile_domain_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9567 (-0.0189) | 0.9774 -> 0.9621 (-0.0154) | 0.9973 -> 0.9950 (-0.0023) | acc 0/1, bacc 0/1, f1 0/1 | Keep top causal-proxy features, quantile-normalize, then append domain-shift features |
| propensity_quantile_domain_feature | breast_cancer | 1 | 0.9790 -> 0.9650 (-0.0140) | 0.9756 -> 0.9645 (-0.0111) | 0.9774 -> 0.9627 (-0.0148) | 0.9973 -> 0.9971 (-0.0002) | acc 0/1, bacc 0/1, f1 0/1 | Quantile-normalize features and append domain-shift plus propensity features |

## Top Single-Run Metric Improvements

_No positive metric deltas._
