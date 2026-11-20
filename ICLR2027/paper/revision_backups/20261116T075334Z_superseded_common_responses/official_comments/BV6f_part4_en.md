## Part 4: High-confidence errors and extreme regression branches

> The mechanism has relevant limitations: the entropy gate suppresses corrections to confident Base errors, and median-based support scoring does not prevent an extreme regression branch from pulling the weighted mean. These cases matter for the paper’s robustness motivation.
> How often can TabACE correct high-confidence Base errors? How sensitive is the regression output to one extreme alternative prediction? Are branch predictions bounded or clipped?

Thank you for identifying these failure modes. **Official Comment C4** gives their definitions, frequencies, and original-versus-extended results; the full analysis is in Appendix E.7, Tables 29–31 (pp. 43–44).

- **Confident errors:** Base predictions with maximum class probability ≥0.9 that are incorrect affect **2.06%–2.44%** of query–condition pairs. The extension corrects **17.09%/26.43%** of these errors for TabICL/TabPFN, raising overall Acc from **67.5532% to 67.8380% / 66.1437% to 66.6759%**. Aggregation is unchanged; the added confidence/disagreement/consensus score is calibrated on training folds only.
- **Single extremes:** original scalar predictions are not additionally clipped. Only **0.109%–0.379%** of queries have one alternative beyond the median-centered ±3 source-label-SD interval. Clipping changes ordinary average $R^2$ by **<0.0007**, while reducing stress-test output displacement P95 from **24.264 to 0.996** source SDs for TabICL corruptions.
- **Joint errors:** material fusion harm affects **0.021%–0.547%** of query–condition pairs. Source-calibrated residual correction and risk-based contraction modestly improve average regression performance, recovering **30.32%/64.75%** of excess fusion loss within the fixed severe-joint-error cohort.

**These cases have limited measured aggregate impact, and the additional diagnostics motivated useful improvements while preserving the frozen-backbone design.**
