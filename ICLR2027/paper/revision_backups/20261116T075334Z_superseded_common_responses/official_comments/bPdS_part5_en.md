## Part 5: Correlated regression errors and confident classification errors

> Regression gate robustness. Since Ureg increases with Base–aggregate disagreement, updates are largest when the alternatives disagree most with Base. How does the method behave when these alternatives are correlated and jointly incorrect?
> Failure under confident errors. In classification, a low-entropy but incorrect Base prediction closes the gate, potentially preventing useful corrections. How frequently does this occur, and how does it affect overall performance?

Thank you for identifying these limitations. **Official Comment C4** provides complete diagnostics and extension results (Appendix E.7, Tables 29–31, pp. 43–44).

**Regression:** the update $\alpha_i=\alpha_{\max}U_{\mathrm{reg},i}E_i$ depends on both disagreement and context evidence. Correlated alternatives can still agree on an incorrect answer. Material fusion harm in the severe-joint-error cohort affects **0.021%–0.547%** of all query–condition pairs. Source-calibrated residual correction and risk contraction improve corruption/shift-average $R^2$ from **−0.12264 to −0.09363** for TabICL and **−0.19912 to −0.16110** for TabPFN, recovering **30.32%/64.75%** of this cohort's excess fusion loss.

**Classification:** high-confidence Base errors (maximum probability ≥0.9) affect **2.06%–2.44%** of queries. Adding relative confidence, disagreement, and consensus to update-strength estimation corrects **17.09%/26.43%** of these errors; overall Acc becomes **67.8380%/66.6759%**, versus 67.5532%/66.1437% originally. These cases have limited measured aggregate impact, and the extensions provide modest overall improvements with frozen backbones.

We thank you for helping us strengthen the context-budget tests, ensemble baselines, generalization evidence, and deployment details. If these additional results and clarifications resolve your main concerns, we would greatly appreciate your consideration of a more favorable score.
