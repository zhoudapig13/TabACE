## Part 3: Generalization beyond perturbation–constructor matching

> The perturbations appear closely matched to their corresponding constructors: Shortcut corrupts the top feature, while its constructor drops the top-correlated feature; Regression Concept introduces smooth drift while its constructor removes the strongest smooth component. Gains diminish when these matched constructors are absent and are most pronounced when shifts appear in the Shared bank.
> Without such a match (i.e., when the perturbation/shift is not included in the Shared Constructor bank), performance is nearly identical to Base. The gain is also minimal in Folktables settings when the shift is not by design. These results raise concerns about generalizability beyond the specifically matched perturbations and constructors.

Thank you for raising this concern. **Official Comment C1** separates calibration coverage and constructor matching in a $2\times2$ Shortcut experiment. C/D, which omit the matched constructor, achieve **0.71187/0.85757 and 0.69333/0.84662 Acc/AUC**, versus Base's 0.51719/0.65690; D also excludes Shortcut from calibration.

More directly, **Shared improves Acc and $R^2$ under all six mechanisms absent from calibration and dedicated constructor design** (C1; Appendix E.4, Tables 23–24). This evidence addresses the claim that performance is nearly identical to Base without a match. Specific exploits problem knowledge when available, consistent with problem-aware context construction [5], while **Shared requires no mechanism identifier**.

We agree that Folktables gains vary by task and migration setting. **C3** adds hospital-system and admission-source shifts: all six clinical ID/OOD settings improve, and readmission OOD Base/Shared/Specific accuracy is **57.27%/58.97%/59.24%** in the 1:1 case–control evaluation. Together these tests extend the evidence beyond specifically matched synthetic mechanisms.

**References**

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.
