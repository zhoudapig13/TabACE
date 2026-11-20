## Part 3: Generalization coverage and distribution of gains

> The robustness evidence is narrower than the overall results suggest. Calibration covers the same registered mechanisms used in evaluation. The natural-shift study contains only two migration settings and three tasks. Gains also vary substantially across families. Based on Table 15, Domain and Temporal account for roughly 86% of Shared’s total regression improvement across the seven families.

Thank you for highlighting both coverage and uneven gains. **Official Comment C1** reports the mechanism-held-out Shortcut ablation and six unseen mechanisms; D achieves **0.69333/0.84662 Acc/AUC** without either Shortcut calibration or a matched constructor, versus Base's 0.51719/0.65690. Shared improves Acc and $R^2$ in all six new settings.

**Official Comment C3** broadens the natural-shift evidence beyond Folktables to hospital-system and admission-source transfer. All three medical OOD settings improve; readmission Base/Shared/Specific accuracy is **57.27%/58.97%/59.24%** in the 1:1 case–control evaluation.

**We agree that gains vary substantially by family**, with Domain and Temporal contributing most to the aggregate regression improvement. Appendix Table 15 reports each family separately: averaged over nonzero strengths, **Shared improves $R^2$ in all seven families**. The family-level results and additional held-out-mechanism/clinical tests show that the benefits extend beyond the two largest contributors.
