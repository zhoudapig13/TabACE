## Part 2: Robustness under real-world distribution shifts

> The only natural-shift benchmark is Folktables, where the improvements are mostly small and half of the times similar to the improvements observed on the unshifted test data, suggesting a generic ensembling effect.
> And does it work across other real-world shifts, such as in the medical domain between eICU and MIMIC?

Thank you for suggesting broader natural-shift tests. **Official Comment C3** reports all twelve Folktables ID/OOD comparisons and six clinical settings. Both variants outperform Base, DistPFN–TabICL-v2, and Drift-Resilient in all twelve Folktables comparisons. For geographic TravelTime OOD, Base/Shared/Specific accuracy is **0.5472/0.5818/0.6012**.

While eICU/MIMIC authorization is pending, we tested bidirectional hospital-system transfer on PhysioNet Sepsis 2019 and admission-source transfer on UCI Diabetes 130-US Hospitals, using 1:1 case–control evaluations. **Specific > Shared > Base in all six ID/OOD settings**; readmission OOD accuracy is **57.27%/58.97%/59.24%** for Base/Shared/Specific. These are naturally occurring shifts, extending the evidence beyond Folktables (Appendix E.5, Tables 25–27, pp. 41–42).

We address the ordinary-ensembling explanation in **Part 4 and Official Comment C2**, with the same deployed prediction budget.
