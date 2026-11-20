## Part 1: Generalization beyond calibration and constructor matching

> The Shortcut corruption corrupts the most predictive feature, and one of the five Shared classification contexts simply drops that feature.
>
> The weighting policy is calibrated on the same corruption types and strengths
> the Specific variant is even told which one is present.
> How does TabACE perform on corruptions unseen during calibration?

Thank you for raising this central concern. **Official Comment C1** reports the $2\times2$ Shortcut ablation and six unseen-mechanism evaluations. Without either Shortcut calibration or its matched constructor, D attains **0.69333/0.84662 Acc/AUC**, versus Base's 0.51719/0.65690. Shared also improves all four metrics under all six unseen mechanisms. **The gains therefore extend beyond matched repairs and calibration on the evaluated mechanism.**

Specific exploits family knowledge when available, as in problem-aware context construction [5]; **Shared uses no mechanism identifier and remains effective when that knowledge is unavailable**. Full protocols and results are in Appendix E.3–E.4 (Tables 22–24, p. 39).

**References**

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.
