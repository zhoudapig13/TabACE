# General Response to Reviewers

We sincerely thank all three reviewers for the time and care devoted to our submission. We especially appreciate the recognition of TabACE's **simple, backbone-agnostic design** (SbVz and bPdS), its **practical motivation** (BV6f), and the **broad evaluation and informative ablations** (BV6f and bPdS). We are also grateful that SbVz highlighted the preservation of clean performance alongside improvements under corruptions and shifts.

In response to your constructive feedback, we have added the following experiments and clarifications to the revised manuscript. **New appendix material is highlighted in blue.**

- **Stronger ensemble comparisons.** We added native and random-context ensembles, learned static weighting, and established methods including DENSE, PHE, RF-PFN, and OligoICP. The comparisons cover TabICL and TabPFN classification and regression, with deployment budgets and adaptation settings specified (**Appendix E.1, Tables 16–19**).
- **Larger context budgets.** We extended the TabICL evaluation to **1,024, 4,096, and 8,192 context rows** for classification and regression, comparing Base, Shared, and Specific under clean data, corruptions, and shifts (**Appendix E.2, Tables 20–21**).
- **Generalization beyond matched mechanisms.** We added a **2×2 Shortcut ablation** separating calibration coverage from constructor matching, and evaluated **three unseen corruptions and three unseen shifts** without dedicated matching constructors (**Appendices E.3–E.4, Tables 22–24**).
- **Additional natural-shift evidence.** We expanded the Folktables comparisons with **DistPFN–TabICL-v2 and Drift-Resilient TabPFN**, reported repeated-run variability, and added clinical transfer evaluations on **PhysioNet Sepsis 2019 and UCI Diabetes 130-US Hospitals** (**Appendix E.5, Tables 25–27**).
- **Comparison with corruption-based fine-tuning.** We evaluated fine-tuning alone and combined it with Shared/Specific, reporting predictive performance and training overhead to examine the value of inference-time adaptation alongside backbone optimization (**Appendix E.6, Table 28**).
- **Failure-mode analysis and algorithm refinements.** We quantified high-confidence Base errors, single extreme regression predictions, and correlated branch errors, and evaluated extensions addressing these cases while retaining frozen backbones (**Appendix E.7, Tables 29–31**).
- **Clearer deployment and ablation details.** We expanded the context-category and branch-count descriptions and clarified candidate selection, policy fitting, and the scope of the weighting-term ablations (**Appendices B.4–B.5 and D.3**).

Our point-by-point responses below explain how these results address the individual concerns. We thank the reviewers again for helping us strengthen the empirical evidence and clarify the method, and welcome any further questions.
