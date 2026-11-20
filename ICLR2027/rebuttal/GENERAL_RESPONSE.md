# General Response to Reviewers and the Area Chair

We thank the reviewers for carefully reading our paper and offering thoughtful suggestions. We appreciate your recognition of TabACE's **simple, backbone-agnostic design** (SbVz and bPdS), the **practical relevance of context sensitivity** (BV6f), and the **breadth of the evaluation and value of the ablations** (BV6f and bPdS). SbVz also noted that the method maintains clean performance while improving robustness under corruptions and shifts.

Your feedback guided additional comparisons, broader generalization tests, and a closer examination of the method's behavior. We summarize these updates below; **new appendix material is highlighted in blue.**

**Comparisons and context budgets**

- **Ensemble baselines** (SbVz, BV6f, bPdS). We now compare native and random-context ensembling, static learned weights, DENSE, PHE, RF-PFN, and OligoICP across TabICL/TabPFN classification and regression, specifying deployment budgets and adaptation settings (**Appendix E.1, Tables 16–19**).

- **Larger contexts** (bPdS). We evaluate Base, Shared, and Specific with **1,024, 4,096, and 8,192 context rows** under clean data, corruptions, and shifts for both tasks (**Appendix E.2, Tables 20–21**).

- **Fine-tuning** (SbVz). We test corruption-based fine-tuning alone and with Shared/Specific, reporting predictive performance and training overhead to assess their complementarity (**Appendix E.6, Table 28**).

**Generalization across mechanisms and domains**

- **Unseen mechanisms** (SbVz, BV6f, bPdS). A **2×2 Shortcut ablation** separates calibration coverage from constructor matching. We also evaluate **three unseen corruptions and three unseen shifts** without dedicated matching constructors (**Appendices E.3–E.4, Tables 22–24**).

- **Natural and clinical shifts** (SbVz, BV6f, bPdS). We extend Folktables comparisons to **DistPFN–TabICL-v2 and Drift-Resilient TabPFN**, report repeated-run variability, and evaluate clinical transfer on **PhysioNet Sepsis 2019 and UCI Diabetes 130-US Hospitals** (**Appendix E.5, Tables 25–27**).

**Method analysis and implementation details**

- **Failure modes and refinements** (BV6f, bPdS). We measure the frequency and impact of high-confidence Base errors, extreme regression predictions, and correlated branch errors, and evaluate improvements with frozen backbones (**Appendix E.7, Tables 29–31**).

- **Deployment and ablations** (BV6f, bPdS). We detail context categories, branch counts, candidate selection, and policy fitting, and clarify which weighting terms each ablation removes (**Appendices B.4–B.5 and D.3**).

We provide point-by-point responses to each review below. We believe these revisions address most of the concerns raised, and we thank the reviewers for their constructive suggestions and welcome further discussion. We would also appreciate the Area Chair’s assessment of how the additional results and clarifications address the main concerns raised in the reviews.
