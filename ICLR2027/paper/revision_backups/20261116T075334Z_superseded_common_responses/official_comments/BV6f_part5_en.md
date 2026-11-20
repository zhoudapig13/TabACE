## Part 5: Calibration protocol, sample sizes, and uncertainty

> Which datasets and conditions are used to calibrate these policies? Please provide the evaluation sample sizes and uncertainty estimates for the reported gains.

**Calibration excludes the evaluated dataset**, using five-fold dataset-level OOF. Episodes cover registered mechanisms at strengths $\{0,0.2,0.5,0.8,1.0\}$; member selection and policy parameters are frozen before held-out evaluation. There are **72/35 classification/regression datasets for corruptions and 35/33 for shifts**. Classification uses 128 evaluation rows per dataset; regression uses up to 512 for corruptions and 256 for shifts (Appendix B.1.1, Table 4).

**Synthetic response curves include 95% bootstrap confidence intervals** from 10,000 resamples of complete seed–DGP–scenario units (Appendix B.1.2). Three unseen confirmation seeds × five DGPs × twelve non-clean families give **180 aligned units**; each nonzero strength has 75 corruption or 105 shift units. Whole-unit resampling preserves within-experiment dependence. Folktables seed variability is summarized in **C3** and Appendix E.5.3.

We thank you for helping us strengthen the comparisons, deployment details, and failure-mode analysis. If these additional results and clarifications resolve your main concerns, we would greatly appreciate your consideration of a more favorable score.
