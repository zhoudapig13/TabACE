## Part 2: Ensemble baselines and random-context accuracy

> TabACE is an ensemble method over TFMs, yet the paper does not compare it against established or even naive ensemble methods. A single forward pass with a randomly sampled context is a potentially unstable and weak baseline, especially with only 256 context rows. Thus, the reported gains may reflect generic ensemble benefits rather than TabACE’s specific contributions.
> Figure 5 reports latency for random contexts but not their accuracy. Could the authors report accuracy to enable a fair comparison?

Thank you for requesting these predictive comparisons. **Official Comment C2** reports Native-6, Random-6, and Convex Stacking-6, with the same six deployed predictions as Shared. Random-6 reaches **0.523/0.696 Acc/AUC** under corruptions and **0.764/0.828** under shifts; Shared reaches **0.575/0.720** and **0.772/0.831**. For regression, their corruption $R^2$/nRMSE is **−0.397/1.089** versus **0.363/0.783**.

Thus, Random-6 now provides Figure 5's missing accuracy counterpart, and the native/static/established ensemble comparisons show that **generic ensembling does not reproduce Shared's overall robustness**. Complete results, including DENSE [2], PHE [3], RF-PFN, and OligoICP [4], are in Appendix E.1, Tables 16–19 (pp. 36–37).

**References**

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[3] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[4] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.
