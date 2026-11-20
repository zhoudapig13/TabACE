## Official Comment C2: Native and established ensemble baselines

We added native ensembling, random-context averaging, learned static weighting, and the requested TFM ensemble baselines. **Native-6** enables the backbone's native six-estimator ensemble; **Random-6** averages six independently sampled contexts; **Convex Stacking-6** fits nonnegative calibration weights summing to one. **Shared uses one Base and five alternative predictions.** These deployed ensembles use six predictions per query; OligoICP 60→6 selects six deployed members from 60 candidates.

The table reports classification Acc/AUC and regression $R^2$/nRMSE. Complete clean results, adaptation markers, and candidate/deployment budgets appear in **Appendix E.1, Tables 16–19, pp. 36–37**, including DENSE [2], PHE [3], RF-PFN, OligoICP [4], GES, and robust aggregation.

| Backbone | Method | Corruptions: classification | Shifts: classification | Corruptions: regression | Shifts: regression |
|---|---|---|---|---|---|
| TabICL-v2 | Base | 0.494 / 0.676 | 0.763 / 0.826 | −1.077 / 1.270 | 0.273 / 0.730 |
| TabICL-v2 | Native-6 | 0.507 / 0.683 | 0.763 / 0.825 | −0.597 / 1.145 | 0.287 / 0.721 |
| TabICL-v2 | Random-6 | 0.523 / 0.696 | 0.764 / 0.828 | −0.397 / 1.089 | 0.351 / 0.702 |
| TabICL-v2 | DENSE-6 | 0.531 / 0.695 | 0.764 / 0.826 | — | — |
| TabICL-v2 | Convex Stacking-6 | 0.559 / 0.704 | 0.766 / 0.827 | −0.446 / 1.104 | 0.401 / 0.682 |
| TabICL-v2 | Shared | **0.575 / 0.720** | **0.772 / 0.831** | **0.363 / 0.783** | **0.467 / 0.659** |
| TabPFN-v3 | Base | 0.457 / 0.636 | 0.758 / 0.809 | −1.253 / 1.313 | 0.426 / 0.669 |
| TabPFN-v3 | Native-6 | 0.467 / 0.637 | 0.766 / 0.814 | −0.405 / 1.116 | 0.444 / 0.640 |
| TabPFN-v3 | PHE-6 | 0.464 / 0.626 | 0.763 / 0.816 | −0.387 / 1.111 | 0.424 / 0.659 |
| TabPFN-v3 | OligoICP 60→6 | — | — | −0.370 / 1.120 | 0.333 / 0.703 |
| TabPFN-v3 | Shared | **0.553 / 0.697** | **0.770 / 0.817** | **0.237 / 0.834** | **0.460 / 0.656** |

**Shared improves both classification metrics and regression $R^2$ over the displayed ensembles under corruptions and shifts.** It also has lower nRMSE throughout except against TabPFN Native-6 on shifts (0.656 versus 0.640). Native and random-context ensembles help in several settings, but do not reproduce Shared's overall robustness; **additional backbone predictions alone do not explain the gains**. Random-6 supplies the predictive-performance counterpart to Figure 5's latency comparison. The fusion-specific fixed-bank analysis is addressed separately in our response to BV6f, Part 1.

**References**

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[3] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[4] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.
