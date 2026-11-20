## Official Comment C2：原生集成与已有集成基线

我们补充了原生集成、随机上下文平均、静态学习权重及审稿人要求的 TFM 集成基线。**Native-6** 启用骨干原生的六估计器集成；**Random-6** 平均六个独立采样上下文的预测；**Convex Stacking-6** 在校准数据上学习和为一的非负固定权重。**Shared 使用一个 Base 和五个替代预测。** 这些集成部署时均使用六次预测；OligoICP 60→6 从 60 个候选中选取六个部署成员。

表中为分类 Acc/AUC 与回归 $R^2$/nRMSE。完整 Clean 结果、适配标记及候选／部署预算见**附录 E.1，表 16–19，第 36–37 页**，包括 DENSE [2]、PHE [3]、RF-PFN、OligoICP [4]、GES 及鲁棒聚合。

| 骨干 | 方法 | 扰动：分类 | 偏移：分类 | 扰动：回归 | 偏移：回归 |
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

**Shared 在扰动与偏移下的两项分类指标及回归 $R^2$ 上均优于表中集成基线。** nRMSE 也普遍更低，例外是 TabPFN Native-6 的偏移结果（Shared 为 0.656，Native-6 为 0.640）。原生和随机上下文集成在部分设置中有效，但无法复现 Shared 的整体鲁棒性，**仅增加骨干预测次数不能解释这些收益**。Random-6 补充了图 5 延迟对照中相应的预测性能。固定上下文库的融合分析见对 BV6f 的第 1 部分回复。

**参考文献**

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[3] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[4] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.
