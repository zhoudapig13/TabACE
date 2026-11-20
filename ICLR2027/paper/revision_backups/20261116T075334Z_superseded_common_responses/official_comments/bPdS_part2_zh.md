## 第 2 部分：集成基线与随机上下文的准确率

> TabACE 是一种基于 TFM 的集成方法，但论文没有与成熟的、甚至朴素的集成方法比较。随机采样上下文后仅进行一次前向传播，可能构成不稳定且较弱的基线，尤其是在只有 256 行上下文时。因此，报告的收益可能来自普通集成效应，而非 TabACE 的具体贡献。
> 图 5 报告了随机上下文的延迟，却没有报告准确率。作者能否补充准确率，以便公平比较？

感谢您要求补充预测性能。**Official Comment C2** 报告 Native-6、Random-6 与 Convex Stacking-6，均与 Shared 一样部署六次预测。Random-6 在扰动／偏移下的 **Acc/AUC 为 0.523/0.696／0.764/0.828**，Shared 为 **0.575/0.720／0.772/0.831**；回归扰动 $R^2$/nRMSE 分别为 **−0.397/1.089** 与 **0.363/0.783**。

Random-6 因而补齐图 5 缺失的准确率对照；原生／静态／已有集成对照进一步表明，**普通集成无法复现 Shared 的整体鲁棒性**。包含 DENSE [2]、PHE [3]、RF-PFN 与 OligoICP [4] 的完整结果见附录 E.1、表 16–19（第 36–37 页）。

**参考文献**

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[3] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[4] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.
