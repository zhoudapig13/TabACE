# 相关工作（中文版）

## 2.1 表格基础模型与上下文依赖

TabPFN 将表格预测建模为基于合成先验的上下文学习，并通过带标签的训练样本直接预测测试查询；其后续版本进一步扩展了任务规模、特征处理和推理配置。TabICL 采用 column-wise、row-wise 和 dataset-wise 的分层注意力结构，将表格上下文学习扩展到更大数据集；TabDPT 则结合上下文检索、掩码自监督学习和真实表格预训练，以提高跨数据集泛化能力 \citep{muller2023tabpfn,hollmann2025tabpfn,qu2025tabicl,ma2025tabdpt}。近期研究还发现，context size、sampling 和 coverage 会显著影响 TabPFN 的预测精度与稳定性 \citep{abdullah2026contextsampling}。这些工作建立了表格基础模型的主要技术路线，但重点通常是预训练、模型结构与可扩展性，较少研究同一冻结模型如何协调不同 context 所提供的冲突或互补证据。

## 2.2 上下文压缩、检索与选择

一类工作直接优化进入模型的上下文。TuneTables 将大型训练集压缩为可学习的短上下文；LoCalPFN 为每个查询检索近邻样本，并利用检索到的局部上下文进行任务微调；MixturePFN 将训练数据划分为多个簇，为每个簇训练专门的 ICL expert，并将查询路由到相应 expert \citep{feuer2024tunetables,thomas2024localpfn,xu2025mixturepfn}。此外，Tab-AICL 分别使用不确定性、coreset 和二者结合的准则选择进入 context 的标注样本 \citep{treerath2026tabaicl}。

这些方法分别将 context construction 表述为压缩、检索、样本选择或路由问题，核心目标是提高上下文的规模效率或查询相关性。它们通常为一次预测确定一个主要 context 或 expert，因此性能依赖于特定的压缩、相似度或路由准则；当该准则与当前查询不匹配时，模型缺少利用其他 context 纠正预测的机制。

## 2.3 从同质上下文集成到异质上下文集成

另一类工作通过集成降低上下文敏感性。DENSE 将 demonstrations 划分为多个 buckets，分别预测后再聚合输出概率；LARA 将长 demonstrations 分组，并通过无梯度优化学习各组 logits 的组合权重 \citep{khalifa2023dense,huang2026lara}。在表格领域，TabPFN 集成不同的预处理、特征排列和标签排列；BETA 则结合 bootstrap sampling 与多个轻量级 encoders，同时降低模型偏差和预测方差 \citep{hollmann2025tabpfn,liu2025beta}。

这些方法证明了 context-level ensemble 的有效性，但其多样性主要来自同一构造规则下的重复采样、分组或等价输入变换。它们尚未明确研究具有不同归纳偏置的 context constructors 能否提供互补证据。TabACE 关注这一差异：它集成由异质且可扩展的 constructors 产生的 context-conditioned predictions，而不是只对同一种 context 进行重复采样。

## 2.4 上下文不稳定性与分布偏移

表格鲁棒性研究主要围绕预定义的 distribution shift 展开。BBSE 使用源域混淆矩阵估计目标域类别先验，Maximum-Likelihood Label Shift 则依赖经过偏差校正的概率来估计并修正标签比例变化 \citep{lipton2018bbse,alexandari2020labelshift}。近期预印本 DistPFN 将 posterior adjustment 引入表格上下文学习，在测试时修正标签先验失配 \citep{lee2026distpfn}；Drift-Resilient TabPFN 则利用包含时间变化的合成结构因果先验预训练模型，以处理 temporal distribution shift \citep{helli2024drifttabpfn}。在评价方面，TableShift 收集自然产生的跨域偏移任务，TabFSBench 系统研究缺失、增加或变化特征等 feature shifts \citep{gardner2023tableshift,cheng2025tabfsbench}。

这些方法分别针对标签、时间、自然域或特征偏移，但通常依赖预先规定的 shift 类型或相应假设。与此同时，即使训练集与测试集同分布，不同 context sampling 或 construction 也可能使同一查询产生不同预测。本文因此将问题定义为更一般的 context uncertainty：通过集成互补上下文缓解普通 context variability，并在明确设置的分布偏移下检验其鲁棒性，而不把 TabACE 表述为通用的 shift detector。
