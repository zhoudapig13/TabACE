# TabACE 相关工作完整调研

> 调研截止日期：2026-08-03  
> 当前方法：**TabACE — Tabular Adaptive Context Ensemble**  
> 当前标题：**TabACE: Adaptive Ensembling of Configurable Contexts for Reliable Prediction with Tabular Foundation Models**

## 1. 调研范围与方法边界

本文将当前方法理解为：给定一个冻结的表格基础模型（TFM）和同一个查询，构造多个具有不同归纳偏置或信息来源的 context，分别得到 context-induced predictions，再在预测空间进行聚合。目标不是训练多个模型，而是保留不同 context 中稳定、互补的任务信号，并抑制单一 context 携带的偶然噪声、伪相关或错误先验。

当前实现包含两类场景：

1. **Feature/context shift**：KNN、prototype、boundary、subspace、clean historical 等 context view；
2. **Label-prior shift**：由 foundation model、logistic regression、15-NN、cross-validated foundation estimates 等产生多种 prior evidence，再进行组合或后验修正。

以下文献按与该机制的接近程度划分，而不是简单罗列所有表格学习或 ensemble 论文。

## 2. 结论先行

### 2.1 尚未发现完全相同的工作

截至调研日期，没有发现一篇论文同时满足以下五点：

1. 研究对象是 tabular foundation model；
2. 基座模型参数保持冻结；
3. 对同一个查询构造多个**异质且有明确语义**的 context；
4. 每个 context 独立诱导预测，随后在输出层聚合；
5. 目标明确指向 feature/context shift 与 label-prior shift 下的鲁棒性。

因此，TabACE 仍有可以成立的组合性贡献。但是，不应笼统声称“首个 context ensemble”或“首个在 TFM 中集成多个 context 的方法”，因为下面几类工作已经覆盖了其中的大块机制。

### 2.2 最接近且必须正面比较的工作

| 优先级 | 工作 | 与 TabACE 重合的部分 | 关键差异 | Novelty 风险 |
|---|---|---|---|---|
| S | [TabPFN Unleashed / BETA](https://arxiv.org/abs/2502.02527) | 通过多个 bootstrap support sets 构成不同 context，并用 bagging 降低方差 | 同时训练轻量 encoder/部分参数；context 主要是随机 bootstrap，不是面向 shift 的异质 evidence views | **最高** |
| S | [DENSE: Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780) | 把 demonstrations 分成多个 bucket，分别推理，再以 MoE/PoE/max 聚合概率 | 面向语言模型，不是表格 TFM，也不研究 tabular shift 或结构化 context 构造 | **最高，机制先例** |
| S | [LARA: Divide, Reweight, and Conquer](https://aclanthology.org/2026.eacl-long.11/) | 多个 demonstration groups 独立推理并在 logit 层加权组合 | 面向 LLM many-shot ICL，主要解决上下文长度和内存，不研究表格分布偏移 | **高，聚合先例** |
| S | [MixturePFN](https://arxiv.org/abs/2405.16156)（ICLR 2025） | 将大表拆成多个局部 context/prompter | 通常将查询路由到一个局部专家/context，并进行 context-aware fine-tuning；不是让每个查询跨多个异质 context 集成 | **高，名称与局部 context 接近** |
| S | [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6) 及其 post-hoc ensemble | 同一 TabPFN 多次推理后平均；使用不同预处理、feature/class permutations | 多样性主要来自等价变换和预处理配置，而非证据 context 的语义差异或 shift-specific construction | **高，必须控制内置 ensemble** |
| A+ | [Understanding Context Sampling in TabPFN](https://arxiv.org/abs/2607.26628) | 直接研究 context size、随机重复采样、稳定性、覆盖度与 diversity | 以分析为主；没有提出 shift-aware heterogeneous context ensemble | **很新且高度相关** |
| A+ | [Uncertainty-Guided Model Selection for TFMs](https://openreview.net/forum?id=7hIc6bK7tR) | 对不同数据子集上的 TabPFN 预测进行选择和平均，强调 context 敏感性 | 生物分子回归应用；使用不确定性选择 subset-model，范围较窄，非统一 shift 框架 | **高，但属于 workshop/application** |
| A | [Sub-CP](https://arxiv.org/abs/2510.05130) | 将长 context 分块，并显式控制不同块的 diversity，再配合 DENSE 等聚合 | LLM 场景；目标是覆盖与压缩，不是 tabular shift | 中高 |
| A | [Parallel-ICL](https://arxiv.org/abs/2603.16092) | context chunking、并行独立推理、query-relevance weighting、logit-level PoE | 视觉语言模型；核心目标是效率 | 中高 |
| A | [TabFM-Ensemble](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) | 同一 TFM 在 cross features、SVD features 等 32 个配置下集成，学习非负权重 | 属于 feature/preprocessing ensemble；不是对训练 evidence context 的鲁棒组合 | 中高 |
| A | [A Closer Look at TabPFN v2](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c57b3718381cb2bf9c1ccd63377f2448-Abstract-Conference.html) | 分析 TabPFN v2，并系统研究 post-hoc ensemble size | 主要是等价配置/模型输出集成与 scalability，不是多类 context evidence | 中 |
| A | [LoCalPFN](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html) | query-specific KNN context，证明 context construction 很重要 | 每个查询选择单个局部 context，并伴随 fine-tuning；不聚合多种 context | 中 |
| A | [TAAR](https://openreview.net/pdf?id=qBMNsGiE3u) | 用模型注意力选择 task-aligned features 和 context rows | 仍是“寻找一个更好的 context”，而不是保留多个 context-induced predictors | 中 |
| A | [CRUMB](https://arxiv.org/abs/2606.11473) | 用 MMD 为 query cluster 构造 distribution-matched context，对 covariate drift 有韧性 | 每个 query cluster 使用一个缩减 context；目标首先是可扩展推理 | 中 |
| A | [DistPFN](https://arxiv.org/abs/2605.04363) | 直接处理 TFM 对 context label prior 的敏感性；测试时后验调整 | 单一路径的 posterior correction，而非多 context ensemble；ICLR 版本曾撤稿，当前为 arXiv | **label-shift 必须比较** |

### 2.3 最安全的贡献定位

建议把贡献限定为：

> We study **heterogeneous context ensembling for frozen tabular foundation models under distribution shift**. Unlike context selection or routing methods that commit to a single support set, and unlike bootstrap or permutation ensembles that obtain largely homogeneous perturbations, TabACE preserves multiple semantically distinct context-induced predictors and aggregates their stable evidence.

中文含义是：本文研究的是**分布偏移下冻结表格基础模型的异质 context ensemble**。与只选择或路由到一个 support set 的方法不同，也与主要依靠随机 bootstrap 或等价排列得到同质扰动的 ensemble 不同，TabACE 保留多个语义不同的 context 所诱导的预测，并聚合其中稳定的证据。

不建议在没有进一步检索证明时使用：

- “the first context ensemble method”；
- “the first method to ensemble contexts for ICL”；
- “the first multi-context TFM”；
- “the first robust TFM under distribution shift”。

可考虑使用更窄的 first claim：

> To our knowledge, this is the first systematic study of **heterogeneous, inference-time context ensembling across both feature/context and label-prior shifts for frozen tabular foundation models**.

即便采用这句话，也建议写成 “to our knowledge”，并在投稿前再次检索 2026 年下半年的新论文。

## 3. 表格方向：Context 构造、选择、压缩与集成

### 3.1 表格 ICL 与基础模型主干

这些论文用于说明“训练表本身就是推理 context”，以及实验所用基座模型的来源。

1. **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second**，Müller et al.，ICLR 2023。PFN/表格 ICL 的奠基工作；模型以完整训练表作为 context。[OpenReview](https://openreview.net/forum?id=cp5PvcI6w8_)
2. **Accurate Predictions on Small Data with a Tabular Foundation Model**，Hollmann et al.，Nature 2025。TabPFN v2；官方推理已经包含多种预处理和排列配置的预测平均，因此实验必须说明是否关闭或固定该层 ensemble。[Nature](https://www.nature.com/articles/s41586-024-08328-6)
3. **TabICL: A Tabular Foundation Model for In-Context Learning on Large Data**，Qu et al.，ICML 2025。通过 column-then-row 架构扩展 context 规模。[PMLR](https://proceedings.mlr.press/v267/qu25d.html)
4. **TabDPT: Scaling Tabular Foundation Models on Real Data**，Ma et al.，NeurIPS 2025。结合 ICL retrieval 与真实表上的自监督预训练；也是当前实验的核心 backbone。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)
5. **Mixture of In-Context Prompters for Tabular PFNs (MixturePFN)**，Xu et al.，ICLR 2025。将数据分簇成多个局部 prompters，并把查询路由到相关专家。[OpenReview](https://openreview.net/forum?id=2fojNANZSv)
6. **TabPFN-2.5 Model Report**，2025。扩展数据规模和特征数；如果最终实验使用 2.5/2.6/3，需要记录准确 checkpoint 与 ensemble 配置。[arXiv](https://arxiv.org/abs/2511.08667)
7. **TabFM: A Zero-Shot Foundation Model for Tabular Data**，Google Research，2026。混合 TabPFN/TabICL 式架构；其 32-way TabFM-Ensemble 使用交叉特征、SVD 特征、非负最小二乘权重与 Platt scaling。[官方说明](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
8. **MITRA: Mixed Synthetic Priors for Tabular Foundation Models**，NeurIPS 2025。通过混合合成 prior 扩大基础模型归纳偏置，可用于论证一个固定 TFM 已内化多种先验，但不是推理时 context ensemble。[NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/177d68f4adef163b7b123b5c5adb3c60-Paper-Conference.pdf)
9. **Large Scale Transfer Learning for Tabular Data via Language Modeling (TabuLa-8B)**，NeurIPS 2024。LLM-based tabular few-shot baseline，适合作为 TFM 版图背景，不是直接竞争者。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html)
10. **Position: Why Tabular Foundation Models Should Be a Research Priority**，van Breugel & van der Schaar，ICML 2024。概念性背景，强调跨表 contextualization。[OpenReview](https://openreview.net/forum?id=amRSBdZlw9)

### 3.2 单一 context 的选择、检索与压缩

这一组工作的共同点是把 context 当作可优化对象，但通常最终仍提交一个 context。

1. **Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks**，Feuer et al.，2023。系统比较 context summarization、sketching 与 feature selection。[arXiv](https://arxiv.org/abs/2311.10609)
2. **In-Context Data Distillation with TabPFN**，Ma et al.，ICLR ME-FoMo Workshop 2024。直接优化一个压缩 context，使 TabPFN 在固定 memory budget 下处理大表。[arXiv](https://arxiv.org/abs/2402.06971)
3. **TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks**，Feuer et al.，NeurIPS 2024。把大数据压缩成 learned context；核心是 context optimization，不是 ensemble。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97dc07f1253ab33ee514f395a82fa7cc-Abstract-Conference.html)
4. **Retrieval & Fine-Tuning for In-Context Tabular Models (LoCalPFN)**，Thomas et al.，NeurIPS 2024。为每个 query 检索 KNN context，并做 task-specific fine-tuning。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html)
5. **Towards Localization via Data Embedding for TabPFN**，Koshil et al.，TRL@NeurIPS 2024。学习 embedding 后做局部最近邻 context selection，并分析 bias-variance trade-off。[OpenReview](https://openreview.net/forum?id=LFyQyV5HxQ)
6. **MixturePFN**，Xu et al.，ICLR 2025。K-means 划分局部 contexts，路由 query，配合 context-aware fine-tuning。[OpenReview](https://openreview.net/forum?id=2fojNANZSv)
7. **TabDPT**，Ma et al.，NeurIPS 2025。将 retrieval 融入 TFM 的预训练和推理机制。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)
8. **Task-Aligned Attention Retrieval for Scaling Tabular Foundation Models (TAAR)**，ICLR 2026 submission。用 TFM 自身 attention 选择 task-relevant features 和 rows，而非依赖原始欧氏距离。[OpenReview PDF](https://openreview.net/pdf?id=qBMNsGiE3u)
9. **CRUMB: Efficient PFN Inference via Distributionally Matched Context Batching**，Heredge et al.，arXiv 2026。对测试查询分簇，以 MMD 选择匹配训练子集；直接报告 covariate drift 韧性。[arXiv](https://arxiv.org/abs/2606.11473)
10. **Retrieval-aligned TFMs Enable Robust Clinical Risk Prediction... (AWARE)**，Pham et al.，arXiv 2026。通过 supervised embedding 与轻量 adapter 学习 task-aligned retrieval，说明错误检索会在异质、长尾和 shift 下放大不稳定性。[arXiv](https://arxiv.org/abs/2604.01841)
11. **Active In-Context Learning for Tabular Foundation Models**，Treerath & Pittorino，arXiv 2026。将 labeled context 视为主动学习直接优化的对象，比较 uncertainty、coreset 与 hybrid selection。[arXiv](https://arxiv.org/abs/2603.27385)
12. **Understanding Context Sampling in TabPFN on Small Tabular Datasets**，Abdullah，arXiv 2026。重复 context sampling 表明较大 context 提升稳定性；coverage/diversity 比仅匹配 feature means 更能解释准确率。[arXiv](https://arxiv.org/abs/2607.26628)
13. **Context-Constrained Transfer Learning for TFMs via Data Distillation (TL-ANDI)**，2026，under review。按 covariate/posterior shift 选 anchor 并做局部 label smoothing；目前审稿状态较早，只宜作为最新并行工作。[OpenReview](https://openreview.net/forum?id=35LXG0a3Y6)

### 3.3 表格领域中真正进行 aggregation 的工作

1. **TabPFN 内置 heterogeneous ensemble / post-hoc ensemble**。TabPFN 通过 feature order、class order、preprocessor 等配置反复推理并平均；v2 论文明确采用 4-way（回归 8-way）ensemble。[Nature](https://www.nature.com/articles/s41586-024-08328-6)
2. **A Closer Look at TabPFN v2**，Ye et al.，NeurIPS 2025。报告不同 post-hoc ensemble size 对性能的影响，并研究 test-time divide-and-conquer。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c57b3718381cb2bf9c1ccd63377f2448-Abstract-Conference.html)
3. **TabPFN Unleashed / BETA**，Liu & Ye，2025。多个 latent encoders 加上 bootstrapped support sets；这是表格域中与“多 context 后聚合”最接近的工作。[arXiv](https://arxiv.org/abs/2502.02527)
4. **Uncertainty-Guided Model Selection for TFMs in Biomolecule Efficacy Prediction**，Li et al.，NeurIPS 2025 FM4LS Workshop。对不同数据子集上的 TabPFN predictor 做 label-free selection 和 averaging。[OpenReview](https://openreview.net/forum?id=7hIc6bK7tR)
5. **TabFM-Ensemble**，Google Research，2026。对 feature engineering views 做 32-way weighted ensemble。[官方说明](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
6. **Ensembling Tabular Foundation Models — A Diversity Ceiling and a Calibration Trap**，Tanna et al.，arXiv 2026。在 153 个任务上组合六个不同 TFM，发现不同 TFM 的错误高度相关，复杂 stacking 收益很小且可能破坏 calibration。[arXiv](https://arxiv.org/abs/2605.18696)
7. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**，Gorishniy et al.，ICLR 2025。单个网络产生多个预测的高效 ensemble；不是 TFM context ensemble，但属于强 tabular ensemble baseline。[arXiv](https://arxiv.org/abs/2410.24210)
8. **TabArena: A Living Benchmark for Machine Learning on Tabular Data**，Erickson et al.，NeurIPS 2025。说明 configuration ensembling 和 cross-model ensembling 对表格 benchmark 结论影响很大。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
9. **Make Still Further Progress: Chain of Thoughts for Tabular Data Leaderboard**，Liu et al.，arXiv 2025。LLM 根据邻居 context 动态集成多个外部表格模型的预测；这里 ensemble 的是模型输出，LLM 不读取原始特征。[arXiv](https://arxiv.org/abs/2505.13421)
10. **Of Graphs and Tables: Zero-Shot Node Classification with TFMs (TabGFM)**，Hayler et al.，arXiv 2025。多个 TFM 处理不同 subsampled tables 后通过 ensemble selection 聚合；任务是图节点分类，且使用多个 TFM。[arXiv](https://arxiv.org/abs/2509.07143)

## 4. Context / Prompt / Demonstration Ensemble 方向

### 4.1 与 TabACE 数学结构最相近

1. **Exploring Demonstration Ensembling for In-context Learning (DENSE)**，Khalifa et al.，ME-FoMo@ICLR 2023 / arXiv 2023。把 demonstrations 分入 buckets，分别计算 \(p(y\mid B_k,x)\)，再以 mixture-of-experts、product-of-experts 或 max 聚合。这是 TabACE 最直接的跨模态方法先例。[arXiv](https://arxiv.org/abs/2308.08780)
2. **Few-shot Reranking for Multi-hop QA via Language Model Prompting**，Khalifa et al.，ACL 2023。在 prompt 长度不足时对不同 demonstration prompts 独立打分，并用 mean/max ensemble。[ACL Anthology](https://aclanthology.org/2023.acl-long.885/)
3. **Divide, Reweight, and Conquer: A Logit Arithmetic Approach for In-Context Learning (LARA)**，Huang et al.，EACL 2026。把长 demonstrations 分组并行处理，通过无梯度优化学习 logit 权重；B-LARA 进一步选择 group。[ACL Anthology](https://aclanthology.org/2026.eacl-long.11/)
4. **Submodular Context Partitioning and Compression for In-Context Learning (Sub-CP)**，Zheng et al.，arXiv 2025。通过 submodular objective 控制不同 context blocks 的局部一致性和全局 diversity，可与 DENSE 的 MoE/PoE/max 结合。[arXiv](https://arxiv.org/abs/2510.05130)
5. **Parallel In-Context Learning for Large Vision-Language Models**，Yamaguchi et al.，arXiv 2026。分块 context、并行推理，以 query relevance 加权的 product-of-experts 聚合。[arXiv](https://arxiv.org/abs/2603.16092)
6. **Ask Me Anything: A Simple Strategy for Prompting Language Models**，Arora et al.，ICLR 2023。自动构造多个不完美 prompts，再以 weak supervision 建模它们的准确性和相关性进行聚合。[OpenReview](https://openreview.net/forum?id=bhUPJnS2g0X)
7. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**，Wang et al.，ICLR 2023。固定模型和问题，采样多条 reasoning paths，通过 marginalization/majority vote 获得稳定答案；多样性来自推理路径而不是 context。[OpenReview](https://openreview.net/forum?id=1PL1NIMMrw)
8. **Explanation-aware Soft Ensemble Empowers LLM In-context Learning (EASE)**，Yu et al.，arXiv 2023。用 explanation reliability 和 soft probability aggregation 组合多个 ICL outputs。[arXiv](https://arxiv.org/abs/2311.07099)
9. **In-Context Ensemble Learning from Pseudo Labels Improves Video-Language Models...**，Xu et al.，NeurIPS 2024 Workshop。聚合多条 pseudo-labeled SOP paths，并以 consistency regularization 扩展有效 context。[arXiv](https://arxiv.org/abs/2409.15867)
10. **SegGPT: Towards Segmenting Everything in Context**，Wang et al.，ICCV 2023。提出 spatial ensemble 与 feature ensemble 来融合多个视觉 reference contexts；聚合发生在输入拼接或中间特征层，而非最终概率层。[CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.pdf)

### 4.2 证明单一 context 容易带入偏差或干扰

1. **Calibrate Before Use: Improving Few-shot Performance of Language Models**，Zhao et al.，ICML 2021。prompt format、examples 和顺序可导致从随机水平到接近 SOTA 的巨大变化；contextual calibration 可降低偏差。[PMLR](https://proceedings.mlr.press/v139/zhao21c.html)
2. **Fantastically Ordered Prompts and Where to Find Them**，Lu et al.，ACL 2022。仅改变 demonstrations 的顺序即可产生极大性能差异。[ACL Anthology](https://aclanthology.org/2022.acl-long.556/)
3. **Learning To Retrieve Prompts for In-Context Learning**，Rubin et al.，NAACL 2022。学习 query-specific demonstration retriever，证明示例选择对 ICL 性能至关重要。[ACL Anthology](https://aclanthology.org/2022.naacl-main.191/)
4. **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?**，Min et al.，EMNLP 2022。demonstrations 传递 label space、input distribution 和格式等多种信息，支持“context 同时携带有效信号和附加信息”的动机。[ACL Anthology](https://aclanthology.org/2022.emnlp-main.759/)
5. **How Many Demonstrations Do You Need for In-context Learning?**，Chen et al.，EMNLP Findings 2023。区分 positive/negative demonstrations，并观察到多 demo 之间的 interference 和 spurious correlation 可能降低性能。[OpenReview](https://openreview.net/forum?id=JPUx2nVgWa)
6. **Mixture of Demonstrations for In-Context Learning**，Wang et al.，NeurIPS 2024。将 demonstration pool 分组，由多个 retriever experts 协作选择最终 prompt；是 selection/retrieval，不是输出 ensemble。[OpenReview](https://openreview.net/forum?id=uqxSLoCw3K)
7. **MEAL: Stable and Active Learning for Few-Shot Prompting**，Köksal et al.，EMNLP Findings 2023。使用 multiprompt fine-tuning 与 prediction ensembling 降低 few-shot run variability。[ACL Anthology](https://aclanthology.org/2023.findings-emnlp.36/)

## 5. 分布偏移与 TFM 鲁棒性

### 5.1 表格数据的 shift benchmark 与 TFM 评估

1. **TableShift: Benchmarking Distribution Shift in Tabular Data**，Gardner et al.，NeurIPS 2023。15 个真实任务，提供 ID/OOD splits，是自然 shift 实验的重要补充。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html)
2. **TabFSBench: Tabular Benchmark for Feature Shifts in Open Environments**，Cheng et al.，ICML 2025。系统定义四类 tabular feature shift；非常适合规范当前“非 label-prior shift”实验的命名。[PMLR](https://proceedings.mlr.press/v267/cheng25e.html)
3. **RLSbench: Domain Adaptation Under Relaxed Label Shift**，Garg et al.，ICML 2023。包含视觉、文本和表格的 500+ shift pairs，同时考虑 label marginal 与 class-conditional shift。[PMLR](https://proceedings.mlr.press/v202/garg23a.html)
4. **Empirical Evaluation of OOD Performance of Tabular Foundation Models**，Loza et al.，arXiv 2026。比较九个 TFM 在 TableShift 上的 OOD 表现，所有模型均出现系统性退化。[arXiv](https://arxiv.org/abs/2607.26000)
5. **Are TFMs Robust to Realistic Query Distribution Shifts in Microbiome Data?**，Perciballi et al.，arXiv 2026。保持 support context 不变，仅对 queries 施加真实 feature perturbations，直接对应 support-query mismatch。[arXiv](https://arxiv.org/abs/2606.24995)
6. **Robust Tabular Foundation Models**，Peroni et al.，arXiv 2025。以 adversarial synthetic generator 和 fine-tuning 提高模型本身的鲁棒性；与 TabACE 的 frozen-model inference wrapper 不同。[arXiv](https://arxiv.org/abs/2512.03307)
7. **Robustness is Important: Limitations of LLMs for Predictions on Tabular Data**，PNAS Nexus 2026。研究包括 TabPFN/LimiX 在内的模型对 task-irrelevant variations 的敏感性，并讨论通过多次修改输入后平均来降低敏感性。[期刊页面](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520)

### 5.2 表格 test-time adaptation

1. **AdapTable: Test-Time Adaptation for Tabular Data via Shift-Aware Uncertainty Calibrator and Label Distribution Handler**，Kim et al.，TRL@NeurIPS 2024。先做 uncertainty calibration，再估计并调整 target label distribution。[OpenReview](https://openreview.net/forum?id=yhD2icebZ0)
2. **TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules**，Ren et al.，ICML 2024。假设规则结构在 shift 下不变，只适配数值参数和权重。[OpenReview](https://openreview.net/forum?id=LZeixIvQcB)
3. **ODS: Test-Time Adaptation in the Presence of Open-World Data Shift**，Zhou et al.，ICML 2023。将 covariate 与 label distribution shifts 解耦处理。[PMLR PDF](https://proceedings.mlr.press/v202/zhou23e/zhou23e.pdf)
4. **LLM Embeddings Improve Test-Time Adaptation to Tabular \(Y\mid X\)-Shifts**，Zeng et al.，2024。以 LLM 表征缓解 missing-variable 引起的 posterior/concept shift。[OpenReview](https://openreview.net/forum?id=OyjMJjfhiw)
5. **DistPFN: Mitigating Label Shift in Tabular ICL via Test-Time Posterior Adjustment**，Lee et al.，arXiv 2026。TFM label-prior shift 的最直接 baseline。[arXiv](https://arxiv.org/abs/2605.04363)

### 5.3 Label-shift 基础方法

这些论文主要是 label-prior context track 的理论与 baseline 来源。

1. **Detecting and Correcting for Label Shift with Black Box Predictors (BBSE)**，Lipton et al.，ICML 2018。用黑盒预测器的 confusion matrix 估计 target prior。[PMLR](https://proceedings.mlr.press/v80/lipton18a.html)
2. **Regularized Learning for Domain Adaptation under Label Shifts (RLLS)**，Azizzadenesheli et al.，ICLR 2019。对小样本和大 shift 下的 importance weights 做正则估计。[OpenReview](https://openreview.net/forum?id=rJl0r3R9KX)
3. **Maximum Likelihood with Bias-Corrected Calibration Is Hard-To-Beat at Label Shift Adaptation**，Alexandari et al.，ICML 2020。校准后的 maximum-likelihood/EM 是重要强 baseline。[PMLR](https://proceedings.mlr.press/v119/alexandari20a.html)
4. **A Unified View of Label Shift Estimation**，Garg et al.，NeurIPS 2020。统一 BBSE 与 maximum-likelihood label-shift estimation，并分析 calibration error。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html)
5. **Adapting to Online Label Shift with Provable Guarantees**，Bai et al.，NeurIPS 2022。处理随时间变化的 label prior，并提出 online ensemble algorithms。[OpenReview](https://openreview.net/forum?id=dC_Cho7PzT)

## 6. Ensemble 与 multi-view 理论来源

1. **Bagging Predictors**，Breiman，Machine Learning 1996。通过 bootstrap training sets 构造 predictor variants 并聚合；BETA 与随机 context ensemble 的直接经典来源。[DOI](https://doi.org/10.1023/A:1018054314350)
2. **The Random Subspace Method for Constructing Decision Forests**，Ho，TPAMI 1998。通过 feature subsets 制造多样性；与 TabACE 的 subspace context 有直接概念联系。[DOI](https://doi.org/10.1109/34.709601)
3. **Feature Bagging for Outlier Detection**，Lazarevic & Kumar，KDD 2005。在不同特征子空间运行相同算法并聚合结果。[DOI](https://doi.org/10.1145/1081870.1081891)
4. **Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles**，Lakshminarayanan et al.，NeurIPS 2017。说明多样 predictor 的平均可改善 calibration 与 OOD uncertainty。[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
5. **Diversity and Generalization in Neural Network Ensembles**，Ortega et al.，AISTATS 2022。系统联系 ensemble diversity 与 generalization error，覆盖平方损失、交叉熵和 0-1 loss。[PMLR](https://proceedings.mlr.press/v151/ortega22a.html)
6. **Calibrated Ensembles Can Mitigate Accuracy Tradeoffs under Distribution Shift**，Kumar et al.，UAI 2022。组合 standard 与 robust predictors，在 ID/OOD 之间取得更好折中。[OpenReview](https://openreview.net/forum?id=S2EuSP8sqgc)
7. **Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift**，To et al.，ICML 2025。鼓励 prototype classifiers 关注不同模式，在未知 subgroup shift 下提高鲁棒性。[OpenReview](https://openreview.net/forum?id=qUTiOeM57J)
8. **A Survey on Multi-view Learning**，Xu et al.，2013。以 consensus 与 complementarity 两条原则解释多视图学习；可支撑“稳定信号 + 互补信息”的语言，但不应当作直接算法先例。[arXiv](https://arxiv.org/abs/1304.5634)
9. **Wisdom of the Contexts: Active Ensemble Learning for Contextual Anomaly Detection**，Calikus et al.，2021。认为不存在对所有 anomaly 都完美的单一 context，自动构造多个 context 并赋予重要性。[arXiv](https://arxiv.org/abs/2101.11560)
10. **Context-based Ensemble Method for Human Energy Expenditure Estimation**，Gjoreski et al.，Applied Soft Computing 2015。提出 Multiple Contexts Ensemble；“context ensemble”这一名称已有跨领域使用。[DOI](https://doi.org/10.1016/j.asoc.2015.05.001)

## 7. 与 TabACE 的精确差异矩阵

| 方法 | 固定同一基座 | 多个 context | 同一 query 跨 context | 输出聚合 | 异质语义 view | 显式研究 shift |
|---|---:|---:|---:|---:|---:|---:|
| TabACE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TabPFN PHE | ✓ | 部分 | ✓ | ✓ | 否，主要是排列/预处理 | 否 |
| BETA | 部分（含适配） | ✓ | ✓ | ✓ | 否，主要是 bootstrap | 否 |
| MixturePFN | 否/含微调专家 | ✓ | 通常否，主要路由 | 有 mixture 结构 | 局部簇 | 否 |
| LoCalPFN | 否/含 fine-tuning | 单个 query-specific | 否 | 否 | KNN local | 否 |
| TAAR | ✓ | 候选很多、最终选一个 | 否 | 否 | attention-aligned | 部分 |
| CRUMB | ✓ | 每个 query cluster 一个 | 否 | 否 | distribution matching | covariate drift |
| DENSE | ✓ | ✓ | ✓ | ✓ | demonstration buckets | 否，且非 tabular |
| LARA | ✓ | ✓ | ✓ | ✓ | demonstration groups | 否，且非 tabular |
| Sub-CP | ✓ | ✓ | ✓ | 依赖所接框架 | diversity-controlled blocks | 否，且非 tabular |
| TabFM-Ensemble | ✓ | 间接 | ✓ | ✓ | feature transforms | 否 |
| DistPFN | ✓ | 否 | 否 | posterior adjustment | prior evidence 单路径 | label shift |

## 8. 审稿人最可能提出的质疑

### 8.1 “这是不是 BETA 的 bagging support sets？”

必须回答：BETA 的 diversity 主要来自 bootstrap sampling，并以降低一般性 variance 为目标；TabACE 的 context 是有类型的 evidence interventions，分别强调 locality、class prototypes、decision boundary、feature subspace、historical anchor 或 prior estimates，并专门评估不同 shift family。实验上需要加入 **random bootstrap context ensemble**，在相同 context 数、每个 context 大小和总 forward-pass budget 下比较。

### 8.2 “这是不是把 DENSE 从 NLP 搬到表格？”

不能否认 DENSE 给出了通用的“拆 demonstrations、分别预测、聚合”的结构先例。贡献应放在：

- tabular ICL 中 context 表示训练分布而非文本示例序列；
- context construction 与 feature geometry、decision boundary 和 label prior 直接相关；
- 研究问题是 distribution shift robustness，而非 context-window scaling；
- 不仅随机分桶，而且构造并选择异质 evidence views；
- 在多个 TFM backbone、分类/回归和两类 shift 上系统评估。

### 8.3 “TabPFN 本来就有 ensemble，为什么还需要 TabACE？”

需要区分两层随机变量：

- TabPFN PHE 平均的是模型对 feature/class permutations、preprocessors 或 configurations 的响应；
- TabACE 平均的是 \(p_\theta(y\mid x,C_k)\) 对**证据集合/context**的响应。

建议在实验中报告四格消融：base without PHE、base with PHE、TabACE without PHE、TabACE with PHE，并匹配总 forward passes。

### 8.4 “多个 context 只是更多算力带来的收益？”

必须加入 compute-matched controls：

- 同一 base context 重复 K 次；
- K 个 uniform-random contexts；
- K 个 bootstrap contexts；
- 一个大小为 K 倍的 concatenated context（模型容量允许时）；
- oracle best single context（只作为上界，不参与方法）；
- TabACE 的异质 contexts。

### 8.5 “clean historical context 是额外信息，不是 ensemble 的功劳”

回归结果应分开陈述：historical anchor 是额外可用证据。报告 base、history-only、base+history、all-context ensemble，避免把 history access 的收益归因于通用 context diversity。

## 9. 论文必须加入的 baseline 与分析

### 必须 baseline

1. Full/base context；
2. Uniform random single context；
3. KNN/LoCal-style single context；
4. MixturePFN-style single routing context；
5. Random context bagging / BETA-style bootstrap；
6. TabPFN native PHE；
7. DistPFN、BBSE、MLLS/EM、RLLS（label shift）；
8. CatBoost/XGBoost/TabM 等非 TFM 强基线；
9. 不同 TFM 的 model ensemble，至少作为 context ensemble 与 model ensemble 的成本对照。

### 必须分析

1. Context count curve 与 compute-matched curve；
2. Leave-one-context-out；
3. Pairwise disagreement、error correlation、Q-statistic 或 KL/Jensen–Shannon diversity；
4. Diversity 与 ensemble gain 的相关性；
5. 单个 context 的 clean/shift performance 和 worst-case failure；
6. aggregation：mean、confidence weighting、source-only weighting、PoE、max；
7. calibration：NLL、Brier、ECE，而不仅是 accuracy/AUC；
8. shift family 已知和未知两种设置的边界；
9. source-only selection 与所有 target labels 完全隔离的审计；
10. 运行时间、显存和 forward-pass 数。

## 10. 可直接写进论文的 Related Work 结构

建议分成四段，而不是把所有论文堆在一个段落。

### 10.1 Tabular foundation models and in-context prediction

先介绍 TabPFN、TabICL、TabDPT，说明 TFM 以 labeled table 作为 inference context。然后说明本文不改变 pretraining 或 backbone，而是研究 deployment-time context intervention。

### 10.2 Context construction for tabular ICL

讨论 TuneTables/ICD 的 learned compression、LoCalPFN/Localization/TAAR/AWARE 的 retrieval、MixturePFN 的 routing、CRUMB 的 distribution matching。总结它们通常寻找一个压缩或 query-relevant context；TabACE 则保留多个 context-induced hypotheses。

### 10.3 Context and demonstration ensembling

正面引用 DENSE、LARA、Sub-CP、AMA、Self-Consistency、SegGPT。指出多 prompt/context aggregation 在语言和视觉 ICL 已存在；本文的贡献是将这一原则系统化到 tabular foundation models，并让 diversity 对应可解释的 feature geometry、boundary、history 与 class-prior evidence。

### 10.4 Robust tabular prediction under shift

讨论 TableShift、TabFSBench、AdapTable、TabLog、DistPFN、BBSE/MLLS/RLLS。说明传统方法通常估计一种 shift 或更新/校准一个 predictor，而 TabACE 研究多个 context-conditioned predictors 的组合；同时明确当前 benchmark 给定 shift family，并不声称自动识别任意 shift。

## 11. 推荐的 Related Work 核心英文段落草稿

> **Context construction in tabular in-context learning.** Recent tabular foundation models perform prediction by conditioning a pretrained model on a labeled table supplied at inference time. Existing work has therefore treated the context as an object for compression, optimization, or retrieval. TuneTables and in-context data distillation learn compact contexts, whereas LoCalPFN, MixturePFN, TAAR, and CRUMB construct query- or batch-specific support sets. These methods primarily seek a single context or route each query to a local expert. In contrast, we retain multiple context-induced predictors and aggregate their evidence, allowing semantically distinct context views to contribute without committing to any single support set.

> **Context ensembling.** Ensembling demonstrations has been explored in language-model ICL. DENSE partitions demonstrations into buckets and aggregates their predictive probabilities, while LARA and Sub-CP improve the weighting or partitioning of context blocks. TabPFN itself also averages predictions across preprocessing and permutation configurations, and BETA applies bootstrap aggregation to support sets. TabACE differs in its source of diversity and its target setting: it constructs heterogeneous tabular contexts encoding local, prototypical, boundary, subspace, historical, or prior evidence, keeps the foundation model fixed, and studies whether their stable evidence improves robustness under feature/context and label-prior shifts.

## 12. 总体判断

这项工作的 novelty 不应定义为“发明 context ensemble”，因为 DENSE、LARA、BETA、TabPFN PHE 和 TabFM-Ensemble 已经分别覆盖 demonstration、support-set、configuration 和 feature-view aggregation。更有说服力的贡献是：

1. 把 **context-induced predictor** 作为 TFM 鲁棒性的基本分析单位；
2. 设计面向不同 shift mechanism 的异质 context views；
3. 在同一冻结 TFM 上分离“模型多样性”和“证据/context 多样性”；
4. 系统研究哪些 context 信号能在 shift 下保留，哪些只是 context-specific nuisance；
5. 通过严格的 source-only selection、compute-matched controls 和多 backbone 实验说明收益不是普通 bagging 或额外算力。

如果这些区别在方法和实验中被落实，当前题目 **TabACE: Adaptive Ensembling of Configurable Contexts for Reliable Prediction with Tabular Foundation Models** 仍然成立；如果最终方法主要只是随机/bootstrapped support subsets 的平均，则会与 BETA 和 DENSE 过于接近，题目和贡献都需要收窄。
