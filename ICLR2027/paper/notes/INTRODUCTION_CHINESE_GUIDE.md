# TabACE 相关论文 Introduction 中文精读意译

> 整理日期：2026-08-03  
> 说明：本文档不是论文 Introduction 的逐句翻译，也不复现原文，而是按照“研究背景—现有缺口—作者思路—主要贡献—与 TabACE 的关系”进行中文精读意译。需要核对措辞或引用时，请回到每节提供的原论文链接。

## 第一组：与 TabACE 最接近的工作

### 1. TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems（BETA）

[原论文](https://arxiv.org/abs/2502.02527)

TabPFN 展示了无需针对下游数据重新训练、只依靠上下文样本就能完成表格分类的能力，但原始模型仍受到数据规模、特征维度和类别数量的限制。已有扩展方法通常沿两条路线展开：一条通过选择更合适的 context 或局部样本来减小偏差，另一条通过微调模型使其适应目标任务。然而，这些方法往往只改善偏差或方差中的一项，还会增加推理开销。

BETA 从 bias–variance decomposition 出发，认为提高 TabPFN 的实用性需要同时处理分布不匹配造成的偏差和数据扰动造成的方差。它在冻结或基本保持 TabPFN 主体能力的基础上增加轻量输入编码器，使下游数据被映射到多个潜在表示空间，并利用 BatchEnsemble 以较低参数成本产生多条表示路径。推理阶段再对训练数据进行 bootstrap sampling，形成多个 support sets，并聚合相应预测以降低方差。

作者将贡献概括为：提出兼顾偏差与方差的 TabPFN 适配框架；利用轻量编码器、参数高效微调和 bootstrap context 改善高维、大样本、多分类任务；并在大量表格分类数据集上验证其性能和扩展性。

**与 TabACE 的关系：**这是最危险的直接先例。两者都会改变 support context 并聚合预测。区别在于 BETA 的 context diversity 主要来自同质的 bootstrap 扰动，且包含下游参数适配；TabACE 必须突出冻结基座、异质证据 view、分布偏移目标以及 source-only context selection。

### 2. Exploring Demonstration Ensembling for In-context Learning（DENSE）

[原论文](https://arxiv.org/abs/2308.08780)

语言模型的 in-context learning 通常把多个输入—输出示例串联在一个 prompt 中，再附上待预测样本。作者指出这种做法有两个根本问题：第一，所有示例被一次性拼接后，模型几乎无法显式控制每个示例对预测的贡献，不相关或误导性示例可能污染最终判断；第二，Transformer 的 context-window 限制使大量或长文本 demonstrations 无法同时放入 prompt。

DENSE 不再把所有 demonstrations 合并到一个 context，而是将它们划分为多个 bucket。对于同一个测试输入，语言模型分别以不同 bucket 为 context 产生概率分布，再使用 mixture-of-experts、product-of-experts、max 或加权变体聚合。这样既可以在多次短推理中利用更多示例，也能让不同 context 的预测贡献被显式控制。

实验的核心结论是：简单的 demonstration ensembling 可以超过直接拼接，尤其当部分示例相关性不足或上下文长度受限时；不同聚合规则体现不同的“共识”或“最可信专家”假设。

**与 TabACE 的关系：**DENSE 已经给出了几乎相同的数学外壳 \(p(y\mid x,C_k)\rightarrow\operatorname{Aggregate}_k\)。TabACE 的新意不能是这个外壳本身，而应是表格 context 的结构化构造、shift-specific evidence 与跨 backbone 的鲁棒性研究。

### 3. Divide, Reweight, and Conquer: A Logit Arithmetic Approach for In-Context Learning（LARA）

[原论文](https://aclanthology.org/2026.eacl-long.11/)

随着 demonstrations 数量增加，in-context learning 不仅面临二次增长的注意力成本，还可能因长 prompt 中的信息竞争而出现性能下降。已有的 context splitting 方法能够降低单次输入长度，但如果只是平均不同块的结果，就默认所有 context blocks 同样可靠，无法处理冗余、冲突和任务相关性差异。

LARA 将长 demonstration 序列拆成多个可并行处理的短组，对每组分别推理，然后在 logit 空间进行算术组合。它通过不依赖梯度的优化方式估计各组权重，使最终结果能够强调更有用的 context groups。B-LARA 进一步把权重限制为二值，从而同时完成 group selection 和内存压缩。

作者强调，该方法不是简单延长 context，而是把 many-shot ICL 转化成多个短 context 的可控组合问题，并在准确率和内存占用之间取得更好的折中。

**与 TabACE 的关系：**LARA 提醒 TabACE 不能只报告均值聚合，还需要与 source-only weighting、PoE、max 或选择式聚合比较，并说明权重是否使用了目标标签。

### 4. Mixture of In-Context Prompters for Tabular PFNs（MixturePFN）

[原论文](https://openreview.net/forum?id=2fojNANZSv)

表格 ICL 在小数据上表现突出，但 Transformer 对 context length 的二次复杂度使 PFN 难以直接处理大表。简单均匀下采样会丢失局部结构，而把整个大表输入模型又不可行。作者因此把问题解释为：如何让有限 context 的 PFN 在大规模数据中只读取与当前查询最相关的局部证据。

MixturePFN 首先用聚类将训练数据划分为多个局部区域，每个区域构成一个 in-context prompter 或 expert。测试样本根据所在区域被路由到相应 context；作者还提出 context-aware fine-tuning，通过围绕聚类中心构造 KNN contexts 来训练适应局部区域的 PFN experts。

这套方法的核心不是让所有 experts 同时为每个查询投票，而是以 divide-and-conquer 的方式把不同查询分配给更相关的局部 context，从而兼顾扩展性和局部决策能力。

**与 TabACE 的关系：**MixturePFN 是“多个 contexts，但单查询主要选一个”的代表。TabACE 应突出保留多个假设而不是早期路由，并比较 routing、best-single-context 与 ensemble。

### 5. Accurate Predictions on Small Data with a Tabular Foundation Model（TabPFN v2）

[原论文](https://www.nature.com/articles/s41586-024-08328-6)

深度学习在图像、文本等原始数据模态中取得主导地位，但表格任务长期仍由梯度提升树占优。一个重要原因是表格数据之间缺乏统一语义：相同数值在不同表中可能表示完全不同的变量，任务规模又通常较小，使端到端训练和跨数据集迁移都很困难。

TabPFN 将预测算法本身作为预训练对象。模型在大量由结构因果模型和其他生成机制产生的合成任务上学习近似后验预测；面对一个新表时，训练样本及其标签作为 context，测试行作为 query，在一次前向过程中完成预测，无需下游超参数搜索。第二代模型扩大了可处理的数据规模和特征类型，并展示了分类、回归、密度估计、生成和 embedding 等能力。

论文还通过多种 preprocessing、feature/class ordering 和模型配置运行多次预测后平均。这说明 TabPFN 的最终性能并非始终来自严格意义上的单次预测，而已经包含一种 configuration-level ensemble。

**与 TabACE 的关系：**TabACE 要把“固定模型”定义清楚：是否包括 TabPFN 内置的多个配置。最好分别报告 native ensemble 开关状态和总 forward-pass budget。

### 6. Understanding Context Sampling in TabPFN on Small Tabular Datasets

[原论文](https://arxiv.org/abs/2607.26628)

即使数据集不大，实际使用 TabPFN 时仍要决定 context 包含多少行、选择哪些行。由于预测直接依赖这些输入样本，context sampling 不只是工程细节，而可能决定模型的准确率和稳定性。然而，已有工作更多关注模型架构或大规模扩展，对小数据环境下的 sampling variability 缺少系统分析。

论文通过在多个 OpenML 数据集上重复抽取 contexts，研究三个问题：更大的 context 是否降低不同抽样之间的预测波动；context 的作用来自匹配训练分布还是覆盖特征空间；昂贵的 K-means、farthest-point selection 是否优于随机抽样。作者发现，增大 context 通常同时提高准确率和稳定性；随机 context 的成功更多来自平均意义上的 coverage，而不是严格复现 feature means；强行匹配均值甚至可能牺牲 diversity 并降低表现。

作者因此主张把 context diversity、coverage 和选择成本一起评价，而不是只以“代表原分布”衡量一个 context 的质量。

**与 TabACE 的关系：**它直接支持“单 context 有采样方差”这一动机，也要求 TabACE 与 uniform random contexts、K-means 和 coverage-matched contexts 做同预算比较。

### 7. Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction

[原论文](https://openreview.net/forum?id=7hIc6bK7tR)

生物分子效力预测通常数据量有限、特征表示多样，适合使用能够从少量 context 样本快速适应的 TabPFN。但作者观察到，TabPFN 对输入 context 和数据子集高度敏感，因此实践中经常从不同子集得到多个 predictor 再进行 post-hoc ensemble。问题在于，没有测试标签时很难知道哪些成员值得保留。

论文研究 TabPFN 自身输出的不确定性是否能充当无标签的 ensemble-selection 信号。作者以预测区间宽度等量衡量模型不确定性，发现它与真实预测误差显著相关；选择平均不确定性较低的 subset predictors 再平均，可以优于朴素地集成所有成员或只使用完整数据训练的一个模型。

**与 TabACE 的关系：**它已经明确连接了 context sensitivity、subset predictors、label-free selection 和 averaging。TabACE 需要强调多任务、分类/回归、结构化 context types 与 shift evaluation，而不能只声称“按 context 生成多个模型再选”。

### 8. Submodular Context Partitioning and Compression for In-Context Learning（Sub-CP）

[原论文](https://arxiv.org/abs/2510.05130)

把长 context 分块是降低 ICL 成本的常见方法，但随机或连续分块可能让部分信息在多个块中重复，另一些关键信息则完全缺失。这使后续的 ensemble、压缩或 cross-attention 无法充分利用全部 demonstrations。作者认为，context partition 的质量应在每块内部结构和跨块覆盖之间取得平衡。

Sub-CP 用次模目标控制 context blocks 的构成，可以让一个块更强调局部一致性，也可以让不同块覆盖更广的全局信息。由于划分可预计算，该框架可以接入 DENSE 的 MoE、PoE 或 max 等聚合方法，而不必改变底层语言模型。

**与 TabACE 的关系：**这项工作强化了一个重要观点：ensemble gain 依赖成员的结构化 diversity，而不是 context 数量本身。TabACE 应量化 view-level redundancy 和 coverage。

### 9. Parallel In-Context Learning for Large Vision-Language Models

[原论文](https://arxiv.org/abs/2603.16092)

视觉语言模型使用多模态 demonstrations 适应新任务，但更多 demonstrations 会迅速提高注意力成本和推理延迟。直接缩短 context 会损失任务信息，而一次处理全部 context 又缺乏可扩展性。

Parallel-ICL 把 demonstrations 划分为多个短 context，分别并行处理，再通过加权 product-of-experts 在 logit 层整合。分块时利用聚类增加不同 blocks 的差异，聚合时依据 query 与 context 的相似度调整权重。其目标是用多个并行短推理逼近完整长 context 的能力。

**与 TabACE 的关系：**它同时包含 diversity-aware partition 和 query-aware aggregation，是异质 context ensemble 的强跨模态参照；TabACE 的优势必须来自表格 shift 机制，而非一般性并行分块。

### 10. TabFM: A Zero-Shot Foundation Model for Tabular Data

[官方介绍](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)

传统表格建模通常依赖模型训练、超参数搜索和人工特征工程。Google 将 tabular prediction 重新表述为 in-context learning：把历史训练行和待预测行组成统一输入，让预训练模型从当前表中推断任务关系。TabFM 采用混合架构，先交替进行行列注意力，再把每行压缩成向量，最后在行表示上执行高效 ICL。

除单次 zero-shot 配置外，TabFM-Ensemble 还构造交叉特征和 SVD 特征等多种输入表示，形成 32 路预测，并用非负最小二乘学习组合权重；分类任务再做 Platt scaling。也就是说，它已经把“同一 TFM + 不同 feature views + 加权预测组合”作为提高性能的正式配置。

**与 TabACE 的关系：**两者都可能被称为 view ensemble。关键差别是 TabFM 改变 feature representation/preprocessing，TabACE 改变 labeled evidence context，并研究 context/query distribution shift。

### 11. A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities

[原论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c57b3718381cb2bf9c1ccd63377f2448-Abstract-Conference.html)

TabPFN v2 在多类表格任务上表现突出，但它为何能跨异构表泛化、内部学到了什么、以及如何突破高维和大数据限制仍不清楚。作者从输入 token、内部表征和测试时策略三个层面研究其机制。

研究发现，即使 attribute tokens 被随机化，模型仍能推断特征之间的关系；把 TabPFN v2 当作 feature extractor 时，其表示具有较强类别可分性。作者还采用 test-time divide-and-conquer 和 post-hoc ensemble，分析 ensemble size 对性能的增益与饱和趋势。

**与 TabACE 的关系：**这是报告 TabPFN configuration/post-hoc ensemble 的重要来源，也提示 TabACE 要分析 context 数增加后的边际收益和饱和点。

### 12. Retrieval & Fine-Tuning for In-Context Tabular Models（LoCalPFN）

[原论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html)

表格任务的异质性使统一深度模型难以适应所有数据集。TabPFN 通过合成任务上的预训练和 ICL 在小表上取得强结果，但上下文注意力的二次复杂度限制了其规模，并且一个全局 context 可能包含与当前 query 无关或相互冲突的样本。

LoCalPFN 对每个 query 检索最近邻，只把局部子集作为 context，从而在有限长度内突出相关决策结构；随后使用同类检索 context 对模型进行任务特定的 fine-tuning，使模型的参数适配真实数据上的局部规律。论文在大规模基准中展示了 retrieval 与 fine-tuning 的互补性。

**与 TabACE 的关系：**LoCalPFN 代表“选最相关的一个 context”。TabACE 可以把 KNN view 作为成员之一，但必须证明其他 views 的补充价值，而不只是复现局部检索。

### 13. Task-Aligned Attention Retrieval for Scaling Tabular Foundation Models（TAAR）

[原论文](https://openreview.net/pdf?id=qBMNsGiE3u)

最近邻检索可以缓解 TFM 的 context-size 限制，但原始特征空间中的欧氏距离不一定对应任务相关性。混合类型、长尾类别、无关特征和 covariate shift 都可能使几何上接近的样本在标签机制上并不相似。

TAAR 利用基础模型自身的 attention 信号识别当前任务最有预测力的特征，并据此衡量 query 与候选 context rows 的相似性。它不训练新的检索器，作为 drop-in wrapper 用一次额外前向获得 task-aligned retrieval scores，然后构造最终 context。

**与 TabACE 的关系：**TAAR 仍在寻找一个最佳 context，但它比普通 KNN 更强，适合作为 task-aligned single-context baseline。

### 14. CRUMB: Efficient PFN Inference via Distributionally Matched Context Batching

[原论文](https://arxiv.org/abs/2606.11473)

PFN 将标注训练集作为 context，因此当训练表很大时推理成本会变得不可接受。逐查询 KNN 虽然相关，但不利于批处理；固定局部簇又可能无法跟随当前测试批次的分布变化。

CRUMB 先对测试 queries 聚类，再为每个 query cluster 从训练集选择一个小 support context。选择目标是最小化训练 context 与该测试簇之间的 maximum mean discrepancy，使 context 在分布层面匹配当前 queries。随后对每个缩减后的 context–query batch 执行标准 PFN 推理。

论文认为 MMD matching 除了提高效率，还自然提供一定的 covariate-drift robustness，因为 context 会随当前 query distribution 改变。

**与 TabACE 的关系：**CRUMB 是 distribution-matched single context，必须与 TabACE 的多 context aggregation 区分；也可作为 feature-shift 下很强的 context constructor。

### 15. DistPFN: Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment

[原论文](https://arxiv.org/abs/2605.04363)

TabPFN 的预测不仅取决于特征—标签关系，也会受到 context 中类别比例的影响。当训练 context 与测试环境的 label prior 不一致时，模型可能过度偏向 context 中的多数类。传统 label-shift correction 已经研究如何在不获得测试标签的情况下调整后验，但这些方法并未针对 tabular ICL 中“训练先验直接进入 context”这一机制设计。

DistPFN 在测试时重新缩放 TabPFN 输出，降低训练 context prior 对预测的影响，并增强模型后验信息的贡献；DistPFN-T 使用温度参数根据 prior–posterior 差异调节修正强度。方法不改变网络结构，也不要求额外训练。

**与 TabACE 的关系：**它是 label-prior shift 的最近直接方法。TabACE 需要说明多个 prior estimators/context signals 相比单一 posterior adjustment 的额外价值，并公平比较 calibration 和 target-prior estimation。

## 第二组：表格基础模型与 Context 优化

### 16. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second

[原论文](https://openreview.net/forum?id=cp5PvcI6w8_)

传统表格分类器通常需要为每个新数据集重新训练和调参，而 AutoML 又带来较大计算成本。该论文提出 Prior-Data Fitted Network：先从事先定义的任务先验中生成大量数据集，训练 Transformer 直接近似贝叶斯后验预测。部署时，模型把一个新任务的训练样本和测试样本一起作为输入，在单次前向中输出测试标签概率。TabPFN 由此把“拟合一个分类器”改写为“在 context 中执行预训练好的学习算法”。

**与 TabACE 的关系：**这是 context-induced prediction 的基础定义；TabACE 相当于研究同一个 PFN 对多个 \(C_k\) 的响应。

### 17. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

[原论文](https://proceedings.mlr.press/v267/qu25d.html)

TabPFN 证明了表格 ICL 的有效性，但交替行列注意力使其难以扩展到更长训练 context。TabICL 将表示学习和任务推断拆成两阶段：先通过 column-then-row attention 得到固定维度的行表示，再在行序列上执行 ICL。作者据此在保持强小数据性能的同时，将 context 扩展到数十万行，并用大规模分类 benchmark 说明表格 ICL 的优势并不限于小表。

**与 TabACE 的关系：**TabICL 是重要 backbone；它也可测试 TabACE 的收益是否只来自 TabPFN 的 context-length 限制。

### 18. TabDPT: Scaling Tabular Foundation Models on Real Data

[原论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)

早期 TFM 多依赖合成任务预训练，但合成 generator 可能无法涵盖真实表中的规律。TabDPT 结合 retrieval-based ICL 和 self-supervised learning，在大量真实表上构造预训练任务，使模型学习可迁移的表格模式。论文还比较真实与合成预训练数据，发现真实数据能提供合成 prior 难以捕获的信号，并展示模型规模和数据规模带来的可预测提升。

**与 TabACE 的关系：**TabDPT 说明 retrieval 已进入 backbone 训练；TabACE 在其上有效，能支持方法与特定合成 prior 无关的论点。

### 19. TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models

[原论文](https://arxiv.org/abs/2511.08667)

第二代 TabPFN 在小中型表格上很强，但现代生产数据往往包含更多行和特征。2.5 模型报告围绕容量、训练分布、工程效率和更大的输入限制进行升级，使 zero-shot TFM 能覆盖更接近 TabArena 的实际规模。报告同时强调 default、tuned 和 ensemble 配置之间的区别，说明评价 TFM 时必须记录精确模型版本和推理协议。

**与 TabACE 的关系：**如果使用该版本，必须固定它自身的 ensemble/preprocessing，避免把模型升级收益混入 context ensemble。

### 20. MITRA: Mixed Synthetic Priors for Tabular Foundation Models

[原论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/177d68f4adef163b7b123b5c5adb3c60-Paper-Conference.pdf)

PFN 的归纳偏置来自生成预训练任务的 prior，因此单一 generator 会限制它在真实表上的适用范围。MITRA 将多类合成数据机制组合起来，使模型既接触平滑、因果式结构，也接触复杂或非平滑的决策规律。其核心论点是：与其押注某一种“最真实”的 generator，不如通过 mixed priors 扩大任务分布覆盖。

**与 TabACE 的关系：**MITRA 在参数中混合 prior，TabACE 在推理 context 中混合 evidence；二者发生在不同阶段。

### 21. Large Scale Transfer Learning for Tabular Data via Language Modeling（TabuLa-8B）

[原论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04bdc-Abstract-Conference.html)

表格之间 schema 不一致，使传统跨表预训练缺少类似语言 token 的统一接口。TabuLa 将表格 prediction 转换成语言建模问题，利用数百万张表和十亿级数据行继续训练大语言模型，并设计适合表格样本打包和注意力的输入形式。作者希望利用 LLM 的规模化预训练能力实现少样本表格迁移。

**与 TabACE 的关系：**属于另一类 TFM 路线，可作为背景而非直接 context-ensemble 竞争者。

### 22. Position: Why Tabular Foundation Models Should Be a Research Priority

[原论文](https://openreview.net/forum?id=amRSBdZlw9)

论文指出，现实世界中大量关键数据以表格形式存在，但 foundation-model 研究资源主要集中在语言和视觉。表格数据跨领域异构、缺乏统一 schema，却也提供了跨数据集 contextualization、少样本学习、自动数据科学和科学发现的机会。作者主张发展能够在相关表之间迁移和利用上下文的 Large Tabular Models。

**与 TabACE 的关系：**可用于 Introduction 的宏观背景，不应用来支持具体算法有效性。

### 23. Scaling TabPFN: Sketching and Feature Selection for Tabular PFNs

[原论文](https://arxiv.org/abs/2311.10609)

原始 TabPFN 对样本数和特征数都有硬限制，因此在真实任务中必须压缩输入。论文系统研究简单 sketching、row selection 和 feature selection，询问在预训练模型固定时，什么样的数据摘要最适合作为 context。结果说明常规压缩方法对 PFN 并不等价，不恰当的摘要会删除模型推断任务所需的关系。

**与 TabACE 的关系：**表明 context constructor 必须作为算法组件和消融对象，而不是预处理细节。

### 24. In-Context Data Distillation with TabPFN

[原论文](https://arxiv.org/abs/2402.06971)

数据规模限制阻止 TabPFN 直接处理大表。与选择真实行不同，in-context data distillation 直接优化一小组可学习的合成 context points，使它们在有限内存内代表完整数据集。优化目标通过固定 TabPFN 反向传播，寻找能诱导正确下游预测的压缩 context。

**与 TabACE 的关系：**它学习一个高容量 context；TabACE 使用多个可解释 context views。两者可构成“单个优化 context versus 多个构造 context”的对比。

### 25. TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks

[原论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97dc07f1253ab33ee514f395a82fa7cc-Abstract-Conference.html)

TuneTables 将 prompt tuning 的思想引入 PFN：不全面更新 TabPFN，而是学习少量 context embeddings，把大数据集压缩成模型可处理的任务表示。这样既绕过输入长度限制，又能在较低参数成本下适应下游数据。作者还展示 learned context 可被用于解释和公平性目标。

**与 TabACE 的关系：**TuneTables 优化一个 context；TabACE 的优势应体现在无需训练、保留多种证据和 shift robustness。

### 26. Towards Localization via Data Embedding for TabPFN

[原论文](https://openreview.net/forum?id=LFyQyV5HxQ)

直接在原始特征空间做 KNN 会受量纲、无关维度和混合类型影响。该工作学习更适合 TabPFN 的数据表示，再在 embedding space 中选择邻居 context。作者从 bias–variance 角度解释 localization：局部 context 能减少全局异质性造成的偏差，但邻域样本较少也可能提高方差。

**与 TabACE 的关系：**为“单一局部 view 既有优势也有方差”提供理论动机。

### 27. Retrieval-aligned TFMs for Clinical Risk Prediction（AWARE）

[原论文](https://arxiv.org/abs/2604.01841)

电子健康记录具有高维、缺失、类别不平衡和跨机构 shift，原始距离检索常被弱相关维度主导。论文系统评估 TFM 在这些约束下的表现，发现 naive retrieval 随异质性和不平衡增加而退化。AWARE 使用监督表示学习和轻量 adapter，把检索空间与临床预测目标对齐，从而提高邻域 label consistency。

**与 TabACE 的关系：**说明 KNN context 本身可能是噪声源；TabACE 应测试检索质量下降时 ensemble 是否真正抑制错误。

### 28. Active In-Context Learning for Tabular Foundation Models

[原论文](https://arxiv.org/abs/2603.27385)

主动学习希望用尽量少的标签获得高性能，但小样本阶段传统模型的不确定性往往不可靠。TFM 在冻结参数下即可依据少量 labeled context 输出概率，因此可以把“选择下一条标注数据”直接理解为优化未来 context。论文比较 uncertainty、coreset、两者混合及可扩展代理策略。

**与 TabACE 的关系：**证明 context 可以被主动设计；TabACE 则关注已有证据如何构成多个 views 并聚合。

### 29. Context-Constrained Transfer Learning via Data Distillation（TL-ANDI）

[原论文](https://openreview.net/forum?id=35LXG0a3Y6)

将异质 source data 直接并入 TFM context 可能产生 negative transfer，且 context size 限制使这种做法不可扩展。TL-ANDI 根据 covariate 和 posterior shift 选择代表性 anchors，并通过局部 kernel distillation 平滑标签，在固定 context budget 下压缩跨域信息。

**与 TabACE 的关系：**它正面处理 shifted external evidence，但依赖一个 distilled context；属于值得持续跟踪的并行工作。

## 第三组：表格与一般 In-Context Ensemble

### 30. Ensembling Tabular Foundation Models — A Diversity Ceiling and a Calibration Trap

[原论文](https://arxiv.org/abs/2605.18696)

当不同 TFM 在不同数据集上各有胜负时，自然想到通过 ensemble 获得稳定提升。论文却发现多个现代 TFM 的预测错误高度相关，形成 diversity ceiling；复杂 stacking 的准确率收益很小，还可能通过过度锐化决策边界破坏 log-loss 和 calibration。作者因此强调，ensemble 是否有用取决于成员互补性，而不是成员数量或组合器复杂度。

**与 TabACE 的关系：**非常适合支持“context diversity 可能比 model diversity 更有价值”的假设，但 TabACE 必须实际测量错误相关性。

### 31. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

[原论文](https://arxiv.org/abs/2410.24210)

复杂 attention 模型并不总能在表格任务上超过设计良好的 MLP。TabM 借鉴 BatchEnsemble，在共享绝大多数参数的同时产生多个预测分支，以低成本模拟模型 ensemble。单个分支可能较弱，但聚合后表现强，说明表格学习中高效的预测多样性本身就是重要基线。

**与 TabACE 的关系：**用于区分 parameter/model diversity 与 context diversity，并提供效率对照。

### 32. TabArena: A Living Benchmark for Machine Learning on Tabular Data

[原论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)

静态 benchmark 会因数据缺陷、模型版本更新和实现差异迅速过时。TabArena 建立持续维护的数据集、模型和排行榜体系，并统一验证与时间预算。论文发现 validation protocol、超参数配置 ensemble 和 cross-model ensemble 会显著改变排名，因此任何新表格方法都需要明确 default、tuned 和 ensemble 成本。

**与 TabACE 的关系：**要求按计算预算和验证协议公平评价，不应把更多 forward passes 当作免费资源。

### 33. Make Still Further Progress: Chain of Thoughts for Tabular Data Leaderboard

[原论文](https://arxiv.org/abs/2505.13421)

表格比赛中没有一个模型始终最优，人工选择和 stacking 又依赖经验。该工作让 LLM 读取测试实例邻居及多个外部模型的预测，通过 Chain of Tabular Thoughts 进行实例级动态组合。它不依赖原始特征语义，而是把模型输出及局部表现组织为 context，让 LLM 充当 ensemble controller。

**与 TabACE 的关系：**它集成不同模型，TabACE 集成同一模型在不同证据下的预测。

### 34. Of Graphs and Tables: Zero-Shot Node Classification with TFMs（TabGFM）

[原论文](https://arxiv.org/abs/2509.07143)

图基础模型通常需要专门图预训练，而现有图训练集合未必代表真实网络。TabGFM 把节点特征和结构编码成表格列，交给已有 TFM 零样本分类；为获得多样性，它在不同 subsampled tables 上应用多个 TFM，并做 ensemble selection。

**与 TabACE 的关系：**包含 table subsampling 和 aggregation，但模型、任务模态和目标均不同。

### 35. Few-shot Reranking for Multi-hop QA via Language Model Prompting

[原论文](https://aclanthology.org/2023.acl-long.885/)

多跳问答需要从大量候选文档路径中选择支持答案的链条，而监督 reranker 需要昂贵标注。论文用语言模型 prompt 直接给候选路径打分；由于一条 demonstration 已包含多篇文档，context window 无法容纳很多示例，因此对不同 demonstration contexts 分别评分并用 mean/max 聚合。

**与 TabACE 的关系：**这是 DENSE 之前/同期的任务化实例，展示 ensemble 可绕开 context-window 限制。

### 36. Ask Me Anything: A Simple Strategy for Prompting Language Models（AMA）

[原论文](https://openreview.net/forum?id=bhUPJnS2g0X)

少样本 prompting 常依赖人工寻找一个“完美 prompt”，而小模型尤其容易受 prompt format 影响。AMA 自动把任务改写成多个 question-answer prompts，接受每个 prompt 都可能有偏差，再用 weak supervision 估计这些 noisy sources 的准确性和相关性并聚合输出。

**与 TabACE 的关系：**支持“多个不完美 context 经过相关性感知聚合可优于一个精心选择 context”的观点。

### 37. Self-Consistency Improves Chain-of-Thought Reasoning

[原论文](https://openreview.net/forum?id=1PL1NIMMrw)

greedy chain-of-thought 容易锁定一条局部最优推理路径。Self-Consistency 从同一模型采样多条不同推理轨迹，再对最终答案做边缘化或多数投票。若多条相互独立的路径到达同一答案，就把这种一致性视为可靠性信号。

**与 TabACE 的关系：**二者都增加 test-time compute 并用共识降噪，但 TabACE 的多样性来自 context，而不是随机 decoding path。

### 38. Explanation-aware Soft Ensemble Empowers LLM In-context Learning（EASE）

[原论文](https://arxiv.org/abs/2311.07099)

多个 ICL 输出和解释可以被集成，但简单投票把可靠与不可靠解释等同处理。EASE 检查解释与预测的一致性，用解释质量指导 ensemble，并在概率层进行软组合，从而减少不可信 rationale 对最终结果的影响。

**与 TabACE 的关系：**提示可根据 context reliability 而非仅置信度学习权重。

### 39. In-Context Ensemble Learning from Pseudo Labels for Video-Language Models

[原论文](https://arxiv.org/abs/2409.15867)

视频语言模型难以从长视频生成细粒度操作流程，context window 又无法放入所有示范。论文生成多条可能的 pseudo-labeled SOP paths，再通过 in-context ensemble 聚合，使模型利用超过单次窗口容量的证据，并以一致性减少单一路径错误。

**与 TabACE 的关系：**属于多模态 context ensemble 的名称和机制先例。

### 40. SegGPT: Towards Segmenting Everything in Context

[原论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.pdf)

视觉分割任务通常为每个数据集训练专门模型。SegGPT 把不同分割任务统一为 in-context coloring：参考图像及其彩色 mask 定义任务，模型再为 query 图像生成对应输出。为利用多个 references，论文提出 spatial ensemble（拼接参考）和 feature ensemble（分别处理后在中间层平均 query features）。

**与 TabACE 的关系：**证明 context ensemble 不一定只发生在最终概率层；也提醒本文应说明选择 late fusion 的理由。

### 41. Calibrate Before Use: Improving Few-shot Performance of Language Models

[原论文](https://proceedings.mlr.press/v139/zhao21c.html)

少样本语言模型会因 prompt 格式、示例选择和顺序产生巨大波动。作者发现一部分不稳定来自模型对特定答案的先验偏好，并用 content-free input 估计该偏差，再校准输出分布。方法显著提高不同 prompts 下的准确率和稳定性。

**与 TabACE 的关系：**支持 context 会注入 label bias；label-prior track 应同时评价 calibration。

### 42. Fantastically Ordered Prompts and Where to Find Them

[原论文](https://aclanthology.org/2022.acl-long.556/)

在 demonstrations 完全相同的情况下，仅改变顺序就可能让性能从接近随机变为接近 SOTA。论文用模型生成的人工开发集和 entropy 指标选择较好排列，说明 context composition 中存在与任务信号无关的偶然因素。

**与 TabACE 的关系：**为“单 context 携带附加 nuisance information”提供直接经验依据。

### 43. Learning To Retrieve Prompts for In-Context Learning

[原论文](https://aclanthology.org/2022.naacl-main.191/)

随机选择 demonstrations 无法保证与 query 相关。论文用语言模型对候选示例进行正负标注，再训练 dense retriever，在测试时检索更有帮助的 prompts。结果表明 demonstration selection 是 ICL 性能的重要决定因素。

**与 TabACE 的关系：**是 query-specific context selection 的通用先例，但没有保留多个 context hypotheses。

### 44. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

[原论文](https://aclanthology.org/2022.emnlp-main.759/)

论文追问 ICL 究竟从 demonstrations 中学到什么。实验发现，在若干语言分类任务中随机替换 demonstration labels 只造成有限损失，反而 label space、input distribution 和格式等信息对表现很重要。由此说明 context 同时传递任务映射和多种外围线索，模型可能依赖比预期更简单的统计规律。

**与 TabACE 的关系：**与“context 能传递 prior/feature distribution，也会夹带附加信息”的核心动机高度一致。

### 45. How Many Demonstrations Do You Need for In-context Learning?

[原论文](https://openreview.net/forum?id=JPUx2nVgWa)

更多 demonstrations 并不必然更好。论文把对某个 query 有助于正确答案的示例称为 positive demos，把引向错误答案的示例称为 negative demos，并观察到多示例之间会产生干扰和伪相关；一个合适示例有时优于多个混合示例。

**与 TabACE 的关系：**说明简单拼接 contexts 不等同于 ensemble，也支持保留独立预测后再组合。

### 46. Mixture of Demonstrations for In-Context Learning（MoD）

[原论文](https://openreview.net/forum?id=uqxSLoCw3K)

从大型 demonstration pool 中直接检索面临巨大搜索空间和噪声。MoD 先将示例池划成多个 groups，每组由一个 retriever expert 管理，再让 experts 协作检索最终 prompt，以降低无用示例对学习检索器的影响。

**与 TabACE 的关系：**属于多专家检索，而非多 context 输出聚合。

### 47. MEAL: Stable and Active Learning for Few-Shot Prompting

[原论文](https://aclanthology.org/2023.findings-emnlp.36/)

few-shot prompt learning 同时对样本选择和微调随机性敏感，导致结果难以复现。MEAL 结合 multiprompt fine-tuning、prediction ensembling 与主动学习，重点降低不同运行之间的波动，并用信息量更高的样本扩充小训练集。

**与 TabACE 的关系：**可引用其“ensemble 提升 few-shot stability”的动机，但它更新模型参数。

## 第四组：表格分布偏移与 Label Shift

### 48. TableShift: Benchmarking Distribution Shift in Tabular Data

[原论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html)

表格模型大多在随机 IID 划分上评价，无法反映跨地区、时间、人群或机构部署时的退化。TableShift 从真实数据中构造具有明确 domain split 的任务，统一 ID/OOD evaluation，并发现常见模型在自然 shift 下普遍下降，复杂 domain-generalization 方法也不稳定超过良好 ERM 基线。

**与 TabACE 的关系：**是补充自然 shift 实验和避免只依赖合成扰动的重要 benchmark。

### 49. TabFSBench: Tabular Benchmark for Feature Shifts in Open Environments

[原论文](https://proceedings.mlr.press/v267/cheng25e.html)

现有 distribution-shift 研究多假设训练和测试阶段具有同一组特征，但开放环境中的表格会出现新增、缺失、重命名或语义变化的列。TabFSBench 系统定义多类 feature shifts，评价传统、深度、LLM 和 TFM 方法，强调 feature availability/schema change 是表格数据特有而长期被忽略的问题。

**与 TabACE 的关系：**可规范 feature shift 的术语，但当前 feature-context stress 是否完全符合其定义需要逐项核对。

### 50. RLSbench: Domain Adaptation Under Relaxed Label Shift

[原论文](https://proceedings.mlr.press/v202/garg23a.html)

经典 label shift 假设 \(p(x\mid y)\) 不变，但现实中 label proportions 和 class-conditionals 往往同时变化。RLSbench 构建跨视觉、文本和表格的大规模 shift pairs，显示许多 domain-adaptation 方法在 label proportion shift 下失败。作者提出先进行 pseudo-balancing，再按估计 target prior 调整分类器的通用流程。

**与 TabACE 的关系：**提醒 label-prior 实验要讨论纯 label shift 假设是否成立，并考虑 relaxed shift。

### 51. Empirical Evaluation of OOD Performance of Tabular Foundation Models

[原论文](https://arxiv.org/abs/2607.26000)

TFM 的 benchmark 进展主要来自 IID 数据，而其 OOD 行为尚缺统一比较。论文在 TableShift 的自然 domain splits 上评价九个现代 TFM，发现无论预训练策略和架构如何，模型都出现系统性 shift gap；高性能模型还伴随明显硬件成本。

**与 TabACE 的关系：**提供最新的 TFM-OOD 证据，也说明应在多个新 backbone 上复核方法。

### 52. Are TFMs Robust to Realistic Query Distribution Shifts in Microbiome Data?

[原论文](https://arxiv.org/abs/2606.24995)

微生物组数据会因实验平台、测序深度和生物过程产生结构化 feature perturbation。论文保持 support context 不变，只对 queries 施加有现实含义的变化，从而隔离 support–query mismatch 对 TFM 的影响，并检验模型是否把 context 中规律错误外推到变化后的查询。

**与 TabACE 的关系：**与“context 提供的分布信息可能误导 shifted query”高度一致。

### 53. Robust Tabular Foundation Models

[原论文](https://arxiv.org/abs/2512.03307)

通用 TFM 的 synthetic prior 未必覆盖恶意或困难分布，导致模型在扰动下脆弱。论文通过 adversarial data generator 主动寻找模型弱点，再用这些任务进行训练或微调，把鲁棒性写入模型参数。

**与 TabACE 的关系：**它改变模型，TabACE 改变推理 context；两者属于 training-time robustness 与 inference-time robustness 两条路线。

### 54. Robustness Is Important: Limitations of LLMs for Predictions on Tabular Data

[原论文](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520)

语言模型和专用 TFM 虽然能完成表格预测，但可能对与任务无关的输入变化异常敏感。论文通过重复模拟和多种无关扰动检验模型稳定性，发现专用 TFM 通常比通用 LLM 准确，却仍不能免疫顺序、表示等 nuisance changes；多次变换输入并平均可缓和部分敏感性。

**与 TabACE 的关系：**支持 nuisance averaging，但需区分 input invariance ensemble 与 evidence context ensemble。

### 55. AdapTable: Test-Time Adaptation for Tabular Data

[原论文](https://openreview.net/forum?id=yhD2icebZ0)

视觉 TTA 方法直接用于表格时容易因异质特征、复杂边界、置信度偏差和类别不平衡而崩溃。AdapTable 先用 shift-aware uncertainty calibrator 修正预测可信度，再估计 target label distribution 并调整输出，使模型不再固守 source prior。

**与 TabACE 的关系：**是表格 shift-aware calibration 强基线，尤其适合比较是否需要多个 contexts。

### 56. TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules

[原论文](https://openreview.net/forum?id=LZeixIvQcB)

表格特征依赖复杂、类型混合，通用表示对齐不容易解释。TabLog 假设 shift 下逻辑规则的结构保持相对稳定，而规则参数和权重可以变化；它用可微逻辑网络表示规则，通过对比目标在测试时适配数值部分。

**与 TabACE 的关系：**代表更新模型内部规则的 TTA，与冻结模型的 context intervention 形成对照。

### 57. ODS: Test-Time Adaptation in the Presence of Open-World Data Shift

[原论文](https://proceedings.mlr.press/v202/zhou23e/zhou23e.pdf)

许多 TTA 只考虑 covariate shift，却假设测试类别比例稳定。ODS 指出开放环境会同时改变 \(P(X)\) 和 \(P(Y)\)，并提出先解耦混合 shift，再分别处理 feature/covariate 与 label-distribution components 的框架。

**与 TabACE 的关系：**支持将 feature/context shift 和 label-prior shift 分轨实验，而不是统称一种 shift。

### 58. LLM Embeddings Improve Test-Time Adaptation to Tabular \(Y\mid X\)-Shifts

[原论文](https://openreview.net/forum?id=OyjMJjfhiw)

缺失变量会使不同环境中的 \(P(Y\mid X)\) 发生变化，单纯 covariate alignment 无法解决。论文将列名和值序列化后用 LLM 编码，利用语义先验得到更易用少量目标标签适配的表示。

**与 TabACE 的关系：**属于 concept/posterior shift；当前方法若未覆盖此类 shift，不应把结论外推到所有 distribution shift。

### 59. Detecting and Correcting for Label Shift with Black Box Predictors（BBSE）

[原论文](https://proceedings.mlr.press/v80/lipton18a.html)

在 label shift 下，类别先验改变而 \(P(X\mid Y)\) 保持不变。BBSE 使用任意黑盒分类器在 source validation 上的 confusion matrix，以及它在无标签 target 上的预测频率，反推出 target label distribution，并据此重加权预测。理论上，只要 confusion matrix 可逆，即使分类器不完美或未校准也可得到一致估计。

**与 TabACE 的关系：**是 prior-estimation track 必须包含的经典 baseline。

### 60. Regularized Learning for Domain Adaptation under Label Shifts（RLLS）

[原论文](https://openreview.net/forum?id=rJl0r3R9KX)

BBSE 式权重估计在目标样本少或 confusion matrix 病态时方差很大。RLLS 对 importance weights 加入正则，并从泛化界推导适合小样本、大 shift 的估计方式，随后用权重重新训练 source classifier。

**与 TabACE 的关系：**提供小样本 prior estimation 的稳健对照，但它包含重新训练。

### 61. Maximum Likelihood with Bias-Corrected Calibration Is Hard-To-Beat at Label Shift Adaptation

[原论文](https://proceedings.mlr.press/v119/alexandari20a.html)

经典 EM/maximum-likelihood label-shift correction 依赖 calibrated posterior，现代神经网络往往不满足。论文发现，先进行带 class-specific bias 的 calibration，再用最大似然估计 target prior，通常优于 BBSE/RLLS。作者还证明目标函数具有良好优化性质，并强调 source prior 的估计方式会影响稳健性。

**与 TabACE 的关系：**MLLS/EM 是强 baseline；如果 TabACE 使用多个 prior evidence，应证明不是因为遗漏了正确 calibration。

### 62. A Unified View of Label Shift Estimation

[原论文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html)

BBSE 使用混淆矩阵矩匹配，MLLS 使用完整软预测做最大似然，二者看似不同。论文把它们统一为 distribution matching，给出 MLLS 一致性的 calibration 和可识别条件，并将有限样本误差分解为估计误差与 miscalibration error。它解释了 BBSE 因粗粒度离散预测而损失信息的原因。

**与 TabACE 的关系：**为比较 hard/soft prior estimators 和分析 calibration error 提供理论依据。

### 63. Adapting to Online Label Shift with Provable Guarantees

[原论文](https://openreview.net/forum?id=dC_Cho7PzT)

现实中的 target prior 可能随时间连续变化，而不是一次性从 source 跳到固定 target。论文构造无标签在线风险估计器，并用在线 ensemble algorithms 跟踪环境变化，给出随 shift 强度变化的 dynamic-regret 保证。

**与 TabACE 的关系：**若未来扩展到流式 context，可将不同 prior contexts 看作在线专家；当前静态实验不应声称解决 online shift。

## 第五组：Ensemble、Multi-view 与 Context 理论

### 64. Bagging Predictors

[原论文](https://doi.org/10.1023/A:1018054314350)

不稳定学习算法会因训练样本的轻微变化产生不同预测。Bagging 对训练集做 bootstrap，训练多个 predictor，再对回归结果平均或对分类结果投票，以消除部分采样方差。它有效的前提不是“模型越多越好”，而是基础算法对数据扰动敏感且成员错误不完全一致。

**与 TabACE 的关系：**random/bootstrapped context ensemble 是必须匹配的经典控制组。

### 65. The Random Subspace Method for Constructing Decision Forests

[原论文](https://doi.org/10.1109/34.709601)

当特征很多时，不同特征子空间可能支持不同但互补的分类边界。随机子空间法在多个 feature subsets 上训练分类器，再组合其判断，以降低高维噪声和成员相关性。

**与 TabACE 的关系：**subspace context 继承这一思想，但固定 TFM 的 context-conditioned prediction 与重新训练森林不同。

### 66. Feature Bagging for Outlier Detection

[原论文](https://doi.org/10.1145/1081870.1081891)

高维异常检测容易被无关特征和距离集中现象破坏。论文在多个随机特征子集上运行相同异常检测器，再组合异常分数，使真正异常有机会在相关子空间中显现，同时平均掉单个 noisy view 的错误。

**与 TabACE 的关系：**与“原始模型容易被误导的正确信息可在某些子空间 context 中恢复”这一叙述接近。

### 67. Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles

[原论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)

神经网络常给出过度自信的错误预测，贝叶斯神经网络又实现复杂。Deep Ensembles 独立训练多个网络并平均预测，获得较强准确率、校准和 OOD uncertainty。成员在未知输入上的分歧也可作为 epistemic uncertainty 信号。

**与 TabACE 的关系：**可借鉴 disagreement 分析，但 TabACE 的成员共享参数，不能直接声称具有完整 model-ensemble uncertainty。

### 68. Diversity and Generalization in Neural Network Ensembles

[原论文](https://proceedings.mlr.press/v151/ortega22a.html)

ensemble 的收益经常被归因于 diversity，但不同论文使用互不一致的指标。该工作在平方损失、交叉熵和 0-1 loss 下统一分析 diversity 与 ensemble error，说明有效成员需要同时具备个体准确性和互补错误。

**与 TabACE 的关系：**为 pairwise disagreement、Q-statistic、KL diversity 与 gain 的相关分析提供依据。

### 69. Calibrated Ensembles Can Mitigate Accuracy Tradeoffs under Distribution Shift

[原论文](https://openreview.net/forum?id=S2EuSP8sqgc)

robust training 常提高 OOD accuracy，却牺牲 ID performance。论文发现，把标准模型和鲁棒模型先在 ID 数据上校准再组合，可以同时继承标准模型的 ID 优势与鲁棒模型的 OOD 优势。成功依赖成员校准和 OOD spurious features 的性质。

**与 TabACE 的关系：**与“保留 base clean performance，同时吸收 shift-aware context”高度相似，建议引用并报告 ID/OOD 双重权衡。

### 70. Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift

[原论文](https://openreview.net/forum?id=qUTiOeM57J)

未知 subpopulation proportions 改变时，平均表现好的分类器可能在少数群体上失败。该方法训练多个 prototype classifiers，并显式鼓励它们关注不同模式，使 ensemble 在没有 subgroup labels 的情况下覆盖潜在子群。

**与 TabACE 的关系：**prototype context 与 subpopulation coverage 有直接概念联系，但其成员是训练得到的多个分类器。

### 71. A Survey on Multi-view Learning

[原论文](https://arxiv.org/abs/1304.5634)

同一对象可由不同特征集合、模态或数据来源描述。Multi-view learning 的收益通常建立在两个原则上：不同 views 对正确结构应具有一定共识，同时每个 view 又包含其他 view 缺失的互补信息。若 views 只是重复或共同带偏，组合并不会自动改善。

**与 TabACE 的关系：**可用于形式化 signal preservation、consensus 和 complementarity，但要避免把人工 context views 误称为天然独立模态。

### 72. Wisdom of the Contexts: Active Ensemble Learning for Contextual Anomaly Detection

[原论文](https://arxiv.org/abs/2101.11560)

一个对象是否异常取决于所处 context，而真实数据中可能同时存在多种异常机制，没有单一 context 能发现全部异常。WisCon 从特征中自动生成多个 contexts，利用主动学习反馈估计重要性，并把不同 contextual detectors 组成 ensemble。

**与 TabACE 的关系：**“没有一个完美 context，需要利用多个 context 的智慧”与本文动机相近，但任务和 context 定义不同。

### 73. Context-based Ensemble Method for Human Energy Expenditure Estimation

[原论文](https://doi.org/10.1016/j.asoc.2015.05.001)

人体能量消耗与活动、身体状态和多种传感器条件有关。论文把不同 sensor features 分别视为 context，在每个 context value 下训练回归模型，再根据当前样本的多个 context 组装 ensemble。其目标是用多种情境视角提高估计精度。

**与 TabACE 的关系：**说明 “Multiple Contexts Ensemble” 名称早已出现，虽然算法、领域和 ICL 含义完全不同。

## 阅读顺序建议

如果目标是尽快写 TabACE 论文，建议按以下顺序阅读原文：

1. **BETA → DENSE → LARA → MixturePFN → TabPFN v2**：首先明确 novelty 边界；
2. **Understanding Context Sampling → LoCalPFN → TAAR → CRUMB → DistPFN**：建立 tabular context 与 shift 相关工作；
3. **Rethinking Demonstrations → Calibrate Before Use → Ordered Prompts → Sub-CP**：补充 context 信号、偏差和 diversity 动机；
4. **TableShift → TabFSBench → RLSbench → AdapTable**：规范 shift 定义和实验；
5. **Bagging → Diversity and Generalization → Calibrated Ensembles**：构建 ensemble 理论叙述。

## 使用这些意译时的注意事项

- 本文档用于理解和组织 Related Work，不能代替引用原论文。
- 不要把这里的中文句子写成“作者原话”；正式写作时应重新组织并附引用。
- 对 2026 年预印本、withdrawn submission 和 workshop papers，应在正文或参考文献中准确标注状态。
- 如果需要逐句精确翻译某篇论文，请把你合法获得的 PDF 或 Introduction 原文放入工作区；届时可以按你提供的文本逐段翻译。
