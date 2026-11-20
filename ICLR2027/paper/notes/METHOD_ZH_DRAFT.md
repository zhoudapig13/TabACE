# 4 方法

> 对应英文源文件：`paper/sections/04_method.tex`  
> 说明：本稿用于中文审阅与后续论文修改。公式、符号和逻辑与英文 Method 保持一致；为便于和英文正文及方法图对应，保留 Base、Context、TFM 等术语。

## 4.1 方法概述

TabACE 在保持表格基础模型参数固定的前提下，采用一个三阶段推理流程。首先，一组可配置分支分别生成一个 Base（基准）输出和多个候选 Context（上下文）输出。其次，证据引导的聚合机制为各个 Context 分配相对权重，并将其合并为单一的聚合预测。最后，一个独立的融合系数决定该聚合预测应在多大程度上修正 Base。我们将后两个阶段分开，是因为 Context 之间的一致性能够说明不同候选证据应当如何组合，却不能单独决定这些证据是否足以取代 Base。当 Base 的不确定性较高，且多个 Context 一致支持一个置信度更高的候选结果时，聚合预测会获得更大的影响力；当 Base 较为自信，或 Context 证据较弱、彼此冲突时，最终预测则保持在更接近 Base 的位置。

## 4.2 可配置 Context 与输出接口

TabACE 使用一个可配置的推理时分支接口。记
$\Phi=\{\phi_0,\ldots,\phi_R\}$ 为分支配置集合，$F_{\phi_r}$ 为配置
$\phi_r$ 对应的预测器。对于查询 $x_i^t$，有

\[
o_i^{(r)}
=F_{\phi_r}(\mathcal D_s,\mathcal X_t,x_i^t;f_\theta),
\qquad r=0,\ldots,R.
\tag{1}
\]

基于 TFM 的分支首先构造
$C_i^{(r)}=G_{\phi_r}(\mathcal D_s,\mathcal X_t,x_i^t)$，随后计算
$f_\theta(C_i^{(r)},x_i^t)$；与该接口兼容的源端回归分支也可以直接输出一个标量预测。在分类任务中，$o_i^{(r)}=p_i^{(r)}$ 是类别概率向量；在点回归任务中，$o_i^{(r)}=\hat y_i^{(r)}$ 是标量。每个分支可以使用带标签的参考集、当前查询，以及其配置所规定的无标签目标域统计量，但不能使用目标标签。

配置集合 $\Phi$ 及其 Base 分支 $\phi_0$ 在目标域推理开始前，根据源端信息和应用知识确定。此后，各个分支可以使用其配置预先规定的无标签目标域统计量。

我们将 $\phi_0$ 预留给 Base 分支，并以 $o_i^{(0)}$ 表示其输出。其余配置生成候选输出库

\[
\mathcal O_i(\Phi)=\{o_i^{(r)}\}_{r=1}^{R}.
\tag{2}
\]

其中，$\Phi$ 定义分支规则，而 $\mathcal O_i(\Phi)$ 是这些规则在给定数据和查询上的具体输出。

## 4.3 证据引导的 Context 聚合

对于多 Context 分类（$R\geq2$），记 $p_i^{(0)}$ 为 Base 预测，
$\{p_i^{(r)}\}_{r=1}^{R}$ 为候选 Context 预测。每个候选 Context 都有一个从源端学习得到的先验权重 $\pi_r>0$，并满足
$\sum_{r=1}^{R}\pi_r=1$。由于同一 Context 对不同查询的有效性可能不同，我们进一步利用三个无需标签的成员级信号对这些全局先验进行调整。对于包含 $K$ 个类别的分类任务，定义

\[
\begin{aligned}
Q_i^{(r)}
  &=1-\frac{H\!\left(p_i^{(r)}\right)}{\log K},
&\mu_i&=\frac{1}{R}\sum_{s=1}^{R}p_i^{(s)},\\
S_i^{(r)}
  &=1-\frac{1}{2}\left\lVert p_i^{(r)}-\mu_i\right\rVert_1,
&D_i^{(r)}
  &=\frac{1}{2}\left\lVert p_i^{(r)}-p_i^{(0)}\right\rVert_1.
\end{aligned}
\tag{3}
\]

其中，$H(p)=-\sum_{k=1}^{K}p_k\log p_k$。$Q_i^{(r)}$、
$S_i^{(r)}$ 和 $D_i^{(r)}$ 分别衡量归一化熵置信度、该 Context 与候选平均预测的一致性，以及它与 Base 预测之间的距离。

记 $\boldsymbol\eta_Q$、$\boldsymbol\eta_S$ 和
$\boldsymbol\eta_D\in\mathbb R^R$ 为学习得到的系数向量。最终的查询条件化 Context 权重及聚合预测为

\[
\begin{aligned}
w_i^{(r)}
  &=\frac{\pi_r\exp\!\left(
      \eta_{Q,r}Q_i^{(r)}+
      \eta_{S,r}S_i^{(r)}+
      \eta_{D,r}D_i^{(r)}
    \right)}
    {\sum_{s=1}^{R}\pi_s\exp\!\left(
      \eta_{Q,s}Q_i^{(s)}+
      \eta_{S,s}S_i^{(s)}+
      \eta_{D,s}D_i^{(s)}
    \right)},\\
\bar p_i
  &=\sum_{r=1}^{R}w_i^{(r)}p_i^{(r)}.
\end{aligned}
\tag{4}
\]

因此，$\pi_r$ 表示源端对 Context $r$ 的全局偏好，而
$w_i^{(r)}$ 表示该 Context 对查询 $i$ 的最终权重。当所有证据系数均为零时，式（4）严格退化为 $w_i^{(r)}=\pi_r$。

对于点回归，每个配置分支产生一个标量预测。与任务相适配的策略 $\psi$ 将该预测库映射为

\[
\bar y_i
=\operatorname{Agg}_{\psi}
\!\left(\left\{\hat y_i^{(r)}\right\}_{r=1}^{R}\right).
\tag{5}
\]

策略 $\psi$ 可以指定参与聚合的成员、聚合算子，以及该算子所需的任意非负成员权重。式（5）不限定分支的固定数量或语义类型，因此可以通过同一标量预测接口加入新的 Context 分支或证据分支。

## 4.4 与 Base 预测的融合

Context 权重决定候选预测应当如何聚合，但不决定该聚合结果应在多大程度上修正 Base。对于分类任务，我们从置信度增益、跨 Context 一致性以及与 Base 的距离三个方面刻画这一候选更新：

\[
\begin{aligned}
G_i
  &=\max_y\bar p_i(y)-\max_y p_i^{(0)}(y),\\
A_i
  &=1-\frac{1}{2R}\sum_{r=1}^{R}
    \left\lVert p_i^{(r)}-\bar p_i\right\rVert_1,\\
B_i
  &=\frac{1}{2}\left\lVert\bar p_i-p_i^{(0)}\right\rVert_1.
\end{aligned}
\tag{6}
\]

$G_i$ 衡量聚合预测是否比 Base 更有置信度，$A_i$ 衡量各个 Context 在聚合预测周围的一致程度，$B_i$ 则衡量候选更新的幅度。随后，我们将查询相关的融合系数和最终分类预测定义为

\[
\begin{aligned}
\alpha_i
  &=\sigma\!\left(
    b+\gamma_GG_i+\gamma_AA_i+\gamma_BB_i
  \right),\\
p_i^{\mathrm{final}}
  &=(1-\alpha_i)p_i^{(0)}+\alpha_i\bar p_i,
\end{aligned}
\tag{7}
\]

其中，$b$ 为截距，
$\boldsymbol\gamma=(\gamma_G,\gamma_A,\gamma_B)$ 为融合系数。令
$\boldsymbol\gamma=\mathbf 0$，即可在同一接口内恢复固定融合系数
$\alpha_i=\sigma(b)$。

回归任务使用策略级融合系数，而不使用分类任务特有的查询统计量：

\[
\hat y_i^{\mathrm{final}}
=(1-\alpha)\hat y_i^{(0)}+\alpha\bar y_i,
\qquad \alpha\in[0,1].
\tag{8}
\]

对于由同一个回归策略处理的所有目标查询，均使用相同的融合系数。

### 仅使用源端数据的策略学习

我们将分类 Context 的先验参数化为
$\boldsymbol\pi=\operatorname{softmax}(\mathbf u)$，并将所有学习得到的分类参数记为
$\Theta=(\mathbf u,\boldsymbol\eta_Q,\boldsymbol\eta_S,
\boldsymbol\eta_D,b,\boldsymbol\gamma)$。对于回归任务，$\Theta$ 包含
$\operatorname{Agg}_{\psi}$ 中所有需要校准的参数以及 $\alpha$；预先给定的取值则视为固定参数。记 $\mathcal Q_s$ 为一个有限的源端内部校准查询集合。对于每个
$(x,y)\in\mathcal Q_s$，在构造各分支时隐藏标签 $y$，仅在计算校准损失时使用该标签。任务特定策略通过求解以下优化问题学习：

\[
\Theta^*
\in\arg\min_{\Theta}
\mathbb E_{(x,y)\sim\operatorname{Unif}(\mathcal Q_s)}
\left[
  \mathcal L_{\mathrm{cal}}
  \!\left(o^{\mathrm{final}}(x;\Theta),y\right)
\right].
\tag{9}
\]

其中，$\mathcal L_{\mathrm{cal}}$ 是与任务相适配的校准损失。学习得到的策略与 $\Phi$ 均在目标域推理开始前固定；若改变已配置的输出库，则需要重新校准相应的策略参数。

## 4.5 推理复杂度

记 $T_F^{(r)}$ 为生成分支 $r$ 输出的总成本，其中包括可能涉及的检索、变换或冻结模型的前向传播；记 $T_{\mathrm{agg}}$ 为任务特定的聚合与融合成本。每个查询的串行计算成本为

\[
T_{\mathrm{seq}}
=\mathcal O\!\left(
  \sum_{r=0}^{R}T_F^{(r)}+T_{\mathrm{agg}}
\right).
\tag{10}
\]

对于分类任务，计算 Context 均值、熵与总变差距离信号、加权聚合预测以及融合统计量的成本为
$T_{\mathrm{agg}}=\mathcal O(RK)$。常见的标量回归聚合器需要
$T_{\mathrm{agg}}=\mathcal O(R)$。

$R+1$ 个分支可以批量执行或并行求值。在计算资源充足时，对应的关键路径成本为

\[
T_{\mathrm{parallel}}
=\mathcal O\!\left(
  \max_{0\leq r\leq R}T_F^{(r)}+T_{\mathrm{agg}}
\right).
\tag{11}
\]

冻结的 TFM 参数 $\theta$ 只需存储一份；辅助回归分支则需要保存其自身的预测器状态。对于每个分类查询，分支预测和成员级信号需要
$\mathcal O((R+1)K+R)$ 的辅助内存；对于标量回归，辅助内存为
$\mathcal O(R)$。
