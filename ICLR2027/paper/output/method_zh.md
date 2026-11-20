# 方法

## 问题设定与方法概览

令

$$
\mathcal D_s=\{(x_j^s,y_j^s)\}_{j=1}^{n_s}
$$

表示带标签的参考数据，令

$$
\mathcal X_t=\{x_i^t\}_{i=1}^{n_t}
$$

表示一次测试调用中可用的无标签目标查询，其中 $x_j^s,x_i^t\in\mathbb R^d$。分类任务的标签空间为 $\mathcal Y=\{1,\ldots,L\}$，回归任务的标签空间为 $\mathcal Y\subseteq\mathbb R$。冻结的表格基础模型 $f_\theta$ 根据构造得到的带标签上下文对每个查询进行预测。上下文由源样本构成；对于预先登记的选择偏移构造器，上下文还可以包含带有伪标签的目标样本。真实目标标签仅用于评估，测试时 $f_\theta$ 保持冻结。

参考样本与目标样本分别服从 $P_s(X,Y)$ 和 $P_t(X,Y)$。我们研究可访问上下文的受控变化以及满足 $P_t\neq P_s$ 的分布偏移。推理时，每个构造器使用 $\mathcal D_s$、当前查询，以及其预先登记操作所需的目标特征。

如图 1 所示，TabACE 包含一个 Base 分支和多个替代上下文分支，同时保持 $f_\theta$ 冻结。置信度与跨上下文一致性决定各替代分支的相对影响；任务相关门控和聚合上下文支持度决定相对于 Base 的更新幅度。因此，更新强度由当前查询能够提供的证据决定。

## 可配置上下文与输出接口

令 $\phi_0$ 表示 Base 配置，令

$$
\Phi=\{\phi_1,\ldots,\phi_R\}
$$

表示在目标推理前依据源侧信息和应用知识选定的替代分支配置。记其索引集合为 $\mathcal I(\Phi)=\{1,\ldots,R\}$。令 $F_{\phi_r}$ 表示由配置 $\phi_r$ 诱导的完整预测映射。对于一个可用查询 $x_i^t$，有

$$
o_i^{(r)}
=F_{\phi_r}(\mathcal D_s,\mathcal X_t,x_i^t;f_\theta),
\qquad r=0,\ldots,R.
\tag{1}
$$

其中，参数 $\mathcal X_t$ 提供配置 $\phi_r$ 所需的目标子集；仅使用源数据的构造器会忽略该参数。

对于表格基础模型分支，该映射由可配置上下文构造器与冻结预测器复合得到。共享上下文构造器首先形成

$$
C^{(r)}=G_{\phi_r}(\mathcal D_s,\mathcal X_t),
$$

并在不同查询之间复用该上下文，因此

$$
F_{\phi_r}(\mathcal D_s,\mathcal X_t,x_i^t;f_\theta)
=f_\theta(C^{(r)},x_i^t).
$$

查询相关构造器则形成

$$
C_i^{(r)}=G_{\phi_r}(\mathcal D_s,\mathcal X_t,x_i^t).
$$

兼容的传统预测器把构造得到的上下文作为对应分支的训练数据，并使用相同的输出接口。对于分类任务，$o_i^{(r)}=p_i^{(r)}\in\Delta^{L-1}$ 是类别概率向量；对于点回归任务，$o_i^{(r)}=\hat y_i^{(r)}$ 是标量。

记 $o_i^{(0)}$ 为 Base 输出。其构造器 $G_{\phi_0}$ 遵循相应 backbone 的标准采样规则，利用可用的带标签参考数据形成规范上下文，并保持所选特征及其标签不变。对于 $r\geq1$，$G_{\phi_r}$ 通过选择或重加权参考样本、选择或变换特征，或者应用预先规定的源数据响应调整来构造互补上下文。

选择偏移构造器还可以为目标 peer 附加 Base 伪标签。对于查询 $x_i^t$，这些样本满足

$$
\widehat{\mathcal D}_{t,-i}^{(0)}
\subseteq\{(x_j^t,\widetilde y_j^{(0)}):j\neq i\},
$$

其中

$$
\widetilde y_j^{(0)}
=\arg\max_{\ell\in\mathcal Y}p_{j,\ell}^{(0)}.
$$

当前查询不包含在该 peer 集合中。构造器使用参考信息；预先登记的目标相关构造器还可以使用由其可用目标子集计算得到的统计量。所有构造器均无法访问目标标签。当构造器选择特征子集时，相同的投影同时作用于上下文和查询。最终得到替代输出库

$$
\mathcal O_i(\Phi)=\{o_i^{(r)}:r\in\mathcal I(\Phi)\}.
\tag{2}
$$

其中，$\Phi$ 指定上下文构造规则，$\mathcal O_i(\Phi)$ 汇集这些规则针对查询 $i$ 产生的输出。

## 基于置信度与一致性的上下文聚合

对于查询 $x_i^t$，$R$ 个替代上下文产生式（2）中的输出。我们构造与任务相适配的支持分数，并使用统一的归一化加权规则聚合这些预测。

对于分类任务，令 $p_i^{(r)}\in\Delta^{L-1}$ 表示上下文 $r$ 的预测。给定从源数据学习得到的 $\eta_{Q,r},\eta_{S,r}\geq0$，计算

$$
\begin{aligned}
\mu_i&=\frac{1}{R}\sum_{s=1}^{R}p_i^{(s)},
&Q_i^{(r)}&=1-\frac{H(p_i^{(r)})}{\log L},\\
S_i^{(r)}&=1-\frac12\lVert p_i^{(r)}-\mu_i\rVert_1,
&\rho_i^{(r)}&=\eta_{Q,r}Q_i^{(r)}+\eta_{S,r}S_i^{(r)}.
\end{aligned}
\tag{3}
$$

其中

$$
H(p)=-\sum_{\ell=1}^{L}p_\ell\log p_\ell.
$$

$Q_i^{(r)}$ 衡量归一化熵置信度，$S_i^{(r)}$ 衡量该预测与替代分支平均预测之间的一致性；两者较高时，$\rho_i^{(r)}$ 相应增大。

对于回归任务，令 $\hat y_i^{(r)}\in\mathbb R$ 表示上下文 $r$ 的预测。定义

$$
\begin{aligned}
m_i&=\operatorname{median}\{\hat y_i^{(s)}\}_{s=1}^{R},
&S_i^{(r)}&=\exp\!\left(-\frac{|\hat y_i^{(r)}-m_i|}{\tau_y}\right),\\
\rho_i^{(r)}&=\eta_{S,r}^{\mathrm{reg}}S_i^{(r)},
\qquad &\eta_{S,r}^{\mathrm{reg}}&\geq0.
\end{aligned}
\tag{4}
$$

中位数 $m_i$ 是稳健中心，$\tau_y>0$ 由源数据确定；距离该中心较远的预测获得较低的支持度。

两个任务随后使用相同的归一化加权与聚合规则：

$$
\begin{aligned}
w_i^{(r)}&=\frac{\pi_r e^{\rho_i^{(r)}}}
{\sum_{s=1}^{R}\pi_s e^{\rho_i^{(s)}}},\\
\bar o_i&=\sum_{r=1}^{R}w_i^{(r)}o_i^{(r)},
\qquad \sum_{r=1}^{R}w_i^{(r)}=1.
\end{aligned}
\tag{5}
$$

其中，$\pi_r>0$ 是由源数据学习得到的上下文 $r$ 的先验可靠性，并满足 $\sum_r\pi_r=1$。因此，对于分类任务，$(o_i^{(r)},\bar o_i)=(p_i^{(r)},\bar p_i)$；对于回归任务，$(o_i^{(r)},\bar o_i)=(\hat y_i^{(r)},\bar y_i)$。随后，将聚合结果 $\bar o_i$ 与 Base 进行整合。

## 与 Base 的查询相关整合

我们利用任务相关门控 $U_i$ 和聚合上下文支持度 $E_i$，将 $\bar o_i$ 与 Base 整合。对于分类任务，$U_i$ 表示 Base 的内在不确定性；对于回归任务，$U_i$ 表示 Base 与聚合结果之间的差异；$E_i$ 衡量替代分支的支持程度：

$$
\begin{aligned}
U_i^{\mathrm{cls}}&=\frac{H(p_i^{(0)})}{\log L},
&E_i^{\mathrm{cls}}&=\sum_{r=1}^{R}w_i^{(r)}Q_i^{(r)}S_i^{(r)},\\
U_i^{\mathrm{reg}}&=1-\exp\!\left(-\frac{|\bar y_i-\hat y_i^{(0)}|}{\tau_y}\right),
&E_i^{\mathrm{reg}}&=\sum_{r=1}^{R}w_i^{(r)}S_i^{(r)}.
\end{aligned}
\tag{6}
$$

下文中，$U_i$ 和 $E_i$ 表示当前任务对应的这一组量。

两个任务使用相同的整合规则：

$$
\alpha_i=\alpha_{\max}U_iE_i,
\qquad
o_i^{\mathrm{final}}
=(1-\alpha_i)o_i^{(0)}+\alpha_i\bar o_i.
\tag{7}
$$

由于 $U_i,E_i,\alpha_{\max}\in[0,1]$，当任务相关门控或上下文支持度较低时，更新会保持接近 Base。

### 外层折策略学习

由外层训练数据集构成的校准 episode 用于拟合上下文先验、评分系数、输出尺度和整合上限。每个 episode 将一个带标签源上下文与一个伪目标特征集合配对，其标签仅进入校准损失。拟合后的策略与 $\Phi$ 在 held-out 推理前固定。
