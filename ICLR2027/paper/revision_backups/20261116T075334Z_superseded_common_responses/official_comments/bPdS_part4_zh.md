## 第 4 部分：Folktables 配置、随机种子波动与偏移方法对照

> 在 Folktables 评估中，Specific 指定了哪个家族？作者能否报告不同随机种子下的方差，并与 DistPFN 和 Drift-Resilient TabPFN 比较？

感谢您建议补充实现细节与对照实验。我们在 Employment、Income 和 TravelTime 上评估 California → Michigan（2018）及 California 2014 → 2018。**Specific 采用源域到目标域的分布偏移配置**，保留五类 Shared 上下文并增加三类适配上下文：**Shared／Specific 分别使用五／八个替代分支，每个最多 512 行**，另加 Base。构造过程使用源标签与无标签目标特征。

**上下文构造（附录 E.5.1，第 40 页）。** 记 $z_i$ 为稳健标准化的源特征，$d_i$ 为到目标批次十近邻的平均平方距离。离群分数与源标签冲突度分别为：

$$
o_i=\frac{1}{p}\sum_{j=1}^{p}z_{ij}^{2},\qquad
\delta_i=1-\max_c\frac{1}{10}\sum_{k\in N_{10}(i)}\mathbf{1}[y_k=c].
$$

其中 $p$ 为特征数，$N_{10}(i)$ 为源域十近邻。除特别说明外，保留分数最低的源样本，再无放回抽样。

- **目标邻域（Shared／Specific）：** 由目标批次各维的 10%、30%、50%、70%、90% 分位点组成锚点集 $A_Q$，按 $s_i=\min_{a\in A_Q}\lVert z_i-a\rVert_2^2$ 排序，保留前 80%。
- **代表性样本（Shared／Specific）：** 按 $s_i=d_i+0.35o_i$ 排序，保留前 70%。
- **局部稳定（Shared／Specific）：** 按 $s_i=\delta_i+0.10d_i$ 排序，保留前 80%。
- **特征子空间（Shared／Specific）：** 从上下文与查询中同步移除 $j^*=\arg\max_j\lvert\operatorname{corr}(X_j,y)\rvert$，再抽取源样本。
- **低冲突（Shared／Specific）：** 按 $s_i=\delta_i+0.20o_i$ 排序，保留前 75%。
- **目标匹配 Bootstrap（Specific）：** 连续特征标准化、类别特征 One-hot 编码后，按 $w_i\propto\exp(-d_i^{\mathrm{mix}}/h)$ 有放回抽样；$d_i^{\mathrm{mix}}$ 为到目标批次十近邻的平均平方距离，$h$ 为正距离中位数，并保留类别覆盖。
- **密度比 Bootstrap（Specific）：** 使用逻辑回归域分类器估计 $g(x)$，按 $w_i\propto[\min\{\frac{g(x_i)}{1-g(x_i)}\frac{n_s}{n_t},10\}]^{1/2}$ 有放回抽样并保留类别覆盖；$n_s,n_t$ 为源域、目标域样本数。
- **连续特征对齐（Specific）：** 对预先指定的连续特征施加 $T(x)=0.5x+0.5[(x-m_s)s_t/s_s+m_t]$，其中 $m$ 为中位数，$s=\max(\mathrm{IQR},1)$。无放回抽取源样本，保留源标签与类别特征。

**偏移方法对照与不确定性汇总于 Official Comment C3。** Shared 和 Specific 在十二项 ID/OOD 对照中均优于 Base、DistPFN–TabICL-v2 [6] 与 Drift-Resilient TabPFN [7]。例如地域 TravelTime OOD，Shared／Specific 的准确率为 **0.5818 ± 0.0349／0.6012 ± 0.0062**，两种基线为 0.5802 ± 0.0071 与 0.5667 ± 0.0338。完整对照及 seed=0–4 的方差见附录 E.5.2–E.5.3（表 25–26，第 41 页）；单独的适配重复配置中，Specific 在全部 60 项配对评估中优于 Shared。

**参考文献**

[6] Lee. [Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](https://arxiv.org/abs/2605.04363). ICML, 2026.

[7] Helli et al. [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634). NeurIPS, 2024.
