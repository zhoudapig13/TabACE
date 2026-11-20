## Official Comment C3：自然偏移、医疗迁移与重复实验

我们扩展了 Employment、Income 和 TravelTime 的自然偏移对照：**California → Michigan（2018）**的地域迁移，以及 **California 2014 → 2018** 的时间迁移，并比较 DistPFN–TabICL-v2 [6] 与 Drift-Resilient TabPFN [7]。**Specific 使用源域到目标域的分布偏移配置**；其八类替代上下文保留五类 Shared 构造并增加三类适配构造，每类最多 512 行，另加 Base。构造使用源标签及无标签目标特征。附录 E.5.1 列出具体规则；E.5.2、表 25（第 41 页）给出对照。含 ± 的单元格表示准确率与样本标准差。

| 场景 | 任务 | 域 | Base | DistPFN–TabICL v2 | Drift-Resilient | Shared | Specific |
|---|---|---|---:|---:|---:|---:|---:|
| 地域 | Employment | ID | 0.7922 | 0.7892 ± 0.0030 | 0.7885 ± 0.0071 | **0.7952 ± 0.0031** | **0.8089 ± 0.0042** |
| 地域 | Employment | OOD | 0.8043 | 0.8034 ± 0.0033 | 0.8022 ± 0.0067 | **0.8106 ± 0.0047** | **0.8245 ± 0.0040** |
| 地域 | Income | ID | 0.7888 | 0.7930 ± 0.0100 | 0.7856 ± 0.0147 | **0.8118 ± 0.0020** | **0.8201 ± 0.0003** |
| 地域 | Income | OOD | 0.7655 | 0.7768 ± 0.0288 | 0.7733 ± 0.0209 | **0.7823 ± 0.0063** | **0.7914 ± 0.0029** |
| 地域 | TravelTime | ID | 0.5912 | 0.5731 ± 0.0253 | 0.5723 ± 0.0252 | **0.6225 ± 0.0212** | **0.6569 ± 0.0101** |
| 地域 | TravelTime | OOD | 0.5472 | 0.5802 ± 0.0071 | 0.5667 ± 0.0338 | **0.5818 ± 0.0349** | **0.6012 ± 0.0062** |
| 时间 | Employment | ID | 0.7704 | 0.7770 ± 0.0043 | 0.7699 ± 0.0095 | **0.7861 ± 0.0028** | **0.7994 ± 0.0031** |
| 时间 | Employment | OOD | 0.7847 | 0.7804 ± 0.0090 | 0.7742 ± 0.0140 | **0.7922 ± 0.0053** | **0.8030 ± 0.0015** |
| 时间 | Income | ID | 0.7926 | 0.8009 ± 0.0050 | 0.8010 ± 0.0055 | **0.8126 ± 0.0042** | **0.8240 ± 0.0011** |
| 时间 | Income | OOD | 0.7844 | 0.7894 ± 0.0040 | 0.7867 ± 0.0054 | **0.8000 ± 0.0027** | **0.8106 ± 0.0031** |
| 时间 | TravelTime | ID | 0.5951 | 0.5937 ± 0.0235 | 0.5956 ± 0.0145 | **0.6237 ± 0.0076** | **0.6526 ± 0.0088** |
| 时间 | TravelTime | OOD | 0.5908 | 0.5731 ± 0.0309 | 0.5692 ± 0.0179 | **0.6045 ± 0.0126** | **0.6350 ± 0.0114** |

**在全部十二项 ID/OOD 对照中，Shared 和 Specific 均优于 Base 与两种基线，Specific 始终最优。** 附录 E.5.3、表 26（第 41 页）单独报告源域监督适配配置在 **seed=0–4**、固定划分和超参数下的方差；Specific 在全部 60 项配对评估中优于 Shared。

**医疗偏移（附录 E.5.4，表 27，第 42 页）。** eICU/MIMIC 的访问授权尚未取得，因此我们测试 PhysioNet Sepsis 2019 的医院系统 A→B、B→A 迁移，以及 UCI Diabetes 130-US Hospitals 的再入院任务。再入院按 TableShift 的定义，以非急诊入院来源为 ID、急诊入院为 OOD，准确率采用 **1:1 病例对照评估**。

| 数据集 | 任务 | 域 | Base | Shared | Specific |
|---|---|---|---:|---:|---:|
| PhysioNet Sepsis 2019 | Sepsis A $\rightarrow$ B | ID | 70.20% | **70.71%** | **71.72%** |
| PhysioNet Sepsis 2019 | Sepsis A $\rightarrow$ B | OOD | 61.59% | **62.14%** | **63.04%** |
| PhysioNet Sepsis 2019 | Sepsis B $\rightarrow$ A | ID | 76.19% | **77.38%** | **78.57%** |
| PhysioNet Sepsis 2019 | Sepsis B $\rightarrow$ A | OOD | 63.79% | **64.60%** | **64.91%** |
| UCI Diabetes 130-US Hospitals | 再入院 | ID | 54.82% | **55.26%** | **56.87%** |
| UCI Diabetes 130-US Hospitals | 再入院 | OOD | 57.27% | **58.97%** | **59.24%** |

**全部六项医疗 ID/OOD 设置均满足 Specific > Shared > Base。** 这些评估将证据延伸至自然发生的医院系统与入院来源偏移，无需人为构造所评估的扰动。

**参考文献**

[6] Lee. [Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](https://arxiv.org/abs/2605.04363). ICML, 2026.

[7] Helli et al. [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634). NeurIPS, 2024.
