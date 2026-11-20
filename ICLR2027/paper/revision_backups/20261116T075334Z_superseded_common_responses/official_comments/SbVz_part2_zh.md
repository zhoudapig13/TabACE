## 第 2 部分：真实世界分布偏移下的鲁棒性

> 唯一的自然偏移基准是 Folktables，其中的提升大多较小，而且有一半情况下与未发生偏移的测试数据上的提升相近，提示这些收益可能来自普通集成效应。
> TabACE 能否应对其他真实世界的分布偏移，例如医疗领域中 eICU 与 MIMIC 之间的偏移？

感谢您建议扩展自然偏移测试。**Official Comment C3** 汇总了十二项 Folktables ID/OOD 对照与六项医疗设置。两个版本在十二项 Folktables 对照中均优于 Base、DistPFN–TabICL-v2 与 Drift-Resilient。例如，地域 TravelTime OOD 的 Base／Shared／Specific 准确率为 **0.5472/0.5818/0.6012**。

eICU/MIMIC 的授权尚未取得，因此我们在 PhysioNet Sepsis 2019 上测试双向医院系统迁移，并在 UCI Diabetes 130-US Hospitals 上测试入院来源迁移，采用 1:1 病例对照评估。**全部六项 ID/OOD 设置均满足 Specific > Shared > Base**；再入院 OOD 的 Base／Shared／Specific 为 **57.27%/58.97%/59.24%**。这些自然发生的偏移将证据延伸至 Folktables 之外（附录 E.5，表 25–27，第 41–42 页）。

普通集成效应的问题在**第 4 部分与 Official Comment C2** 中通过相同部署预测预算的对照回应。
