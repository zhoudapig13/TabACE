## 第 3 部分：超出扰动与构造器匹配范围的泛化

> 扰动似乎与相应构造器紧密匹配：Shortcut 损坏最重要的特征，而其构造器删除相关性最高的特征；回归 Concept 引入平滑漂移，而其构造器移除最强的平滑分量。移除这些匹配构造器后，收益会减小；当偏移被 Shared 上下文库覆盖时，收益最明显。
> 缺乏这种匹配时，即扰动／偏移没有被纳入 Shared 构造器库时，性能几乎与 Base 相同。在偏移并非人为设计的 Folktables 设置中，收益也很小。这些结果令人担忧：方法能否泛化到这些专门匹配的扰动和构造器之外？

感谢您提出这一疑虑。**Official Comment C1** 通过 $2\times2$ Shortcut 实验区分校准覆盖与构造器匹配。移除匹配构造器的 C/D，其 **Acc/AUC 为 0.71187/0.85757 与 0.69333/0.84662**，Base 为 0.51719/0.65690；D 还从校准中排除 Shortcut。

更直接地，**Shared 在六种未参与校准且没有专门匹配构造器的机制上，Acc 和 $R^2$ 均有提升**（C1；附录 E.4、表 23–24）。这回应了没有匹配时性能几乎等于 Base 的判断。Specific 在可获得问题知识时加以利用，与问题感知上下文构造 [5] 一致，而 **Shared 无需机制标识**。

我们认同 Folktables 收益因任务和迁移设置而异。**C3** 增加医院系统与入院来源偏移：六项医疗 ID/OOD 设置均有提升；1:1 病例对照下，再入院 OOD 的 Base／Shared／Specific 为 **57.27%/58.97%/59.24%**。这些测试共同将证据延伸至专门匹配的合成机制之外。

**参考文献**

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.
