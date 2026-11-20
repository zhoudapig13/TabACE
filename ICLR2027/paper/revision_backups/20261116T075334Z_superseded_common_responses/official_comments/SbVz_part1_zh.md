## 第 1 部分：超出校准覆盖与构造器匹配范围的泛化

> Shortcut 扰动破坏了预测能力最强的特征，而五种 Shared 分类上下文中的一种恰好直接删除了该特征。
>
> 加权策略是在与评估相同的扰动类型和强度上校准的。
> Specific 版本甚至被告知了当前出现的是哪一种扰动。
> TabACE 在校准阶段未见过的扰动上表现如何？

感谢您提出这一核心问题。**Official Comment C1** 汇总了 $2\times2$ Shortcut 消融与六种未知机制评估。在既没有 Shortcut 校准也没有匹配构造器时，D 的 **Acc/AUC 为 0.69333/0.84662**，Base 为 0.51719/0.65690。Shared 在六种未知机制的四项指标上也全部提升。**这些收益能够超出匹配修复与对评估机制的校准覆盖。**

Specific 在可获得家族知识时利用这些信息，与问题感知上下文构造 [5] 的思路一致；**Shared 不使用机制标识，在无法获得这些知识时仍然有效**。完整配置与结果见附录 E.3–E.4（表 22–24，第 39 页）。

**参考文献**

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.
