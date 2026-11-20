# V28 family-agreement risk router smoke notes

## 目标

V28 尝试把 V24/V27 的 no-shift feature-evidence 方法升级为多 context family 一致性路由器。候选 family 包括 boundary、diverse、prototype、clean 和 minority-density，并加入 minority-risk guard 与 risk score。

## 小规模验证

验证集：

- `openml_31_credit_g` seed 2, 3
- `openml_1050_pc3` seed 2, 3
- `openml_1498_sa_heart` seed 2, 3

这些样本覆盖一个 V24/V27 正例、一个极端少数类负例、一个 boundary 误触发负例。

## 结果

| 版本 | 平均 Delta Acc | 平均 Delta BAcc | 平均 Delta Macro-F1 | 平均 Delta Minority Recall | 结论 |
|---|---:|---:|---:|---:|---|
| r1 | -0.0026 | -0.0042 | -0.0037 | -0.0083 | `sa_heart seed=2` 仍掉点 |
| r2/r3 | -0.0026 | -0.0035 | -0.0047 | -0.0069 | 修掉 `sa_heart`，但新增 `credit_g seed=3` 掉点 |
| r4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 安全但过于保守，没有涨点 |

## 掉点原因

`credit_g seed=3` 的 r2/r3 掉点说明：多个 feature-evidence family 同意同一个候选类别，并不保证这个候选是真修正。该错误翻转伤害了少数类召回，说明 family agreement 本身仍会受到局部邻域偏差影响。

`sa_heart seed=2` 的 r1 掉点说明：如果只要求 family votes 和风险分数，boundary 型误触发仍可能通过，因此需要更严格的少数类风险保护。

## 当前处理

r4 将 baseline minority class 到 candidate majority class 的翻转全部拒绝，因此修掉了上述少数类召回风险。但这也导致方法没有任何有效改动，不能作为最终算法。

## 下一步

V28 不应继续扩大数据集测试。更合理的下一步是 V31 negative-control stability test：以 V27 的候选为主，不重新生成激进候选；对每个候选翻转做 context 删除、替换、重采样扰动，只保留稳定率足够高的候选。这样有机会保留 V27 的正收益，同时进一步过滤偶然翻转。
