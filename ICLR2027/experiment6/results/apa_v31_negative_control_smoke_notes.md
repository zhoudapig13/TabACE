# V31 negative-control stability smoke notes

## 目标

V31 的目标是把 V27 的候选翻转再经过负控稳定性测试：对 context 做轻微扰动，只有候选类别在扰动后仍稳定出现时才保留。

## smoke 结果

验证集为：

- `openml_31_credit_g` seed 2, 3
- `openml_1050_pc3` seed 2, 3
- `openml_1498_sa_heart` seed 2, 3

第一版 V31 使用高风险候选稳定率阈值 0.50。结果在 `openml_31_credit_g seed=3` 上出现准确率不变但 Balanced Accuracy 和 Macro-F1 下降的问题。该错误说明：只对高风险候选做稳定性测试不够，且少数类到多数类的翻转会伤害平衡指标。

第二版将阈值提高到 0.75 后，问题仍存在，说明该坏翻转没有被当前 high-risk 标记充分覆盖。

当前修正版要求所有候选都满足稳定率至少 0.75，并且直接拒绝 baseline minority class 到 candidate majority class 的翻转。在 `openml_31_credit_g seed=3` 上，修正版结果为：

| 数据集 | seed | Delta Acc | Delta BAcc | Delta Macro-F1 |
|---|---:|---:|---:|---:|
| `openml_31_credit_g` | 3 | +0.0000 | +0.0000 | +0.0000 |

## 结论

当前 V31 能拦住已知坏翻转，但也过滤掉了 V27 的正收益，因此不能作为比 V27 更强的主方法。它的主要价值是证明了一个重要风险：稳定性测试如果只看候选类别重复出现，仍可能保留伤害少数类的翻转；必须结合 minority-risk guard。

## 后续方向

暂时不扩测 V31。下一步更合理的是：

- 将 V31 改成候选级扰动，只对 V27 已经选中的少量候选重跑 context，降低运行成本。
- 把稳定性测试作为 V27 的附加诊断，而不是替换 V27 的候选选择。
- 对少数类到多数类的翻转默认拒绝；除非后续用 leave-one-out meta-router 证明它在训练内部是稳定正收益。
