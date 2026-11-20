# v25 minority-aware context 小探针结果

本轮测试目标是解决 `pc3`、`planning_relax` 这类 Accuracy 不低但少数类召回很差的问题。新增 context family 为 `minority_contrast`：对每个测试样本构造“最近少数类锚点 + 当前 baseline 预测类对照 + KNN 补齐”的 feature-evidence context。

## 方法设计

v25 初版允许少数类锚点在低 margin、低 disagreement、context vote 一致时触发。初版发现一个问题：少数类锚点过强时，会把并非少数类塌缩的数据也拉向少数类，造成负迁移。

因此修复版加入两个门控：

- `minority collapse gate`：baseline 的少数类预测比例必须低于训练少数类比例的 70%。
- `candidate minority gate`：候选 context 的预测类别必须是训练集少数类。

## 初版与修复版平均表现

| 版本 | 方法 | 组数 | 平均 ΔAcc | 平均 ΔBAcc | 平均 ΔMacro-F1 | 平均 ΔMinority Recall | 平均 ΔECE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v25 初版 | minority router | 6 | +0.0000 | -0.0014 | -0.0012 | -0.0062 | -0.0049 | 有正例但出现负迁移 |
| v25 修复版 | minority router | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 消除负迁移，但过于保守 |
| v24 对照 | boundary-union router | 6 | +0.0078 | +0.0068 | +0.0076 | +0.0062 | +0.0009 | 当前更好 |

## 逐数据集表现

| 数据集 | Baseline Acc | v25 初版 ΔAcc | v25 修复版 ΔAcc | v24 ΔAcc | 现象 |
|---|---:|---:|---:|---:|---|
| openml_1050_pc3 | 0.9531 | +0.0000 | +0.0000 | +0.0000 | 极端少数类，当前少数类锚点仍未找到可安全翻转样本 |
| openml_1490_planning_relax | 0.6296 | +0.0000 | +0.0000 | +0.0000 | baseline 偏单类，但 context 候选证据不足 |
| openml_1464_blood_transfusion | 0.3750 | +0.0000 | +0.0000 | +0.0000 | batch prior JS 高，no-shift 分支被拦截 |
| openml_31_credit_g | 0.7344 | +0.0000 | +0.0000 | +0.0156 | v24 boundary 更适合 |
| openml_37_diabetes | 0.7500 | -0.0156 | +0.0000 | +0.0156 | 初版误触发，修复版成功拦住 |
| openml_43_haberman | 0.7344 | +0.0156 | +0.0000 | +0.0156 | 修复版拦掉了正例，说明门控过紧 |

## 原因分析

v25 初版的问题很明确：`minority_contrast` 会故意提高少数类样本在 context 中的比例，如果没有额外约束，它可能把本来正确的多数类边界样本翻成少数类。`diabetes` 就是这个反例，Accuracy、Balanced Accuracy、Macro-F1 都下降。

修复版加入 collapse gate 后，`diabetes` 的掉点被消除，但 `haberman` 的正向样本也被拦住。这说明“少数类预测比例塌缩”这个 batch 级条件太粗，不能单独判断一个样本是否应该被少数类 context 修正。

当前结论是：v25 的 idea 有研究价值，但还不是主算法。下一步如果继续这条线，应做 sample-level minority evidence，而不是 batch-level collapse gate，例如：

- 计算测试样本到少数类原型的相对距离；
- 比较少数类近邻密度与 baseline 类近邻密度；
- 只在“少数类距离优势明显、且 boundary/prototype context 同时支持”时触发；
- 保持每个 batch 最多翻转 1 个样本，避免牺牲 Accuracy。

当前主线仍应以 v24 boundary-union 为主，因为它在不饱和探针中已经做到 3 个正向组、0 个负向组。

Artifacts:

- `apa_v25_minority_probe_detail.csv`
- `apa_v25_minority_probe_summary.csv`
- `apa_v25_minority_probe_r2_detail.csv`
- `apa_v25_minority_probe_r2_summary.csv`
