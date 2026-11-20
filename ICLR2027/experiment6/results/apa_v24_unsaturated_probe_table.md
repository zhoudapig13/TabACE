# v24 不饱和数据集 no-shift 探针结果

本轮只测试 shift=0 的原始数据，不做人为 label shift。选择 8 个 baseline 尚未饱和或类别边界可能较难的数据集，每个数据集先跑 1 个 seed，用于判断新提出的 boundary-union 是否值得继续扩大测试。

## 方法平均表现

| 方法 | 组数 | 平均 Accuracy | 平均 ΔAcc | 平均 ΔBalanced Acc | 平均 ΔMacro-F1 | 平均 ΔMinority Recall | 平均 ΔECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabICL baseline | 8 | 0.7018 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| boundary router | 8 | 0.7057 | +0.0039 | +0.0028 | +0.0033 | +0.0000 | -0.0012 |
| v23 union router | 8 | 0.7037 | +0.0020 | +0.0023 | +0.0024 | +0.0046 | +0.0018 |
| v24 boundary-union router | 8 | 0.7076 | +0.0059 | +0.0051 | +0.0057 | +0.0046 | +0.0004 |

## 逐数据集表现

| 数据集 | Baseline Acc | boundary ΔAcc | v23 ΔAcc | v24 ΔAcc | v24 ΔBAcc | v24 ΔMacro-F1 | v24 ΔECE | 现象 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| openml_31_credit_g | 0.7344 | +0.0156 | +0.0000 | +0.0156 | +0.0116 | +0.0137 | +0.0049 | boundary 有效，v24 成功合入 |
| openml_37_diabetes | 0.7500 | +0.0000 | +0.0156 | +0.0156 | +0.0185 | +0.0193 | +0.0162 | diverse/prototype 证据有效，v24 保留 v23 收益 |
| openml_1480_ilpd | 0.7344 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0008 | 有候选概率扰动，但未改变预测类别 |
| openml_43_haberman | 0.7344 | +0.0156 | +0.0000 | +0.0156 | +0.0104 | +0.0128 | -0.0159 | boundary 有效，v24 成功合入 |
| openml_1498_sa_heart | 0.7031 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0014 | 低 margin 样本证据不足，未触发有效类别翻转 |
| openml_1464_blood_transfusion | 0.3750 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | batch prior shift 较强，no-shift 分支被安全拦截 |
| openml_1050_pc3 | 0.9531 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | Accuracy 已高但 minority recall 为 0，当前 accuracy 导向门控不适合解决少数类 |
| openml_1490_planning_relax | 0.6296 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | baseline 预测偏单类，当前局部证据不支持安全翻转 |

## 正负迁移统计

| 方法 | 正向组数 | 负向组数 | 不变组数 | 说明 |
|---|---:|---:|---:|---|
| boundary router | 2 | 0 | 6 | 在 credit_g、haberman 上涨点 |
| v23 union router | 1 | 0 | 7 | 在 diabetes 上涨点 |
| v24 boundary-union router | 3 | 0 | 5 | 合并 boundary 与 v23 的互补收益，当前没有掉点 |

## 初步原因分析

v24 的收益来自两类互补证据。第一类是 boundary contrast：在 `credit_g` 和 `haberman` 上，普通 KNN 或 diverse context 没有触发，但 top-1 类和 top-2 竞争类的对比 context 能修正一个边界样本，因此 Accuracy +0.0156。第二类是 diverse/prototype/subspace 证据：在 `diabetes` 上，v23 已经能修正一个样本，v24 保留了这个收益。

没有涨点的数据不能简单解释为算法无效。`blood_transfusion` 的 batch prior JS 约为 0.0374，高于 no-shift 分支阈值 0.012，所以当前分支被安全拦截；这类数据应该交给 shift 分支或重新设计自然 shift 检测。`pc3` 的 Accuracy 很高但少数类召回为 0，单纯追求 Accuracy 的 no-shift 边界修正不一定能解决，后续应单独做 minority-aware context。`planning_relax` 也是类似问题，baseline 几乎偏向单类，当前局部证据不足以支持安全翻转。

下一步更值得做的是：在 v24 基础上新增 minority-aware boundary/prototype context，只对少数类召回为 0 或极低、但 Accuracy 已经较高的数据生效；同时继续保持 changed fraction 很小，避免为了召回牺牲整体 Accuracy。
