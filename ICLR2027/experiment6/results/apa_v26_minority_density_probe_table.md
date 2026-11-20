# v26 sample-level minority-density context 小探针结果

本轮继续 v25 的少数类方向，但把 batch-level collapse gate 改成 sample-level minority evidence。新增方法为 `apa_cf_no_shift_minority_density_router`，核心判断是：只有当某个测试样本在特征空间中更接近少数类锚点，并且 `minority_contrast` context 的候选预测也指向少数类时，才允许小比例翻转。

## 方法平均表现

| 方法 | 组数 | 平均 ΔAcc | 平均 ΔBalanced Acc | 平均 ΔMacro-F1 | 平均 ΔMinority Recall | 平均 ΔECE |
|---|---:|---:|---:|---:|---:|---:|
| v26 minority-density router | 6 | +0.0026 | +0.0031 | +0.0032 | +0.0062 | +0.0030 |
| v24 boundary-union router | 6 | +0.0078 | +0.0068 | +0.0076 | +0.0062 | +0.0009 |
| TabICL baseline | 6 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## 逐数据集表现

| 数据集 | Baseline Acc | v26 ΔAcc | v26 ΔBAcc | v24 ΔAcc | 现象 |
|---|---:|---:|---:|---:|---|
| openml_1050_pc3 | 0.9531 | +0.0000 | +0.0000 | +0.0000 | 极端少数类，仍未找到可安全翻转样本 |
| openml_1490_planning_relax | 0.6296 | +0.0000 | +0.0000 | +0.0000 | 只有轻微概率扰动，未改变预测类别 |
| openml_1464_blood_transfusion | 0.3750 | +0.0000 | +0.0000 | +0.0000 | batch prior JS 高，no-shift 分支被拦截 |
| openml_31_credit_g | 0.7344 | +0.0000 | +0.0000 | +0.0156 | boundary-union 更适合 |
| openml_37_diabetes | 0.7500 | +0.0156 | +0.0185 | +0.0156 | v26 找到正确少数类边界翻转 |
| openml_43_haberman | 0.7344 | +0.0000 | +0.0000 | +0.0156 | v26 未触发，v24 boundary 更适合 |

## 与 v25 的关系

v25 初版在 `diabetes` 上出现 -0.0156 的负迁移，因为少数类锚点过强，误把正确样本翻错。v26 使用样本级少数类距离优势后，`diabetes` 不再掉点，反而 +0.0156，说明 sample-level evidence 比 batch-level collapse gate 更合理。

但 v26 仍没有解决 `pc3` 和 `planning_relax`。这说明极端少数类召回问题不能只靠“少数类更近”这一条证据，还需要更强的局部密度或验证集信号。

## 当前判断

v26 是一个有价值的旁路：它更快、逻辑更清晰，也比 v25 初版安全。但从当前数据看，主算法仍应是 v24 boundary-union，因为 v24 在同一批数据上有 2 个正向组，而 v26 只有 1 个。

后续如果继续少数类方向，建议不要再单独扩大 v26，而是把 v26 作为 v24 的一个低优先级候选：只有当 boundary-union 不触发、且 minority-distance evidence 很强时，才尝试 v26。

Artifacts:

- `apa_v26_minority_density_probe_detail.csv`
- `apa_v26_minority_density_probe_summary.csv`
- `apa_v26_minority_density_probe_report.md`
