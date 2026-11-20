# v24 不饱和数据集 3-seed 稳定性结果

本表合并了两批结果：

- seed=1：`apa_v24_unsaturated_probe_detail.csv`
- seed=2,3：`apa_v24_unsaturated_stability_seed23_detail.csv`

所有结果都是 shift=0 原始数据，不做人为 label shift。`Δ` 表示 v24 boundary-union router 相对原始 TabICL baseline 的变化。

## 总体结果

| 组数 | 平均 ΔAcc | 平均 ΔBalanced Acc | 平均 ΔMacro-F1 | 平均 ΔMinority Recall | 平均 ΔECE | 正向组数 | 负向组数 | 不变组数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | +0.0059 | +0.0020 | +0.0022 | -0.0038 | -0.0031 | 8 | 2 | 14 |

## 逐数据集 3-seed 表现

| 数据集 | seeds | Baseline Acc 均值 | v24 Acc 均值 | 平均 ΔAcc | 平均 ΔBAcc | 平均 ΔF1 | 平均 ΔECE | 正/负/零 | seed 明细 ΔAcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `openml_1050_pc3` | 3 | 0.9219 | 0.9167 | -0.0052 | -0.0278 | -0.0310 | -0.0040 | 0/1/2 | s1:+0.0000, s2:-0.0156, s3:+0.0000 |
| `openml_1464_blood_transfusion` | 3 | 0.7031 | 0.7031 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/0/3 | s1:+0.0000, s2:+0.0000, s3:+0.0000 |
| `openml_1480_ilpd` | 3 | 0.6927 | 0.7083 | +0.0156 | +0.0151 | +0.0173 | -0.0062 | 2/0/1 | s1:+0.0000, s2:+0.0312, s3:+0.0156 |
| `openml_1490_planning_relax` | 3 | 0.6296 | 0.6296 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/0/3 | s1:+0.0000, s2:+0.0000, s3:+0.0000 |
| `openml_1498_sa_heart` | 3 | 0.7240 | 0.7188 | -0.0052 | -0.0083 | -0.0075 | -0.0059 | 0/1/2 | s1:+0.0000, s2:-0.0156, s3:+0.0000 |
| `openml_31_credit_g` | 3 | 0.7135 | 0.7292 | +0.0156 | +0.0122 | +0.0124 | +0.0063 | 2/0/1 | s1:+0.0156, s2:+0.0000, s3:+0.0312 |
| `openml_37_diabetes` | 3 | 0.7344 | 0.7500 | +0.0156 | +0.0175 | +0.0184 | -0.0045 | 2/0/1 | s1:+0.0156, s2:+0.0312, s3:+0.0000 |
| `openml_43_haberman` | 3 | 0.7344 | 0.7448 | +0.0104 | +0.0071 | +0.0082 | -0.0108 | 2/0/1 | s1:+0.0156, s2:+0.0156, s3:+0.0000 |

## 正向数据集

| 数据集 | 平均 ΔAcc | 正向 seed | 说明 |
|---|---:|---:|---|
| `openml_31_credit_g` | +0.0156 | 2/3 | boundary contrast 在多个 split 中有效 |
| `openml_37_diabetes` | +0.0156 | 2/3 | diverse/prototype 与 minority-density 型证据均能找到边界样本 |
| `openml_1480_ilpd` | +0.0156 | 2/3 | seed=1 不触发，但 seed=2/3 明显涨点 |
| `openml_43_haberman` | +0.0104 | 2/3 | boundary contrast 稳定有效 |

## 负向数据集

| 数据集 | seed | ΔAcc | ΔBAcc | ΔMacro-F1 | 可能原因 |
|---|---:|---:|---:|---:|---|
| `openml_1050_pc3` | 2 | -0.0156 | -0.0833 | -0.0929 | 极端少数类数据，单个翻转会显著伤害 Balanced Accuracy |
| `openml_1498_sa_heart` | 2 | -0.0156 | -0.0250 | -0.0224 | boundary 证据误触发，候选翻转不是稳定边界修正 |

## 结论

v24 在 3-seed 检查中仍保持平均 Accuracy 正向，且正向组数多于负向组数。但它已经不再是“完全无负迁移”：`pc3 seed=2` 和 `sa_heart seed=2` 暴露出两个需要保护的场景。

下一步不应该继续盲目扩大 v24，而应新增 v27 风险保护：

- 对训练集极端不平衡、测试集少数类很少的数据，限制或禁止会伤害少数类预测的翻转。
- 如果候选修改只改变 1 个样本，但该数据的 minority recall 已经很脆弱，需要额外要求 boundary 与 prototype 两路同时支持。
- 对 `sa_heart` 这类误触发，需要加入 context family agreement：boundary、diverse、prototype 至少两路支持同一翻转时才允许改。

Artifacts:

- `apa_v24_unsaturated_probe_detail.csv`
- `apa_v24_unsaturated_stability_seed23_detail.csv`
- `apa_v24_unsaturated_stability_seed23_summary.csv`
- `apa_v24_unsaturated_stability_seed23_report.md`
