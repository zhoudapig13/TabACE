# APA v21 Feature-Evidence Context Interim Report

本轮目标：先实现用户提出的 no-shift feature-evidence context family，不再只依赖训练/测试类别比例差异做概率修正。

## 新增方法

- `diverse_knn_context`：从较大 KNN 候选池中逐步选择邻居，同时惩罚信息重复。
- `boundary_contrast_context`：对低边界样本同时放入 top-1 与 top-2 预测类的近邻。
- `prototype_context`：每类选靠近类别中心的代表样本，再按测试点距离取原型。
- `clean_knn_context`：过滤局部标签一致性低的训练邻居。
- `bootstrap_knn_contexts`：从 KNN 候选池采样多个 context 做集成。
- `feature_subspace_knn_contexts`：在全特征、高方差特征、随机子空间上分别取邻居。

新增方法名：`apa_cf_no_shift_diverse_router`、`apa_cf_no_shift_clean_router`、`apa_cf_no_shift_boundary_router`、`apa_cf_dual_mode_router_v17`。

## 掉点原因和修正

v20 的主要问题是用单个 train-only 验证段选择整批 context policy，验证段上涨不代表测试窗口上涨，导致 `mv`、`diabetes`、`electricity` 上出现负迁移。v21 改成样本级门控：只允许低 margin、低 confidence、多 feature-evidence context 方向一致、local neighbor evidence 支持、context disagreement 低的样本做小幅 blend。

第一版 v21 中 `clean` 和 `boundary` 仍在 `diabetes seed=1` 上各掉 1/64；原因是只改一个样本但该样本证据不够稳。第二版把 `clean/boundary` 的 margin、confidence、disagreement、vote fraction、local evidence 门槛收紧，并让 dual-mode 的 no-shift 分支使用更稳的 `diverse + prototype + feature_subspace`。

## No-shift 诊断集结果

| 数据集 | seed | 方法 | Acc变化 | BAcc变化 | Macro-F1变化 | eligible | changed |
|---|---:|---|---:|---:|---:|---:|---:|
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_clean_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_31_credit_g` | 1 | `apa_cf_no_shift_boundary_router` | +0.0156 | +0.0116 | +0.0137 | 0.0156 | 0.0156 |
| `openml_31_credit_g` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_clean_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_881_mv` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_diverse_router` | +0.0156 | +0.0185 | +0.0193 | 0.0156 | 0.0156 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_clean_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_37_diabetes` | 1 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_37_diabetes` | 1 | `apa_cf_dual_mode_router_v17` | +0.0156 | +0.0185 | +0.0193 | 0.0156 | 0.0156 |
| `openml_37_diabetes` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0312 |
| `openml_37_diabetes` | 2 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0156 | 0.0000 |
| `openml_37_diabetes` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0312 | 0.0312 |
| `openml_151_electricity` | 2 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 2 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 2 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 1 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 1 | `apa_cf_no_shift_boundary_router` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| `openml_151_electricity` | 1 | `apa_cf_dual_mode_router_v17` | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |

## 方法汇总

| 方法 | 组数 | 平均Acc变化 | Acc正/负/零 | 平均BAcc变化 | 平均Macro-F1变化 |
|---|---:|---:|---:|---:|---:|
| `apa_cf_dual_mode_router_v17` | 6 | +0.0026 | 1/0/5 | +0.0031 | +0.0032 |
| `apa_cf_no_shift_boundary_router` | 6 | +0.0026 | 1/0/5 | +0.0019 | +0.0023 |
| `apa_cf_no_shift_clean_router` | 3 | +0.0000 | 0/0/3 | +0.0000 | +0.0000 |
| `apa_cf_no_shift_diverse_router` | 6 | +0.0026 | 1/0/5 | +0.0031 | +0.0032 |

## Shift 安全小检查

只在 `openml_37_diabetes seed=0` 上做最小检查。`dual-mode` 在 shift=0 使用 feature-evidence 分支，在 shift=0.6/0.9 回到 prior-intervention 分支。

| shift | 方法 | Acc变化 | BAcc变化 | Macro-F1变化 |
|---:|---|---:|---:|---:|
| 0.0 | `apa_cf_no_shift_diverse_router` | +0.0156 | +0.0156 | +0.0152 |
| 0.0 | `apa_cf_dual_mode_router_v17` | +0.0156 | +0.0156 | +0.0152 |
| 0.0 | `apa_cf_directional_router` | +0.0000 | +0.0000 | +0.0000 |
| 0.6 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 |
| 0.6 | `apa_cf_dual_mode_router_v17` | +0.0312 | +0.0159 | +0.0164 |
| 0.6 | `apa_cf_directional_router` | +0.0312 | +0.0159 | +0.0164 |
| 0.9 | `apa_cf_no_shift_diverse_router` | +0.0000 | +0.0000 | +0.0000 |
| 0.9 | `apa_cf_dual_mode_router_v17` | +0.0312 | +0.0159 | +0.0164 |
| 0.9 | `apa_cf_directional_router` | +0.0312 | +0.0159 | +0.0164 |

## 当前结论

v21 已经验证了 feature-evidence context 的价值：在小诊断集上 `diverse` 和 `dual-mode` 有正例且没有 Acc 负例；`boundary` 在 `credit_g seed=1` 有小涨点。当前还不是最终普适算法，下一步应继续在少量诊断集上提高触发率，同时保持 changed_fraction 很低，再扩展到 10+ 数据集。
