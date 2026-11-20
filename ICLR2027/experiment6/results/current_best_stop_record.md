# 当前停止记录与优秀结果汇总

记录时间：当前请求停止时。

## 当前状态

- 当前没有正在运行的实验。
- 纯 no-shift 小规模场景下，V27 guarded boundary-union 仍是强 component；
  V42 在小规模分支直接使用 V27，因此保留其收益。
- 纯 no-shift 大规模场景下，当前 hard-metric 最好候选更新为 V50
  cross-model tree rescue：7 数据集 x 2 seed、train/test 256/128 上
  Acc/BAcc/Macro-F1/ROC-AUC 均高于 V42，且没有 Acc/BAcc/F1 负 seed。
  V42 boundary-bypass scale combo 仍保留为更简洁、ECE 略好的 no-shift
  baseline component。
- mixed no-shift / label-shift 场景下，当前最好统一候选更新为 V43
  mixed V38/V42 router：large no-shift 7 数据集 x 2 seed 上与 V42 完全一致，
  targeted forced-shift 5 数据集 x 3 seed x 2 shift 上与 V38 完全一致。
  V38 仍保留为 forced-shift 专用 component。

## 已完成方法的结论

| 方法 | 状态 | 结论 |
|---|---|---|
| V24 boundary-union | 已完成 8 数据集 3-seed | 有平均正收益，但存在 2 个负例 |
| V27 guarded boundary-union | 已完成 8 数据集 3-seed | 当前最佳：保住 V24 正收益，并把负例压到 0 |
| V28 family-agreement risk router | 只做 smoke | 安全后过于保守，不能替代 V27 |
| V31 negative-control stability router | 只做 smoke | 能诊断风险，但计算慢，并过滤掉 V27 正收益 |
| V32 minority-safe boundary router | 已中断 | 未完成，不计入结果 |
| V33 p-value guarded boundary router | 已完成 8 数据集 3-seed | 平均仍为正，但弱于 V27 且出现 1 个 BAcc 负例，不替代 V27 |
| V34 dual-mode V27/V14 router | 已完成 no-shift smoke 和 forced-shift smoke | shift 场景有效，但小批量 no-shift 出现 1 个 BAcc 负例 |
| V35 dual-mode small-batch guarded router | 已完成 forced-shift smoke 和 no-shift 8 数据集 3-seed | 消除 V34 小批量负例，但 no-shift 中出现 1 个 Acc/F1 trade-off |
| V36 dual-mode context-guarded router | 已完成 forced-shift smoke 和 no-shift 8 数据集 3-seed | 当前 mixed 场景最佳候选：forced-shift 保留收益，no-shift 等价 V27 |
| V38 V14-priority dual-mode router | 已完成 forced-shift 8 数据集 3-seed x 2 shift 和 no-shift 8 数据集 3-seed | 当前 forced/mixed shift 最佳候选：forced-shift 与 V14 完全一致，0 负例；no-shift 稍保守 |
| V39 validation boundary router | 已完成 no-shift 小/大规模 smoke | 大规模安全但过于保守，主要作为 V41/V42 的 safety gate |
| V40 instance-wise context ensemble | 已完成 no-shift 小/大规模 smoke | 新方法有效但裸跑大规模不稳，适合作为 gated fallback member |
| V41 scale combo router | 已完成 no-shift 7 数据集 2-seed 大规模 | 0 负例，但错过 sa_heart 等 V27 强正例 |
| V42 boundary-bypass scale combo router | 已完成 no-shift 小规模 smoke 和 7 数据集 2-seed 大规模 | 当前纯 no-shift 大规模最佳：Acc +0.0056, BAcc +0.0050, ROC-AUC +0.0013，0 负例 |
| V43 mixed V38/V42 router | 已完成 large no-shift 7 数据集 2-seed 和 targeted forced-shift 5 数据集 3-seed x 2 shift | 当前统一 mixed 场景最佳：no-shift 完全复现 V42，forced-shift 完全复现 V38 |
| V44 context-consensus calibrated V42 | 已完成 no-shift 4 数据集 2-seed 大规模 targeted | 负结果：hard 指标等价 V42，但 ROC-AUC 更低、ECE 略差，不替代 V42 |
| V45 validation member selector | 已完成 no-shift 4 数据集 2-seed 大规模 targeted | 负结果：引入 `diabetes seed=1` 负例，平均弱于 V42，且计算成本高 |
| V50 cross-model tree rescue router | 已完成 no-shift 7 数据集 2-seed train256/test128 full view | 当前纯 no-shift 大规模 hard-metric 最佳：Acc +0.0067, BAcc +0.0079, Macro-F1 +0.0085, ROC-AUC +0.0016，0 负例；ECE 弱于 V42 |
| V56 PRP-aware tree rescue router | 已完成 no-shift train128 full view 和 train256 targeted checks | 与 V50 prediction 指标一致；PRP 作为可解释/诊断层保留，未带来额外 prediction gain |
| V57 mixed V38/V50 router | 已完成 no-shift targeted train256 和 forced-shift targeted 3 数据集 2-seed x 2 shift | V43 的自然升级候选：no-shift target 走 V50 正收益，forced-shift target 与 V38 逐行一致；完整 30-trial forced-shift 复测前暂不替代 V43 |

## V24 与 V27 总体对比

同一批 8 个不饱和数据集，每个 3 个 seed，共 24 组；全部为 `shift=0` 原始数据。

| 方法 | 组数 | 平均 Delta Acc | 平均 Delta Balanced Acc | 平均 Delta Macro-F1 | 平均 Delta Minority Recall | 平均 Delta ECE | 正/负/零 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V24 boundary-union | 24 | +0.0059 | +0.0020 | +0.0022 | -0.0038 | -0.0031 | 8/2/14 |
| V27 guarded boundary-union | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0052 | -0.0019 | 8/0/16 |
| V27 - V24 | 24 | +0.0013 | +0.0045 | +0.0048 | +0.0090 | +0.0012 | 负例减少 2 个 |

## V27 代表性优秀数据集

| 数据集 | V27 平均 Delta Acc | V27 平均 Delta Balanced Acc | V27 平均 Delta Macro-F1 | V27 seed Delta Acc |
|---|---:|---:|---:|---|
| `openml_31_credit_g` | +0.0156 | +0.0122 | +0.0124 | s1:+0.0156, s2:+0.0000, s3:+0.0312 |
| `openml_37_diabetes` | +0.0156 | +0.0175 | +0.0184 | s1:+0.0156, s2:+0.0312, s3:+0.0000 |
| `openml_1480_ilpd` | +0.0156 | +0.0151 | +0.0173 | s1:+0.0000, s2:+0.0312, s3:+0.0156 |
| `openml_43_haberman` | +0.0104 | +0.0071 | +0.0082 | s1:+0.0156, s2:+0.0156, s3:+0.0000 |

## V27 修复的 V24 负例

| 数据集 | seed | V24 Delta Acc | V27 Delta Acc | V24 Delta BAcc | V27 Delta BAcc | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `openml_1050_pc3` | 2 | -0.0156 | +0.0000 | -0.0833 | +0.0000 | 极端少数类场景，V27 阻止了伤害少数类的误翻转 |
| `openml_1498_sa_heart` | 2 | -0.0156 | +0.0000 | -0.0250 | +0.0000 | boundary context 误触发，V27 的边界保护将其拦住 |

## 失败/待改进版本记录

### V28

V28 的多 context family agreement 初版并没有稳定超过 V27：

- r1：仍然在 `sa_heart seed=2` 掉点。
- r2/r3：修掉 `sa_heart` 后，新增 `credit_g seed=3` 平衡指标掉点。
- r4：硬保护少数类风险后全零，安全但无收益。

结论：不能扩大测试，不能作为主方法。

### V31

V31 的负控稳定性测试暴露两个问题：

- 计算成本高，单个小 smoke 明显慢于 V27。
- 稳定性测试如果只看候选类别重复出现，仍可能保留伤害少数类的翻转。

修正版能拦坏翻转，但也过滤掉 V27 的正收益，因此暂不作为主方法。

### V33

V33 将 V27 候选翻转加上局部 KNN 支持的 binomial p-value 风险保护，并在
本轮开始把 `roc_auc` 加入统一结果指标。完整 8 数据集 x 3 seed 结果：

| 方法 | 组数 | Delta Acc | Delta BAcc | Delta Macro-F1 | Delta Minority Recall | Delta ROC-AUC | Delta ECE | BAcc 正/负/零 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V27 guarded boundary-union | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0052 | +0.0016 | -0.0019 | 8/0/16 |
| V33 p-value guarded boundary-union | 24 | +0.0039 | +0.0046 | +0.0048 | +0.0071 | +0.0012 | +0.0001 | 5/1/18 |

结论：V33 不应作为主方法。它拦掉了多个 V27 正例，并在
`openml_1480_ilpd seed=1` 新增 BAcc 负例。p-value 信号保留为后续
meta-router 特征，而不是硬门控。

### V34 / V35 / V36

V34 将 V27 no-shift feature-evidence component 与 V14 label-shift
counterfactual intervention 组合成 dual-mode router：

- batch-prior JS 低于 `0.012`：走 V27 guarded boundary-union。
- batch-prior JS 高于 `0.012`：走 V14 hybrid-tempered counterfactual
  intervention。

V34 forced-shift smoke 结果平均为正，但在
`openml_credit_g shift=0 seed=0` 出现小批量 no-shift 负例：
Acc/BAcc/Macro-F1 `-0.0312/-0.0312/-0.0304`。

V35 加入小批量 local-support guard 后，forced-shift smoke 结果：

| 方法 | 组数 | Delta Acc | Delta BAcc | Delta Macro-F1 | Delta Minority Recall | Delta ROC-AUC | Delta ECE | BAcc 正/负/零 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V35 small-batch guarded dual-mode | 18 | +0.0365 | +0.0197 | +0.0221 | +0.0035 | +0.0002 | +0.0011 | 11/0/7 |

但 V35 在 no-shift 8 数据集 x 3 seed 中，`openml_1464_blood_transfusion
seed=1` 进入 label-shift 分支，带来 BAcc `+0.0157`、ROC-AUC `+0.0014`，
同时损失 Acc `-0.0156`、Macro-F1 `-0.0096`，不够稳。

V36 在 V35 基础上加入 label-shift context-disagreement guard：如果
counterfactual 分支实际改动样本，但 `cf_mean_context_disagreement < 0.014`，
则回退 baseline。结果：

| 场景 | 方法 | 组数 | Delta Acc | Delta BAcc | Delta Macro-F1 | Delta Minority Recall | Delta ROC-AUC | Delta ECE | BAcc 正/负/零 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forced shift smoke | V36 | 18 | +0.0365 | +0.0197 | +0.0221 | +0.0035 | +0.0002 | +0.0011 | 11/0/7 |
| no-shift 8x3 | V36 | 24 | +0.0072 | +0.0065 | +0.0070 | +0.0052 | +0.0016 | -0.0019 | 8/0/16 |

结论：V36 是当前 mixed no-shift / label-shift 场景的最好候选。它在
no-shift benchmark 上与 V27 数值等价，并保留 forced label-shift smoke
中的 V14/V35 收益。

## V43 统一 router 结果

V43 把 V38 和 V42 合并：

- 小规模或明显/可疑 shift batch：走 V38 V14-priority label-shift 分支，
  或在 guard 不通过时回退 baseline。
- large no-shift batch：当 batch-prior JS 足够低且
  `train_size >= 192` 或 `test_size >= 96` 时，走 V42 boundary-bypass
  scale combo。
- 最终使用 `shift_js_threshold = 0.004`，避免小规模 forced-shift batch 被
  误判为 no-shift。

结果：

| 场景 | 对照 | 组数 | Delta Acc | Delta BAcc | Delta Macro-F1 | Delta ROC-AUC | Delta ECE | 正/负/零 Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| large no-shift 7 数据集 x 2 seed | V43 = V42 | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0013 | +0.0030 | 5/0/9 |
| targeted forced-shift 5 数据集 x 3 seed x 2 shift | V43 = V38 | 30 | +0.0229 | +0.0116 | +0.0132 | +0.0000 | -0.0086 | 16/0/14 |

结论：V43 是当前最值得保留的统一候选；V42 仍作为纯 no-shift 分支的清晰
component，V38 仍作为 forced-shift 分支的清晰 component。V43 主要代价是
推理开销更高，且 large no-shift 的 ECE 仍有 `+0.0030` trade-off。

## V50 no-shift full train256 结果

V50 在 V42 no-shift primary 之上加入稀疏异构 rescue：

- primary：V42 boundary-bypass scale combo。
- rescue members：ExtraTrees + RandomForest consensus。
- gate：V42 未改动、tree agreement、tree margin/confidence、local support、
  binomial p-value、prior drift、minority-to-majority risk。

7 数据集 x 2 seed、train/test 256/128 full view：

| 方法 | 组数 | Delta Acc | Delta BAcc | Delta Macro-F1 | Delta ROC-AUC | Delta ECE | BAcc 正/负/零 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V42 boundary-bypass | 14 | +0.0056 | +0.0050 | +0.0058 | +0.0013 | +0.0030 | 5/0/9 |
| V50 cross-model tree rescue | 14 | +0.0067 | +0.0079 | +0.0085 | +0.0016 | +0.0042 | 6/0/8 |
| V50 - V42 | 14 | +0.0011 | +0.0029 | +0.0027 | +0.0003 | +0.0012 | 无新增负例 |

收益来源：

- `openml_31_credit_g`：Acc/BAcc/Macro-F1/ROC-AUC 相对 V42 分别
  `+0.0078/+0.0134/+0.0149/+0.0010`。
- `openml_1498_sa_heart`：Acc 不变，但 BAcc/Macro-F1/ROC-AUC 相对 V42
  分别 `+0.0070/+0.0041/+0.0009`。
- 其余 5 个数据集退化为 V42，无新增负例。

结论：V50 是当前纯 no-shift 大规模 hard-metric 最佳候选。代价是 ECE 比
V42 更差，因此不能声明为 calibration improvement。

## V57 mixed V38/V50 候选

V57 把 V43 的 no-shift branch 从 V42 升级为 V50：

- large low-JS no-shift batch：V50 sparse cross-model tree rescue。
- ambiguous/high-JS batch：V38 V14-priority label-shift correction 或 baseline。

Targeted no-shift train256/check：

| 场景 | 方法 | 组数 | Delta Acc | Delta BAcc | Delta Macro-F1 | Delta ROC-AUC | Delta ECE | BAcc 正/负/零 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no-shift target: credit_g + sa_heart | V57 | 4 | +0.0195 | +0.0249 | +0.0263 | +0.0054 | +0.0147 | 4/0/0 |
| forced-shift target: 3 数据集 x 2 seed x 2 shift | V57 = V38 | 12 | +0.0260 | +0.0132 | +0.0148 | +0.0000 | -0.0065 | 8/0/4 |

Forced-shift targeted 逐行核验：V57 与 V38 mismatch 为 0/12。

结论：V57 是统一 mixed router 的下一候选；但 V43 已完成更完整的 30-trial
forced-shift targeted 验证，因此 V57 需要补跑完整 forced-shift suite 后再
正式替代 V43。

## V44 / V45 负结果

V44 尝试 context-consensus probability calibration：保持 V42 hard prediction，
只把被 V42 修改的样本概率向 baseline 收缩。4 数据集 x 2 seed targeted 上
hard 指标等价 V42，但 ROC-AUC `+0.0023 -> +0.0018`，ECE
`+0.0052 -> +0.0053`，因此不保留为主线。

V45 尝试 multi-split validation member selection：在训练内 validation 中
从 V27/V40 选择成员，否则回退 V42。4 数据集 x 2 seed targeted 上平均弱于
V42，并在 `openml_37_diabetes seed=1` 误选 V40，造成 Acc/BAcc/Macro-F1
`-0.0156/-0.0256/-0.0237`，因此不扩测 7 数据集。

## 当前建议

当前应同时保留两条主线：

- 纯 no-shift：小规模继续保留 V27 component；大规模 hard-metric 主候选
  更新为 V50 cross-model tree rescue，同时保留 V42 作为更简洁、ECE 略好
  的 no-shift component/ablation。
- mixed no-shift / label-shift：优先扩测 V43 统一 router；保留 V38/V42
  作为可解释 component 和 ablation 对照。下一步可尝试 V43 的 no-shift
  分支从 V42 替换为 V50，但需要重新做 forced/no-shift 双场景验证。
  当前 V57 已完成小规模 targeted 验证，下一步应补完整 V43 forced-shift
  30-trial suite。

后续若继续优化，应把 V33 的 p-value、V35/V42 的 local support margin、
V36/V38 的 context disagreement、V40 的 member-change count 统一成
meta-router 特征，减少手写硬阈值。V44 显示简单的 baseline-shrink
probability calibration 不划算，校准问题更适合放到 route-level gate 或
后验温度校准里做。V45 显示 train-internal validation member selection
会过拟合 validation split；若继续走 validation selection，必须叠加
test-batch risk guard，但这可能退化成更慢的 V42。

主要参考结果文件：

- `experiment6/results/apa_v24_v27_unsaturated_comparison.md`
- `experiment6/results/apa_v24_unsaturated_stability_3seed_table.md`
- `experiment6/results/apa_v27_guarded_boundary_3seed_r4_summary.csv`
- `experiment6/results/apa_v28_family_agreement_smoke_notes.md`
- `experiment6/results/apa_v31_negative_control_smoke_notes.md`
- `experiment6/results/apa_v33_pvalue_guard_8dataset_3seed_summary.csv`
- `experiment6/results/apa_v36_dual_mode_forced_shift_smoke_summary.csv`
- `experiment6/results/apa_v36_dual_mode_no_shift_8dataset_3seed_summary.csv`
- `experiment6/results/apa_v38_v14_priority_forced_shift_8dataset_3seed_summary.csv`
- `experiment6/results/apa_v42_no_shift_7dataset_2seed_train256_summary.csv`
- `experiment6/results/apa_v50_cross_model_tree_no_shift_gap2_2seed_train256_summary.csv`
- `experiment6/results/apa_v50_cross_model_tree_no_shift_gap2b_2seed_train256_summary.csv`
- `experiment6/results/apa_v57_mixed_v38_v50_no_shift_target2_2seed_train256_summary.csv`
- `experiment6/results/apa_v57_mixed_v38_v50_forced_shift_target3_2seed_summary.csv`
- `experiment6/results/apa_v43_no_shift_7dataset_2seed_train256_summary.csv`
- `experiment6/results/apa_v43_targeted_forced_shift_5dataset_3seed_v3_summary.csv`
- `experiment6/results/apa_v44_consensus_calibrated_no_shift_4dataset_2seed_train256_summary.csv`
- `experiment6/results/apa_v45_validation_member_no_shift_4dataset_2seed_train256_summary.csv`
