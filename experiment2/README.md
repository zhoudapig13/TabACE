# Experiment 2: Innovation Sandbox

This directory is prepared for a new algorithmic idea on top of the reproduced
TabICL baseline. It keeps the same data split and metric protocol as
`experiment1`, then evaluates:

- `baseline`: unchanged TabICL.
- `candidate`: TabICL after a pluggable dataset transformation from
  `experiment2/methods.py`.

Outputs are isolated under `experiment2/results/`.

## Quick smoke run

```bash
python experiment2/run_innovation.py \
  --datasets synthetic_classification \
  --methods identity,row_norm_feature \
  --n-estimators 1 \
  --device auto
```

## Where to add the new algorithm

Add a function in `experiment2/methods.py`, then register it in
`METHOD_REGISTRY`. The function receives a `DatasetSpec` and returns a new
`DatasetSpec` with transformed features.

Keep the baseline untouched. Only the candidate branch should change.

## Broad search workflow

Run a screening pass across weak OpenML datasets, local sklearn datasets, and
synthetic stress datasets:

```bash
python experiment2/search_methods.py \
  --openml weak \
  --random-states 42 \
  --max-samples 600 \
  --device cuda

python experiment2/summarize_search.py
```

Outputs:

- `experiment2/results/search_results.csv`: raw per dataset/seed/method metrics.
- `experiment2/results/search_summary.csv`: grouped method-dataset deltas.
- `experiment2/results/search_method_summary.csv`: benchmark-style method-level
  stability across datasets, especially useful for accuracy/AUC screening.
- `experiment2/results/search_individual_deltas.csv`: per-run metric deltas.
- `experiment2/results/search_report.md`: readable summary with metric changes
  like `baseline -> candidate (delta)` and positive-run counts.

## Benchmark workflow

For benchmark-style stability, use the official OpenML-CC18 suite. `cc18`
loads the 72 OpenML-CC18 data ids from OpenML study `99`, with a cached id list
as fallback. `cc18_small` runs the first 16 ids for a faster smoke sweep.

```bash
python experiment2/search_methods.py \
  --openml cc18_small \
  --sklearn "" \
  --methods baseline_n1,strong_weight_diagnostic_causal_combo_policy,strong_weight_combo_diagnostic_prior_logit_validation_accuracy_guarded_balanced_accuracy,quantile_domain_shift_feature,quantile_domain_probe_feature \
  --random-states 42 \
  --max-samples 500 \
  --device cuda \
  --output cc18_small_accuracy_auc_screen_results.csv \
  --environment-output cc18_small_accuracy_auc_screen_environment.json

python experiment2/summarize_search.py \
  --input cc18_small_accuracy_auc_screen_results.csv \
  --summary-output cc18_small_accuracy_auc_screen_summary.csv \
  --individual-output cc18_small_accuracy_auc_screen_individual_deltas.csv \
  --report-output cc18_small_accuracy_auc_screen_report.md
```

For this benchmark pass, prioritize classification `accuracy` and `roc_auc`.
Balanced accuracy and macro F1 are still recorded, but they are secondary unless
the dataset is explicitly imbalanced.

## Current experiment log

See `experiment2/RESULTS_SUMMARY.md` for the running Chinese/English summary of
which algorithms improved which metrics, on which dataset types, and whether the
gain was single-run or repeated across seeds.

## Algorithm name guide

When discussing a method, prefer saying the Chinese meaning first, then the code
name. This keeps experiment notes readable while preserving exact reproducible
method names.

| Chinese meaning | Code name | What it does |
| --- | --- | --- |
| 原始 TabICL 单模型基线 | `baseline_n1` | Use unchanged TabICL with `n_estimators=1`. |
| TabICL 四模型集成 | `ensemble_n4` | Use unchanged features, but increase TabICL ensemble size to 4. |
| 四模型集成加更尖锐温度 | `ensemble_n4_temp07` | Same as `ensemble_n4`, with `softmax_temperature=0.7`. |
| 四模型集成并平均概率 | `ensemble_n4_avg_probs` | Same as `ensemble_n4`, but average probabilities instead of logits. |
| 按处理/选择组平衡上下文 | `balanced_treatment` | Balance synthetic treatment or selection groups in the context. |
| 按类别平衡上下文 | `label_balance_context` | Downsample context rows so each class has similar count. |
| 稀有类别过采样 | `rare_class_oversample` | Duplicate low-frequency class rows in the training context. |
| 稀有类别加边界样本上下文 | `rare_boundary_context` | Keep rare-class and class-boundary rows. |
| 稀有类别过采样加域偏移特征 | `rare_domain_shift_oversample` | Oversample rare classes and append train/test domain-shift features. |
| 稀有边界样本加域偏移特征 | `rare_boundary_domain_shift` | Keep rare/boundary rows and append domain-shift features. |
| 倾向分数特征 | `propensity_feature` | Append synthetic propensity score and treatment flag. |
| 倾向分数 logit 特征 | `propensity_logit_feature` | Append logit propensity score and treatment flag. |
| 倾向分数排序特征 | `propensity_rank_feature` | Append propensity percentile rank and treatment flag. |
| 倾向分数与重要特征交互 | `propensity_interactions` | Add interactions between propensity and top causal-proxy features. |
| 倾向分数加因果代理特征筛选 | `propensity_causal_kbest` | Keep top supervised causal-proxy features, then append propensity features. |
| 倾向分数重叠区间上下文裁剪 | `propensity_overlap` | Trim context rows to the train/test propensity-overlap region. |
| 倾向分数重叠区间裁剪 | `propensity_overlap_feature` | Keep context rows in train/test propensity overlap and append propensity features. |
| 倾向分数分箱平衡 | `propensity_bin_balance` | Balance training rows across propensity-score bins. |
| 因果代理权重缩放 | `causal_scaled` | Scale features by supervised causal-proxy weights. |
| 因果代理 Top-K 特征筛选 | `causal_kbest` | Keep top supervised causal-proxy features. |
| 训练期特征因果偏置，推理期关闭 | `train_only_causal_feature_bias` | Duplicate causally biased context rows while leaving inference/test inputs unchanged. |
| 因果辅助任务表征代理 | `causal_auxiliary_summary_feature` | Append compact causal-role summary features as a proxy for auxiliary causal heads. |
| 因果 dropout/mask 数据增强 | `causal_mask_dropout_augmentation` | Augment training context by masking weak causal-proxy features more often. |
| 样本间因果 attention 代理 | `causal_sample_attention_context` | Keep context rows closest to test rows under causal-weighted sample distance. |
| 不平衡诊断因果 mask 策略 | `diagnostic_causal_mask_policy` | Enable causal mask/dropout only for imbalanced medium-feature binary diagnostics. |
| 权重集中诊断样本因果 attention 策略 | `diagnostic_sample_attention_policy` | Enable sample-level causal attention only when causal-proxy weights are concentrated. |
| 诊断组合因果策略 | `diagnostic_causal_combo_policy` | Use train/test diagnostics to choose causal mask, sample causal attention, or identity. |
| 强因果权重诊断样本 attention 策略 | `strong_weight_sample_attention_policy` | Enable sample-level causal attention when causal-proxy weights are very concentrated in balanced medium-feature binary data. |
| 强权重诊断组合因果策略 | `strong_weight_diagnostic_causal_combo_policy` | Use causal mask for imbalanced diagnostics, strong-weight sample attention for credit-like diagnostics, otherwise identity. |
| 分位数正态化特征 | `quantile_features` | Apply quantile-normal transform before TabICL. |
| 稳健缩放特征 | `robust_features` | Apply robust scaling before TabICL. |
| 极值截断特征 | `winsorized_features` | Clip feature tails by train quantile bounds. |
| 异常值指示特征 | `outlier_indicator_feature` | Append row-level robust outlier indicators. |
| 极值截断加异常值指示 | `outlier_winsor_combo` | Clip feature tails and append outlier indicators. |
| 符号对数压缩特征 | `signed_log_features` | Compress large deviations around feature medians. |
| 域偏移特征 | `domain_shift_feature` | Append features predicting whether a row looks train-like or test-like. |
| 分位数正态化加域偏移特征 | `quantile_domain_shift_feature` | Quantile-normalize features, then append train/test domain-shift features. |
| 分位数正态化加线性探针特征 | `quantile_linear_probe_feature` | Quantile-normalize, then append linear-probe confidence/probability features. |
| 分位数正态化加域偏移和线性探针 | `quantile_domain_probe_feature` | Quantile-normalize, append domain-shift features, then append linear-probe features. |
| 因果代理筛选加分位数正态化和域偏移 | `causal_quantile_domain_feature` | Keep top causal-proxy features, quantile-normalize, then append domain-shift features. |
| 倾向分数加分位数正态化和域偏移 | `propensity_quantile_domain_feature` | Quantile-normalize, append domain-shift features, then append propensity features. |
| 分位数域偏移探针加四模型集成 | `quantile_domain_probe_ensemble_n4` | `quantile_domain_probe_feature` plus `n_estimators=4`. |
| 域偏移特征加四模型集成 | `domain_shift_ensemble_n4` | `domain_shift_feature` plus `n_estimators=4`. |
| 分位数域偏移特征加四模型集成 | `quantile_domain_shift_ensemble_n4` | `quantile_domain_shift_feature` plus `n_estimators=4`. |
| 因果分位数域偏移加四模型集成 | `causal_quantile_domain_ensemble_n4` | `causal_quantile_domain_feature` plus `n_estimators=4`. |
| 倾向分数分位数域偏移加四模型集成 | `propensity_quantile_domain_ensemble_n4` | `propensity_quantile_domain_feature` plus `n_estimators=4`. |
| 极值异常特征加四模型集成 | `outlier_winsor_ensemble_n4` | `outlier_winsor_combo` plus `n_estimators=4`. |
| 线性探针特征加四模型集成 | `linear_probe_ensemble_n4` | `linear_probe_feature` plus `n_estimators=4`. |
| 强权重诊断组合因果策略加四模型集成 | `strong_weight_diagnostic_causal_combo_ensemble_n4` | `strong_weight_diagnostic_causal_combo_policy` plus `n_estimators=4`. |
| 准确率/AUC保护的内部验证集成门控 | `accuracy_auc_guarded_ensemble_gate` | Use training-only validation to choose baseline, plain four-model ensemble, or strong-weight causal combo ensemble, with an accuracy guard. |
| 准确率/AUC保护的宽候选内部验证集成门控 | `accuracy_auc_guarded_wide_ensemble_gate` | Same accuracy/AUC gate, but also considers previous positive feature+ensemble variants such as outlier-winsor, domain-shift, quantile-domain-probe, and linear-probe ensembles. |
| 线性探针辅助特征 | `linear_probe_feature` | Append balanced linear-probe confidence, uncertainty, and class-probability features. |
| 困难样本过采样 | `hard_example_oversample` | Oversample low-confidence or linear-probe-misclassified rows. |
| 线性探针特征加困难样本过采样 | `hard_probe_combo` | Append probe features and oversample hard rows. |
| 局部密度特征 | `local_density_feature` | Append local neighborhood density features. |
| 类中心距离特征 | `centroid_distance_feature` | Append distances to class centroids. |
| KNN 标签分布特征 | `knn_label_feature` | Append leave-one-out KNN label distribution features. |
| 域匹配上下文选择 | `domain_matched_context` | Keep context rows most similar to the test feature distribution. |
| 域偏移与因果代理交互 | `domain_causal_interactions` | Add interactions between domain-shift score and top causal-proxy features. |
| 单次内部验证门控选择 | `validation_gated_feature` | Use one training-only validation split to choose identity or a candidate transform. |
| 多次内部验证门控选择 | `validation_gated_repeated_feature` | Average three internal validation splits before choosing a transform. |
| 因果方向单次内部验证门控 | `causal_validation_gated_feature` | Use one training-only validation split to choose identity or one causal proxy transform. |
| 因果方向多次内部验证门控 | `causal_validation_gated_repeated_feature` | Average three internal validation splits before choosing a causal proxy transform. |
| 因果方向 TabICL 单次内部验证门控 | `causal_tabicl_gated_feature` | Use one training-only split and real TabICL validation to choose a causal proxy transform. |
| 因果方向 TabICL 多次内部验证门控 | `causal_tabicl_gated_repeated_feature` | Average three real TabICL validation splits before choosing a causal proxy transform. |
| 二分类 balanced accuracy 阈值校准 | `threshold_balanced_accuracy` | Tune the positive-class threshold on an internal validation split for balanced accuracy. |
| 二分类 macro F1 阈值校准 | `threshold_macro_f1` | Tune the positive-class threshold on an internal validation split for macro F1. |
| 域偏移特征加 balanced accuracy 阈值校准 | `domain_shift_threshold_balanced_accuracy` | Add domain-shift features, then tune a binary threshold for balanced accuracy. |
| 极值异常特征加 balanced accuracy 阈值校准 | `outlier_winsor_threshold_balanced_accuracy` | Add outlier-winsor features, then tune a binary threshold for balanced accuracy. |
| 因果分位数域偏移加 balanced accuracy 阈值校准 | `causal_quantile_domain_threshold_balanced_accuracy` | Use causal-quantile-domain features, then tune a binary threshold. |
| 倾向分数分位数域偏移加 balanced accuracy 阈值校准 | `propensity_quantile_domain_threshold_balanced_accuracy` | Use propensity-quantile-domain features, then tune a binary threshold. |
| 类别先验对数校正 | `class_prior_logit_adjustment` | Subtract training class log-priors from predicted logits before choosing a class. |
| 四分之一强度类别先验对数校正 | `class_prior_logit_adjustment_quarter` | Apply a mild inverse-prior logit correction at quarter strength. |
| 平方根类别先验对数校正 | `class_prior_logit_adjustment_sqrt` | Apply a milder inverse-prior logit correction using square-root strength. |
| 四分之三强度类别先验对数校正 | `class_prior_logit_adjustment_three_quarter` | Apply inverse-prior logit correction at three-quarter strength. |
| 强权重诊断组合因果策略加类别先验对数校正 | `strong_weight_combo_prior_logit_adjustment` | Use the strong-weight causal diagnostic combo, then apply inverse-prior logit correction. |
| 强权重诊断组合因果策略加诊断式四分之一强度类别先验对数校正 | `strong_weight_combo_diagnostic_prior_logit_adjustment_quarter` | Apply quarter-strength prior-logit correction only when causal/imbalance diagnostics enable it. |
| 强权重诊断组合因果策略加诊断式平方根类别先验对数校正 | `strong_weight_combo_diagnostic_prior_logit_adjustment_sqrt` | Apply mild prior-logit correction only when causal/imbalance diagnostics enable it. |
| 强权重诊断组合因果策略加诊断式四分之三强度类别先验对数校正 | `strong_weight_combo_diagnostic_prior_logit_adjustment_three_quarter` | Apply three-quarter-strength prior-logit correction only when causal/imbalance diagnostics enable it. |
| 内部验证选择类别先验强度优化 balanced accuracy | `class_prior_logit_validation_balanced_accuracy` | Choose prior-logit correction strength on an internal validation split for balanced accuracy. |
| 内部验证选择类别先验强度优化 macro F1 | `class_prior_logit_validation_macro_f1` | Choose prior-logit correction strength on an internal validation split for macro F1. |
| 强权重诊断组合因果策略加诊断式内部验证类别先验强度选择 | `strong_weight_combo_diagnostic_prior_logit_validation_balanced_accuracy` | Use diagnostics, then choose prior-logit correction strength by internal balanced accuracy. |
| 准确率保护的内部验证类别先验强度选择 | `class_prior_logit_validation_accuracy_guarded_balanced_accuracy` | Choose prior-logit strength by internal balanced accuracy, but only among alphas with validation accuracy within 0.01 of default. |
| 严格准确率保护的内部验证类别先验强度选择 | `class_prior_logit_validation_strict_accuracy_guarded_balanced_accuracy` | Choose prior-logit strength by internal balanced accuracy while requiring validation accuracy at least as high as default. |
| 强权重诊断组合因果策略加准确率保护的内部验证类别先验强度选择 | `strong_weight_combo_diagnostic_prior_logit_validation_accuracy_guarded_balanced_accuracy` | Use diagnostics, then choose prior-logit strength with the 0.01 validation accuracy guard. |
| 强权重诊断组合因果策略加 balanced accuracy 阈值校准 | `strong_weight_combo_threshold_balanced_accuracy` | Use the strong-weight causal diagnostic combo, then tune a binary threshold for balanced accuracy. |
| 强权重诊断组合因果策略加 macro F1 阈值校准 | `strong_weight_combo_threshold_macro_f1` | Use the strong-weight causal diagnostic combo, then tune a binary threshold for macro F1. |
| 强权重诊断组合因果策略加诊断式 balanced accuracy 阈值校准 | `strong_weight_combo_diagnostic_threshold_balanced_accuracy` | Use threshold tuning only when diagnostics indicate non-identity causal action or clear imbalance. |
| 强权重诊断组合因果策略加诊断式 macro F1 阈值校准 | `strong_weight_combo_diagnostic_threshold_macro_f1` | Use macro-F1 threshold tuning only when diagnostics indicate non-identity causal action or clear imbalance. |
