# Experiment Artifacts Index

This file records the experiment result groups currently stored under
`experiment2/results/`. Each group normally contains some or all of:

- `<prefix>_report.md`: readable report.
- `<prefix>_summary.csv`: dataset-level summary.
- `<prefix>_method_summary.csv`: method-level aggregate summary.
- `<prefix>_individual_deltas.csv`: per-run/per-dataset deltas against baseline.
- `<prefix>_results.csv`: raw run outputs.
- `<prefix>_environment.json`: runtime/environment metadata.

## Final Recommended Line

- `compact_unified_defense_10_5seed_v2`: final 10-dataset main result for
  `compact_unified_defense_policy`.
- `compact_unified_defense_expanded_5seed`: expanded 11-dataset safety check.
- `cc18_small_compact_unified_defense_3seed_v2`: CC18-small 16-dataset safety
  check.
- `paper_defense_policy_10_5seed`: stronger but more branch-heavy paper-defense
  reference.
- `paper_defense_policy_expanded_5seed`: expanded safety/reference run for the
  branch-heavy paper-defense policy.

## Accuracy/AUC Policy Development

- `accuracy_auc_strict_gate_smoke`
- `broad_accuracy_auc_policy_5seed`
- `broad_accuracy_auc_policy_tuned_5seed`
- `diagnostic_accuracy_auc_policy_5seed`
- `diagnostic_accuracy_auc_policy_smoke`
- `effect_size_accuracy_auc_policy_5seed`
- `effect_size_accuracy_auc_policy_5seed_v2`
- `effect_size_accuracy_auc_policy_expanded_5seed`
- `effect_size_accuracy_auc_policy_expanded_5seed_v2`
- `paper_policy_candidate_10_5seed`
- `unified_probability_blend_10_3seed`

## Context And Causal-Proxy Method Screens

- `causal_directions_multiseed`
- `causal_directions_smoke`
- `causal_gated_imputed_multiseed`
- `causal_gated_imputed_smoke`
- `causal_gated_multiseed`
- `causal_gated_smoke`
- `causal_tabicl_gated_multiseed`
- `causal_tabicl_gated_smoke`
- `combo_smoke`
- `context_effect_size_policy_expanded_5seed`
- `context_effect_size_policy_expanded_5seed_v2`
- `diagnostic_causal_policy_multiseed`
- `diagnostic_causal_policy_smoke`
- `extra_context_screen_3seed`
- `new_context_methods_expanded_3seed`
- `new_context_methods_expanded_5seed`
- `strong_weight_sample_attention_multiseed`
- `strong_weight_sample_attention_smoke`
- `strong_weight_threshold_multiseed`
- `strong_weight_threshold_smoke`

## Gate And Prior-Logit Experiments

- `gated_repeated_smoke`
- `gated_smoke`
- `model_gate_smoke`
- `prior_logit_accuracy_guard_multiseed`
- `prior_logit_accuracy_guard_smoke`
- `prior_logit_alpha_sweep_fixed_multiseed`
- `prior_logit_alpha_sweep_fixed_smoke`
- `prior_logit_alpha_sweep_smoke`
- `prior_logit_multiseed`
- `prior_logit_smoke`
- `prior_logit_validation_alpha_multiseed`
- `prior_logit_validation_alpha_smoke`

## Weak OpenML And Ensemble Screens

- `weak_openml_combo_multiseed`
- `weak_openml_ensemble_multiseed`
- `weak_openml_feature_ensemble_multiseed`
- `weak_openml_gated_multiseed`
- `weak_openml_gated_repeated_multiseed`
- `weak_openml_search`
- `weak_openml_top_multiseed`

## CC18-Small Screens

- `cc18_small_accuracy_auc_screen`
- `cc18_small_broad_policy_3seed`
- `cc18_small_broad_policy_smoke`
- `cc18_small_compact_unified_defense_3seed`
- `cc18_small_compact_unified_defense_3seed_v2`
- `cc18_small_context_effect_size_policy_3seed`
- `cc18_small_context_effect_size_policy_3seed_v2`
- `cc18_small_effect_size_policy_3seed`
- `cc18_small_effective_ensemble_screen`
- `cc18_small_ensemble_accuracy_auc_screen`
- `cc18_small_ensemble_top_3seed`
- `cc18_small_paper_defense_policy_3seed`
- `cc18_small_strong_combo_ensemble_3seed`

## Baseline Search Artifacts

- `diagnostic_threshold_multiseed`
- `diagnostic_threshold_smoke`
- `smoke_search`
