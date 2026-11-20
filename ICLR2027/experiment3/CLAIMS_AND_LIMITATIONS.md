# Claims And Limitations

## Supported Claims

- Causal-aware treated-control context retrieval improves PEHE over random/kNN
  context retrieval in context-limited ICL-style CATE estimation.
- `matched_pair_diverse` is the strongest method on the extended synthetic
  benchmark.
- External validation changes the ranking: on scaled IHDP and ACIC 2016
  condition 1, the simpler `matched_pair` is more robust than
  `matched_pair_diverse`.
- Common-support filtering alone is not enough: `overlap_aware` is a useful
  negative control because it often hurts PEHE.

## Weak Or Secondary Claims

- ATE error and policy value can improve, but they are less stable than PEHE and
  should remain secondary.
- Real TabICL is callable in a tiny smoke test, but the current evidence does
  not establish final TabICL model-level improvement.

## Unsupported Claims

- This is not a new causal PFN.
- This is not proof that TabICL or TabPFN improves on CATE tasks after model
  adaptation.
- This does not show `matched_pair_diverse` is uniformly best.
- This does not show context-limited methods beat full-data classical causal
  estimators overall.
- This should not be mixed with experiment2 accuracy/AUC policy artifacts.

## Next Work

- Run a paper-scale real TabICL evaluation once compute budget is available.
- Supply `TABPFN_TOKEN` and rerun the TabPFN model probe, then decide whether a
  full TabPFN CATE wrapper is worthwhile.
- Expand ACIC beyond condition 1 and increase split seeds/evaluated test units.
- Study why diversity helps synthetic regimes but hurts or weakens external
  IHDP/ACIC settings.
