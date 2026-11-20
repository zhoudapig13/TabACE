# Claim checklist

Keep these constraints visible while drafting and during every table update.

- **Name:** TabACE expands to **Tabular Adaptive Context Ensemble**. Do not
  later redefine the acronym. Older audited reports may display `CoCE`; that is
  a legacy paper-facing name for the same result fields, not a separate method.
- **Core invariant:** the foundation model and query are fixed; accessible
  training/evidence contexts vary and their induced predictions are combined.
- **No target-label access:** construction, selection, update factors, fusion, HPO, and
  shift-direction selection must not use target labels. Benchmark construction
  and final scoring are separate and must be described explicitly.
- **Known experimental track:** current experiments do not prove that an
  unsupervised router can identify an arbitrary unknown shift family.
- **Classification context count:** the defensible statement is that the first
  two or three source-ranked contexts give complementary AUC gains and then
  saturate. Do not claim that every context improves accuracy.
- **Regression:** the strong result depends mainly on a clean historical
  anchor, which is extra information. Adding all five contexts is slightly
  worse than the clean anchor alone in the latest v4 audit.
- **TabACE+DistPFN:** do not claim a significant improvement over TabACE unless the
  final paired confidence interval excludes zero. It is safe to describe the
  combined method as preserving TabACE while outperforming DistPFN only when the
  corresponding audited comparison supports that wording.
- **Natural OOD:** the single Hospital Readmission pilot mainly supports soft
  suppression of unsupported updates and does not replace a multi-task
  natural-shift benchmark.
- **Numbers:** regenerate headline tables from final audited CSVs. Do not mix
  older Experiment 10 numbers, candidate-review tables, and the v4 supplement.
- **Denominators:** print datasets, seeds, and failure counts wherever aggregate
  results from different tracks are compared.
