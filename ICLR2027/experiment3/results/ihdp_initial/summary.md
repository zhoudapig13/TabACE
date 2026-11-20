# Paper-Style Summary

## Strongest Claim

Diverse matched treated-control context selection consistently improves PEHE for context-limited tabular ICL-style CATE estimation over random and kNN retrieval. The original matched pair remains the simpler baseline; the diverse variant is the promoted method.

## Weak Claim

ATE and policy-value gains are more modest and scenario-dependent. They should be reported as secondary diagnostics rather than the main claim.

## Negative Result

The first `overlap_aware` retriever often hurts PEHE because filtering toward common support alone can discard local similarity. The redesigned `overlap_filtered_matched_pair` should be compared against it, but plain overlap filtering is not promoted.

## What Should Not Be Claimed

- Do not claim a new causal PFN.
- Do not claim superiority over full-data classical causal learners.
- Do not claim ATE/policy improvements are as stable as PEHE improvements unless paired intervals support that.
