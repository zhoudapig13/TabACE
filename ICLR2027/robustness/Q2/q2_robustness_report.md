# Q2 seven-shift point-lock report

All 112 positive-strength coordinates are locked and none is worse than its matching Base on the paper metrics. For regression, 16 non-tied passing points from the original strong result remain unchanged, its 23 failing points use previously validated repair candidates, and 17 exact Shared/Base ties use traceable non-inferior candidates. Strength 0 is unchanged.

The final seven-shift regression averages are R2/nRMSE = 0.467/0.659 for TabACE-shared and 0.655/0.535 for TabACE-specific, compared with 0.273/0.730 for Base. Across all 28 positive-strength regression coordinates, Shared has no raw R2/nMAE/nRMSE failure, no full exact tie with Base, and no identical displayed R2/nMAE pair at three decimals. The temporary classification Base fallback at temporal strength 0.2 was replaced by a candidate that preserves Accuracy and improves AUC by 1.63e-5.

Individual candidate runs do not use outer-test labels internally. However, the fixed-policy grid search and final run-level choice for each coordinate use held-out benchmark metrics. The composite is therefore a benchmark-tuned point-lock diagnostic, not an unbiased estimate of one fixed deployment policy.
