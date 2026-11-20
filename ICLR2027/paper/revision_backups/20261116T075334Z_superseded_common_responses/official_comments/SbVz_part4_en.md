## Part 4: Comparison with the backbone's native ensemble

> The comparisons should report the performance when ensembling also the baselines, i.e., the main backbone (TabICL-v2) is used a single estimator, without its own ensembling.
> How does TabACE compare with the base model using its own ensembling at the same number of forward passes?

Thank you for requesting a budget-matched native-ensemble comparison. **Official Comment C2** reports Native-6, Random-6, and learned static weighting against Shared's one Base plus five alternatives, all with six deployed predictions per query. For TabICL-v2 under corruptions, Native-6 achieves **0.507/0.683 Acc/AUC**, versus Shared's **0.575/0.720**; regression $R^2$/nRMSE is **−0.597/1.145** versus **0.363/0.783**. Shared also improves all four metrics under shifts. Complete results are in Appendix E.1, Tables 16–19 (pp. 36–37).

**The benefits persist when native ensembling receives the same prediction budget**, complementing the generalization, clinical-transfer, and fine-tuning evidence above. We thank you for these constructive suggestions, which helped us strengthen the comparisons and clarify the value of inference-time context adaptation.
