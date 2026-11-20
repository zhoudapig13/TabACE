# Follow-up Response to Reviewer SbVz

We thank you for your follow-up and for recognizing the additional unseen-mechanism and fine-tuning experiments. We apologize for the inconsistent reporting and have corrected the comparisons as follows.

**Question 1 — Different Shared/Specific results in Table 28 and the main table**

> In Table 28, fine-tuned TabICL-v2 is compared against "adapted" versions of Shared and Specific that score much higher on corruptions than in Table 1 (e.g., 0.754 vs. 0.575 for Shared) despite nearly identical Base accuracy.

You are right to question this discrepancy. **Our earlier Table 28 used inconsistent context-length settings across the corruption and shift evaluations**, making the comparison with the main-table configuration inappropriate. We have now unified these settings, rerun the comparison, and **replaced Table 28 in Appendix E.6**. The corrected results are below. Classification cells report Acc/AUROC; regression cells report $R^2$/nRMSE.

| Task | Method | Clean | Five corruptions | Seven shifts |
| --- | --- | --- | --- | --- |
| Classification | Base | 0.8033 / 0.9007 | 0.4973 / 0.6729 | 0.7634 / 0.8246 |
| Classification | Shared | 0.8048 / 0.9030 | 0.5818 / 0.7253 | 0.7747 / 0.8392 |
| Classification | Specific | 0.8041 / 0.9013 | 0.6132 / 0.7661 | 0.8197 / 0.8576 |
| Classification | FT | 0.7970 / 0.8998 | 0.5399 / 0.6978 | 0.7687 / 0.8301 |
| Classification | FT+Specific | 0.8018 / 0.9020 | 0.6135 / 0.7706 | 0.8220 / 0.8682 |
| Regression | Base | 0.7425 / 0.4803 | −1.0763 / 1.2685 | 0.2746 / 0.7288 |
| Regression | Shared | 0.7400 / 0.4825 | 0.3637 / 0.7845 | 0.4652 / 0.6564 |
| Regression | Specific | 0.7519 / 0.4710 | 0.4855 / 0.7069 | 0.6531 / 0.5365 |
| Regression | FT | 0.6942 / 0.5275 | 0.3102 / 0.8494 | 0.3897 / 0.6898 |
| Regression | FT+Specific | 0.7253 / 0.4994 | 0.4401 / 0.7199 | 0.6101 / 0.5693 |

The corrected results show that **Shared outperforms FT in both robustness blocks for classification and regression**, while applying TabACE to FT further improves the fine-tuned backbone. These are the corrected results on which we base our comparison.

**Question 2 — Inconsistent Folktables results between Tables 2 and 25**

> Similarly, Shared results on Folktables differ between Table 25 and Table 2 for the same Base, and the paper does not explain what changed in either case.

Following Reviewer bPdS's request, we expanded the original single-run Folktables evaluation to **five seeds (0–4)**. We failed to synchronize the original single-run table with the updated repeated-run results, causing the inconsistency you identified. **The main Folktables table (Table 2) has now been updated to the repeated-run Shared/Specific means, consistent with Table 25.** We take responsibility for this reporting error.

**Question 3 — Whether the gains reflect generic ensembling**

> Finally, gains on natural shifts also appear on ID data, and neither these nor the unseen-mechanism results are compared against ensemble baselines (Native-6, Random-6), so a generic ensembling effect cannot be ruled out.

With the discussion period ending, we have not completed the corresponding ensemble controls on Folktables and the unseen-mechanism suite. Our existing **OpenML comparisons, including CC18 classification**, nevertheless directly compare Shared with native ensembling, random-context averaging, and learned static weighting, all using **six predictions per query** (Appendix E.1, Tables 16–17):

| Task | Evaluation | Base | Native-6 | Random-6 | Convex Stacking-6 | Shared |
|---|---|---|---|---|---|---|
| Classification | Clean | 0.804 / 0.901 | 0.803 / 0.901 | 0.800 / 0.897 | 0.800 / 0.898 | 0.804 / 0.901 |
| Classification | Five corruptions | 0.494 / 0.676 | 0.507 / 0.683 | 0.523 / 0.696 | 0.559 / 0.704 | **0.575 / 0.720** |
| Classification | Seven shifts | 0.763 / 0.826 | 0.763 / 0.825 | 0.764 / 0.828 | 0.766 / 0.827 | **0.772 / 0.831** |
| Regression | Clean | 0.739 / 0.476 | 0.740 / 0.487 | 0.742 / 0.505 | 0.733 / 0.499 | 0.738 / 0.478 |
| Regression | Five corruptions | −1.077 / 1.270 | −0.597 / 1.145 | −0.397 / 1.089 | −0.446 / 1.104 | **0.363 / 0.783** |
| Regression | Seven shifts | 0.273 / 0.730 | 0.287 / 0.721 | 0.351 / 0.702 | 0.401 / 0.682 | **0.467 / 0.659** |

**Shared exceeds these ensemble controls in both robustness blocks**, showing that ordinary ensembling does not reproduce TabACE's gains on the controlled benchmarks. If accepted, we will include the complete Folktables ensemble ablations in the camera-ready version.

We appreciate your careful reading and kindly ask that these clarifications and corrections be considered in the reviewer–AC discussion. We welcome further questions and discussion.
