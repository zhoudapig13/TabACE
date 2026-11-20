## Part 3: Comparison and complementarity with robust fine-tuning

> I am also not sure why these context operations/augmentations should be hardcoded at inference time instead of learned. I think a baseline is a TFM fine-tuned on contexts with the same corruptions, which should learn to ignore those and possibly to generalize other similar corruptions.
> Would a TFM fine-tuned on contexts with the same corruptions be as robust, and does TabACE still improve its performance?

Thank you for suggesting this comparison. **FT means fine-tuning the pretrained backbone on corruption-augmented training episodes.** For TabICL-v2, we use 20% clean and 80% matched-corruption episodes at strengths $\{0.2,0.5,0.8,1.0\}$, updating the last four ICL blocks and prediction components with AdamW. Checkpoints are selected on held-out training data; final evaluation queries are excluded from training and selection. FT checkpoints are then frozen for FT+Shared/Specific.

All methods within this comparison use the same splits, queries, conditions, strength grid, and metrics. The supplementary protocol is in Appendix E.6, Table 28 (p. 42).

| Task | Method | Clean | Five corruptions (avg.) | Seven shifts (avg.) |
|---|---|---:|---:|---:|
| Classification | Base | 0.8033 / 0.9007 | 0.4973 / 0.6729 | 0.7634 / 0.8246 |
| Classification | Shared | 0.8058 / 0.9030 | 0.7535 / 0.8647 | 0.7708 / 0.8334 |
| Classification | Specific | 0.8167 / 0.9157 | 0.8196 / 0.9176 | 0.7795 / 0.8462 |
| Classification | FT | 0.7970 / 0.8998 | 0.6474 / 0.7865 | 0.7596 / 0.8272 |
| Classification | FT+Shared | 0.7994 / 0.9019 | 0.7695 / 0.8765 | 0.7687 / 0.8374 |
| Classification | FT+Specific | 0.8018 / 0.9020 | 0.8120 / 0.9056 | 0.7943 / 0.8525 |
| Regression | Base | 0.7425 / 0.4803 | −1.0763 / 1.2685 | 0.2746 / 0.7288 |
| Regression | Shared | 0.7400 / 0.4825 | 0.6733 / 0.5469 | 0.6796 / 0.5105 |
| Regression | Specific | 0.7519 / 0.4710 | 0.7578 / 0.4659 | 0.7722 / 0.4008 |
| Regression | FT | 0.6942 / 0.5275 | 0.6118 / 0.5901 | 0.6171 / 0.5599 |
| Regression | FT+Shared | 0.7178 / 0.5062 | 0.7160 / 0.5074 | 0.7120 / 0.4796 |
| Regression | FT+Specific | 0.7253 / 0.4994 | 0.7421 / 0.4829 | 0.7498 / 0.4301 |

**FT is effective, while Shared achieves comparable or better performance without backbone fine-tuning.** Shared/Specific further improve FT, and the combined variants perform broadly similarly to their direct inference-time counterparts. **Inference-time context construction therefore remains useful alongside corruption-based fine-tuning.**

Across 175 dataset–track jobs on **eight RTX 5090 GPUs (32 GB each; one GPU per job)**, selected-model training averaged **41.4 s/model**; all recorded candidate training summed to **147.8 GPU-minutes**. Direct TabACE avoids this backbone-training stage. The paper reports Shared median inference latency of **872 ms for classification and 141 ms for regression**, using one Base plus five alternative passes. This comparison supports **lower upfront adaptation cost with broadly comparable prediction quality**.
