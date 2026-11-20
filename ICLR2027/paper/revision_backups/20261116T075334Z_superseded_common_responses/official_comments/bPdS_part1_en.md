## Part 1: Context size and larger-context evaluation

**Weakness 1 — justification for the context budget**

> The experiments use only 256 context rows without justification, despite several CC18 datasets being much larger (e.g., CIFAR-10 has 60K rows and Devanagari-Script about 92K). This is particularly concerning given that TabICLv2 and TabPFNv3 support much larger contexts (48K rows or more). Such aggressive subsampling may degrade model performance and increase prediction instability, potentially favoring ensembling. The authors should justify this choice and evaluate larger context sizes.

**Question — classification/regression context budgets and persistence of gains**

> Why use 256 labeled rows for classification but up to 2,048 for regression? Do the reported gains persist with larger context sizes?

Thank you for highlighting the context-budget choice. We used a compact 256-row setting to keep the broad robustness evaluation computationally manageable, informed by the small-context setting examined in TabDPT [1] (its inference ablation explicitly includes 256 rows). Classification and regression used separate sampling protocols, with up to 2,048 source-context rows for regression. **We have now extended the evaluation to common context caps of 1,024, 4,096, and 8,192 rows for both tasks.** The following TabICL-v2 results cover 72 classification and 35 regression datasets for clean/corrupted data, and 35 classification and 33 regression datasets for shifts.

**Classification (Acc / AUC; Appendix E.2, Tables 20–21, p. 38).**

| Context cap | Method | Clean (72 datasets) | Five corruptions (72) | Seven shifts (35) |
|---:|---|---:|---:|---:|
| 1024 | Base | 0.8254 / 0.9218 | 0.5332 / 0.7015 | 0.7635 / 0.8474 |
| 1024 | Shared | 0.8268 / 0.9223 | 0.6586 / 0.8044 | 0.7762 / 0.8661 |
| 1024 | Specific | **0.8289 / 0.9229** | **0.6917 / 0.8374** | **0.8047 / 0.8767** |
| 4096 | Base | 0.8381 / 0.9304 | 0.5446 / 0.7105 | 0.7607 / 0.8554 |
| 4096 | Shared | 0.8402 / 0.9306 | 0.6790 / 0.8116 | 0.7738 / 0.8724 |
| 4096 | Specific | **0.8436 / 0.9308** | **0.7071 / 0.8433** | **0.7982 / 0.8806** |
| 8192 | Base | 0.8416 / 0.9314 | 0.5469 / 0.7107 | 0.7625 / 0.8576 |
| 8192 | Shared | 0.8433 / 0.9315 | 0.6814 / 0.8124 | 0.7746 / 0.8734 |
| 8192 | Specific | **0.8456 / 0.9316** | **0.7115 / 0.8440** | **0.8042 / 0.8824** |

**Regression ($R^2$ / nRMSE).**

| Context cap | Method | Clean (35 datasets) | Five corruptions (35) | Seven shifts (33) |
|---:|---|---:|---:|---:|
| 1024 | Base | 0.7557 / 0.4680 | −1.0547 / 1.2598 | 0.2891 / 0.6659 |
| 1024 | Shared | 0.7570 / 0.4662 | 0.4779 / 0.6803 | 0.5816 / 0.5533 |
| 1024 | Specific | **0.7573 / 0.4633** | **0.5968 / 0.6051** | **0.7322 / 0.4569** |
| 4096 | Base | 0.7590 / 0.4647 | −1.0380 / 1.2560 | 0.3265 / 0.6379 |
| 4096 | Shared | 0.7599 / 0.4627 | 0.4800 / 0.6793 | 0.5980 / 0.5325 |
| 4096 | Specific | **0.7604 / 0.4622** | **0.5985 / 0.6040** | **0.7512 / 0.4306** |
| 8192 | Base | 0.7596 / 0.4642 | −1.0317 / 1.2545 | 0.3287 / 0.6310 |
| 8192 | Shared | 0.7603 / 0.4623 | 0.4809 / 0.6790 | 0.5864 / 0.5328 |
| 8192 | Specific | **0.7609 / 0.4621** | **0.5989 / 0.6038** | **0.7649 / 0.4109** |

**Shared and Specific continue to outperform Base under both corruptions and shifts at every evaluated context cap, while maintaining close clean performance.** The larger contexts also strengthen Base on clean data, yet the robustness benefits remain. These results directly address the concern that TabACE’s gains arise only from a small, unstable 256-row context.

**References**

[1] Ma et al. [TabDPT: Scaling Tabular Foundation Models on Real Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html). NeurIPS, 2025.
