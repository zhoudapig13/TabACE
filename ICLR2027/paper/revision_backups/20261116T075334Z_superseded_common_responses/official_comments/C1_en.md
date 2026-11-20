## Official Comment C1: Generalization beyond calibration and constructor matching

We thank the reviewers for asking whether TabACE depends on matching the evaluation mechanism. We tested **calibration coverage and constructor matching separately**, then evaluated mechanisms absent from both calibration and dedicated constructor design.

**Shortcut ablation (Appendix E.3, Table 22, p. 39).** B/D exclude Shortcut from policy calibration; C/D use nonmatching constructors. Available supervision is the same across configurations.

| Configuration | Shortcut in calibration | Matched constructor | Acc $\uparrow$ | AUC $\uparrow$ |
|---|---|---|---:|---:|
| Base | — | — | 0.51719 | 0.65690 |
| A | Yes | Yes | **0.74394** | **0.87909** |
| B | No | Yes | 0.70777 | 0.85370 |
| C | Yes | No | 0.71187 | 0.85757 |
| D | No | No | 0.69333 | 0.84662 |

**C/D improve over Base without the matched feature-dropping constructor; B/D retain gains when Shortcut is excluded from calibration.** Thus, the benefit extends beyond a directly matched repair branch and calibration exposure to that mechanism.

**Unseen mechanisms (Appendix E.4, Tables 23–24, p. 39).** We tested three new corruptions and three new shifts at strengths $\{0.2,0.5,0.8,1.0\}$, averaging over strengths and the listed datasets. All six mechanisms were excluded from policy calibration, with no dedicated matching constructors, additional target labels, or mechanism identifiers. Corruptions modify source contexts; rotation and nonlinear sensor transformations modify classification queries or regression source-context features; interaction-dependent sampling reweights complete source rows with their labels.

| Unseen mechanism | Datasets (Cls./Reg.) | Base: Acc / AUC | Shared: Acc / AUC | Base: $R^2$ / nRMSE | Shared: $R^2$ / nRMSE |
|---|---:|---:|---:|---:|---:|
| Heavy-tailed noise | 64 / 35 | 0.7875 / 0.8932 | **0.8176 / 0.9068** | 0.6380 / 0.5417 | **0.6757 / 0.5161** |
| Column misalignment | 72 / 35 | 0.7824 / 0.8920 | **0.8186 / 0.9100** | 0.6179 / 0.5591 | **0.6751 / 0.5170** |
| Feature quantization | 64 / 35 | 0.7070 / 0.8782 | **0.7576 / 0.8902** | 0.3743 / 0.7070 | **0.5023 / 0.6452** |
| Feature rotation | 31 / 31 | 0.7609 / 0.8217 | **0.7861 / 0.8420** | 0.3375 / 0.7527 | **0.4663 / 0.6810** |
| Nonlinear sensor transformations | 32 / 33 | 0.8002 / 0.8524 | **0.8208 / 0.8600** | 0.1009 / 0.8235 | **0.4413 / 0.6665** |
| Interaction-dependent sampling | 35 / 33 | 0.8214 / 0.8676 | **0.8288 / 0.8779** | 0.6487 / 0.5350 | **0.6860 / 0.5042** |

**Shared improves all four metrics in all six settings.** These results directly address whether gains persist outside the registered calibration mechanisms and dedicated constructors.

Specific uses available family knowledge to add problem-directed contexts, consistent with the problem-aware context construction studied in SPN [5]. **Specific provides an option for exploiting available problem knowledge, while Shared establishes performance when that knowledge is unavailable.** The unseen-mechanism results above demonstrate that Shared remains effective in this latter setting.

**References**

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.
