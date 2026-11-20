## Official Comment C4: Confident errors and regression robustness

We thank the reviewers for identifying these failure modes. We quantified their frequency and impact on reconstructed benchmarks and evaluated extensions with frozen backbones.

**High-confidence Base errors affect only 2.06%–2.44% of all query–condition pairs** in the five-corruption evaluation (Appendix E.7.1, Table 29, p. 43). High confidence means maximum Base class probability ≥0.9. We add relative Base/context confidence, disagreement, and branch consensus to update-strength estimation, allowing strong alternative evidence to correct confident Base predictions. Aggregation is unchanged; the added score is calibrated on training folds only.

| Backbone | High-confidence Base errors / all queries | Error correction rate: original → extended | Corrected errors | Overall Acc: original → extended |
|---|---:|---:|---:|---:|
| TabICL | 2.06% | 0 → **17.09%** | 648 | 67.5532% → **67.8380%** |
| TabPFN | 2.44% | 0 → **26.43%** | 1191 | 66.1437% → **66.6759%** |

Correction rates are conditional on high-confidence Base errors; Acc is macro-averaged over datasets and conditions. **The extension corrects a substantial fraction of these errors, with modest overall gains of 0.285 and 0.532 percentage points.**

**One extreme regression branch: ordinary impact is small.** Across four backbone/track settings, **0.109%–0.379%** of queries have exactly one alternative outside the interval below; clipping changes average $R^2$ by **<0.0007** in every setting. Original scalar branch predictions are not additionally clipped; the output is a convex combination of Base and the alternative aggregate. We add a median-centered bound requiring no target labels before recomputing fusion:

$$
\widetilde y_i^{(r)}=m_i+\operatorname{clip}\!\left(\widehat y_i^{(r)}-m_i,-3s_{\mathrm{src}},3s_{\mathrm{src}}\right),
$$

Here $m_i$ is the alternative median and $s_{\mathrm{src}}$ the source-label standard deviation; Base is unchanged. With one branch replaced by an extreme prediction, clipping reduces the TabICL five-corruption output displacement P95 from **24.264 to 0.996 source standard deviations**, **suppressing extreme-branch influence while preserving ordinary performance.**

The regression update is $\alpha_i=\alpha_{\max}U_{\mathrm{reg},i}E_i$: disagreement affects $U_{\mathrm{reg},i}$, while context evidence also controls the update.

**Correlated errors with material fusion harm affect only 0.021%–0.547% of query–condition pairs** (Appendix E.7.2, Tables 30–31, pp. 43–44). Severe joint error means all five alternatives err in the same direction, each by >3 evaluation-target standard deviations; material harm means fusion squared loss exceeds Base's by >0.01 target variance. Incidence is over all nonzero-strength query–condition pairs. Evaluation labels are used only for these diagnostics.

| Backbone | Track | Severe joint error | Severe joint error with material fusion harm |
|---|---|---:|---:|
| TabICL | Five corruptions | 3.147% | 0.219% |
| TabICL | Seven shifts | 0.081% | 0.025% |
| TabPFN | Five corruptions | 4.017% | 0.547% |
| TabPFN | Seven shifts | 0.084% | 0.021% |

We combine source-calibrated residual correction with a learned harm-risk score to contract risky updates toward Base, bounding the final update by the source scale. Reconstruction results below macro-average within each track, then weight the corruption and shift tracks equally.

| Backbone | Clean $R^2$: original → extended | Corruption/shift $R^2$: original → extended | Corruption/shift nRMSE: original → extended | Excess fusion-loss recovery in the severe-joint-error cohort |
|---|---:|---:|---:|---:|
| TabICL | 0.71918 → **0.72052** | −0.12264 → **−0.09363** | 0.89654 → **0.88471** | **30.32%** |
| TabPFN | 0.72387 → **0.72542** | −0.19912 → **−0.16110** | 0.91558 → **0.90061** | **64.75%** |

Excess fusion loss is $\max\{\ell(\widehat y_{\mathrm{TabACE}},y)-\ell(\widehat y_{\mathrm{Base}},y),0\}$; recovery is relative to its original total in the fixed severe-joint-error cohort. **The extension modestly improves average performance and recovers substantial fusion harm in this infrequent cohort.** We appreciate your suggestion, which motivated these improvements.
