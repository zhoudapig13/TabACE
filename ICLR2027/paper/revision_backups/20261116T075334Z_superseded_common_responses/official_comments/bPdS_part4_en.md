## Part 4: Folktables configuration, seed variability, and shift-specific baselines

> Folktables evaluation. Which family was specified for the Specific setting? Could the authors report variance across random seeds and compare against DistPFN and Drift-Resilient TabPFN?

Thank you for requesting these implementation details and comparisons. We evaluate California → Michigan (2018) and California 2014 → 2018 on Employment, Income, and TravelTime. **Specific uses a source-to-target domain-shift configuration**, retaining five Shared context types and adding three adaptation types: **five Shared and eight Specific alternatives, each capped at 512 rows**, plus Base. Construction uses source labels and unlabeled target features.

**Context construction (Appendix E.5.1, p. 40).** Let $z_i$ be robustly standardized source features and $d_i$ the mean squared distance to ten nearest target examples. Outlier and source-label conflict scores are:

$$
o_i=\frac{1}{p}\sum_{j=1}^{p}z_{ij}^{2},\qquad
\delta_i=1-\max_c\frac{1}{10}\sum_{k\in N_{10}(i)}\mathbf{1}[y_k=c].
$$

Here $p$ is the feature count and $N_{10}(i)$ the ten source neighbors. Unless noted, retain the lowest scores and sample without replacement.

- **Target neighborhood (Shared/Specific):** Form target anchors $A_Q$ from the coordinate-wise 10th, 30th, 50th, 70th, and 90th percentiles; rank by $s_i=\min_{a\in A_Q}\lVert z_i-a\rVert_2^2$ and retain 80%.
- **Representative samples (Shared/Specific):** Rank by $s_i=d_i+0.35o_i$ and retain 70%.
- **Local stability (Shared/Specific):** Rank by $s_i=\delta_i+0.10d_i$ and retain 80%.
- **Feature subspace (Shared/Specific):** Remove $j^*=\arg\max_j\lvert\operatorname{corr}(X_j,y)\rvert$ from both context and query, then sample source rows.
- **Low conflict (Shared/Specific):** Rank by $s_i=\delta_i+0.20o_i$ and retain 75%.
- **Target-matched bootstrap (Specific):** Standardize continuous and one-hot encode categorical features; bootstrap with $w_i\propto\exp(-d_i^{\mathrm{mix}}/h)$, where $d_i^{\mathrm{mix}}$ is the ten-neighbor target distance and $h$ is its positive median, preserving class coverage.
- **Density-ratio bootstrap (Specific):** Fit a logistic domain classifier $g(x)$; bootstrap with $w_i\propto[\min\{\frac{g(x_i)}{1-g(x_i)}\frac{n_s}{n_t},10\}]^{1/2}$, preserving class coverage; $n_s,n_t$ are source and target sample counts.
- **Continuous-feature alignment (Specific):** For prespecified continuous features, use $T(x)=0.5x+0.5[(x-m_s)s_t/s_s+m_t]$, where $m$ is the median and $s=\max(\mathrm{IQR},1)$. Sample source rows without replacement, retaining source labels and categorical features.

**Shift-specific comparisons and uncertainty are consolidated in Official Comment C3.** Shared and Specific outperform Base, DistPFN–TabICL-v2 [6], and Drift-Resilient TabPFN [7] in all twelve ID/OOD comparisons. For geographic TravelTime OOD, accuracy is **0.5818 ± 0.0349 / 0.6012 ± 0.0062** for Shared/Specific, versus 0.5802 ± 0.0071 and 0.5667 ± 0.0338 for the two baselines. Full comparisons and seed-0–4 variance reporting are in Appendix E.5.2–E.5.3 (Tables 25–26, p. 41); the separate repeated adapted configuration has Specific above Shared in all 60 paired evaluations.

**References**

[6] Lee. [Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](https://arxiv.org/abs/2605.04363). ICML, 2026.

[7] Helli et al. [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634). NeurIPS, 2024.
