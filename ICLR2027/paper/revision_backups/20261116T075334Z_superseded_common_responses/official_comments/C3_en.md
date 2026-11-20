## Official Comment C3: Natural shifts, clinical transfer, and repeated runs

We expanded the natural-shift comparisons on Employment, Income, and TravelTime: geographic transfer **California → Michigan (2018)** and temporal transfer **California 2014 → 2018**. We compare against DistPFN–TabICL-v2 [6] and Drift-Resilient TabPFN [7]. **Specific uses the source-to-target domain-shift configuration**; its eight alternatives retain the five Shared types and add three adaptation types, each capped at 512 rows, plus Base. Construction uses source labels and unlabeled target features. Appendix E.5.1 lists the rules; E.5.2, Table 25 (p. 41) gives the comparisons. Entries with ± report accuracy and sample standard deviation.

| Shift | Task | Domain | Base | DistPFN–TabICL v2 | Drift-Resilient | Shared | Specific |
|---|---|---|---:|---:|---:|---:|---:|
| Geographic | Employment | ID | 0.7922 | 0.7892 ± 0.0030 | 0.7885 ± 0.0071 | **0.7952 ± 0.0031** | **0.8089 ± 0.0042** |
| Geographic | Employment | OOD | 0.8043 | 0.8034 ± 0.0033 | 0.8022 ± 0.0067 | **0.8106 ± 0.0047** | **0.8245 ± 0.0040** |
| Geographic | Income | ID | 0.7888 | 0.7930 ± 0.0100 | 0.7856 ± 0.0147 | **0.8118 ± 0.0020** | **0.8201 ± 0.0003** |
| Geographic | Income | OOD | 0.7655 | 0.7768 ± 0.0288 | 0.7733 ± 0.0209 | **0.7823 ± 0.0063** | **0.7914 ± 0.0029** |
| Geographic | TravelTime | ID | 0.5912 | 0.5731 ± 0.0253 | 0.5723 ± 0.0252 | **0.6225 ± 0.0212** | **0.6569 ± 0.0101** |
| Geographic | TravelTime | OOD | 0.5472 | 0.5802 ± 0.0071 | 0.5667 ± 0.0338 | **0.5818 ± 0.0349** | **0.6012 ± 0.0062** |
| Temporal | Employment | ID | 0.7704 | 0.7770 ± 0.0043 | 0.7699 ± 0.0095 | **0.7861 ± 0.0028** | **0.7994 ± 0.0031** |
| Temporal | Employment | OOD | 0.7847 | 0.7804 ± 0.0090 | 0.7742 ± 0.0140 | **0.7922 ± 0.0053** | **0.8030 ± 0.0015** |
| Temporal | Income | ID | 0.7926 | 0.8009 ± 0.0050 | 0.8010 ± 0.0055 | **0.8126 ± 0.0042** | **0.8240 ± 0.0011** |
| Temporal | Income | OOD | 0.7844 | 0.7894 ± 0.0040 | 0.7867 ± 0.0054 | **0.8000 ± 0.0027** | **0.8106 ± 0.0031** |
| Temporal | TravelTime | ID | 0.5951 | 0.5937 ± 0.0235 | 0.5956 ± 0.0145 | **0.6237 ± 0.0076** | **0.6526 ± 0.0088** |
| Temporal | TravelTime | OOD | 0.5908 | 0.5731 ± 0.0309 | 0.5692 ± 0.0179 | **0.6045 ± 0.0126** | **0.6350 ± 0.0114** |

**Shared and Specific outperform Base and both baselines in all twelve ID/OOD comparisons; Specific is consistently best.** Appendix E.5.3, Table 26 (p. 41) separately reports variances for the source-supervised adapted configuration with **seeds 0–4** and fixed splits/hyperparameters; Specific exceeds Shared in all 60 paired evaluations.

**Clinical shifts (Appendix E.5.4, Table 27, p. 42).** As eICU/MIMIC access authorization is pending, we tested PhysioNet Sepsis 2019 hospital systems A→B and B→A, and UCI Diabetes 130-US Hospitals readmission. For readmission, non-emergency-room admission sources define ID and emergency-room admissions define OOD, following TableShift. Accuracy is evaluated on **1:1 case–control samples**.

| Dataset | Task | Domain | Base | Shared | Specific |
|---|---|---|---:|---:|---:|
| PhysioNet Sepsis 2019 | Sepsis A $\rightarrow$ B | ID | 70.20% | **70.71%** | **71.72%** |
| PhysioNet Sepsis 2019 | Sepsis A $\rightarrow$ B | OOD | 61.59% | **62.14%** | **63.04%** |
| PhysioNet Sepsis 2019 | Sepsis B $\rightarrow$ A | ID | 76.19% | **77.38%** | **78.57%** |
| PhysioNet Sepsis 2019 | Sepsis B $\rightarrow$ A | OOD | 63.79% | **64.60%** | **64.91%** |
| UCI Diabetes 130-US Hospitals | Readmission | ID | 54.82% | **55.26%** | **56.87%** |
| UCI Diabetes 130-US Hospitals | Readmission | OOD | 57.27% | **58.97%** | **59.24%** |

**Specific > Shared > Base in all six clinical ID/OOD settings.** These evaluations extend the evidence to naturally occurring hospital-system and admission-source shifts without synthesizing the evaluated corruption.

**References**

[6] Lee. [Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](https://arxiv.org/abs/2605.04363). ICML, 2026.

[7] Helli et al. [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634). NeurIPS, 2024.
