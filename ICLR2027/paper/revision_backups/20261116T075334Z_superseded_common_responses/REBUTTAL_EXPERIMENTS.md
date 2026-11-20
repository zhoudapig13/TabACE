# Official Comments — Common Responses

We thank the reviewers for their constructive feedback. We consolidate the shared experimental evidence in C1–C4 below and retain a separate, numbered response to each reviewer's questions.

---

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

---

## Official Comment C2: Native and established ensemble baselines

We added native ensembling, random-context averaging, learned static weighting, and the requested TFM ensemble baselines. **Native-6** enables the backbone's native six-estimator ensemble; **Random-6** averages six independently sampled contexts; **Convex Stacking-6** fits nonnegative calibration weights summing to one. **Shared uses one Base and five alternative predictions.** These deployed ensembles use six predictions per query; OligoICP 60→6 selects six deployed members from 60 candidates.

The table reports classification Acc/AUC and regression $R^2$/nRMSE. Complete clean results, adaptation markers, and candidate/deployment budgets appear in **Appendix E.1, Tables 16–19, pp. 36–37**, including DENSE [2], PHE [3], RF-PFN, OligoICP [4], GES, and robust aggregation.

| Backbone | Method | Corruptions: classification | Shifts: classification | Corruptions: regression | Shifts: regression |
|---|---|---|---|---|---|
| TabICL-v2 | Base | 0.494 / 0.676 | 0.763 / 0.826 | −1.077 / 1.270 | 0.273 / 0.730 |
| TabICL-v2 | Native-6 | 0.507 / 0.683 | 0.763 / 0.825 | −0.597 / 1.145 | 0.287 / 0.721 |
| TabICL-v2 | Random-6 | 0.523 / 0.696 | 0.764 / 0.828 | −0.397 / 1.089 | 0.351 / 0.702 |
| TabICL-v2 | DENSE-6 | 0.531 / 0.695 | 0.764 / 0.826 | — | — |
| TabICL-v2 | Convex Stacking-6 | 0.559 / 0.704 | 0.766 / 0.827 | −0.446 / 1.104 | 0.401 / 0.682 |
| TabICL-v2 | Shared | **0.575 / 0.720** | **0.772 / 0.831** | **0.363 / 0.783** | **0.467 / 0.659** |
| TabPFN-v3 | Base | 0.457 / 0.636 | 0.758 / 0.809 | −1.253 / 1.313 | 0.426 / 0.669 |
| TabPFN-v3 | Native-6 | 0.467 / 0.637 | 0.766 / 0.814 | −0.405 / 1.116 | 0.444 / 0.640 |
| TabPFN-v3 | PHE-6 | 0.464 / 0.626 | 0.763 / 0.816 | −0.387 / 1.111 | 0.424 / 0.659 |
| TabPFN-v3 | OligoICP 60→6 | — | — | −0.370 / 1.120 | 0.333 / 0.703 |
| TabPFN-v3 | Shared | **0.553 / 0.697** | **0.770 / 0.817** | **0.237 / 0.834** | **0.460 / 0.656** |

**Shared improves both classification metrics and regression $R^2$ over the displayed ensembles under corruptions and shifts.** It also has lower nRMSE throughout except against TabPFN Native-6 on shifts (0.656 versus 0.640). Native and random-context ensembles help in several settings, but do not reproduce Shared's overall robustness; **additional backbone predictions alone do not explain the gains**. Random-6 supplies the predictive-performance counterpart to Figure 5's latency comparison. The fusion-specific fixed-bank analysis is addressed separately in our response to BV6f, Part 1.

---

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

---

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

---

# Response to Reviewer SbVz

## Part 1: Generalization beyond calibration and constructor matching

> The Shortcut corruption corrupts the most predictive feature, and one of the five Shared classification contexts simply drops that feature.
>
> The weighting policy is calibrated on the same corruption types and strengths
> the Specific variant is even told which one is present.
> How does TabACE perform on corruptions unseen during calibration?

Thank you for raising this central concern. **Official Comment C1** reports the $2\times2$ Shortcut ablation and six unseen-mechanism evaluations. Without either Shortcut calibration or its matched constructor, D attains **0.69333/0.84662 Acc/AUC**, versus Base's 0.51719/0.65690. Shared also improves all four metrics under all six unseen mechanisms. **The gains therefore extend beyond matched repairs and calibration on the evaluated mechanism.**

Specific exploits family knowledge when available, as in problem-aware context construction [5]; **Shared uses no mechanism identifier and remains effective when that knowledge is unavailable**. Full protocols and results are in Appendix E.3–E.4 (Tables 22–24, p. 39).

## Part 2: Robustness under real-world distribution shifts

> The only natural-shift benchmark is Folktables, where the improvements are mostly small and half of the times similar to the improvements observed on the unshifted test data, suggesting a generic ensembling effect.
> And does it work across other real-world shifts, such as in the medical domain between eICU and MIMIC?

Thank you for suggesting broader natural-shift tests. **Official Comment C3** reports all twelve Folktables ID/OOD comparisons and six clinical settings. Both variants outperform Base, DistPFN–TabICL-v2, and Drift-Resilient in all twelve Folktables comparisons. For geographic TravelTime OOD, Base/Shared/Specific accuracy is **0.5472/0.5818/0.6012**.

While eICU/MIMIC authorization is pending, we tested bidirectional hospital-system transfer on PhysioNet Sepsis 2019 and admission-source transfer on UCI Diabetes 130-US Hospitals, using 1:1 case–control evaluations. **Specific > Shared > Base in all six ID/OOD settings**; readmission OOD accuracy is **57.27%/58.97%/59.24%** for Base/Shared/Specific. These are naturally occurring shifts, extending the evidence beyond Folktables (Appendix E.5, Tables 25–27, pp. 41–42).

We address the ordinary-ensembling explanation in **Part 4 and Official Comment C2**, with the same deployed prediction budget.

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

## Part 4: Comparison with the backbone's native ensemble

> The comparisons should report the performance when ensembling also the baselines, i.e., the main backbone (TabICL-v2) is used a single estimator, without its own ensembling.
> How does TabACE compare with the base model using its own ensembling at the same number of forward passes?

Thank you for requesting a budget-matched native-ensemble comparison. **Official Comment C2** reports Native-6, Random-6, and learned static weighting against Shared's one Base plus five alternatives, all with six deployed predictions per query. For TabICL-v2 under corruptions, Native-6 achieves **0.507/0.683 Acc/AUC**, versus Shared's **0.575/0.720**; regression $R^2$/nRMSE is **−0.597/1.145** versus **0.363/0.783**. Shared also improves all four metrics under shifts. Complete results are in Appendix E.1, Tables 16–19 (pp. 36–37).

**The benefits persist when native ensembling receives the same prediction budget**, complementing the generalization, clinical-transfer, and fine-tuning evidence above. We thank you for these constructive suggestions, which helped us strengthen the comparisons and clarify the value of inference-time context adaptation.

---

# Response to Reviewer BV6f

## Part 1: Independent contribution of query-dependent fusion

> The independent contribution of the fusion mechanism remains unclear. DENSE, TabPFN PHE, and OligoICP provide relevant ensemble strategies. The paper needs comparisons that match context membership, calibration information, and inference budget. The mean and median baselines also remove the Base gate, so their gaps do not isolate the weighting rule. The component ablations help, but some signals remain active elsewhere after their weighting coefficients are removed.
> With the same context bank, calibration data, Base gate, and inference budget, how does the full method compare with static learned weights and uncertainty-only weighting? This would clarify the value of query-dependent agreement.

Thank you for distinguishing ensemble competitiveness from fusion-specific evidence. **Official Comment C2** reports DENSE, PHE, OligoICP, native/random ensembles, and static stacking. For example, TabICL Convex Stacking-6 achieves **0.559/0.704 Acc/AUC** under corruptions, versus Shared's **0.575/0.720**.

**Fixed-bank ablations (Appendix D.3).** Each fold's five-context bank is fixed, and policies are fitted on outer-training datasets. Remaining aggregation/gate mechanisms are retained when one weighting component is removed. Results equally weight the corruption and shift tracks after averaging nonzero strengths.

| Fixed-bank configuration | Classification Acc | Regression nRMSE |
|---|---:|---:|
| Uniform prior ($\pi_r=1/R$) | 0.636 | 0.835 |
| Without agreement weighting ($\eta_S=0$) | 0.639 | 0.805 |
| Full fusion | **0.672** | **0.739** |

**Full fusion improves over uniform priors and confidence-based weighting without the agreement term**; displayed differences are significant under the appendix's Holm-corrected paired tests.

These are **weighting-term ablations**: $\eta_S=0$ removes agreement from branch weights while retaining it in context evidence; $\eta_Q=0$ retains confidence in the gate. We clarify this in Appendix D.3. The comparisons measure the incremental value of these weighting terms within the deployed architecture. Mean/median comparisons in D.2 instead compare aggregation pipelines with different gating choices.

## Part 2: Specific context selection and policy optimization

> Specific’s deployed branch counts and selection procedure are also missing. These gaps make it difficult to separate gains from better contexts, learned reliability, and the proposed query-dependent fusion.
> Please report the deployed context lists and branch counts for Specific. How are these lists selected? Which optimization settings are used to fit the fusion policy?

Thank you for requesting the deployment details. **Appendix B.4 reports the Shared registry (Table 8), Specific candidates (Tables 9–10), branch-count convention (B.4.5), and selection/calibration (B.4.6); B.5, Table 11 gives implementation settings.**

For declared family $z$, Specific adds registered family-directed candidates to Shared, then deploys the frozen selected list $\Phi_z$:

$$
\mathcal C_{\mathrm{Specific}}(z)=\mathcal C_{\mathrm{Shared}}\cup\mathcal C_z,
\qquad R_z=\lvert\Phi_z\rvert,\qquad N_{\mathrm{passes}}=1+R_z.
$$

- **Information:** family $z$, labeled source context, and available unlabeled target features; no target labels or strength input.
- **Candidates:** Shortcut includes predictive-feature removal, weak-feature subspaces, clipping, and stable-row filtering; Boundary uses neighborhood repair/stable subsets; Domain/Covariate uses target matching and location-scale or quantile transport. Different parameterizations are separate candidates.
- **Selection:** five-fold dataset-level OOF; select members and fit the family policy on the other four folds, then freeze both for the held-out dataset. Candidate-pool size differs from deployed branch count.
- **Objective:** classification cross-entropy or regression squared error, over calibration strengths $\{0,0.2,0.5,0.8,1\}$.
- **Optimization:** fit priors $\pi_r$, evidence coefficients $\eta_{Q,r},\eta_{S,r}$, and cap $\alpha_{\max}$; regression also calibrates $\tau_y$. Equation 46 uses softmax priors, softplus nonnegative coefficients/positive scale, and a sigmoid cap, with the backbone frozen.

Shared has **five alternatives plus Base** for either task. The explicit Folktables deployment has **five/eight alternatives for Shared/Specific**, or six/nine predictions including Base (Appendix E.5.1, p. 40; Official Comment C3).

The complementary ablations show that **constructed contexts supply useful alternatives, learned priors suppress unreliable branches, and query-dependent agreement favors predictions supported by other branches** (C1 and Part 1 above).

## Part 3: Generalization coverage and distribution of gains

> The robustness evidence is narrower than the overall results suggest. Calibration covers the same registered mechanisms used in evaluation. The natural-shift study contains only two migration settings and three tasks. Gains also vary substantially across families. Based on Table 15, Domain and Temporal account for roughly 86% of Shared’s total regression improvement across the seven families.

Thank you for highlighting both coverage and uneven gains. **Official Comment C1** reports the mechanism-held-out Shortcut ablation and six unseen mechanisms; D achieves **0.69333/0.84662 Acc/AUC** without either Shortcut calibration or a matched constructor, versus Base's 0.51719/0.65690. Shared improves Acc and $R^2$ in all six new settings.

**Official Comment C3** broadens the natural-shift evidence beyond Folktables to hospital-system and admission-source transfer. All three medical OOD settings improve; readmission Base/Shared/Specific accuracy is **57.27%/58.97%/59.24%** in the 1:1 case–control evaluation.

**We agree that gains vary substantially by family**, with Domain and Temporal contributing most to the aggregate regression improvement. Appendix Table 15 reports each family separately: averaged over nonzero strengths, **Shared improves $R^2$ in all seven families**. The family-level results and additional held-out-mechanism/clinical tests show that the benefits extend beyond the two largest contributors.

## Part 4: High-confidence errors and extreme regression branches

> The mechanism has relevant limitations: the entropy gate suppresses corrections to confident Base errors, and median-based support scoring does not prevent an extreme regression branch from pulling the weighted mean. These cases matter for the paper’s robustness motivation.
> How often can TabACE correct high-confidence Base errors? How sensitive is the regression output to one extreme alternative prediction? Are branch predictions bounded or clipped?

Thank you for identifying these failure modes. **Official Comment C4** gives their definitions, frequencies, and original-versus-extended results; the full analysis is in Appendix E.7, Tables 29–31 (pp. 43–44).

- **Confident errors:** Base predictions with maximum class probability ≥0.9 that are incorrect affect **2.06%–2.44%** of query–condition pairs. The extension corrects **17.09%/26.43%** of these errors for TabICL/TabPFN, raising overall Acc from **67.5532% to 67.8380% / 66.1437% to 66.6759%**. Aggregation is unchanged; the added confidence/disagreement/consensus score is calibrated on training folds only.
- **Single extremes:** original scalar predictions are not additionally clipped. Only **0.109%–0.379%** of queries have one alternative beyond the median-centered ±3 source-label-SD interval. Clipping changes ordinary average $R^2$ by **<0.0007**, while reducing stress-test output displacement P95 from **24.264 to 0.996** source SDs for TabICL corruptions.
- **Joint errors:** material fusion harm affects **0.021%–0.547%** of query–condition pairs. Source-calibrated residual correction and risk-based contraction modestly improve average regression performance, recovering **30.32%/64.75%** of excess fusion loss within the fixed severe-joint-error cohort.

**These cases have limited measured aggregate impact, and the additional diagnostics motivated useful improvements while preserving the frozen-backbone design.**

## Part 5: Calibration protocol, sample sizes, and uncertainty

> Which datasets and conditions are used to calibrate these policies? Please provide the evaluation sample sizes and uncertainty estimates for the reported gains.

**Calibration excludes the evaluated dataset**, using five-fold dataset-level OOF. Episodes cover registered mechanisms at strengths $\{0,0.2,0.5,0.8,1.0\}$; member selection and policy parameters are frozen before held-out evaluation. There are **72/35 classification/regression datasets for corruptions and 35/33 for shifts**. Classification uses 128 evaluation rows per dataset; regression uses up to 512 for corruptions and 256 for shifts (Appendix B.1.1, Table 4).

**Synthetic response curves include 95% bootstrap confidence intervals** from 10,000 resamples of complete seed–DGP–scenario units (Appendix B.1.2). Three unseen confirmation seeds × five DGPs × twelve non-clean families give **180 aligned units**; each nonzero strength has 75 corruption or 105 shift units. Whole-unit resampling preserves within-experiment dependence. Folktables seed variability is summarized in **C3** and Appendix E.5.3.

We thank you for helping us strengthen the comparisons, deployment details, and failure-mode analysis. If these additional results and clarifications resolve your main concerns, we would greatly appreciate your consideration of a more favorable score.

---

# Response to Reviewer bPdS

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

## Part 2: Ensemble baselines and random-context accuracy

> TabACE is an ensemble method over TFMs, yet the paper does not compare it against established or even naive ensemble methods. A single forward pass with a randomly sampled context is a potentially unstable and weak baseline, especially with only 256 context rows. Thus, the reported gains may reflect generic ensemble benefits rather than TabACE’s specific contributions.
> Figure 5 reports latency for random contexts but not their accuracy. Could the authors report accuracy to enable a fair comparison?

Thank you for requesting these predictive comparisons. **Official Comment C2** reports Native-6, Random-6, and Convex Stacking-6, with the same six deployed predictions as Shared. Random-6 reaches **0.523/0.696 Acc/AUC** under corruptions and **0.764/0.828** under shifts; Shared reaches **0.575/0.720** and **0.772/0.831**. For regression, their corruption $R^2$/nRMSE is **−0.397/1.089** versus **0.363/0.783**.

Thus, Random-6 now provides Figure 5's missing accuracy counterpart, and the native/static/established ensemble comparisons show that **generic ensembling does not reproduce Shared's overall robustness**. Complete results, including DENSE [2], PHE [3], RF-PFN, and OligoICP [4], are in Appendix E.1, Tables 16–19 (pp. 36–37).

## Part 3: Generalization beyond perturbation–constructor matching

> The perturbations appear closely matched to their corresponding constructors: Shortcut corrupts the top feature, while its constructor drops the top-correlated feature; Regression Concept introduces smooth drift while its constructor removes the strongest smooth component. Gains diminish when these matched constructors are absent and are most pronounced when shifts appear in the Shared bank.
> Without such a match (i.e., when the perturbation/shift is not included in the Shared Constructor bank), performance is nearly identical to Base. The gain is also minimal in Folktables settings when the shift is not by design. These results raise concerns about generalizability beyond the specifically matched perturbations and constructors.

Thank you for raising this concern. **Official Comment C1** separates calibration coverage and constructor matching in a $2\times2$ Shortcut experiment. C/D, which omit the matched constructor, achieve **0.71187/0.85757 and 0.69333/0.84662 Acc/AUC**, versus Base's 0.51719/0.65690; D also excludes Shortcut from calibration.

More directly, **Shared improves Acc and $R^2$ under all six mechanisms absent from calibration and dedicated constructor design** (C1; Appendix E.4, Tables 23–24). This evidence addresses the claim that performance is nearly identical to Base without a match. Specific exploits problem knowledge when available, consistent with problem-aware context construction [5], while **Shared requires no mechanism identifier**.

We agree that Folktables gains vary by task and migration setting. **C3** adds hospital-system and admission-source shifts: all six clinical ID/OOD settings improve, and readmission OOD Base/Shared/Specific accuracy is **57.27%/58.97%/59.24%** in the 1:1 case–control evaluation. Together these tests extend the evidence beyond specifically matched synthetic mechanisms.

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

## Part 5: Correlated regression errors and confident classification errors

> Regression gate robustness. Since Ureg increases with Base–aggregate disagreement, updates are largest when the alternatives disagree most with Base. How does the method behave when these alternatives are correlated and jointly incorrect?
> Failure under confident errors. In classification, a low-entropy but incorrect Base prediction closes the gate, potentially preventing useful corrections. How frequently does this occur, and how does it affect overall performance?

Thank you for identifying these limitations. **Official Comment C4** provides complete diagnostics and extension results (Appendix E.7, Tables 29–31, pp. 43–44).

**Regression:** the update $\alpha_i=\alpha_{\max}U_{\mathrm{reg},i}E_i$ depends on both disagreement and context evidence. Correlated alternatives can still agree on an incorrect answer. Material fusion harm in the severe-joint-error cohort affects **0.021%–0.547%** of all query–condition pairs. Source-calibrated residual correction and risk contraction improve corruption/shift-average $R^2$ from **−0.12264 to −0.09363** for TabICL and **−0.19912 to −0.16110** for TabPFN, recovering **30.32%/64.75%** of this cohort's excess fusion loss.

**Classification:** high-confidence Base errors (maximum probability ≥0.9) affect **2.06%–2.44%** of queries. Adding relative confidence, disagreement, and consensus to update-strength estimation corrects **17.09%/26.43%** of these errors; overall Acc becomes **67.8380%/66.6759%**, versus 67.5532%/66.1437% originally. These cases have limited measured aggregate impact, and the extensions provide modest overall improvements with frozen backbones.

We thank you for helping us strengthen the context-budget tests, ensemble baselines, generalization evidence, and deployment details. If these additional results and clarifications resolve your main concerns, we would greatly appreciate your consideration of a more favorable score.

---

# References

[1] Ma et al. [TabDPT: Scaling Tabular Foundation Models on Real Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html). NeurIPS, 2025.

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[3] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[4] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.

[6] Lee. [Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](https://arxiv.org/abs/2605.04363). ICML, 2026.

[7] Helli et al. [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634). NeurIPS, 2024.
