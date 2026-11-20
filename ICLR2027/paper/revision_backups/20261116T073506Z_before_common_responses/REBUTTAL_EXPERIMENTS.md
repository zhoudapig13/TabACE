# Response to Reviewer SbVz

## Part 1: Generalization beyond calibration and constructor matching

Thank you for raising this important question about whether the gains depend on matching the evaluation conditions. We have examined this directly through the following additional experiments.

> The Shortcut corruption corrupts the most predictive feature, and one of the five Shared classification contexts simply drops that feature.
>
> The weighting policy is calibrated on the same corruption types and strengths

**Calibration and constructor matching (Appendix E.3, Table 22, p. 39).** To examine these two factors directly, we conducted a $2\times2$ Shortcut ablation, complementing our weighting-component ablations. We exclude Shortcut from policy calibration in B/D and use nonmatching context constructions in C/D, with the same available supervision across conditions. We find that C achieves 0.71187/0.85757 Acc/AUC; even with both calibration coverage and the matched constructor absent, D achieves 0.69333/0.84662, compared with Base's 0.51719/0.65690.

| Configuration | Shortcut in calibration | Matched constructor | Acc $\uparrow$ | AUC $\uparrow$ |
|---|---|---|---:|---:|
| Base | — | — | 0.51719 | 0.65690 |
| A | Yes | Yes | **0.74394** | **0.87909** |
| B | No | Yes | 0.70777 | 0.85370 |
| C | Yes | No | 0.71187 | 0.85757 |
| D | No | No | 0.69333 | 0.84662 |

We believe these results directly address both concerns: **constructor matching** and **calibration on the evaluated corruption types**. **C/D outperform Base without the matched feature-dropping constructor**, showing that the benefit extends beyond direct corruption–constructor pairing. **B/D also outperform Base when Shortcut is excluded from calibration**; together with the six unseen-mechanism results below, this demonstrates that **the learned weighting policy transfers to mechanisms absent from calibration**.

> the Specific variant is even told which one is present.

**Specific and Shared.** We would like to clarify the different deployment settings of the two variants. We designed Specific to use mechanism knowledge when it is available, allowing additional contexts to reflect the problem structure. Problem-aware context construction also appears in [SPN (Lv et al., ICML 2026)](https://proceedings.mlr.press/v306/lv26g.html) [1], which constructs examples for strategically modified inputs. We also evaluate Shared, which receives no mechanism identifier and uses fixed constructor rules. Thus, **Specific provides an option for exploiting available problem knowledge, while Shared establishes performance when that knowledge is unavailable.** In the experiments below, we further show that Shared consistently outperforms Base under three unseen corruptions and three unseen shifts.

> How does TabACE perform on corruptions unseen during calibration?

**Unseen mechanisms (Appendix E.4, Tables 23–24, p. 39).** We further tested Shared on three additional corruptions and three distribution shifts, excluding all six mechanisms from policy calibration and providing no dedicated matching constructors. We evaluated strengths $\{0.2,0.5,0.8,1.0\}$ and report averages over these strengths and the datasets listed below. All comparisons use the same available supervision, with no additional target labels or mechanism identifiers.

We apply the three corruptions to source contexts. For feature rotation and nonlinear sensor transformations, we modify query features in classification and source-context features in regression. Interaction-dependent sampling reweights complete source rows while retaining their associated labels.

| Unseen mechanism | Datasets (Cls./Reg.) | Base: Acc / AUC | Shared: Acc / AUC | Base: $R^2$ / nRMSE | Shared: $R^2$ / nRMSE |
|---|---:|---:|---:|---:|---:|
| Heavy-tailed noise | 64 / 35 | 0.7875 / 0.8932 | **0.8176 / 0.9068** | 0.6380 / 0.5417 | **0.6757 / 0.5161** |
| Column misalignment | 72 / 35 | 0.7824 / 0.8920 | **0.8186 / 0.9100** | 0.6179 / 0.5591 | **0.6751 / 0.5170** |
| Feature quantization | 64 / 35 | 0.7070 / 0.8782 | **0.7576 / 0.8902** | 0.3743 / 0.7070 | **0.5023 / 0.6452** |
| Feature rotation | 31 / 31 | 0.7609 / 0.8217 | **0.7861 / 0.8420** | 0.3375 / 0.7527 | **0.4663 / 0.6810** |
| Nonlinear sensor transformations | 32 / 33 | 0.8002 / 0.8524 | **0.8208 / 0.8600** | 0.1009 / 0.8235 | **0.4413 / 0.6665** |
| Interaction-dependent sampling | 35 / 33 | 0.8214 / 0.8676 | **0.8288 / 0.8779** | 0.6487 / 0.5350 | **0.6860 / 0.5042** |

We observe that Shared outperforms Base on **all four metrics in every setting**, despite receiving no indication of which mechanism is present. These results directly answer your question about unseen corruptions: Shared retains its benefits beyond the mechanisms represented in calibration and beyond those addressed by dedicated constructors.

## Part 2: Robustness under real-world distribution shifts

**Weakness 1 — natural shifts and generic ensembling effects**

> The only natural-shift benchmark is Folktables, where the improvements are mostly small and half of the times similar to the improvements observed on the unshifted test data, suggesting a generic ensembling effect.

Thank you for asking us to examine whether the natural-shift gains can be explained by ordinary ensembling. We repeated the Folktables experiments and observed consistent gains across geographic shifts (California $\rightarrow$ Michigan, 2018) and temporal shifts (California, 2014 $\rightarrow$ 2018), on Employment, Income, and TravelTime. We also compared against **DistPFN–TabICL v2** and **Drift-Resilient**, using the same geographic and temporal task definitions; DistPFN uses TabICL v2 as its backbone (Appendix E.5.2, Table 25, p. 41).

| Shift | Task | Domain | Base | DistPFN–TabICL v2 | Drift-Resilient | Shared | Specific |
|---|---|---|---:|---:|---:|---:|---:|
| Geographic | Employment | ID | 0.7922 | 0.7892 | 0.7885 | 0.7952 | **0.8089** |
| Geographic | Employment | OOD | 0.8043 | 0.8034 | 0.8022 | 0.8106 | **0.8245** |
| Geographic | Income | ID | 0.7888 | 0.7930 | 0.7856 | 0.8118 | **0.8201** |
| Geographic | Income | OOD | 0.7655 | 0.7768 | 0.7733 | 0.7823 | **0.7914** |
| Geographic | TravelTime | ID | 0.5912 | 0.5731 | 0.5723 | 0.6225 | **0.6569** |
| Geographic | TravelTime | OOD | 0.5472 | 0.5802 | 0.5667 | 0.5818 | **0.6012** |
| Temporal | Employment | ID | 0.7704 | 0.7770 | 0.7699 | 0.7861 | **0.7994** |
| Temporal | Employment | OOD | 0.7847 | 0.7804 | 0.7742 | 0.7922 | **0.8030** |
| Temporal | Income | ID | 0.7926 | 0.8009 | 0.8010 | 0.8126 | **0.8240** |
| Temporal | Income | OOD | 0.7844 | 0.7894 | 0.7867 | 0.8000 | **0.8106** |
| Temporal | TravelTime | ID | 0.5951 | 0.5937 | 0.5956 | 0.6237 | **0.6526** |
| Temporal | TravelTime | OOD | 0.5908 | 0.5731 | 0.5692 | 0.6045 | **0.6350** |

We find that **both Shared and Specific outperform Base, DistPFN–TabICL v2, and Drift-Resilient in all 12 ID/OOD comparisons**, with Specific giving the best accuracy throughout. These repeated results show that the gains persist across tasks and both types of natural shift, rather than arising from an isolated favorable setting.

We address the generic-ensembling explanation directly in **Part 4**, using native ensembling, random-context ensembling, and learned static weighting under the same prediction budget.

**Question 1 — other real-world shifts**

> And does it work across other real-world shifts, such as in the medical domain between eICU and MIMIC?

Thank you for suggesting medical shifts. We do not yet have access authorization for eICU and MIMIC, so we evaluated two publicly available clinical datasets: **PhysioNet Sepsis 2019** and **UCI Diabetes 130-US Hospitals**. For Sepsis, we test hospital-system transfer in both directions, A $\rightarrow$ B and B $\rightarrow$ A; ID patients come from the source system and OOD patients from the target system. For readmission, we follow the [TableShift admission-source domain definition](https://tableshift.org/datasets.html#hospital-readmission): non-emergency-room admission sources form the source/ID domain, while emergency-room admissions form the OOD domain. We report Accuracy on 1:1 case–control evaluations (Appendix E.5.4, Table 27, p. 42).

| Dataset | Task | Domain | Base | Shared | Specific |
|---|---|---|---:|---:|---:|
| PhysioNet Sepsis 2019 | Sepsis A $\rightarrow$ B | ID | 70.20% | **70.71%** | **71.72%** |
| PhysioNet Sepsis 2019 | Sepsis A $\rightarrow$ B | OOD | 61.59% | **62.14%** | **63.04%** |
| PhysioNet Sepsis 2019 | Sepsis B $\rightarrow$ A | ID | 76.19% | **77.38%** | **78.57%** |
| PhysioNet Sepsis 2019 | Sepsis B $\rightarrow$ A | OOD | 63.79% | **64.60%** | **64.91%** |
| UCI Diabetes 130-US Hospitals | Readmission | ID | 54.82% | **55.26%** | **56.87%** |
| UCI Diabetes 130-US Hospitals | Readmission | OOD | 57.27% | **58.97%** | **59.24%** |

We observe **Specific > Shared > Base in all six medical ID/OOD settings**. These results directly address your request for evidence beyond Folktables: we obtain improvements under naturally occurring hospital-system and admission-source shifts, without constructing synthetic corruptions for these evaluations.

## Part 3: Comparison and complementarity with robust fine-tuning

**Weakness 2 — inference-time context construction versus learning**

> I am also not sure why these context operations/augmentations should be hardcoded at inference time instead of learned. I think a baseline is a TFM fine-tuned on contexts with the same corruptions, which should learn to ignore those and possibly to generalize other similar corruptions.

**Question 2 — robustness after fine-tuning and additional TabACE gains**

> Would a TFM fine-tuned on contexts with the same corruptions be as robust, and does TabACE still improve its performance?

Thank you for suggesting this comparison. FT (fine-tuning) denotes updating the pretrained backbone on corruption-augmented training episodes. We fine-tuned **TabICL-v2** using 20% clean and 80% matched-corruption training episodes, with strengths $\{0.2,0.5,0.8,1.0\}$. We updated the last four ICL blocks and prediction components using AdamW, selected checkpoints on held-out training data, and excluded the final evaluation queries from training and selection. We then froze the FT checkpoints and applied Shared or Specific at inference time, yielding FT+Shared and FT+Specific (Appendix E.6, Table 28, p. 42).

Within this comparison, we evaluate all methods on the same dataset splits and query sets, under identical perturbation/shift conditions and strength grids, using the same evaluation metrics. The detailed supplementary protocol is given in Appendix E.6, Table 28, p. 42.

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

We agree that **FT is effective against the tested corruptions**. We also find that **Shared already achieves comparable or better performance without backbone fine-tuning**. Adding Shared or Specific further improves the FT model, while the combined variants reach broadly similar performance levels to their direct inference-time counterparts. We therefore believe these results address your question: **inference-time context construction remains valuable when corruption-based fine-tuning is available**, and the two approaches can also be combined.

We additionally measured the training overhead. We ran 175 dataset–track fine-tuning jobs on **eight NVIDIA GeForce RTX 5090 GPUs (32 GB each)**, with one GPU per job. Training runs for the selected models averaged **41.4 seconds per model**; the recorded candidate training runs required **147.8 GPU-minutes in total**, summed across GPUs rather than measured as wall-clock time. Applying TabACE directly to the pretrained backbone avoids this additional backbone-optimization stage. Its computational advantage here is therefore **lower upfront adaptation cost while achieving broadly comparable predictive performance**.

For comparison, the paper reports Shared median inference latencies of **872 ms for classification and 141 ms for regression**, using one Base and five alternative-context passes. Direct TabACE requires no backbone fine-tuning, avoiding the training cost above.

## Part 4: Comparison with the backbone's native ensemble

**Weakness 3 — the backbone is evaluated as a single estimator**

> The comparisons should report the performance when ensembling also the baselines, i.e., the main backbone (TabICL-v2) is used a single estimator, without its own ensembling.

**Question 3 — comparison at the same number of forward passes**

> How does TabACE compare with the base model using its own ensembling at the same number of forward passes?

Thank you for suggesting this comparison. We added **Native-6**, which enables TabICL-v2's native six-estimator ensemble, and compare it with Shared, which uses **one Base and five alternative-context predictions**. We also include **Random-6**, averaging predictions from six independently sampled contexts, and **Convex Stacking-6**, learning fixed nonnegative weights that sum to one on calibration data. All four ensemble methods use the same backbone and six predictions per query; the table reports Clean results and averages over the five corruptions and seven shifts.

| Task | Method | Clean | Five corruptions (avg.) | Seven shifts (avg.) |
|---|---|---:|---:|---:|
| Classification | Base | 0.804 / 0.901 | 0.494 / 0.676 | 0.763 / 0.826 |
| Classification | Native-6 | 0.803 / 0.901 | 0.507 / 0.683 | 0.763 / 0.825 |
| Classification | Random-6 | 0.800 / 0.897 | 0.523 / 0.696 | 0.764 / 0.828 |
| Classification | Convex Stacking-6 | 0.800 / 0.898 | 0.559 / 0.704 | 0.766 / 0.827 |
| Classification | Shared | **0.804 / 0.901** | **0.575 / 0.720** | **0.772 / 0.831** |
| Regression | Base | 0.739 / 0.476 | −1.077 / 1.270 | 0.273 / 0.730 |
| Regression | Native-6 | 0.740 / 0.487 | −0.597 / 1.145 | 0.287 / 0.721 |
| Regression | Random-6 | 0.742 / 0.505 | −0.397 / 1.089 | 0.351 / 0.702 |
| Regression | Convex Stacking-6 | 0.733 / 0.499 | −0.446 / 1.104 | 0.401 / 0.682 |
| Regression | Shared | 0.738 / 0.478 | **0.363 / 0.783** | **0.467 / 0.659** |

We find that **Shared outperforms native ensembling, random-context averaging, and static learned weighting on both classification metrics and both regression metrics under corruptions and shifts**. Native ensembling helps in some settings, but does not reproduce Shared's robustness. These comparisons directly address your concern: **the gains are not explained solely by increasing the number of backbone predictions**.

Our broader comparison also covers DENSE [2], GES, robust aggregation, PHE, RF-PFN, and OligoICP. **The complete baseline tables and budget conventions are provided in Appendix E.1, Tables 16–19, pp. 36–37**, covering TabICL and TabPFN classification and regression under clean data, the five corruptions, and the seven shifts.

Taken together, these experiments show that **TabACE's benefits extend beyond matched corruptions and constructors, remain complementary to backbone fine-tuning, and persist when the backbone's native ensemble receives the same prediction budget**. The additional natural-shift and clinical evaluations further support its usefulness beyond the controlled benchmarks. We thank you for these constructive suggestions, which helped us strengthen the experimental comparisons and clarify the contribution of inference-time context adaptation.

## References

[1] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

---

# Response to Reviewer BV6f

## Part 1: Independent contribution of query-dependent fusion

**Weakness 1 — matched ensemble comparisons and component isolation**

> The independent contribution of the fusion mechanism remains unclear. DENSE, TabPFN PHE, and OligoICP provide relevant ensemble strategies. The paper needs comparisons that match context membership, calibration information, and inference budget. The mean and median baselines also remove the Base gate, so their gaps do not isolate the weighting rule. The component ablations help, but some signals remain active elsewhere after their weighting coefficients are removed.

**Question 2**

> With the same context bank, calibration data, Base gate, and inference budget, how does the full method compare with static learned weights and uncertainty-only weighting? This would clarify the value of query-dependent agreement.

Thank you for highlighting the need to distinguish ensemble competitiveness from the contribution of the fusion rule. We address these through **comparisons with established ensembles** and **fixed-bank component ablations**.

**Comparison with ensemble methods.** We added DENSE [1], TabPFN PHE [2], OligoICP [3], and learned static stacking, alongside the native and random-context ensembles. The representative results below use TabICL-v2 or TabPFN-v3, evaluated on classification (Acc/AUC) and regression ($R^2$/nRMSE). The listed ensemble configurations deploy six predictions per query; OligoICP 60→6 selects six members from 60 candidates. Complete results, adaptation markers, and candidate/deployment budget conventions are provided in Appendix E.1, Tables 16–19, pp. 36–37.

| Backbone | Method | Five corruptions: classification | Seven shifts: classification | Five corruptions: regression | Seven shifts: regression |
|---|---|---:|---:|---:|---:|
| TabICL-v2 | DENSE-6 | 0.531 / 0.695 | 0.764 / 0.826 | — | — |
| TabICL-v2 | Convex Stacking-6 | 0.559 / 0.704 | 0.766 / 0.827 | −0.446 / 1.104 | 0.401 / 0.682 |
| TabICL-v2 | Shared | **0.575 / 0.720** | **0.772 / 0.831** | **0.363 / 0.783** | **0.467 / 0.659** |
| TabPFN-v3 | PHE-6 | 0.464 / 0.626 | 0.763 / 0.816 | −0.387 / 1.111 | 0.424 / 0.659 |
| TabPFN-v3 | OligoICP 60→6 | — | — | −0.370 / 1.120 | 0.333 / 0.703 |
| TabPFN-v3 | Shared | **0.553 / 0.697** | **0.770 / 0.817** | **0.237 / 0.834** | **0.460 / 0.656** |

We find that **Shared outperforms the highlighted established ensembles and static stacking under both corruptions and shifts**. These comparisons address the missing-baseline concern; to examine fusion independently of context construction, we turn to the fixed-bank experiment below.

**Contribution of query-dependent weighting.** Appendix D.3 fixes each fold's five-context bank and uses the corresponding outer-training folds for policy fitting, retaining the remaining aggregation and gate mechanisms when removing one component. The results equally weight the five-corruption and seven-shift tracks after averaging over nonzero strengths.

| Fixed-bank configuration | Classification Acc | Regression nRMSE |
|---|---:|---:|
| Uniform prior ($\pi_r=1/R$) | 0.636 | 0.835 |
| Without agreement weighting ($\eta_S=0$) | 0.639 | 0.805 |
| Full fusion | **0.672** | **0.739** |

In classification, setting $\eta_S=0$ leaves confidence-based branch weighting and the gate in place. **Adding agreement to branch weighting improves performance even when the context bank and the remaining gate mechanisms are retained**; all displayed differences from full fusion are significant under the appendix's Holm-corrected paired tests. The uniform-prior comparison further shows that learned context reliability and query-dependent agreement contribute complementary information.

We also appreciate your observation that some signals remain active in other paths. **These are weighting-term ablations**: setting $\eta_S=0$ removes agreement from branch weighting while retaining it in context evidence, and setting $\eta_Q=0$ similarly retains confidence in the gate. We have made these definitions explicit in Appendix D.3. This interpretation directly isolates the incremental value of the weighting terms within the deployed architecture; the mean/median comparisons in Appendix D.2 evaluate alternative aggregation pipelines, including their different gating choices.

## Part 2: Specific context selection and policy optimization

**Weakness 1 — deployed branch counts and selection procedure**

> Specific’s deployed branch counts and selection procedure are also missing. These gaps make it difficult to separate gains from better contexts, learned reliability, and the proposed query-dependent fusion.

**Question 1**

> Please report the deployed context lists and branch counts for Specific. How are these lists selected? Which optimization settings are used to fit the fusion policy?

Thank you for asking us to clarify the deployed configuration. **Appendix B.4 specifies the Shared registry (Table 8), family-directed Specific constructors (Tables 9–10), and candidate selection and aggregation (B.4.6); Appendix B.5, Table 11, summarizes implementation settings.** We summarize the TabACE settings below.

**Candidate construction and selection.** For a declared family $z$, Specific retains the Shared registry and adds the registered family-directed candidates:

$$
\mathcal C_{\mathrm{Specific}}(z)=\mathcal C_{\mathrm{Shared}}\cup\mathcal C_z,
\qquad R_z=|\Phi_z|,
\qquad N_{\mathrm{passes}}=1+R_z.
$$

Here $\Phi_z$ is the member list selected on the outer-training datasets and frozen before evaluating the held-out dataset. The candidate pool and deployed list are distinct: registered coverage, repair, feature-subspace, and transport levels define separate candidates, while $R_z$ counts only deployed alternatives.

- **Information supplied**: Declared family $z$, labeled source context, and available unlabeled target features; no target labels or corruption-strength input
- **Representative family-directed candidates**: Shortcut: predictive-feature removal, weak-feature subspaces, clipping, and stable-row filtering; Boundary: neighborhood label repair and stable subsets; Domain/Covariate: target matching and location-scale or quantile transport
- **Selection**: Five-fold dataset-level OOF; select the member list and fit the family-specific policy using the other four folds, then freeze both for the held-out dataset
- **Policy objective**: Classification cross-entropy; regression squared error, over calibration episodes with strengths $\{0,0.2,0.5,0.8,1\}$
- **Parameters fitted**: Context priors $\pi_r$, evidence coefficients $\eta_{Q,r},\eta_{S,r}$, and integration cap $\alpha_{\max}$; regression additionally calibrates $\tau_y$ and uses agreement coefficients

**Fusion-policy fitting.** Equation 46 defines calibration of the final prediction loss with the backbone frozen. The parameterization uses softmax for normalized context priors, softplus for nonnegative evidence coefficients and positive regression scale, and sigmoid for $\alpha_{\max}\in[0,1]$. **The family identifier determines the admissible constructor registry; the selected member list and fusion policy are fixed before held-out inference.**

The Shortcut constructor-matching ablation and the uniform-prior and agreement-weighting ablations in Appendix D.3 show how **context construction supplies effective alternative predictions, learned reliability priors reduce the influence of unreliable branches, and agreement weighting favors predictions supported by other branches for the current query**. We now explicitly report the context categories and branch-count convention in Appendix B.4.5: Shared uses five alternatives plus Base for both classification and regression; Specific uses the frozen selected list $\Phi_z$, with $R_z=|\Phi_z|$ alternatives and $1+R_z$ predictions including Base. Appendix E.5.1, p. 40 enumerates the deployed Folktables list: five Shared alternatives and eight Specific alternatives, giving six and nine total predictions, respectively. Candidate selection for the main benchmark is described in Appendix B.4.6.

## Part 3: Generalization coverage and distribution of gains

**Weakness 2 — calibration coverage, natural shifts, and concentrated regression gains**

> The robustness evidence is narrower than the overall results suggest. Calibration covers the same registered mechanisms used in evaluation. The natural-shift study contains only two migration settings and three tasks. Gains also vary substantially across families. Based on Table 15, Domain and Temporal account for roughly 86% of Shared’s total regression improvement across the seven families.

Thank you for pointing out both the calibration coverage and the uneven distribution of gains. We have added **mechanism-held-out ablations, six unseen mechanisms, and additional natural-shift evaluations**, and examine the regression results by family.

**Transfer beyond calibration and constructor matching (Appendix E.3, Table 22, p. 39).** We conduct a $2\times2$ Shortcut experiment, independently varying whether calibration includes Shortcut and whether a matched constructor is available, with the same available supervision across configurations.

| Configuration | Shortcut in calibration | Matched constructor | Acc | AUC |
|---|---|---|---:|---:|
| Base | — | — | 0.51719 | 0.65690 |
| A | Yes | Yes | **0.74394** | **0.87909** |
| B | No | Yes | 0.70777 | 0.85370 |
| C | Yes | No | 0.71187 | 0.85757 |
| D | No | No | 0.69333 | 0.84662 |

**B/D outperform Base when Shortcut is absent from calibration, and D also excludes the matched constructor, showing that the benefit extends beyond direct calibration–constructor matching.**

We further evaluate Shared on **three unseen corruptions and three unseen shifts**, all excluded from policy calibration and supplied with no dedicated matching constructors or mechanism identifiers, at strengths $\{0.2,0.5,0.8,1.0\}$. **Shared improves classification Acc and regression $R^2$ in all six settings**, extending the evidence beyond the registered calibration mechanisms. The complete results are provided in Appendix E.4, Tables 23–24, p. 39.

**Additional natural shifts.** Beyond the Folktables geographic and temporal evaluations, we test bidirectional hospital-system transfer on PhysioNet Sepsis 2019 and non-emergency-room to emergency-room admission-source transfer on UCI Diabetes 130-US Hospitals. The table reports OOD Acc for the 1:1 case–control evaluation (Appendix E.5.4, Table 27, p. 42).

| Dataset | Transfer task | Base | Shared | Specific |
|---|---|---:|---:|---:|
| PhysioNet Sepsis 2019 | Sepsis A→B | 61.59% | **62.14%** | **63.04%** |
| PhysioNet Sepsis 2019 | Sepsis B→A | 63.79% | **64.60%** | **64.91%** |
| UCI Diabetes 130-US Hospitals | Readmission | 57.27% | **58.97%** | **59.24%** |

**Shared and Specific improve Base in all three medical OOD settings, extending the natural-shift evidence beyond Folktables.**

**Family-specific gains. We agree that the magnitude of improvement varies substantially across families**, with Domain and Temporal contributing the largest share of the aggregate regression gain. Appendix Table 15 already reports each family separately: **averaged over the four nonzero strengths, Shared improves $R^2$ over Base in all seven families**. Together with the mechanism-held-out and clinical results above, this shows that the benefit extends beyond the two families contributing most to the aggregate improvement.

## Part 4: High-confidence errors and extreme regression branches

**Weakness 2 — limitations of the gate and support scoring**

> The mechanism has relevant limitations: the entropy gate suppresses corrections to confident Base errors, and median-based support scoring does not prevent an extreme regression branch from pulling the weighted mean. These cases matter for the paper’s robustness motivation.

**Question 3**

> How often can TabACE correct high-confidence Base errors? How sensitive is the regression output to one extreme alternative prediction? Are branch predictions bounded or clipped?

Thank you for highlighting these failure modes. We quantified their frequency and impact on reconstructed benchmarks and developed extensions with frozen backbones.

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

## Part 5: Calibration protocol, sample sizes, and uncertainty

**Question 4**

> Which datasets and conditions are used to calibrate these policies? Please provide the evaluation sample sizes and uncertainty estimates for the reported gains.

Thank you. **We calibrate on datasets separate from the evaluated dataset**, using five-fold dataset-level OOF evaluation. Calibration episodes cover the registered perturbations/shifts at strengths $\{0,0.2,0.5,0.8,1.0\}$; context selection and policy parameters are frozen before held-out evaluation. The public benchmark contains **72 classification / 35 regression datasets for perturbations and 35 / 33 for shifts**. Classification uses 128 evaluation rows per dataset; regression uses up to 512 for perturbations and 256 for shifts. Appendix B.1.1, Table 4 gives the partition details.

**Uncertainty estimates are already reported for the synthetic experiments:** the response curves include 95% bootstrap confidence intervals from 10,000 resamples of complete seed–DGP–scenario units (Appendix B.1.2). Confirmation uses three unseen seeds, five DGPs, and twelve non-clean families, giving **180 aligned units**, with 75 units per nonzero perturbation-strength point and 105 per shift-strength point. Resampling complete units preserves the dependence among queries and methods within each experiment.

We thank you for the constructive feedback, which helped us strengthen both the evaluation and the method. **The ensemble comparisons, fixed-bank ablations, and unseen-mechanism tests provide complementary evidence that TabACE’s benefits extend beyond generic ensembling and directly matched constructors.** Your questions also motivated explicit analyses of confident errors and extreme regression predictions, leading to targeted improvements while keeping the backbone frozen. We hope these results and clarifications address your concerns and make the contribution and deployment protocol clearer. If these additional results and clarifications resolve your main concerns, we would greatly appreciate your consideration of a more favorable score.

## References

[1] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[2] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[3] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.

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

**Weakness 2 — inadequate ensemble baselines**

> TabACE is an ensemble method over TFMs, yet the paper does not compare it against established or even naive ensemble methods. A single forward pass with a randomly sampled context is a potentially unstable and weak baseline, especially with only 256 context rows. Thus, the reported gains may reflect generic ensemble benefits rather than TabACE’s specific contributions.

**Question — accuracy of random-context ensembles**

> Figure 5 reports latency for random contexts but not their accuracy. Could the authors report accuracy to enable a fair comparison?

Thank you for suggesting these comparisons. **We now report the predictive performance of random-context ensembles alongside the native ensemble and learned static weighting.** Native-6 enables TabICL-v2’s six-estimator native ensemble; Random-6 averages predictions from six independently sampled contexts; Convex Stacking-6 learns fixed nonnegative weights summing to one on calibration data. Shared uses one Base and five alternative-context predictions. **All four ensemble methods use the same backbone and six predictions per query.** The table reports classification Acc/AUC and regression $R^2$/nRMSE under clean data, the five corruptions, and the seven shifts.

| Task | Method | Clean | Five corruptions (avg.) | Seven shifts (avg.) |
|---|---|---:|---:|---:|
| Classification | Base | 0.804 / 0.901 | 0.494 / 0.676 | 0.763 / 0.826 |
| Classification | Native-6 | 0.803 / 0.901 | 0.507 / 0.683 | 0.763 / 0.825 |
| Classification | Random-6 | 0.800 / 0.897 | 0.523 / 0.696 | 0.764 / 0.828 |
| Classification | Convex Stacking-6 | 0.800 / 0.898 | 0.559 / 0.704 | 0.766 / 0.827 |
| Classification | Shared | **0.804 / 0.901** | **0.575 / 0.720** | **0.772 / 0.831** |
| Regression | Base | 0.739 / 0.476 | −1.077 / 1.270 | 0.273 / 0.730 |
| Regression | Native-6 | 0.740 / 0.487 | −0.597 / 1.145 | 0.287 / 0.721 |
| Regression | Random-6 | 0.742 / 0.505 | −0.397 / 1.089 | 0.351 / 0.702 |
| Regression | Convex Stacking-6 | 0.733 / 0.499 | −0.446 / 1.104 | 0.401 / 0.682 |
| Regression | Shared | 0.738 / 0.478 | **0.363 / 0.783** | **0.467 / 0.659** |

**Shared outperforms native ensembling, random-context averaging, and static learned weighting on both classification metrics and both regression metrics under corruptions and shifts.** Random-context ensembling improves over Base in several settings, but does not reproduce Shared’s robustness. This directly addresses the concern that the improvements merely result from repeated backbone predictions with a small context.

We additionally evaluated DENSE [2], GES, robust aggregation, TabPFN PHE [3], RF-PFN, and OligoICP [4]. **The complete baseline tables and budget conventions are provided in Appendix E.1, Tables 16–19, pp. 36–37**, covering both TabICL and TabPFN classification and regression. The Random-6 row above supplies the predictive-performance counterpart to the random-context latency comparison in Figure 5.

## Part 3: Generalization beyond perturbation–constructor matching

**Weakness 3 — matched mechanisms and limited generalizability**

> The perturbations appear closely matched to their corresponding constructors: Shortcut corrupts the top feature, while its constructor drops the top-correlated feature; Regression Concept introduces smooth drift while its constructor removes the strongest smooth component. Gains diminish when these matched constructors are absent and are most pronounced when shifts appear in the Shared bank.

> Without such a match (i.e., when the perturbation/shift is not included in the Shared Constructor bank), performance is nearly identical to Base. The gain is also minimal in Folktables settings when the shift is not by design. These results raise concerns about generalizability beyond the specifically matched perturbations and constructors.

Thank you for raising this concern. **We directly tested whether TabACE remains effective without a matching constructor or calibration exposure to the evaluated mechanism.**

**Separating calibration coverage from constructor matching (Appendix E.3, Table 22, p. 39).** We conduct a $2\times2$ Shortcut ablation, independently varying these two factors, with the same available supervision across configurations. C/D use nonmatching constructions; B/D exclude Shortcut from policy calibration.

| Configuration | Shortcut in calibration | Matched constructor | Acc | AUC |
|---|---|---|---:|---:|
| Base | — | — | 0.51719 | 0.65690 |
| A | Yes | Yes | **0.74394** | **0.87909** |
| B | No | Yes | 0.70777 | 0.85370 |
| C | Yes | No | 0.71187 | 0.85757 |
| D | No | No | 0.69333 | 0.84662 |

**C/D outperform Base without the matched feature-dropping constructor; D retains this advantage even when Shortcut is also absent from calibration.** These results show that useful corrections can arise from the remaining contexts rather than requiring a directly matched repair branch.

Problem-aware construction is also a useful option when problem knowledge is available: SPN [5] constructs strategic in-context examples to adapt frozen TFMs to post-manipulation inputs. Specific uses available family knowledge to construct additional candidates, while **Shared requires no mechanism identifier and remains effective without dedicated matching constructors**, as tested below.

**Mechanisms outside the Shared bank (Appendix E.4, Tables 23–24, p. 39).** We evaluate three new corruptions and three new shifts, all excluded from policy calibration and without adding dedicated matching constructors. Results average strengths $\{0.2,0.5,0.8,1.0\}$ and the listed datasets, with the same available supervision and no additional target labels. The corruptions modify source contexts; rotation and nonlinear sensor transformations modify classification queries or regression source-context features; interaction-dependent sampling reweights complete source rows with their labels.

| Unseen mechanism | Datasets (Cls./Reg.) | Classification Acc: Base / Shared | Regression $R^2$: Base / Shared |
|---|---:|---:|---:|
| Heavy-tailed noise | 64 / 35 | 0.7875 / **0.8176** | 0.6380 / **0.6757** |
| Column misalignment | 72 / 35 | 0.7824 / **0.8186** | 0.6179 / **0.6751** |
| Feature quantization | 64 / 35 | 0.7070 / **0.7576** | 0.3743 / **0.5023** |
| Feature rotation | 31 / 31 | 0.7609 / **0.7861** | 0.3375 / **0.4663** |
| Nonlinear sensor transformations | 32 / 33 | 0.8002 / **0.8208** | 0.1009 / **0.4413** |
| Interaction-dependent sampling | 35 / 33 | 0.8214 / **0.8288** | 0.6487 / **0.6860** |

**Shared improves both classification Acc and regression $R^2$ in all six settings.** This directly addresses the statement that performance is nearly identical to Base when the evaluated mechanism is absent from the constructor bank: the additional evaluations demonstrate gains under precisely that condition.

**Natural shifts.** We agree that some Folktables gains are modest and depend on the task and migration setting. We therefore broadened the evaluation to bidirectional hospital-system transfer on PhysioNet Sepsis 2019 and admission-source transfer on UCI Diabetes 130-US Hospitals (non-emergency-room → emergency-room). Shared and Specific improve OOD Acc in all three transfer settings; for example, readmission Acc is **57.27% / 58.97% / 59.24% for Base / Shared / Specific**. Together with the unseen-mechanism tests, these clinical results provide further evidence of generalization beyond the specifically matched synthetic conditions. Complete clinical results for the 1:1 case–control evaluation are provided in Appendix E.5.4, Table 27, p. 42.

## Part 4: Folktables configuration, seed variability, and shift-specific baselines

**Question — Specific’s family information, uncertainty, and comparative methods**

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

**Shift-specific comparisons (Appendix E.5.2, Table 25, p. 41).** We added DistPFN–TabICL-v2 [6] and Drift-Resilient TabPFN [7]. Entries with ± report accuracy and sample standard deviation.

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

**Shared and Specific outperform Base and both baselines in all twelve ID/OOD comparisons; Specific is consistently best.**

**Seeds and variance.** Appendix E.5.3 (Table 26, p. 41) reports variances for the separate source-supervised adapted configuration, with **seeds 0–4** and fixed splits/hyperparameters. **Specific exceeds Shared in all 60 paired evaluations.**


## Part 5: Correlated regression errors and confident classification errors

**Question — regression gate robustness**

> Regression gate robustness. Since Ureg increases with Base–aggregate disagreement, updates are largest when the alternatives disagree most with Base. How does the method behave when these alternatives are correlated and jointly incorrect?

**Question — failure under confident errors**

> Failure under confident errors. In classification, a low-entropy but incorrect Base prediction closes the gate, potentially preventing useful corrections. How frequently does this occur, and how does it affect overall performance?

Thank you for identifying these failure modes. We agree that correlated alternatives can be jointly wrong and that low-entropy Base errors can suppress useful corrections. We quantified their frequency and impact on reconstructed benchmarks and used these findings to improve the update while keeping the backbone frozen.

For regression, the actual update is $\alpha_i=\alpha_{\max}U_{\mathrm{reg},i}E_i$: Base–aggregate disagreement increases $U_{\mathrm{reg},i}$, while context evidence also controls the final update. Mutually consistent errors can nevertheless support a wrong aggregate, which motivates the following diagnostic and extension.

**Correlated errors and harmful updates (Appendix E.7.2, Tables 30–31, pp. 43–44). Severe joint errors with material fusion harm are infrequent in this evaluation, affecting 0.021%–0.547% of query–condition pairs.** We nevertheless explicitly examine this failure mode: all five alternatives make same-direction errors, each exceeding three evaluation-target standard deviations. A joint error is counted as materially harmful when the original fusion's squared loss exceeds Base's by more than 0.01 times the target variance. Incidence uses all nonzero-strength query–condition pairs as its denominator; evaluation labels define these diagnostic events only.

| Backbone | Track | Severe joint error | Severe joint error with material fusion harm |
|---|---|---:|---:|
| TabICL | Five corruptions | 3.147% | 0.219% |
| TabICL | Seven shifts | 0.081% | 0.025% |
| TabPFN | Five corruptions | 4.017% | 0.547% |
| TabPFN | Seven shifts | 0.084% | 0.021% |

To reduce harmful common-mode updates, we further combine source-calibrated residual correction with a learned harm-risk score that contracts risky updates toward Base, with the final update bounded by the source scale. The following reconstruction results macro-average within each track and give the five-corruption and seven-shift tracks equal weight.

| Backbone | Clean $R^2$: original → extended | Corruption/shift $R^2$: original → extended | Corruption/shift nRMSE: original → extended | Excess fusion-loss recovery in the severe-joint-error cohort |
|---|---:|---:|---:|---:|
| TabICL | 0.71918 → **0.72052** | −0.12264 → **−0.09363** | 0.89654 → **0.88471** | **30.32%** |
| TabPFN | 0.72387 → **0.72542** | −0.19912 → **−0.16110** | 0.91558 → **0.90061** | **64.75%** |

Excess fusion loss is $\max\{\ell(\widehat y_{\mathrm{TabACE}},y)-\ell(\widehat y_{\mathrm{Base}},y),0\}$; recovery is measured against its original total within the fixed severe-joint-error cohort. **The extension produces modest average performance improvements while recovering a substantial fraction of the additional fusion harm within this infrequent cohort.**

**High-confidence Base errors affect a small fraction of the evaluated queries:** (Appendix E.7.1, Table 29, p. 43) on the five-corruption benchmark, they account for only **2.06%–2.44% of all query–condition pairs**. We define high confidence as a maximum Base class probability of at least 0.9. We extend the update-strength estimator with relative Base/context confidence, prediction disagreement, and branch consensus, so sufficiently strong alternative evidence can also correct a confident Base prediction. The branch aggregation is unchanged, and the added score is calibrated on training folds only.

| Backbone | High-confidence Base errors / all queries | Error correction rate: original → extended | Corrected errors | Overall Acc: original → extended |
|---|---:|---:|---:|---:|
| TabICL | 2.06% | 0 → **17.09%** | 648 | 67.5532% → **67.8380%** |
| TabPFN | 2.44% | 0 → **26.43%** | 1191 | 66.1437% → **66.6759%** |

Correction rates are conditional on high-confidence Base errors; overall Acc is macro-averaged across datasets and corruption conditions. **The extension corrects a meaningful fraction of these errors, with modest net gains in overall accuracy of 0.285 and 0.532 percentage points.**

We thank you again for the constructive suggestions. **The larger-context evaluations, ensemble comparisons, mechanism-held-out tests, and natural-shift results provide complementary evidence for TabACE’s robustness across context budgets and deployment settings.** Your questions also helped us clarify the context configurations and seed variability, and improve the treatment of confident errors and correlated regression failures. We hope these additions address your concerns and make the contribution and empirical evidence clearer. If these additional results and clarifications resolve your main concerns, we would greatly appreciate your consideration of a more favorable score.

## References

[1] Ma et al. [TabDPT: Scaling Tabular Foundation Models on Real Data](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html). NeurIPS, 2025.

[2] Khalifa et al. [Exploring Demonstration Ensembling for In-context Learning](https://arxiv.org/abs/2308.08780). 2023.

[3] Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). Nature, 2025.

[4] Li et al. [Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](https://arxiv.org/abs/2510.02476). NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences, 2025.

[5] Lv et al. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://proceedings.mlr.press/v306/lv26g.html). ICML, 2026.

[6] Lee. [Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](https://arxiv.org/abs/2605.04363). ICML, 2026.

[7] Helli et al. [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634). NeurIPS, 2024.
