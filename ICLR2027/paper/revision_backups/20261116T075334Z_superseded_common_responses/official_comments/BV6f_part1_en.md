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
