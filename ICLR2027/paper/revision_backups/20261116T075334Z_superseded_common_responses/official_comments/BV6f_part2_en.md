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
