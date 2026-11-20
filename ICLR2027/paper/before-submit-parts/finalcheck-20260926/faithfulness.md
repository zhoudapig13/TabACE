# Final Internal Faithfulness Check — `main_iclr.tex`

_Generated: 2026-09-26 · Scope: assembled ICLR 2027 review manuscript · No manuscript files modified_

## Summary
- 🔴 Desk-reject risks: 0
- 🟠 Reviewers will frown: 2
- 🔵 Optional polish: 0

## 🟠 Reviewers will frown

### Cross-section method consistency

- 🟠 **sections/00_abstract_iclr.tex:5–6 ↔ sections/04_method.tex:161–165** — the abstract calls TabACE a “training-free” strategy, while the method explicitly fits context priors, score coefficients, an output scale, and the integration cap on outer-training calibration episodes. Freezing the backbone and fitting a separate fusion policy are compatible, but “training-free” ordinarily denies the latter and is therefore internally misleading. _Fix:_ use a narrowly accurate phrase such as “backbone-training-free” or “without backbone updates,” while retaining the existing calibration description.
  ```latex
  % sections/00_abstract_iclr.tex:5-6
  contexts, we propose \textbf{\method{}} (Adaptive Context Ensembling), a model-agnostic
  and training-free inference strategy to enhance prediction robustness.
  ```
  ```latex
  % sections/04_method.tex:161-165
  \paragraph{Outer-fold policy learning.}
  Calibration episodes from outer-training datasets fit the context priors, score
  coefficients, output scale, and integration cap. Each episode pairs a labeled
  source context with a pseudo-target feature set whose labels enter only the
  calibration loss. The fitted policy and $\Phi$ are fixed before held-out inference.
  ```

- 🟠 **sections/context_construction_details.tex:274–280 ↔ sections/appendix.tex:1291–1295 ↔ sections/06_results.tex:97–103 (Figure `fig:inference-cost`)** — the deployed Specific branch set is described once as a selected member list $\Phi_z$ with variable count $R_z$, but the implementation table says each configuration retains its complete registered pool. The cost figure visibly includes a Specific bar under a five-alternative-context caption, while the construction text explicitly states only that the latency experiment fixes $R=5$ for Shared. Together these passages leave it unclear whether Specific deploys the full pool or a selected subset, and whether its plotted latency is a fixed-$R=5$ diagnostic or the cost of the reported Specific configuration. _Fix:_ make the three descriptions use one deployment rule and explicitly identify the Specific branch count used in the latency figure.
  ```latex
  % sections/context_construction_details.tex:274-280
  Specific retains all Shared contexts and activates the complete registered pool
  for family $z$. The pool has no protocol-level candidate-count cap. Each range,
  repair strength, subspace, and transport level below defines a separate
  candidate. This pool is the outer-fold selection space, not the deployed branch
  count: the selected member list $\Phi_z$ is frozen before held-out inference,
  which uses one Base pass and $R_z=|\Phi_z|$ alternative passes. The latency
  experiment fixes $R=5$ for Shared.
  ```
  ```latex
  % sections/appendix.tex:1291-1295
      Candidate selection
        & Each configuration retains its complete registered pool. Five-fold
          dataset-level OOF fits source-side policy parameters on the other four
          folds and freezes them for the held-out dataset, using cross-entropy for
          classification and squared error for regression. \\
  ```
  ```latex
  % sections/06_results.tex:97-103
  \begin{figure}[H]
    \centering
    \includegraphics[width=0.96\textwidth]{figures/tabace_inference_cost_ablation.pdf}
    \caption{Inference-cost ablation with five alternative contexts and a
    separately evaluated Base branch. Panels (a--b) decompose median latency;
    (c) compares original and cached inference; (d) includes the one-time cache
    setup in the amortized speedup.}
  ```

## What I checked / skipped
- ✅ Read the assembled main-text chain and active appendix chain, excluding `%` comments and the `\iffalse ... \fi` legacy block.
- ✅ Cross-checked the headline dataset/family counts, synthetic-unit count, stability win rates, clean/nonzero-strength claims, natural-shift winners, conventional-predictor gains, caching speedups, ablation values, and gate-bin values against their tables, captions, or plotted annotations. No numerical mismatch beyond legitimate rounding or different-track subsets was found.
- ✅ Visually inspected the active framework figure; the main OpenML strength curves; synthetic stability and controlled-response figures; inference-cost figure; all three additional-backbone strength figures; and the four appendix ablation/diagnostic figures. The stated main trends and annotated values agree with the rendered figures.
- ✅ Checked the principal notation flow across method, theory, Figure 1, and appendix: $x_i^t$, $o_i^{(r)}$, $p_i^{(r)}$, $\hat y_i^{(r)}$, $w_i^{(r)}$, $U_i$, $E_i$, $\alpha_i$, $\Phi$, and $R$ are otherwise materially consistent.
- ⏭️ External citation truth, venue compliance, anonymization, and compilation/page count were outside this fragment’s scope.
- ℹ️ The original limitations/future-work paragraph was left as written, per instruction.
