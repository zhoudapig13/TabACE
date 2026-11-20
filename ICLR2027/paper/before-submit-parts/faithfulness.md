- 🔴 **sections/00_abstract_iclr.tex:15–16 ↔ sections/06_results.tex:133–139 (Figure~`fig:inference-cost`)** — the abstract says repeated inference is accelerated “by up to fivefold,” while the results report about $5.7\times$ and the rendered figure labels the cached variants $5.8\times$ and $5.7\times$. Because “up to” states an upper bound, 5 is not a faithful rounding of 5.7–5.8. _Fix:_ reconcile the abstract with the measured result (for example, “up to $5.8\times$” or “nearly sixfold”), after confirming which plotted value is the intended headline.
  ```latex
  % sections/00_abstract_iclr.tex:15–16
  caching reusable context computations accelerates repeated inference by up to
  fivefold.
  ```
  ```latex
  % sections/06_results.tex:133–139
  \paragraph{Inference cost.}
  For the fixed five-context Shared bank, Figure~\ref{fig:inference-cost}
  decomposes wall-clock latency into six backbone passes (Base plus five
  alternatives) and context construction/fusion. Caching reusable propensity
  quantities yields about $5.7\times$ faster repeated inference. After including
  the one-time cache construction, it retains about $3.9\times$ amortized speedup
  over ten batches.
  ```

- 🟠 **sections/context_construction_details.tex:272–278 ↔ sections/appendix.tex:1291–1295 ↔ sections/06_results.tex:97–103 (Figure~`fig:inference-cost`)** — the Specific deployment is described once as a selected member list whose size $R_z$ can differ from the full admissible pool, but the implementation table says every configuration retains its complete pool. The latency figure contains both Shared and Specific bars while its caption assigns five alternatives without resolving which interpretation was timed. A reader therefore cannot determine the Specific forward-pass count represented by the figure. _Fix:_ state one consistent deployed-pool rule and give the corresponding branch count for each timed variant; if five candidates were selected for the latency diagnostic, say so explicitly for Specific as well as Shared.
  ```latex
  % sections/context_construction_details.tex:272–278
  Specific retains all Shared contexts and activates the complete registered pool
  for family $z$. The pool has no protocol-level candidate-count cap. Each range,
  repair strength, subspace, and transport level below defines a separate
  candidate. This pool is the outer-fold selection space, not the deployed branch
  count: the selected member list $\Phi_z$ is frozen before held-out inference,
  which uses one Base pass and $R_z=|\Phi_z|$ alternative passes. The latency
  experiment fixes $R_z=5$ for Shared.
  ```
  ```latex
  % sections/appendix.tex:1291–1295
  Candidate selection
    & Each configuration retains its complete registered pool. Five-fold
      dataset-level OOF fits source-side policy parameters on the other four
      folds and freezes them for the held-out dataset, using cross-entropy for
      classification and squared error for regression. \\
  ```
  ```latex
  % sections/06_results.tex:97–103
  \begin{figure}[H]
    \centering
    \includegraphics[width=0.96\textwidth]{figures/tabace_inference_cost_ablation.pdf}
    \caption{Inference-cost ablation with five alternative contexts and a
    separately evaluated Base branch. Panels (a--b) decompose median latency;
    (c) compares original and cached inference; (d) includes the one-time cache
    setup in the amortized speedup.}
  ```

- 🔵 **tables/overall_results.tex:61–65 (Table~`tab:overall-performance`)** — boldface is applied to every Shared/Specific value without being defined in the caption, including cells that are not best in their comparison group (for example, TabICL-v2 clean AUC is 0.901 for Base/Shared but bold 0.900 for Specific; clean $R^2$ is 0.739 for Base but bold 0.738 for both variants). If readers interpret bold as the usual winner marker, the table is misleading. _Fix:_ either reserve bold for the true column-wise winner/ties within each backbone block, or state explicitly that boldface only identifies TabACE rows and does not denote the best value.
  ```latex
  % tables/overall_results.tex:61–65
  TabICL-v2 & 0.804 & 0.901 & 0.739 & 0.476 & 0.494 & 0.676 & -1.077 & 1.270 & 0.763 & 0.826 & 0.273 & 0.730 \\
  \rowcolor{tabaceSharedRow}
  \textcolor{tabaceSharedText}{\textbf{+ \method{}-shared}} & \textcolor{tabaceSharedText}{\textbf{0.804}} & \textcolor{tabaceSharedText}{\textbf{0.901}} & \textcolor{tabaceSharedText}{\textbf{0.738}} & \textcolor{tabaceSharedText}{\textbf{0.478}} & \textcolor{tabaceSharedText}{\textbf{0.575}} & \textcolor{tabaceSharedText}{\textbf{0.720}} & \textcolor{tabaceSharedText}{\textbf{0.363}} & \textcolor{tabaceSharedText}{\textbf{0.783}} & \textcolor{tabaceSharedText}{\textbf{0.772}} & \textcolor{tabaceSharedText}{\textbf{0.831}} & \textcolor{tabaceSharedText}{\textbf{0.467}} & \textcolor{tabaceSharedText}{\textbf{0.659}} \\
  \rowcolor{tabaceSpecificRow}
  \textcolor{tabaceSpecificText}{\textbf{+ \method{}-specific}} & \textcolor{tabaceSpecificText}{\textbf{0.804}} & \textcolor{tabaceSpecificText}{\textbf{0.900}} & \textcolor{tabaceSpecificText}{\textbf{0.738}} & \textcolor{tabaceSpecificText}{\textbf{0.476}} & \textcolor{tabaceSpecificText}{\textbf{0.610}} & \textcolor{tabaceSpecificText}{\textbf{0.752}} & \textcolor{tabaceSpecificText}{\textbf{0.485}} & \textcolor{tabaceSpecificText}{\textbf{0.708}} & \textcolor{tabaceSpecificText}{\textbf{0.805}} & \textcolor{tabaceSpecificText}{\textbf{0.854}} & \textcolor{tabaceSpecificText}{\textbf{0.655}} & \textcolor{tabaceSpecificText}{\textbf{0.535}} \\
  ```

- 🟠 **sections/appendix.tex:481–485 ↔ sections/appendix.tex:1399–1405, 1423–1428, 1445–1446 (Figures~`fig:context-count`, `fig:aggregation-ablation`, and `fig:k5-formula-diagnostic`)** — the metric protocol says nMAE/nRMSE are compared only within a track because perturbation and shift tracks use different denominators, but the ablation figures form equal-weight cross-track nMAE/nRMSE summaries. This is an explicit protocol contradiction, not a rounding issue. _Fix:_ either report cross-track $R^2$ ablations as prescribed, or define and justify a common normalization before pooling nMAE/nRMSE; otherwise revise the protocol sentence so it accurately scopes the exception.
  ```latex
  % sections/appendix.tex:481–485
  Both are population standard deviations (\texttt{ddof=0}).  Because the two
  tracks use different normalization scales, nMAE and nRMSE are compared only
  within a track; cross-track regression summaries use $R^2$.  In the shift
  track, $\operatorname{std}(y_{\mathrm{target}})$ is computed only after all
  predictions have been produced and is used solely for metric evaluation.
  ```
  ```latex
  % sections/appendix.tex:1399–1405
  \caption{Context-count ablation over four nonzero strengths. Each curve
  starts from Base and continues through $R=1,\ldots,5$; dashed horizontal
  lines mark the Base level. The five-perturbation and seven-shift tracks
  receive equal macro weight. With only a few contexts, shift coverage is
  limited. Performance improves as complementary contexts are added, and
  $R=5$ outperforms Base on all six metrics. Arrows indicate the preferred
  direction.}
  ```
  ```latex
  % sections/appendix.tex:1423–1428
  \begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/tabace_aggregation_ablation.pdf}
    \caption{Aggregation-rule ablation at $R=5$, equally weighting the
    five-perturbation and seven-shift tracks. Panels compare classification
    Accuracy and regression nRMSE.}
  ```
  ```latex
  % sections/appendix.tex:1445–1446
  includes $Q_i$. The four nonzero strengths are averaged within the
  five-perturbation and seven-shift tracks before giving the tracks equal weight.
  ```
