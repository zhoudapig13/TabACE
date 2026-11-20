# Cross-Media Consistency Audit

**Verdict: FAILED**

The displayed manuscript numbers are internally consistent. The final verdict remains `FAILED` because seven raw CSV files named by the synthetic-figure provenance record are absent from the current workspace, so those figures cannot be traced back to their stated raw sources here.

## Scope and sources of truth

- Scope: manuscript text, Table 1, the main OpenML strength figure, backbone-wise appendix tables and figures, natural-shift and conventional-predictor tables, ablations, and the final ICLR PDF.
- OpenML sources: the canonical `summary_oof.csv` files consumed by the table and figure generators.
- Seven-shift freeze: `robustness/Q2/q2_robustness_summary.json` and the two locked composite summaries.
- Visual identity: Base `#555555`, TabACE-shared `#56B4E9`, and TabACE-specific `#D55E00`.

## Audit result

- Table 1 exactly matches its current generator and canonical inputs. All 24 Shared-over-Base nonzero block comparisons and all 24 Specific-over-Shared comparisons pass at unrounded precision.
- The main strength figure contains 120 trace rows; its maximum absolute difference from the canonical summaries is `4.95e-11`. Specific is best in all 32 nonzero panel-strength comparisons.
- The appendix contains 2,520 backbone-strength trace rows. All 144 source aggregate checks pass, and all 54 aggregates shared with Table 1 agree exactly.
- The seven-shift lock contains 112 positive-strength method coordinates and zero failing metric cells under the recorded noninferiority criterion. For TabACE-shared regression, all 28 positive-strength coordinates are non-inferior on raw R2/nMAE/nRMSE, none is a full exact tie with Base, and none repeats the displayed Base R2/nMAE pair at three decimals. Four individual metric ties remain elsewhere in the full 112-coordinate lock, but no affected method coordinate duplicates every reported metric.
- Repeated manuscript claims agree with their displayed evidence: dataset counts, the 13-baseline count, natural-shift rankings, eight conventional-predictor comparisons, context-count trends, and the reported `5.7x`/`3.9x` inference-cost values.
- The classification class-count symbol was inconsistent across source sections (`K` versus `L`). The unused preliminaries source has been corrected to use `L`, matching Method and Appendix.
- The old red code is absent from manuscript sources and generated vector artifacts. Specific is orange throughout; Shared remains blue and Base remains gray.
- The ICLR PDF compiles to 31 pages with zero LaTeX errors, undefined references, or overfull boxes. Visual inspection of Table 1, the main strength figure, appendix strength tables, and additional-backbone figures found no clipping or overlap.

## Unresolved divergence

`paper/figures/tabace_synthetic_figures_audit.json` names seven files under `experiment41/results/confirmation/`, but that directory is absent. The affected consumers are the synthetic stability/response figures and the associated `67.0%` and `87.6%` claims. No disagreement is visible between the current manuscript locations, but the raw-source link cannot be independently rechecked until those files are restored or the provenance record is corrected.

Repair owner: synthetic-experiment artifact/provenance owner.

## Scientific-scope note

Individual seven-shift candidate runs exclude outer-test labels internally, but the fixed-policy grid search and pointwise run-level meta-selection used held-out benchmark metrics. The result is therefore a benchmark-tuned diagnostic composite rather than an unbiased evaluation of one fixed deployment policy. This note is retained in the internal audit and was not inserted into the manuscript.
