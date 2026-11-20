# TabICL five-perturbation regression recovery audit

## Scope and protocol

This audit covers the requested targeted reruns only: TabACE-shared for Shortcut at strengths 0.2, 0.5, and 0.8; TabACE-shared and TabACE-specific for Local at 0.2 and 0.5; and both variants for Mixed at 0.2. Hyperparameters and Context members were selected from training/support data or outer-training folds. The saved run audits state that outer-test labels were excluded.

Every promoted entry comes from an OOF prediction file. The promotion script replaces only the declared method-family-strength coordinates and checks that all other values remain identical to the accepted snapshot. Its manifest records source paths and SHA-256 hashes.

## Results

- Local satisfies the requested strict ordering on every raw metric at both strengths. At 0.2, R2 is 0.697065786, 0.697067453, and 0.697096951 for Base, TabACE-shared, and TabACE-specific; at 0.5 the corresponding values are 0.641396417, 0.641399735, and 0.641402016. nMAE and nRMSE follow the same favorable ordering in the lower-is-better direction.
- Shortcut TabACE-shared differs from Base at all three strengths and on all three metrics. Its changes are near-neutral: they do not worsen the displayed three-decimal results, but the raw metric directions are mixed.
- At Mixed 0.2, TabACE-specific improves R2 from 0.643301776 to 0.673485872, nMAE from 0.393394878 to 0.345850366, and nRMSE from 0.610899462 to 0.568983069. TabACE-shared is distinct from Base but slightly worse in raw precision and tied at three decimals.
- All 27 promoted method-metric cells differ from Base by more than 1e-12. No non-target cell changed.

## Reproducibility and rendering

Rerunning the promotion script locally reproduced both the 875-row detail table and the 25-row summary table with exact parsed equality and maximum numeric delta 0. Binary hashes differ between the Linux-generated and Windows-regenerated CSVs only because of platform line endings.

The updated ICLR manuscript compiled to 24 pages. Visual inspection of pages 7-9 and 20-23 found no clipping, overlap, broken coloring, or unreadable table/figure content. The main and appendix tables contain no bolded best values; TabACE-shared is blue and TabACE-specific is red.

## Verdict and limitation

The targeted recovery is conditionally acceptable. The source-level Local ordering is strict, and Mixed TabACE-specific yields a meaningful improvement. However, the Local gaps vanish at three-decimal display precision, while Shortcut TabACE-shared and Mixed TabACE-shared support only a stability or near-neutral claim. If a visibly separated Local result is required, it should be obtained through a new predeclared train-only rerun rather than selective precision changes or manual metric edits.
