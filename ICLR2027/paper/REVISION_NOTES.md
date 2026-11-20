# ICLR 2027 discussion revision

Official source: [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), checked 2026-11-12.

- Discussion revisions are allowed until the public discussion closes on November 18, 2026.
- The revised main text may contain up to 10 pages; references and appendices are outside that limit.
- OpenReview applies a PDF difference comparison to the original submission.
- The official instructions ask authors to communicate changes clearly; they do not prescribe manual change colors. Blue is an editorial aid used here, not a mandatory ICLR convention.
- The current work changes the local manuscript only and does not upload a revision.

## Files and provenance

- `sections/rebuttal_additions.tex`: new Appendix E.
- `tables/revision/`: the 16 new numeric tables.
- `scripts/generate_rebuttal_appendix.py`: reproducible table generator.
- `revision_data/rebuttal_tables.json`: the author-provided numeric snapshot with per-table source notes.
- Existing main results are retained. Supplementary adaptations and task reconstruction are described separately.
- The larger-context regression recovery protocol keeps the original corrupted core and adds clean training rows.
- Folktables repeated-run variability is kept separate from the different supplied comparison point estimates.
- Clinical Accuracy is labelled as a 1:1 case-control component evaluation.

The compiled location index will list the final appendix sections, table numbers, and PDF pages.
