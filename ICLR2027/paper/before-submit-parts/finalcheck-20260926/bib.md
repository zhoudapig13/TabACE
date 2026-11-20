# Final bibliography check — `main_iclr.tex`

_Checked: 2026-09-26 · Scope: cited references in the assembled ICLR 2027 review manuscript · Manuscript and bibliography were not modified._

## Result

No clear, submission-relevant bibliography problem was found.

## What I checked

- ✅ **Citation-key integrity** — the assembled `main_iclr.aux` contains 52 distinct cited keys; all 52 resolve to entries in `references.bib`. `main_iclr.blg` and `main_iclr.log` contain no missing-citation or undefined-reference warning.
- ✅ **Comments excluded** — citation usage was taken from the assembled document, so commented-out LaTeX did not count as a rendered citation.
- ✅ **Duplicate entries** — checked all 54 BibTeX records by exact key and normalized/fuzzy title; no duplicate paper was found.
- ✅ **Required metadata** — no cited entry is missing the required author/title/venue/year fields for its BibTeX type, and all years are numeric.
- ✅ **Existence and metadata** — `verify_refs.py` verified 49/54 records directly. Its five residual flags were rechecked live and were false positives: XGBoost is the ACM KDD 2016 paper; mixup is the ICLR 2018 paper; OpenML-CTR23 is the AutoML Conference 2023 workshop paper; and the SSA and Elliptic Loss papers are both ICLR 2025 papers. Their current BibTeX metadata matches the primary proceedings/OpenReview records.
- ✅ **Retraction/withdrawal** — live OpenAlex checks returned `is_retracted = false` for all 24 DOI-bearing records, and Crossref showed no retraction update. The recent arXiv records were also checked against their current primary pages; no withdrawal notice was found.
- ✅ **Published-version preference** — the cited arXiv-only records were checked for later archival versions. No clear archival replacement was found as of 2026-09-26. DENSE has an ICLR 2023 workshop presentation, so retaining the arXiv record is acceptable under the non-archival-workshop exception.
- ℹ️ **Non-rendering entries** — `huang2026lara` and `liu2025beta` are present in `references.bib` but are not cited by `main_iclr.tex`. They do not appear in the rendered bibliography and were not elevated as findings in this light final check.

## Sources used to adjudicate script flags

- ACM record for XGBoost: <https://doi.org/10.1145/2939672.2939785>
- OpenReview record for mixup: <https://openreview.net/forum?id=r1Ddp1-Rb>
- AutoML 2023 accepted-paper list for OpenML-CTR23: <https://2023.automl.cc/program/accepted_papers/>
- ICLR 2025 proceedings for SSA: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/c19c4def293b6a63db1ff27143dd4f10-Abstract-Conference.html>
- ICLR 2025 proceedings for Elliptic Loss Regularization: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/bf3ee5a5422b0e2a88b0c9c6ed3b6144-Abstract-Conference.html>
