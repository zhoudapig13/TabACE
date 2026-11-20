# Before-Submit Report — `main_iclr.tex`

_Generated: 2026-09-26 · Venue: ICLR 2027 conference track (live author guidelines and AI policy, checked 2026-09-26) · Version: review_

## Summary

- 🔴 Desk-reject risks: 0
- 🟠 Reviewers will frown: 2
- ⚪ Optional polish: 1

The manuscript is structurally submission-ready: the review style is active, the authors are anonymous, the main text ends on page 9, the required AI Use Statement is present, references are followed by the appendix, and the current PDF compiles without undefined citations or cross-references. The points below are deliberately limited to clear residual issues.

## 🔴 Desk-reject risk

### None confirmed

- No definite desk-reject issue was found in the assembled review manuscript.

## 🟠 Reviewers will frown

### AI-use disclosure specificity

- **sections/08_conclusion.tex:31** — The required section exists, but “other supporting tasks” is too broad to establish whether any of ICLR 2027's specifically required disclosure categories were involved. This is a compliance ambiguity, not a request for a longer statement. _Fix:_ make the paper statement and submission-form answers match; name any required-use categories that were actually involved. If none were involved beyond writing/polishing/drafting, remove the vague catch-all rather than implying undisclosed uses.
  ```latex
The authors used generative AI tools to aid and polish writing, draft portions
of the manuscript, and assist with other supporting tasks. All AI-assisted
material was reviewed and verified by the authors, who
take responsibility for the final manuscript.
  ```

### Notation consistency

- **sections/04_method.tex:6 ↔ sections/context_construction_details.tex:3** — The main method denotes the available target-query set by `\mathcal X_t`, while the construction appendix independently switches to `X_t` for the same available target features without relating the symbols. The distinction may be intended as set versus feature matrix, but it is currently unstated. _Fix:_ minimally say that `X_t` is the feature-matrix representation of `\mathcal X_t`, or use one symbol consistently.
  ```latex
% sections/04_method.tex:6
Let $\mathcal D_s=\{(x_j^s,y_j^s)\}_{j=1}^{n_s}$ denote labeled reference
data and $\mathcal X_t=\{x_i^t\}_{i=1}^{n_t}$ available unlabeled target queries, with
$x_j^s,x_i^t\in\mathbb R^d$.

% sections/context_construction_details.tex:3
Let $(X_c,y_c)$ denote the labeled context available in an outer fold and let
$X_t$ denote the unlabeled target features available to a test-time call.
  ```

## ⚪ Optional polish

### Experimental framing consistency

- **sections/00_abstract_iclr.tex:11 ↔ sections/01_introduction.tex:35** — The abstract and introductory summary still list controlled perturbations/distribution shifts alongside synthetic/natural shifts as parallel categories. Elsewhere, the paper correctly treats the former as evaluation conditions and the latter as supporting data/evaluation sources. This is not a factual contradiction, but the leftover phrasing blurs the experimental organization. _Fix:_ preserve the numbers and claims while using the already adopted relation, e.g. “OpenML benchmarks spanning five controlled context perturbations and seven distribution shifts, complemented by fully synthetic and natural-shift evaluations.”
  ```latex
% sections/00_abstract_iclr.tex:11
harmful updates. Across OpenML benchmarks spanning five context perturbations
and seven distribution shifts, fully synthetic settings, and natural shifts,
\method{} consistently mitigates performance degradation across foundation and
conventional predictors while preserving clean performance.

% sections/01_introduction.tex:35
supports reusable context computations to substantially reduce latency. Across
four frozen TFMs on OpenML benchmarks spanning five controlled context
perturbations and seven distribution shifts, as well as synthetic and natural
shifts, \method{} consistently improves prediction robustness while preserving
clean performance.
  ```

## What I checked / skipped

- ✅ Assembled input chain — checked `main_iclr.tex` and all 14 reachable `\input` files; orphaned drafts and Chinese mirrors were excluded.
- ✅ Compile — `latexmk` completed; BibTeX reported 52 used entries and zero warnings.
- ✅ Cross-references and citations — no undefined or multiply defined references/citations in the compile log.
- ✅ Page count and order — PDF has 35 letter-size pages; conclusion ends on page 9, reproducibility/AI statements and references begin on page 10, and the appendix starts after the bibliography on page 15.
- ✅ Anonymization — `Anonymous Authors`, review mode active, `\iclrfinalcopy` commented, PDF author metadata empty, and no identity-bearing URL/email/name found in active sources or comments.
- ✅ Mandatory/style checks — official `iclr2027_conference` style, required AI Use Statement present, optional Reproducibility Statement present, references excluded from main count, appendix after references.
- ✅ Mechanical LaTeX checks — captions are placed correctly; no active TODO/FIXME/AI-chat residue; no confirmed bad labels, broken references, unescaped text ampersands, or page-breaking overfull boxes.
- ✅ Main-text prose and terminology — read for concrete grammar and terminology conflicts; no other clear error worth changing at this late stage.
- ⏭️ Future-work rewriting — intentionally skipped because the author requested retaining the current paragraph.
- ⏭️ Reference existence/metadata and numerical faithfulness — outside this fragment's LaTeX/compliance scope.

## Venue rules used

- Source: [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) and [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors), checked live on 2026-09-26.
- Page limit: review main text ≤9 pages; references excluded; unlimited appendix after references.
- Review requirements: double blind; author identity in main or supplementary material is desk-rejectable.
- Statements: AI Use Statement required and excluded from the page limit; Reproducibility Statement recommended and excluded from the page limit.

