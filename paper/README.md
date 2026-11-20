# TabACE paper

This directory is the working manuscript for:

> **TabACE: Adaptive Ensembling of Configurable Contexts for Reliable Prediction with Tabular Foundation Models**

The method name is **TabACE (Tabular Adaptive Context Ensemble)**. In prose,
use "TabACE" as an uncountable method name (for example, "we apply TabACE") and
use "context ensemble" for the general mechanism. The context configuration
set is a configurable instantiation rather than a fixed part of the algorithm.
Legacy experiment fields
named `context` and older reports displaying `CoCE` should remain unchanged for
reproducibility; all new paper-facing text and tables should display `TabACE`.

## Manuscript formats

The source has two entry points that share `macros.tex`, `sections/`, and
`references.bib`:

- `main.tex`: complete, single-column arXiv preprint using the standard
  `article` class and packages available in TeX Live.
- `main_iclr.tex`: anonymous ICLR 2027 review manuscript using the
  vendored official conference style.

arXiv does not require a particular document class. The preprint deliberately
does not depend on an unofficial `arxiv.sty` package.

## Build

Build the arXiv preprint (the default target):

```bash
make
```

Build the ICLR review version:

```bash
make iclr
```

Both commands use `latexmk`. If it is unavailable, run `pdflatex`, `bibtex`,
and `pdflatex` twice for the chosen entry point.

## Preparing the arXiv upload

Before packaging, replace `Anonymous Authors` in `main.tex` with the public
author list and affiliations, and resolve every `\TODO` in the manuscript.
Then run:

```bash
make arxiv-package
```

This compiles the paper and writes `dist/tabace-arxiv-source.zip`. The archive
contains the TeX sources, bibliography source, generated `main.bbl`, and no
auxiliary build files or compiled PDF. After uploading it, select PDFLaTeX and
carefully inspect arXiv's generated preview before completing the submission.

If figures are added later, add only the figures referenced by the manuscript
to the `arxiv-package` recipe; do not upload unused plots or exploratory files.

## Submission mode

`main_iclr.tex` is anonymous by default. Keep `\iclrfinalcopy` commented out
for review. For a camera-ready version, uncomment it and replace the anonymous
author block with the final author information.

## Writing map

- `main.tex`: arXiv metadata, package setup, and complete section order.
- `main_iclr.tex`: ICLR package setup and complete section order.
- `macros.tex`: notation shared by both formats.
- `sections/00_abstract.tex`: abstract shared by both formats.
- `sections/`: one source file per paper section.
- `references.bib`: manuscript bibliography.
- `figures/`: final paper figures (vector PDF preferred for plots).
- `tables/`: compact, paper-facing tables generated from audited results.
- `notes/CLAIM_CHECKLIST.md`: claims that must remain qualified.

Do not manually copy numbers from exploratory reports. Final tables should be
generated from the audited Experiment 11 outputs, and every headline number
should be checked against its source CSV before submission.
