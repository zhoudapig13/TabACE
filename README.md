# TabACE

**TabACE: Robust Inference in Tabular Foundation Models via Adaptive Context Ensembling**

TabACE constructs alternative labeled contexts for a frozen tabular foundation model and combines their predictions using calibrated reliability, confidence, agreement, and a query-dependent Base gate.

## Repository contents

- `src/tabicl/`: vendored TabICL backbone implementation.
- `experiments/`, `experiment*/`, `scripts/`: experiment implementations and utilities.
- `fair_budget_experiments/`, `robustness/`, `rebuttal/`: comparison and robustness experiment sources.
- `paper/`: manuscript source, tables, figures, and compiled ICLR manuscript.
- `REBUTTAL_*.md`: author-response drafts.

## Installation

Use Python 3.10 or newer and install the local backbone package:

```bash
pip install -e .
```

Experiment-specific dependencies and invocation details are documented in the corresponding directories. The Python distribution remains named `tabicl` to preserve existing imports.

## Manuscript

The review manuscript is available at [paper/main_iclr.pdf](paper/main_iclr.pdf). To compile its LaTeX source:

```bash
cd paper
make iclr
```

## Upstream attribution

This project includes code from [TabICL](https://github.com/soda-inria/tabicl). Its license and original notices are retained; the upstream README is preserved at [docs/TABICL_UPSTREAM_README.md](docs/TABICL_UPSTREAM_README.md).

## Snapshot scope

This is a source-and-documentation export of the ICLR2027 working directory. Raw datasets, per-example predictions, model checkpoints, caches, temporary files, and the original Git history are excluded. Paths to local datasets or checkpoints in experiment scripts may need to be configured on a new machine.
