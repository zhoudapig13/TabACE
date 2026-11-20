# TabACE

## Robust Inference in Tabular Foundation Models via Adaptive Context Ensembling

TabACE is an inference framework for improving the robustness of tabular foundation models under context corruption and distribution shift. It constructs complementary labeled contexts, runs the same frozen backbone on these contexts, and combines their predictions using calibrated reliability, confidence, and cross-context agreement. A query-dependent gate controls how far the final prediction departs from the default Base prediction.

**Manuscript:** [TabACE — ICLR manuscript](ICLR2027/paper/main_iclr.pdf)  
**Working project:** [ICLR2027/](ICLR2027/)  
**Complete backup:** [GitHub Releases](https://github.com/zhoudapig13/TabACE/releases)

## Method

For a query, TabACE produces a Base prediction and an ensemble of alternative-context predictions. Its final output has the form

$$o_{\mathrm{TabACE}}(x)=(1-\alpha(x))o_0(x)+\alpha(x)\sum_{r=1}^{R}w_r(x)o_r(x),\qquad \sum_r w_r(x)=1.$$

The branch weights combine learned reliability with query-level evidence. The gate uses Base uncertainty and support from the alternative contexts to determine the update strength. Classification uses class probabilities; regression uses scalar predictions.

- **Shared:** a reusable context bank for settings without a known corruption or shift family.
- **Specific:** context configurations that incorporate available problem knowledge.
- **Backbones:** the project includes experiments with TabICL, TabPFN, TabDPT, and LimiX, together with conventional predictor comparisons.
- **Evaluation:** clean data, five context-corruption families, seven shift families, synthetic tasks, and natural shifts.

## Project layout

```text
TabACE/
├── README.md
├── LICENSE
├── ICLR2027/                 # Original project-relative directory layout
│   ├── src/tabicl/           # Vendored TabICL backbone
│   ├── experiments/
│   ├── fair_budget_experiments/
│   ├── rebuttal/
│   ├── scripts/
│   └── paper/                # Manuscript, figures, tables, and PDF
└── backup/                   # Backup manifest and restoration instructions
```

## Installation and experiments

Use Python 3.10 or newer:

```bash
pip install -e ./ICLR2027
cd ICLR2027
```

Experiment dependencies, configurations, and commands are documented in the corresponding experiment directories. The vendored backbone distribution remains named `tabicl`, preserving the existing imports. Dataset and checkpoint paths in historical experiment scripts may need to be adjusted for a new machine.

To compile the paper:

```bash
cd ICLR2027/paper
make iclr
```

## Full project backup

Git tracks the source, documents, and lightweight reports. Large files and the complete original working-directory snapshot are preserved as split release assets when uploaded. See [backup/README.md](backup/README.md) for the manifest, checksums, and restore commands. The archive, rather than the lightweight Git checkout, is the restoration source for excluded datasets, checkpoints, and historical outputs.

## Attribution

This project includes [TabICL](https://github.com/soda-inria/tabicl) code. Its license and upstream attribution are retained. The original project README remains at [ICLR2027/README.md](ICLR2027/README.md).
