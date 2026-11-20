# Tables

Store generated paper-facing `*.tex` fragments here. Avoid hand-editing result
values: generation scripts should control rounding, bolding, denominators, and
confidence intervals.

Regenerate the overall performance table with

```bash
python paper/scripts/generate_overall_results_table.py
```

The TFM and TabACE rows are read from these designated reports:

- `PaperResults/TABACE_FINAL_RESULTS_ZH.md`
- `PaperResults/TABACE_TABPFN_V3_FINAL_RESULTS_ZH.md`
- `PaperResults/TABACE_TABDPT_V1_2_FINAL_RESULTS_ZH.md`

The non-TFM rows remain sourced from the audited baseline CSV summary.
