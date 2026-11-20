# Experiment 43: Context-budget inference efficiency

This experiment extends the synchronized component profiler from Experiment 11
to Context budgets `K=1,...,5`.  It uses the same three classification and
three regression datasets, seven distribution-shift families, strength 0.5,
and one-estimator TabICL inference path.

`Base` and every `K` are measured on the same shifted cell and repeat.  Shift
construction and one-time checkpoint loading are outside the timed region.
Context construction, `K` TabICL calls, aggregation, and CUDA synchronization
are inside it.  The shared-bank prefix is fixed across `K`; therefore each
successive point adds one Context without changing earlier members.

Run on a CUDA device:

```bash
python fair_budget_experiments/exp43_context_budget_efficiency/src/profile_context_budget.py
```

The script writes `timing_detail.csv`, `timing_summary.csv`, and
`metadata.json` under `results/`.
