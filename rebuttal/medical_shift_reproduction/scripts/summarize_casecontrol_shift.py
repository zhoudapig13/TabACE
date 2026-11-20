"""Aggregate the separately declared, exploratory case-control comparisons."""
import json
import pandas as pd
import summarize_public_shift as summary


if __name__ == '__main__':
    summary.DATA = summary.ROOT/'data/processed/public_casecontrol_v1'
    summary.OUT = summary.ROOT/'results/public_casecontrol_v1'
    summary.TASKS = ['sepsis_A_to_B', 'sepsis_B_to_A']
    summary.main()
    status = json.loads((summary.OUT/'status.json').read_text())
    status.update(exploratory_casecontrol=True, prevalence_fixed_at_half=True,
                  previously_evaluated_public_patients=True,
                  natural_cohort_results_preserved=True)
    summary.write_json(summary.OUT/'status.json', status)
    table = pd.read_csv(summary.OUT/'aggregate_fixed_threshold_diagnostic.csv')
    print('FIXED_0.5_DIAGNOSTIC_ONLY')
    print(table[['task', 'domain', 'method', 'accuracy']].to_string(index=False))
