"""Audit final risk-gate and fixed-composition results without hiding weak cases."""
import hashlib,json
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,write_json

DEST=OUT/'shared_error_rebuttal_summary'

def main():
    natural=[];budgets=[];freq=[]
    for b in ['tabicl','tabpfn']:
        for t in ['five','seven_reimplemented']:
            f=pd.read_csv(OUT/'joint_harm_frequency'/b/t/'summary.csv')
            freq.extend(f.to_dict('records'))
            for version in ['excess_risk_v10','composed_risk_v11']:
                root=OUT/version/b/t;d=pd.read_csv(root/'performance.csv')
                for scope,subset in [('clean',d[d.rho==0]),('nonclean',d[d.rho!=0])]:
                    means=subset.groupby(['method','data_id'])[['r2','nrmse']].mean().groupby('method').mean()
                    task=subset.groupby(['data_id','method']).r2.mean().unstack();old=means.loc['Original']
                    for method,row in means.iterrows():
                        delta=(task[method]-task.Original).to_numpy();rng=np.random.default_rng(0)
                        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                        natural.append(dict(version=version,backbone=b,track=t,scope=scope,method=method,r2=float(row.r2),nrmse=float(row.nrmse),
                            delta_r2=float(row.r2-old.r2),normalized_mse_reduction_percent=100*(row.r2-old.r2)/(1-old.r2),
                            dataset_bootstrap_ci_lower=float(np.quantile(boot,.025)),dataset_bootstrap_ci_upper=float(np.quantile(boot,.975))))
                r=pd.read_csv(root/'condition_budgets.csv')
                for cohort,g in r.groupby('cohort'):
                    sums=g.groupby('method')[['n_total','n_cohort','sum_normalized_loss','sum_excess_vs_Base','no_longer_materially_worse_than_Base']].sum();baseline=sums.loc['Original']
                    for method,row in sums.iterrows():
                        budgets.append(dict(version=version,backbone=b,track=t,cohort=cohort,method=method,n_cohort=int(row.n_cohort),
                            cohort_percent=100*row.n_cohort/row.n_total,
                            total_loss_reduction_percent=100*(1-row.sum_normalized_loss/max(baseline.sum_normalized_loss,1e-12)),
                            excess_fusion_loss_recovery_percent=100*(1-row.sum_excess_vs_Base/max(baseline.sum_excess_vs_Base,1e-12)),
                            no_longer_materially_worse_than_Base_percent=100*row.no_longer_materially_worse_than_Base/max(row.n_cohort,1)))
    pd.DataFrame(natural).to_csv(DEST/'final_natural_metrics.csv',index=False)
    pd.DataFrame(budgets).to_csv(DEST/'final_error_budgets.csv',index=False)
    pd.DataFrame(freq).to_csv(DEST/'final_incidence_all_thresholds.csv',index=False)
    original=json.loads((OUT/'frozen_plan.json').read_text());policies=[p for c in original['cases'] for p in c['policies']]
    for p in policies:assert hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()==p['sha256']
    for version,script in [('excess_risk_v10','excess_risk_gate.py'),('composed_risk_v11','composed_risk_calibration.py')]:
        plan=json.loads((OUT/version/'frozen_plan.json').read_text())
        assert hashlib.sha256((ROOT/'rebuttal/gate_sensitivity_replay/scripts'/script).read_bytes()).hexdigest()==plan['source_sha256']
    for version in ['gate_preserving_v8','excess_risk_v10']:
        for b in ['tabicl','tabpfn']:
            for t in ['five','seven_reimplemented']:
                for p in (OUT/version/b/t).glob('fold_*.json'):
                    h=json.loads(p.read_text());assert set(h['source_data_ids']).isdisjoint(h['evaluation_data_ids'])
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    write_json(DEST/'final_audit.json',dict(purpose='Support a scoped rebuttal with incidence, performance, and actual shared-error loss recovery',
        settings=dict(seed=0,original_prediction_banks_and_benchmarks_unchanged=True,new_backbone_calls=0,
            model_parameters_and_within_run_thresholds_fit_on_source_datasets_only=True),
        results='All final natural, incidence, and error-budget tables saved; all preceding failures retained',
        analysis='Fixed composition improves clean and nonclean mean R2 in all four cases. Large shared-error excess-loss recovery is supported for five corruptions only; seven-shift subgroup recovery is small or slightly negative. Overall R2 gain is not equated with recovery of all shared errors.',
        evaluation_status='Iterative reconstructed development evaluation, not a fresh independent test',
        definitions=dict(serious_joint='All 5 alternatives err in same direction by >3 evaluation-target SD',
            harmful_joint='Serious joint AND Original squared loss exceeds Base by >0.01 target variance',
            excess_loss_recovery='Net reduction of positive fusion loss relative to Base within the fixed serious-joint cohort'),
        attribution='No matched random-context intervention: context diversity is design motivation, not proven cause of rarity',
        next_steps='Freeze one candidate and validate on independent data before treating development gains as confirmatory',
        original_policy_hashes_verified=len(policies),markdown_bytes=0))
    print('FINAL_NONCLEAN');print(pd.DataFrame(natural).query('version=="composed_risk_v11" and scope=="nonclean"').to_string(index=False))
    print('FINAL_RECOVERY');print(pd.DataFrame(budgets).query('version=="composed_risk_v11" and cohort=="joint3"').to_string(index=False))

if __name__=='__main__':main()
