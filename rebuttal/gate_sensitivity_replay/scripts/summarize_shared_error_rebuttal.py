"""Keep incidence, overall gains, and conditional loss recovery separate."""
import hashlib,json
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,write_json

DEST=OUT/'shared_error_rebuttal_summary'
VERSIONS=['common_mode_v7','gate_preserving_v8','reference_range_v9']

def main():
    DEST.mkdir(exist_ok=True)
    natural=[];recovery=[];frequency=[];stress=[]
    for b in ['tabicl','tabpfn']:
        for t in ['five','seven_reimplemented']:
            f=pd.read_csv(OUT/'joint_harm_frequency'/b/t/'summary.csv');frequency.extend(f.to_dict('records'))
            for version in VERSIONS:
                root=OUT/version/b/t;df=pd.read_csv(root/'performance.csv')
                for scope,d in [('clean',df[df.rho==0]),('nonclean',df[df.rho!=0])]:
                    means=d.groupby(['method','data_id'])[['r2','nrmse']].mean().groupby('method').mean()
                    baseline=means.loc['Original']
                    task=d.groupby(['data_id','method']).r2.mean().unstack()
                    for method,row in means.iterrows():
                        delta=(task[method]-task.Original).to_numpy();rng=np.random.default_rng(0)
                        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                        natural.append(dict(version=version,backbone=b,track=t,scope=scope,method=method,
                            r2=float(row.r2),nrmse=float(row.nrmse),delta_r2=float(row.r2-baseline.r2),
                            normalized_mse_reduction_percent=100*(row.r2-baseline.r2)/(1-baseline.r2),
                            dataset_bootstrap_ci_lower=float(np.quantile(boot,.025)),dataset_bootstrap_ci_upper=float(np.quantile(boot,.975))))
            for version,path in [('gate_preserving_v8',OUT/'gated_error_budget'/b/t/'condition_budgets.csv'),
                                 ('reference_range_v9',OUT/'reference_range_v9'/b/t/'condition_budgets.csv')]:
                d=pd.read_csv(path)
                for cohort,g in d.groupby('cohort'):
                    sums=g.groupby('method')[['n_total','n_cohort','sum_normalized_loss','sum_excess_vs_Base','no_longer_materially_worse_than_Base']].sum()
                    original=sums.loc['Original']
                    for method,row in sums.iterrows():
                        recovery.append(dict(version=version,backbone=b,track=t,cohort=cohort,method=method,
                            n_cohort=int(row.n_cohort),cohort_percent=100*row.n_cohort/row.n_total,
                            total_loss_reduction_percent=100*(1-row.sum_normalized_loss/max(original.sum_normalized_loss,1e-12)),
                            excess_fusion_loss_recovery_percent=100*(1-row.sum_excess_vs_Base/max(original.sum_excess_vs_Base,1e-12)),
                            no_longer_materially_worse_than_Base_percent=100*row.no_longer_materially_worse_than_Base/max(row.n_cohort,1)))
            for version in ['gate_preserving_v8','reference_range_v9']:
                d=pd.read_csv(OUT/version/b/t/'joint_error_records.csv')
                for (magnitude,threshold),g in d.groupby(['magnitude_source_sd','threshold_evaluation_sd']):
                    result=[]
                    for method,h in g.groupby('method'):
                        tasks=h.groupby('data_id')[['joint_sum_normalized_se','n_joint']].sum();valid=tasks.n_joint>0
                        mse=float((tasks.loc[valid,'joint_sum_normalized_se']/tasks.loc[valid,'n_joint']).mean())
                        result.append(dict(method=method,conditional_normalized_mse=mse))
                    baseline=next(x for x in result if x['method']=='Original')['conditional_normalized_mse']
                    for row in result:
                        stress.append(dict(version=version,backbone=b,track=t,magnitude_source_sd=magnitude,
                            threshold_evaluation_sd=threshold,**row,mse_reduction_percent=100*(1-row['conditional_normalized_mse']/max(baseline,1e-12))))
    for name,records in [('natural_metrics',natural),('natural_error_recovery',recovery),('original_joint_incidence',frequency),('controlled_joint_error',stress)]:
        pd.DataFrame(records).to_csv(DEST/f'{name}.csv',index=False)
    scripts=['common_mode_residual.py','gate_preserving_residual.py','reference_range_residual.py']
    for version,script in zip(VERSIONS,scripts):
        plan=json.loads((OUT/version/'frozen_plan.json').read_text())
        assert hashlib.sha256((ROOT/'rebuttal/gate_sensitivity_replay/scripts'/script).read_bytes()).hexdigest()==plan['source_sha256']
        for b in ['tabicl','tabpfn']:
            for t in ['five','seven_reimplemented']:
                for path in (OUT/version/b/t).glob('fold_*.json'):
                    h=json.loads(path.read_text());assert set(h['source_data_ids']).isdisjoint(h['evaluation_data_ids'])
    parent=json.loads((OUT/'frozen_plan.json').read_text());policies=[p for c in parent['cases'] for p in c['policies']]
    for p in policies:assert hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()==p['sha256']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    write_json(DEST/'experiment_log.json',dict(purpose='Quantify rare serious joint harm and optimize calibration with explicit error-budget denominators',
        settings=dict(seed=0,all_original_benchmarks_and_prediction_banks_preserved=True,new_backbone_calls=0),
        results='See original_joint_incidence.csv, natural_metrics.csv, natural_error_recovery.csv, controlled_joint_error.csv',
        analysis='No claim that all shared regression errors are rare. Serious shared error plus material fusion harm is a restricted event. No causal attribution to context curation without matched random-context controls. Overall R2 gain, subgroup error reduction, and recoverable excess fusion harm are distinct quantities.',
        next_steps='Freeze the selected development variant and confirm on independent data; do not combine incidence from natural data with recovery from injected-bias stress as if they were one cohort',
        evaluation_status='Iterative reconstructed development evaluation; no held-out labels enter model fitting or within-run parameter selection',
        source_test_separation_verified=True,original_policy_hashes_verified=len(policies),markdown_bytes=0))
    n=pd.DataFrame(natural);r=pd.DataFrame(recovery)
    print('LATEST_NATURAL');print(n[(n.version=='reference_range_v9')].to_string(index=False))
    print('LATEST_RECOVERY');print(r[(r.version=='reference_range_v9')&r.cohort.isin(['joint3','harmful_joint3','all'])].to_string(index=False))

if __name__=='__main__':main()
