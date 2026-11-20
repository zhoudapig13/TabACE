"""Consolidate natural failure incidence and all development iterations."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import ROOT,OUT,write_json

def average(d):
    return d.groupby(['method','data_id'])[['r2','nrmse']].mean().groupby('method').mean()

def main():
    dest=OUT/'development_comparison';dest.mkdir(exist_ok=True)
    natural=[];performance=[];paired=[];stress=[]
    for b in ['tabicl','tabpfn']:
        for t in ['five','seven_reimplemented']:
            nr=OUT/'natural_failure_audit'/b/t
            counts=pd.read_csv(nr/'condition_counts.csv');counts=counts[counts.rho!=0]
            n=int(counts.n_queries.sum())
            pure=pd.read_csv(nr/'performance.csv');pure=average(pure[pure.rho!=0])
            row=dict(backbone=b,track=t,datasets=counts.data_id.nunique(),n_query_conditions=n,
                single_extreme_percent=100*counts.single_extreme_branch_queries.sum()/n,
                any_extreme_percent=100*counts.any_extreme_branch_queries.sum()/n,
                changed_by_clipping_percent=100*counts.changed_after_clipping.sum()/n,
                common_wrong_3of5_1sd_percent=100*counts.common_wrong_3of5_1sd.sum()/n,
                common_wrong_3of5_3sd_percent=100*counts.common_wrong_3of5_3sd.sum()/n,
                common_wrong_5of5_3sd_percent=100*counts.common_wrong_5of5_3sd.sum()/n,
                original_r2=float(pure.loc['Original','r2']),clip3_r2=float(pure.loc['Clip3_no_refit','r2']),
                clip3_delta_r2=float(pure.loc['Clip3_no_refit','r2']-pure.loc['Original','r2']),
                trust3_delta_r2=float(pure.loc['Trust3_no_refit','r2']-pure.loc['Original','r2']))
            natural.append(row)
            paths=[('no_refit',nr/'performance.csv'),
                ('v2',OUT/'algorithm_repairs_v2'/b/t/'performance.csv'),
                ('v3',OUT/'algorithm_repairs_v3'/b/t/'performance.csv')]
            for iteration,path in paths:
                df=pd.read_csv(path)
                for scope,subset in [('clean',df[df.rho==0]),('nonclean',df[df.rho!=0])]:
                    a=average(subset)
                    for method,r in a.iterrows():
                        performance.append(dict(backbone=b,track=t,iteration=iteration,scope=scope,method=method,
                            datasets=subset.data_id.nunique(),r2=float(r.r2),nrmse=float(r.nrmse),
                            delta_r2=float(r.r2-a.loc['Original','r2']),delta_nrmse=float(r.nrmse-a.loc['Original','nrmse'])))
                    tasks=subset.groupby(['data_id','method']).r2.mean().unstack()
                    for method in tasks.columns:
                        if method=='Original':continue
                        delta=(tasks[method]-tasks.Original).to_numpy()
                        rng=np.random.default_rng(0)
                        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                        paired.append(dict(backbone=b,track=t,iteration=iteration,scope=scope,method=method,
                            delta_r2=float(delta.mean()),dataset_bootstrap_ci_lower=float(np.quantile(boot,.025)),
                            dataset_bootstrap_ci_upper=float(np.quantile(boot,.975)),
                            improved_datasets=int((delta>1e-10).sum()),worsened_datasets=int((delta<-1e-10).sum()),
                            unchanged_datasets=int((np.abs(delta)<=1e-10).sum()),bootstrap_seed=0,
                            interval_scope='Dataset-block uncertainty conditional on this development evaluation, not a new independent test'))
            stress.extend(pd.read_csv(OUT/'algorithm_repairs_v3'/b/t/'stress_pooled.csv').to_dict('records'))
    for name,records in [('natural_failure_summary',natural),('all_iteration_performance',performance),
                         ('paired_dataset_changes',paired),('stress_summary',stress)]:
        pd.DataFrame(records).to_csv(dest/f'{name}.csv',index=False)
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    policies=[p for case in plan['cases'] for p in case['policies']]
    assert all(hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()==p['sha256'] for p in policies)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    checks=[]
    for version in ['algorithm_repairs_v2','algorithm_repairs_v3']:
        vp=OUT/version;frozen=json.loads((vp/'frozen_plan.json').read_text())
        code='calibrated_regression_repairs.py' if version.endswith('v2') else 'bounded_gain_gate.py'
        assert hashlib.sha256((ROOT/'rebuttal/gate_sensitivity_replay/scripts'/code).read_bytes()).hexdigest()==frozen['source_sha256']
        for b in ['tabicl','tabpfn']:
            for t in ['five','seven_reimplemented']:
                for path in (vp/b/t).rglob('fold_*.json'):
                    v=json.loads(path.read_text());assert set(v['source_data_ids']).isdisjoint(v['evaluation_data_ids'])
                    checks.append(dict(path=str(path.relative_to(ROOT)),source_test_disjoint=True,optimizer_success=v['optimization']['success']))
    write_json(dest/'experiment_log.json',dict(purpose='Quantify natural incidence and performance effects, then test source-calibrated algorithm repairs',
        settings=dict(seed=0,backbone_calls=0,benchmark_unchanged=True,additional_target_supervision=False),
        results=dict(natural_failure_summary=natural,all_iterations_preserved=True,optimizer_checks=checks),
        analysis='Severe common errors and single extreme branches are infrequent at 3 source SD; common errors at 1 source SD are not rare. Clipping alone changes average R2 by less than 0.0007. New source-calibrated variants improve nonclean mean R2 numerically in all four cases, with clean and stress tradeoffs reported.',
        evaluation_status='Development evaluation: model architectures were iterated after inspecting prior aggregate evaluations. Held-out labels never enter parameter fitting; results are not an untouched final test or original-paper reproduction.',
        next_steps='Freeze a chosen variant and verify on separate datasets or a newly held-out evaluation before confirmatory claims',
        clipping_count_audit='Current frequency uses abs(branch-query_median)>3*source_SD. Earlier exact floating-point inequality k!=bounded could count numerical reconstruction roundoff.',
        original_policy_hashes_verified=len(policies),markdown_bytes=0))
    print('NATURAL_FREQUENCY');print(pd.DataFrame(natural).to_string(index=False))
    p=pd.DataFrame(performance)
    print('LATEST');print(p[(p.scope=='nonclean')&(((p.iteration=='v2')&(p.method=='C_calibrated_guard'))|((p.iteration=='v3')&(p.method=='A_bounded_gain')))].to_string(index=False))
    print('AUDIT_COMPLETE',len(checks),'heads')

if __name__=='__main__':main()
