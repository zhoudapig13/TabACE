"""Aggregate strict ABCD separately from constructor-availability diagnostics."""
import json
import numpy as np
import pandas as pd
import abcd_metric_aligned_v100 as study
import abcd_fresh_confirmation_v101 as fresh
import abcd_reference_repair_v102 as repair
from prepare_data import atomic_json

OUT=study.RUN/'results/abcd_comparison_v103'


def main():
    assert json.loads((fresh.OUT/'audit.json').read_text())['status']=='PASS'
    assert json.loads((repair.OUT/'audit.json').read_text())['status']=='PASS'
    assert json.loads((study.OUT/'execution.json').read_text())['status']=='COMPLETE'
    assert json.loads((study.OUT/'plan.json').read_text())['code_sha256']==study.origin.sha(study.__file__)
    assert json.loads((fresh.OUT/'plan.json').read_text())['code_sha256']==study.origin.sha(fresh.__file__)
    assert json.loads((repair.OUT/'plan.json').read_text())['code_sha256']==study.origin.sha(repair.__file__)
    for folder,sealed,base in [(study.OUT,'policy_seal.json',study.OUT),(fresh.OUT,'frozen_policies.json',study.OUT),(repair.OUT,'policy_seal.json',repair.OUT)]:
        hashes=json.loads((folder/sealed).read_text())
        assert all(study.origin.sha(base/path)==h for path,h in hashes.items())
    OUT.mkdir(exist_ok=True,parents=True)
    dev=pd.read_csv(study.OUT/'development_tables.csv');dev['cohort']='development'
    new=pd.read_csv(fresh.OUT/'tables.csv');new['cohort']='additional_queries'
    table=pd.concat([dev,new],ignore_index=True)
    table.to_csv(OUT/'strict_ABCD_tables.csv',index=False)
    first=pd.read_csv(study.OUT/'development_contrasts.csv');first['cohort']='development'
    second=pd.read_csv(fresh.OUT/'contrasts.csv');second['cohort']='additional_queries'
    alternative=pd.read_csv(repair.OUT/'contrasts.csv')
    alternative=alternative[alternative.cohort=='fresh'].copy();alternative['task']='regression'
    alternative['datasets']=22;alternative['cohort']='additional_queries'
    alternative['version']='constructor_availability_diagnostic'
    primary=second.copy();primary['version']='strict_ABCD'
    family=pd.concat([primary,alternative],ignore_index=True);assert len(family)==12
    prev=0.
    for rank,i in enumerate(sorted(family.index,key=lambda j:family.loc[j,'p_two_sided'])):
        prev=max(prev,min(1.,(12-rank)*family.loc[i,'p_two_sided']));family.loc[i,'p_holm_all12']=prev
    family['positive_significant_all12']=(family.gain_ci_low>0)&(family.p_holm_all12<.05)
    family.to_csv(OUT/'fresh_all_versions_contrasts.csv',index=False)
    pd.concat([first,second],ignore_index=True).to_csv(OUT/'strict_ABCD_contrasts.csv',index=False)
    # Recompute published aggregates directly from per-dataset detail rows.
    for name,source in [('development',study.OUT/'development_detail.csv'),('additional_queries',fresh.OUT/'detail.csv')]:
        detail=pd.read_csv(source)
        for task,g in detail.groupby('task'):
            key='accuracy' if task=='classification' else 'r2'
            v=g[g.rho>0].groupby(['data_id','method'])[key].mean()
            for method in g.method.unique():
                expected=float(v.xs(method,level='method').mean())
                row=table[(table.cohort==name)&(table.task==task)&(table.method==method)].iloc[0]
                assert np.isclose(expected,row[key],atol=1e-12)
                assert len(v.xs(method,level='method'))==row.datasets
    assert (study.RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    log={'purpose':'Rerun and test A-B,A-C,B-D,C-D; no ordering constraints in loss or selection',
        'setting':{'seed':0,'context_cap':256,'alternative_branches':5,'classification_target_labels':64,
            'regression_target_labels':0,'benchmark_and_source_inputs_unchanged':True,
            'classification_training':'CE with nested calibration Accuracy selection',
            'regression_training':'normalized MSE with other-dataset inner R2 selection'},
        'results':{'strict_tables':json.loads(table.to_json(orient='records')),
            'additional_query_contrasts':json.loads(family.to_json(orient='records'))},
        'analysis':{'diagnostic_constructor_deployment':pd.read_csv(repair.OUT/'constructor_deployment.csv').to_dict('records'),
            'availability_fallback_does_not_validate_a_matched_constructor':True,
            'all_trials_and_unfavorable_results_retained':True,
            'new_reference_repair_passed_synthetic_signal_preservation_check_but_not_actual_calibration_selection':True},
        'next_steps':'Use actual pairwise differences and limits in rebuttal. Any further method change needs calibration-only selection and separately disclosed confirmation scope; do not enforce method ranking on evaluated queries.',
        'audit':{'status':'PASS','policy_hashes_unchanged':True,'aggregate_metrics_recomputed':True,
            'freshness_scope':'rows absent from previous ABCD query cohorts only; not global project historical independence',
            'original_paper_replication':False,'Markdown_bytes':0}}
    atomic_json(OUT/'experiment_log.json',log)
    print(table[table.method.isin(['Base','A','B','C','D'])].to_string(index=False),flush=True)
    print(family.to_string(index=False),flush=True)


if __name__=='__main__':main()
