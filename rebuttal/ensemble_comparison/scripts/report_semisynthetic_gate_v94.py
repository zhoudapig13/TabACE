"""Audit the strict-control gate experiment and retain all exploratory versions."""
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd

import semisynthetic_gate_only_v94 as study
from report_abcd_v84 import paired_statistics
from run_tabicl import metrics
from prepare_data import atomic_json


def main():
    files=list((study.OUT/'folds/classification').glob('*/detail.csv')); assert len(files)==5
    all_new=pd.concat([pd.read_csv(p) for p in files],ignore_index=True); assert len(all_new)==3600
    methods=['Base','Single-matched','A','B','C','D']
    selected=all_new[all_new.method.isin(methods)].copy(); assert len(selected)==2160
    selected['version']='gate_selected'
    other=all_new[~all_new.method.isin(methods)].copy(); other['method']=other.method.str.split('_').str[0]
    controls=selected[selected.method.isin(['Base','Single-matched'])]
    other=pd.concat([other,controls],ignore_index=True);other['version']='gate_unselected'
    previous=pd.read_csv(study.source.OUT/'all_versions_detail.csv')
    data=pd.concat([previous,selected,other],ignore_index=True)
    rows=[]
    for (task,version),group in data.groupby(['task','version']):
        first,second=('auc','accuracy') if task=='classification' else ('r2','nrmse')
        scored=group[group.rho>0].groupby(['data_id','method'])[[first,second]].mean()
        clean=group[group.rho==0].set_index(['data_id','method'])[[first,second]]
        base=scored.xs('Base',level='method');clean_base=clean.xs('Base',level='method')
        for method in methods:
            v=scored.xs(method,level='method').loc[base.index];c=clean.xs(method,level='method').loc[base.index]
            rows.append({'task':task,'version':version,'method':method,'datasets':len(base),first:v[first].mean(),second:v[second].mean(),
                'clean_primary_change':(c[first]-clean_base[first]).mean(),'clean_secondary_change':(c[second]-clean_base[second]).mean(),
                **paired_statistics(v[first]-base[first])})
    table=pd.DataFrame(rows);tested=table.index[table.method.isin(['A','B','C','D'])].tolist();assert len(tested)==20
    last=0.
    for rank,i in enumerate(sorted(tested,key=lambda j:table.loc[j,'p_two_sided'])):
        last=max(last,min(1.,(20-rank)*table.loc[i,'p_two_sided']));table.loc[i,'p_holm']=last
    table['positive_significant']=(table.gain_ci_low>0)&(table.p_holm<.05)
    primary=table[table.version.isin(['gate_selected','concept_calibrated_adapted'])]
    all_new.to_csv(study.OUT/'all_gate_detail.csv',index=False)
    table.to_csv(study.OUT/'all_exploratory_versions_tables.csv',index=False)
    primary.to_csv(study.OUT/'primary_tables.csv',index=False)
    cells={matched:{(c['meta']['task_id'],c['meta']['rho']):c for c in study.source.load(matched,'density_adapted')
                    if c['meta']['track']=='five_semisynthetic'} for matched in (True,False)}
    for r in all_new.to_dict('records'):
        role=r['method'] if r['method'] in ['Base','Single-matched'] else r['method'].split('_')[0]
        c=cells[role not in ('C','D')][(r['task_id'],r['rho'])]
        folder=study.OUT/'folds/classification'/str(r['fold'])
        p=folder/'predictions'/str(r['task_id'])/f'{r["rho"]}_{r["method"]}.npz'
        with np.load(p) as z:prediction,y=z['prediction'],z['y']
        assert np.array_equal(y,c['y'])
        if role=='Base':expected=c['base']
        elif role=='Single-matched':expected=c['bank'][3]
        else:
            policy=json.loads((folder/f'{role}_policy.json').read_text())
            assert set(policy['training_data_ids']).isdisjoint(policy['excluded_data_ids'])
            selected_model=min(policy['candidates'],key=lambda t:t['inner_CE'])
            assert selected_model['gate']==policy['selected_gate']
            candidate=next(t for t in policy['candidates'] if t['gate']==r['gate'])
            expected=study.predict(c,candidate['model'])
            assert (r['method']==role)==(r['gate']==policy['selected_gate'])
        assert np.allclose(prediction,expected,atol=1e-12,rtol=1e-12),p
        for key,value in metrics(y,prediction,'classification',1.).items():assert np.isclose(value,r[key],atol=1e-11,rtol=1e-11),p
    plan=json.loads((study.OUT/'plan.json').read_text())
    assert plan['code_sha256']==study.source.original.sha(study.__file__)
    assert not plan['feature_partition_proposal_used']
    assert (study.source.original.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    main_tests=primary[primary.method.isin(['A','B','C','D'])]
    atomic_json(study.OUT/'audit.json',{'status':'PASS','all_candidate_metric_and_mapping_rows_recomputed':len(all_new),
        'same_v92_context_bank_and_Base':True,'feature_suppression_not_reintroduced_in_C_D':True,
        'Source_only_gate_selection_verified':True,'Holm_exploratory_family':20,
        'all_eight_primary_ABCD_positive_significant':bool(main_tests.positive_significant.all()),
        'upstream_input_audit':str(study.source.OUT/'audit.json'),'original_paper_replication':False,
        'independent_confirmation':False,'Markdown_bytes':0})
    print(primary.to_string(index=False),flush=True)


if __name__=='__main__':main()
