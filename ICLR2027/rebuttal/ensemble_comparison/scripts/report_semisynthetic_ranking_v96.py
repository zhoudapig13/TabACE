"""Recompute ranking calibration, validation selection, and exploratory tests."""
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch

import semisynthetic_ranking_calibration_v96 as study
import semisynthetic_gate_only_v94 as old_gate
from prepare_data import atomic_json
from run_tabicl import metrics
from report_abcd_v84 import paired_statistics


def combined_history(new):
    previous=pd.read_csv(study.source.OUT/'all_versions_detail.csv')
    gates=pd.read_csv(old_gate.OUT/'all_gate_detail.csv')
    names=['Base','Single-matched','A','B','C','D']
    chosen=gates[gates.method.isin(names)].copy();chosen['version']='gate_selected'
    controls=chosen[chosen.method.isin(['Base','Single-matched'])]
    alternative=gates[~gates.method.isin(names)].copy();alternative['method']=alternative.method.str.split('_').str[0]
    alternative=pd.concat([alternative,controls],ignore_index=True);alternative['version']='gate_unselected'
    return pd.concat([previous,chosen,alternative,new],ignore_index=True)


def main():
    torch.set_num_threads(1)
    paths=sorted((study.OUT/'folds/classification').glob('*/detail.csv'));assert len(paths)==5
    new=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);assert len(new)==2160
    data=combined_history(new)
    assert not data.duplicated(['task','version','data_id','rho','method']).any()
    rows=[]
    for (task,version),group in data.groupby(['task','version']):
        first,second=('auc','accuracy') if task=='classification' else ('r2','nrmse')
        g=group[group.rho>0].groupby(['data_id','method'])[[first,second]].mean()
        c=group[group.rho==0].set_index(['data_id','method'])[[first,second]]
        base=g.xs('Base',level='method');bc=c.xs('Base',level='method')
        for method in ['Base','Single-matched','A','B','C','D']:
            v=g.xs(method,level='method').loc[base.index];clean=c.xs(method,level='method').loc[base.index]
            rows.append({'task':task,'version':version,'method':method,'datasets':len(base),
                first:v[first].mean(),second:v[second].mean(),'clean_primary':clean[first].mean(),
                'clean_secondary':clean[second].mean(),'clean_primary_change':(clean[first]-bc[first]).mean(),
                'clean_secondary_change':(clean[second]-bc[second]).mean(),**paired_statistics(v[first]-base[first])})
    table=pd.DataFrame(rows);tests=table.index[table.method.isin(['A','B','C','D'])].tolist();assert len(tests)==24
    last=0.
    for rank,i in enumerate(sorted(tests,key=lambda j:table.loc[j,'p_two_sided'])):
        last=max(last,min(1.,(24-rank)*table.loc[i,'p_two_sided']));table.loc[i,'p_holm']=last
    table['positive_significant']=(table.gain_ci_low>0)&(table.p_holm<.05)
    primary=table[table.version.isin(['ranking_selected','concept_calibrated_adapted'])].copy()
    data.to_csv(study.OUT/'all_exploratory_detail.csv',index=False)
    table.to_csv(study.OUT/'all_exploratory_versions_tables.csv',index=False)
    primary.to_csv(study.OUT/'primary_tables.csv',index=False)
    cell_sets={matched:study.source.load(matched,'density_adapted') for matched in (True,False)}
    policies={};validation_audits=[]
    for path in (study.OUT/'folds/classification').glob('*/*_policy.json'):
        policy=json.loads(path.read_text());fold=int(path.parent.name);method=policy['method']
        cells=cell_sets[policy['matched']]
        train=[c for c in cells if c['fold']!=fold and (policy['seen'] or not study.source.original.exposed(c,'classification'))]
        train_ids={c['meta']['data_id'] for c in train}
        excluded={c['meta']['data_id'] for c in cells if c['fold']==fold}
        assert train_ids==set(policy['training_data_ids']) and excluded==set(policy['excluded_data_ids'])
        assert train_ids.isdisjoint(excluded)
        assert policy['seen'] or not any(study.source.original.exposed(c,'classification') for c in train)
        inner=policy['inner_fold'];validation=[c for c in train if c['fold']==inner]
        best=study.auc_score(validation,{'kind':'base'});selected=None
        assert abs(best-policy['trials'][0]['inner_auc'])<1e-12
        for trial in policy['trials'][1:]:
            value=study.auc_score(validation,trial['inner_model'])
            assert abs(value-trial['inner_auc'])<1e-12,(path,trial['regularization'])
            if value>best+1e-12:best=value;selected=trial['regularization']
        assert selected==policy['selected_regularization']
        assert abs(best-policy['selected_inner_auc'])<1e-12
        if policy['policy']['kind']!='base':
            tensors=study.pair_data(train);p=torch.as_tensor(policy['policy']['parameters'],dtype=torch.float64)
            positive=study.mapped_probability(p,tensors,'positive').clamp(1e-7,1-1e-7)
            negative=study.mapped_probability(p,tensors,'negative').clamp(1e-7,1-1e-7)
            penalty=(p[:5]-p[:5].mean()).square().mean()+(p[5:15]-1).square().mean()
            objective=(torch.nn.functional.softplus(-(torch.logit(positive)-torch.logit(negative)))*tensors['weights']).sum()
            objective+=policy['policy']['regularization']*penalty
            assert abs(float(objective)-policy['policy']['training_objective'])<1e-8,path
            assert len(tensors['weights'])==policy['policy']['sampled_pairs']
        policies[(fold,method)]=policy
        validation_audits.append({'fold':fold,'method':method,'validation_auc_recomputed':True,
            'full_training_ranking_objective_recomputed':policy['policy']['kind']!='base','identity_fallback':selected is None})
    assert len(policies)==20
    query_cells={matched:{(c['meta']['task_id'],c['meta']['rho']):c for c in cells
                           if c['meta']['track']=='five_semisynthetic'} for matched,cells in cell_sets.items()}
    for r in new.to_dict('records'):
        matched=r['method'] not in ('C','D');c=query_cells[matched][(r['task_id'],r['rho'])]
        folder=study.OUT/'folds/classification'/str(r['fold'])
        path=folder/'predictions'/str(r['task_id'])/f'{r["rho"]}_{r["method"]}.npz'
        with np.load(path) as z:prediction,y=z['prediction'],z['y']
        assert np.array_equal(y,c['y'])
        if r['method']=='Base':expected=c['base']
        elif r['method']=='Single-matched':expected=c['bank'][3]
        else:expected=study.predict(c,policies[(r['fold'],r['method'])]['policy'])
        assert np.allclose(prediction,expected,atol=1e-12,rtol=1e-12),path
        for key,value in metrics(y,prediction,'classification',1.).items():assert np.isclose(value,r[key],atol=1e-11,rtol=1e-11),path
    plan=json.loads((study.OUT/'plan.json').read_text())
    assert plan['code_sha256']==study.source.original.sha(study.__file__)
    assert (study.source.original.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    main_tests=primary[primary.method.isin(['A','B','C','D'])]
    atomic_json(study.OUT/'audit.json',{'status':'PASS','classifier_prediction_and_metric_rows_recomputed':len(new),
        'Source_validation_auc_and_hyperparameter_selections_recomputed':20,
        'full_training_pair_objectives_verified':validation_audits,
        'same_Base_and_context_bank_as_v92':True,'no_new_feature_suppression_in_C_D':True,
        'Holm_exploratory_family':24,'all_eight_primary_ABCD_positive_significant':bool(main_tests.positive_significant.all()),
        'original_paper_replication':False,'independent_confirmation':False,'Markdown_bytes':0})
    print(primary.to_string(index=False),flush=True)


if __name__=='__main__':main()
