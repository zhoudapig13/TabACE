"""Source-prior weighted reference fusion; frozen v272 expert forecasts reused."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import shared_reference_calibration_v272 as parent
import prior_weighted_reference_fusion_v274 as fusion
from prepare_data import ROOT,RUN,atomic_json
from run_tabicl import metrics

OUT=RUN/'results/shared_prior_weighted_calibration_v275'


def fit_all():
    assert json.loads((parent.OUT/'independent_audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'REFERENCE_SUPERVISION_FUSION_ADAPTATION_NOT_ORIGINAL_SHARED',
        'seed':0,'extra_reference_labels':64,'new_TFM_inference':False,
        'only_change_from_v272':'classification CE weighted by source/reference class frequency ratio, capped10',
        'class_priors_from_final_evaluation_labels':False,'regression_unchanged':True,
        'fitting_uses_only_source_and_reference_labels':True,
        'context_cap':256,'script_sha256':parent.source.sha(__file__),
        'fusion_sha256':parent.source.sha(fusion.__file__),
        'expected_conditions':1327,'independent_confirmation':False,'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)
    count=0
    refs={(r['task'],r['task_id']):r for r in json.loads((parent.reference.OUT/'reference_manifest.json').read_text())}
    for c in json.loads((parent.source.OUT/'case_manifest.json').read_text()):
        if c['track']!='perturbation':continue
        task,tid=c['task'],c['task_id']
        with np.load(c['input_file']) as z:sy=z['y_source']
        with np.load(refs[(task,tid)]['reference_file']) as z:ry=z['y_reference']
        for src in sorted((parent.OUT/'predictions'/task/str(tid)).glob('*.npz')):
            meta=json.loads(src.with_suffix('.json').read_text())
            assert parent.source.sha(src)==meta['output_sha256']
            with np.load(src) as z:
                model=fusion.fit(z['reference_expert_predictions'],ry,sy,task,max(float(np.std(ry)),1e-12))
                pred=fusion.predict(z['query_expert_predictions'],model)
            dest=OUT/'predictions'/task/str(tid)/src.name
            parent.source.npwrite(dest,Shared_prior_weighted=pred)
            atomic_json(dest.with_suffix('.json'),{k:meta[k] for k in ('task','task_id','data_id','family','rho')}|
                {'parent_file':str(src),'parent_sha256':meta['output_sha256'],
                 'output_sha256':parent.source.sha(dest),'model':model,'final_query_labels_opened':False})
            count+=1
    assert count==1327
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','conditions':count,'all_models_frozen_before_report':True})
    print('SOURCE_PRIOR_CALIBRATION_COMPLETE',count,flush=True)


def report():
    assert json.loads((OUT/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((OUT/'plan.json').read_text())
    assert parent.source.sha(__file__)==plan['script_sha256']
    assert parent.source.sha(fusion.__file__)==plan['fusion_sha256']
    refs={(r['task'],r['task_id']):r for r in json.loads((parent.reference.OUT/'reference_manifest.json').read_text())}
    records=[];count=0
    for c in json.loads((parent.source.OUT/'case_manifest.json').read_text()):
        if c['track']!='perturbation':continue
        task,tid=c['task'],c['task_id']
        with np.load(c['input_file']) as z:sy=z['y_source']
        with np.load(refs[(task,tid)]['reference_file']) as z:ry=z['y_reference']
        assert parent.source.sha(c['truth_file'])==c['truth_sha256']
        with np.load(c['truth_file']) as z:truth=z['y_target']
        for path in sorted((OUT/'predictions'/task/str(tid)).glob('*.npz')):
            m=json.loads(path.with_suffix('.json').read_text())
            assert parent.source.sha(path)==m['output_sha256']
            assert parent.source.sha(m['parent_file'])==m['parent_sha256']
            assert not m['final_query_labels_opened']
            with np.load(m['parent_file']) as old,np.load(path) as z:
                model=fusion.fit(old['reference_expert_predictions'],ry,sy,task,max(float(np.std(ry)),1e-12))
                np.testing.assert_allclose(model['weights'],m['model']['weights'],rtol=0,atol=1e-10)
                replay=fusion.predict(old['query_expert_predictions'],model)
                np.testing.assert_allclose(replay,z['Shared_prior_weighted'],rtol=0,atol=1e-10)
                if task=='regression':np.testing.assert_allclose(replay,old['Shared_reference_calibrated'],rtol=0,atol=1e-10)
                predictions={'Base':old['Base'],'Base_plus_same_reference':old['Base_plus_same_reference'],
                    'Shared_unweighted_reference':old['Shared_reference_calibrated'],'Shared_prior_weighted':replay}
                for method,p in predictions.items():
                    records.append({'task':task,'data_id':c['data_id'],'family':m['family'],'rho':m['rho'],
                        'method':method,**metrics(truth,p,task,max(float(np.std(truth)),1e-12))})
            count+=1
    assert count==1327
    df=pd.DataFrame(records);df.to_csv(OUT/'condition_metrics.csv',index=False)
    keys,cols=['task','family','method'],['accuracy','auc','r2','nrmse']
    ds=df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table=ds.groupby(keys)[cols].mean().reset_index()
    table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(OUT/'family_averages.csv',index=False)
    gains=[]
    for (task,family),g in table[table.family!='clean'].groupby(['task','family']):
        metric='accuracy' if task=='classification' else 'r2'
        b=float(g.loc[g.method=='Base',metric].iloc[0]);s=float(g.loc[g.method=='Shared_prior_weighted',metric].iloc[0])
        gains.append({'task':task,'family':family,'Base':b,'Shared':s,'gain':s-b,'threshold':.05,'passed':s-b>=.05})
    atomic_json(OUT/'numerical_acceptance.json',{'family_results':gains,'all_six_pass':all(g['passed'] for g in gains)})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'independent_audit.json',{'status':'PASS','conditions':count,'local_models_refitted':count,
        'classification_prior_uses_source_labels_only':True,'regression_predictions_unchanged':True,
        'predictions_replayed':count,'parent_reference_crossfit_audit_passed':True,'Markdown_bytes':0})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fit','report']);a=p.parse_args()
    fit_all() if a.phase=='fit' else report()
