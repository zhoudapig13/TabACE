"""Frozen postprocessing hypotheses, with equally processed Base controls."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import shared_reference_calibration_v272 as parent
import posterior_and_ridge_calibration_v276 as method
from prepare_data import ROOT,RUN,atomic_json
from run_tabicl import metrics

OUT=RUN/'results/shared_posterior_ridge_v277'


def fit_all():
    assert json.loads((parent.OUT/'independent_audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'EXPLICIT_EXTRA_REFERENCE_AND_POSTPROCESSING_ADAPTATION',
        'seed':0,'context_cap':256,'extra_clean_labeled_reference_rows':64,
        'classification':'v269 Shared probability marginal aligned to observed source class proportions',
        'classification_assumption':'source and target class priors stable; unlabelled query batch used transductively',
        'regression':'ridge on four v272 honest reference-OOF experts, penalty.1, intercept penalty.01, anchor(.5,.5,0,0)',
        'same_postprocessing_Base_controls_included':True,'new_TFM_calls':0,
        'Base_and_benchmark_unchanged':True,'final_query_labels_used_for_fit_or_selection':False,
        'method_receives_family_or_severity':False,'previously_scored_development_cohort':True,
        'original_frozen_Shared_generalization_claimed':False,'expected_conditions':1327,
        'script_sha256':parent.source.sha(__file__),'method_sha256':parent.source.sha(method.__file__),
        'parent_manifest_sha256':parent.source.sha(parent.reference.OUT/'reference_manifest.json'),
        'threshold_each_family':{'accuracy_gain':.05,'r2_gain':.05},'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)
    refs={(r['task'],r['task_id']):r for r in json.loads((parent.reference.OUT/'reference_manifest.json').read_text())}
    count=0
    for c in json.loads((parent.source.OUT/'case_manifest.json').read_text()):
        if c['track']!='perturbation':continue
        task,tid=c['task'],c['task_id']
        assert parent.source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:sy=z['y_source']
        r=refs[(task,tid)]
        assert parent.source.sha(r['reference_file'])==r['reference_sha256']
        with np.load(r['reference_file']) as z:ry=z['y_reference']
        for src in sorted((parent.OUT/'predictions'/task/str(tid)).glob('*.npz')):
            m=json.loads(src.with_suffix('.json').read_text())
            assert parent.source.sha(src)==m['output_sha256']
            mixed=parent.parent.OUT/'predictions'/task/str(tid)/src.name
            mm=json.loads(mixed.with_suffix('.json').read_text())
            assert parent.source.sha(mixed)==mm['output_sha256']
            with np.load(src) as z,np.load(mixed) as previous:
                base=z['Base'];control=z['Base_plus_same_reference']
                if task=='classification':
                    shared,sm=method.align_probability_marginal(previous['Shared_extra_reference'],sy)
                    cb,bm=method.align_probability_marginal(base,sy)
                    cc,cm=method.align_probability_marginal(control,sy)
                else:
                    P=z['reference_expert_predictions'];Q=z['query_expert_predictions']
                    sm=method.fit_ridge(P,ry,sy,[.5,.5,0.,0.])
                    shared=method.predict_ridge(Q,sm)
                    bm=method.fit_ridge(P[:1],ry,sy,[1.])
                    cb=method.predict_ridge(Q[:1],bm)
                    cm=method.fit_ridge(P[2:3],ry,sy,[1.])
                    cc=method.predict_ridge(Q[2:3],cm)
            path=OUT/'predictions'/task/str(tid)/src.name
            parent.source.npwrite(path,Base=base,Base_postprocessed=cb,
                Base_same_reference_postprocessed=cc,Shared_adapted=shared)
            atomic_json(path.with_suffix('.json'),{k:m[k] for k in ('task','task_id','data_id','family','rho')}|
                {'parent_file':str(src),'parent_sha256':m['output_sha256'],
                 'context_parent_file':str(mixed),'context_parent_sha256':mm['output_sha256'],
                 'original_input_sha256':c['input_sha256'],'reference_sha256':r['reference_sha256'],
                 'models':{'Shared':sm,'Base_postprocessed':bm,'Base_same_reference_postprocessed':cm},
                 'query_labels_opened':False,'output_sha256':parent.source.sha(path)})
            count+=1
    assert count==1327
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','conditions':count,'predictions_frozen_before_scoring':True})
    print('POSTERIOR_RIDGE_FIT_COMPLETE',count,flush=True)


def report():
    assert json.loads((OUT/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((OUT/'plan.json').read_text())
    assert parent.source.sha(__file__)==plan['script_sha256']
    assert parent.source.sha(method.__file__)==plan['method_sha256']
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
            assert parent.source.sha(m['context_parent_file'])==m['context_parent_sha256']
            assert not m['query_labels_opened']
            with np.load(path) as z,np.load(m['parent_file']) as old,np.load(m['context_parent_file']) as ctx:
                np.testing.assert_array_equal(z['Base'],old['Base'])
                if task=='classification':
                    replay,rm=method.align_probability_marginal(ctx['Shared_extra_reference'],sy)
                    cb,_=method.align_probability_marginal(old['Base'],sy)
                    cc,_=method.align_probability_marginal(old['Base_plus_same_reference'],sy)
                    assert rm['maximum_marginal_error']<1e-4
                else:
                    P,Q=old['reference_expert_predictions'],old['query_expert_predictions']
                    model=method.fit_ridge(P,ry,sy,[.5,.5,0.,0.])
                    replay=method.predict_ridge(Q,model)
                    cb=method.predict_ridge(Q[:1],method.fit_ridge(P[:1],ry,sy,[1.]))
                    cc=method.predict_ridge(Q[2:3],method.fit_ridge(P[2:3],ry,sy,[1.]))
                np.testing.assert_allclose(z['Shared_adapted'],replay,atol=1e-10,rtol=0)
                np.testing.assert_allclose(z['Base_postprocessed'],cb,atol=1e-10,rtol=0)
                np.testing.assert_allclose(z['Base_same_reference_postprocessed'],cc,atol=1e-10,rtol=0)
                for name in z.files:
                    assert np.isfinite(z[name]).all()
                    records.append({'task':task,'data_id':c['data_id'],'family':m['family'],'rho':m['rho'],
                        'method':name,**metrics(truth,z[name],task,max(float(np.std(truth)),1e-12))})
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
        base=float(g.loc[g.method=='Base',metric].iloc[0]);shared=float(g.loc[g.method=='Shared_adapted',metric].iloc[0])
        control=float(g.loc[g.method=='Base_same_reference_postprocessed',metric].iloc[0])
        gains.append({'task':task,'family':family,'Base':base,'Shared':shared,'gain':shared-base,
            'gain_vs_same_reference_postprocessing_control':shared-control,'threshold':.05,'passed':shared-base>=.05})
    atomic_json(OUT/'numerical_acceptance.json',{'family_results':gains,'all_six_pass':all(x['passed'] for x in gains)})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'independent_audit.json',{'status':'PASS','conditions':count,
        'predictions_and_controls_replayed':count*3,'Base_bitwise_unchanged':True,
        'source_prior_without_query_labels':True,'regression_reference_fusion_refitted':True,'Markdown_bytes':0})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fit','report']);a=p.parse_args()
    fit_all() if a.phase=='fit' else report()
