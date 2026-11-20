"""Additional query rows for metric-aligned ABCD, without ordering selection.

Freshness is limited to previous ABCD query cohorts, not all project history.
Unavailable small tasks are recorded structurally, never selected by scores.
Classification retains semisynthetic label-conditional generation and target64
adaptation. Regression retains v87 source/history information, with no target
labels supplied to fitting or inference. Final truth is opened after freezing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import abcd_metric_aligned_v100 as study
import target_calibration_split_v97 as old
import seven_shifts as seven
from prepare_data import RUN,LOCAL_CC18,atomic_json,protocol
from shared_reference import contexts
from run_tabicl import Predictor,preserving_bootstrap,metrics
from report_abcd_v84 import paired_statistics

OUT=RUN/'results/abcd_fresh_confirmation_v101'


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'code_sha256':study.origin.sha(__file__),
        'purpose':'confirm ABCD contrasts on additional previously unqueried ABCD rows',
        'classification_max_rows':96,'regression_max_rows':256,'minimum_rows':16,
        'classification_exclusions':['original source256','original five-query128','old seven-query','v97 query rows','feature duplicates of previous source/query'],
        'regression_exclusions':['previous Concept/seven query rows','feature duplicates of source and previous Concept queries'],
        'datasets_selected_by':'available disjoint distinct rows only; no model score filtering',
        'classification_target_labels':64,'regression_target_labels':0,
        'constructor_bank_and_benchmark_parameters':'unchanged from v98/v87',
        'old_query_scores_not_used_to_choose_fresh_configuration':True,
        'fresh_evaluation_truth_opened_after_all_policies_frozen':True,
        'freshness_scope':'prior ABCD queries only; not global project-wide historical independence',
        'statistical_family':'eight A-B,A-C,B-D,C-D primary contrasts; classification Accuracy/regression R2',
        'original_paper_replication':False,'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def class_prepare(meta,seven_ids):
    task_id=int(meta['task_id']);out=OUT/'inputs/classification'/f'{task_id}.npz'
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    folder=LOCAL_CC18/str(task_id)
    frame=pd.read_parquet(folder/'X.parquet');raw_y=pd.read_parquet(folder/'y.parquet').iloc[:,0]
    valid=np.flatnonzero(~raw_y.isna().to_numpy())
    with np.load(RUN/'cache/five/classification'/f'{task_id}.npz') as z:
        src=z['source_indices'];oldq=z['target_indices'];sy=z['y_source']
    with np.load(old.OUT/'inputs'/f'{task_id}.npz') as z:previous={k:z[k] for k in z.files}
    encoder=LabelEncoder().fit(raw_y.iloc[src])
    excluded=set(src)|set(oldq)|set(previous['evaluation_raw_ids'])
    if task_id in seven_ids:
        labels=LabelEncoder().fit_transform(raw_y.iloc[valid])
        subset=protocol._balanced_cap_indices(labels,448,0)
        _,q=train_test_split(subset,train_size=320,test_size=128,stratify=labels[subset],random_state=0)
        excluded.update(valid[q])
    candidates=np.array([i for i in valid if i not in excluded],dtype=int)
    candidate_labels=encoder.transform(raw_y.iloc[candidates])
    if len(candidates)>2048:
        cap=protocol._balanced_cap_indices(candidate_labels,2048,101000+task_id)
        candidates,candidate_labels=candidates[cap],candidate_labels[cap]
    prior_ids=np.r_[oldq,previous['evaluation_raw_ids']]
    S,allQ,_=old.numeric_source_only(frame.iloc[src],frame.iloc[np.r_[prior_ids,candidates]])
    prevQ,C=allQ[:len(prior_ids)],allQ[len(prior_ids):]
    used=set(old.sha_rows(np.vstack([S,prevQ])));rng=np.random.default_rng(101000+task_id)
    queues={int(k):rng.permutation(np.flatnonzero(candidate_labels==k)).tolist() for k in np.unique(candidate_labels)}
    chosen=[]
    while len(chosen)<96 and any(queues.values()):
        for k in sorted(queues):
            while queues[k]:
                i=queues[k].pop();h=old.sha_rows(C[i:i+1])[0]
                if h not in used:chosen.append(i);used.add(h);break
            if len(chosen)>=96:break
    info={'task':'classification','task_id':task_id,'data_id':int(meta['data_id']),'dataset':meta['dataset'],
        'evaluation_rows':len(chosen),'status':'COMPLETE' if len(chosen)>=16 else 'UNAVAILABLE',
        'selection_uses_performance':False}
    if len(chosen)<16:
        atomic_json(out.with_suffix('.json'),info);return info
    ids=candidates[chosen];Q=C[chosen];truth=candidate_labels[chosen]
    center=S.mean(0);scale=np.maximum(S.std(0),1e-6)
    A=np.clip((S-center)/scale,-5,5);B=np.clip((Q-center)/scale,-5,5)
    if A.shape[1]>16:
        projector=TruncatedSVD(n_components=16,random_state=0,n_iter=5).fit(A)
        A,B=projector.transform(A),projector.transform(B)
    c=A.mean(0);s=np.maximum(A.std(0),1e-6);A,B=(A-c)/s,(B-c)/s
    prior=old.parent.generator.latent(meta);width=A.shape[1];rng=np.random.default_rng(1010000+task_id)
    latent={'source_real':A,'target_real':B,'source_noise':prior['source_noise'][:,:width],
        'target_noise':rng.normal(size=B.shape),'source_z_noise':prior['source_z_noise'],
        'target_z_noise':rng.normal(size=len(ids)),'source_corruption_noise':prior['source_corruption_noise'],
        'y_source':sy,'y_target':truth}
    config=old.parent.selection()['configuration'];arrays={}
    for rho in (0,20,50,80,100):
        case=old.parent.generator.materialize(meta,config,rho,latent)
        assert np.array_equal(case['X_source'],previous[f'X_source_{rho}'])
        arrays[f'X_source_{rho}']=case['X_source']
        if rho==0:arrays['X_evaluation']=case['X_target']
    arrays.update(y_source=sy,X_calibration=previous['X_calibration'],y_calibration=previous['y_calibration'],evaluation_raw_ids=ids)
    assert set(ids).isdisjoint(excluded)
    out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,**arrays)
    p=OUT/'truth/classification'/f'{task_id}.npz';p.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(p,y_evaluation=truth,raw_ids=ids)
    info.update(input_sha256=study.origin.sha(out),truth_sha256=study.origin.sha(p),
        evaluation_classes=len(np.unique(truth)),synthetic_generation_uses_labels=True,
        original_source_inputs_unchanged=True,extra_target_labels=64)
    atomic_json(out.with_suffix('.json'),info);return info


def reg_prepare(meta):
    task_id=int(meta['task_id']);out=OUT/'inputs/regression'/f'{task_id}.npz'
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    with np.load(RUN/'cache/five/regression'/f'{task_id}.npz') as z:
        sourceX=z['X_source'];sourcey=z['y_source'];Q=z['X_target'];truth=z['y_target'];raw_ids=z['target_indices']
    rng=np.random.default_rng(2718);rng.permutation(len(sourcey))[:min(len(sourcey),1024)]
    old_ids=rng.choice(len(truth),min(256,len(truth)),replace=False)
    candidates=np.setdiff1d(np.arange(len(truth)),old_ids)
    rng=np.random.default_rng(101000+task_id);rng.shuffle(candidates)
    used=set(old.sha_rows(np.vstack([sourceX,Q[old_ids]])))
    kept=[]
    for i in candidates:
        h=old.sha_rows(Q[i:i+1])[0]
        if h not in used:kept.append(i);used.add(h)
        if len(kept)>=256:break
    ids=np.array(kept,dtype=int)
    info={'task':'regression','task_id':task_id,'data_id':int(meta['data_id']),'dataset':meta['dataset'],
        'status':'COMPLETE' if len(ids)>=16 else 'UNAVAILABLE','evaluation_rows':len(ids),'selection_uses_performance':False}
    if len(ids)<16:
        atomic_json(out.with_suffix('.json'),info);return info
    clean=list(seven.regression_clean(meta));assert np.array_equal(clean[-2],Q[old_ids])
    clean[-2]=Q[ids];clean[-1]=np.zeros(len(ids))
    seven.PARAMETERS['regression_response_drift_sd']=1.
    arrays={'X_evaluation':Q[ids],'evaluation_raw_ids':raw_ids[ids]}
    for rho in (0,20,50,80,100):
        values,_=seven.materialize(meta,'concept',rho,clean)
        for k in ('X_source','y_source','X_pool','y_pool'):arrays[f'{k}_{rho}']=values[k]
    out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,**arrays)
    p=OUT/'truth/regression'/f'{task_id}.npz';p.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(p,y_evaluation=truth[ids],raw_ids=raw_ids[ids])
    info.update(input_sha256=study.origin.sha(out),truth_sha256=study.origin.sha(p),extra_target_labels=0)
    atomic_json(out.with_suffix('.json'),info);return info


def prepare():
    freeze();ids={int(m['task_id']) for m in seven.registry() if m['task']=='classification'}
    jobs=[('classification',m) for m in old.parent.generator.cohort()]
    jobs += [('regression',m) for m in seven.registry() if m['task']=='regression']
    records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(class_prepare,m,ids) if t=='classification' else pool.submit(reg_prepare,m) for t,m in jobs]
        for f in as_completed(futures):
            r=f.result();records.append(r);print('FRESH_COHORT',r['task'],r['task_id'],r['status'],r['evaluation_rows'],flush=True)
    pd.DataFrame(records).to_csv(OUT/'cohorts.csv',index=False)
    atomic_json(OUT/'preparation.json',{'status':'COMPLETE','tasks':len(records),'availability':pd.DataFrame(records).groupby(['task','status']).size().to_dict().__str__()})


def infer_worker(shard,shards):
    torch.set_num_threads(1);predictors={}
    paths=sorted((OUT/'inputs').glob('*/*.npz'))
    for i,path in enumerate(paths):
        if i%shards!=shard:continue
        task=path.parent.name;tid=int(path.stem);meta=json.loads(path.with_suffix('.json').read_text())
        assert study.origin.sha(path)==meta['input_sha256']
        with np.load(path) as z:data={k:z[k] for k in z.files}
        if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
        model=predictors[task]
        def infer(X,y,Q):
            for estimator in model.models.values():
                root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                if root is not None and hasattr(root,'root'):root.root=None
            torch.cuda.empty_cache()
            return model(X,y,Q,n_classes=len(np.unique(data['y_source'])) if task=='classification' else None)[0]
        Q=data['X_evaluation']
        for rho in (0,20,50,80,100):
            out=OUT/'banks'/task/str(tid)/f'{rho}.npz'
            if out.with_suffix('.json').exists():continue
            if task=='classification':
                X,y=data[f'X_source_{rho}'],data['y_source'];T,t=data['X_calibration'],data['y_calibration']
                union=np.vstack([T,Q]);query=np.arange(len(Q))+len(T)
                A,b=study.cls.banks.native_context(X,y,T,t)
                views=contexts(np.vstack([X,T]),np.r_[y,t],union,A,b,256,task)
                C,d=study.cls.banks.target_context(X,y,T,t);views[1]=('target_labeled',C,d,union)
                base=infer(A,b,union)[query];bank=np.stack([infer(U,v,V)[query] for _,U,v,V in views])
                ids=preserving_bootstrap(b,np.random.default_rng(0))
                other=infer(A[ids],b[ids],union)[query]
                arrays={'base':base,'members':bank,'bootstrap':other,'zero':infer(X,y,union)[query]}
            else:
                X,y,P,py=(data[f'{k}_{rho}'] for k in ('X_source','y_source','X_pool','y_pool'))
                views=contexts(P,py,Q,X,y,256,task)
                bank=np.stack([infer(A,b,V) for _,A,b,V in views]);A,b,_=study.origin.repaired_context(X,y)
                arrays={'base':infer(X,y,Q),'members':bank,'repair':infer(A,b,Q),
                    'mu':float(py.mean()),'sigma':max(float(py.std()),1e-9)}
            out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,**arrays)
            atomic_json(out.with_suffix('.json'),{'status':'COMPLETE','input_sha256':meta['input_sha256'],
                'evaluation_labels_used':False,'context_rows':[len(v[2]) for v in views]})
        print('FRESH_INFER',task,tid,flush=True)


def infer():
    assert json.loads((OUT/'preparation.json').read_text())['status']=='COMPLETE'
    assert json.loads((study.OUT/'execution.json').read_text())['status']=='COMPLETE'
    assert json.loads((study.OUT/'plan.json').read_text())['code_sha256']==study.origin.sha(study.__file__)
    policies=list((study.OUT/'classification').glob('*/*_policy.json'))+list((study.OUT/'regression').glob('*/*_policy.json'))
    atomic_json(OUT/'frozen_policies.json',{str(p.relative_to(study.OUT)):study.origin.sha(p) for p in policies})
    def run(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i)
        with (OUT/f'infer_{i}.log').open('a') as f:
            p=subprocess.run([sys.executable,__file__,'infer-worker','--shard',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode}
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(run,i) for i in range(8)]):
            r=f.result();records.append(r);print(r,flush=True)
    atomic_json(OUT/'inference.json',{'status':'COMPLETE' if not any(r['returncode'] for r in records) else 'FAILED','workers':records})


def report():
    assert json.loads((OUT/'inference.json').read_text())['status']=='COMPLETE'
    seal=json.loads((OUT/'frozen_policies.json').read_text())
    assert all(study.origin.sha(study.OUT/p)==h for p,h in seal.items())
    mapping,_=study.cls.fold_mapping('regression','five');rows=[]
    for path in sorted((OUT/'inputs').glob('*/*.npz')):
        task=path.parent.name;tid=int(path.stem);meta=json.loads(path.with_suffix('.json').read_text())
        truth=OUT/'truth'/task/f'{tid}.npz';assert study.origin.sha(truth)==meta['truth_sha256']
        with np.load(truth) as z:y=z['y_evaluation']
        for rho in (0,20,50,80,100):
            with np.load(OUT/'banks'/task/str(tid)/f'{rho}.npz') as z:values={k:z[k] for k in z.files}
            for method,seen,matched in study.METHODS:
                bank=values['members'].copy()
                if task=='classification' and not matched:bank[3]=values['bootstrap']
                if task=='regression' and matched:bank[0]=values['repair']
                c={'task':task,'base':values['base'],'bank':bank}
                if task=='regression':c.update(mu=float(values['mu']),sigma=float(values['sigma']))
                folder=study.OUT/('classification/'+str(tid) if task=='classification' else 'regression/'+str(mapping[meta['data_id']]))
                policy=json.loads((folder/f'{method}_policy.json').read_text())['policy']
                predictions={method:study.predict(c,policy)}
                if method=='A':
                    predictions.update(Base=c['base'],Single_matched=bank[3 if task=='classification' else 0])
                    if task=='classification':predictions.update(Base_zero=values['zero'],Target_only=bank[1])
                for name,q in predictions.items():
                    p=OUT/'predictions'/task/str(tid)/f'{rho}_{name}.npz';p.parent.mkdir(exist_ok=True,parents=True)
                    np.savez_compressed(p,prediction=q)
                    rows.append({'task':task,'task_id':tid,'data_id':meta['data_id'],'rho':rho,'method':name,
                        'query_rows':len(y),**metrics(y,q,task,max(float(y.std()),1e-12))})
    detail=pd.DataFrame(rows);detail.to_csv(OUT/'detail.csv',index=False);tables=[];tests=[]
    for task,group in detail.groupby('task'):
        first,second=('accuracy','auc') if task=='classification' else ('r2','nrmse')
        values=group[group.rho>0].groupby(['data_id','method'])[[first,second]].mean()
        clean=group[group.rho==0].set_index(['data_id','method'])[[first,second]];base=values.xs('Base',level='method')
        for method in group.method.unique():
            v=values.xs(method,level='method');cl=clean.xs(method,level='method')
            tables.append({'task':task,'method':method,'datasets':len(v),first:v[first].mean(),second:v[second].mean(),
                'clean_primary':cl[first].mean(),'gain_vs_Base':(v[first]-base[first]).mean()})
        for a,b in study.CONTRASTS:
            delta=(values.xs(a,level='method')[first]-values.xs(b,level='method')[first]).dropna()
            tests.append({'task':task,'contrast':a+'-'+b,'datasets':len(delta),**paired_statistics(delta)})
    table=pd.DataFrame(tables);stats=pd.DataFrame(tests);prev=0.
    for rank,i in enumerate(sorted(stats.index,key=lambda j:stats.loc[j,'p_two_sided'])):
        prev=max(prev,min(1.,(8-rank)*stats.loc[i,'p_two_sided']));stats.loc[i,'p_holm']=prev
    stats['positive_significant']=(stats.gain_ci_low>0)&(stats.p_holm<.05)
    table.to_csv(OUT/'tables.csv',index=False);stats.to_csv(OUT/'contrasts.csv',index=False)
    atomic_json(OUT/'audit.json',{'status':'PASS','policy_hashes_unchanged':True,'truth_opened_after_frozen_policies':True,
        'all_requested_contrast_means_positive':bool((stats.gain>0).all()),
        'all_requested_contrasts_positive_significant':bool(stats.positive_significant.all()),
        'no_ordering_criterion_in_calibration':True,'datasets':detail.groupby('task').task_id.nunique().to_dict(),
        'historical_global_independence_not_claimed':True,'Markdown_bytes':(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(table.to_string(index=False),flush=True);print(stats.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer','infer-worker','report']);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='infer':infer()
    elif a.phase=='infer-worker':infer_worker(a.shard,8)
    else:report()
