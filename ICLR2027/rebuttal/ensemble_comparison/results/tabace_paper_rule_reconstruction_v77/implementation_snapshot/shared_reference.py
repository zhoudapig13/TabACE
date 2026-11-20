"""Appendix-based Shared constructors and a matched frozen-backbone reference.

This is an explicitly labelled reconstruction, not a claim that the archived
main-table TabACE implementation has been recovered. Constructors never receive
family, strength, or target labels. On the seven-shift reconstruction every
method has the same accessible source/history/reference pool and context cap.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from prepare_data import RUN,atomic_json,protocol
import sys
sys.path.insert(0,str(RUN.parents[1]/'src'))
from run_tabicl import Predictor,normalize,metrics


def geometry(X,Q):
    center=np.median(X,axis=0)
    scale=np.maximum(np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0),1e-6)
    return (X-center)/scale,(Q-center)/scale,center,scale


def class_subset(y,score,fraction,budget):
    count=min(budget,max(len(np.unique(y)),int(np.ceil(len(y)*fraction))))
    rank=np.argsort(score,kind='stable');keep=list(rank[:count])
    for label in np.unique(y):
        if np.any(y[keep]==label):continue
        candidate=next(int(i) for i in rank if y[i]==label)
        for position in range(len(keep)-1,-1,-1):
            if np.sum(y[keep]==y[keep[position]])>1:
                keep[position]=candidate;break
    return np.asarray(keep,dtype=int)


def matched_rows(Z,V,n,seed):
    distances=pairwise_distances(Z,V,metric='sqeuclidean')
    k=min(10,len(V));nearest=np.partition(distances,k-1,axis=1)[:,:k].mean(1)
    positive=nearest[nearest>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    weights=np.exp(np.clip(-nearest/max(bandwidth,1e-12),-700,0))
    weights/=weights.sum()
    return np.random.default_rng(seed).choice(len(Z),size=n,replace=True,p=weights)


def contexts(X,y,Q,canonical_X,canonical_y,budget,task,base=None):
    """Five Shared registry roles from context_construction_details.tex."""
    Z,V,center,scale=geometry(X,Q)
    if task=='classification':
        anchors=np.quantile(V,[.1,.3,.5,.7,.9],axis=0)
        anchor_distance=pairwise_distances(Z,anchors,metric='sqeuclidean').min(1)
        target_distances=pairwise_distances(Z,V,metric='sqeuclidean')
        target_k=min(10,len(V))
        target_distance=np.partition(target_distances,target_k-1,axis=1)[:,:target_k].mean(1)
        outlying=np.minimum(np.abs(Z),10.)**2
        outlying=outlying.mean(1)
        distance=pairwise_distances(Z,metric='sqeuclidean')
        np.fill_diagonal(distance,np.inf)
        k=min(10,len(y)-1)
        neighbors=np.argpartition(distance,k-1,axis=1)[:,:k]
        consensus=np.max(np.stack([(y[neighbors]==label).mean(1) for label in np.unique(y)],axis=1),axis=1)
        disagreement=1-consensus
        specifications=[('target_neighbor',anchor_distance,.8),
            ('representative',target_distance+.35*outlying,.7),
            ('locally_stable',disagreement+.10*target_distance,.8),
            ('low_conflict',disagreement+.20*outlying,.75)]
        result=[]
        for name,score,fraction in specifications:
            ids=class_subset(y,score,fraction,budget)
            result.append((name,X[ids],y[ids],Q))
        numeric=y.astype(float);association=np.abs(np.sum((X-X.mean(0))*(numeric-numeric.mean())[:,None],axis=0))
        association/=np.maximum(np.sqrt(np.sum((X-X.mean(0))**2,axis=0)*np.sum((numeric-numeric.mean())**2)),1e-12)
        selected=np.array([j for j in range(X.shape[1]) if j!=int(np.argmax(association))])
        if not len(selected):selected=np.array([0])
        result.insert(3,('feature_subspace',canonical_X[:,selected],canonical_y,Q[:,selected]))
        return result
    rng=np.random.default_rng(0)
    bootstrap=rng.choice(len(y),size=budget,replace=True)
    ordered=np.argsort(y,kind='stable');parts=[p for p in np.array_split(ordered,8) if len(p)]
    counts=np.full(len(parts),budget//len(parts));counts[:budget%len(parts)]+=1
    balanced=np.concatenate([rng.choice(p,int(n),replace=True) for p,n in zip(parts,counts)])
    rng.shuffle(balanced)
    matched=matched_rows(Z,V,budget,0)
    target_center=np.median(Q,axis=0)
    target_scale=np.maximum(np.percentile(Q,75,axis=0)-np.percentile(Q,25,axis=0),1e-6)
    aligned=Z*target_scale+target_center
    # On a larger allowed pool the aligned branch retains the canonical context
    # identities. Pool-based sampling is used only by the registered resamplers.
    canonical_aligned=(canonical_X-center)/scale*target_scale+target_center
    aligned_Z=(aligned-center)/scale
    aligned_matched=matched_rows(aligned_Z,V,budget,0)
    return [('uniform_bootstrap',X[bootstrap],y[bootstrap],Q),
        ('response_balanced',X[balanced],y[balanced],Q),
        ('target_matched',X[matched],y[matched],Q),
        ('distribution_aligned',canonical_aligned,canonical_y,Q),
        ('aligned_target_matched',aligned[aligned_matched],y[aligned_matched],Q)]


class PFNPredictor:
    def __init__(self,task):self.task=task;self.models={}
    def __call__(self,X,y,Q,*,members=1,n_classes=None,**kwargs):
        from run_tabpfn import model,fit_predict
        if members not in self.models:
            self.models[members]=model(self.task,'cuda:0')
            self.models[members].n_estimators=members
        return fit_predict(self.models[members],X,y,Q,self.task,n_classes),None


def specifications(track):
    if track.startswith('five'):
        output=[]
        for task in ['classification','regression']:
            for p in sorted((RUN/'cache/five'/task).glob('*.json')):
                info=json.loads(p.read_text())
                if track in ['five_corrected','five_calibrated','five_feature_calibrated'] and task!='classification':continue
                output.append(info|{'track':track})
        return output
    from seven_shifts import registry
    values=registry()
    if track in ['seven_calibrated','seven_calibrated_v2']:return [m|{'track':track} for m in values if m['task']=='regression']
    return values


def load_condition(meta,family,rho):
    if meta['track']=='seven_calibrated_v2':
        from calibrated_seven_protocol import materialize
        from seven_shifts import regression_clean
        configuration=json.loads((RUN/'configs/seven_regression_protocol_v2.json').read_text())
        source=RUN/'cache/seven_calibrated_v2'/meta['task']/str(meta['task_id'])/f'{family}_{rho}.npz'
        if not source.with_suffix('.json').exists():
            data,info=materialize(meta,family,rho,regression_clean(meta),configuration['family_amplitude_multipliers'][family])
            info.update(track='seven_calibrated_v2',condition_materialized=True,n_classes=None,
                protocol_sha256=configuration['protocol_sha256'],severity_selection_data_role='benchmark development',
                calibration_protocol='familywise amplitude selected using benchmark-development Base only',
                original_paper_generator_recovered=False)
            source.parent.mkdir(parents=True,exist_ok=True);tmp=source.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**data);tmp.replace(source);atomic_json(source.with_suffix('.json'),info)
        info=json.loads(source.with_suffix('.json').read_text());assert info['protocol_sha256']==configuration['protocol_sha256']
        with np.load(source) as f:data={k:f[k] for k in f.files}
        return data,info
    if meta['track']=='five_feature_calibrated':
        from feature_calibrated_protocol import load
        return load(meta,family,rho)
    if meta['track'].startswith('five'):
        with np.load(RUN/'cache/five'/meta['task']/f"{meta['task_id']}.npz") as f:
            arrays={k:f[k] for k in f.files}
        active=protocol
        if meta['track']=='five_calibrated':
            from calibrated_five_protocol import corrupt
            frozen=json.loads((RUN/'configs/calibrated_protocol_frozen.json').read_text())
            assert frozen['status']=='FROZEN'
            configuration=frozen['five_classification']
            A,b,details=corrupt(arrays['X_source'],arrays['y_source'],family=family,rho=rho/100.,
                              **{k:v for k,v in configuration.items() if k!='name'})
            classes=len(np.unique(arrays['y_source']))
            arrays['X_source'],arrays['y_source']=A,b
            return arrays,meta|{'n_classes':classes,'family':family,'rho':rho,'condition_materialized':True,
                'calibrated_protocol_sha256':frozen['protocol_sha256'],'severity_parameters':configuration,
                'calibration_block_sha256':__import__('hashlib').sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
                'X_context_sha256':protocol.array_sha256(A),'y_context_sha256':protocol.array_sha256(b)}
        if meta['track']=='five_corrected':import corrected_source_protocol as active
        changed=active.manipulate_source_context(meta['task'],arrays['X_source'],arrays['y_source'],
            family=family,rho=rho/100.,seed=1729+rho,
            shift_scale=6. if meta['task']=='classification' else 10.,label_scale=1.)
        classes=len(np.unique(arrays['y_source'])) if meta['task']=='classification' else None
        arrays['X_source'],arrays['y_source']=changed.X_context,changed.y_context
        extra={}
        if meta['track']=='five_corrected':extra={'polarity_revision':active.REVISION,'condition_materialized':True,
            'X_context_sha256':protocol.array_sha256(changed.X_context),'y_context_sha256':protocol.array_sha256(changed.y_context)}
        return arrays,meta|{'n_classes':classes,'family':family,'rho':rho}|extra
    if meta['track']=='seven_calibrated':
        from calibrated_seven_protocol import materialize
        from seven_shifts import regression_clean
        frozen=json.loads((RUN/'configs/calibrated_protocol_frozen.json').read_text());assert frozen['status']=='FROZEN'
        source=RUN/'cache/seven_calibrated'/meta['task']/str(meta['task_id'])/f'{family}_{rho}.npz'
        if not source.with_suffix('.json').exists():
            arrays,info=materialize(meta,family,rho,regression_clean(meta),frozen['seven_regression']['amplitude_multiplier'])
            info.update(track='seven_calibrated',condition_materialized=True,n_classes=None,
                        calibrated_protocol_sha256=frozen['protocol_sha256'])
            source.parent.mkdir(parents=True,exist_ok=True)
            temp=source.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(source)
            atomic_json(source.with_suffix('.json'),info)
        info=json.loads(source.with_suffix('.json').read_text());assert info['calibrated_protocol_sha256']==frozen['protocol_sha256']
        with np.load(source) as f:arrays={k:f[k] for k in f.files}
        return arrays,info
    source=RUN/'cache/seven_reimplemented'/meta['task']/str(meta['task_id'])/f'{family}_{rho}.npz'
    with np.load(source) as f:arrays={k:f[k] for k in f.files}
    return arrays,json.loads(source.with_suffix('.json').read_text())|{'n_classes':2 if meta['task']=='classification' else None}


def run_one(predictor,data,meta,out,backbone,context_factory=contexts,variant='shared',include_native=True):
    X,y,Q,truth=(data[k] for k in ['X_source','y_source','X_target','y_target'])
    pool_X=data.get('X_pool',X);pool_y=data.get('y_pool',y)
    count=meta['n_classes'];task=meta['task'];budget=len(y)
    old=RUN/'results'/backbone/meta['track']/task/str(meta['task_id'])/f"{meta['family']}_{meta['rho']}.npz"
    if meta['rho']==0 and not old.exists():
        old=old.with_name('original_0.npz' if meta['track'].startswith('five') else 'covariate_0.npz')
    key='base_protocol_check' if backbone=='tabicl' else 'tabpfn_v3_base_check'
    if old.exists():
        with np.load(old) as f:base=f['prediction__'+key]
        previous=json.loads(old.with_suffix('.json').read_text())
        base_seconds=previous['methods'][key]['seconds'];base_reused=True
    else:
        start=time.perf_counter();base,_=predictor(X,y,Q,n_classes=count)
        torch.cuda.synchronize();base_seconds=time.perf_counter()-start;base_reused=False
    torch.cuda.synchronize();start=time.perf_counter()
    views=context_factory(pool_X,pool_y,Q,X,y,budget,task,base=base)
    construction_seconds=time.perf_counter()-start
    members=[];seconds=[]
    for name,A,b,T in views:
        torch.cuda.synchronize();start=time.perf_counter();p,_=predictor(A,b,T,n_classes=count)
        torch.cuda.synchronize();seconds.append(time.perf_counter()-start);members.append(p)
    bank=np.stack(members);mean_name=('Shared' if variant=='shared' else 'Specific')+'_context_mean_reimplemented'
    predictions={mean_name:bank.mean(0)}
    timing={mean_name:sum(seconds)+construction_seconds}
    if backbone=='tabpfn' and include_native:
        for n in [6,8]:
            torch.cuda.synchronize();start=time.perf_counter()
            predictions[f'tabpfn_native_{n}'],_=predictor(X,y,Q,members=n,n_classes=count)
            torch.cuda.synchronize();timing[f'tabpfn_native_{n}']=time.perf_counter()-start
    evaluation_scale=max(float(truth.std()),1e-12) if task=='regression' and meta['track'].startswith('seven') else meta['scale']
    methods={name:metrics(truth,p,task,evaluation_scale)|{'seconds':timing[name]} for name,p in predictions.items()}
    arrays={'base':base,'members':bank,'y_target':truth,
        'source_scale':np.asarray(max(float(pool_y.std()),1e-9)),'source_mean':np.asarray(float(pool_y.mean()))}
    arrays.update({'prediction__'+name:p for name,p in predictions.items()})
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(out)
    atomic_json(out.with_suffix('.json'),meta|{'status':'COMPLETE','backbone':backbone,
        'reference_status':f'appendix_{variant}_R5_reimplementation_not_archived_implementation','registry_revision':2,
        'context_roles':[v[0] for v in views],'context_rows':[len(v[2]) for v in views],
        'context_budget':budget,'accessible_source_pool_rows':len(pool_y),
        'base_reused':base_reused,'base_seconds':base_seconds,'context_seconds':seconds,
        'construction_seconds':construction_seconds,'methods':methods,
        'constructors_receive_target_labels':False,'constructors_receive_family':variant=='specific',
        'constructors_receive_strength':False,
        'weights_training':'separate dataset-level OOF policy fit','seed':0})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl')
    parser.add_argument('--track',choices=['five','five_corrected','five_calibrated','seven_calibrated','seven_reimplemented','both'],default='both')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);predictors={};failures=0
    import logging,warnings
    logging.disable(logging.INFO);warnings.filterwarnings('ignore',category=FutureWarning)
    tracks=['five','seven_reimplemented'] if args.track=='both' else [args.track]
    jobs=[(track,m) for track in tracks for m in specifications(track)]
    if args.limit:jobs=jobs[:args.limit]
    for job_index,(track,meta) in enumerate(jobs):
        if track.startswith('five'):conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        else:
            from seven_shifts import FAMILIES
            conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        task=meta['task']
        if task not in predictors:predictors[task]=Predictor(task,'cuda:0') if args.backbone=='tabicl' else PFNPredictor(task)
        for condition_index,(family,rho) in enumerate(conditions):
            if (job_index*len(conditions)+condition_index)%args.shards!=args.shard:continue
            out=RUN/'results/reference'/args.backbone/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            if track=='five_corrected' and rho==0:
                from corrected_source_protocol import reuse_unchanged_clean
                old=RUN/'results/reference'/args.backbone/'five'/task/str(meta['task_id'])/'original_0.npz'
                if reuse_unchanged_clean(old,out):continue
            print('START',args.backbone,track,task,meta['task_id'],family,rho,flush=True)
            try:
                arrays,info=load_condition(meta,family,rho)
                run_one(predictors[task],arrays,info,out,args.backbone)
                print('DONE',args.backbone,track,task,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;error=meta|{'family':family,'rho':rho,'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f"reference_{args.backbone}_{track}_{task}_{meta['task_id']}_{family}_{rho}.json",error)
                print('FAILED',json.dumps(error),flush=True);torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
