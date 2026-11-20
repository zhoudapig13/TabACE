"""Apply frozen benchmark-development policies once to reserved, disjoint queries.

Query labels are passed only to the final metric calculation. Contexts and
features are constructed before they are read for scoring. These are new-query
tests on previously seen datasets, not tests on unseen datasets.
"""
import argparse, hashlib, json, logging, time, traceback, warnings
from functools import partial
import numpy as np
import torch
from prepare_data import RUN, atomic_json, protocol
from shared_reference import Predictor, PFNPredictor, contexts, load_condition
from specific_reference import specific_contexts
from run_tabicl import metrics
from fit_benchmark_reference import predict


def policy_paths(backbone,track):
    root=RUN/'results/benchmark_tuning'/backbone/track/('regression' if track.startswith('seven') else 'classification')
    families=list(protocol.FAMILIES)
    if track.startswith('seven'):
        from seven_shifts import FAMILIES
        families=list(FAMILIES)
    return root/'shared_all.json',{f:root/f'specific_{f}.json' for f in families}


def cell(meta,base,bank,y,source_mean,source_scale):
    c={'meta':meta,'base':base,'bank':bank,'y':y,'source_mean':source_mean,'source_scale':source_scale}
    if meta['task']=='classification':
        k=base.shape[1];a=np.clip(bank,1e-12,1)
        q=1+np.sum(a*np.log(a),axis=2)/np.log(k)
        s=1-.5*np.sum(np.abs(bank-bank.mean(0)[None]),axis=2)
        b=np.clip(base,1e-12,1);u=-np.sum(b*np.log(b),axis=1)/np.log(k)
        c.update(Q=q.T,S=s.T,U=u)
    else:
        members=(bank-source_mean)/source_scale;b=(base-source_mean)/source_scale
        c.update(M=members.T,B=b,D=np.abs(members-np.median(members,axis=0)[None]).T)
    return c


def build_input(meta,track,family,rho,Q,t):
    if track=='five_feature_calibrated':
        from feature_calibrated_protocol import materialize
        return materialize(meta,family,rho,query=(Q,t))
    if track=='five_calibrated':
        data,info=load_condition(meta|{'track':track},family,rho)
        data['X_target'],data['y_target']=Q,t
        return data,info
    from seven_shifts import regression_clean
    from calibrated_seven_protocol import materialize
    clean=list(regression_clean(meta));clean[-2:]=[Q,t]
    frozen=json.loads((RUN/'configs/calibrated_protocol_frozen.json').read_text())
    data,info=materialize(meta,family,rho,clean,frozen['seven_regression']['amplitude_multiplier'])
    return data,info|{'track':track,'n_classes':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    manifest=json.loads((RUN/'configs/benchmark_tuning_holdout_manifest.json').read_text())
    jobs=[]
    for track in ['five_calibrated','five_feature_calibrated','seven_calibrated']:
        key='seven_regression' if track.startswith('seven') else 'classification'
        jobs.extend((track,m,key) for m in manifest[key] if m['holdout_status']=='READY')
    models={};failures=0
    for track,meta,key in jobs[a.shard::a.shards]:
        shared_path,specific_paths=policy_paths(a.backbone,track)
        while not all(p.exists() for p in [shared_path,*specific_paths.values()]):time.sleep(5)
        shared=json.loads(shared_path.read_text());specific={f:json.loads(p.read_text()) for f,p in specific_paths.items()}
        # Save the exact frozen policies with every independently evaluated task.
        signatures={'shared':hashlib.sha256(shared_path.read_bytes()).hexdigest(),
            'specific':{f:hashlib.sha256(p.read_bytes()).hexdigest() for f,p in specific_paths.items()}}
        task=meta['task']
        if task not in models:models[task]=Predictor(task,'cuda:0') if a.backbone=='tabicl' else PFNPredictor(task)
        predictor=models[task]
        with np.load(RUN/'cache/tuning_holdout'/key/f"{meta['task_id']}.npz") as f:
            Q,t,ids=f['X_target'],f['y_target'],f['query_indices']
        families=list(specific)
        conditions=[(f,r) for f in families for r in [20,50,80,100]]
        for family,rho in conditions:
            out=RUN/'results/benchmark_tuning_holdout'/a.backbone/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():
                old=json.loads(out.with_suffix('.json').read_text());assert old['policy_signatures']==signatures
                continue
            try:
                data,info=build_input(meta,track,family,rho,Q,t)
                X,y,T=data['X_source'],data['y_source'],data['X_target'];pool=data.get('X_pool',X);labels=data.get('y_pool',y)
                count=info['n_classes'];budget=len(y);base,_=predictor(X,y,T,n_classes=count)
                mean=float(labels.mean());scale=max(float(labels.std()),1e-9)
                output={'Base':base};banks={}
                for variant,policy in [('Shared',shared),('Specific',specific[family])]:
                    if variant=='Shared':views=contexts(pool,labels,T,X,y,budget,task,base=base)
                    else:
                        views=specific_contexts(pool,labels,T,X,y,budget,task,base=base,family=family,track=track,
                            history=(data.get('X_history'),data.get('y_history')),
                            reference=(data.get('X_reference'),data.get('y_reference')))
                    bank=np.stack([predictor(A,b,V,n_classes=count)[0] for _,A,b,V in views]);banks[variant]=bank
                    output[f'{variant}_context_mean']=bank.mean(0)
                    c=cell(info,base,bank,data['y_target'],mean,scale)
                    output[f'TabACE_{variant}_benchmark_tuned_R5']=predict(c,np.array(policy['parameters']),task,policy['gain'])
                eval_scale=max(float(data['y_target'].std()),1e-12) if task=='regression' else 1.
                scores={name:metrics(data['y_target'],prediction,task,eval_scale) for name,prediction in output.items()}
                out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**{'prediction__'+k:v for k,v in output.items()},
                    y_target=data['y_target'],query_indices=ids,shared_bank=banks['Shared'],specific_bank=banks['Specific']);tmp.replace(out)
                atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':a.backbone,'methods':scores,
                    'results_role':'independent queries on benchmark-development datasets',
                    'holdout_query_labels_used_for_fit_or_selection':False,'new_dataset_evaluation':False,
                    'policy_signatures':signatures,'query_indices_sha256':protocol.array_sha256(ids),
                    'X_shifted_holdout_sha256':protocol.array_sha256(T),'seed':0})
                print('HOLDOUT_DONE',a.backbone,track,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;print('HOLDOUT_FAILED',a.backbone,track,meta['task_id'],family,rho,repr(exc),flush=True)
                atomic_json(RUN/'results/inference_failures'/f"holdout_{a.backbone}_{track}_{meta['task_id']}_{family}_{rho}.json",
                    {'error':repr(exc),'traceback':traceback.format_exc()})
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
