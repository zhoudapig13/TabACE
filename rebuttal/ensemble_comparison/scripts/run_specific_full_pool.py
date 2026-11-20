"""Materialize all declared five-family Specific candidates, with cached aliases."""
import argparse,json,logging,traceback,warnings
from functools import partial
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition,contexts,Predictor,PFNPredictor
from specific_reference import specific_contexts
from specific_full_registry import extra_contexts


def signature(view):return tuple(protocol.array_sha256(np.asarray(v)) for v in view[1:])


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five','five_feature_calibrated'],required=True)
    p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    task='regression' if a.track=='five' else 'classification'
    predictor=Predictor(task,'cuda:0') if a.backbone=='tabicl' else PFNPredictor(task)
    conditions=[(f,r) for f in protocol.FAMILIES for r in [0,20,50,80,100]]
    jobs=[(m,f,r) for m in specifications(a.track) if m['task']==task for f,r in conditions][a.shard::a.shards]
    failures=0
    for meta,family,rho in jobs:
        out=RUN/'results/reference_specific_full'/a.backbone/a.track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        shared=RUN/'results/reference'/a.backbone/a.track/task/str(meta['task_id'])/('original_0.npz' if rho==0 else out.name)
        directed=RUN/'results/reference_specific'/a.backbone/a.track/task/str(meta['task_id'])/out.name
        if not shared.with_suffix('.json').exists() or not directed.with_suffix('.json').exists():
            # Resumable prerequisite wait does not run an alternate input protocol.
            import time
            while not shared.with_suffix('.json').exists() or not directed.with_suffix('.json').exists():time.sleep(5)
        try:
            data,info=load_condition(meta,family,rho)
            X,y,Q=(data[k] for k in ['X_source','y_source','X_target']);PX=data.get('X_pool',X);py=data.get('y_pool',y);budget=len(y)
            with np.load(shared) as f:base=f['base'];sb=f['members'];mean=f['source_mean'];scale=f['source_scale']
            with np.load(directed) as f:db=f['members'];assert np.array_equal(f['base'],base)
            common=contexts(PX,py,Q,X,y,budget,task,base=base)
            special=specific_contexts(PX,py,Q,X,y,budget,task,family=family,track=a.track,base=base)
            all_views=common+special;predictions=list(sb)+list(db);cached={}
            for view,prediction in zip(all_views,predictions):cached[signature(view)]=prediction
            aliases=0;new_calls=0
            for view in extra_contexts(PX,py,Q,budget,task,family):
                key=signature(view)
                if key in cached:prediction=cached[key];aliases+=1
                else:
                    _,A,b,T=view;prediction,_=predictor(A,b,T,n_classes=info['n_classes']);cached[key]=prediction;new_calls+=1
                all_views.append(view);predictions.append(prediction)
            bank=np.stack(predictions);assert np.isfinite(bank).all()
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,base=base,members=bank,y_target=data['y_target'],source_mean=mean,source_scale=scale);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':a.backbone,
                'context_roles':[v[0] for v in all_views],'context_rows':[len(v[2]) for v in all_views],
                'candidate_count':len(all_views),'unique_input_contexts':len(cached),'new_backbone_calls':new_calls,
                'context_budget':budget,'registry_budget_revision':2,
                'aliases_reused':aliases,'context_count_cap':None,'includes_shared_registry':True,
                'constructors_receive_target_labels':False,'constructors_receive_strength':False,
                'original_registered_pool_bit_exact':False,'registry':'declared five-family Appendix ranges',
                'reference_status':'Specific full declared range reconstruction','seed':0})
            print('FULL_SPECIFIC_BANK',a.backbone,a.track,meta['task_id'],family,rho,len(all_views),new_calls,flush=True)
        except Exception as exc:
            failures+=1;atomic_json(RUN/'results/inference_failures'/f"specific_full_{a.backbone}_{a.track}_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()});print('FULL_SPECIFIC_FAILED',repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
