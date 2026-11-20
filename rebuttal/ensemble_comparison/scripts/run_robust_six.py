"""Frozen TabICL six-view adaptations, preserving the original comparison outputs."""
import argparse,json,logging,time,traceback,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabicl import Predictor,metrics,normalize,preserving_bootstrap
from robust_six_views import construct,density_and_proxy


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--always-views',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    jobs=[]
    for track in ['five_feature_calibrated','seven_reimplemented']:
        families=protocol.FAMILIES
        if track.startswith('seven'):
            from seven_shifts import FAMILIES
            families=FAMILIES
        conditions=[('original' if track.startswith('five') else 'covariate',0)]+[(f,r) for f in families for r in [20,50,80,100]]
        jobs.extend((m,f,r) for m in specifications(track) if m['task']=='classification' for f,r in conditions)
    model=Predictor('classification','cuda:0');failures=0
    for meta,family,rho in jobs[a.shard::a.shards]:
        root='robust_six_all' if a.always_views else 'robust_six'
        out=RUN/'results'/root/'tabicl'/meta['track']/'classification'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        try:
            data,info=load_condition(meta,family,rho);X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
            original=RUN/'results/tabicl'/meta['track']/'classification'/str(meta['task_id'])/out.name
            with np.load(original) as f:base=f['prediction__base_protocol_check'];raw_native=f['prediction__native_6'];raw_random=f['bootstrap_bank']
            tick=time.perf_counter()
            views,diagnostics=construct(X,y,Q,history=(data.get('X_history'),data.get('y_history')),
                recent=(data['X_reference'],data['y_reference']) if 'X_reference' in data else None,force_views=a.always_views)
            previous=RUN/'results/robust_six/tabicl'/meta['track']/'classification'/str(meta['task_id'])/out.name
            if a.always_views and previous.with_suffix('.json').exists() and diagnostics['detected']:
                previous_info=json.loads(previous.with_suffix('.json').read_text())
                with np.load(previous) as f:
                    bank=f['bank'];densities=f['density'];native=f['native_adapted']
                selected=previous_info['native_selected_view'];proxy=previous_info['native_source_proxy_scores']
            elif diagnostics['detected'] or a.always_views:
                bank=[base];densities=[];proxy=[]
                for i,(name,A,b,T) in enumerate(views):
                    density,reliability,_,_=density_and_proxy(A,b,T);densities.append(density);proxy.append(reliability*reliability*float(density.mean()))
                    if i:
                        rows=preserving_bootstrap(b,np.random.default_rng(1009*(i+1)),size=len(y))
                        prediction,_=model(A[rows],b[rows],T,n_classes=info['n_classes'],random_state=i)
                        bank.append(prediction)
                selected=int(np.argmax(proxy))
                if selected:
                    _,A,b,T=views[selected]
                    native,_=model(A,b,T,n_classes=info['n_classes'],members=6)
                else:native=raw_native
                bank=np.stack(bank);densities=np.stack(densities)
            else:
                bank=np.repeat(base[None],6,axis=0);densities=np.ones((6,len(t)));proxy=[0.]*6;selected=0;native=raw_native
            mean=normalize(bank.mean(0));out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,bank=bank,base=base,raw_bank=raw_random,raw_native=raw_native,native_adapted=native,
                density=densities,y_target=t,prediction__Robust_Random6_adapted=mean,prediction__Robust_Native6_adapted=native)
            tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':'tabicl','seed':0,
                'adaptation_revision':2 if a.always_views else 1,'always_compute_distinct_views':a.always_views,
                'roles':[v[0] for v in views],'context_rows':[len(v[2]) for v in views],
                'view_diagnostics':diagnostics,'native_selected_view':selected,'native_source_proxy_scores':proxy,
                'source_row_cap':len(y),'logical_candidate_count':6,'bank_backbone_call_budget':6,'native_call_budget':6,
                'feature_adapter_uses_target_labels':False,'feature_adapter_uses_family_or_strength':False,
                'backbone_weights_frozen':True,'Shared_or_paper_scores_used':False,'seconds':time.perf_counter()-tick,
                'methods':{'Robust_Random6_adapted':metrics(t,mean,'classification',1.),
                           'Robust_Native6_adapted':metrics(t,native,'classification',1.)}})
            print('ROBUST_SIX_DONE',meta['track'],meta['task_id'],family,rho,diagnostics['detected'],selected,flush=True)
        except Exception as exc:
            failures+=1;atomic_json(RUN/'results/inference_failures'/f"robust_six_{meta['track']}_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()});print('ROBUST_SIX_FAILED',repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
