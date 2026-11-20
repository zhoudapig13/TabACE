"""OligoICP's 10%-retained IQR rule at 40 and original-size 400 candidates.

All candidate evaluation costs are counted. Matching 40/400-member bagging is
also reported, so uncertainty selection is compared at the same candidate cost.
The already computed six TabICL regression candidates are reused exactly.
"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabicl import Predictor,metrics


def run_one(predictor,data,meta,out,total):
    task='regression';X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
    PX=data.get('X_pool',X);py=data.get('y_pool',y)
    if out.exists():
        with np.load(out) as f:bank=list(f['bank']);iqr=list(f['iqr']);times=list(f['candidate_seconds'])
    elif out.with_suffix('.partial.npz').exists():
        with np.load(out.with_suffix('.partial.npz')) as f:bank=list(f['bank']);iqr=list(f['iqr']);times=list(f['candidate_seconds'])
    else:
        source=RUN/'results/tabicl'/meta['track']/task/str(meta['task_id'])/f"{meta['family']}_{meta['rho']}.npz"
        old=json.loads(source.with_suffix('.json').read_text());assert old.get('bootstrap_revision',1)>=2
        with np.load(source) as f:bank=list(f['bootstrap_bank']);iqr=list(f['bootstrap_iqr'])
        times=[old['methods']['bootstrap_mean_6']['seconds']/6]*6
    for i in range(len(bank),total):
        rng=np.random.default_rng(1009*(i+1));rows=rng.choice(len(py),len(y),replace=True)
        torch.cuda.synchronize();start=time.perf_counter()
        mean,spread=predictor(PX[rows],py[rows],Q,quantiles=True)
        torch.cuda.synchronize();times.append(time.perf_counter()-start);bank.append(mean);iqr.append(spread)
        if (i+1)%40==0:
            out.parent.mkdir(parents=True,exist_ok=True);partial=out.with_suffix('.partial.npz');temp=out.with_suffix('.tmp.npz')
            np.savez_compressed(temp,bank=np.stack(bank),iqr=np.stack(iqr),candidate_seconds=np.asarray(times));temp.replace(partial)
            print('CANDIDATES',meta['track'],meta['task_id'],meta['family'],meta['rho'],i+1,flush=True)
    bank=np.stack(bank);iqr=np.stack(iqr);times=np.asarray(times)
    scale=max(float(t.std()),1e-12) if meta['track'].startswith('seven') else meta['scale']
    methods={};predictions={};selections={}
    for count in [40,60,400]:
        if count>len(bank):continue
        keep=count//10;selected=np.argsort(iqr[:count].mean(1),kind='stable')[:keep]
        for name,p in [(f'OligoICP_IQR{count}_keep{keep}_adapted',bank[selected].mean(0)),
                       (f'Bootstrap_mean{count}_extended_budget',bank[:count].mean(0))]:
            methods[name]=metrics(t,p,task,scale)|{'seconds':float(times[:count].sum()),'candidate_evaluations':count}
            predictions['prediction__'+name]=p
        selections[str(count)]=selected.tolist()
    out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp.npz')
    np.savez_compressed(temp,bank=bank,iqr=iqr,candidate_seconds=times,y_target=t,**predictions);temp.replace(out)
    atomic_json(out.with_suffix('.json'),meta|{'status':'COMPLETE','backbone':'tabicl','seed':0,
        'methods':methods,'candidate_pool_size':len(bank),'quantiles':[.15,.85],'retained_fraction':.1,
        'selected_candidates':selections,'first_six_predictions_reused':True,
        'every_candidate_scored':True,'target_labels_used_for_selection':False,
        'adaptations':'TabICL backbone; source/context budgets follow this benchmark; 400-candidate pool matches original count',
        'budget_group':'extended, not six-member matched budget'})
    partial=out.with_suffix('.partial.npz')
    if partial.exists():partial.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--candidates',type=int,choices=[40,60,400],default=400);p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    p.add_argument('--track',choices=['both','five','seven_reimplemented','seven_calibrated','seven_calibrated_v2'],default='both')
    p.add_argument('--condition-sharding',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);predictor=Predictor('regression','cuda:0');failures=0
    tracks=['five','seven_reimplemented'] if args.track=='both' else [args.track]
    jobs=[(track,m) for track in tracks for m in specifications(track) if m['task']=='regression']
    if not args.condition_sharding:jobs=jobs[args.shard::args.shards]
    if args.limit:jobs=jobs[:args.limit]
    for job_index,(track,meta) in enumerate(jobs):
        if track=='five':conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        else:
            from seven_shifts import FAMILIES
            conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        for condition_index,(family,rho) in enumerate(conditions):
            if args.condition_sharding and (job_index*len(conditions)+condition_index)%args.shards!=args.shard:continue
            out=RUN/'results/oligo_extended/tabicl'/track/'regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():
                existing=json.loads(out.with_suffix('.json').read_text())
                if existing.get('candidate_pool_size',0)>=args.candidates and (args.candidates!=60 or 'OligoICP_IQR60_keep6_adapted' in existing['methods']):continue
            print('START',track,meta['task_id'],family,rho,flush=True)
            try:
                data,info=load_condition(meta,family,rho)
                if rho==0 and track=='seven_calibrated_v2':
                    old=RUN/'results/oligo_extended/tabicl/seven_reimplemented/regression'/str(meta['task_id'])/out.name
                    if old.with_suffix('.json').exists():
                        previous=json.loads(old.with_suffix('.json').read_text())
                        if previous.get('candidate_pool_size',0)>=args.candidates:
                            for key in ['X_source','y_source','X_pool','y_pool','X_target','y_target']:
                                assert previous[key+'_sha256']==protocol.array_sha256(data[key]),key
                            out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,out)
                            info['unchanged_clean_predictions_reused_from']=str(old.relative_to(RUN))
                run_one(predictor,data,info,out,args.candidates)
                print('DONE',track,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;row=meta|{'family':family,'rho':rho,'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f"oligo_{track}_{meta['task_id']}_{family}_{rho}.json",row)
                print('FAILED',json.dumps(row),flush=True);torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
