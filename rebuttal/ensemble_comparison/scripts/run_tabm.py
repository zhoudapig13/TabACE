"""Official TabM architecture, source-only early stopping, full-source refit."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import tabm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabicl import metrics,normalize

CONFIG={'architecture':'official_tabm_0.0.3_make_defaults','k':32,'d_block':512,'n_blocks':3,
        'dropout':.1,'optimizer':'AdamW','learning_rate':.002,'weight_decay':.0003,
        'batch_size':256,'max_epochs':256,'patience':16,'validation_fraction':.25,
        'refit_all_source':True,'seed':0,'input_preprocessing':'source-fit sklearn StandardScaler; constant scale=1',
        'preprocessing_revision':2}


def make_model(d,classes,task):
    torch.manual_seed(0)
    return tabm.TabM.make(n_num_features=d,d_out=classes if task=='classification' else 1).cuda()


def loss(model,X,y,task):
    p=model(X)
    if task=='classification':return F.cross_entropy(p.flatten(0,1),y.repeat_interleave(p.shape[1]))
    return torch.mean((p.squeeze(-1)-y[:,None])**2)


@torch.no_grad()
def predict(model,X,task):
    model.eval();out=[]
    for batch in X.split(256):
        value=model(batch)
        out.append(torch.softmax(value,-1).mean(1) if task=='classification' else value.squeeze(-1).mean(1))
    return torch.cat(out)


def train_epoch(model,optimizer,X,y,task,generator):
    model.train();order=torch.randperm(len(y),generator=generator,device='cuda')
    for idx in order.split(256):
        optimizer.zero_grad(set_to_none=True);value=loss(model,X[idx],y[idx],task)
        if not torch.isfinite(value):raise ValueError('nonfinite training loss')
        value.backward();optimizer.step()


def run_one(data,meta,out):
    task=meta['task'];X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
    classes=meta['n_classes'];start=time.perf_counter()
    if len(np.unique(y))==1:
        prediction=(np.eye(classes)[np.full(len(Q),int(y[0]))] if task=='classification' else np.full(len(Q),float(y[0])))
        training={'constant_source':True,'best_epoch':0,'validation_curve':[],'training_seconds':0.}
    else:
        # Split positions depend on the clean source, never on perturbation strength.
        if meta['track'].startswith('five'):
            with np.load(RUN/'cache/five'/task/f"{meta['task_id']}.npz") as f:clean_y=f['y_source']
        else:
            with np.load(RUN/'cache/seven_reimplemented'/task/str(meta['task_id'])/'covariate_0.npz') as f:clean_y=f['y_source']
        stratify=clean_y if task=='classification' and np.unique(clean_y,return_counts=True)[1].min()>=2 else None
        train,val=train_test_split(np.arange(len(y)),test_size=.25,random_state=0,stratify=stratify)
        scaler=StandardScaler().fit(np.asarray(X[train],dtype=np.float64))
        A=torch.as_tensor(scaler.transform(np.asarray(X,dtype=np.float64)),dtype=torch.float32,device='cuda')
        mu=float(y[train].mean()) if task=='regression' else 0.
        sigma=max(float(y[train].std()),1e-9) if task=='regression' else 1.
        targets=torch.as_tensor((y-mu)/sigma if task=='regression' else y,
            dtype=torch.float32 if task=='regression' else torch.long,device='cuda')
        model=make_model(X.shape[1],classes,task)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
        generator=torch.Generator(device='cuda').manual_seed(0)
        best=np.inf;best_epoch=0;curve=[]
        for epoch in range(256):
            train_epoch(model,optimizer,A[train],targets[train],task,generator)
            v=predict(model,A[val],task)
            value=float((-torch.log(v[torch.arange(len(val),device='cuda'),targets[val]].clamp_min(1e-12))).mean()
                if task=='classification' else torch.mean((v-targets[val])**2))
            curve.append(value)
            if value<best-1e-7:best=value;best_epoch=epoch
            if epoch-best_epoch>=16:break
        del optimizer,model,A,targets
        # Refit the selected epoch budget on every available labeled source row.
        scaler=StandardScaler().fit(np.asarray(X,dtype=np.float64))
        A=torch.as_tensor(scaler.transform(np.asarray(X,dtype=np.float64)),dtype=torch.float32,device='cuda')
        query=torch.as_tensor(scaler.transform(np.asarray(Q,dtype=np.float64)),dtype=torch.float32,device='cuda')
        mu=float(y.mean()) if task=='regression' else 0.;sigma=max(float(y.std()),1e-9) if task=='regression' else 1.
        targets=torch.as_tensor((y-mu)/sigma if task=='regression' else y,
            dtype=torch.float32 if task=='regression' else torch.long,device='cuda')
        model=make_model(X.shape[1],classes,task)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
        generator=torch.Generator(device='cuda').manual_seed(0)
        for _ in range(best_epoch+1):train_epoch(model,optimizer,A,targets,task,generator)
        prediction=predict(model,query,task).cpu().numpy()*sigma+mu
        training={'constant_source':False,'best_epoch':best_epoch+1,'validation_loss':best,
            'validation_curve':curve,'validation_indices_sha256':protocol.array_sha256(val),
            'source_rows_after_refit':len(y),'target_labels_used_for_fit':False}
        del model,optimizer,A,query,targets
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    if task=='classification':prediction=normalize(prediction)
    evaluation_scale=max(float(t.std()),1e-12) if task=='regression' and meta['track'].startswith('seven') else meta['scale']
    observed=metrics(t,prediction,task,evaluation_scale)|{'seconds':seconds}
    if not np.isfinite(prediction).all():raise ValueError('nonfinite TabM output')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,prediction__TabM_official_source_refit=prediction,y_target=t);tmp.replace(out)
    atomic_json(out.with_suffix('.json'),meta|{'status':'COMPLETE','backbone':'tabm','seed':0,
        'methods':{'TabM_official_source_refit':observed},'configuration':CONFIG,'training':training,
        'adaptation_of_pretrained_backbone':False,'target_labels_used_for_fit':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    p.add_argument('--track',choices=['both','all','five','five_corrected','five_calibrated','seven_calibrated','seven_calibrated_v2','seven_reimplemented'],default='both')
    p.add_argument('--task',choices=['both','classification','regression'],default='both');args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    tracks=['five','seven_reimplemented'] if args.track=='both' else ['five','five_corrected','five_calibrated','seven_calibrated','seven_calibrated_v2','seven_reimplemented'] if args.track=='all' else [args.track]
    jobs=[(track,m) for track in tracks for m in specifications(track) if args.task in ['both',m['task']]]
    if args.limit:jobs=jobs[:args.limit]
    failures=0
    for job_index,(track,meta) in enumerate(jobs):
        if track.startswith('five'):conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        else:
            from seven_shifts import FAMILIES
            conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        for condition_index,(family,rho) in enumerate(conditions):
            if (job_index*len(conditions)+condition_index)%args.shards!=args.shard:continue
            out=RUN/'results/tabm'/track/meta['task']/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            print('START',track,meta['task'],meta['task_id'],family,rho,flush=True)
            try:
                data,info=load_condition(meta,family,rho);run_one(data,info,out)
                print('DONE',track,meta['task'],meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;row=meta|{'family':family,'rho':rho,'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f"tabm_{track}_{meta['task']}_{meta['task_id']}_{family}_{rho}.json",row)
                print('FAILED',json.dumps(row),flush=True)
            torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
