"""Real updated-checkpoint ensembles; calibration uses clean TRAIN holdouts only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from scipy.optimize import minimize
from prepare_data import RUN,atomic_json,protocol
from internal_regression_benchmark import manifest
from run_internal_regression_ensembles import jobs
from tabicl_regression_comparison_v2 import load_condition
from train_tabicl_robust_regression_v3 import OUT,source_data,augment
from run_tabicl import Predictor,metrics

METHODS=['FT_single','Native6','Random6','GES6','ConvexStacking6','RidgeStacking6','Median6','FeatureMean6','TrimmedMean6','IQR6']


class UpdatedPredictor(Predictor):
    def __init__(self,path):super().__init__('regression','cuda:0');self.path=path
    def __call__(self,X,y,Q,*,members=1,quantiles=False,**kwargs):
        from tabicl import TabICLRegressor
        if members not in self.models:
            self.models[members]=TabICLRegressor(n_estimators=members,device='cuda:0',model_path=str(self.path),
                allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0)
        return super().__call__(X,y,Q,members=members,quantiles=quantiles,**kwargs)


def bank(predictor,X,y,Q):
    from run_feature_views import projections
    single,_=predictor(X,y,Q,quantiles=True);native,_=predictor(X,y,Q,members=6)
    feature=np.stack([single if 'identity' in name else predictor(A,y,T)[0] for name,A,T in projections(X,Q)])
    boot=[];iqr=[]
    for i in range(6):
        ids=np.random.default_rng(1009*(i+1)).choice(len(y),size=len(y),replace=True)
        p,u=predictor(X[ids],y[ids],Q,quantiles=True);boot.append(p);iqr.append(u)
    result={'feature':feature,'bootstrap':np.stack(boot),'iqr':np.stack(iqr),'single':single,'native':native}
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def calibration(m,predictor,tag,checkpoint_sha):
    from seven_shifts import FAMILIES as SEVEN
    source=source_data(m);order=np.random.default_rng(1234).permutation(source['train'])
    ctx=order[:min(256,len(order))];remaining=order[len(ctx):];qry=source['val'][:128];cells=[]
    for track,families in [('five',protocol.FAMILIES),('seven',SEVEN)]:
        for family,rho in [('original' if track=='five' else 'covariate',0)]+[(f,r) for f in families for r in [50,100]]:
            path=OUT/'source_validation_banks'/tag/str(m['task_id'])/f'{track}_{family}_{rho}.npz'
            if not(path.exists() and path.with_suffix('.json').exists()):
                X,y,Q,t=augment(m,source,ctx,qry,remaining,track,family,rho);b=bank(predictor,X,y,Q)
                path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**b,y=t);tmp.replace(path)
                atomic_json(path.with_suffix('.json'),{'track':track,'family':family,'rho':rho,'role':'official TRAIN validation rows',
                    'checkpoint_sha256':checkpoint_sha,'benchmark_query_labels_used':False,
                    'context_row_ids':source['source_indices'][ctx].tolist(),'query_row_ids':source['source_indices'][qry].tolist()})
            meta=json.loads(path.with_suffix('.json').read_text());assert meta['checkpoint_sha256']==checkpoint_sha
            with np.load(path) as z:cell={k:z[k] for k in z.files}
            cells.append(cell|meta)
    return cells


def prediction(c,method,model):
    if method=='FT_single':return c['single']
    if method=='Native6':return c['native']
    if method=='Random6':return c['bootstrap'].mean(0)
    if method=='IQR6':
        ids=np.argsort(c['iqr'].mean(1),kind='stable')[:model['keep']]
        return c['bootstrap'][ids].mean(0)
    values=c[model['bank']]
    if method=='FeatureMean6':return values.mean(0)
    if method=='Median6':return np.median(values,axis=0)
    if method=='TrimmedMean6':return np.sort(values,axis=0)[1:-1].mean(0)
    return np.asarray(model['weights'])@values


def gram(cells,bank_name,half):
    G=np.zeros((6,6));masses=[]
    for c in cells:
        masses.append(.1/2 if c['rho']==0 else .45/(10 if c['track']=='five' else 14))
    for c,mass in zip(cells,masses):
        n=len(c['y']);ids=np.arange(n//2) if half else np.arange(n)
        sd=max(float(c['y'].std()),1e-8);E=(c[bank_name][:,ids]-c['y'][ids])/sd
        G+=mass*(E@E.T/len(ids))
    return G


def fitted_weights(G,kind,lam):
    prior=np.full(6,1/6)
    if kind=='GES6':
        counts=np.zeros(6);best=None
        for i in range(25):
            candidates=(counts[None,:]+np.eye(6))/(i+1);loss=np.einsum('ij,jk,ik->i',candidates,G,candidates)
            k=int(np.argmin(loss));counts[k]+=1;beta=counts/(i+1);value=float(beta@G@beta)
            if best is None or value<best[0]:best=(value,beta.copy())
        return best[1]
    if kind=='RidgeStacking6':
        B=np.vstack([-np.ones(5),np.eye(5)]);z=np.linalg.lstsq(B.T@G@B+lam*(B.T@B),-B.T@G@prior,rcond=1e-12)[0]
        beta=prior+B@z
        if np.linalg.norm(beta)>10:raise ValueError('Unstable ridge weights')
        return beta
    unit=max(1.,float(np.max(abs(G))))
    def fun(w):
        delta=w-prior
        return (float(w@G@w+lam*(delta@delta))/unit,(2*G@w+2*lam*delta)/unit)
    fit=minimize(fun,prior,jac=True,method='SLSQP',bounds=[(0,1)]*6,
        constraints=[{'type':'eq','fun':lambda w:float(w.sum()-1),'jac':lambda w:np.ones(6)}],options={'ftol':1e-12,'maxiter':300})
    if not fit.success:raise RuntimeError(str(fit.message))
    beta=np.clip(fit.x,0,1);return beta/beta.sum()


def score(cells,method,model,half=True):
    gains={'five':[],'seven':[]}
    for c in cells:
        if not c['rho']:continue
        n=len(c['y']);ids=np.arange(n//2,n) if half else np.arange(n)
        p=prediction(c,method,model)[ids];baseline=c['single'][ids];t=c['y'][ids]
        denom=max(float(c['y'].var()),1e-12)
        gains[c['track']].append(float(np.mean((baseline-t)**2)-np.mean((p-t)**2))/denom)
    a=float(np.mean(gains['five']));b=float(np.mean(gains['seven']))
    return {'five_gain_over_FT_single':a,'seven_gain_over_FT_single':b,'selection_score':min(a,b)+.05*(a+b)}


def calibrate(cells):
    models={m:{} for m in ['FT_single','Native6','Random6']};trials={}
    for method in METHODS[3:]:
        candidates=[]
        if method=='IQR6':options=[{'keep':k} for k in [1,3,6]]
        elif method in ['FeatureMean6','Median6','TrimmedMean6']:options=[{'bank':b} for b in ['feature','bootstrap']]
        else:options=[{'bank':b,'regularization':v} for b in ['feature','bootstrap'] for v in ([0.] if method=='GES6' else [.001,.01,.1,1.])]
        for option in options:
            try:
                model=option.copy()
                if method in ['GES6','ConvexStacking6','RidgeStacking6']:
                    model['weights']=fitted_weights(gram(cells,model['bank'],True),method,model['regularization']).tolist()
                candidates.append({'model':model,'score':score(cells,method,model)})
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:candidates.append({'model':option,'invalid':repr(e)})
        valid=[r for r in candidates if 'score' in r];assert valid
        selected=max(valid,key=lambda r:r['score']['selection_score'])['model'].copy()
        if method in ['GES6','ConvexStacking6','RidgeStacking6']:
            selected['weights']=fitted_weights(gram(cells,selected['bank'],False),method,selected['regularization']).tolist()
        models[method]=selected;trials[method]=candidates
    return models,trials


def evaluate(m,tag):
    ck=OUT/'checkpoints'/tag/str(m['task_id']);deadline=time.monotonic()+1800
    while not (ck/'training_result.json').exists():
        if time.monotonic()>deadline:raise TimeoutError(f"Training not complete: {m['dataset']}")
        time.sleep(5)
    training=json.loads((ck/'training_result.json').read_text())
    assert training['actual_backbone_weights_updated']
    path=ck/'best.ckpt';checkpoint_sha=hashlib.sha256(path.read_bytes()).hexdigest();predictor=UpdatedPredictor(path)
    policies=OUT/'combiner_policies'/tag/f"{m['task_id']}.json"
    if policies.exists():
        policy=json.loads(policies.read_text());assert policy['checkpoint_sha256']==checkpoint_sha;models=policy['models']
    else:
        cells=calibration(m,predictor,tag,checkpoint_sha);models,trials=calibrate(cells)
        policy={'status':'COMPLETE','models':models,'calibration_trials':trials,'checkpoint_sha256':checkpoint_sha,
            'role':'clean official TRAIN validation labels, extra supervision authorized','benchmark_query_labels_used':False,
            'original_Base_fallback':False,'backbone_weights_updated':True,'annotation':'†‡§'}
        atomic_json(policies,policy)
    task_jobs=[j for j in jobs() if j[1]['task_id']==m['task_id']]
    for track,meta,family,rho in task_jobs:
        out=OUT/'benchmark_predictions'/tag/track/str(m['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        tick=time.perf_counter();data,info,base=load_condition(track,meta,family,rho)
        X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
        b=bank(predictor,X,y,Q);preds={method:prediction(b,method,models[method]) for method in METHODS}
        observed={method:metrics(t,p,'regression',info['scale']) for method,p in preds.items()}
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**b,y_target=t,original_Base=base,**{'prediction__'+k:v for k,v in preds.items()});tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','track':track,'checkpoint_sha256':checkpoint_sha,
            'checkpoint':str(path.relative_to(RUN)),'methods':observed,'actual_backbone_weights_updated':True,
            'extra_clean_TRAIN_label_supervision':True,'benchmark_query_labels_used_for_training_or_selection':False,
            'candidate_budget':6,'raw_Base_fallback':False,'noisy_canonical_context_labels_preserved_at_inference':True,
            'input_hashes':{k:protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target','y_target']},
            'seconds':time.perf_counter()-tick,'seed':0,'annotation':'†‡§'})
        print('ACTUAL_FT_ENSEMBLE_DONE',track,m['dataset'],family,rho,flush=True)
    print('ACTUAL_FT_DATASET_COMPLETE',m['dataset'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--tag',default='initial128');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    work=manifest()[a.shard::a.shards]
    if a.limit:work=work[:a.limit]
    for m in work:evaluate(m,a.tag)
