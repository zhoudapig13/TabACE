"""Dataset-OOF calibration of the paper's confidence/agreement/Base-gate rule."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize
from fit_combiners import fold_mapping,sample_weights
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics


def load_cells(backbone,track,task,variant='shared'):
    mapping,source=fold_mapping(task,track);cells=[]
    root='reference' if variant=='shared' else 'reference_specific_full' if variant=='specific_full' else 'reference_specific'
    for path in sorted((RUN/'results'/root/backbone/track/task).rglob('*.npz')):
        m=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as f:
            b=f['base'];bank=f['members'];y=f['y_target'];scale=float(f['source_scale']);mu=float(f['source_mean'])
        c={'meta':m,'path':path,'base':b,'bank':bank,'y':y,'source_scale':scale,'source_mean':mu,
           'fold':mapping[int(m['data_id'])],'clean_weight':(5. if track.startswith('five') else 7.) if m['rho']==0 and variant=='shared' else 1.}
        if task=='classification':
            k=b.shape[-1];a=np.clip(bank,1e-12,1);q=1+np.sum(a*np.log(a),axis=2)/np.log(k)
            s=1-.5*np.sum(np.abs(bank-bank.mean(0)[None]),axis=2)
            base=np.clip(b,1e-12,1);u=-np.sum(base*np.log(base),axis=1)/np.log(k)
            c.update(Q=q.T,S=s.T,U=u,true_members=bank[:,np.arange(len(y)),y.astype(int)].T,
                     true_base=b[np.arange(len(y)),y.astype(int)])
        else:
            eval_scale=max(float(y.std()),1e-12) if track.startswith('seven') else m['scale']
            members=(bank-mu)/scale;base=(b-mu)/scale
            c.update(M=members.T,B=base,Y=(y-mu)/scale,
                     D=np.abs(members-np.median(members,axis=0)[None]).T,
                     loss_scale=(scale/eval_scale)**2)
        cells.append(c)
    return cells,source


def tensor_data(cells,task):
    w=np.concatenate(sample_weights(cells))
    if task=='classification':
        names=['Q','S','U','true_members','true_base'];data={k:np.concatenate([c[k] for c in cells]) for k in names}
    else:
        names=['M','B','Y','D'];data={k:np.concatenate([c[k] for c in cells]) for k in names}
        w=w*np.concatenate([np.full(len(c['y']),c['loss_scale']) for c in cells])
    return {k:torch.as_tensor(v,dtype=torch.float64) for k,v in data.items()}|{'weights':torch.as_tensor(w,dtype=torch.float64)}


def mapping(parameters,data,task):
    if task=='classification':
        logits=parameters[:5]+data['Q']*parameters[5:10]+data['S']*parameters[10:15]
        weights=torch.softmax(logits,dim=1)
        aggregate=(weights*data['true_members']).sum(1)
        support=(weights*data['Q']*data['S']).sum(1)
        alpha=parameters[-1]*data['U']*support
        return (1-alpha)*data['true_base']+alpha*aggregate
    tau=torch.exp(parameters[-2])
    agreement=torch.exp(-data['D']/tau)
    weights=torch.softmax(parameters[:5]+agreement*parameters[5:10],dim=1)
    aggregate=(weights*data['M']).sum(1)
    uncertainty=1-torch.exp(-torch.abs(aggregate-data['B'])/tau)
    support=(weights*agreement).sum(1)
    alpha=parameters[-1]*uncertainty*support
    return (1-alpha)*data['B']+alpha*aggregate


def fit(cells,task,regularization,initial_parameters=None):
    data=tensor_data(cells,task)
    initial=np.r_[np.zeros(5),np.ones(10),.5] if task=='classification' else np.r_[np.zeros(5),np.ones(5),0.,.5]
    if initial_parameters is not None:initial=np.asarray(initial_parameters,dtype=np.float64)
    bounds=[(-8,8)]*5+([(0,10)]*10+[(0,1)] if task=='classification' else [(0,10)]*5+[(-4,4),(0,1)])
    def objective(x):
        p=torch.tensor(x,dtype=torch.float64,requires_grad=True);pred=mapping(p,data,task)
        loss=(( -torch.log(torch.clamp(pred,min=1e-12,max=1.)) if task=='classification'
                else (pred-data['Y'])**2)*data['weights']).sum()
        prior=p[:5]-p[:5].mean();scores=p[5:15] if task=='classification' else p[5:10]
        loss=loss+regularization*((prior**2).mean()+((scores-1)**2).mean())
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    chunks=[];current=initial
    for budget in [60,180,360]:
        result=minimize(objective,current,jac=True,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':budget,'ftol':1e-10,'gtol':1e-7})
        chunks.append({'budget':budget,'iterations':int(result.nit),'success':bool(result.success),'loss':float(result.fun)})
        if result.success:break
        current=result.x
    if not np.all(np.isfinite(result.x)):raise ValueError('nonfinite policy')
    return result.x,{'optimizer_success':bool(result.success),'message':str(result.message),
                     'iterations':sum(c['iterations'] for c in chunks),'optimization_chunks':chunks,
                     'calibration_regularized_loss':float(result.fun)}


def loss_on(cells,parameters,task):
    data=tensor_data(cells,task)
    with torch.no_grad():
        pred=mapping(torch.as_tensor(parameters,dtype=torch.float64),data,task)
        losses=-torch.log(pred.clamp(1e-12,1.)) if task=='classification' else (pred-data['Y'])**2
        return float((losses*data['weights']).sum())


def predict(c,parameters,task):
    if task=='classification':
        logits=parameters[:5]+c['Q']*parameters[5:10]+c['S']*parameters[10:15]
        w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        aggregate=np.einsum('nm,mnc->nc',w,c['bank'])
        support=(w*c['Q']*c['S']).sum(1)
        alpha=parameters[-1]*c['U']*support
        return normalize((1-alpha[:,None])*c['base']+alpha[:,None]*aggregate)
    tau=np.exp(parameters[-2]);agreement=np.exp(-c['D']/tau)
    logits=parameters[:5]+agreement*parameters[5:10]
    w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    aggregate=(w*c['M']).sum(1)
    alpha=parameters[-1]*(1-np.exp(-np.abs(aggregate-c['B'])/tau))*(w*agreement).sum(1)
    return ((1-alpha)*c['B']+alpha*aggregate)*c['source_scale']+c['source_mean']


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl')
    p.add_argument('--track',choices=['five','five_corrected','five_calibrated','seven_calibrated','seven_reimplemented'],default='five');p.add_argument('--wait',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    dimensions=[('classification',1512),('regression',735)] if args.track=='five' else [('classification',1015),('regression',957)]
    if args.track=='five_corrected':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track=='five_calibrated':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track=='seven_calibrated':dimensions=[('regression',1155 if 'specific_reference' in __file__ else 957)]
    for task,expected in dimensions:
        while len(list((RUN/'results/reference'/args.backbone/args.track/task).rglob('*.json')))<expected:
            if not args.wait:raise RuntimeError('incomplete reference bank')
            time.sleep(10)
        cells,source=load_cells(args.backbone,args.track,task);assert len(cells)==expected
        for fold in range(5):
            out=RUN/'results/combiners'/args.track/task/f'fold_{fold}_tabace_{args.backbone}_reimplemented.json'
            if out.exists():continue
            train=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
            train_ids={int(c['meta']['data_id']) for c in train};test_ids={int(c['meta']['data_id']) for c in test}
            assert train_ids.isdisjoint(test_ids)
            inner=min(c['fold'] for c in train);inner_train=[c for c in train if c['fold']!=inner];inner_val=[c for c in train if c['fold']==inner]
            candidates=[]
            for regularization in [.0001,.001,.01]:
                parameter,_=fit(inner_train,task,regularization)
                candidates.append((loss_on(inner_val,parameter,task),regularization))
            _,best=min(candidates);parameters,optimization=fit(train,task,best)
            rows=[]
            for c in test:
                prediction=predict(c,parameters,task)
                scale=max(float(c['y'].std()),1e-12) if task=='regression' and args.track.startswith('seven') else c['meta']['scale']
                row=c['meta']|{'method':'TabACE_Shared_R5_reimplemented','backbone':args.backbone,'outer_fold':fold,
                              'seconds':c['meta']['base_seconds']+sum(c['meta']['context_seconds'])+c['meta']['construction_seconds']}
                row.update(metrics(c['y'],prediction,task,scale));rows.append(row)
                target=RUN/'results/reference_predictions'/args.backbone/args.track/task/str(c['meta']['task_id'])/c['path'].name
                target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,prediction=prediction,y_target=c['y']);tmp.replace(target)
            atomic_json(out,{'status':'COMPLETE','backbone':args.backbone,'track':args.track,'task':task,'fold':fold,'seed':0,
                'method':'TabACE_Shared_R5_reimplemented','parameters':parameters.tolist(),'parameter_layout':
                'prior_logits[5],eta_Q[5],eta_S[5],alpha_max' if task=='classification' else 'prior_logits[5],eta_S[5],log_tau_source_scale,alpha_max',
                'regularization':best,'inner_scores':candidates,'optimization':optimization,
                'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'fold_map_source':source,
                'held_out_labels_used_for_fit':False,'rows':rows})
            print('DONE',args.backbone,args.track,task,fold,optimization,flush=True)


if __name__=='__main__':main()
