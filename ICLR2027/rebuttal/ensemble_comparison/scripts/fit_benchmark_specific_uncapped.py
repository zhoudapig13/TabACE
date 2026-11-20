"""Fuse every available Shared and family-specific context with no R cap.

The current reconstructed bank has five Shared and five directed roles. The
parameterization derives its dimension from the bank and accepts any additional
registered roles. This is not a claim to recover the archived uncapped pool.
"""
import argparse,json,time
import numpy as np
import torch
from scipy.optimize import minimize
from prepare_data import RUN,atomic_json
from fit_shared_reference import load_cells,tensor_data
from fit_combiners import sample_weights
from fit_benchmark_reference import training_loss
from evaluate_benchmark_tuning_holdout import cell
from run_tabicl import metrics,normalize


def predict(c,parameters,task,gain):
    n=c['bank'].shape[0]
    if task=='classification':
        z=parameters[:n]+c['Q']*parameters[n:2*n]+c['S']*parameters[2*n:3*n]
        w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        mixed=np.einsum('nm,mnc->nc',w,c['bank'])
        alpha=np.clip(gain*parameters[-1]*c['U']*(w*c['Q']*c['S']).sum(1),0,1)
        return normalize((1-alpha[:,None])*c['base']+alpha[:,None]*mixed)
    tau=np.exp(parameters[-2]);s=np.exp(-c['D']/tau)
    z=parameters[:n]+s*parameters[n:2*n];w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    mixed=(w*c['M']).sum(1)
    alpha=np.clip(gain*parameters[-1]*(1-np.exp(-np.abs(mixed-c['B'])/tau))*(w*s).sum(1),0,1)
    return ((1-alpha)*c['B']+alpha*mixed)*c['source_scale']+c['source_mean']


def fit(cells,task,regularization,gain):
    n=cells[0]['bank'].shape[0];assert all(c['bank'].shape[0]==n for c in cells)
    data=tensor_data(cells,task)
    initial=np.r_[np.zeros(n),np.ones(2*n),.5] if task=='classification' else np.r_[np.zeros(n),np.ones(n),0.,.5]
    bounds=[(-8,8)]*n+([(0,10)]*(2*n)+[(0,1)] if task=='classification' else [(0,10)]*n+[(-4,4),(0,1)])
    def objective(values):
        p=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        if task=='classification':
            w=torch.softmax(p[:n]+data['Q']*p[n:2*n]+data['S']*p[2*n:3*n],dim=1)
            mixed=(w*data['true_members']).sum(1)
            alpha=(gain*p[-1]*data['U']*(w*data['Q']*data['S']).sum(1)).clamp(0,1)
            prediction=(1-alpha)*data['true_base']+alpha*mixed
            losses=-torch.log(prediction.clamp(1e-12,1));scores=p[n:3*n]
        else:
            tau=torch.exp(p[-2]);s=torch.exp(-data['D']/tau)
            w=torch.softmax(p[:n]+s*p[n:2*n],dim=1);mixed=(w*data['M']).sum(1)
            alpha=(gain*p[-1]*(1-torch.exp(-torch.abs(mixed-data['B'])/tau))*(w*s).sum(1)).clamp(0,1)
            losses=((1-alpha)*data['B']+alpha*mixed-data['Y'])**2;scores=p[n:2*n]
        loss=(losses*data['weights']).sum()+regularization*((p[:n]-p[:n].mean()).square().mean()+(scores-1).square().mean())
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    current=initial;chunks=[]
    for budget in [120,360,720]:
        result=minimize(objective,current,jac=True,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':budget,'ftol':1e-10,'gtol':1e-7})
        chunks.append({'iterations':int(result.nit),'success':bool(result.success),'loss':float(result.fun)})
        if result.success:break
        current=result.x
    return result.x,{'optimizer_success':bool(result.success),'chunks':chunks}


def loss(cells,parameters,task,gain):
    total=0.
    for c,w in zip(cells,sample_weights(cells)):
        prediction=predict(c,parameters,task,gain)
        if task=='classification':l=-np.log(np.clip(prediction[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1))
        else:
            scale=max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
            l=((prediction-c['y'])/scale)**2
        total+=float(w@l)
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
    p.add_argument('--track',choices=['five','five_calibrated','five_feature_calibrated','seven_reimplemented','seven_calibrated'],required=True)
    p.add_argument('--pool',choices=['available','registered'],default='available')
    a=p.parse_args();torch.set_num_threads(1)
    task='regression' if a.track in ['five','seven_calibrated'] else 'classification'
    n=(35 if task=='regression' else 72) if a.track.startswith('five') else (33 if task=='regression' else 35)
    nf=5 if a.track.startswith('five') else 7
    requirements=[('reference_specific_full',n*nf*5)] if a.pool=='registered' else [('reference',n*(1+nf*4)),('reference_specific',n*nf*5)]
    for root,count in requirements:
        while len(list((RUN/'results'/root/a.backbone/a.track/task).glob('*/*.json')))<count:time.sleep(5)
    shared,_=load_cells(a.backbone,a.track,task)
    specific,_=load_cells(a.backbone,a.track,task,variant='specific_full' if a.pool=='registered' else 'specific')
    def key(c):return (c['meta']['task_id'],c['meta']['family'] if c['meta']['rho'] else 'clean',c['meta']['rho'])
    common={key(c):c for c in shared}
    for family in sorted({c['meta']['family'] for c in specific}):
        out=RUN/('results/benchmark_specific_uncapped_full' if a.pool=='registered' else 'results/benchmark_specific_uncapped')/a.backbone/a.track/task/f'specific_{family}.json'
        if out.exists():continue
        cells=[]
        for c in specific:
            if c['meta']['family']!=family:continue
            s=common[key(c)];assert np.array_equal(s['y'],c['y'])
            bank=c['bank'] if a.pool=='registered' else np.concatenate([s['bank'],c['bank']],axis=0)
            v=cell(c['meta'],c['base'],bank,c['y'],c['source_mean'],c['source_scale'])
            if task=='regression':
                eval_scale=max(float(c['y'].std()),1e-12) if a.track.startswith('seven') else c['meta']['scale']
                v.update(Y=(c['y']-c['source_mean'])/c['source_scale'],loss_scale=(c['source_scale']/eval_scale)**2)
            else:v.update(true_members=bank[:,np.arange(len(c['y'])),c['y'].astype(int)].T,
                true_base=c['base'][np.arange(len(c['y'])),c['y'].astype(int)])
            cells.append(c|v)
        best=None;trials=[]
        for gain in [1.,2.,4.]:
            for reg in [.0001,.001,.01]:
                parameters,optimization=fit(cells,task,reg,gain);observed=loss(cells,parameters,task,gain)
                trials.append({'gain':gain,'regularization':reg,'training_loss':observed,'optimization':optimization})
                if best is None or observed<best[0]:best=(observed,gain,reg,parameters,optimization)
        observed,gain,reg,parameters,optimization=best;rows=[]
        for c in cells:
            prediction=predict(c,parameters,task,gain)
            scale=max(float(c['y'].std()),1e-12) if task=='regression' and a.track.startswith('seven') else c['meta']['scale']
            rows.append(c['meta']|{'method':'TabACE_Specific_benchmark_tuned_uncapped_available_pool',
                'benchmark_labels_used_for_fit':True,'results_role':'benchmark development',**metrics(c['y'],prediction,task,scale)})
            target=RUN/('results/benchmark_specific_uncapped_full_predictions' if a.pool=='registered' else 'results/benchmark_specific_uncapped_predictions')/a.backbone/a.track/task/str(c['meta']['task_id'])/c['path'].name
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=c['y']);tmp.replace(target)
        atomic_json(out,{'status':'COMPLETE','backbone':a.backbone,'track':a.track,'task':task,'family':family,
            'parameters':parameters.tolist(),'gain':gain,'regularization':reg,'optimization':optimization,'trials':trials,
            'context_cap':None,'candidate_count':cells[0]['bank'].shape[0],'deployed_alternative_branches':cells[0]['bank'].shape[0],
            'candidate_roles':c['meta']['context_roles'] if a.pool=='registered' else s['meta']['context_roles']+c['meta']['context_roles'],
            'candidate_pool':a.pool+'; all materialized Shared plus family-directed contexts',
            'full_original_registered_ranges_restored':False,'benchmark_labels_used_for_fit':True,
            'independent_test_score':False,'strength_used_for_inference':False,'rows':rows})
        print('UNCAPPED_POLICY_DONE',a.backbone,a.track,family,cells[0]['bank'].shape[0],observed,flush=True)


if __name__=='__main__':main()
