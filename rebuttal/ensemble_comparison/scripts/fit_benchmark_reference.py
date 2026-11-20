"""User-authorized development-set fusion tuning, explicitly not a test score.

Frozen backbone/context predictions are reused. Shared remains family-blind;
Specific fits one policy per family. No strength value enters either mapping.
"""
import argparse,json,time
import numpy as np
import torch
from scipy.optimize import minimize
from prepare_data import RUN,atomic_json
from fit_shared_reference import load_cells,tensor_data
from run_tabicl import normalize,metrics


def fit(cells,task,reg,gain):
    data=tensor_data(cells,task)
    initial=np.r_[np.zeros(5),np.ones(10),.5] if task=='classification' else np.r_[np.zeros(5),np.ones(5),0.,.5]
    bounds=[(-8,8)]*5+([(0,10)]*10+[(0,1)] if task=='classification' else [(0,10)]*5+[(-4,4),(0,1)])
    def objective(values):
        p=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        if task=='classification':
            w=torch.softmax(p[:5]+data['Q']*p[5:10]+data['S']*p[10:15],dim=1)
            mixed=(w*data['true_members']).sum(1);support=(w*data['Q']*data['S']).sum(1)
            alpha=(gain*p[-1]*data['U']*support).clamp(0,1)
            prediction=(1-alpha)*data['true_base']+alpha*mixed
            losses=-torch.log(prediction.clamp(1e-12,1));scores=p[5:15]
        else:
            tau=torch.exp(p[-2]);s=torch.exp(-data['D']/tau)
            w=torch.softmax(p[:5]+s*p[5:10],dim=1);mixed=(w*data['M']).sum(1)
            alpha=(gain*p[-1]*(1-torch.exp(-torch.abs(mixed-data['B'])/tau))*(w*s).sum(1)).clamp(0,1)
            prediction=(1-alpha)*data['B']+alpha*mixed;losses=(prediction-data['Y'])**2;scores=p[5:10]
        centered=p[:5]-p[:5].mean()
        loss=(losses*data['weights']).sum()+reg*((centered**2).mean()+((scores-1)**2).mean())
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    chunks=[];current=initial
    for budget in [120,360]:
        result=minimize(objective,current,jac=True,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':budget,'ftol':1e-10,'gtol':1e-7})
        chunks.append({'iterations':int(result.nit),'success':bool(result.success),'loss':float(result.fun)})
        if result.success:break
        current=result.x
    assert np.isfinite(result.x).all()
    return result.x,{'optimizer_success':bool(result.success),'chunks':chunks,'regularized_training_loss':float(result.fun)}


def predict(c,parameters,task,gain):
    if task=='classification':
        z=parameters[:5]+c['Q']*parameters[5:10]+c['S']*parameters[10:15]
        w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        mixed=np.einsum('nm,mnc->nc',w,c['bank'])
        alpha=np.clip(gain*parameters[-1]*c['U']*(w*c['Q']*c['S']).sum(1),0,1)
        return normalize((1-alpha[:,None])*c['base']+alpha[:,None]*mixed)
    tau=np.exp(parameters[-2]);s=np.exp(-c['D']/tau);z=parameters[:5]+s*parameters[5:10]
    w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True);mixed=(w*c['M']).sum(1)
    alpha=np.clip(gain*parameters[-1]*(1-np.exp(-np.abs(mixed-c['B'])/tau))*(w*s).sum(1),0,1)
    return ((1-alpha)*c['B']+alpha*mixed)*c['source_scale']+c['source_mean']


def training_loss(cells,parameters,task,gain):
    from fit_combiners import sample_weights
    total=0.
    for c,w in zip(cells,sample_weights(cells)):
        p=predict(c,parameters,task,gain)
        if task=='classification':loss=-np.log(np.clip(p[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1))
        else:
            scale=max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
            loss=((p-c['y'])/scale)**2
        total+=float(w@loss)
    return total


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
    parser.add_argument('--track',choices=['five_calibrated','five_feature_calibrated','seven_calibrated','five','seven_reimplemented'],required=True)
    parser.add_argument('--variant',choices=['shared','specific'],required=True);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    task='classification' if args.track in ['five_calibrated','five_feature_calibrated'] else 'regression' if args.track in ['seven_calibrated','five'] else 'classification'
    n=(72 if task=='classification' else 35) if args.track.startswith('five') else (35 if task=='classification' else 33)
    families=5 if args.track.startswith('five') else 7
    expected=n*(families*5 if args.variant=='specific' else 1+families*4)
    root=RUN/'results'/('reference' if args.variant=='shared' else 'reference_specific')/args.backbone/args.track/task
    while len(list(root.glob('*/*.json')))<expected:
        if not args.wait:raise RuntimeError('incomplete context bank')
        time.sleep(5)
    all_cells,source=load_cells(args.backbone,args.track,task,variant=args.variant)
    assert len(all_cells)==expected
    groups={'all':all_cells} if args.variant=='shared' else {f:[c for c in all_cells if c['meta']['family']==f] for f in sorted({c['meta']['family'] for c in all_cells})}
    for family,cells in groups.items():
        out=RUN/'results/benchmark_tuning'/args.backbone/args.track/task/f'{args.variant}_{family}.json'
        if out.exists():continue
        trials=[];best=None
        for gain in [1.,2.,4.]:
            for regularization in [.0001,.001,.01]:
                parameters,optimization=fit(cells,task,regularization,gain)
                loss=training_loss(cells,parameters,task,gain)
                trials.append({'gain':gain,'regularization':regularization,'training_loss':loss,'optimization':optimization})
                if best is None or loss<best[0]:best=(loss,gain,regularization,parameters,optimization)
                print('BENCHMARK_FIT',args.backbone,args.track,args.variant,family,gain,regularization,loss,flush=True)
        loss,gain,regularization,parameters,optimization=best;rows=[]
        for c in cells:
            prediction=predict(c,parameters,task,gain)
            scale=max(float(c['y'].std()),1e-12) if args.track.startswith('seven') and task=='regression' else c['meta']['scale']
            method=f'TabACE_{args.variant.capitalize()}_benchmark_tuned_R5'
            row=c['meta']|{'backbone':args.backbone,'method':method,'results_role':'benchmark development (training-query scores)',
                'benchmark_labels_used_for_fit':True,'weights_training':'all benchmark development queries; not OOF test scores',
                **metrics(c['y'],prediction,task,scale)};rows.append(row)
            target=RUN/'results/benchmark_tuned_predictions'/args.backbone/args.track/task/args.variant/str(c['meta']['task_id'])/c['path'].name
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp.npz')
            np.savez_compressed(temp,prediction=prediction,y_target=c['y']);temp.replace(target)
        atomic_json(out,{'status':'COMPLETE','backbone':args.backbone,'track':args.track,'task':task,'variant':args.variant,
            'family':family,'parameters':parameters.tolist(),'gain':gain,'regularization':regularization,
            'optimization':optimization,'trials':trials,'rows':rows,'source_fold_mapping':source,
            'fitting_data_ids':sorted({int(c['meta']['data_id']) for c in cells}),
            'benchmark_labels_used_for_fit':True,'independent_test_score':False,
            'family_identifier_visible':args.variant=='specific','strength_used_for_inference':False,
            'parameter_choice':'minimum prediction loss over nine predeclared gain/regularization combinations, no score rewriting'})
        print('BENCHMARK_POLICY_DONE',args.backbone,args.track,args.variant,family,loss,flush=True)


if __name__=='__main__':main()
