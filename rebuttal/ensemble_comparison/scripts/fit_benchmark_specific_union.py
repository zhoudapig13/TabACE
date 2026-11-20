"""Restore the documented Shared-union-family candidate interface at fixed R=5.

This is a declared reconstruction with ten cached candidates. It does not claim
to restore the paper's uncapped Specific pool or its original member selection.
Selection uses only the authorized benchmark development set, never new queries.
"""
import argparse,json,time
import numpy as np
import torch
from prepare_data import RUN,atomic_json
from fit_shared_reference import load_cells
from fit_combiners import sample_weights
from fit_benchmark_reference import fit,predict,training_loss
from evaluate_benchmark_tuning_holdout import cell
from run_tabicl import metrics


def key(c):return (c['meta']['task_id'],c['meta']['family'] if c['meta']['rho'] else 'clean',c['meta']['rho'])


def select(cells):
    task=cells[0]['meta']['task'];values=[];truth=[]
    for c in cells:
        if task=='classification':values.append(c['bank'][:,np.arange(len(c['y'])),c['y'].astype(int)])
        else:
            scale=max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
            values.append((c['bank']-c['source_mean'])/scale);truth.append((c['y']-c['source_mean'])/scale)
    A=np.concatenate(values,axis=1);w=np.concatenate(sample_weights(cells));y=np.concatenate(truth) if truth else None
    chosen=[];history=[];running=np.zeros(A.shape[1])
    for step in range(5):
        options=[]
        for j in range(10):
            if j in chosen:continue
            p=(running+A[j])/(step+1)
            loss=float((-np.log(np.clip(p,1e-12,1))*w).sum() if task=='classification' else ((p-y)**2*w).sum())
            options.append((loss,j))
        loss,j=min(options);chosen.append(j);running+=A[j];history.append({'selected':j,'loss':loss,'all_candidates':options})
    return chosen,history


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
    p.add_argument('--track',choices=['five','five_calibrated','five_feature_calibrated','seven_reimplemented','seven_calibrated'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    task='regression' if a.track in ['five','seven_calibrated'] else 'classification'
    n=(35 if task=='regression' else 72) if a.track.startswith('five') else (33 if task=='regression' else 35)
    nf=5 if a.track.startswith('five') else 7
    for root,count in [('reference',n*(1+nf*4)),('reference_specific',n*nf*5)]:
        while len(list((RUN/'results'/root/a.backbone/a.track/task).glob('*/*.json')))<count:time.sleep(5)
    shared,_=load_cells(a.backbone,a.track,task);specific,source=load_cells(a.backbone,a.track,task,variant='specific')
    common={key(c):c for c in shared}
    for family in sorted({c['meta']['family'] for c in specific}):
        out=RUN/'results/benchmark_specific_union'/a.backbone/a.track/task/f'specific_{family}.json'
        if out.exists():continue
        merged=[]
        for c in specific:
            if c['meta']['family']!=family:continue
            shared_cell=common[key(c)];assert np.array_equal(shared_cell['y'],c['y'])
            bank=np.concatenate([shared_cell['bank'],c['bank']],axis=0)
            merged.append(c|{'bank':bank})
        selected,selection=select(merged)
        cells=[]
        for c in merged:
            bank=c['bank'][selected]
            values=cell(c['meta'],c['base'],bank,c['y'],c['source_mean'],c['source_scale'])
            if task=='regression':
                eval_scale=max(float(c['y'].std()),1e-12) if a.track.startswith('seven') else c['meta']['scale']
                values.update(Y=(c['y']-c['source_mean'])/c['source_scale'],loss_scale=(c['source_scale']/eval_scale)**2)
            else:
                values.update(true_members=bank[:,np.arange(len(c['y'])),c['y'].astype(int)].T,
                    true_base=c['base'][np.arange(len(c['y'])),c['y'].astype(int)])
            cells.append(c|values)
        best=None;trials=[]
        for gain in [1.,2.,4.]:
            for reg in [.0001,.001,.01]:
                parameters,optimization=fit(cells,task,reg,gain);loss=training_loss(cells,parameters,task,gain)
                trials.append({'gain':gain,'regularization':reg,'loss':loss,'optimization':optimization})
                if best is None or loss<best[0]:best=(loss,gain,reg,parameters,optimization)
        loss,gain,reg,parameters,optimization=best;rows=[]
        for c in cells:
            prediction=predict(c,parameters,task,gain)
            scale=max(float(c['y'].std()),1e-12) if task=='regression' and a.track.startswith('seven') else c['meta']['scale']
            rows.append(c['meta']|{'method':'TabACE_SpecificUnion_benchmark_tuned_R5','benchmark_labels_used_for_fit':True,
                'results_role':'benchmark development','deployed_alternative_branches':5,
                **metrics(c['y'],prediction,task,scale)})
            target=RUN/'results/benchmark_specific_union_predictions'/a.backbone/a.track/task/str(c['meta']['task_id'])/c['path'].name
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=c['y']);tmp.replace(target)
        atomic_json(out,{'status':'COMPLETE','backbone':a.backbone,'track':a.track,'task':task,'family':family,
            'method':'TabACE_SpecificUnion_benchmark_tuned_R5','parameters':parameters.tolist(),'gain':gain,
            'regularization':reg,'optimization':optimization,'trials':trials,'selection':selection,
            'selected_candidate_indices':selected,'candidate_count':10,'deployed_alternative_branches':5,
            'candidate_order':'five Shared roles, followed by five family-specific roles',
            'selection_and_fit_use_benchmark_development_labels':True,'independent_test_score':False,
            'original_uncapped_pool_restored':False,'strength_used_for_inference':False,'rows':rows})
        print('UNION_POLICY_DONE',a.backbone,a.track,family,selected,loss,flush=True)


if __name__=='__main__':main()
