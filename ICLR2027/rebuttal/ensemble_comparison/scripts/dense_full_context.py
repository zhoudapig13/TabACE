"""DENSE aggregation at the full six-context budget, with OOF rule selection.

The original disjoint-bucket adaptation remains available. This stronger
adaptation reuses the six full-size bootstrap contexts, so every member has the
same per-context row cap as the other six-member ensemble comparisons.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabicl import normalize,metrics,preserving_bootstrap
from fit_combiners import fold_mapping,sample_weights

NAMES=[f'DENSE_fullcontext_{weight}_{rule}_adapted' for weight in ['uniform','cosine'] for rule in ['MoE','PoE','Max']]


def candidates(bank,X,y,Q,PX,py):
    loc=PX.mean(0);scale=np.maximum(PX.std(0),1e-6)
    Z=(PX-loc)/scale;query=(Q-loc)/scale
    normalized=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    query/=np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
    similarities=[]
    for i in range(6):
        rows=preserving_bootstrap(py,np.random.default_rng(1009*(i+1)),size=len(y))
        similarities.append(np.maximum(query@normalized[rows].mean(0),0)+1e-6)
    weights=np.stack(similarities);weights/=weights.sum(0,keepdims=True)
    outputs=[]
    for w in [np.ones_like(weights)/6,weights]:
        outputs.append(normalize((bank*w[:,:,None]).sum(0)))
        z=(np.log(np.clip(bank,1e-12,1))*w[:,:,None]).sum(0)
        outputs.append(normalize(np.exp(z-z.max(1,keepdims=True))))
        outputs.append(normalize((bank*w[:,:,None]).max(0)))
    return np.stack(outputs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--track',choices=['both','five','five_corrected','five_calibrated','five_feature_calibrated','seven_calibrated','seven_reimplemented'],default='both');args=parser.parse_args()
    tracks=['five','seven_reimplemented'] if args.track=='both' else [args.track]
    for track in tracks:
        task='classification';mapping,source=fold_mapping(task,track);cells=[]
        if track.startswith('five'):conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        else:
            from seven_shifts import FAMILIES
            conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        for meta in specifications(track):
            if meta['task']!=task:continue
            for family,rho in conditions:
                out=RUN/'results/dense_full_context/tabicl'/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
                if not out.with_suffix('.json').exists():
                    arrays,info=load_condition(meta,family,rho)
                    original=RUN/'results/tabicl'/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
                    raw_meta=json.loads(original.with_suffix('.json').read_text())
                    with np.load(original) as f:bank=f['bootstrap_bank'];truth=f['y_target']
                    X,y,Q=(arrays[k] for k in ['X_source','y_source','X_target'])
                    PX=arrays.get('X_pool',X);py=arrays.get('y_pool',y)
                    outputs=candidates(bank,X,y,Q,PX,py)
                    methods={name:metrics(truth,p,task,1.)|{'seconds':raw_meta['methods']['bootstrap_mean_6']['seconds']} for name,p in zip(NAMES,outputs)}
                    out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp.npz')
                    np.savez_compressed(temp,candidates=outputs,y_target=truth,**{'prediction__'+n:v for n,v in zip(NAMES,outputs)})
                    temp.replace(out)
                    atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':'tabicl','seed':0,'methods':methods,
                        'bootstrap_predictions_reused':True,'members':6,'rows_per_context':len(y),
                        'adaptation':'DENSE probability fusion on full-size bootstrap contexts, tabular cosine; no language embedding model',
                        'target_labels_used_for_prediction':False})
                info=json.loads(out.with_suffix('.json').read_text())
                with np.load(out) as f:outputs=f['candidates'];truth=f['y_target']
                cells.append({'meta':info,'path':out,'outputs':outputs,'y':truth,
                    'fold':mapping[int(meta['data_id'])],'clean_weight':(5. if track.startswith('five') else 7.) if rho==0 else 1.})
            print('DENSE_FULL_READY',track,meta['task_id'],flush=True)
        for fold in range(5):
            out=RUN/'results/combiners'/track/task/f'fold_{fold}_dense_full_context.json'
            if out.exists():continue
            train=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
            train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test};assert train_ids.isdisjoint(test_ids)
            losses=np.zeros(6)
            for c,w in zip(train,sample_weights(train)):
                true=c['outputs'][:,np.arange(len(c['y'])),c['y'].astype(int)]
                losses+=(-np.log(np.clip(true,1e-12,1))*w[None]).sum(1)
            selected=int(np.argmin(losses));rows=[]
            for c in test:
                observed=metrics(c['y'],c['outputs'][selected],task,1.)
                rows.append(c['meta']|{'method':'DENSE_fullcontext_OOF_selected_adapted','backbone':'tabicl',
                    'outer_fold':fold,'selected_rule':NAMES[selected]}|observed)
            atomic_json(out,{'status':'COMPLETE','backbone':'tabicl','track':track,'task':task,'fold':fold,'seed':0,
                'method':'DENSE_fullcontext_OOF_selected_adapted','rules':NAMES,'calibration_losses':losses.tolist(),
                'selected_rule':NAMES[selected],'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
                'held_out_labels_used_for_selection':False,'fold_map_source':source,'rows':rows})
            print('DENSE_FULL_FOLD',track,fold,NAMES[selected],flush=True)


if __name__=='__main__':main()
