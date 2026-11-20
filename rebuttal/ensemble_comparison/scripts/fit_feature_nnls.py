"""OOF NNLS and source-learned probability calibration of six feature views."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from scipy.optimize import nnls
from fit_combiners import fold_mapping,sample_weights
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics


def load_cells(track,task):
    mapping,source=fold_mapping(task,track);cells=[]
    for p in sorted((RUN/'results/feature_views/tabicl'/track/task).rglob('*.npz')):
        m=json.loads(p.with_suffix('.json').read_text())
        with np.load(p) as f:bank=f['bank'];y=f['y_target'];mu=float(f['source_mean'])
        cells.append({'meta':m,'bank':bank,'y':y,'mu':mu,'fold':mapping[int(m['data_id'])],
            'clean_weight':(5. if track.startswith('five') else 7.) if m['rho']==0 else 1.})
    return cells,source


def fit(cells,task):
    matrices=[];targets=[]
    for c,w in zip(cells,sample_weights(cells)):
        if task=='classification':
            k=c['bank'].shape[-1];A=c['bank'].transpose(1,2,0).reshape(-1,6)
            target=np.eye(k)[c['y'].astype(int)].reshape(-1);weight=np.repeat(w,k)
        else:
            scale=max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
            A=(c['bank']-c['mu']).T/scale;target=(c['y']-c['mu'])/scale;weight=w
        root=np.sqrt(weight);matrices.append(A*root[:,None]);targets.append(target*root)
    weights,_=nnls(np.concatenate(matrices),np.concatenate(targets),maxiter=500)
    if task=='classification':weights=weights/max(weights.sum(),1e-12) if weights.sum()>1e-12 else np.ones(6)/6
    if not np.isfinite(weights).all():raise ValueError('nonfinite NNLS')
    return weights


def predict(c,weights,task,temperature=1.):
    if task=='regression':return weights@(c['bank']-c['mu'])+c['mu']
    p=normalize(np.einsum('m,mnc->nc',weights,c['bank']))
    z=np.log(np.clip(p,1e-12,1))/temperature
    return normalize(np.exp(z-z.max(1,keepdims=True)))


def loss(cells,weights,temperature):
    return sum(float(w@(-np.log(np.clip(predict(c,weights,'classification',temperature)[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1))))
               for c,w in zip(cells,sample_weights(cells)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five','five_corrected','five_calibrated','five_feature_calibrated','seven_calibrated','seven_calibrated_v2','seven_reimplemented'],default='five');p.add_argument('--wait',action='store_true');args=p.parse_args()
    dimensions=[('classification',1512),('regression',735)] if args.track=='five' else [('classification',1015),('regression',957)]
    if args.track=='five_corrected':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track in ['five_calibrated','five_feature_calibrated']:dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track in ['seven_calibrated','seven_calibrated_v2']:dimensions=[('regression',1155 if 'specific_reference' in __file__ else 957)]
    for task,expected in dimensions:
        root=RUN/'results/feature_views/tabicl'/args.track/task
        while len(list(root.rglob('*.json')))<expected:
            if not args.wait:raise RuntimeError('incomplete feature-view bank')
            time.sleep(10)
        cells,source=load_cells(args.track,task);assert len(cells)==expected
        for fold in range(5):
            out=RUN/'results/combiners'/args.track/task/f'fold_{fold}_feature_nnls.json'
            if out.exists():continue
            train=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
            train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test};assert train_ids.isdisjoint(test_ids)
            temperature=1.;validation=[]
            if task=='classification':
                inner=min(c['fold'] for c in train)
                inner_train=[c for c in train if c['fold']!=inner];inner_val=[c for c in train if c['fold']==inner]
                weights=fit(inner_train,task)
                validation=[(loss(inner_val,weights,t),t) for t in [.5,.75,1.,1.25,1.5,2.]]
                temperature=min(validation)[1]
            weights=fit(train,task);rows=[]
            for c in test:
                pred=predict(c,weights,task,temperature)
                scale=max(float(c['y'].std()),1e-12) if task=='regression' and args.track.startswith('seven') else c['meta']['scale']
                rows.append(c['meta']|{'method':'TabFM_feature_views6_NNLS_adapted','backbone':'tabicl','outer_fold':fold}|metrics(c['y'],pred,task,scale))
            atomic_json(out,{'status':'COMPLETE','backbone':'tabicl','track':args.track,'task':task,'fold':fold,'seed':0,
                'method':'TabFM_feature_views6_NNLS_adapted','weights':weights.tolist(),'temperature':temperature,
                'inner_validation':validation,'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
                'held_out_labels_used_for_fit':False,'fold_map_source':source,'rows':rows})
            print('DONE',args.track,task,fold,'weights',weights,'temperature',temperature,flush=True)


if __name__=='__main__':main()
