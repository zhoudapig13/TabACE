"""Nested dataset-OOF regression adaptations on a frozen internal benchmark."""
from __future__ import annotations
import argparse
import json
import numpy as np
from prepare_data import RUN,atomic_json
from internal_regression_benchmark import OUT
from fit_combiners import fold_mapping,sample_weights
from fit_regression_expansion_static import train as static_train
from run_tabicl import metrics

METHODS=['Native6','GES6','ConvexStacking6','RidgeStacking6','Median6','FeatureMean6','TrimmedMean6','IQR6_keep1_adapted']
ALPHAS=[0.,.1,.25,.5,.75,1.]
PRIOR=np.array([1.,0,0,0,0,0])


def load(track):
    mapping,source=fold_mapping('regression','five' if track=='five' else 'seven_calibrated_v2')
    files=sorted((OUT/'ensemble_banks'/track).glob('*/*.npz'))
    assert len(files)==(735 if track=='five' else 957),(track,len(files))
    cells=[]
    for p in files:
        meta=json.loads(p.with_suffix('.json').read_text())
        with np.load(p) as z:
            c={'meta':meta,'path':p,'base':z['prediction__Base'],'native_raw':z['prediction__Native6_raw'],
               'native_quantile':z['prediction__Native6_source_quantiles'],'feature':z['feature_bank'],'bootstrap':z['iqr_bank'],
               'iqr':z['iqr'],'y':z['y_target'],'mu':float(z['source_mean']),'sigma':float(z['source_scale']),
               'fold':mapping[int(meta['data_id'])],
               'clean_weight':(5. if track=='five' else 7.) if meta['rho']==0 else 1.}
        assert np.array_equal(c['feature'][0],c['base']) and np.array_equal(c['bootstrap'][0],c['base'])
        cells.append(c)
    return cells,source


def train(cells,method,bank,param):
    if method=='Native6':return {'variant':param,'bank':'native'}
    if method in ['TrimmedMean6','IQR6_keep1_adapted']:
        return {'alpha':param,'bank':bank}
    model=static_train([c|{'bank':c[bank]} for c in cells],method,param)
    return model|{'bank':bank}


def prediction(c,method,model):
    if method=='Native6':return c['native_raw'] if model['variant']=='raw' else c['native_quantile']
    bank=c[model['bank']]
    if 'weights' in model:return np.asarray(model['weights'])@bank
    if method=='Median6':raw=np.median(bank,axis=0)
    elif method=='FeatureMean6':raw=bank.mean(0)
    elif method=='TrimmedMean6':raw=np.sort(bank,axis=0)[1:-1].mean(0)
    elif method=='IQR6_keep1_adapted':raw=bank[int(np.argmin(c['iqr'].mean(1)))]
    else:raise ValueError(method)
    return (1-model['alpha'])*bank[0]+model['alpha']*raw


def clean(cells,method=None,model=None):
    values=[]
    for c in cells:
        if c['meta']['rho']==0:
            values.append(metrics(c['y'],c['base'] if method is None else prediction(c,method,model),
                                  'regression',c['meta']['scale']))
    return {k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']}


def loss(cells,method,model):
    return sum(float(w@(((prediction(c,method,model)-c['y'])/c['meta']['scale'])**2))
               for c,w in zip(cells,sample_weights(cells)))


def feasible(observed,base,tolerance=.005):
    return observed['r2']>=base['r2']-tolerance and observed['nrmse']<=base['nrmse']+tolerance


def fit(track,fold):
    required=['frozen_clean.json','frozen_five.json'] if track=='five' else ['frozen_seven.json']
    frozen={p:(OUT/p).read_bytes() for p in required}
    cells,fold_source=load(track);training=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
    train_ids={int(c['meta']['data_id']) for c in training};test_ids={int(c['meta']['data_id']) for c in test}
    assert train_ids.isdisjoint(test_ids)
    inner=min(c['fold'] for c in training);tr=[c for c in training if c['fold']!=inner];val=[c for c in training if c['fold']==inner]
    base_clean=clean(val);calibration_clean=clean(training)
    for method in METHODS:
        out=OUT/'policies'/track/f'fold_{fold}_{method}.json'
        if out.exists():continue
        if method=='Native6':options=[('native',v) for v in ['raw','quantile']]
        elif method=='IQR6_keep1_adapted':options=[('bootstrap',a) for a in ALPHAS]
        elif method=='ConvexStacking6':options=[(b,p) for b in ['feature','bootstrap'] for p in [0.,.001,.01,.1,1.,10.]]
        elif method=='RidgeStacking6':options=[(b,p) for b in ['feature','bootstrap'] for p in [.001,.01,.1,1.,10.,100.]]
        else:options=[(b,p) for b in ['feature','bootstrap'] for p in ALPHAS]
        trials=[]
        for bank,param in options:
            try:
                model=train(tr,method,bank,param);value=loss(val,method,model);observed=clean(val,method,model)
                trials.append({'bank':bank,'parameter':param,'loss':value,'Clean':observed,
                               'Clean_feasible':feasible(observed,base_clean)})
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append({'bank':bank,'parameter':param,'invalid':repr(exc),'Clean_feasible':False})
        valid=[r for r in trials if 'loss' in r];assert valid
        ok=[r for r in valid if r['Clean_feasible']]
        selected=min(ok or valid,key=lambda r:r['loss'])
        model=train(training,method,selected['bank'],selected['parameter'])
        refit_trials=[]
        if method!='Native6':
            best=None
            for eta in ALPHAS:
                shrunk=model.copy()
                if 'weights' in model:shrunk['weights']=((1-eta)*PRIOR+eta*np.asarray(model['weights'])).tolist()
                else:shrunk['alpha']=eta*model['alpha']
                observed=clean(training,method,shrunk);value=loss(training,method,shrunk)
                legal=feasible(observed,calibration_clean,.0025)
                refit_trials.append({'eta':eta,'Clean':observed,'loss':value,'Clean_feasible':legal})
                if legal and (best is None or value<best[0]):best=(value,shrunk)
            assert best is not None
            model=best[1]
        if method=='Native6':raw_model={'variant':'raw','bank':'native'}
        else:
            raw_param=0. if method=='ConvexStacking6' else .01 if method=='RidgeStacking6' else 1.
            raw_model=train(training,method,'bootstrap' if method=='IQR6_keep1_adapted' else 'feature',raw_param)
        rows=[];raw_rows=[]
        for c in test:
            summary={k:c['meta'][k] for k in ['task_id','data_id','dataset','family','rho']}
            for stage,m,values in [('adapted',model,rows),('raw',raw_model,raw_rows)]:
                p=prediction(c,method,m);assert np.isfinite(p).all()
                row=summary|{'method':method,'stage':stage,'fold':fold}|metrics(c['y'],p,'regression',c['meta']['scale']);values.append(row)
                dest=OUT/'predictions'/track/(method+'_'+stage)/str(c['meta']['task_id'])/f"{c['meta']['family']}_{c['meta']['rho']}.npz"
                dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,prediction=p,y_target=c['y']);tmp.replace(dest)
        atomic_json(out,{'status':'COMPLETE','method':method,'track':track,'fold':fold,'seed':0,
            'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'inner_validation_fold':inner,
            'fold_map_source':fold_source,'model':model,'raw_model':raw_model,'trials':trials,
            'refit_Clean_trials':refit_trials,'inner_Clean_feasible':bool(ok),'context_count':6,
            'Clean_tolerance':.005,'refit_calibration_Clean_margin':.0025,
            'test_labels_used_for_method_fitting':False,'historical_Shared_bounds_used_for_selection':False,
            'benchmark_changed_during_method_fitting':False,'internal_only':True,'Shared_run':False,
            'rows':rows,'raw_rows':raw_rows})
        assert all((OUT/p).read_bytes()==old for p,old in frozen.items())
        print('METHOD_DONE',track,fold,method,selected['bank'],selected['parameter'],bool(ok),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five','seven'],required=True);p.add_argument('--fold',type=int,required=True)
    a=p.parse_args();assert a.fold in range(5);fit(a.track,a.fold)
