"""Cross-fitted calibration safety gate; preserve every original algorithm.

The gate never reads historical Shared scores. An explicit Base fallback is a
deployment decision, not an ensemble gain or a reproduction of the raw method.
"""
import argparse
import json
import math
import numpy as np
from scipy.stats import norm
from prepare_data import atomic_json
from tabicl_regression_comparison_v2 import OUT
import fit_internal_regression_ensembles as original
from fit_regression_expansion_static import ridge,convex
from run_tabicl import metrics

METHODS=['Native6','GES6','ConvexStacking6','RidgeStacking6','Median6','FeatureMean6','TrimmedMean6','IQR6']
PRIOR=np.array([1.,0,0,0,0,0])
ALPHAS=[.1,.25,.5,.75,1.]


def prepare(cells):
    for c in cells:
        c['target_variance']=max(float(np.var(c['y'])),1e-24)
        c['error_grams']={}
        c['nonlinear_grams']={}
        for bank in ['feature','bootstrap']:
            E=c[bank]-c['y'][None,:]
            c['error_grams'][bank]=E@E.T/len(c['y'])
            for method,raw in [('Median6',np.median(c[bank],axis=0)),
                               ('TrimmedMean6',np.sort(c[bank],axis=0)[1:-1].mean(0))]:
                errors=np.stack([c['base']-c['y'],raw-c['y']])
                c['nonlinear_grams'][(method,bank)]=errors@errors.T/len(c['y'])
        c['base_mse']=float(c['error_grams']['feature'][0,0])
        c['native_mse']={v:float(np.mean((c['native_'+v]-c['y'])**2)) for v in ['raw','quantile']}
        c['iqr_order']=np.argsort(c['iqr'].mean(1),kind='stable')


def options(method):
    if method=='Native6':return [('native',v) for v in ['raw','quantile']]
    if method=='IQR6':return [('bootstrap',{'keep':k,'alpha':a}) for k in [1,3,6] for a in [.1,.25,.5,1.]]
    grid=([0.,.001,.01,.1,1.,10.] if method=='ConvexStacking6' else
          [.001,.01,.1,1.,10.,100.] if method=='RidgeStacking6' else ALPHAS)
    return [(bank,param) for bank in ['feature','bootstrap'] for param in grid]


def gram(cells,bank,cache):
    key=(tuple(sorted({int(c['meta']['data_id']) for c in cells})),bank)
    if key not in cache:
        totals={}
        for c in cells:
            ds=int(c['meta']['data_id']);totals[ds]=totals.get(ds,0.)+c['clean_weight']
        G=np.zeros((6,6))
        for c in cells:
            mass=c['clean_weight']/totals[int(c['meta']['data_id'])]/len(totals)
            G+=mass*c['error_grams'][bank]/c['meta']['scale']**2
        cache[key]=G
    return cache[key]


def ges(G):
    counts=np.zeros(6);best=None
    for i in range(25):
        candidates=(counts[None,:]+np.eye(6))/(i+1)
        losses=np.einsum('ij,jk,ik->i',candidates,G,candidates)
        k=int(np.argmin(losses));counts[k]+=1;beta=counts/(i+1);value=float(beta@G@beta)
        if best is None or value<best[0]:best=(value,beta.copy())
    return best[1]


def train(cells,method,bank,param,cache):
    model={'method':method,'bank':bank,'fallback':False}
    if method=='Native6':return model|{'variant':param}
    if method=='IQR6':return model|param
    if method in ['Median6','FeatureMean6','TrimmedMean6']:return model|{'alpha':param}
    G=gram(cells,bank,cache)
    if method=='GES6':
        key=('GES',tuple(sorted({int(c['meta']['data_id']) for c in cells})),bank)
        if key not in cache:cache[key]=ges(G)
        beta=(1-param)*PRIOR+param*cache[key]
    elif method=='RidgeStacking6':beta=ridge(G,np.zeros(6),param)
    else:beta=convex(G,np.zeros(6),param)
    assert np.isclose(beta.sum(),1.,atol=1e-10)
    return model|{'weights':beta.tolist(),'parameter':param}


def beta(c,model):
    if 'weights' in model:return np.asarray(model['weights'])
    if model['method']=='FeatureMean6':return (1-model['alpha'])*PRIOR+model['alpha']/6
    out=(1-model['alpha'])*PRIOR
    out[c['iqr_order'][:model['keep']]]+=model['alpha']/model['keep']
    return out


def mse(c,model):
    if model['fallback']:return c['base_mse']
    method=model['method']
    if method=='Native6':return c['native_mse'][model['variant']]
    if method in ['Median6','TrimmedMean6']:
        w=np.array([1-model['alpha'],model['alpha']]);return max(0.,float(w@c['nonlinear_grams'][(method,model['bank'])]@w))
    w=beta(c,model);return max(0.,float(w@c['error_grams'][model['bank']]@w))


def predict(c,model):
    if model['fallback']:return c['base'].copy()
    method=model['method']
    if method=='Native6':return c['native_'+model['variant']]
    if method in ['Median6','TrimmedMean6']:
        bank=c[model['bank']]
        raw=np.median(bank,axis=0) if method=='Median6' else np.sort(bank,axis=0)[1:-1].mean(0)
        return (1-model['alpha'])*c['base']+model['alpha']*raw
    return beta(c,model)@c[model['bank']]


def case_scores(c,model):
    value=mse(c,model);base=c['base_mse'];scale=c['meta']['scale']
    return {'data_id':int(c['meta']['data_id']),'rho':c['meta']['rho'],
        'gain_R2':(base-value)/c['target_variance'],
        'gain_nRMSE':(math.sqrt(base)-math.sqrt(value))/scale,
        'risk':.5*value/c['target_variance']+.5*math.sqrt(value)/scale,
        'clean_weight':c['clean_weight']}


def evidence(rows,z):
    shift=[r for r in rows if r['rho']>0];clean=[r for r in rows if r['rho']==0]
    ids=sorted({r['data_id'] for r in shift});result={}
    for metric in ['gain_R2','gain_nRMSE']:
        values=np.array([np.mean([r[metric] for r in shift if r['data_id']==ds]) for ds in ids])
        se=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else float('inf')
        result[metric]={'mean':float(values.mean()),'standard_error':se,'lower_bound':float(values.mean()-z*se)}
    result['Clean_gain_R2']=float(np.mean([r['gain_R2'] for r in clean]))
    result['Clean_gain_nRMSE']=float(np.mean([r['gain_nRMSE'] for r in clean]))
    result['Clean_feasible']=result['Clean_gain_R2']>=-.005 and result['Clean_gain_nRMSE']>=-.005
    result['risk']=float(np.mean([sum(r['risk']*r['clean_weight'] for r in rows if r['data_id']==ds)/
        sum(r['clean_weight'] for r in rows if r['data_id']==ds) for ds in ids]))
    result['passes']=result['Clean_feasible'] and all(result[k]['lower_bound']>1e-12 for k in ['gain_R2','gain_nRMSE'])
    return result


def clean_feasible(cells,model):
    rows=[case_scores(c,model) for c in cells if c['meta']['rho']==0]
    return all(float(np.mean([r[k] for r in rows]))>=-.0025 for k in ['gain_R2','gain_nRMSE'])


def fit(track,fold):
    original.OUT=OUT;cells,fold_source=original.load(track);prepare(cells)
    cal=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
    calids={int(c['meta']['data_id']) for c in cal};testids={int(c['meta']['data_id']) for c in test}
    assert calids.isdisjoint(testids);cache={};inner_folds=sorted({c['fold'] for c in cal})
    frozen={name:(OUT/name).read_bytes() for name in ['frozen_clean.json','frozen_five.json','frozen_seven.json']}
    splits=[]
    for inner in inner_folds:
        tr={int(c['meta']['data_id']) for c in cal if c['fold']!=inner}
        val={int(c['meta']['data_id']) for c in cal if c['fold']==inner}
        assert tr.isdisjoint(val) and (tr|val).isdisjoint(testids)
        splits.append({'validation_fold':inner,'train_data_ids':sorted(tr),'validation_data_ids':sorted(val)})
    for method in METHODS:
        dest=OUT/'safety_policies'/track/f'fold_{fold}_{method}.json'
        if dest.exists():continue
        grid=options(method);z=float(norm.ppf(1-.05/(2*len(grid))));trials=[]
        for bank,param in grid:
            inner_rows=[];models=[]
            try:
                for inner in inner_folds:
                    tr=[c for c in cal if c['fold']!=inner];val=[c for c in cal if c['fold']==inner]
                    model=train(tr,method,bank,param,cache);models.append({'validation_fold':inner,'model':model})
                    inner_rows.extend(case_scores(c,model) for c in val)
                ev=evidence(inner_rows,z)
                trials.append({'bank':bank,'parameter':param,'evidence':ev,'inner_models':models})
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append({'bank':bank,'parameter':param,'invalid':repr(exc)})
        passing=[r for r in trials if r.get('evidence',{}).get('passes',False)]
        selected=min(passing,key=lambda r:r['evidence']['risk']) if passing else None
        model={'method':method,'fallback':True,'reason':'Insufficient calibration evidence for joint R2 and nRMSE gain'}
        refit=[]
        if selected:
            proposed=train(cal,method,selected['bank'],selected['parameter'],cache)
            etas=[1.] if method=='Native6' else [1.,.75,.5,.25,.1,0.]
            for eta in etas:
                candidate=proposed.copy()
                if eta==0.:candidate={'method':method,'fallback':True,'reason':'Refit Clean safeguard'}
                elif 'weights' in candidate:candidate['weights']=((1-eta)*PRIOR+eta*np.asarray(candidate['weights'])).tolist()
                elif 'alpha' in candidate:candidate['alpha']*=eta
                legal=clean_feasible(cal,candidate);refit.append({'eta':eta,'Clean_feasible':legal})
                if legal:model=candidate;break
        rows=[]
        for c in test:
            prediction=predict(c,model);actual=metrics(c['y'],prediction,'regression',c['meta']['scale'])
            # Sufficient-statistic risk must agree with direct prediction error.
            assert np.isclose(mse(c,model),np.mean((prediction-c['y'])**2),rtol=1e-8,atol=1e-8)
            row={k:c['meta'][k] for k in ['task_id','data_id','dataset','family','rho']}
            rows.append(row|actual|{'method':method,'fold':fold,'Base_fallback':model['fallback']})
            path=OUT/'safety_predictions'/track/method/str(c['meta']['task_id'])/c['path'].name
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=c['y']);tmp.replace(path)
        atomic_json(dest,{'status':'COMPLETE','method':method,'track':track,'fold':fold,'seed':0,
            'variant':'cross_fitted_calibration_safety_gate','model':model,'Base_fallback':model['fallback'],
            'calibration_data_ids':sorted(calids),'test_data_ids':sorted(testids),'inner_splits':splits,
            'fold_map_source':fold_source,'z_critical':z,'familywise_calibration_alpha':.05,
            'candidate_count':len(grid),'calibration_trials':trials,'selected':selected,'refit_Clean_checks':refit,
            'rows':rows,'test_labels_used_for_fitting':False,'Shared_bounds_used_for_selection':False,
            'benchmark_changed':False,'backbone_weights_updated':False,'annotation':'†‡',
            'candidate_budget_maximum':6,'deployed_context_budget_maximum':1 if model['fallback'] else 6,
            'nonzero_linear_fusion_weights':int(np.count_nonzero(np.abs(model['weights'])>1e-12)) if 'weights' in model else None,
            'fallback_is_ensemble_gain':False,'original_method_results_preserved':True,
            'calibration_normal_bounds_are_selection_heuristic_not_coverage_guarantee':True,
            'internal_development_only':True})
        assert all((OUT/name).read_bytes()==value for name,value in frozen.items())
        print('SAFETY_FIT_DONE',track,fold,method,'Base fallback' if model['fallback'] else 'ensemble',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five','seven'],required=True);p.add_argument('--fold',type=int,required=True)
    a=p.parse_args();assert a.fold in range(5);fit(a.track,a.fold)
