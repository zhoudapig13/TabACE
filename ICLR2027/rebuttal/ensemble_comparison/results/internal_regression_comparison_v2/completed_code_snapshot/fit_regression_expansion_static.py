"""Source-view regression combiners with dataset-nested calibration.

All six predictions already exist. Identity is one member, so static calibration
shrinkage costs no additional backbone call. Raw versions are retained too.
"""
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from fit_feature_nnls import load_cells
from fit_combiners import sample_weights
from fit_convex_combiners import design,fit_ges
from prepare_data import RUN,atomic_json
from run_tabicl import metrics

PRIOR=np.array([1.,0,0,0,0,0])
LAMBDAS=[.001,.01,.1,1.,10.,100.]
ALPHAS=[.25,.5,.75,1.]


def gram(cells):
    A,y,w=design(cells,'regression')
    return (A*w)@A.T,A@(w*y)


def ridge(G,b,regularization):
    B=np.vstack([-np.ones(5),np.eye(5)])
    H=B.T@G@B+regularization*(B.T@B)
    rhs=B.T@(b-G@PRIOR)
    z=np.linalg.lstsq(H,rhs,rcond=1e-12)[0]
    beta=PRIOR+B@z
    if not np.isfinite(beta).all() or np.linalg.norm(beta)>100:
        raise ValueError('unstable affine ridge coefficients')
    return beta


def convex(G,b,regularization):
    unit=max(1.,float(np.abs(G).max()),float(np.abs(b).max()))
    def fun(beta):
        delta=beta-PRIOR
        return ((float(beta@G@beta-2*b@beta)+regularization*float(delta@delta))/unit,
                (2*(G@beta-b)+2*regularization*delta)/unit)
    for initial in [PRIOR,np.full(6,1/6)]:
        fit=minimize(fun,initial,jac=True,method='SLSQP',bounds=[(0,1)]*6,
            constraints=[{'type':'eq','fun':lambda x:float(x.sum()-1),'jac':lambda x:np.ones(6)}],
            options={'maxiter':200,'ftol':1e-12})
        if fit.success:
            beta=np.clip(fit.x,0,1);return beta/beta.sum()
    raise RuntimeError(str(fit.message))


def prediction(c,method,model):
    if method in ['Median6','FeatureMean6']:
        raw=np.median(c['bank'],axis=0) if method=='Median6' else c['bank'].mean(0)
        return (1-model['alpha'])*c['bank'][0]+model['alpha']*raw
    return np.asarray(model['weights'])@c['bank']


def clean(cells,method=None,model=None):
    rows=[]
    for c in cells:
        if c['meta']['rho']!=0:continue
        p=c['bank'][0] if method is None else prediction(c,method,model)
        scale=max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
        rows.append(metrics(c['y'],p,'regression',scale))
    return {k:float(np.mean([r[k] for r in rows])) for k in ['r2','nrmse']}


def loss(cells,method,model):
    total=0.
    for c,w in zip(cells,sample_weights(cells)):
        scale=max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
        total+=float(w@(((prediction(c,method,model)-c['y'])/scale)**2))
    return total


def train(cells,method,param):
    if method in ['Median6','FeatureMean6']:return {'alpha':param}
    if method=='GES6':
        beta,_=fit_ges(*design(cells,'regression'),'regression')
        return {'weights':((1-param)*PRIOR+param*beta).tolist(),'alpha':param}
    G,b=gram(cells)
    beta=ridge(G,b,param) if method=='RidgeStacking6' else convex(G,b,param)
    return {'weights':beta.tolist(),'regularization':param}


def save_rows(cells,method,model,track,fold,suffix,result_root):
    rows=[]
    for c in cells:
        p=prediction(c,method,model)
        assert np.isfinite(p).all()
        scale=max(float(c['y'].std()),1e-12) if track.startswith('seven') else c['meta']['scale']
        rows.append(c['meta'] | {'method':method+suffix,'outer_fold':fold} | metrics(c['y'],p,'regression',scale))
        out=RUN/'results'/result_root/'predictions'/track/(method+suffix)/str(c['meta']['task_id'])/f"{c['meta']['family']}_{c['meta']['rho']}.npz"
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=p,y_target=c['y']);tmp.replace(out)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--track',choices=['five','seven_calibrated_v2'],required=True)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--clean-refit-margin',type=float)
    a=parser.parse_args()
    result_root='regression_expansion_static_clean' if a.clean_refit_margin is not None else 'regression_expansion_static'
    cells,source=load_cells(a.track,'regression');assert len(cells)==(735 if a.track=='five' else 957)
    for c in cells:
        raw=RUN/'results/tabicl'/a.track/'regression'/str(c['meta']['task_id'])/f"{c['meta']['family']}_{c['meta']['rho']}.npz"
        with np.load(raw) as f:assert np.array_equal(c['bank'][0],f['prediction__base_protocol_check'])
    trainset=[c for c in cells if c['fold']!=a.fold];test=[c for c in cells if c['fold']==a.fold]
    train_ids={c['meta']['data_id'] for c in trainset};test_ids={c['meta']['data_id'] for c in test}
    assert train_ids.isdisjoint(test_ids)
    inner=min(c['fold'] for c in trainset)
    tr=[c for c in trainset if c['fold']!=inner];val=[c for c in trainset if c['fold']==inner]
    base_clean=clean(val)
    for method in ['GES6','ConvexStacking6','RidgeStacking6','Median6','FeatureMean6']:
        out=RUN/'results'/result_root/'policies'/a.track/f'fold_{a.fold}_{method}.json'
        if out.exists():continue
        params=ALPHAS if method in ['GES6','Median6','FeatureMean6'] else ([0.]+LAMBDAS if method=='ConvexStacking6' else LAMBDAS)
        candidates=[]
        for p in params:
            try:
                fitted=train(tr,method,p);cc=clean(val,method,fitted);score=loss(val,method,fitted)
                feasible=cc['r2']>=base_clean['r2']-.005 and cc['nrmse']<=base_clean['nrmse']+.005
                candidates.append({'parameter':p,'clean':cc,'loss':score,'clean_feasible':feasible})
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                candidates.append({'parameter':p,'invalid':repr(exc),'clean_feasible':False})
        valid=[r for r in candidates if 'loss' in r];assert valid
        feasible=[r for r in valid if r['clean_feasible']]
        selected=min(feasible or valid,key=lambda r:r['loss'])
        fitted=train(trainset,method,selected['parameter'])
        refit_trials=[];refit_alpha=1.
        if a.clean_refit_margin is not None:
            calibration_base=clean(trainset)
            refit_original=fitted.copy();best_refit=None
            for eta in [0.,.1,.25,.5,.75,1.]:
                if 'weights' in refit_original:
                    shrunk=refit_original|{'weights':((1-eta)*PRIOR+eta*np.asarray(refit_original['weights'])).tolist()}
                else:shrunk={'alpha':eta*refit_original['alpha']}
                observed=clean(trainset,method,shrunk)
                ok=observed['r2']>=calibration_base['r2']-a.clean_refit_margin and observed['nrmse']<=calibration_base['nrmse']+a.clean_refit_margin
                objective=loss(trainset,method,shrunk)
                refit_trials.append({'alpha':eta,'Clean':observed,'feasible':ok,'calibration_loss':objective})
                if ok and (best_refit is None or objective<best_refit[0]):best_refit=(objective,eta,shrunk)
            assert best_refit is not None
            _,refit_alpha,fitted=best_refit
        default_parameter=1. if method in ['GES6','Median6','FeatureMean6'] else (0. if method=='ConvexStacking6' else .01)
        default=train(trainset,method,default_parameter)
        rows=save_rows(test,method,fitted,a.track,a.fold,'_calibrated',result_root)
        raw_rows=save_rows(test,method,default,a.track,a.fold,'_raw',result_root)
        atomic_json(out,{'status':'COMPLETE','seed':0,'track':a.track,'fold':a.fold,'method':method,
            'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'inner_validation_fold':inner,
            'fold_map_source':source,'selected_parameter':selected['parameter'],'model':fitted,
            'raw_model':default,'candidates':candidates,'calibration_feasible':bool(feasible),
            'calibration_base_clean':base_clean,'clean_limit':.005,'candidate_count':6,
            'final_refit_Clean_margin':a.clean_refit_margin,'refit_source_alpha':refit_alpha,
            'refit_source_calibration_trials':refit_trials,
            'context_bank':'identity, source SVD16/32/64, source cross32/64; fixed six source views',
            'identity_prior':'static calibration regularization, not query-dependent gating',
            'source_access':'Canonical Base input; no history/reference augmentation',
            'target_test_labels_used_for_fit':False,'Shared_or_score_ordering_used_for_selection':False,
            'Shared_run':False,'Markdown_written':False,'rows':rows,'raw_rows':raw_rows})
        print('STATIC_DONE',a.track,a.fold,method,selected['parameter'],bool(feasible),flush=True)


if __name__=='__main__':main()
