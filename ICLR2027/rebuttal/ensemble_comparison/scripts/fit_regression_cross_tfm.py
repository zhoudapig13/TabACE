"""Two-backbone, six-candidate regression ensemble with OOF calibration."""
import argparse
import json
import numpy as np
from fit_feature_nnls import load_cells
from fit_regression_expansion_static import train,prediction,clean,loss,PRIOR,ALPHAS
from prepare_data import RUN,atomic_json
from run_tabicl import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',required=True,choices=['five','seven_calibrated_v2'])
    p.add_argument('--fold',required=True,type=int);a=p.parse_args()
    cells,source=load_cells(a.track,'regression')
    assert len(cells)==(735 if a.track=='five' else 957)
    for c in cells:
        meta=c['meta'];path=RUN/'results/regression_expansion/pfn'/a.track/'regression'/str(meta['task_id'])/f"{meta['family']}_{meta['rho']}.npz"
        with np.load(path) as f:
            assert np.array_equal(c['y'],f['y_target'])
            # Same response units and canonical labelled source; 3 candidates
            # per backbone, 6 total logical inference members.
            c['bank']=np.stack([c['bank'][0],c['bank'][2],c['bank'][4],f['base'],f['bank'][0],f['bank'][1]])
    cal=[c for c in cells if c['fold']!=a.fold];test=[c for c in cells if c['fold']==a.fold]
    train_ids={c['meta']['data_id'] for c in cal};test_ids={c['meta']['data_id'] for c in test}
    assert train_ids.isdisjoint(test_ids)
    inner=min(c['fold'] for c in cal);tr=[c for c in cal if c['fold']!=inner];val=[c for c in cal if c['fold']==inner]
    base_clean=clean(val)
    for label,method in [('CrossTFMMean6','FeatureMean6'),('CrossTFMGES6','GES6')]:
        out=RUN/'results/regression_cross_tfm/policies'/a.track/f'fold_{a.fold}_{label}.json'
        if out.exists():continue
        trials=[]
        for alpha in ALPHAS:
            fitted=train(tr,method,alpha);cl=clean(val,method,fitted)
            ok=cl['r2']>=base_clean['r2']-.005 and cl['nrmse']<=base_clean['nrmse']+.005
            trials.append({'alpha':alpha,'Clean':cl,'feasible':ok,'loss':loss(val,method,fitted)})
        feasible=[r for r in trials if r['feasible']]
        selected=min(feasible or trials,key=lambda r:r['loss'])
        fitted=train(cal,method,selected['alpha']);raw=train(cal,method,1.)
        original=fitted.copy();refbase=clean(cal);refit=[];best=None
        for eta in [0.,.1,.25,.5,.75,1.]:
            model=(original|{'weights':((1-eta)*PRIOR+eta*np.asarray(original['weights'])).tolist()}
                   if 'weights' in original else {'alpha':eta*original['alpha']})
            cc=clean(cal,method,model)
            ok=cc['r2']>=refbase['r2']-.0025 and cc['nrmse']<=refbase['nrmse']+.0025
            score=loss(cal,method,model);refit.append({'alpha':eta,'Clean':cc,'feasible':ok,'loss':score})
            if ok and (best is None or score<best[0]):best=(score,eta,model)
        assert best is not None
        _,eta,fitted=best;rows=[];raw_rows=[]
        for suffix,model,collection in [('_calibrated',fitted,rows),('_raw',raw,raw_rows)]:
            for c in test:
                pred=prediction(c,method,model)
                assert np.isfinite(pred).all()
                scale=max(float(c['y'].std()),1e-12) if a.track.startswith('seven') else c['meta']['scale']
                collection.append(c['meta']|{'method':label+suffix,'outer_fold':a.fold}|metrics(c['y'],pred,'regression',scale))
                target=RUN/'results/regression_cross_tfm/predictions'/a.track/(label+suffix)/str(c['meta']['task_id'])/f"{c['meta']['family']}_{c['meta']['rho']}.npz"
                target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,prediction=pred,y_target=c['y']);tmp.replace(target)
        atomic_json(out,{'status':'COMPLETE','seed':0,'track':a.track,'fold':a.fold,'method':label,
            'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'inner_validation_fold':inner,
            'fold_map_source':source,'selected_alpha':selected['alpha'],'model':fitted,'raw_model':raw,
            'refit_source_alpha':eta,'refit_trials':refit,'validation_trials':trials,
            'candidate_count':6,'backbones':['TabICL-v2','TabPFN-v3'],
            'candidate_roles':['TabICL identity','TabICL SVD32','TabICL cross32','TabPFN identity','TabPFN bootstrap1','TabPFN bootstrap2'],
            'clean_reference':'TabICL identity; both backbone Base scores reported separately',
            'source_access':'Same canonical labelled context as Base',
            'target_test_labels_used_for_fit':False,'Shared_or_order_bounds_used_for_selection':False,
            'Shared_run':False,'Markdown_written':False,'rows':rows,'raw_rows':raw_rows})
        print('CROSS_DONE',a.track,a.fold,label,selected['alpha'],eta,flush=True)


if __name__=='__main__':main()
