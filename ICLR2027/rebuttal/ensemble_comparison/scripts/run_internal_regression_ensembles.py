"""Six-member TabICL regression banks on frozen internal-only benchmarks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,metrics
from internal_regression_benchmark import OUT,manifest,five_condition


def fingerprint(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def declare():
    p=RUN/'configs/internal_regression_ensemble_plan.json'
    if p.exists():raise RuntimeError('Comparison plan already exists')
    atomic_json(p,{'status':'DECLARED','backbone':'TabICL-v2','seed':0,'logical_members':6,
       'methods':['Native6','GES6','ConvexStacking6','RidgeStacking6','Median6','FeatureMean6','TrimmedMean6','IQR6_keep1_adapted'],
       'Native_adaptation':['unchanged native6','source-only Gaussian quantiles then native6'],
       'feature_bank':['identity','source SVD16','source SVD32','source SVD64','source crosses32','source crosses64'],
       'IQR_bank':'Base exact canonical context plus five canonical bootstrap contexts, total six',
       'IQR_quantiles':[.15,.85],'IQR_method_is_original_OligoICP_replication':False,
       'source_access':'Every method uses only Base canonical current-context rows; no extra history/reference rows',
       'existing_unrestricted_pool_results_preserved':True,'clean_decline_tolerance':.005,
       'method_selection':'nested dataset-OOF normalized loss, Clean calibration constraint',
       'static_shrinkage_prior':'identity member already in six-member bank; no seventh inference',
       'historical_Shared_bounds_used_for_method_selection':False,'all_raw_and_adapted_variants_retained':True,
       'Random_excluded_from_main_table':True,'Shared_run':False,'internal_only':True,'Markdown_written':False})
    print('COMPARISON_PLAN_DECLARED',flush=True)


def jobs():
    from seven_shifts import registry,FAMILIES
    five=[('five',m,f,r) for m in manifest() for f,r in [('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]]
    seven=[('seven',m,f,r) for m in registry() if m['task']=='regression'
           for f,r in [('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]]
    assert len(five)==735 and len(seven)==957
    return five+seven


def load_condition(track,m,family,rho):
    if track=='five':
        settings=json.loads((OUT/'frozen_five.json').read_text())
        data,info=five_condition(m,family,rho,settings['generator_parameters'])
        path=OUT/'five_candidates'/settings['candidate']/str(m['task_id'])/f'{family}_{rho}.npz'
        with np.load(path) as z:base=z['prediction'];assert np.array_equal(data['y_target'],z['y_target'])
        previous=json.loads(path.with_suffix('.json').read_text())
        assert previous['input_hashes']=={k:protocol.array_sha256(a) for k,a in data.items()}
    else:
        from shared_reference import load_condition as legacy_load
        assert (OUT/'frozen_seven.json').exists()
        data,info=legacy_load(m|{'track':'seven_calibrated_v2'},family,rho)
        path=RUN/'results/tabicl/seven_calibrated_v2/regression'/str(m['task_id'])/f'{family}_{rho}.npz'
        with np.load(path) as z:base=z['prediction__base_protocol_check'];assert np.array_equal(data['y_target'],z['y_target'])
        info=info|{'track':'seven_internal_regression_v1','family':family,'rho':rho,'strength':rho/100.,
                   'strength_rescaled':False,'scale':max(float(data['y_target'].std()),1e-12),'internal_only':True}
    return data,info,base


def run(shard,shards,limit,tracks):
    import torch,tabicl
    from run_feature_views import projections
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    required=[]
    if 'five' in tracks:required+=['frozen_clean.json','frozen_five.json']
    if 'seven' in tracks:required+=['frozen_seven.json']
    for name in required:assert (OUT/name).exists()
    predictor=Predictor('regression','cuda:0')
    work=[j for j in jobs() if j[0] in tracks][shard::shards]
    if limit:work=work[:limit]
    for track,m,family,rho in work:
        out=OUT/'ensemble_banks'/track/str(m['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        tick=time.perf_counter();data,info,base=load_condition(track,m,family,rho)
        X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
        scale=info['scale'];feature_reused=False;bootstrap_reused=False
        if track=='seven':
            raw=RUN/'results/tabicl/seven_calibrated_v2/regression'/str(m['task_id'])/out.name
            with np.load(raw) as z:native=z['prediction__native_6']
            old=RUN/'results/feature_views/tabicl/seven_calibrated_v2/regression'/str(m['task_id'])/out.name
            with np.load(old) as z:feature=z['bank'];assert np.array_equal(t,z['y_target']);assert np.array_equal(feature[0],base)
            feature_reused=True
        else:
            native,_=predictor(X,y,Q,members=6)
            feature=[]
            for name,A,T in projections(X,Q):feature.append(base if 'identity' in name else predictor(A,y,T)[0])
            feature=np.stack(feature)
        qt=QuantileTransformer(n_quantiles=min(64,len(y)),output_distribution='normal',random_state=0,subsample=None)
        A=qt.fit_transform(X);T=qt.transform(Q);native_quantile,_=predictor(A,y,T,members=6)
        mean,base_iqr=predictor(X,y,Q,quantiles=True)
        assert np.allclose(mean,base,rtol=1e-6,atol=1e-6),(track,m['task_id'],family,rho,float(np.max(abs(mean-base))))
        ibank=[base];iqrs=[base_iqr]
        old=RUN/'results/canonical_input/tabicl/seven_calibrated_v2/regression'/str(m['task_id'])/out.name
        if track=='seven' and old.with_suffix('.json').exists():
            saved=json.loads(old.with_suffix('.json').read_text())
            assert saved.get('resampling_fraction',1.)==1.
            expected_indices=[fingerprint(np.random.default_rng(1009*(i+1)).choice(len(y),size=len(y),replace=True)) for i in range(6)]
            assert saved['bootstrap_indices_sha256']==expected_indices,'Bootstrap construction differs'
            assert saved['source_X_sha256']==fingerprint(X) and saved['source_y_sha256']==fingerprint(y)
            assert saved['query_X_sha256']==fingerprint(Q)
            with np.load(old) as z:
                assert np.array_equal(t,z['y_target'])
                ibank.extend(z['bootstrap_bank'][:5]);iqrs.extend(z['bootstrap_iqr'][:5])
            bootstrap_reused=True
        else:
            for i in range(5):
                ids=np.random.default_rng(1009*(i+1)).choice(len(y),size=len(y),replace=True)
                pred,iqr=predictor(X[ids],y[ids],Q,quantiles=True);ibank.append(pred);iqrs.append(iqr)
        ibank=np.stack(ibank);iqrs=np.stack(iqrs)
        assert feature.shape==ibank.shape==iqrs.shape==(6,len(t))
        assert np.isfinite(feature).all() and np.isfinite(ibank).all() and np.isfinite(iqrs).all()
        selected=int(np.argmin(iqrs.mean(1)))
        predictions={'Base':base,'Native6_raw':native,'Native6_source_quantiles':native_quantile,
          'FeatureMean6_raw':feature.mean(0),'Median6_raw':np.median(feature,axis=0),
          'TrimmedMean6_raw':np.sort(feature,axis=0)[1:-1].mean(0),'IQR6_keep1_raw_adapted':ibank[selected]}
        observed={name:metrics(t,p,'regression',scale) for name,p in predictions.items()}
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,feature_bank=feature,iqr_bank=ibank,iqr=iqrs,y_target=t,
             source_mean=np.asarray(float(y.mean())),source_scale=np.asarray(max(float(y.std()),1e-12)),
             **{'prediction__'+name:p for name,p in predictions.items()});tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','model_seed':0,'methods':observed,
             'canonical_context_rows':len(y),'eligible_rows_for_all_methods':len(y),'context_count':6,
             'additional_history_reference_rows_used':False,'selected_IQR_candidate':selected,
             'IQR_bank_contains_Base':True,'source_only_preprocessing':True,'backbone_weights_frozen':True,
             'feature_predictions_reused':feature_reused,'bootstrap_predictions_reused':bootstrap_reused,
             'benchmark_parameters_already_frozen':True,'Shared_run':False,'TabICL_library':tabicl.__file__,
             'input_hashes':{k:protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target','y_target']},
             'seconds':time.perf_counter()-tick,'all_raw_variants_retained':True})
        print('BANKS_DONE',track,m['dataset'],family,rho,flush=True)
    print('WORKER_FINISHED',shard,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--tracks',nargs='+',default=['five','seven']);a=p.parse_args()
    if a.declare:declare()
    else:run(a.shard,a.shards,a.limit,a.tracks)
