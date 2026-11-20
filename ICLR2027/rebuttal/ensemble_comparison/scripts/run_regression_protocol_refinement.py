"""Finite native/PHE preprocessing check and representative IQR support bank.

Frozen source-only quantile normalization is a native preprocessing adaptation.
IQR uses six representative k-of-20 source supports, including full source.
Neither the perturbation protocol nor test labels enter configuration selection.
"""
import argparse
import gc
import json
import logging
import time
import traceback
import warnings
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from prepare_data import RUN,atomic_json
from shared_reference import load_condition
from run_tabicl import Predictor,metrics
from run_regression_expansion_gpu import work,quantile_prediction
from run_tabpfn import (model,fit_predict,AutoPostHocEnsemblePredictor,
    TabPFNBaseModelSource,TaskType)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    jobs=work()[args.shard::args.shards]
    if args.limit:jobs=jobs[:args.limit]
    tt=Predictor('regression','cuda:0');single=model('regression','cuda:0')
    native=model('regression','cuda:0');native.n_estimators=6
    failures=0
    for meta,family,rho in jobs:
        out=RUN/'results/regression_protocol_refinement'/meta['track']/'regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        try:
            data,info=load_condition(meta,family,rho)
            X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
            tick=time.perf_counter();times={};predictions={};arrays={}
            transform=QuantileTransformer(n_quantiles=min(64,len(y)),output_distribution='normal',subsample=None,random_state=0)
            A=transform.fit_transform(X).astype(np.float32);T=transform.transform(Q).astype(np.float32)
            start=time.perf_counter();p,_=tt(A,y,T,members=6)
            predictions['TabICL_Native6_source_quantiles']=p;times['TabICL_Native6_source_quantiles']=time.perf_counter()-start
            start=time.perf_counter();p=fit_predict(native,A,y,T,'regression',None)
            predictions['TabPFN_Native6_source_quantiles']=p;times['TabPFN_Native6_source_quantiles']=time.perf_counter()-start
            start=time.perf_counter()
            if np.std(y)<1e-12 or A.shape[1]==0 or np.all(np.ptp(A,axis=0)==0):
                p=np.full(len(t),float(y.mean()));nonzero=0
            else:
                configs=[(f'seed_{i}',model('regression','cuda:0',seed=i)) for i in range(6)]
                phe=AutoPostHocEnsemblePredictor(preset='default',max_time=30,task_type=TaskType.REGRESSION,
                    ges_scoring_string='mse',device='cuda',bm_random_state=0,ges_random_state=0,
                    tabpfn_base_model_source=TabPFNBaseModelSource.CUSTOM,custom_tabpfn_models=configs,
                    max_models=6,n_repeats=1,holdout_fraction=.25,ges_n_iterations=25,ignore_pretraining_limits=True)
                phe.fit(A,y);p=np.asarray(phe.predict(T)).reshape(-1)
                nonzero=int(np.count_nonzero(phe._ens_model.weights_));del phe,configs
            predictions['PHE6_source_quantiles']=p;times['PHE6_source_quantiles']=time.perf_counter()-start
            start=time.perf_counter();indices=np.random.default_rng(0).permutation(len(y));parts=np.array_split(indices,20)
            bank=[];iqrs=[];sizes=[]
            for i,k in enumerate([1,5,9,13,17,20]):
                selected_parts=np.random.default_rng(1009*(i+1)).choice(20,k,replace=False)
                ids=np.concatenate([parts[j] for j in selected_parts])
                # Use the canonical order for the full-support candidate, so
                # it is a valid paired check against the stored single Base.
                if k==20:ids=np.arange(len(y))
                single.random_state=0
                pred,iqr=quantile_prediction(single,X[ids],y[ids],Q)
                bank.append(pred);iqrs.append(iqr);sizes.append(len(ids))
            bank,iqrs=np.stack(bank),np.stack(iqrs);selected=int(iqrs.mean(1).argmin())
            predictions['IQR6_representative_support']=bank[selected]
            times['IQR6_representative_support']=time.perf_counter()-start
            arrays.update(support_bank=bank,support_iqr=iqrs)
            scale=max(float(t.std()),1e-12) if meta['track'].startswith('seven') else info['scale']
            scores={}
            for name,p in predictions.items():
                assert np.asarray(p).shape==t.shape and np.isfinite(p).all()
                arrays['prediction__'+name]=p;scores[name]=metrics(t,p,'regression',scale)|{'seconds':times[name]}
            arrays['y_target']=t
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**arrays);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','seed':0,'family':family,'rho':rho,
                'source_preprocessing':'Source-only QuantileTransformer64 to Gaussian; identical fitted transform on query',
                'source_preprocessing_uses_target_features_or_labels_for_fit':False,
                'IQR_support_k_of_20':[1,5,9,13,17,20],'IQR_support_rows':sizes,'IQR_selected':selected,
                'IQR_role':'Representative six-candidate support-size adaptation, not original 400-candidate experiment',
                'IQR_full_source_candidate_included':True,'PHE_nonzero_members':nonzero,
                'native_member_count':6,'PHE_candidate_limit':6,'IQR_candidate_count':6,
                'canonical_source_only':True,'target_labels_used_for_selection':False,
                'Shared_or_score_ordering_used':False,'Shared_run':False,'Markdown_written':False,
                'seconds':time.perf_counter()-tick,'methods':scores})
            print('REFINEMENT_DONE',meta['track'],meta['task_id'],family,rho,selected,flush=True)
            gc.collect();torch.cuda.empty_cache()
        except Exception as exc:
            failures+=1
            atomic_json(RUN/'results/inference_failures'/f"reg_refine_{meta['track']}_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()})
            print('REFINEMENT_FAILED',repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
