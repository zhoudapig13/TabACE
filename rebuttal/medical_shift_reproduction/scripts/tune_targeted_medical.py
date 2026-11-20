"""Matched-supervision medical development on a fixed old-test adaptation split.

Remaining evaluation labels are deliberately absent from this module's inputs.
All selection is on the unchanged original development cohort, seed=0.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from tabicl import TabICLClassifier
import torch
import run_public_shift as exp
from develop_specific_residual import calibrate,new_learner,fit_residual,apply_residual
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class TargetedWorker:
    def __init__(self,task,domain,gpu,data_root=None):
        self.task=task;self.domain=domain
        self.data=(exp.ROOT/'data/processed/medical_test_assisted_v26' if data_root is None else Path(data_root))/(task+'_'+domain)
        self.meta=json.loads((self.data/'metadata.json').read_text())
        with np.load(self.data/'inputs.npz') as z:self.pack={k:z[k] for k in z.files}
        p=self.pack
        assert {k:exp.array_hash(v) for k,v in p.items()}==self.meta['input_hashes']
        self.ns=len(p['y_source_train']);self.nt=len(p['y_target_train'])
        self.X=np.r_[p['X_source_train'],p['X_target_train'],p['X_adaptation_fit']]
        self.y=np.r_[p['y_source_train'],p['y_target_train'],p['y_adaptation_fit']]
        self.mandatory=np.arange(self.ns+self.nt,len(self.y))
        self.fit_domains=np.r_[np.zeros(self.ns),np.ones(self.nt),np.full(len(self.mandatory),domain=='ood')]
        self.nsc=len(p['y_source_cal']);self.ntc=len(p['y_target_cal'])
        self.cal_y=np.r_[p['y_source_cal'],p['y_target_cal'],p['y_adaptation_cal']]
        self.cal_domains=np.r_[np.zeros(self.nsc),np.ones(self.ntc),np.full(len(p['y_adaptation_cal']),domain=='ood')]
        self.ncal=len(self.cal_y);self.ndev=len(p['y_development'])
        self.queries=np.r_[p['X_source_cal'],p['X_target_cal'],p['X_adaptation_cal'],
                           p['X_development'],p['X_evaluation']]
        self.dev=slice(self.ncal,self.ncal+self.ndev)
        self.evaluation=slice(self.ncal+self.ndev,len(self.queries))
        self.query_domains=np.r_[self.cal_domains,np.full(len(self.queries)-self.ncal,domain=='ood')]
        self.query_hash=exp.array_hash(self.queries)
        self.cache=self.data/'member_cache_native8';self.cache.mkdir(exist_ok=True)
        self.model=TabICLClassifier(n_estimators=8,device=f'cuda:{gpu}',model_path=str(exp.CHECKPOINT),
            allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0)
        a,b,_,_=exp.geometry(self.X,p['X_development'])
        dist=NearestNeighbors(n_neighbors=10).fit(b).kneighbors(a)[0].mean(1)
        outlier=np.minimum(a*a,100).mean(1)
        neighbors=NearestNeighbors(n_neighbors=11).fit(a).kneighbors(a,return_distance=False)
        neighbors=np.stack([row[row!=i][:10] for i,row in enumerate(neighbors)])
        conflict=np.abs(self.y-self.y[neighbors].mean(1))
        self.scores=[dist,dist+.35*outlier,conflict+.1*dist,np.zeros(len(self.y)),conflict+.2*outlier]
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        self.subspace=np.flatnonzero(np.arange(self.X.shape[1])!=association.argmax())

    def weights(self,share):
        # share is the calibration weight of the evaluated domain.
        selected=self.cal_domains==int(self.domain=='ood')
        return np.where(selected,share/selected.sum(),(1-share)/(~selected).sum())

    def score(self,p,threshold):
        return float(np.mean((p[self.dev]>=threshold)==self.pack['y_development']))

    def rows(self,use_target,weight,seed):
        rng=np.random.default_rng(seed)
        target=np.arange(self.ns,self.ns+self.nt) if use_target else np.empty(0,dtype=int)
        count=256-len(self.mandatory)-len(target)
        source=[]
        for label in [0,1]:
            pool=np.flatnonzero(self.y[:self.ns]==label)
            w=np.maximum(weight[pool],1e-10);w=w/w.sum()
            source.append(rng.choice(pool,count//2,replace=False,p=w))
        selected=np.r_[*source,target,self.mandatory];rng.shuffle(selected)
        assert len(selected)==256 and len(np.unique(selected))==256
        assert set(self.mandatory).issubset(selected)
        return selected

    def predict(self,rows,columns=None):
        columns=np.arange(self.X.shape[1]) if columns is None else columns
        context=self.X[rows][:,columns]
        key=hashlib.sha256(context.tobytes()+self.y[rows].tobytes()+columns.tobytes()+
                           self.query_hash.encode()+b'native8_seed0').hexdigest()
        path=self.cache/(key+'.npz')
        if path.exists():
            with np.load(path) as z:return z['p']
        root=getattr(getattr(self.model,'model_',None),'icl_predictor',None)
        if root is not None and hasattr(root,'root'):root.root=None
        self.model.fit(context,self.y[rows]);self.model._load_model=lambda:None
        result=np.concatenate([self.model.predict_proba(self.queries[i:i+1024][:,columns])[:,1]
                               for i in range(0,len(self.queries),1024)])
        assert np.isfinite(result).all()
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,p=result,rows=rows,columns=columns)
        temporary.replace(path)
        print('TFM_MEMBER',self.task,self.domain,'rows',len(rows),'features',len(columns),flush=True)
        return result

    def bank(self,use_target,strength):
        predictions=[]
        for i,score in enumerate(self.scores):
            rank=np.argsort(np.argsort(score,kind='stable'),kind='stable')/max(1,len(score)-1)
            rows=self.rows(use_target,np.exp(-6*strength*rank),10+i)
            predictions.append(self.predict(rows,self.subspace if i==3 else None))
        return np.asarray(predictions)

    def run(self):
        shares=getattr(self,'calibration_shares',[.5,.8]);trials=[];best_score=-1
        base=None;base_config=None
        for p,conf in getattr(self,'additional_base_candidates',lambda:[])():
            value=self.score(p,conf['threshold'])
            trials.append(dict(stage='Base',frozen_candidate=conf,development_accuracy=value))
            if value>best_score+1e-12:
                base=p;best_score=value;base_config=conf
        for use_target in [False,True]:
            p=self.predict(self.rows(use_target,np.ones(len(self.y)),0))
            for share in shares:
                threshold=exp.threshold(p[:self.ncal],self.cal_y,self.weights(share))
                value=self.score(p,threshold)
                trials.append(dict(stage='Base',use_original_target_context=use_target,
                    calibration_domain_share=share,threshold=threshold,development_accuracy=value))
                if value>best_score+1e-12:
                    base=p;best_score=value
                    base_config=dict(use_original_target_context=use_target,
                        calibration_domain_share=share,threshold=threshold)
        threshold=base_config['threshold'];shared=base.copy();shared_score=best_score
        shared_config=dict(kind='Base fallback',threshold=threshold)
        frozen_shared_selected=False
        for p,conf in getattr(self,'additional_shared_candidates',lambda threshold:[])(threshold):
            value=self.score(p,threshold)
            trials.append(dict(stage='Shared',frozen_candidate=conf,development_accuracy=value))
            if value>shared_score+1e-12:
                shared=p;shared_score=value;shared_config=conf;frozen_shared_selected=True
        selected_bank=None;mean_best=-1;mean_p=None;mean_threshold=None
        exp.features=lambda b,k,m:floor_features(b,k,m,.25)
        try:
            for use_target in [False,True]:
                for strength in [.5,1.]:
                    bank=self.bank(use_target,strength)
                    raw_mean=bank.mean(0)
                    for share in shares:
                        weight=self.weights(share)
                        tau=exp.threshold(raw_mean[:self.ncal],self.cal_y,weight)
                        value=self.score(raw_mean,tau)
                        if value>mean_best+1e-12:
                            mean_best=value;mean_p=raw_mean;mean_threshold=tau
                        for reg in [1.,.1]:
                            policy=exp.fit_gate(base[:self.ncal],bank[:,:self.ncal],self.cal_y,weight,
                                                len(bank),3.,reg)
                            full=exp.probability(base,bank,policy)
                            for cap in [.25,.5]:
                                p=(1-cap)*base+cap*full;value=self.score(p,threshold)
                                trials.append(dict(stage='Shared',use_original_target_context=use_target,
                                    sampling_strength=strength,calibration_domain_share=share,
                                    regularization=reg,update_cap=cap,development_accuracy=value))
                                if value>shared_score+1e-12:
                                    shared=p;shared_score=value;selected_bank=bank.copy()
                                    frozen_shared_selected=False
                                    shared_config=dict(kind='five-context probability gate',threshold=threshold,
                                        use_original_target_context=use_target,sampling_strength=strength,
                                        calibration_domain_share=share,policy=policy,update_cap=cap)
                    print('SHARED_DEVELOPMENT',self.task,self.domain,shared_score,flush=True)
            shared_oof=base[:self.ncal].copy()
            if frozen_shared_selected:
                shared_oof=self.frozen_shared_oof(threshold)
            if selected_bank is not None:
                policy=shared_config['policy'];weight=self.weights(shared_config['calibration_domain_share'])
                strata=self.cal_y+2*self.cal_domains
                for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(self.ncal),strata):
                    gate=exp.fit_gate(base[fit],selected_bank[:,fit],self.cal_y[fit],weight[fit],
                                      len(selected_bank),3.,policy['regularization'])
                    full=exp.probability(base[val],selected_bank[:,val],gate)
                    cap=shared_config['update_cap'];shared_oof[val]=(1-cap)*base[val]+cap*full
        finally:exp.features=ORIGINAL_FEATURES
        specific=shared.copy();specific_score=shared_score;specific_config=dict(kind='Shared fallback',threshold=threshold)
        for p,conf in getattr(self,'additional_specific_candidates',lambda threshold:[])(threshold):
            value=self.score(p,threshold)
            trials.append(dict(stage='Specific',frozen_candidate=conf,development_accuracy=value))
            if value>specific_score+1e-12:
                specific=p;specific_score=value;specific_config=conf
        aux_best=-1;auxiliary=None;aux_config=None;model_count=0
        specs=getattr(self,'auxiliary_specs',[dict(kind='logistic',C=.1),dict(kind='logistic',C=1.),
               dict(kind='forest',depth=4),dict(kind='forest',depth=8),
               dict(kind='lightgbm',depth=3),dict(kind='lightgbm',depth=4)])
        domain=self.fit_domains.astype(bool)
        x=np.c_[self.X,self.fit_domains];q=np.c_[self.queries,self.query_domains]
        strata=self.cal_y+2*self.cal_domains
        s=logit(np.clip(shared_oof,1e-6,1-1e-6))-logit(threshold)
        centered=logit(np.clip(shared,1e-6,1-1e-6))-logit(threshold)
        masses=[.2,.5] if self.domain=='ood' else [0.,.2]
        for spec in specs:
            for mass in masses:
                mask=np.ones(len(self.y),dtype=bool) if mass else ~domain
                fitw=np.where(domain,mass/max(1,domain.sum()),(1-mass)/(~domain).sum())*mask.sum()
                model=new_learner(spec);model.fit(x[mask],self.y[mask],sample_weight=fitw[mask]);model_count+=1
                rawlogit=logit(np.clip(model.predict_proba(q)[:,1],1e-6,1-1e-6))
                for share in shares:
                    weight=self.weights(share);oof=np.zeros(self.ncal)
                    for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(self.ncal),strata):
                        a,b=calibrate(rawlogit[fit],self.cal_y[fit],weight[fit]);oof[val]=expit(a*rawlogit[val]+b)
                    a,b=calibrate(rawlogit[:self.ncal],self.cal_y,weight)
                    teacher=expit(a*rawlogit+b);tau=exp.threshold(oof,self.cal_y,weight)
                    aux_score=self.score(teacher,tau)
                    common=dict(learner=spec,target_training_mass=mass,calibration_domain_share=share,
                                platt_a=float(a),platt_b=float(b),expert_threshold=tau,threshold=threshold)
                    if aux_score>aux_best+1e-12:
                        auxiliary=teacher;aux_best=aux_score;aux_config=dict(common)
                    expert_centered=logit(np.clip(teacher,1e-6,1-1e-6))-logit(tau)
                    candidates=[]
                    for alpha in [.15,.3,.45]:
                        candidates.append((expit((1-alpha)*centered+alpha*expert_centered+logit(threshold)),
                                           dict(common,kind='constant logit pool',alpha=alpha)))
                    t=logit(np.clip(oof,1e-6,1-1e-6))-logit(tau)
                    for cap in [.25,.5]:
                        for reg in [.1,.01]:
                            gate=fit_residual(s,t,self.cal_y,weight,cap,reg)
                            candidates.append((apply_residual(shared,teacher,threshold,tau,gate),
                                               dict(common,kind='cross-fitted residual gate',gate=gate)))
                    for p,conf in candidates:
                        value=self.score(p,threshold)
                        trials.append(dict(stage='Specific',configuration=conf,development_accuracy=value))
                        if value>specific_score+1e-12:
                            specific=p;specific_score=value;specific_config=conf
                print('SPECIFIC_DEVELOPMENT',self.task,self.domain,model_count,
                      dict(Base=best_score,Shared=shared_score,Specific=specific_score,Auxiliary=aux_best),flush=True)
        output=dict(Base=base,Shared=shared,Specific=specific,Auxiliary=auxiliary,Mean5=mean_p)
        assert all(np.isfinite(p).all() for p in output.values())
        scores=dict(Base=best_score,Shared=shared_score,Specific=specific_score,Auxiliary=aux_best,Mean5=mean_best)
        configuration=dict(Base=base_config,Shared=shared_config,Specific=specific_config,
                           Auxiliary=dict(aux_config,threshold=aux_config['expert_threshold']),Mean5=dict(threshold=mean_threshold))
        report=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),task=self.task,domain=self.domain,
            data=str(self.data.relative_to(exp.ROOT)),metadata=self.meta,
            configurations=configuration,development_accuracy=scores,trials=trials,
            query_layout=dict(calibration=[0,self.ncal],development=[self.dev.start,self.dev.stop],
                              evaluation=[self.evaluation.start,self.evaluation.stop]),
            common_native_estimators_per_tfm_branch=8,common_context_cap=256,
            mandatory_new_adaptation_context_rows=16,additional_adaptation_calibration_rows=16,
            original_test_labels_used_for_development=32,evaluation_labels_opened=False,
            label_access_matched_across_methods=True,selected_by='Original development accuracy only; strict-improvement fallback',
            auxiliary_model_count=model_count,not_original_tabace=True,not_original_paper_reproduction=True,
            previous_test_cohorts_observed=True,protocol=json.loads((exp.ROOT/'configs/medical_test_assisted_v26.json').read_text()))
        report.update(context_replacement_protocol=getattr(self,'replacement_protocol','Independent resampling, v26'),
            original_frozen_candidates_available=hasattr(self,'additional_base_candidates'),
            script_sha256_at_freeze=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        dest=exp.ROOT/'results'/getattr(self,'results_tag','medical_test_assisted_v26')/(self.task+'_'+self.domain)
        exp.write_json(dest/'frozen_selection.json',report)
        np.savez_compressed(self.data/getattr(self,'predictions_filename','frozen_predictions.npz'),**output)
        print('TARGETED_FROZEN',self.task,self.domain,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True)
    p.add_argument('--domain',choices=['id','ood'],required=True);p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):TargetedWorker(args.task,args.domain,args.gpu).run()
