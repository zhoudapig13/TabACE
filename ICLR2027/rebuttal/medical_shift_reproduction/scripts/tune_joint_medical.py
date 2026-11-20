"""Joint ID/OOD development optimization; never open test outcome files."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
from datetime import datetime,timezone
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
from threadpoolctl import threadpool_limits
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_medical_limited import LimitedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class JointWorker(LimitedWorker):
    def __init__(self,task,gpu,representation):
        super().__init__(task,gpu,context_budget=32,target_calibration_budget=32)
        self.representation=representation
        source=np.arange(self.n_source_cal)
        fit,validation=train_test_split(source,test_size=.5,random_state=0,stratify=self.cal_y[source])
        rng=np.random.default_rng(0)
        groups=[validation[self.cal_y[validation]==c] for c in [0,1]]
        n=min(map(len,groups))
        self.id_validation=np.sort(np.r_[rng.choice(groups[0],n,replace=False),rng.choice(groups[1],n,replace=False)])
        self.source_calibration_fit=np.sort(fit)
        self.calibration_fit=np.r_[self.source_calibration_fit,np.arange(self.n_source_cal,len(self.cal_y))]
        assert not set(self.id_validation).intersection(self.calibration_fit)
        self.projection=None
        if representation=='risk':self.add_common_projection()
        self.cache_dir=main.DATA/self.task/('member_cache_joint_'+representation)
        self.cache_dir.mkdir(exist_ok=True)

    def add_common_projection(self):
        weight=np.r_[np.full(self.ns,.8/self.ns),np.full(self.nt,.2/self.nt)]*len(self.y)
        oof=np.zeros(len(self.y));folds=StratifiedKFold(5,shuffle=True,random_state=0)
        for fit,val in folds.split(self.X,self.y):
            model=LogisticRegression(C=.1,max_iter=500,random_state=0)
            model.fit(self.X[fit],self.y[fit],sample_weight=weight[fit])
            oof[val]=model.decision_function(self.X[val])
        model=LogisticRegression(C=.1,max_iter=500,random_state=0)
        model.fit(self.X,self.y,sample_weight=weight)
        query_score=np.clip(model.decision_function(self.queries),-8,8)
        self.projection=dict(C=.1,coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),
                             training_features_cross_fitted=True,fit_on_context_pool_only=True,
                             validation_and_calibration_labels_used=False,common_to_all_methods=True)
        self.X=np.c_[self.X,np.clip(oof,-8,8)].astype(np.float32)
        self.queries=np.c_[self.queries,query_score].astype(np.float32)
        self.A,self.B,_,_=exp.geometry(self.X,self.queries[self.slices['ood']])
        distance=NearestNeighbors(n_neighbors=10).fit(self.B).kneighbors(self.A)[0].mean(1)
        outlier=np.minimum(self.A*self.A,100).mean(1)
        ids=NearestNeighbors(n_neighbors=11).fit(self.A).kneighbors(self.A,return_distance=False)
        ids=np.stack([row[row!=i][:10] for i,row in enumerate(ids)])
        unstable=np.abs(self.y-self.y[ids].mean(1))
        self.scores=[distance,distance+.35*outlier,unstable+.1*distance,np.zeros(len(self.y)),unstable+.2*outlier]
        self.match=distance
        domain=LogisticRegression(C=.1,max_iter=300,random_state=0)
        domain.fit(np.r_[self.A,self.B],np.r_[np.zeros(len(self.A)),np.ones(len(self.B))])
        p=np.clip(domain.predict_proba(self.A)[:,1],1e-6,1-1e-6)
        self.ratio=np.minimum(p/(1-p)*len(self.A)/len(self.B),10)**.5
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        self.subspace=np.flatnonzero(np.arange(self.X.shape[1])!=corr.argmax())

    def development(self,p,threshold):
        return dict(id=float(np.mean((p[self.id_validation]>=threshold)==self.cal_y[self.id_validation])),
                    ood=self.score(p,threshold))

    def robust_key(self,p,threshold,reference):
        s=self.development(p,threshold)
        changes=[s[d]-reference[d] for d in ['id','ood']]
        return (min(changes),np.mean(changes))

    def fit_weights(self,target_share):
        return np.r_[np.full(len(self.source_calibration_fit),(1-target_share)/len(self.source_calibration_fit)),
                     np.full(self.n_target_cal,target_share/self.n_target_cal)]

    def joint_fuse(self,anchor,bank,fallback,fallback_config,threshold,reference):
        best=fallback;conf=fallback_config.copy();best_key=self.robust_key(best,threshold,reference);trials=[]
        fit=self.calibration_fit
        exp.features=lambda b,k,m:floor_features(b,k,m,.25)
        try:
            for reg in [1.,.1]:
                for share in [.25,.5]:
                    policy=exp.fit_gate(anchor[fit],bank[:,fit],self.cal_y[fit],self.fit_weights(share),len(bank),3.,reg)
                    full=exp.probability(anchor,bank,policy)
                    for cap in [.15,.25]:
                        prediction=(1-cap)*anchor+cap*full
                        score=self.development(prediction,threshold)
                        key=self.robust_key(prediction,threshold,reference)
                        trials.append(dict(regularization=reg,target_share=share,update_cap=cap,
                                           development_accuracy=score,robust_key=list(key)))
                        if key[0]>best_key[0]+1e-12 or (abs(key[0]-best_key[0])<1e-12 and key[1]>best_key[1]+1e-12):
                            best=prediction;best_key=key
                            conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                                      update_cap=cap,target_calibration_share=share)
        finally:exp.features=ORIGINAL_FEATURES
        return best,conf,best_key,trials

    def run_joint(self):
        best_macro=-1;base=None;base_conf=None;base_trials=[]
        for fraction in [0.,1.]:
            rows=self.rows(fraction,np.ones(len(self.y)),0);p=self.predict(rows)
            for share in [.25,.5]:
                threshold=exp.threshold(p[self.calibration_fit],self.cal_y[self.calibration_fit],self.fit_weights(share))
                scores=self.development(p,threshold);macro=np.mean(list(scores.values()))
                base_trials.append(dict(fraction=fraction,target_share=share,threshold=threshold,
                                        development_accuracy=scores,macro_accuracy=macro))
                if macro>best_macro+1e-12:
                    best_macro=macro;base=p;base_conf=dict(policy=dict(kind='base',threshold=threshold),
                        context_fraction=fraction,target_calibration_share=share,actual_target_context_rows=int((rows>=self.ns).sum()))
        threshold=base_conf['policy']['threshold'];base_scores=self.development(base,threshold)
        shared=base;sc=base_conf.copy();best_key=(0.,0.);shared_bank=None;trials=[]
        for fraction in [0.,1.]:
            for strength in [.5,1.]:
                bank=self.bank(fraction,strength)
                p,conf,key,choices=self.joint_fuse(base,bank,shared,sc,threshold,base_scores)
                trials.append(dict(stage='Shared',fraction=fraction,strength=strength,choices=choices))
                if key[0]>best_key[0]+1e-12 or (abs(key[0]-best_key[0])<1e-12 and key[1]>best_key[1]+1e-12):
                    shared=p;sc=dict(conf,context_fraction=fraction,sampling_strength=strength);shared_bank=bank;best_key=key
                print('JOINT_SHARED',self.task,self.representation,fraction,strength,self.development(shared,threshold),flush=True)
        if shared_bank is None:shared_bank=self.bank(base_conf['context_fraction'],.5)
        shared_scores=self.development(shared,threshold)
        specific=shared;pc=sc.copy();specific_key=(0.,0.)
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-corr,kind='stable')
        original=main.ROOT/self.meta['original_inputs']
        numeric=json.loads((original/'metadata.json').read_text())['numeric_features']
        phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int) if self.task.startswith('sepsis') else np.sort(ranked[:64])
        if self.representation=='risk':phenotype=np.unique(np.r_[phenotype,self.X.shape[1]-1])
        for vi,columns in enumerate([np.sort(ranked[:32]),phenotype]):
            for size in [32,64,128,256]:
                near,_=self.local(columns,size)
                a=self.predict(self.rows(base_conf['context_fraction'],np.ones(len(self.y)),201),columns)
                b=self.predict(self.rows(sc.get('context_fraction',base_conf['context_fraction']),np.ones(len(self.y)),202),phenotype)
                bank=np.r_[shared_bank,np.array([near,a,b])]
                p,conf,key,choices=self.joint_fuse(shared,bank,specific,pc,threshold,shared_scores)
                trials.append(dict(stage='Specific',view=vi,context_size=size,choices=choices))
                if key[0]>specific_key[0]+1e-12 or (abs(key[0]-specific_key[0])<1e-12 and key[1]>specific_key[1]+1e-12):
                    specific=p;pc=dict(conf,view=vi,columns=columns.tolist(),context_size=size,anchor='selected Shared');specific_key=key
                print('JOINT_SPECIFIC',self.task,self.representation,vi,size,self.development(specific,threshold),flush=True)
        scores={m:self.development(p,threshold) for m,p in [('Base',base),('Shared',shared),('Specific',specific)]}
        tag='joint_medical_v14' if self.representation=='raw' else 'joint_medical_risk_v15'
        dest=main.ROOT/'results'/tag/self.task
        snapshot=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),task=self.task,
            representation=self.representation,test_labels_opened=False,configurations=dict(Base=base_conf,Shared=sc,Specific=pc),
            development_accuracy=scores,base_trials=base_trials,trials=trials,
            source_calibration_fit_indices=self.source_calibration_fit.tolist(),id_development_indices=self.id_validation.tolist(),
            target_context_indices=self.target_context_indices.tolist(),target_calibration_indices=self.target_calibration_indices.tolist(),
            source_calibration_fit_n=len(self.source_calibration_fit),target_fitting_labels=64,
            id_development_n=len(self.id_validation),ood_development_n=len(self.val_y),
            data=str(main.DATA.relative_to(main.ROOT)/self.task),metadata=self.meta,common_projection=self.projection,
            protocol=json.loads((main.ROOT/'configs/joint_id_ood_development_v14.json').read_text()),
            selected_by='Base best macro accuracy; ensembles maximize worst ID/OOD development gain, then macro gain. No final outcomes used.',
            single_threshold_all_methods=True,all_source_and_target_label_access_matched=True,
            method_status='DAGGER exploratory joint-domain adaptation; source-only paper method not claimed',
            previous_test_patients_observed=True)
        exp.write_json(dest/'frozen_selection.json',snapshot)
        np.savez_compressed(main.DATA/self.task/('joint_'+self.representation+'_predictions.npz'),
                             Base=base,Shared=shared,Specific=specific)
        print('JOINT_FROZEN',self.task,self.representation,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--representation',choices=['raw','risk'],default='raw');args=p.parse_args()
    if args.task=='readmission_other_to_ER':
        main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):
        JointWorker(args.task,args.gpu,args.representation).run_joint()
