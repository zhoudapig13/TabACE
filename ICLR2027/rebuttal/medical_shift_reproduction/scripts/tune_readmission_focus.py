"""Readmission-only development: calibration, shrinkage, and local TFM experts.

Uses the fixed v26 adaptation/evaluation partition. Remaining evaluation labels
are never loaded. All unrelated tasks and old results remain unchanged.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
from tune_targeted_medical_paired import PairedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES
from develop_specific_residual import calibrate,new_learner,fit_residual,apply_residual


class ReadmissionWorker(PairedWorker):
    def __init__(self,gpu):
        super().__init__('readmission_other_to_ER','ood',gpu)
        self.parent_path=exp.ROOT/'results/medical_test_assisted_paired_v27/readmission_other_to_ER_ood/frozen_selection.json'
        self.focus_parent=json.loads(self.parent_path.read_text())
        with np.load(self.data/'paired_frozen_predictions_v27.npz') as z:
            self.pp={k:z[k] for k in z.files}
        self.threshold=self.focus_parent['configurations']['Base']['threshold']
        self.trials=[];self.standalone_best=-1;self.standalone=None;self.standalone_config=None

    def parent_shared_oof(self):
        conf=self.focus_parent['configurations']['Shared']
        if conf['kind']=='Frozen original Native-8 Shared':return self.frozen_shared_oof(self.threshold)
        base=self.pp['Base'];oof=base[:self.ncal].copy()
        if conf['kind']=='Base fallback':return oof
        bank=self.bank(conf['use_original_target_context'],conf['sampling_strength'])
        weight=self.weights(conf['calibration_domain_share']);strata=self.cal_y+2*self.cal_domains
        exp.features=lambda b,k,m:floor_features(b,k,m,.25)
        try:
            for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(self.ncal),strata):
                gate=exp.fit_gate(base[fit],bank[:,fit],self.cal_y[fit],weight[fit],len(bank),
                                  conf['policy']['gain'],conf['policy']['regularization'])
                full=exp.probability(base[val],bank[:,val],gate)
                oof[val]=(1-conf['update_cap'])*base[val]+conf['update_cap']*full
        finally:exp.features=ORIGINAL_FEATURES
        return oof

    def calibrated(self,all_logits,cal_logits,share):
        weight=self.weights(share);oof=np.zeros(self.ncal)
        strata=self.cal_y+2*self.cal_domains
        for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(self.ncal),strata):
            a,b=calibrate(cal_logits[fit],self.cal_y[fit],weight[fit])
            oof[val]=expit(a*cal_logits[val]+b)
        a,b=calibrate(cal_logits,self.cal_y,weight)
        full=expit(a*all_logits+b);tau=exp.threshold(oof,self.cal_y,weight)
        return full,oof,tau,dict(platt_a=float(a),platt_b=float(b),expert_threshold=tau,
                                  calibration_domain_share=share)

    def mapped(self,p,tau):
        if abs(tau-self.threshold)<1e-12:return p.copy()
        return expit(logit(np.clip(p,1e-6,1-1e-6))-logit(tau)+logit(self.threshold))

    def consider_expert(self,raw,description):
        logits=logit(np.clip(raw,1e-6,1-1e-6))
        for share in [.5,.8,1.]:
            teacher,oof,tau,calconf=self.calibrated(logits,logits[:self.ncal],share)
            value=self.score(teacher,tau)
            if value>self.standalone_best+1e-12:
                self.standalone_best=value;self.standalone=teacher
                self.standalone_config=dict(description,**calconf,threshold=tau)
            centered=logit(np.clip(self.shared,1e-6,1-1e-6))-logit(self.threshold)
            expert_centered=logit(np.clip(teacher,1e-6,1-1e-6))-logit(tau)
            candidates=[]
            for alpha in getattr(self,'pooling_strengths',[.15,.3,.45]):
                candidates.append((expit((1-alpha)*centered+alpha*expert_centered+logit(self.threshold)),
                                   dict(kind='constant logit pool',alpha=alpha)))
            s=logit(np.clip(self.shared_oof,1e-6,1-1e-6))-logit(self.threshold)
            t=logit(np.clip(oof,1e-6,1-1e-6))-logit(tau)
            for cap in getattr(self,'residual_caps',[.25,.5]):
                gate=fit_residual(s,t,self.cal_y,self.weights(share),cap,.1)
                candidates.append((apply_residual(self.shared,teacher,self.threshold,tau,gate),
                                   dict(kind='cross-fitted residual gate',gate=gate)))
            for p,mix in candidates:
                score=self.score(p,self.threshold);configuration=dict(description,**calconf,**mix,threshold=self.threshold)
                self.trials.append(dict(stage='Specific',configuration=configuration,development_accuracy=score))
                if score>self.specific_score+1e-12:
                    self.specific=p;self.specific_score=score;self.specific_config=configuration

    def retrieval(self):
        x=np.c_[self.X,self.fit_domains];q=np.c_[self.queries,self.query_domains]
        domain=self.fit_domains.astype(bool)
        weights=np.where(domain,.3/domain.sum(),.7/(~domain).sum())*len(self.y)
        forest=RandomForestClassifier(n_estimators=64,max_depth=4,min_samples_leaf=8,n_jobs=1,random_state=0)
        forest.fit(x,self.y,sample_weight=weights)
        pl=forest.apply(x);ql=forest.apply(q)
        anchors=[];nearest=np.zeros(len(x));candidate=int(np.random.default_rng(0).integers(len(x)))
        for _ in range(4):
            anchors.append(candidate);nearest=np.maximum(nearest,(pl==pl[candidate]).mean(1))
            distance=1-nearest;distance[anchors]=-1;candidate=int(distance.argmax())
        anchor=pl[anchors];pool_sim=(pl[:,None,:]==anchor[None,:,:]).mean(2)
        query_sim=(ql[:,None,:]==anchor[None,:,:]).mean(2);groups=query_sim.argmax(1)
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        columns=[np.arange(self.X.shape[1]),np.sort(np.argsort(-association,kind='stable')[:64])]
        for view,cols in enumerate(columns):
            prediction=np.empty(len(q));rowsets=[]
            for j in range(4):
                # Every local context uses exactly 256 unique rows, including
                # the same16 adaptation patients and all32 original target rows.
                source=np.argsort(-pool_sim[:self.ns,j],kind='stable')[:208]
                rows=np.r_[source,np.arange(self.ns,self.ns+self.nt),self.mandatory]
                assert len(rows)==256 and len(np.unique(rows))==256
                qi=np.flatnonzero(groups==j);rowsets.append(rows.tolist())
                if len(qi):prediction[qi]=self.predict(rows,cols)[qi]
            self.consider_expert(prediction,dict(expert='forest-guided query-local Native-8 TFM',
                view=view,columns=cols.tolist(),anchor_count=4,anchors=anchors,context_rows=rowsets,
                context_size=256,forest_parameters=forest.get_params(),extra_label_budget=0))
            print('READMISSION_LOCAL_DEVELOPMENT',view,self.specific_score,flush=True)

    def run_focus(self):
        base=self.pp['Base'];self.shared=self.pp['Shared'].copy()
        self.shared_oof=self.parent_shared_oof();original_oof=self.shared_oof.copy()
        shared_config=dict(kind='Frozen v27 Shared',threshold=self.threshold)
        shared_score=self.score(self.shared,self.threshold)
        raw_logits=logit(np.clip(self.shared,1e-6,1-1e-6))
        oof_logits=logit(np.clip(original_oof,1e-6,1-1e-6))
        for share in [.5,.8,1.]:
            p,oof,tau,conf=self.calibrated(raw_logits,oof_logits,share)
            mapped=self.mapped(p,tau);score=self.score(mapped,self.threshold)
            self.trials.append(dict(stage='Shared calibration',configuration=conf,development_accuracy=score))
            if score>shared_score+1e-12:
                self.shared=mapped;self.shared_oof=self.mapped(oof,tau);shared_score=score
                shared_config=dict(conf,kind='Cross-fitted calibration of frozen five-context Shared',threshold=self.threshold)
        self.specific=self.shared.copy();self.specific_score=shared_score
        self.specific_config=dict(kind='Shared fallback',threshold=self.threshold)
        prior_score=self.score(self.pp['Specific'],self.threshold)
        if prior_score>self.specific_score+1e-12:
            self.specific=self.pp['Specific'].copy();self.specific_score=prior_score
            self.specific_config=dict(kind='Frozen v27 Specific',threshold=self.threshold,
                                      full_configuration=self.focus_parent['configurations']['Specific'])
        print('READMISSION_FROZEN_ANCHORS',dict(Base=self.score(base,self.threshold),Shared=shared_score,
                                               Specific=self.specific_score),flush=True)
        domain=self.fit_domains.astype(bool)
        x=np.c_[self.X,self.fit_domains];q=np.c_[self.queries,self.query_domains]
        model_count=0
        for c in [.001,.003,.01,.03]:
            for mass in [0.,.2,.5]:
                selected=np.ones(len(self.y),dtype=bool) if mass else ~domain
                weight=np.where(domain,mass/domain.sum(),(1-mass)/(~domain).sum())*selected.sum()
                model=new_learner(dict(kind='logistic',C=c))
                model.fit(x[selected],self.y[selected],sample_weight=weight[selected]);model_count+=1
                self.consider_expert(model.predict_proba(q)[:,1],dict(expert='strongly regularized logistic',
                    C=c,target_training_mass=mass,fit_patients=int(selected.sum()),extra_label_budget=0))
                print('READMISSION_REGULARIZATION',model_count,'Specific',self.specific_score,
                      'standalone',self.standalone_best,flush=True)
        self.retrieval()
        output=dict(Base=base,Shared=self.shared,Specific=self.specific,Auxiliary=self.standalone,
                    Mean5=self.pp['Mean5'])
        assert np.array_equal(output['Base'],self.pp['Base'])
        assert all(len(p)==len(self.queries) and np.isfinite(p).all() for p in output.values())
        scores={m:self.score(p,self.standalone_config['threshold'] if m=='Auxiliary' else
                  (self.focus_parent['configurations']['Mean5']['threshold'] if m=='Mean5' else self.threshold))
                for m,p in output.items()}
        report=dict(self.focus_parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_labels_opened=False,
            development_accuracy=scores,trials=self.trials,
            configurations=dict(Base=self.focus_parent['configurations']['Base'],Shared=shared_config,
                Specific=self.specific_config,Auxiliary=self.standalone_config,
                Mean5=self.focus_parent['configurations']['Mean5']),
            parent_snapshot=str(self.parent_path.relative_to(exp.ROOT)),
            parent_sha256=hashlib.sha256(self.parent_path.read_bytes()).hexdigest(),
            original_base_predictions_bitwise_unchanged=True,other_tasks_not_run=True,
            new_logistic_learners=model_count,new_local_expert_configurations=2,
            all_tfm_contexts_exactly256=True,no_additional_labels_beyond_v26=True,
            method_status='Medical development adaptation; calibration and extra supervised retrieval/auxiliary experts, not original TabACE',
            script_sha256_at_freeze=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        exp.write_json(exp.ROOT/'results/readmission_focused_v28/frozen_selection.json',report)
        np.savez_compressed(self.data/'readmission_focused_predictions_v28.npz',**output)
        np.savez_compressed(self.data/'readmission_focused_calibration_oof_v28.npz',Shared=self.shared_oof)
        print('READMISSION_FOCUS_FROZEN',scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):ReadmissionWorker(a.gpu).run_focus()
