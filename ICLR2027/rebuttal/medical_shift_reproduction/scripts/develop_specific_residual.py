"""Medical Specific-R: independently developed supervised residual extension.

Base/Shared stay frozen. Auxiliary fitting never sees calibration, development,
or test labels. Development outcomes select configurations; test truth is not read.
"""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_expanded_medical import ExpandedProbability
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


def calibrate(logits,y,weights):
    L,Y,W=[torch.as_tensor(a,dtype=torch.float64) for a in [logits,y,weights/weights.sum()]]
    def obj(x):
        c=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        z=c[0]*L+c[1]
        loss=(W*torch.nn.functional.binary_cross_entropy_with_logits(z,Y,reduction='none')).sum()+.1*((c[0]-1)**2+c[1]**2)
        loss.backward();return float(loss.detach()),c.grad.detach().numpy()
    r=minimize(obj,np.array([1.,0.]),jac=True,method='L-BFGS-B',bounds=[(.05,5),(-5,5)],options={'maxiter':100})
    return r.x


def gate_features(s,t):
    return np.c_[np.ones(len(s)),np.minimum(np.abs(s),6)/6,
                 np.minimum(np.abs(t),6)/6,np.minimum(np.abs(s-t),8)/8]


def fit_residual(s,t,y,weight,cap,reg):
    F,S,T,Y,W=[torch.as_tensor(a,dtype=torch.float64) for a in [gate_features(s,t),s,t,y,weight/weight.sum()]]
    def obj(x):
        c=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        alpha=cap*torch.sigmoid(F@c)
        z=(1-alpha)*S+alpha*T
        loss=(W*torch.nn.functional.binary_cross_entropy_with_logits(z,Y,reduction='none')).sum()+reg*c.square().mean()
        loss.backward();return float(loss.detach()),c.grad.detach().numpy()
    result=minimize(obj,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-8,8)]*4,options={'maxiter':100,'ftol':1e-9})
    return dict(coefficients=result.x.tolist(),alpha_cap=cap,regularization=reg)


def apply_residual(shared,teacher,shared_threshold,teacher_threshold,gate):
    s=logit(np.clip(shared,1e-6,1-1e-6))-logit(shared_threshold)
    t=logit(np.clip(teacher,1e-6,1-1e-6))-logit(teacher_threshold)
    alpha=gate['alpha_cap']*expit(gate_features(s,t)@np.asarray(gate['coefficients']))
    return expit((1-alpha)*s+alpha*t+logit(shared_threshold))


def learner_specs():
    return ([dict(kind='logistic',C=c) for c in [.03,.3,3.]]+
            [dict(kind='forest',depth=d) for d in [4,8]]+
            [dict(kind='lightgbm',depth=d) for d in [2,3,4]])


def new_learner(spec):
    if spec['kind']=='logistic':return LogisticRegression(C=spec['C'],max_iter=1000,random_state=0)
    if spec['kind']=='forest':return RandomForestClassifier(n_estimators=300,max_depth=spec['depth'],
        min_samples_leaf=8,n_jobs=1,random_state=0)
    depth=spec['depth']
    return LGBMClassifier(n_estimators=200 if depth==2 else (250 if depth==3 else 300),
        learning_rate=.03,max_depth=depth,num_leaves=2**depth-1,min_child_samples=32 if depth==4 else 16,
        reg_lambda=10 if depth in [2,4] else 5,n_jobs=1,verbosity=-1,random_state=0,
        deterministic=True,force_col_wise=True)


class ResidualWorker(ExpandedProbability):
    def shared_oof(self,domain,parent,probabilities):
        conf=parent['configurations'][domain]['Shared'];policy=conf['policy']
        base=probabilities[domain+'_Base'];ncal=len(self.cal_y)
        if policy['kind']=='base':return base[:ncal].copy()
        assert policy['kind']=='gate','v20 parent is the conservative probability variant.'
        self.reference_domain(domain)
        fraction=conf.get('context_fraction',parent['configurations'][domain]['Base']['context_fraction'])
        bank=self.bank(fraction,conf.get('sampling_strength',.5))
        full_weights=self.fit_weights(conf.get('target_calibration_share',.25))
        strata=self.cal_y+2*(np.arange(ncal)>=self.n_source_cal)
        folds=StratifiedKFold(5,shuffle=True,random_state=0);oof=np.zeros(ncal)
        exp.features=lambda b,k,m:floor_features(b,k,m,.25)
        try:
            for fit,val in folds.split(np.arange(ncal),strata):
                gate=exp.fit_gate(base[fit],bank[:,fit],self.cal_y[fit],full_weights[fit],len(bank),
                                  policy['gain'],policy['regularization'])
                pred=exp.probability(base[val],bank[:,val],gate)
                oof[val]=(1-conf['update_cap'])*base[val]+conf['update_cap']*pred
        finally:exp.features=ORIGINAL_FEATURES
        return oof

    def run_residual(self):
        parent_path=main.ROOT/'results/expanded_medical_probability_v18'/self.task/'frozen_selection.json'
        parent=json.loads(parent_path.read_text())
        with np.load(main.DATA/self.task/'expanded_probability_predictions.npz') as z:pp={k:z[k] for k in z.files}
        self.cache_dir=main.DATA/self.task/'member_cache_known_domain_raw'
        ncal=len(self.cal_y)
        qdomain=np.zeros(len(self.queries));qdomain[self.n_source_cal:ncal]=1
        qdomain[self.slices['validation']]=1;qdomain[self.slices['ood']]=1
        x=np.c_[self.X,np.r_[np.zeros(self.ns),np.ones(self.nt)]]
        q=np.c_[self.queries,qdomain]
        oof_shared={d:self.shared_oof(d,parent,pp) for d in ['id','ood']}
        predictions={};configs={};aux_configs={};best_score={};best_aux_score={}
        for d in ['id','ood']:
            threshold=parent['configurations'][d]['Base']['policy']['threshold']
            predictions[d+'_Base']=pp[d+'_Base'];predictions[d+'_Shared']=pp[d+'_Shared']
            predictions[d+'_Specific_R']=pp[d+'_Shared'].copy()
            configs[d]=dict(kind='Shared fallback',threshold=threshold)
            best_score[d]=self.domain_score(pp[d+'_Shared'],threshold,d);best_aux_score[d]=-1
        trials=[];model_count=0
        for spec in learner_specs():
            for target_mass in [0.,.2]:
                model_count+=1;model=new_learner(spec)
                if target_mass:
                    fitx,fity=x,self.y
                    fitw=np.r_[np.full(self.ns,(1-target_mass)/self.ns),np.full(self.nt,target_mass/self.nt)]*len(self.y)
                else:fitx,fity=x[:self.ns],self.y[:self.ns];fitw=np.ones(self.ns)
                model.fit(fitx,fity,sample_weight=fitw)
                raw=np.clip(model.predict_proba(q)[:,1],1e-6,1-1e-6)
                all_logits=logit(raw)
                for domain in ['id','ood']:
                    shared=pp[domain+'_Shared'];threshold=parent['configurations'][domain]['Base']['policy']['threshold']
                    domain_choices=[]
                    for share in ([0.,.25] if domain=='id' else [.5,.75]):
                        weights=self.fit_weights(share)
                        strata=self.cal_y+2*(np.arange(ncal)>=self.n_source_cal)
                        cal_oof=np.zeros(ncal)
                        for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),strata):
                            a,b=calibrate(all_logits[fit],self.cal_y[fit],weights[fit])
                            cal_oof[val]=expit(a*all_logits[val]+b)
                        a,b=calibrate(all_logits[:ncal],self.cal_y,weights)
                        teacher=expit(a*all_logits+b)
                        tau=exp.threshold(cal_oof,self.cal_y,weights)
                        aux_score=self.domain_score(teacher,tau,domain)
                        if aux_score>best_aux_score[domain]+1e-12:
                            best_aux_score[domain]=aux_score;predictions[domain+'_Auxiliary']=teacher
                            aux_configs[domain]=dict(learner=spec,target_training_mass=target_mass,platt_a=float(a),
                                platt_b=float(b),threshold=tau,target_calibration_share=share)
                        s=logit(np.clip(oof_shared[domain],1e-6,1-1e-6))-logit(threshold)
                        t=logit(np.clip(cal_oof,1e-6,1-1e-6))-logit(tau)
                        for cap in [.15,.25]:
                            for reg in [.1,.01]:
                                gate=fit_residual(s,t,self.cal_y,weights,cap,reg)
                                probability=apply_residual(shared,teacher,threshold,tau,gate)
                                score=self.domain_score(probability,threshold,domain)
                                domain_choices.append(dict(target_calibration_share=share,alpha_cap=cap,
                                    gate_regularization=reg,specific_development_accuracy=score,auxiliary_development_accuracy=aux_score))
                                if score>best_score[domain]+1e-12:
                                    best_score[domain]=score;predictions[domain+'_Specific_R']=probability
                                    configs[domain]=dict(learner=spec,target_training_mass=target_mass,
                                        platt_a=float(a),platt_b=float(b),teacher_threshold=tau,threshold=threshold,
                                        gate=gate,target_calibration_share=share,
                                        source_auxiliary_training_labels=self.ns,target_auxiliary_training_labels=self.nt if target_mass else 0)
                    trials.append(dict(learner=spec,target_training_mass=target_mass,domain=domain,choices=domain_choices))
                print('SPECIFIC_RESIDUAL_TRIAL',self.task,model_count,'Specific-R',best_score,'Auxiliary',best_aux_score,flush=True)
        out=main.ROOT/'results/medical_specific_residual_v20'/self.task
        scores={m:{d:self.domain_score(predictions[d+'_'+m],
            aux_configs[d]['threshold'] if m=='Auxiliary' else parent['configurations'][d]['Base']['policy']['threshold'],d)
            for d in ['id','ood']} for m in ['Base','Shared','Specific_R','Auxiliary']}
        report=dict(parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            parent_snapshot=str(parent_path.relative_to(main.ROOT)),base_shared_unchanged=True,
            specific_residual_configurations=configs,auxiliary_standalone_configurations=aux_configs,
            development_accuracy=scores,residual_trials=trials,
            protocol=json.loads((main.ROOT/'configs/medical_specific_residual_v20.json').read_text()),
            source_training_labels=self.ns,target_auxiliary_training_budget=32,target_calibration_budget=32,
            auxiliary_model_count=model_count,known_domain_information=True,
            method_status='Specific-R DAGGER independently developed supervised hybrid residual; not original TabACE Specific',
            shared_meta_training_predictions_cross_fitted=True,auxiliary_calibration_predictions_cross_fitted=True)
        exp.write_json(out/'frozen_selection.json',report)
        np.savez_compressed(main.DATA/self.task/'specific_residual_predictions.npz',**predictions)
        print('SPECIFIC_RESIDUAL_FROZEN',self.task,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args();main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):ResidualWorker(args.task,args.gpu,'raw').run_residual()
