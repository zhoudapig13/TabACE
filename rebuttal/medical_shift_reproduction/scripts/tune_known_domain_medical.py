"""Known hospital/admission-domain routing; all methods get the same metadata."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_joint_medical import JointWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class DomainWorker(JointWorker):
    def reference_domain(self,domain):
        reference=self.queries[:self.n_source_cal] if domain=='id' else self.queries[self.slices['ood']]
        self.A,self.B,_,_=exp.geometry(self.X,reference)
        distance=NearestNeighbors(n_neighbors=10).fit(self.B).kneighbors(self.A)[0].mean(1)
        outlier=np.minimum(self.A*self.A,100).mean(1)
        ids=NearestNeighbors(n_neighbors=11).fit(self.A).kneighbors(self.A,return_distance=False)
        ids=np.stack([row[row!=i][:10] for i,row in enumerate(ids)])
        unstable=np.abs(self.y-self.y[ids].mean(1))
        self.match=distance
        self.scores=[distance,distance+.35*outlier,unstable+.1*distance,np.zeros(len(self.y)),unstable+.2*outlier]
        model=LogisticRegression(C=.1,max_iter=300,random_state=0)
        model.fit(np.r_[self.A,self.B],np.r_[np.zeros(len(self.A)),np.ones(len(self.B))])
        p=np.clip(model.predict_proba(self.A)[:,1],1e-6,1-1e-6)
        self.ratio=np.minimum(p/(1-p)*len(self.A)/len(self.B),10)**.5

    def domain_score(self,p,threshold,domain):return self.development(p,threshold)[domain]

    def domain_fuse(self,anchor,bank,fallback,fallback_conf,threshold,domain):
        best=fallback;conf=fallback_conf.copy();score=self.domain_score(best,threshold,domain);trials=[]
        shares=[0.,.25] if domain=='id' else [.5,.75]
        fit=self.calibration_fit
        exp.features=lambda b,k,m:floor_features(b,k,m,.25)
        try:
            for reg in [1.,.1]:
                for share in shares:
                    policy=exp.fit_gate(anchor[fit],bank[:,fit],self.cal_y[fit],self.fit_weights(share),len(bank),3.,reg)
                    full=exp.probability(anchor,bank,policy)
                    for cap in [.15,.25]:
                        p=(1-cap)*anchor+cap*full;value=self.domain_score(p,threshold,domain)
                        trials.append(dict(regularization=reg,target_calibration_share=share,update_cap=cap,
                                           development_accuracy=value))
                        if value>score+1e-12:
                            best=p;score=value;conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                                update_cap=cap,target_calibration_share=share)
        finally:exp.features=ORIGINAL_FEATURES
        return best,conf,score,trials

    def run_domain(self):
        self.cache_dir=main.DATA/self.task/getattr(self,'cache_tag','member_cache_known_domain_raw');self.cache_dir.mkdir(exist_ok=True)
        predictions={};configurations={};scores={m:{} for m in ['Base','Shared','Specific']};trials=[]
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-corr,kind='stable')
        original=main.ROOT/self.meta['original_inputs']
        numeric=json.loads((original/'metadata.json').read_text())['numeric_features']
        phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int) if self.task.startswith('sepsis') else np.sort(ranked[:64])
        for domain in ['id','ood']:
            self.reference_domain(domain)
            best_score=-1;base=None;base_conf=None;base_trials=[]
            for fraction in [0.,1.]:
                rows=self.rows(fraction,np.ones(len(self.y)),0);p=self.predict(rows)
                for share in ([0.,.25] if domain=='id' else [.5,.75]):
                    threshold=exp.threshold(p[self.calibration_fit],self.cal_y[self.calibration_fit],self.fit_weights(share))
                    value=self.domain_score(p,threshold,domain)
                    base_trials.append(dict(fraction=fraction,target_share=share,threshold=threshold,development_accuracy=value))
                    if value>best_score+1e-12:
                        best_score=value;base=p;base_conf=dict(policy=dict(kind='base',threshold=threshold),
                            context_fraction=fraction,target_calibration_share=share,actual_target_context_rows=int((rows>=self.ns).sum()))
            threshold=base_conf['policy']['threshold'];shared=base;sc=base_conf.copy();shared_score=best_score;shared_bank=None
            for fraction in [0.,1.]:
                for strength in [.5,1.]:
                    bank=self.bank(fraction,strength)
                    p,conf,value,choices=self.domain_fuse(base,bank,shared,sc,threshold,domain)
                    trials.append(dict(domain=domain,stage='Shared',fraction=fraction,strength=strength,choices=choices))
                    if value>shared_score+1e-12:
                        shared=p;sc=dict(conf,context_fraction=fraction,sampling_strength=strength);shared_score=value;shared_bank=bank
            if shared_bank is None:shared_bank=self.bank(base_conf['context_fraction'],.5)
            specific=shared;pc=sc.copy();specific_score=shared_score
            for vi,columns in enumerate([np.sort(ranked[:16]),np.sort(ranked[:64]),phenotype]):
                for size in [32,64,128,256]:
                    # Known source hospital: query-local branch may use source-only contexts.
                    nt=self.nt
                    try:
                        if domain=='id':self.nt=0
                        near,_=self.local(columns,size)
                    finally:self.nt=nt
                    a=self.predict(self.rows(base_conf['context_fraction'],np.ones(len(self.y)),201),columns)
                    b=self.predict(self.rows(sc.get('context_fraction',base_conf['context_fraction']),np.ones(len(self.y)),202),phenotype)
                    bank=np.r_[shared_bank,np.array([near,a,b])]
                    p,conf,value,choices=self.domain_fuse(shared,bank,specific,pc,threshold,domain)
                    trials.append(dict(domain=domain,stage='Specific',view=vi,size=size,choices=choices))
                    if value>specific_score+1e-12:
                        specific=p;pc=dict(conf,view=vi,columns=columns.tolist(),local_context_size=size,anchor='selected Shared')
                        specific_score=value
            configurations[domain]=dict(Base=base_conf,Shared=sc,Specific=pc,base_trials=base_trials)
            for name,p in [('Base',base),('Shared',shared),('Specific',specific)]:
                predictions[domain+'_'+name]=p;scores[name][domain]=self.domain_score(p,threshold,domain)
            print('KNOWN_DOMAIN_DEVELOPMENT',self.task,domain,{m:s[domain] for m,s in scores.items()},flush=True)
        out=main.ROOT/'results'/getattr(self,'results_tag','known_domain_medical_v16')/self.task
        exp.write_json(out/'frozen_selection.json',dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),task=self.task,test_labels_opened=False,
            known_domain_metadata_available_to_all=True,domain_routing='Known hospital or admission-source group before clinical outcome',
            configurations=configurations,development_accuracy=scores,trials=trials,
            target_fitting_budget=64,source_calibration_fit_indices=self.source_calibration_fit.tolist(),
            id_development_indices=self.id_validation.tolist(),target_context_indices=self.target_context_indices.tolist(),
            target_calibration_indices=self.target_calibration_indices.tolist(),id_development_n=len(self.id_validation),
            ood_development_n=len(self.val_y),data=str(main.DATA.relative_to(main.ROOT)/self.task),metadata=self.meta,
            protocol=json.loads(getattr(self,'protocol_path',main.ROOT/'configs/known_domain_medical_v16.json').read_text()),
            method_status='DAGGER known-domain adaptation, not domain-blind original TabACE',
            previous_test_patients_observed=True))
        np.savez_compressed(main.DATA/self.task/getattr(self,'predictions_filename','known_domain_raw_predictions.npz'),**predictions)
        print('KNOWN_DOMAIN_FROZEN',self.task,scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.task=='readmission_other_to_ER':
        main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):DomainWorker(args.task,args.gpu,'raw').run_domain()
