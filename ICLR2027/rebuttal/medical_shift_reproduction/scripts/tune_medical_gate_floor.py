"""Confidence-floor ablation/adaptation, ranked on the development set only."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import torch
import run_public_shift as exp
from tune_medical_development import ROOT,DATA
from tune_medical_local import LocalWorker

ORIGINAL_FEATURES=exp.features
BASE_RESULTS=ROOT/'results/target_development_v2'
LOCAL_RESULTS=ROOT/'results/target_development_local_v3'
FLOOR_OUT=ROOT/'results/target_development_floor_v4'


def floor_features(base,bank,m,floor):
    p,q,s,u=ORIGINAL_FEATURES(base,bank,m)
    return p,floor+(1-floor)*q,s,u


class FloorWorker(LocalWorker):
    def floor_fuse(self,anchor,bank,fallback_p,fallback_conf):
        best=fallback_p;conf=fallback_conf.copy();score=self.score(best,conf['policy']['threshold'])
        trials=[]
        for floor in [0.,.25,.5,1.]:
            exp.features=lambda base,members,m:floor_features(base,members,m,floor)
            try:
                for share in [.25,.5,1.]:
                    weight=self.weights(share)
                    policy=exp.fit_gate(anchor[self.slices['cal']],bank[:,self.slices['cal']],
                                        self.cal_y,weight,len(bank),3.,.01)
                    p=exp.probability(anchor,bank,policy)
                    t=exp.threshold(p[self.slices['cal']],self.cal_y,weight)
                    policy.update(threshold=t,confidence_floor=floor)
                    value=self.score(p,t)
                    trials.append(dict(confidence_floor=floor,target_calibration_share=share,
                                       validation_accuracy=value,threshold=t))
                    if value>score+1e-12:
                        best=p;score=value;conf=dict(policy=policy,target_calibration_share=share,
                                                   adapter='confidence floor')
            finally:exp.features=ORIGINAL_FEATURES
        return best,conf,score,trials

    def run_floor(self):
        v2=json.loads((BASE_RESULTS/self.task/'frozen_development_selection.json').read_text())
        v3=json.loads((LOCAL_RESULTS/self.task/'frozen_development_selection.json').read_text())
        with np.load(DATA/self.task/'selected_local_predictions.npz') as z:
            base=z['Base'];shared=z['Shared'];specific=z['Specific']
        base_conf=v2['configurations']['Base'];shared_conf=v2['configurations']['Shared'].copy()
        trials=[];shared_score=self.score(shared,shared_conf['policy']['threshold'])
        best_bank=self.bank(shared_conf.get('context_fraction',base_conf['target_fraction']),
                            shared_conf.get('sampling_strength',.5))
        for fraction in [0.,.25,.5,1.]:
            for strength in [0.,.5,1.]:
                bank=self.bank(fraction,strength)
                p,conf,value,ft=self.floor_fuse(base,bank,shared,shared_conf)
                trials.append(dict(stage='Shared',fraction=fraction,strength=strength,choices=ft))
                if value>shared_score+1e-12:
                    shared=p;shared_conf=dict(conf,context_fraction=fraction,sampling_strength=strength)
                    best_bank=bank;shared_score=value
                print('FLOOR_SHARED_TRIAL',self.task,fraction,strength,value,flush=True)
        previous_conf=v3['configurations']['Specific']
        if self.score(specific,previous_conf['policy']['threshold'])>=shared_score:
            specific_conf=previous_conf.copy()
        else:
            specific=shared;specific_conf=shared_conf.copy()
        specific_score=self.score(specific,specific_conf['policy']['threshold'])
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-association,kind='stable')
        old=ROOT/self.meta['original_inputs']
        numeric=json.loads((old/'metadata.json').read_text())['numeric_features']
        phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int) if self.task.startswith('sepsis') else np.sort(ranked[:32])
        views=[np.sort(ranked[:32]),np.sort(ranked[:64]),np.sort(ranked[:128]),phenotype]
        for ci,columns in enumerate(views):
            for size in [64,128,256]:
                local,_=self.local(columns,size)
                other=self.predict(self.rows(base_conf['target_fraction'],np.ones(len(self.y)),201),columns)
                last=self.predict(self.rows(v2['configurations']['Shared'].get('context_fraction',base_conf['target_fraction']),np.ones(len(self.y)),202),phenotype)
                bank=np.r_[best_bank,np.array([local,other,last])]
                p,conf,value,ft=self.floor_fuse(shared,bank,specific,specific_conf)
                trials.append(dict(stage='Specific',view=ci,local_size=size,choices=ft))
                if value>specific_score+1e-12:
                    specific=p;specific_conf=dict(conf,view=ci,columns=columns.tolist(),local_size=size,anchor='v4 Shared')
                    specific_score=value
                print('FLOOR_SPECIFIC_TRIAL',self.task,ci,size,value,flush=True)
        scores=dict(Base=v2['development_validation_accuracy']['Base'],Shared=shared_score,Specific=specific_score)
        dest=FLOOR_OUT/self.task
        exp.write_json(dest/'frozen_development_selection.json',dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            configurations=dict(Base=base_conf,Shared=shared_conf,Specific=specific_conf),
            development_validation_accuracy=scores,trials=trials,
            no_final_test_metrics_written=True,extra_target_labels=True,
            method_status='DAGGER confidence-floor and query-local development adapters; changed fusion policy',
            confidence_floor_formula='q_adjusted = floor + (1-floor)*q; gate family otherwise unchanged'))
        np.savez_compressed(DATA/self.task/'selected_floor_predictions.npz',Base=base,Shared=shared,Specific=specific)
        print('FLOOR_DEVELOPMENT_SELECTED',self.task,scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    FloorWorker(args.task,args.gpu).run_floor()
