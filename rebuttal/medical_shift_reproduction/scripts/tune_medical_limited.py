"""Reduced target supervision and capped updates; development metrics only."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class LimitedWorker(local.LocalWorker):
    def limited_fuse(self,anchor,bank,fallback,fallback_conf,threshold):
        best=fallback;conf=fallback_conf.copy();value=self.score(best,threshold);trials=[]
        exp.features=lambda base,members,m:floor_features(base,members,m,.25)
        try:
            for reg in [1.,.1]:
                policy=exp.fit_gate(anchor[self.slices['cal']],bank[:,self.slices['cal']],
                                    self.cal_y,self.weights(.25),len(bank),3.,reg)
                full=exp.probability(anchor,bank,policy)
                for cap in [.15,.25]:
                    prediction=(1-cap)*anchor+cap*full
                    score=self.score(prediction,threshold)
                    trials.append(dict(regularization=reg,update_cap=cap,validation_accuracy=score))
                    if score>value+1e-12:
                        best=prediction;value=score
                        conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                                  update_cap=cap,adapter='reduced-supervision capped update')
        finally:exp.features=ORIGINAL_FEATURES
        return best,conf,value,trials

    def run_limited(self):
        best_score=-1;base=None;base_conf=None;base_trials=[]
        for fraction in [0.,1.]:
            rows=self.rows(fraction,np.ones(len(self.y)),0)
            p=self.predict(rows)
            t=exp.threshold(p[self.slices['cal']],self.cal_y,self.weights(.25))
            score=self.score(p,t)
            base_trials.append(dict(requested_target_fraction=fraction,actual_target_context_rows=int((rows>=self.ns).sum()),
                                    threshold=t,validation_accuracy=score))
            if score>best_score+1e-12:
                base=p;best_score=score;base_conf=dict(policy=dict(kind='base',threshold=t),
                    context_fraction=fraction,actual_target_context_rows=int((rows>=self.ns).sum()))
        threshold=base_conf['policy']['threshold'];shared=base;shared_conf=base_conf.copy();shared_score=best_score
        trials=[];best_bank=None
        for fraction in [0.,1.]:
            for strength in [.5,1.]:
                bank=self.bank(fraction,strength)
                p,conf,score,ft=self.limited_fuse(base,bank,shared,shared_conf,threshold)
                trials.append(dict(stage='Shared',fraction=fraction,strength=strength,choices=ft))
                if score>shared_score+1e-12:
                    shared=p;shared_conf=dict(conf,context_fraction=fraction,sampling_strength=strength)
                    shared_score=score;best_bank=bank
                print('LIMITED_SHARED_TRIAL',self.task,fraction,strength,score,flush=True)
        if best_bank is None:best_bank=self.bank(base_conf['context_fraction'],.5)
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-association,kind='stable')
        old=main.ROOT/self.meta['original_inputs']
        numeric=json.loads((old/'metadata.json').read_text())['numeric_features']
        phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int) if self.task.startswith('sepsis') else np.sort(ranked[:64])
        views=[np.sort(ranked[:32]),phenotype]
        specific=shared;specific_conf=shared_conf.copy();specific_score=shared_score
        for vi,columns in enumerate(views):
            for size in [128,256]:
                query_local,rows=self.local(columns,size)
                a=self.predict(self.rows(base_conf['context_fraction'],np.ones(len(self.y)),201),columns)
                b=self.predict(self.rows(shared_conf.get('context_fraction',base_conf['context_fraction']),np.ones(len(self.y)),202),phenotype)
                bank=np.r_[best_bank,np.array([query_local,a,b])]
                p,conf,score,ft=self.limited_fuse(shared,bank,specific,specific_conf,threshold)
                trials.append(dict(stage='Specific',view=vi,local_context_size=size,choices=ft))
                if score>specific_score+1e-12:
                    specific=p;specific_conf=dict(conf,view=vi,columns=columns.tolist(),local_context_size=size,
                                                  anchor='limited Shared')
                    specific_score=score
                print('LIMITED_SPECIFIC_TRIAL',self.task,vi,size,score,flush=True)
        dest=main.ROOT/'results/medical_limited_development_v8'/self.task
        scores=dict(Base=best_score,Shared=shared_score,Specific=specific_score)
        exp.write_json(dest/'frozen_development_selection.json',dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            configurations=dict(Base=base_conf,Shared=shared_conf,Specific=specific_conf),
            development_validation_accuracy=scores,base_trials=base_trials,trials=trials,
            metadata=self.meta,extra_target_context_labels=self.nt,
            extra_target_calibration_labels=self.n_target_cal,
            validation_labels_used_for_selection=len(self.val_y),source_calibration_labels=self.n_source_cal,
            shared_and_specific_use_same_base_threshold=True,
            protocol=json.loads((main.ROOT/'configs/medical_limited_development_v8.json').read_text()),
            selected_by='development-validation accuracy only; no target test or numeric-band filter',
            method_status='DAGGER64 labeled target fitting patients, query-local and capped-gate development adapters',
            test_patients_previously_observed=True))
        np.savez_compressed(main.DATA/self.task/'selected_limited_predictions.npz',Base=base,Shared=shared,Specific=specific,
                            target_context_fit_indices=self.target_context_indices,
                            target_calibration_indices=self.target_calibration_indices)
        print('LIMITED_DEVELOPMENT_SELECTED',self.task,scores,'target_fitting_labels',self.nt+self.n_target_cal,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args()
    if args.task=='readmission_other_to_ER':
        main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5'
        local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    LimitedWorker(args.task,args.gpu,context_budget=32,target_calibration_budget=32).run_limited()
