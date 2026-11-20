"""Refine remaining Specific adapters using the SAME64 fitting labels and capped updates."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_medical_limited import LimitedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class RefineWorker(LimitedWorker):
    def refine(self):
        old=main.ROOT/'results/medical_limited_development_v8'/self.task
        frozen=json.loads((old/'frozen_development_selection.json').read_text())
        with np.load(main.DATA/self.task/'selected_limited_predictions.npz') as z:
            base=z['Base'];shared=z['Shared'];best=z['Specific']
        configurations=frozen['configurations'];shared_conf=configurations['Shared']
        threshold=configurations['Base']['policy']['threshold']
        best_conf=configurations['Specific'].copy();best_score=self.score(best,threshold)
        original=main.ROOT/self.meta['original_inputs']
        numeric=json.loads((original/'metadata.json').read_text())['numeric_features']
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-corr,kind='stable')
        phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int) if self.task.startswith('sepsis') else np.sort(ranked[:64])
        views=[np.sort(ranked[:16]),np.sort(ranked[:64]),np.sort(ranked[:128]),phenotype]
        bank_shared=self.bank(shared_conf.get('context_fraction',configurations['Base']['context_fraction']),
                              shared_conf.get('sampling_strength',.5))
        trials=[]
        for vi,columns in enumerate(views):
            for size in [32,64,128,256]:
                lp,_=self.local(columns,size)
                first=self.predict(self.rows(configurations['Base']['context_fraction'],np.ones(len(self.y)),201),columns)
                second=self.predict(self.rows(shared_conf.get('context_fraction',0),np.ones(len(self.y)),202),phenotype)
                bank=np.r_[bank_shared,np.array([lp,first,second])]
                choices=[]
                exp.features=lambda b,k,m:floor_features(b,k,m,.25)
                try:
                    for reg in [1.,.1]:
                        for share in [.25,.5]:
                            policy=exp.fit_gate(shared[self.slices['cal']],bank[:,self.slices['cal']],
                                                self.cal_y,self.weights(share),len(bank),3.,reg)
                            full=exp.probability(shared,bank,policy)
                            for cap in [.15,.25]:
                                p=(1-cap)*shared+cap*full
                                score=self.score(p,threshold)
                                choices.append(dict(regularization=reg,target_calibration_share=share,
                                                    update_cap=cap,validation_accuracy=score))
                                if score>best_score+1e-12:
                                    best=p;best_score=score
                                    best_conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                                        update_cap=cap,target_calibration_share=share,feature_view=vi,
                                        columns=columns.tolist(),local_context_size=size,anchor='v8 Shared')
                finally:exp.features=ORIGINAL_FEATURES
                trials.append(dict(view=vi,local_context_size=size,choices=choices))
                print('REFINED_SPECIFIC_TRIAL',self.task,vi,size,best_score,flush=True)
        result=dict(frozen)
        result.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
                      refinement='Additional predefined local feature/size candidates; same64 fitting labels, capped25% updates, unchanged Base/Shared and validation rows',
                      refinement_trials=trials)
        result['configurations']=dict(configurations,Specific=best_conf)
        result['development_validation_accuracy']=dict(frozen['development_validation_accuracy'],Specific=best_score)
        dest=main.ROOT/'results/medical_limited_specific_v9'/self.task
        exp.write_json(dest/'frozen_development_selection.json',result)
        np.savez_compressed(main.DATA/self.task/'selected_refined_predictions.npz',Base=base,Shared=shared,Specific=best)
        print('REFINED_DEVELOPMENT_SELECTED',self.task,result['development_validation_accuracy'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args()
    if args.task=='readmission_other_to_ER':
        main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    RefineWorker(args.task,args.gpu,context_budget=32,target_calibration_budget=32).refine()
