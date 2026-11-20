"""Cross-fitted member calibration, without expanding labels or update caps."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_medical_limited import LimitedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class CalibrationWorker(LimitedWorker):
    def calibrate(self,bank,C):
        ncal=len(self.cal_y);corrected=bank.copy();parameters=[]
        folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),self.cal_y))
        weight=self.weights(.5)*ncal
        for i,values in enumerate(bank):
            predictor=logit(np.clip(values,1e-6,1-1e-6))[:,None]
            for fit,val in folds:
                model=LogisticRegression(C=C,max_iter=300,random_state=0)
                model.fit(predictor[fit],self.cal_y[fit],sample_weight=weight[fit])
                corrected[i,val]=model.predict_proba(predictor[val])[:,1]
            model=LogisticRegression(C=C,max_iter=300,random_state=0)
            model.fit(predictor[:ncal],self.cal_y,sample_weight=weight)
            corrected[i,ncal:]=model.predict_proba(predictor[ncal:])[:,1]
            parameters.append(dict(coefficient=float(model.coef_[0,0]),intercept=float(model.intercept_[0])))
        return corrected,parameters

    def run_calibration(self):
        parent=main.ROOT/'results/medical_limited_specific_v9'/self.task
        frozen=json.loads((parent/'frozen_development_selection.json').read_text())
        configs=frozen['configurations'];shared_conf=configs['Shared']
        with np.load(main.DATA/self.task/'selected_refined_predictions.npz') as z:
            base=z['Base'];shared=z['Shared'];best=z['Specific']
        threshold=configs['Base']['policy']['threshold'];best_conf=configs['Specific'].copy()
        best_score=self.score(best,threshold)
        shared_bank=self.bank(shared_conf.get('context_fraction',configs['Base']['context_fraction']),
                              shared_conf.get('sampling_strength',.5))
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-association,kind='stable')
        views=[np.sort(ranked[:16]),np.sort(ranked[:32]),np.sort(ranked[:64]),np.sort(ranked[:128])]
        trials=[]
        for vi,columns in enumerate(views):
            query_local,_=self.local(columns,64)
            first=self.predict(self.rows(configs['Base']['context_fraction'],np.ones(len(self.y)),201),columns)
            second=self.predict(self.rows(shared_conf.get('context_fraction',0),np.ones(len(self.y)),202),np.sort(ranked[:64]))
            raw_bank=np.r_[shared_bank,np.array([query_local,first,second])]
            for C in [.1,1.,10.]:
                bank,calibrators=self.calibrate(raw_bank,C);choices=[]
                exp.features=lambda b,k,m:floor_features(b,k,m,.25)
                try:
                    for reg in [1.,.1]:
                        for share in [.25,.5]:
                            policy=exp.fit_gate(shared[self.slices['cal']],bank[:,self.slices['cal']],
                                               self.cal_y,self.weights(share),len(bank),3.,reg)
                            full=exp.probability(shared,bank,policy)
                            for cap in [.15,.25]:
                                p=(1-cap)*shared+cap*full;score=self.score(p,threshold)
                                choices.append(dict(regularization=reg,target_calibration_share=share,update_cap=cap,
                                                    validation_accuracy=score))
                                if score>best_score+1e-12:
                                    best=p;best_score=score
                                    best_conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                                        update_cap=cap,member_calibration_C=C,member_calibrators=calibrators,
                                        view=vi,columns=columns.tolist(),local_context_size=64,anchor='v8 Shared',
                                        target_calibration_share=share,adapter='cross-fitted member calibration')
                finally:exp.features=ORIGINAL_FEATURES
                trials.append(dict(view=vi,calibration_C=C,choices=choices))
                print('MEMBER_CALIBRATION_TRIAL',self.task,vi,C,best_score,flush=True)
        result=dict(frozen)
        result.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
                      member_calibration_trials=trials,
                      refinement='5fold member Platt calibration, same64 labels and25percent update cap')
        result['configurations']=dict(configs,Specific=best_conf)
        result['development_validation_accuracy']=dict(frozen['development_validation_accuracy'],Specific=best_score)
        dest=main.ROOT/'results/medical_limited_member_calibration_v10'/self.task
        exp.write_json(dest/'frozen_development_selection.json',result)
        np.savez_compressed(main.DATA/self.task/'selected_calibrated_predictions.npz',Base=base,Shared=shared,Specific=best)
        print('MEMBER_CALIBRATION_SELECTED',self.task,result['development_validation_accuracy'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
    main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    CalibrationWorker('readmission_other_to_ER',args.gpu,context_budget=32,target_calibration_budget=32).run_calibration()
