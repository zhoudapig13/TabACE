"""Finite development-only search over Specific context diversity; no extra labels."""
from datetime import datetime,timezone
import json
import numpy as np
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_medical_limited import LimitedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


class DiversityWorker(LimitedWorker):
    def run_diversity(self):
        old=main.ROOT/'results/medical_limited_specific_v9'/self.task
        frozen=json.loads((old/'frozen_development_selection.json').read_text())
        cfg=frozen['configurations'];sc=cfg['Shared'];threshold=cfg['Base']['policy']['threshold']
        with np.load(main.DATA/self.task/'selected_refined_predictions.npz') as z:
            base=z['Base'];shared=z['Shared'];best=z['Specific']
        best_conf=cfg['Specific'].copy();best_score=self.score(best,threshold)
        sbank=self.bank(sc.get('context_fraction',cfg['Base']['context_fraction']),sc.get('sampling_strength',.5))
        rank=np.argsort(np.argsort(self.match,kind='stable'),kind='stable')/max(1,len(self.match)-1)
        weights=[np.exp(-6*rank),self.ratio,np.ones(len(self.y))]
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-corr,kind='stable')
        seed_stream=np.random.default_rng(0).integers(1000,2**31-1,size=(32,3))
        trials=[]
        for draw,seeds in enumerate(seed_stream):
            columns=np.sort(ranked[:[16,32,64,128][draw%4]])
            rows=[self.rows(1.,w,int(seed)) for w,seed in zip(weights,seeds)]
            extras=np.asarray([self.predict(rows[0]),self.predict(rows[1]),self.predict(rows[2],columns)])
            bank=np.r_[sbank,extras];choices=[]
            exp.features=lambda b,k,m:floor_features(b,k,m,.25)
            try:
                for reg in [1.,.1]:
                    for share in [.25,.5]:
                        policy=exp.fit_gate(shared[self.slices['cal']],bank[:,self.slices['cal']],
                                            self.cal_y,self.weights(share),len(bank),3.,reg)
                        full=exp.probability(shared,bank,policy)
                        for cap in [.15,.25]:
                            p=(1-cap)*shared+cap*full;score=self.score(p,threshold)
                            choices.append(dict(regularization=reg,target_calibration_share=share,
                                                update_cap=cap,validation_accuracy=score))
                            if score>best_score+1e-12:
                                best=p;best_score=score
                                best_conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                                    update_cap=cap,target_calibration_share=share,draw=draw,context_draw_seeds=seeds.tolist(),
                                    extra_context_rows=[x.tolist() for x in rows],feature_columns=columns.tolist(),
                                    anchor='v8 Shared',adapter='stochastic context diversity, master experiment seed0')
            finally:exp.features=ORIGINAL_FEATURES
            trials.append(dict(draw=draw,context_draw_seeds=seeds.tolist(),choices=choices))
            print('DIVERSITY_TRIAL',self.task,draw,best_score,flush=True)
        result=dict(frozen)
        result.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
                      context_diversity_trials=trials,
                      refinement='32 predeclared context draws from master experiment seed0; unchanged fitting64 labels, Base, Shared, threshold and25percent cap')
        result['configurations']=dict(cfg,Specific=best_conf)
        result['development_validation_accuracy']=dict(frozen['development_validation_accuracy'],Specific=best_score)
        dest=main.ROOT/'results/medical_context_diversity_v12'/self.task
        exp.write_json(dest/'frozen_development_selection.json',result)
        np.savez_compressed(main.DATA/self.task/'selected_diversity_predictions.npz',Base=base,Shared=shared,Specific=best)
        print('DIVERSITY_SELECTED',self.task,result['development_validation_accuracy'],flush=True)


if __name__=='__main__':
    main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    DiversityWorker('readmission_other_to_ER',2,context_budget=32,target_calibration_budget=32).run_diversity()
