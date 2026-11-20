"""Specific-S: selective expert correction with a label-free4.5percent edit cap."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import json
import numpy as np
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from develop_specific_residual import ResidualWorker,calibrate


def selection_features(shared,expert,shared_threshold,expert_threshold):
    s=np.clip(logit(np.clip(shared,1e-6,1-1e-6))-logit(shared_threshold),-8,8)
    t=np.clip(logit(np.clip(expert,1e-6,1-1e-6))-logit(expert_threshold),-8,8)
    return np.c_[s,t,np.abs(s),np.abs(t),np.abs(s-t),(s>=0).astype(float)]


def budgeted_override(shared,expert,win_probability,shared_threshold,expert_threshold,cutoff,budget=.045):
    """No true labels are accepted: budget and ranking depend on predictions only."""
    answer=shared.copy()
    expert_label=expert>=expert_threshold
    eligible=np.flatnonzero((expert_label!=(shared>=shared_threshold))&(win_probability>=cutoff))
    # Random tie-breaking avoids exploiting any case/control ordering in data.
    tie=np.random.default_rng(0).random(len(shared))
    order=np.lexsort((tie[eligible],-win_probability[eligible]))
    chosen=eligible[order[:int(np.floor(budget*len(shared)))]]
    mapped=expit(logit(np.clip(expert[chosen],1e-6,1-1e-6))-logit(expert_threshold)+logit(shared_threshold))
    # Preserve the explicitly defined>= decision at floating-point boundaries.
    target_label=expert_label[chosen]
    mapped[target_label&(mapped<shared_threshold)]=np.nextafter(shared_threshold,1.)
    mapped[(~target_label)&(mapped>=shared_threshold)]=np.nextafter(shared_threshold,0.)
    answer[chosen]=mapped
    changes=int(np.count_nonzero((answer>=shared_threshold)!=(shared>=shared_threshold)))
    assert changes<=int(np.floor(budget*len(shared)))
    return answer,changes


class SelectiveWorker(ResidualWorker):
    def run_selective(self):
        parent=json.loads((main.ROOT/'results/expanded_medical_probability_v18'/self.task/'frozen_selection.json').read_text())
        pf=json.loads((main.ROOT/'results/medical_specific_simple_v22'/self.task/'frozen_selection.json').read_text())
        with np.load(main.DATA/self.task/'expanded_probability_predictions.npz') as z:old={k:z[k] for k in z.files}
        with np.load(main.DATA/self.task/'specific_simple_predictions.npz') as z:experts={k:z[k] for k in z.files}
        self.cache_dir=main.DATA/self.task/'member_cache_known_domain_raw'
        reference={d:self.shared_oof(d,parent,old) for d in ['id','ood']}
        ncal=len(self.cal_y);strata=self.cal_y+2*(np.arange(ncal)>=self.n_source_cal)
        predictions={};configs={};scores={m:{} for m in ['Base','Shared','Specific_S']};trials=[]
        for domain in ['id','ood']:
            shared=old[domain+'_Shared'];threshold=parent['configurations'][domain]['Base']['policy']['threshold']
            best=shared.copy();best_score=self.domain_score(best,threshold,domain)
            best_conf=dict(kind='Shared fallback',threshold=threshold)
            for kind,key in [('TFM_context','TFM_expert'),('Supervised_auxiliary','Auxiliary')]:
                desc=pf['pooling_expert_configurations'][domain][kind]
                full=experts[domain+'_'+key]
                raw=(logit(np.clip(full,1e-10,1-1e-10))-desc['platt_b'])/desc['platt_a']
                for share in ([0.,.25] if domain=='id' else [.5,.75]):
                    weights=self.fit_weights(share);cal_oof=np.zeros(ncal)
                    for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),strata):
                        a,b=calibrate(raw[fit],self.cal_y[fit],weights[fit]);cal_oof[val]=expit(a*raw[val]+b)
                    a,b=calibrate(raw[:ncal],self.cal_y,weights);expert=expit(a*raw+b)
                    tau=exp.threshold(cal_oof,self.cal_y,weights)
                    disagreement=(reference[domain]>=threshold)!=(cal_oof>=tau)
                    features=selection_features(reference[domain],cal_oof,threshold,tau)
                    query_features=selection_features(shared,expert,threshold,tau)
                    target=(cal_oof>=tau)==self.cal_y
                    fit_ids=np.flatnonzero(disagreement&(weights>0))
                    if len(fit_ids)<4:continue
                    local_weights=weights[fit_ids];local_weights=local_weights/local_weights.sum()*len(fit_ids)
                    scaler=StandardScaler().fit(features[fit_ids],sample_weight=local_weights)
                    train=scaler.transform(features[fit_ids]);queries=scaler.transform(query_features)
                    for C in [.1,1.]:
                        if len(np.unique(target[fit_ids]))<2:
                            constant=float((np.average(target[fit_ids],weights=local_weights)*len(fit_ids)+1)/(len(fit_ids)+2))
                            win=np.full(len(shared),constant);model_conf=dict(constant_probability=constant)
                        else:
                            selector=LogisticRegression(C=C,max_iter=500,random_state=0)
                            selector.fit(train,target[fit_ids].astype(int),sample_weight=local_weights)
                            win=selector.predict_proba(queries)[:,1]
                            model_conf=dict(coefficients=selector.coef_[0].tolist(),intercept=float(selector.intercept_[0]),
                                feature_center=scaler.mean_.tolist(),feature_scale=scaler.scale_.tolist())
                        for cutoff in [.5,.65]:
                            pred=shared.copy();counts={}
                            for split in [domain+'_development',domain+'_test']:
                                sl=self.query_slices[split]
                                pred[sl],counts[split]=budgeted_override(shared[sl],expert[sl],win[sl],threshold,tau,cutoff)
                            value=self.domain_score(pred,threshold,domain)
                            trials.append(dict(domain=domain,expert=kind,target_calibration_share=share,
                                selector_C=C,min_probability=cutoff,development_accuracy=value,
                                edits_per_batch=counts,calibration_disagreement_examples=len(fit_ids)))
                            if value>best_score+1e-12:
                                best=pred;best_score=value
                                best_conf=dict(expert=kind,expert_descriptor=desc,platt_a=float(a),platt_b=float(b),
                                    expert_threshold=tau,threshold=threshold,selector_C=C,min_probability=cutoff,
                                    selector=model_conf,target_calibration_share=share,max_edit_fraction=.045,
                                    edits_per_batch=counts,selection_tie_break_seed=0)
            configs[domain]=best_conf
            for name,p in [('Base',old[domain+'_Base']),('Shared',shared),('Specific_S',best)]:
                predictions[domain+'_'+name]=p;scores[name][domain]=self.domain_score(p,threshold,domain)
            print('SELECTIVE_SPECIFIC_DEVELOPMENT',self.task,domain,{m:s[domain] for m,s in scores.items()},flush=True)
        report=dict(parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            selective_specific_configurations=configs,development_accuracy=scores,selective_trials=trials,
            base_shared_unchanged=True,target_fitting_budget=64,max_edit_fraction=.045,
            prediction_budget_uses_no_true_labels=True,
            protocol=json.loads((main.ROOT/'configs/medical_selective_specific_v23.json').read_text()),
            method_status='Specific-S DAGGER learned selective expert correction, not original Specific')
        exp.write_json(main.ROOT/'results/medical_selective_specific_v23'/self.task/'frozen_selection.json',report)
        np.savez_compressed(main.DATA/self.task/'specific_selective_predictions.npz',**predictions)
        print('SELECTIVE_SPECIFIC_FROZEN',self.task,scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):SelectiveWorker(args.task,args.gpu,'raw').run_selective()
