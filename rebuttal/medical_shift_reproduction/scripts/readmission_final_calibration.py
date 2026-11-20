"""Recalibrate the operating point of frozen hybrid scores using OOF calibration."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
import torch
from threadpoolctl import threadpool_limits
from develop_specific_residual import new_learner
from readmission_fresh_specialist import FreshWorker
import run_public_shift as exp


def main():
    w=FreshWorker(0);w.anchors()
    path=exp.ROOT/'results/readmission_fresh_specialist_v32/frozen_selection.json';parent=json.loads(path.read_text())
    with np.load(w.data/'fresh_specialist_predictions_v32.npz') as z:p={k:z[k] for k in z.files}
    best=p['Specific'].copy();best_score=w.score(best,w.threshold)
    best_config=dict(parent['configurations']['Specific']);trials=[]
    pc=parent['configurations']['Specific'];assert pc['learner']['kind']=='logistic'
    domain=w.fit_domains.astype(bool);mass=pc['target_training_mass']
    weight=np.where(domain,mass/domain.sum(),(1-mass)/(~domain).sum())*len(domain)
    model=new_learner(pc['learner']);model.fit(np.c_[w.X,w.fit_domains],w.y,sample_weight=weight)
    lrraw=logit(np.clip(model.predict_proba(np.c_[w.queries,w.query_domains])[:,1],1e-6,1-1e-6))
    rfconf=parent['configurations']['Auxiliary']
    rfraw=(logit(np.clip(p['Auxiliary'],1e-6,1-1e-6))-rfconf['platt_b'])/rfconf['platt_a']
    cs=logit(np.clip(w.shared_oof,1e-6,1-1e-6))-logit(w.threshold)
    s=logit(np.clip(p['Shared'],1e-6,1-1e-6))-logit(w.threshold)
    for name,raw in [('Frozen LR expert',lrraw),('Frozen RF expert',rfraw)]:
        for share in [.8,1.]:
            teacher,oof,tau,calconf=w.calibrated(raw,raw[:w.ncal],share)
            t=logit(np.clip(teacher,1e-6,1-1e-6))-logit(tau)
            ct=logit(np.clip(oof,1e-6,1-1e-6))-logit(tau)
            for alpha in [.2,.4,.6,.8,.9,.95]:
                full=expit((1-alpha)*s+alpha*t+logit(w.threshold))
                cal_oof=expit((1-alpha)*cs+alpha*ct+logit(w.threshold))
                threshold=exp.threshold(cal_oof,w.cal_y,w.weights(share))
                score=w.score(full,threshold)
                config=dict(kind='Extra-supervision hybrid with OOF-calibrated operating threshold',
                    expert=name,alpha=alpha,expert_threshold=tau,calibration=calconf,threshold=threshold,
                    threshold_fitted_from='OOF calibration scores only',development_labels_used_to_fit_threshold=False)
                trials.append(dict(configuration=config,development_accuracy=score))
                if score>best_score+1e-12:best=full;best_score=score;best_config=config
    output=dict(p,Specific=best)
    config=dict(parent['configurations'],Specific=best_config)
    scores={m:w.score(v,config[m]['threshold']) for m,v in output.items()}
    report=dict(parent)
    report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_labels_opened=False,
        configurations=config,development_accuracy=scores,final_threshold_trials=trials,
        base_shared_predictions_bitwise_unchanged=True,method_specific_threshold_explicit=True,
        original_base_threshold_preserved=True,extra_supervision_unchanged=True,
        method_status='Extra-supervision hybrid with independent final threshold calibration; not original TabACE Specific',
        parent_snapshot=str(path.relative_to(exp.ROOT)),parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    exp.write_json(exp.ROOT/'results/readmission_final_calibration_v34/frozen_selection.json',report)
    np.savez_compressed(w.data/'readmission_final_calibration_predictions_v34.npz',**output)
    print('FINAL_OPERATING_POINT_FROZEN',scores,best_config,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):main()
