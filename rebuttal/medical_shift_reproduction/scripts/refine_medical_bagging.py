"""Fixed eight-draw averaging reduces Specific context-selection variance."""
from datetime import datetime,timezone
import json
import numpy as np
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_medical_limited import LimitedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES


if __name__=='__main__':
    main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    w=LimitedWorker('readmission_other_to_ER',2,context_budget=32,target_calibration_budget=32)
    old=main.ROOT/'results/medical_limited_development_v8'/w.task
    parent=json.loads((old/'frozen_development_selection.json').read_text());cfg=parent['configurations'];sc=cfg['Shared']
    with np.load(main.DATA/w.task/'selected_limited_predictions.npz') as z:base,shared=z['Base'],z['Shared']
    threshold=cfg['Base']['policy']['threshold']
    sbank=w.bank(sc.get('context_fraction',cfg['Base']['context_fraction']),sc.get('sampling_strength',.5))
    rank=np.argsort(np.argsort(w.match,kind='stable'),kind='stable')/max(1,len(w.match)-1)
    weights=[np.exp(-6*rank),w.ratio,np.ones(len(w.y))]
    xc=w.X-w.X.mean(0);yc=w.y-w.y.mean()
    corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
    ranked=np.argsort(-corr,kind='stable')
    draws=np.random.default_rng(0).integers(1000,2**31-1,size=(8,3));families=[]
    for i,seeds in enumerate(draws):
        cols=np.sort(ranked[:[16,32,64,128][i%4]])
        rows=[w.rows(1.,weight,int(seed)) for weight,seed in zip(weights,seeds)]
        families.append(np.asarray([w.predict(rows[0]),w.predict(rows[1]),w.predict(rows[2],cols)]))
    means=np.mean(families,axis=0);bank=np.r_[sbank,means]
    best=shared;best_conf=sc.copy();best_score=w.score(shared,threshold);trials=[]
    exp.features=lambda b,k,m:floor_features(b,k,m,.25)
    try:
        for reg in [1.,.1]:
            for share in [.25,.5]:
                policy=exp.fit_gate(shared[w.slices['cal']],bank[:,w.slices['cal']],w.cal_y,w.weights(share),8,3.,reg)
                full=exp.probability(shared,bank,policy);p=.75*shared+.25*full
                score=w.score(p,threshold)
                trials.append(dict(regularization=reg,target_calibration_share=share,validation_accuracy=score))
                if score>best_score+1e-12:
                    best=p;best_score=score;best_conf=dict(policy=dict(policy,threshold=threshold,confidence_floor=.25),
                        update_cap=.25,target_calibration_share=share,context_draws=8,anchor='v8 Shared',
                        adapter='fixed eight-draw context family means')
    finally:exp.features=ORIGINAL_FEATURES
    scores=dict(parent['development_validation_accuracy'],Specific=best_score)
    result=dict(parent)
    result.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
                  configurations=dict(cfg,Specific=best_conf),development_validation_accuracy=scores,
                  bagging_trials=trials,bagging_count_fixed_before_fit=8,
                  inference_note='Shared: Base+five contexts=6; candidate Specific: Base+five Shared+3families*8draws=30 backbone members. All branches cap256. Fallback deployment may use fewer.',
                  additional_fitting_labels=False,not_compute_matched_to_shared=True)
    dest=main.ROOT/'results/medical_bagged_specific_v13'/w.task
    exp.write_json(dest/'frozen_development_selection.json',result)
    np.savez_compressed(main.DATA/w.task/'selected_bagged_predictions.npz',Base=base,Shared=shared,Specific=best)
    print('BAGGED_DEVELOPMENT_SELECTED',scores,flush=True)
