"""Add independently selected forest-only controls before any v21 final evaluation."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import json
import numpy as np
from scipy.special import logit,expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import torch
import tune_medical_development as main
import tune_medical_local as local
from tune_expanded_medical import ExpandedProbability
from develop_specific_residual import calibrate
import run_public_shift as exp


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);args=parser.parse_args()
    main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        w=ExpandedProbability(args.task,0,'raw')
        path=main.ROOT/'results/medical_specific_context_v21'/args.task/'frozen_selection.json'
        report=json.loads(path.read_text());ncal=len(w.cal_y)
        qdomain=np.zeros(len(w.queries));qdomain[w.n_source_cal:ncal]=1
        qdomain[w.slices['validation']]=1;qdomain[w.slices['ood']]=1
        x=np.c_[w.X,np.r_[np.zeros(w.ns),np.ones(w.nt)]];q=np.c_[w.queries,qdomain]
        forest=RandomForestClassifier(n_estimators=64,max_depth=4,min_samples_leaf=8,n_jobs=1,random_state=0)
        weights=np.r_[np.full(w.ns,.8/w.ns),np.full(w.nt,.2/w.nt)]*len(w.y)
        forest.fit(x,w.y,sample_weight=weights);rawlogit=logit(np.clip(forest.predict_proba(q)[:,1],1e-6,1-1e-6))
        with np.load(main.DATA/args.task/'specific_context_predictions.npz') as z:p={k:z[k] for k in z.files}
        configurations={};scores={}
        strata=w.cal_y+2*(np.arange(ncal)>=w.n_source_cal)
        for domain in ['id','ood']:
            best=-1;choices=[]
            for share in ([0.,.25] if domain=='id' else [.5,.75]):
                weight=w.fit_weights(share);oof=np.zeros(ncal)
                for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),strata):
                    a,b=calibrate(rawlogit[fit],w.cal_y[fit],weight[fit]);oof[val]=expit(a*rawlogit[val]+b)
                a,b=calibrate(rawlogit[:ncal],w.cal_y,weight);probability=expit(a*rawlogit+b)
                threshold=exp.threshold(oof,w.cal_y,weight);score=w.domain_score(probability,threshold,domain)
                choices.append(dict(target_calibration_share=share,threshold=threshold,development_accuracy=score))
                if score>best+1e-12:
                    best=score;p[domain+'_Forest']=probability
                    configurations[domain]=dict(platt_a=float(a),platt_b=float(b),threshold=threshold,
                                                target_calibration_share=share,forest_parameters=forest.get_params())
            configurations[domain]['development_choices']=choices;scores[domain]=best
        report['forest_standalone_configurations']=configurations
        report['forest_reference_frozen_at_utc']=datetime.now(timezone.utc).isoformat()
        report['development_accuracy']['Forest']=scores
        report['forest_reference_not_used_to_reselect_specific']=True
        path.write_text(json.dumps(report,indent=2))
        np.savez_compressed(main.DATA/args.task/'specific_context_predictions.npz',**p)
        print('FOREST_REFERENCE_FROZEN',args.task,scores)
