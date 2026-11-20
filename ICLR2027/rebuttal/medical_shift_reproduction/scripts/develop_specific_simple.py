"""Low-complexity score fusion: no fitted residual gate and no extra supervision."""
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
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from develop_specific_context import ContextWorker
from develop_specific_residual import calibrate


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):
        w=ContextWorker(args.task,args.gpu,'raw')
        root=main.ROOT
        parent=json.loads((root/'results/expanded_medical_probability_v18'/args.task/'frozen_selection.json').read_text())
        cf=json.loads((root/'results/medical_specific_context_v21'/args.task/'frozen_selection.json').read_text())
        rf=json.loads((root/'results/medical_specific_residual_v20'/args.task/'frozen_selection.json').read_text())
        with np.load(main.DATA/args.task/'expanded_probability_predictions.npz') as z:base={k:z[k] for k in z.files}
        with np.load(main.DATA/args.task/'specific_residual_predictions.npz') as z:rpred={k:z[k] for k in z.files}
        ncal=len(w.cal_y);domain_flag=np.zeros(len(w.queries));domain_flag[w.n_source_cal:ncal]=1
        domain_flag[w.slices['validation']]=1;domain_flag[w.slices['ood']]=1
        x=np.c_[w.X,np.r_[np.zeros(w.ns),np.ones(w.nt)]];q=np.c_[w.queries,domain_flag]
        forest=RandomForestClassifier(n_estimators=64,max_depth=4,min_samples_leaf=8,n_jobs=1,random_state=0)
        weights=np.r_[np.full(w.ns,.8/w.ns),np.full(w.nt,.2/w.nt)]*len(w.y)
        forest.fit(x,w.y,sample_weight=weights);pl=forest.apply(x);ql=forest.apply(q)
        w.context_cache=main.DATA/args.task/'forest_context_cache_v21'
        anchors=[];near=np.zeros(len(x));candidate=int(np.random.default_rng(0).integers(len(x)))
        for _ in range(16):
            anchors.append(candidate);near=np.maximum(near,(pl==pl[candidate]).mean(1))
            distance=1-near;distance[anchors]=-1;candidate=int(distance.argmax())
        output={};configs={};expert_configs={};scores={m:{} for m in ['Base','Shared','Specific_P']};trials=[]
        for domain in ['id','ood']:
            shared=base[domain+'_Shared'];threshold=parent['configurations'][domain]['Base']['policy']['threshold']
            cc=cf['context_specific_configurations'][domain]
            selected_anchors=cc.get('anchors',anchors)
            size=cc.get('context_size',128)
            raw,_=w.retrieve(pl,ql,selected_anchors,size,domain)
            rawlogit=logit(np.clip(raw,1e-6,1-1e-6));best_c_score=-1
            for share in ([0.,.25] if domain=='id' else [.5,.75]):
                weight=w.fit_weights(share);oof=np.zeros(ncal)
                strata=w.cal_y+2*(np.arange(ncal)>=w.n_source_cal)
                for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),strata):
                    a,b=calibrate(rawlogit[fit],w.cal_y[fit],weight[fit]);oof[val]=expit(a*rawlogit[val]+b)
                a,b=calibrate(rawlogit[:ncal],w.cal_y,weight);expert=expit(a*rawlogit+b)
                tau=exp.threshold(oof,w.cal_y,weight);value=w.domain_score(expert,tau,domain)
                if value>best_c_score+1e-12:
                    best_c_score=value;c_expert=expert;c_tau=tau;c_config=dict(platt_a=float(a),platt_b=float(b),
                        threshold=tau,anchors=selected_anchors,context_size=size,target_calibration_share=share)
            r_expert=rpred[domain+'_Auxiliary'];r_tau=rf['auxiliary_standalone_configurations'][domain]['threshold']
            output[domain+'_TFM_expert']=c_expert
            output[domain+'_Auxiliary']=r_expert
            expert_configs[domain]=dict(TFM_context=c_config,Supervised_auxiliary=rf['auxiliary_standalone_configurations'][domain])
            best=shared;best_score=w.domain_score(shared,threshold,domain);configuration=dict(kind='Shared fallback',threshold=threshold)
            centered=logit(np.clip(shared,1e-6,1-1e-6))-logit(threshold)
            for name,expert,tau in [('TFM_context',c_expert,c_tau),('Supervised_auxiliary',r_expert,r_tau)]:
                expert_centered=logit(np.clip(expert,1e-6,1-1e-6))-logit(tau)
                for alpha in [.15,.25,.35]:
                    p=expit((1-alpha)*centered+alpha*expert_centered+logit(threshold))
                    value=w.domain_score(p,threshold,domain)
                    trials.append(dict(domain=domain,expert=name,alpha=alpha,development_accuracy=value))
                    if value>best_score+1e-12:
                        best=p;best_score=value;configuration=dict(expert=name,alpha=alpha,threshold=threshold,
                            expert_threshold=tau,context_calibration=c_config if name=='TFM_context' else None,
                            auxiliary_configuration=rf['auxiliary_standalone_configurations'][domain] if name=='Supervised_auxiliary' else None)
            configs[domain]=configuration
            for m,p in [('Base',base[domain+'_Base']),('Shared',shared),('Specific_P',best)]:
                output[domain+'_'+m]=p;scores[m][domain]=w.domain_score(p,threshold,domain)
            print('SIMPLE_SPECIFIC_DEVELOPMENT',args.task,domain,{m:s[domain] for m,s in scores.items()},flush=True)
        report=dict(parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            simple_specific_configurations=configs,pooling_expert_configurations=expert_configs,
            development_accuracy=scores,simple_fusion_trials=trials,
            base_shared_unchanged=True,target_fitting_budget=64,
            protocol=json.loads((root/'configs/medical_specific_simple_v22.json').read_text()),
            method_status='Specific-P DAGGER simple score pooling development adaptation, not original Specific')
        exp.write_json(root/'results/medical_specific_simple_v22'/args.task/'frozen_selection.json',report)
        np.savez_compressed(main.DATA/args.task/'specific_simple_predictions.npz',**output)
        print('SIMPLE_SPECIFIC_FROZEN',args.task,scores,flush=True)
