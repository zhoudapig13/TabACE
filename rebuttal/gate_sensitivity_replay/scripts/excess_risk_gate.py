"""Cost-sensitive source calibration of whether fusion harms the Base."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from replay_diagnostics import OUT,ROOT,cells,write_json
from correlated_risk_calibration import load,pack,policy_records,second_moment,state

DEST=OUT/'excess_risk_v10'
THRESHOLDS=[.5,.7,.9,.95,.99]
STRENGTHS=[.5,1.]

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite evaluated experiment')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Directly recover naturally occurring shared-error excess loss instead of treating overall R2 gains as subgroup recovery',
        training_label='Original squared error > Base squared error',
        training_weight='Absolute loss difference, normalized for source calibration task variance',
        score='Cost-sensitive harm-risk score, not a calibrated population probability',
        thresholds=THRESHOLDS,attenuation_strengths=STRENGTHS,
        prediction='Base + (1-strength*I(score>=threshold))*(Original-Base)',
        selection='Source inner validation: natural loss<=Original+0.001 and clean loss<=Original+0.005; maximize recovery of excess fusion loss on original joint1 harmful cohort; natural loss tie-break',
        fallback='Original fusion exactly',learner=dict(max_iter=100,max_leaf_nodes=15,max_depth=4,min_samples_leaf=80,l2_regularization=5.,learning_rate=.07,random_state=0),
        held_out_labels_used_for_training_or_selection=False,original_benchmark_unchanged=True,
        backbone_calls=0,markdown_written=False,evaluation_status='Reconstructed development evaluation, not independent confirmation'))

def features(b,k,t,moment):
    x,d,old=state(b,k,t,moment)
    return np.c_[x,np.tanh((k-b[:,None])/3.)].astype(np.float32),old

def fit(data,t,moment):
    b,k,y,w=data;x,p=features(b,k,t,moment)
    difference=(p-y)**2-(b-y)**2;target=(difference>0).astype(int)
    weight=w*np.abs(difference);valid=weight>1e-16
    weight=weight[valid];weight/=weight.mean()
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,max_depth=4,min_samples_leaf=80,
        l2_regularization=5.,learning_rate=.07,early_stopping=False,random_state=0)
    model.fit(x[valid],target[valid],sample_weight=weight)
    return model

def train(backbone,track):
    dataset=load(backbone,track)
    for record in policy_records(backbone,track):
        p=json.loads((ROOT/record['path']).read_text());t=np.asarray(p['parameters'])
        train_ids=set(p['train_data_ids']);test_ids=set(p['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        outer=[c for c in dataset if c['id'] in train_ids];inner=min(c['fold'] for c in outer)
        fitdata=pack([c for c in outer if c['fold']!=inner],track);moment=second_moment(fitdata);model=fit(fitdata,t,moment)
        val=[c for c in outer if c['fold']==inner]
        candidates=[]
        for threshold in THRESHOLDS:
            for strength in STRENGTHS:
                values={}
                for scope,subset in [('natural',[c for c in val if c['meta']['rho']!=0]),('clean',[c for c in val if c['meta']['rho']==0])]:
                    b,k,y,w=pack(subset,track);x,old=features(b,k,t,moment)
                    s=model.predict_proba(x)[:,1];new=b+(1-strength*(s>=threshold))*(old-b)
                    values[scope]=float(np.sum(w*(new-y)**2));values['baseline_'+scope]=float(np.sum(w*(old-y)**2))
                    if scope=='natural':
                        # Construct a fixed original-defined joint1 cohort using each
                        # source condition's target SD, which is retained below.
                        weights=[];mask=[];offset=0
                        for c in subset:
                            n=len(c['idx']);e=(c['k'][c['idx']]-c['y'][c['idx'],None])/max(float(c['y'].std()),1e-12)
                            mask.append((e>1).all(1)|(e<-1).all(1));offset+=n
                        mask=np.concatenate(mask)
                        oldex=np.maximum((old-y)**2-(b-y)**2,0.);newex=np.maximum((new-y)**2-(b-y)**2,0.)
                        values['excess_recovery']=float(np.sum(w[mask]*(oldex[mask]-newex[mask]))/max(float(np.sum(w[mask]*oldex[mask])),1e-12))
                candidates.append(dict(threshold=threshold,strength=strength,**values,
                    feasible=values['natural']<=values['baseline_natural']+.001 and values['clean']<=values['baseline_clean']+.005 and values['excess_recovery']>0))
        feasible=[c for c in candidates if c['feasible']]
        choice=min(feasible,key=lambda c:(-c['excess_recovery'],c['natural'])) if feasible else None
        data=pack(outer,track);moment=second_moment(data);model=fit(data,t,moment)
        dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(model=model,moment=moment,choice=choice),dest/f'fold_{p["fold"]}.joblib')
        write_json(dest/f'fold_{p["fold"]}.json',dict(source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),
            source_candidates=candidates,selected=choice,held_out_labels_used_for_fit=False,original_policy_sha256=record['sha256']))
        print('EXCESS_RISK_FIT',backbone,track,p['fold'],choice,flush=True)
    print('EXCESS_RISK_TRAIN_COMPLETE',backbone,track,flush=True)

def evaluate(backbone,track):
    dest=DEST/backbone/track;heads={f:joblib.load(dest/f'fold_{f}.joblib') for f in range(5)}
    scores=[];budgets=[]
    for i,m,rawb,rawk,rawy,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;y=(rawy-sm)/ss;h=heads[fold]
        x,old=features(b,k,t,h['moment']);choice=h['choice']
        risk=h['model'].predict_proba(x)[:,1]
        new=b+(1-choice['strength']*(risk>=choice['threshold']))*(old-b) if choice else old.copy()
        scale=max(float(rawy.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        for method,v in [('Original',old),('Excess_risk_gate',new)]:
            output=v*ss+sm
            scores.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,n=len(y),
                r2=1-float(np.sum((output-rawy)**2))/max(float(np.sum((rawy-rawy.mean())**2)),1e-12),nrmse=float(np.sqrt(np.mean((output-rawy)**2))/scale)))
        assert abs(float(np.sqrt(np.mean(((old*ss+sm)-rawy)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        if m['rho']==0:continue
        sd=max(float(y.std()),1e-12);base_se=((b-y)/sd)**2;oldse=((old-y)/sd)**2;e=(k-y[:,None])/sd
        masks={'all':np.ones(len(y),bool)}
        for threshold in [1,3]:
            joint=(e>threshold).all(1)|(e<-threshold).all(1);masks[f'joint{threshold}']=joint;masks[f'harmful_joint{threshold}']=joint&(oldse>base_se+.01)
        for method,v in [('Original',old),('Excess_risk_gate',new)]:
            se=((v-y)/sd)**2
            for name,mask in masks.items():
                budgets.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,cohort=name,
                    n_total=len(y),n_cohort=int(mask.sum()),sum_normalized_loss=float(se[mask].sum()),sum_excess_vs_Base=float(np.maximum(se-base_se,0.)[mask].sum()),
                    no_longer_materially_worse_than_Base=int((mask&(se<=base_se+.01)).sum()),
                    improved_vs_original=int((mask&(se<oldse-1e-10)).sum()),worsened_vs_original=int((mask&(se>oldse+1e-10)).sum())))
    pd.DataFrame(scores).to_csv(dest/'performance.csv',index=False);pd.DataFrame(budgets).to_csv(dest/'condition_budgets.csv',index=False)
    write_json(dest/'audit.json',dict(original_frozen_metrics_reproduced=True,held_out_labels_used_for_training=False,query_labels_used_for_scoring_only=True))
    print('EXCESS_RISK_EVALUATION_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['train','evaluate']);p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='train':train(a.backbone,a.track)
    else:evaluate(a.backbone,a.track)
