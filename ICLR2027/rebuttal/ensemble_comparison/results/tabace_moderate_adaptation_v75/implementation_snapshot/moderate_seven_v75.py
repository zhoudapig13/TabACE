"""Choose moderate seven-shift classifier residuals on TRAIN calibration only."""
import argparse
import json
import os
from itertools import product
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import balanced_seven_v66 as parent
import moderate_adaptation_v75 as common
from prepare_data import atomic_json,protocol
from run_tabicl import normalize

OUT=common.OUT
POWERS=[0.,.5,1.]
TEMPERATURES=[.7,1.,1.4]
BLENDS=[.01,.02,.05,.1,.15,.2,.25,.3,.4,.5,.6,.75,.9,1.]


def score(P,y):
    P=normalize(P)
    accuracy=float(np.mean(P.argmax(2)==y))
    auc=[]
    for label in [0,1]:
        truth=y==label;n=truth.sum(1);other=y.shape[1]-n
        ranks=rankdata(P[:,:,label],axis=1,method='average')
        numerator=(ranks*truth).sum(1)-n*(n+1)/2
        values=np.divide(numerator,n*other,out=np.full(len(n),np.nan),where=(n*other)>0)
        auc.append(values)
    per_case=np.mean(auc,axis=0)
    return dict(accuracy=accuracy,auc=float(np.nanmean(per_case)))


def correction(P,train,target,power,temperature):
    ratio=np.column_stack([1-target,target])/np.column_stack([1-train,train])
    logs=(np.log(np.clip(P,1e-12,1.))+power*np.log(ratio)[:,None,:])/temperature
    return normalize(np.exp(logs-logs.max(2,keepdims=True)))


def select(cells,y,baseline,role):
    initial=score(baseline,y);best=None;best_predictions=baseline;best_loss=None;trials=[]
    profiles=[0] if role=='shared' else [0,1,2]
    gain_range=(1e-9,.02) if role=='shared' else (.02,.03)
    gain_target=.01 if role=='shared' else .025
    for index,power,temperature in product(profiles,POWERS,TEMPERATURES):
        P=np.asarray([c[role][index] for c in cells]);train=[];target=[]
        for c in cells:
            descriptor=c['meta']['priors'];family=c['meta']['family']
            train.append(descriptor['train_shared'] if role=='shared' else descriptor['train'][family])
            target.append(descriptor['target_shared'] if role=='shared' else descriptor['target'][family])
        value=correction(P,np.asarray(train),np.asarray(target),power,temperature)
        for blend in BLENDS:
            predictions=normalize((1-blend)*baseline+blend*value)
            measured=score(predictions,y);gain=measured['accuracy']-initial['accuracy']
            eligible=bool(gain_range[0]-1e-10<=gain<=gain_range[1]+1e-10 and measured['auc']>initial['auc']+1e-10)
            loss=(abs(gain-gain_target),-measured['auc'],blend)
            trials.append(dict(profile_index=index,power=power,temperature=temperature,blend=blend,
                source_scores=measured,source_accuracy_gain=gain,eligible=eligible))
            if eligible and (best_loss is None or loss<best_loss):
                best_loss=loss;best=dict(profile_index=index,power=power,temperature=temperature,blend=blend)
                best_predictions=predictions
    return dict(chosen=best,source_baseline=initial,trials=trials,source_gain_range=list(gain_range),source_gain_target=gain_target),best_predictions


def run(budget):
    cells=parent.source_cells(budget)
    assert len(cells)==980
    y=np.asarray([c['y_target'] for c in cells]);base=np.asarray([c['base'] for c in cells])
    assert y.shape==(980,128) and base.shape==(980,128,2)
    # AUC is ranked independently for both columns, matching the repository's
    # two one-vs-rest calls, including ties and degenerate-label cases.
    for P in [base,np.asarray([c['shared'][0] for c in cells]),np.asarray([c['specific'][2] for c in cells])]:
        reference=parent.score(cells,{c['key']:p for c,p in zip(cells,P)})[0]
        actual=score(P,y)
        assert all(abs(reference[k]-actual[k])<1e-12 for k in actual),(reference,actual)
    shared,source_sh=select(cells,y,base,'shared')
    specific,_=select(cells,y,source_sh,'specific')
    folder=common.root(budget,'classification','seven');freeze=folder/'all_policies_frozen.json'
    atomic_json(freeze,dict(status='FROZEN_BEFORE_QUERY',Shared=shared,Specific=specific,seed=0,
        Shared_Acc_gain_cap=.02,Specific_Acc_gain_over_Shared_range=[.02,.03],AUC_improvement_required_on_source=True,
        source_calibration='v66 balanced held-out TRAIN; no new independent labels',
        Shared_family_blind=True,Specific_family_visible=True,actual_strength_input=False,
        fast_score_checked_against_repository_metrics=True,query_labels_used_for_fit_or_selection=False,
        auxiliary_models_frozen_from_v54=True,extra_clean_TRAIN_supervision=True,
        final_query_bounds_not_guaranteed_by_source_selection=True))
    rows=[]
    for meta in parent.RAW.frozen.cohort('classification','seven'):
        tid=str(meta['task_id']);pool=parent.RAW.scaling.prepare('classification','seven',meta);split=parent.partition(meta)
        descriptor=parent.priors(meta,pool,split,budget,'query');cache={}
        for family,rho in parent.RAW.case_grid('seven'):
            origin=parent.expert.nn.source.OUT/'banks/query'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
            record=json.loads(origin.with_suffix('.json').read_text())
            with np.load(origin) as z:b=z['base'];truth=z['y_target']
            data,_=parent.RAW.scaling.bank('classification','seven',meta,family,rho);X=data['X_target']
            assert protocol.array_sha256(X)==record['source_input_hashes']['X_target']
            anchor=b;values={}
            for name,role,policy in [('Shared','Shared',shared),('Specific',f'Specific_{family}',specific)]:
                choice=policy['chosen'];p=anchor
                if choice is not None:
                    model_path=parent.expert.OUT/'experts'/str(budget)/'seven'/tid/role/f'query_{choice["profile_index"]}.joblib'
                    if str(model_path) not in cache:cache[str(model_path)]=joblib.load(model_path)
                    value=parent.expert.predict(X,cache[str(model_path)],2)
                    train=descriptor['train_shared'] if name=='Shared' else descriptor['train'][family]
                    target=descriptor['target_shared'] if name=='Shared' else descriptor['target'][family]
                    value=correction(value[None],np.asarray([train]),np.asarray([target]),choice['power'],choice['temperature'])[0]
                    a=choice['blend'];p=normalize((1-a)*anchor+a*value)
                values[name]=p;anchor=p
            record.update(hybrid_auxiliary_classifiers=True,
                deployed_fusion='frozen Native1 backbone plus TRAIN-augmented auxiliary classifier residuals',
                original_TabACE_fusion_reproduction=False)
            c=dict(key=f'classification/seven/{tid}/{family}_{rho}.npz',y=truth,meta=record)
            rows+=common.write_query(c,b,values['Shared'],values['Specific'],record,budget,'classification','seven',freeze)
    pd.DataFrame(rows).to_csv(folder/'query_detail.csv',index=False)
    atomic_json(folder/'completion.json',dict(status='COMPLETE',conditions=980,source_only_selection=True))
    print('MODERATE_CLASSIFICATION_SEVEN_COMPLETE',budget,shared['chosen'],specific['chosen'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,required=True);args=parser.parse_args();run(args.budget)
