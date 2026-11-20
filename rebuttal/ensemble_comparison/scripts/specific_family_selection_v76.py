"""Dataset-cross-fitted family choice and uncertainty-regularized Specific.

All selection uses the held-out TRAIN calibration. Freeze all family recipes
and the global residual coefficient before opening any final-query bank.
"""
import argparse
import json
import os
from itertools import product
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
import specific_balanced_bags_v76 as bags
from prepare_data import atomic_json,protocol
from run_tabicl import normalize

parent=bags.parent
OUT=bags.OUT
LOCAL_BLENDS=[.25,.5,.75,1.]
GLOBAL_BLENDS=np.round(np.arange(.01,1.001,.01),2).tolist()
RECIPES=[dict(kind='legacy',index=i) for i in [0,1,2]]+[dict(kind='balanced_bags',members=n) for n in [2,4]]


def source(budget):
    cells=parent.parent.source_cells(budget)
    y=np.asarray([c['y_target'] for c in cells]);B=np.asarray([c['base'] for c in cells])
    old=json.loads((parent.OUT/'policies'/str(budget)/'classification/seven/all_policies_frozen.json').read_text())
    choice=old['Shared']['chosen'];Sh=B
    if choice is not None:
        train=np.asarray([c['meta']['priors']['train_shared'] for c in cells]);target=np.asarray([c['meta']['priors']['target_shared'] for c in cells])
        value=parent.correction(np.asarray([c['shared'][0] for c in cells]),train,target,choice['power'],choice['temperature'])
        a=choice['blend'];Sh=normalize((1-a)*B+a*value)
    actual=parent.score(Sh,y);assert all(abs(actual[k]-old['Specific']['source_baseline'][k])<1e-12 for k in actual)
    cohort={str(m['task_id']):m for m in parent.parent.RAW.frozen.cohort('classification','seven')}
    for c in cells:
        meta=c['meta'];tid=str(meta['task_id']);p=OUT/'source'/tid/f'{meta["family"]}_{meta["rho"]}.npz'
        record=json.loads(p.with_suffix('.json').read_text())
        with np.load(p) as z:
            assert np.array_equal(z['y_validation'],c['y_target']);c['bags']=z['predictions']
        assert c['bags'].shape==(4,128,2)
        assert record['source_input_hashes']['X_target']==meta['source_input_hashes']['X_target']
        c['bag_train_priors']=record['train_priors'];c['data_id']=int(cohort[tid]['data_id'])
    return cells,y,B,Sh


def candidate(cells,recipe,power,temperature):
    target=np.asarray([c['meta']['priors']['target'][c['meta']['family']] for c in cells])
    if recipe['kind']=='legacy':
        train=np.asarray([c['meta']['priors']['train'][c['meta']['family']] for c in cells])
        return parent.correction(np.asarray([c['specific'][recipe['index']] for c in cells]),train,target,power,temperature)
    values=[]
    for member in range(recipe['members']):
        train=np.asarray([c['bag_train_priors'][member] for c in cells])
        values.append(parent.correction(np.asarray([c['bags'][member] for c in cells]),train,target,power,1.))
    p=np.mean(values,axis=0);logs=np.log(np.clip(p,1e-12,1.))/temperature
    return normalize(np.exp(logs-logs.max(2,keepdims=True)))


def choose(choices,indices,Sh,Y):
    baseline=parent.score(Sh[indices],Y[indices]);best=None;utility=.5*(baseline['accuracy']+baseline['auc'])
    for index,(_,P) in enumerate(choices):
        for blend in LOCAL_BLENDS:
            p=normalize((1-blend)*Sh[indices]+blend*P[indices]);score=parent.score(p,Y[indices])
            current=.5*(score['accuracy']+score['auc'])
            if score['accuracy']>=baseline['accuracy']-1e-10 and score['auc']>=baseline['auc']-1e-10 and current>utility+1e-10:
                best=dict(candidate_index=index,local_blend=blend,training_scores=score);utility=current
    return best


def run(budget):
    cells,Y,B,Sh=source(budget);datasets=sorted({c['data_id'] for c in cells});folds={d:i%4 for i,d in enumerate(datasets)}
    fold=np.asarray([folds[c['data_id']] for c in cells]);families=np.asarray([c['meta']['family'] for c in cells])
    choices=[]
    for recipe,power,temperature in product(RECIPES,parent.POWERS,parent.TEMPERATURES):
        description=dict(recipe=recipe,power=power,temperature=temperature)
        choices.append((description,candidate(cells,recipe,power,temperature)))
    P=Sh.copy();cv=[];final={}
    for family in parent.parent.expert.FAMILIES:
        group=np.flatnonzero(families==family)
        for held in range(4):
            train=group[fold[group]!=held];valid=group[fold[group]==held]
            assert len(train) and len(valid)
            assert not ({cells[i]['data_id'] for i in train}&{cells[i]['data_id'] for i in valid})
            best=choose(choices,train,Sh,Y)
            if best is not None:
                a=best['local_blend'];P[valid]=normalize((1-a)*Sh[valid]+a*choices[best['candidate_index']][1][valid])
            cv.append(dict(family=family,fold=held,chosen=best,
                train_datasets=sorted({cells[i]['data_id'] for i in train}),held_datasets=sorted({cells[i]['data_id'] for i in valid})))
        best=choose(choices,group,Sh,Y)
        final[family]=None if best is None else dict(**choices[best['candidate_index']][0],local_blend=best['local_blend'])
        print('SPECIFIC_FAMILY_CV_COMPLETE',budget,family,flush=True)
    initial=parent.score(Sh,Y);baseline_acc=np.mean(Sh.argmax(2)==Y,axis=1)
    ids=np.asarray([c['data_id'] for c in cells]);rng=np.random.default_rng(0)
    bootstrap=rng.integers(0,len(datasets),size=(1000,len(datasets)))
    best=None;best_loss=None;trials=[]
    for blend in GLOBAL_BLENDS:
        p=normalize((1-blend)*Sh+blend*P);score=parent.score(p,Y);gain=score['accuracy']-initial['accuracy']
        per_case=np.mean(p.argmax(2)==Y,axis=1)-baseline_acc
        per_data_sum=np.asarray([per_case[ids==d].sum() for d in datasets])
        per_data_count=np.asarray([(ids==d).sum() for d in datasets])
        draws=per_data_sum[bootstrap].sum(1)/per_data_count[bootstrap].sum(1)
        upper=float(np.quantile(draws,.90))
        eligible=bool(.02-1e-10<=gain<=.03+1e-10 and score['auc']>initial['auc']+1e-10)
        # A one-sided dataset-bootstrap uncertainty bound is preferred before
        # distance to the midpoint. This rule is fixed before query evaluation.
        loss=(max(0.,upper-.03),abs(gain-.025),-score['auc'],blend)
        trials.append(dict(blend=blend,source_OOF_scores=score,source_accuracy_gain=gain,upper90_dataset_bootstrap_gain=upper,eligible=eligible))
        if eligible and (best_loss is None or loss<best_loss):best_loss=loss;best=dict(blend=blend)
    folder=OUT/'policies'/str(budget)/'classification/seven';freeze=folder/'all_policies_frozen.json'
    atomic_json(freeze,dict(status='FROZEN_BEFORE_QUERY',Specific=best,family_recipes=final,source_baseline=initial,trials=trials,
        candidates=[x[0] for x in choices],family_dataset_crossfits=cv,Shared_frozen_from_v75=True,
        local_blends=LOCAL_BLENDS,global_blends=GLOBAL_BLENDS,
        bootstrap=dict(seed=0,resamples=1000,unit='whole dataset',one_sided_quantile=.90),
        calibration='v66 held-out TRAIN; family selector excludes the entire held dataset',
        Shared_parent_hyperparameters_remain_frozen_from_prior_TRAIN_calibration=True,
        maximum_new_auxiliary_learners=4,new_auxiliary_unique_row_cap=256,extra_clean_TRAIN_supervision=True,
        query_labels_used_for_fit_or_selection=False,development_only=True,independent_test_claim=False,
        original_TabACE_fusion_reproduction=False,seed=0))
    rows=[]
    for meta in parent.parent.RAW.frozen.cohort('classification','seven'):
        tid=str(meta['task_id']);pool=parent.parent.RAW.scaling.prepare('classification','seven',meta);split=parent.parent.partition(meta)
        descriptor=parent.parent.priors(meta,pool,split,budget,'query');cache={}
        for family,rho in parent.parent.RAW.case_grid('seven'):
            old=parent.OUT/'predictions'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
            with np.load(old) as z:base=z['Base'];shared=z['Shared'];truth=z['y_target']
            info=json.loads(old.with_suffix('.json').read_text());specific=shared
            choice=final[family]
            if best is not None and choice is not None:
                raw,_=parent.parent.RAW.scaling.bank('classification','seven',meta,family,rho);X=raw['X_target']
                hashes=info.get('input_hashes') or info.get('source_input_hashes')
                assert hashes and protocol.array_sha256(X)==hashes['X_target']
                recipe=choice['recipe'];power=choice['power'];temperature=choice['temperature'];target=descriptor['target'][family]
                def predict(path):
                    if str(path) not in cache:cache[str(path)]=joblib.load(path)
                    return parent.parent.expert.predict(X,cache[str(path)],2)
                if recipe['kind']=='legacy':
                    model=parent.parent.expert.OUT/'experts'/str(budget)/'seven'/tid/f'Specific_{family}'/f'query_{recipe["index"]}.joblib'
                    value=parent.correction(predict(model)[None],np.array([descriptor['train'][family]]),np.array([target]),power,temperature)[0]
                else:
                    model_root=OUT/'experts'/tid/family;registry=json.loads((model_root/'models_complete.json').read_text());values=[]
                    for member in range(recipe['members']):
                        q=predict(model_root/f'query_{member}.joblib')
                        values.append(parent.correction(q[None],np.array([registry['query_priors'][member]]),np.array([target]),power,1.)[0])
                    q=np.mean(values,axis=0);logs=np.log(np.clip(q,1e-12,1.))/temperature;value=normalize(np.exp(logs-logs.max(1,keepdims=True)))
                a=choice['local_blend'];candidate_p=normalize((1-a)*shared+a*value)
                a=best['blend'];specific=normalize((1-a)*shared+a*candidate_p)
            out=OUT/'predictions'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
            parent.parent.RAW.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
            info.update(adaptation_version='v76',Shared_bitwise_unchanged_from_v75=True,
                Specific_family_crossfit_plus_balanced_bags=True,extra_clean_TRAIN_supervision=True,
                maximum_additional_auxiliary_learners=4,new_auxiliary_unique_row_cap=256,
                matched_inference_budget_claim=False,query_labels_used_for_fit_or_selection=False,
                methods={name:parent.parent.metrics(truth,p,'classification',1.) for name,p in [('Base',base),('Shared',shared),('Specific',specific)]})
            atomic_json(out.with_suffix('.json'),info)
            for name,values in info['methods'].items():rows.append(dict(context=budget,task='classification',block='seven',method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(folder/'query_detail.csv',index=False)
    atomic_json(folder/'completion.json',dict(status='COMPLETE',conditions=980,Shared_unchanged=True))
    print('SPECIFIC_BALANCED_SELECTION_COMPLETE',budget,best,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,required=True);a=parser.parse_args();run(a.budget)
