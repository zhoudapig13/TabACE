"""TRAIN-only augmentation experts for the frozen classification development set.

Shared trains across the registered shift families; Specific additionally
trains family-specific experts. No target severity is supplied at inference.
All labels used for augmentation are clean TRAIN labels, never final-query
labels. The TFM context/predictions and the query benchmark stay unchanged.
The additional supervision and known-generator augmentation are substantial
adaptations and must not be presented as an original TabACE reproduction.
"""
import argparse
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import probability_head_v51 as nn
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import metrics,normalize

OUT=RUN/'results/tabace_augmented_classifier_v54'
PARENT=RUN/'results/tabace_source_expert_v53'
PROFILES=[dict(kind='catboost',iterations=256,depth=6),dict(kind='catboost',iterations=512,depth=8),dict(kind='extra_trees',trees=384)]
BLENDS=[.1,.25,.5,.75,1.]
FAMILIES=['covariate','label','concept','selection','domain','temporal','support']


def intervention(X,y,H,family,rho,cfg):
    """The target-side seven-shift law, applied to TRAIN observations only."""
    s=rho/100.;Q=X.copy();truth=y.copy()
    if family=='label':
        original=float(truth.mean());extreme=.9 if original>=.5 else .1
        if cfg['label_prior_direction']==-1:extreme=1-extreme
        prior=original+s*cfg['label_prior_displacement_coefficient']*(extreme-original)
        rng=np.random.default_rng(0);labels=(rng.random(len(truth))<prior).astype(int)
        if len(np.unique(truth))>1:
            ids=np.asarray([rng.choice(np.flatnonzero(truth==label)) for label in labels]);Q,truth=Q[ids],truth[ids]
    if family in ['concept','temporal']:
        j=int(np.argmax(H.var(0)));order=np.argsort(np.abs(Q[:,j]-np.median(H[:,j])),kind='stable')
        ids=order[:int(np.ceil(cfg['max_boundary_flip_fraction']*s*len(truth)))];truth[ids]=1-truth[ids]
    if family in ['domain','temporal']:
        mean=H.mean(0);sd=np.maximum(H.std(0),1e-6)
        Q=mean+(Q-mean)*(1+cfg['affine_scale_increment']*s)+cfg['affine_translation_sd']*s*sd
    if family=='support':
        sd=np.maximum(H.std(0),1e-6);ids=np.argsort(H.var(0))[-min(3,H.shape[1]):]
        Q[:,ids]+=cfg['support_translation_sd']*s*sd[ids]
    return np.asarray(Q,dtype=np.float32),truth


def training_arrays(pool,partition,budget,block,role):
    if block=='clean':X,y,ids=pool['X'],pool['y'],pool['source_ids'];H=None
    else:
        X=np.vstack([pool['H'],pool['R']]);y=np.r_[pool['hy'],pool['ry']];ids=np.r_[pool['h_ids'],pool['r_ids']]
        keep=partition['h_keep'][:len(pool['H0'])] if role=='validation' else np.ones(len(pool['H0']),bool)
        H=pool['H0'][keep]
    positions=np.arange(len(y))
    if role=='validation':positions=positions[~np.isin(ids,partition['validation_original_ids'])]
    positions=positions[:budget];assert len(np.unique(ids[positions]))==len(positions)<=budget
    assert not np.intersect1d(ids[positions],pool['query_ids']).size
    if role=='validation':assert not np.intersect1d(ids[positions],partition['validation_original_ids']).size
    return X[positions],y[positions],ids[positions],H


def augment(X,y,H,block,families):
    if block=='clean':return X,y,np.ones(len(y))
    cfg=nn.source.adaptation.raw.frozen.configuration()['classification_seven']['generator_parameters']
    conditions=[intervention(X,y,H,f,r,cfg) for f in families for r in [20,50,80,100]]
    A=np.vstack([c[0] for c in conditions]);b=np.concatenate([c[1] for c in conditions])
    return A,b,np.full(len(b),1./len(conditions))


def fit(X,y,w,profile):
    if len(np.unique(y))==1:return dict(constant=int(y[0]))
    if profile['kind']=='catboost':
        model=CatBoostClassifier(iterations=profile['iterations'],depth=profile['depth'],learning_rate=.05,
            loss_function='Logloss' if len(np.unique(y))==2 else 'MultiClass',l2_leaf_reg=5.,random_seed=0,
            task_type='GPU',devices='0',thread_count=2,verbose=False,allow_writing_files=False,
            gpu_ram_part=.65,border_count=64)
    else:model=ExtraTreesClassifier(n_estimators=profile['trees'],min_samples_leaf=1,max_features='sqrt',random_state=0,n_jobs=2)
    model.fit(X,y,sample_weight=w);return dict(estimator=model)


def predict(X,model,classes):
    out=np.zeros((len(X),classes),dtype=np.float64)
    if 'constant' in model:out[:,model['constant']]=1.
    else:
        estimator=model['estimator'];p=estimator.predict_proba(X);out[:,np.asarray(estimator.classes_,dtype=int)]=p
    return normalize(out)


def make_experts(budget,block,limit=None):
    raw=nn.source.adaptation.raw;metas=raw.frozen.cohort('classification',block)
    for meta in metas[:limit] if limit else metas:
        tid=str(meta['task_id']);track='five' if block=='clean' else 'seven'
        folder=OUT/'experts'/str(budget)/block/tid
        pool=raw.scaling.prepare('classification',track,meta)
        path=raw.OUT/'validation_splits/classification'/track/f'{tid}.npz'
        with np.load(path) as z:partition={k:z[k] for k in z.files}
        X,y,ids,H=training_arrays(pool,partition,budget,block,'validation')
        Xq,yq,idsq,Hq=training_arrays(pool,partition,budget,block,'query')
        cells=[c for c in nn.load('validation',budget,block) if str(c['meta']['task_id'])==tid] if limit else None
        if cells is None:
            cells=[]
            for f,r in raw.case_grid(block):
                p=nn.source.OUT/'banks/validation'/str(budget)/'classification'/block/tid/f'{f}_{r}.npz'
                record=json.loads(p.with_suffix('.json').read_text())
                data,_=nn.source.adaptation.validation.materialize('classification',block,meta,f,r,budget,pool,partition)
                assert protocol.array_sha256(data['X_target'])==record['source_input_hashes']['X_target']
                with np.load(p) as z:assert np.array_equal(data['y_target'],z['y_target']);k=z['base'].shape[1]
                cells.append(dict(meta=record,X=data['X_target'],y=data['y_target'],classes=k))
        else:
            for c in cells:
                data,_=nn.source.adaptation.validation.materialize('classification',block,meta,c['meta']['family'],c['meta']['rho'],budget,pool,partition)
                assert protocol.array_sha256(data['X_target'])==c['meta']['source_input_hashes']['X_target']
                c.update(X=data['X_target'],classes=c['base'].shape[1])
        roles=['Shared']+[f'Specific_{f}' for f in (['original'] if block=='clean' else FAMILIES)]
        for role in roles:
            target=folder/role/'source_predictions.npz'
            if target.exists() and target.with_suffix('.json').exists():continue
            families=FAMILIES if role=='Shared' else [role.removeprefix('Specific_')]
            A,b,w=augment(X,y,H,block,families);Af,bf,wf=augment(Xq,yq,Hq,block,families)
            scope=cells if role=='Shared' else [c for c in cells if c['meta']['family']==families[0]]
            indices=[0] if role=='Shared' else list(range(len(PROFILES)))
            values={};paths={}
            for index in indices:
                profile=PROFILES[index];model=fit(A,b,w,profile)
                vpath=folder/role/f'validation_{index}.joblib';vpath.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,vpath,compress=3)
                for c in scope:values[f'{index}_{c["meta"]["family"]}_{c["meta"]["rho"]}']=predict(c['X'],model,c['classes'])
                final=fit(Af,bf,wf,profile);mpath=folder/role/f'query_{index}.joblib';joblib.dump(final,mpath,compress=3);paths[str(index)]=str(mpath)
            raw.scaling.npwrite(target,**values,training_original_ids=ids,validation_original_ids=partition['validation_original_ids'])
            atomic_json(target.with_suffix('.json'),dict(status='SOURCE_EXPERTS_COMPLETE',role=role,query_model_paths=paths,
                source_unique_rows=len(ids),final_unique_rows=len(idsq),source_augmented_rows=len(b),budget=budget,
                source_validation_disjoint=True,query_overlap=0,extra_clean_TRAIN_supervision=True,
                source_hashes={key:protocol.array_sha256(a) for key,a in [('X',X),('y',y),('H',H if H is not None else X[:0])]},
                known_generator_augmentation=block=='seven',query_strength_given=False,query_labels_used=False))
            print('AUGMENTED_SOURCE_ROLE_COMPLETE',budget,block,tid,role,flush=True)


def source_predictions(cells,budget,block,role,profile):
    result={};cache={}
    for c in cells:
        tid=str(c['meta']['task_id']);actual=role if role=='Shared' else f'Specific_{c["meta"]["family"]}'
        p=OUT/'experts'/str(budget)/block/tid/actual/'source_predictions.npz'
        if str(p) not in cache:
            with np.load(p) as z:cache[str(p)]={k:z[k] for k in z.files}
        result[c['key']]=cache[str(p)][f'{profile}_{c["meta"]["family"]}_{c["meta"]["rho"]}']
    return result


def select(cells,baseline,budget,block,role):
    initial,best_value=nn.score(cells,baseline);best=None;best_predictions=baseline;trials=[]
    for index in ([0] if role=='Shared' else range(len(PROFILES))):
        experts=source_predictions(cells,budget,block,role,index)
        for blend in BLENDS:
            combined={c['key']:normalize((1-blend)*baseline[c['key']]+blend*experts[c['key']]) for c in cells}
            values,utility=nn.score(cells,combined)
            eligible=values['accuracy']>=initial['accuracy']-1e-10 and (values['auc'] is None or values['auc']>=initial['auc']-1e-10)
            trials.append(dict(profile_index=index,blend=blend,source_scores=values,utility=utility,eligible=eligible))
            if eligible and utility>best_value+1e-10:best_value=utility;best=dict(profile_index=index,blend=blend);best_predictions=combined
    return dict(chosen=best,source_baseline=initial,trials=trials),best_predictions


def run(budget,block):
    make_experts(budget,block);cells=nn.load('validation',budget,block)
    sh,shpred=select(cells,{c['key']:c['base'] for c in cells},budget,block,'Shared');specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        specific[family],_=select(group,shpred,budget,block,'Specific')
    root=OUT/'policies'/str(budget)/'classification'/block
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Shared=sh,Specific=specific,
        extra_clean_TRAIN_supervision=True,known_generator_augmentation=block=='seven',Shared_family_blind=True,
        Specific_family_visible=True,actual_strength_input=False,seed=0,checkpoint_finetuned=False,
        original_query_labels_used_for_fit=False,development_only=True,original_paper_replication=False))
    query=nn.load('query',budget,block);cache={};rows=[]
    for c in query:
        meta=c['meta'];data,_=nn.source.adaptation.raw.scaling.bank('classification',block,meta,meta['family'],meta['rho'])
        X=data['X_target'];assert protocol.array_sha256(X)==meta['source_input_hashes']['X_target']
        def apply(policy,role,anchor):
            best=policy['chosen']
            if best is None:return anchor
            path=OUT/'experts'/str(budget)/block/str(meta['task_id'])/role/f'query_{best["profile_index"]}.joblib'
            if str(path) not in cache:cache[str(path)]=joblib.load(path)
            value=predict(X,cache[str(path)],c['base'].shape[1]);a=best['blend'];return normalize((1-a)*anchor+a*value)
        shared=apply(sh,'Shared',c['base']);sp=apply(specific[meta['family']],f'Specific_{meta["family"]}',shared)
        out=OUT/'predictions'/str(budget)/c['key'];nn.source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=shared,Specific=sp,y_target=c['y'])
        record={key:meta[key] for key in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        record.update(input_hashes=meta['source_input_hashes'],source_bank=str(c['path']),seed=0,provisional_Base_calibration=False,
            methods={n:metrics(c['y'],p,'classification',1.) for n,p in [('Base',c['base']),('Shared',shared),('Specific',sp)]},
            adaptation_version='v54',extra_clean_TRAIN_supervision=True,known_generator_augmentation=block=='seven',
            original_query_labels_used_for_fit=False,development_only=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),record)
        for name,v in record['methods'].items():rows.append(dict(method=name,context=budget,block=block,task_id=meta['task_id'],**v))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(query),adaptation_marker='dagger'))
    print('AUGMENTED_CLASSIFIER_GROUP_COMPLETE',budget,block,len(query),flush=True)


def plan():
    assert not (OUT/'plan.json').exists();nn.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_CLASSIFIER_EXPERT_TRAINING',seed=0,budgets=[256,1024,4096,8192],
        refit=['classification clean','classification seven'],full_goal_checks=48,profiles=PROFILES,blends=BLENDS,
        augmentation='seven target-side laws applied ONLY to disjoint TRAIN labels; all four registered strengths',
        Shared_training='one family-blind model on equal-weight union of 28 TRAIN transformations',
        Specific_training='family-specific union of four strengths; complete Shared fallback',
        source_calibration='entire existing TRAIN calibration set excluded from auxiliary source fits',
        query_recipe_refit='same capped clean TRAIN pool; no query X/y in fits',source_unique_row_cap='requested context budget',
        extra_clean_TRAIN_supervision=True,known_generator_augmentation=True,
        query_feedback_used_for_development_design=True,query_labels_used_for_fit_or_selection=False,
        Base_and_query_arrays_unchanged=True,checkpoint_finetuned=False,development_only=True,
        original_paper_replication=False,pure_TabACE_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((PARENT/'paper_targets.json').read_text())
    for budget in [256,1024,4096,8192]:
        for task,blocks in [('classification',['five']),('regression',['clean','five','seven'])]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.symlink_to(PARENT/name/str(budget)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit','smoke']);p.add_argument('--budget',type=int,default=256);p.add_argument('--block',choices=['clean','seven'],default='seven');a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='smoke':make_experts(a.budget,a.block,1)
    else:run(a.budget,a.block)
