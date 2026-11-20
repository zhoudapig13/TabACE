"""Core-prior weighted original gates, with dataset-OOF post-calibration.

Keep the existing TFM context banks and the original confidence/agreement/Base
gate. Address TRAIN calibration class-prior mismatch in its loss, and test a
source-selected posterior-prior/temperature transform. Every validation task's
entire dataset is excluded from its gate fit. Query scores never choose policy
parameters; Accuracy and AUC use the same saved probability array.
"""
import argparse
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
import reliability_fusion_v33 as gate
import prior_fusion_v61 as prior
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_weighted_gate_v62'
TEMPERATURES=[.7,1.,1.4,2.]
PRIOR_POWERS=[0.,.25,.5,.75,1.]
BLENDS=[.25,.5,1.]
RAW_TENSORS=gate.fusion.training_tensors


def label_weights(c):
    y=c['y'].astype(int);p=np.bincount(y,minlength=len(c['metric_core_prior']))/len(y)
    w=c['metric_core_prior'][y]/p[y];return w/w.mean()


def tensors(cells,m):
    data=RAW_TENSORS(cells,m)
    w=np.concatenate([label_weights(c) for c in cells]);assert len(w)==len(data['weights'])
    data['weights']=data['weights']*torch.as_tensor(w,dtype=torch.float64)
    return data


def temperatures(cells,m):
    losses=np.zeros((len(TEMPERATURES),m));counts={}
    for c in cells:counts[c['meta']['task_id']]=counts.get(c['meta']['task_id'],0)+1
    for c in cells:
        logs=np.log(np.clip(c['bank'][:m],1e-12,1.));truth=c['y'].astype(int);w=label_weights(c)
        for i,t in enumerate(TEMPERATURES):
            z=logs/t;nll=logsumexp(z,axis=2)-z[:,np.arange(len(truth)),truth]
            losses[i]+=(nll*w[None]).mean(1)/(len(counts)*counts[c['meta']['task_id']])
    return np.asarray(TEMPERATURES)[losses.argmin(0)].tolist()


def configure():
    torch.set_num_threads(1);torch.manual_seed(0)
    # Do not call gate.configure(): it redirects the shared reference module's
    # OUT, which would also redirect v51's immutable validation split lookups.
    gate.fusion.GATE_VARIANT='entropy_or_disagreement'
    gate.fusion.mapping=gate.mapping
    gate.fusion.training_tensors=tensors;gate.temperatures=temperatures


def load(role,budget):
    cells=prior.load(role,budget)
    for c in cells:c.update(task='classification',role=role)
    return cells


def fit_raw(cells,budget,m,name):
    reference=prior.risk.head.source.OUT/'policies'/str(budget)/'classification/clean/Shared.json'
    old=json.loads(reference.read_text())['model']
    assert old['kind']=='gated_fusion'
    gain,regularization=old['gain'],old['regularization']
    datasets=sorted({int(c['meta']['data_id']) for c in cells});groups={d:i%4 for i,d in enumerate(datasets)}
    oof={};root=OUT/'policies'/str(budget)/'classification/clean'/name
    for fold in range(4):
        train=[c for c in cells if groups[int(c['meta']['data_id'])]!=fold]
        valid=[c for c in cells if groups[int(c['meta']['data_id'])]==fold]
        assert train and valid
        assert not ({c['meta']['data_id'] for c in train}&{c['meta']['data_id'] for c in valid})
        model=gate.fit(train,'classification',m,gain,regularization)
        atomic_json(root/f'fold_{fold}.json',dict(model=model,train_datasets=sorted({int(c['meta']['data_id']) for c in train}),
            held_datasets=sorted({int(c['meta']['data_id']) for c in valid}),source_only=True))
        for c in valid:oof[c['key']]=gate.predict(c,model)
    final=gate.fit(cells,'classification',m,gain,regularization)
    atomic_json(root/'raw_model.json',dict(status='FROZEN_BEFORE_QUERY',model=final,
        inherited_fixed_gain=gain,inherited_fixed_regularization=regularization,
        dataset_group_folds=4,source_loss_core_prior_weighted=True))
    return final,oof


def transform(c,p,power,temperature):
    z=(np.log(np.clip(p,1e-12,1.))+power*np.log(c['core_prior']/c['context_prior']))/temperature
    return normalize(np.exp(z-z.max(1,keepdims=True)))


def select(cells,raw_predictions,baseline,folder):
    initial,best_value=prior.score(cells,baseline);best=None;best_pred=baseline;trials=[]
    for source,preds in [('gated',raw_predictions),('base',{c['key']:c['base'] for c in cells})]:
        for power in PRIOR_POWERS:
            for temperature in TEMPERATURES:
                values={c['key']:transform(c,preds[c['key']],power,temperature) for c in cells}
                for blend in BLENDS:
                    mixed={c['key']:normalize((1-blend)*baseline[c['key']]+blend*values[c['key']]) for c in cells}
                    scores,utility=prior.score(cells,mixed)
                    eligible=scores['accuracy']>=initial['accuracy']-1e-10 and scores['auc']>=initial['auc']-1e-10
                    trials.append(dict(source=source,prior_power=power,temperature=temperature,blend=blend,
                        source_weighted_scores=scores,utility=utility,eligible=eligible))
                    if eligible and utility>best_value+1e-10:
                        best_value=utility;best=dict(source=source,prior_power=power,temperature=temperature,blend=blend);best_pred=mixed
    atomic_json(folder/'selection.json',dict(status='FROZEN_BEFORE_QUERY',chosen=best,baseline=initial,trials=trials,
        dataset_OOF_gate_predictions=True,query_labels_used=False))
    return best,best_pred


def apply(c,raw_model,choice,anchor):
    if choice is None:return anchor
    p=gate.predict(c,raw_model) if choice['source']=='gated' else c['base']
    value=transform(c,p,choice['prior_power'],choice['temperature']);a=choice['blend']
    return normalize((1-a)*anchor+a*value)


def run(budget):
    configure();cells=load('validation',budget);root=OUT/'policies'/str(budget)/'classification/clean'
    raw_sh,sh_oof=fit_raw(cells,budget,10,'Shared')
    sh,source_sh=select(cells,sh_oof,{c['key']:c['base'] for c in cells},root/'Shared')
    raw_sp,sp_oof=fit_raw(cells,budget,21,'Specific_original')
    sp,_=select(cells,sp_oof,source_sh,root/'Specific_original')
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Shared=sh,Specific=sp,
        Shared_raw=raw_sh,Specific_raw=raw_sp,dataset_group_folds=4,source_loss_core_prior_weighted=True,
        clean_TRAIN_calibration_supervision=True,query_labels_used_for_fit_or_selection=False,
        target_prior_proxy_uses_protected_TRAIN_core=True,canonical_context_prior_is_member_proxy=True,
        Accuracy_and_AUC_use_same_probability_array=True,TFM_banks_unchanged=True,Shared_members=10,Specific_members=21,Base_additional=True,seed=0))
    rows=[]
    for c in load('query',budget):
        shared=apply(c,raw_sh,sh,c['base']);specific=apply(c,raw_sp,sp,shared);meta=c['meta']
        out=OUT/'predictions'/str(budget)/c['key'];prior.risk.head.source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=shared,Specific=specific,y_target=c['y'])
        record={k:meta[k] for k in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        record.update(input_hashes=meta['source_input_hashes'],source_bank=str(c['path']),role_permutation=c['order'],seed=0,provisional_Base_calibration=False,
            methods={name:metrics(c['y'],p,'classification',1.) for name,p in [('Base',c['base']),('Shared',shared),('Specific',specific)]},
            adaptation_version='v62',clean_TRAIN_calibration_supervision=True,dataset_group_folds=4,
            query_labels_used_for_fit=False,query_feedback_used_for_development_design=True,
            Accuracy_and_AUC_use_same_probability_array=True,checkpoint_finetuned=False,
            development_only=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block='clean',method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=72))
    print('WEIGHTED_GATE_COMPLETE',budget,flush=True)


def plan():
    assert not (OUT/'plan.json').exists();prior.risk.head.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_WEIGHTED_GATE_FITS',seed=0,budgets=[256,1024,4096,8192],
        refit='classification Clean only',full_goal_checks=48,
        change='original confidence/agreement/Base gate trained with TRAIN core-prior importance weights',
        source_validation='four whole-dataset folds; every calibration target dataset excluded from its gate training',
        original_gain_regularization='frozen v44 Shared recipe, same for both member counts',
        post_prior_powers=PRIOR_POWERS,post_temperatures=TEMPERATURES,blends=BLENDS,
        Specific_complete_Shared_fallback=True,Shared_members=10,Specific_members=21,Base_additional=True,
        fixed_TFM_banks=True,extra_TFM_calls=0,clean_TRAIN_calibration_supervision=True,
        query_labels_used_for_fit_or_selection=False,Accuracy_and_AUC_use_same_probability_array=True,
        query_feedback_used_for_development_design=True,development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((prior.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task,blocks,parent in [('classification',['five','seven'],RUN/'results/tabace_augmented_classifier_v54'),('regression',['clean','five','seven'],RUN/'results/tabace_clean_calibration_v59')]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(parent/name/str(b)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);a=p.parse_args();plan() if a.phase=='plan' else run(a.budget)
