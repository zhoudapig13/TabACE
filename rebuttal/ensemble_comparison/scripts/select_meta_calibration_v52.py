"""Select the meta expert using TRAIN OOF predictions, then evaluate queries."""
import argparse
import json
import os
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import meta_calibration_v52 as experiment
import sparse_specific_v47 as parent
from replay_head_oof_v47 import replay
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor,metrics


def parent_oof(cells,budget):
    h=experiment.h;block='seven'
    old=h.OUT/'policies'/str(budget)/'regression/seven'
    shared=json.loads((old/'Shared/policy.json').read_text())['model']
    sh,_=replay(cells,10,block,shared,{c['key']:c['base'] for c in cells})
    result={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        policy=json.loads((old/f'Specific_{family}/policy.json').read_text())['model']
        baseline,_=replay(group,21,block,policy,sh)
        extra=json.loads((parent.OUT/'policies'/str(budget)/'regression/seven'/f'Specific_{family}/policy.json').read_text())['model']
        chosen=extra['chosen']
        if chosen is None:result.update(baseline);continue
        by_task={}
        for i,c in enumerate(group):by_task.setdefault(str(c['meta']['task_id']),[]).append(i)
        values=[np.empty_like(c['y'],dtype=np.float64) for c in group]
        for indices in by_task.values():
            c=group[indices[0]];X=h.calibration(c['meta'],block)['X_validation'];F=h.folds(c['meta'],block,len(c['y']))
            matrices=[parent.matrix(X,group[i],chosen['profile']) for i in indices]
            for fold in range(4):
                train=F!=fold;valid=F==fold
                A=np.concatenate([M[train] for M in matrices]);y=np.concatenate([group[i]['y'][train] for i in indices])
                model=parent.train(A,y,len(indices),int(train.sum()),chosen['profile'])
                for i,M in zip(indices,matrices):values[i][valid]=h.head_predict(M[valid],model)
        a=chosen['blend']
        result.update({c['key']:(1-a)*baseline[c['key']]+a*value for c,value in zip(group,values)})
    return result


def scores(cells,predictions):
    values=[metrics(c['y'],predictions[c['key']],'regression',c['meta']['scale']) for c in cells]
    return {k:float(np.mean([v[k] for v in values])) for k in ['r2','nrmse']}


def select(budget):
    h=experiment.h;root=experiment.OUT/'policies'/str(budget)/'regression/seven'
    assert not (root/'all_policies_frozen.json').exists()
    cells=h.risk.load(h.source.OUT,'validation',budget,'seven');baseline=parent_oof(cells,budget)
    candidates={scope:{} for scope in experiment.SCOPES};verified=0
    for c in cells:
        meta=c['meta'];path=experiment.OUT/'validation'/str(budget)/str(meta['task_id'])/c['path'].name
        record=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            assert np.array_equal(z['y_validation'],c['y'])
            for scope in experiment.SCOPES:
                p=z[scope];assert p.shape==c['y'].shape and np.isfinite(p).all();candidates[scope][c['key']]=p
        assert record['original_query_data_loaded'] is False
        assert all(r['train_target_original_row_overlap']==0 and r['meta_context_rows']<=budget for r in record['records'])
        verified+=1
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family];base=scores(group,baseline);best=None;best_n=base['nrmse'];trials=[]
        for scope in experiment.SCOPES:
            for blend in [.25,.5,.75,1.]:
                mixed={c['key']:(1-blend)*baseline[c['key']]+blend*candidates[scope][c['key']] for c in group}
                values=scores(group,mixed);eligible=values['r2']>=base['r2']-1e-10 and values['nrmse']<best_n-1e-10
                trials.append(dict(scope=scope,blend=blend,source_OOF_scores=values))
                if eligible:best_n=values['nrmse'];best=dict(scope=scope,blend=blend)
        atomic_json(root/f'Specific_{family}.json',dict(status='FROZEN_BEFORE_QUERY',model=best,source_baseline_scores=base,trials=trials,
            selection='minimize source OOF nRMSE, no source R2 loss versus frozen parent',Shared_changed=False,
            clean_TRAIN_supervision_same_as_parent=True,query_labels_used_for_selection=False))
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY_BANKS_LOADED',verified_source_cases=verified,
        member_scopes=experiment.SCOPES,original_query_labels_used_for_selection=False,checkpoint_finetuned=False,
        parent_methods=str(parent.OUT),Specific_max_configured_TFM_forwards=23,Shared_max_configured_TFM_forwards=11,
        extra_clean_TRAIN_supervision=True,development_only=True,original_paper_replication=False))
    print('META_POLICIES_FROZEN',budget,verified,flush=True)


def query_worker(index,count):
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    h=experiment.h;predictor=Predictor('regression','cuda:0');failures=[];completed=0
    for budget,meta,family,rho in experiment.jobs():
        relative=str(experiment.Path(str(budget))/str(meta['task_id'])/f'{family}_{rho}.npz')
        owner=int.from_bytes(experiment.hashlib.sha256(relative.encode()).digest()[:8],'big')%count
        if owner!=index:continue
        out=experiment.OUT/'predictions'/str(budget)/'regression/seven'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        try:
            root=experiment.OUT/'policies'/str(budget)/'regression/seven'
            assert (root/'all_policies_frozen.json').exists()
            choice=json.loads((root/f'Specific_{family}.json').read_text())['model']
            original=parent.OUT/'predictions'/str(budget)/'regression/seven'/str(meta['task_id'])/out.name
            summary=json.loads(original.with_suffix('.json').read_text())
            with np.load(original) as z:base=z['Base'];shared=z['Shared'];specific=z['Specific'];truth=z['y_target']
            rows=0;seconds=0.
            if choice is not None:
                calpath=h.source.OUT/'banks/validation'/str(budget)/'regression/seven'/str(meta['task_id'])/out.name
                querypath=h.source.OUT/'banks/query'/str(budget)/'regression/seven'/str(meta['task_id'])/out.name
                with np.load(calpath) as z:calbank={k:z[k] for k in ['base','members','y_target']}
                with np.load(querypath) as z:qbank={k:z[k] for k in ['base','members']};assert np.array_equal(base,z['base']) and np.array_equal(truth,z['y_target'])
                cal=h.calibration(meta,'seven');assert np.array_equal(cal['y_validation'],calbank['y_target'])
                data,_=h.source.adaptation.raw.scaling.bank('regression','seven',meta,family,rho)
                assert protocol.array_sha256(data['X_target'])==summary['input_hashes']['X_target']
                A=experiment.features(calbank,cal['X_validation'],choice['scope']);Q=experiment.features(qbank,data['X_target'],choice['scope'])
                indices=experiment.selected_rows(np.arange(len(A)),budget);rows=len(indices)
                assert rows<=budget and len(np.unique(cal['validation_original_ids'][indices]))==rows
                start=time.monotonic();p,_=predictor(A[indices],cal['y_validation'][indices],Q,members=1,random_state=0);seconds=time.monotonic()-start
                assert np.isfinite(p).all();a=choice['blend'];specific=(1-a)*specific+a*p
            h.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
            summary.update(adaptation_version='v52',methods={name:metrics(truth,p,'regression',summary['scale']) for name,p in [('Base',base),('Shared',shared),('Specific',specific)]},
                Shared_unchanged_from_v47=True,meta_expert=choice,meta_context_rows=rows,extra_meta_forward_count=int(choice is not None),meta_seconds=seconds,
                maximum_configured_TFM_forwards=23 if choice else 22,clean_TRAIN_supervision_same_as_parent=True,
                checkpoint_finetuned=False,query_labels_used_for_fit=False,independent_test_claim=False)
            atomic_json(out.with_suffix('.json'),summary);completed+=1
            print('META_QUERY_COMPLETE',relative,flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);failures.append(dict(case=relative,error=error))
            atomic_json(experiment.OUT/'errors'/f'query_{index}_{budget}_{meta["task_id"]}_{family}_{rho}.json',failures[-1])
    atomic_json(experiment.OUT/'query_workers'/f'{index}.json',dict(status='COMPLETE' if not failures else 'FAILED',completed=completed,failures=failures))
    if failures:raise SystemExit(1)


def finish():
    h=experiment.h
    for budget in [256,1024,4096,8192]:
        root=experiment.OUT/'predictions'/str(budget)/'regression/seven';paths=[p for p in root.glob('*/*.json') if '.tmp.' not in p.name];assert len(paths)==924
        rows=[]
        for path in paths:
            info=json.loads(path.read_text())
            for name,values in info['methods'].items():rows.append(dict(context=budget,block='seven',method=name,task_id=info['task_id'],**values))
        dest=experiment.OUT/'policies'/str(budget)/'regression/seven';pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
        atomic_json(dest/'completion.json',dict(status='COMPLETE',conditions=924,extra_meta_TFM_forward_disclosed=True))
    print('META_QUERY_GROUPS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','query','finish']);p.add_argument('--budget',type=int);p.add_argument('--index',type=int,default=0);p.add_argument('--count',type=int,default=8);a=p.parse_args()
    select(a.budget) if a.phase=='select' else query_worker(a.index,a.count) if a.phase=='query' else finish()
