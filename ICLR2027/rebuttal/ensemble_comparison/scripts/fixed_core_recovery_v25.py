"""User-authorized NEW protocol: fixed corrupted256 core + clean source extension.

The core, queries and normalization denominators are byte-identical to v22.
Additional rows are distinct clean TRAIN samples from v23. Contamination is
diluted as context grows: this measures recovery, not fixed-fraction robustness.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import tabicl_context_scaling_v23 as original
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabicl_fixed_core_recovery_v25'
BUDGETS=[256,1024,4096,8192]


def jobs():
    return [(track,meta) for track in ['five','seven'] for meta in original.frozen.cohort('regression',track)]


def plan():
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='FROZEN_BEFORE_RECOVERY_PREDICTIONS',
        user_selected='fixed corrupted core, additional clean TRAIN contexts; separately labeled',
        name='fixed_core_clean_extension_recovery',new_protocol=True,development_only=True,
        original_paper_replication=False,fixed_core_rows=256,budgets=BUDGETS,seed=0,n_estimators=1,
        nominal_strengths=[.2,.5,.8,1.],nominal_strength_scope='the fixed256 core only',
        full_context_core_fractions={'256':1.,'1024':.25,'4096':.0625,'8192':.03125},
        inputs='unchanged v22 core and queries; v23 clean eligible source rows after the first canonical256',
        additional_labels='clean TRAIN labels; semisynthetic teacher labels on five track, real TRAIN labels on seven track',
        insufficient_rows='cap to actual distinct eligible source pool size; no duplicate padding',
        source_memberships='nested, fixed order, same as v23 Clean; no score-based selection',
        metric_denominators='unchanged v22',checkpoint='official TabICL-v2 Native1, no finetuning or target adaptation',
        classification_unchanged=True,Clean_reuse_requires_identical_input_hashes=True,
        original256_reuse_requires_identical_input_hashes=True,no_score_acceptance_bounds=True,
        no_query_labels_used_in_fitting=True,semisynthetic_blocks=['clean','five'],
        frozen_v22_sha256=original.sha(original.frozen.CONFIG),script_sha256=original.sha(__file__),Markdown_written=False))


def materialize(pool,task_block,meta,family,rho,budget):
    core,info=original.bank('regression',task_block,meta,family,rho)
    assert len(core['y_source'])==256
    stop=min(budget,len(pool['y']))
    data=core|dict(X_source=np.vstack([core['X_source'],pool['X'][256:stop]]),
                   y_source=np.r_[core['y_source'],pool['y'][256:stop]])
    assert np.array_equal(data['X_source'][:256],core['X_source'])
    assert np.array_equal(data['y_source'][:256],core['y_source'])
    assert not np.intersect1d(pool['source_ids'][:256],pool['source_ids'][256:stop]).size
    assert not np.intersect1d(pool['source_ids'][:stop],pool['query_ids']).size
    return data,info


def worker(shard,shards):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    from seven_shifts import FAMILIES
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0')
    for track,meta in jobs()[shard::shards]:
        try:
            pool=original.prepare('regression',track,meta);tid=str(meta['task_id'])
            grid=([('clean','original',0)]+[('five',f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]) if track=='five' else [('seven',f,r) for f in FAMILIES for r in [20,50,80,100]]
            for block,family,rho in grid:
                for budget in BUDGETS:
                    out=OUT/'predictions'/str(budget)/block/tid/f'{family}_{rho}.npz'
                    if out.exists() and out.with_suffix('.json').exists():continue
                    data,info=materialize(pool,block,meta,family,rho,budget)
                    hashes={k:protocol.array_sha256(v) for k,v in data.items()}
                    start=time.monotonic();reused=None
                    if budget==256:
                        assert hashes==info['input_hashes']
                        reused=original.frozen.OUT/'input_banks/regression'/block/tid/out.name
                        with np.load(reused) as z:pred=z['Base_prediction']
                    elif block=='clean':
                        candidate=original.OUT/'predictions'/str(budget)/'regression/clean'/tid/out.name
                        record=json.loads(candidate.with_suffix('.json').read_text())
                        assert record['input_hashes']==hashes
                        reused=candidate
                        with np.load(reused) as z:pred=z['prediction']
                    else:
                        for small in BUDGETS[1:]:
                            if small>=budget:break
                            prior=OUT/'predictions'/str(small)/block/tid/out.name
                            if prior.with_suffix('.json').exists() and json.loads(prior.with_suffix('.json').read_text())['input_hashes']==hashes:
                                reused=prior
                                with np.load(prior) as z:pred=z['prediction']
                                break
                        if reused is None:pred,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0)
                    assert np.isfinite(pred).all()
                    actual=len(data['y_source'])
                    original.npwrite(out,prediction=pred,y_target=data['y_target'],clean_extension_original_ids=pool['source_ids'][256:actual])
                    atomic_json(out.with_suffix('.json'),dict(task='regression',block=block,task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                        requested_context=budget,actual_context=actual,fixed_core_rows=256,clean_extension_rows=actual-256,
                        core_fraction=256/actual,seed=0,n_estimators=1,scale=info['scale'],input_hashes=hashes,
                        fixed_core_hashes={k:protocol.array_sha256(data[k][:256]) for k in ['X_source','y_source']},
                        clean_extension_hashes={k:protocol.array_sha256(data[k][256:]) for k in ['X_source','y_source']},
                        metrics=metrics(data['y_target'],pred,'regression',info['scale']),elapsed_seconds=time.monotonic()-start,
                        reused_exact_inputs=str(reused) if reused else None,
                        new_protocol='fixed256_corrupted_core_plus_clean_TRAIN_extension',development_only=True,
                        semisynthetic=block in ['clean','five'],target_adaptation=False,finetuning=False))
            print('RECOVERY_DATASET_COMPLETE',track,meta['dataset'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{track}_{meta["task_id"]}.json',dict(error=error))


def report():
    rows=[]
    for p in (OUT/'predictions').glob('*/*/*/*.json'):
        if p.name.endswith('.tmp.json'):continue
        row=json.loads(p.read_text());rows.append({k:v for k,v in row.items() if k not in ['input_hashes','metrics','fixed_core_hashes','clean_extension_hashes']}|row['metrics'])
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'detail.csv',index=False)
    per=df.groupby(['requested_context','block','task_id','dataset'],as_index=False).agg(conditions=('family','size'),actual_context=('actual_context','first'),core_fraction=('core_fraction','first'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per['complete']=per.conditions==per.block.map(dict(clean=1,five=20,seven=28));per.to_csv(OUT/'per_dataset.csv',index=False)
    avg=per[per.complete].groupby(['requested_context','block'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'averages.csv',index=False);print(avg.to_string(index=False))
    table=[]
    for b in BUDGETS:
        row=dict(context_budget=b)
        for block in ['clean','five','seven']:
            a=avg[(avg.requested_context==b)&(avg.block==block)]
            if len(a):row|={block+'_'+k:float(a.iloc[0][k]) for k in ['r2','nrmse','datasets']}
        table.append(row)
    pd.DataFrame(table).to_csv(OUT/'recovery_table.csv',index=False)
    atomic_json(OUT/'progress.json',dict(completed_conditions=len(df),expected_conditions=6636,complete=len(df)==6636,
        errors=len(list((OUT/'errors').glob('*.json'))),Markdown_bytes=(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size))


def audit():
    from run_tabicl import metrics
    from seven_shifts import FAMILIES
    bad=[];missing=[];verified=0
    for track,meta in jobs():
        pool=original.prepare('regression',track,meta)
        grid=([('clean','original',0)]+[('five',f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]) if track=='five' else [('seven',f,r) for f in FAMILIES for r in [20,50,80,100]]
        for block,family,rho in grid:
            previous_ids=None
            for budget in BUDGETS:
                path=OUT/'predictions'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                if not path.exists() or not path.with_suffix('.json').exists():missing.append(str(path));continue
                data,info=materialize(pool,block,meta,family,rho,budget);rec=json.loads(path.with_suffix('.json').read_text())
                hashes={k:protocol.array_sha256(v) for k,v in data.items()}
                if hashes!=rec['input_hashes']:bad.append([str(path),'reconstructed_inputs'])
                for key in ['X_target','y_target']:
                    if hashes[key]!=info['input_hashes'][key]:bad.append([str(path),key])
                for key in ['X_source','y_source']:
                    if rec['fixed_core_hashes'][key]!=info['input_hashes'][key]:bad.append([str(path),'core_changed'])
                    if not np.array_equal(data[key][256:],pool['X' if key=='X_source' else 'y'][256:len(data['y_source'])]):bad.append([str(path),'extension_not_clean'])
                if rec['scale']!=info['scale']:bad.append([str(path),'scale'])
                with np.load(path) as z:pred=z['prediction'];truth=z['y_target'];ids=z['clean_extension_original_ids']
                if previous_ids is not None and not np.array_equal(ids[:len(previous_ids)],previous_ids):bad.append([str(path),'not_nested'])
                previous_ids=ids
                if len(np.unique(ids))!=len(ids) or np.intersect1d(ids,pool['query_ids']).size:bad.append([str(path),'extension_overlap'])
                if protocol.array_sha256(truth)!=info['input_hashes']['y_target']:bad.append([str(path),'saved_truth'])
                measured=metrics(truth,pred,'regression',info['scale'])
                if any(not np.isclose(measured[k],v,atol=1e-12,rtol=0) for k,v in rec['metrics'].items()):bad.append([str(path),'metric'])
                verified+=1
    atomic_json(OUT/'final_audit.json',dict(status='PASS' if not bad and not missing else 'FAILED_OR_INCOMPLETE',
        verified_conditions=verified,expected_conditions=6636,missing=missing,failures=bad,
        fixed256_core_identical_to_v22=True,queries_and_denominators_identical_to_v22=True,
        added_context_rows_are_clean_and_nested=True,new_protocol=True,original_paper_replication=False))
    print('RECOVERY_AUDIT',verified,'missing',len(missing),'failures',bad[:5])
    if bad or missing:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','audit']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards)
    elif a.phase=='report':report()
    else:audit()
