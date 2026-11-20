"""One immutable held-out-TRAIN-selected five-corruption query evaluation."""
import argparse
import json
import os
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import validation_recovery_v26 as v
from prepare_data import atomic_json,protocol

OUT=v.OUT/'frozen_five_query'
FROZEN=OUT/'parameters.json'


def freeze():
    from datetime import datetime,timezone
    assert not FROZEN.exists()
    selected=json.loads((v.OUT/'validation_selection.json').read_text())['best']['five']
    assert selected['validation_goal_met']
    atomic_json(OUT/'validation_selection_snapshot.json',selected)
    atomic_json(FROZEN,dict(status='FROZEN_BEFORE_QUERY_INFERENCE',repair_probability=selected['repair_probability'],
        selected_only_on_held_out_TRAIN=True,validation_goal_met=True,source_refit='all eligible v23 TRAIN rows after selection',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),original_query_score_filter=False,development_only=True,
        independent_test_claim=False,seed=0,n_estimators=1))


def worker(shard,shards):
    import torch,warnings
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    rate=json.loads(FROZEN.read_text())['repair_probability'];fingerprint=v.prior.original.sha(FROZEN)
    predictor=Predictor('regression','cuda:0');jobs=[m for block,m in v.prior.jobs() if block=='five']
    for meta in jobs[shard::shards]:
        pool=v.prior.original.prepare('regression','five',meta)
        for family,rho in v.grid('five'):
            env=v.environment(pool,meta,'five',family,rho,np.arange(len(pool['y'])))
            base,info=v.prior.original.bank('regression','five',meta,family,rho);cache={}
            for budget in v.BUDGETS:
                out=OUT/'predictions'/str(budget)/str(meta['task_id'])/f'{family}_{rho}.npz'
                if out.exists() and out.with_suffix('.json').exists():continue
                A,b,mask=v.context(env,rate,budget);hashes={k:protocol.array_sha256(x) for k,x in (base|dict(X_source=A,y_source=b)).items()}
                key=(hashes['X_source'],hashes['y_source'])
                if budget==256:
                    assert hashes==info['input_hashes']
                    with np.load(v.prior.original.frozen.OUT/'input_banks/regression/five'/str(meta['task_id'])/out.name) as z:p=z['Base_prediction']
                elif key in cache:p=cache[key]
                else:p,_=predictor(A,b,base['X_target'],members=1,random_state=0)
                assert np.isfinite(p).all();cache[key]=p
                v.prior.original.npwrite(out,prediction=p,y_target=base['y_target'],repair_mask=mask)
                atomic_json(out.with_suffix('.json'),dict(task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                    requested_context=budget,actual_context=len(b),repair_probability=rate,input_hashes=hashes,
                    frozen_parameters_sha256=fingerprint,scale=info['scale'],metrics=metrics(base['y_target'],p,'regression',info['scale']),
                    query_configuration_count=1,selected_on_TRAIN_validation=True,development_only=True,seed=0,n_estimators=1))
        print('FROZEN_FIVE_QUERY_COMPLETE',meta['dataset'],rate,flush=True)


def report():
    rows=[]
    for path in (OUT/'predictions').glob('*/*/*.json'):
        if path.name.endswith('.tmp.json'):continue
        r=json.loads(path.read_text());rows.append({k:x for k,x in r.items() if k not in ['metrics','input_hashes']}|r['metrics'])
    df=pd.DataFrame(rows);df.to_csv(OUT/'detail.csv',index=False)
    per=df.groupby(['requested_context','task_id'],as_index=False).agg(conditions=('family','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per=per[per.conditions==20]
    avg=per.groupby('requested_context',as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'averages.csv',index=False);print(avg.to_string(index=False))
    atomic_json(OUT/'completion.json',dict(conditions=len(df),expected=2800,complete=len(df)==2800,query_configurations=1,development_only=True))


def audit():
    from run_tabicl import metrics
    rate=json.loads(FROZEN.read_text())['repair_probability'];fingerprint=v.prior.original.sha(FROZEN);bad=[];count=0
    for block,meta in v.prior.jobs():
        if block!='five':continue
        pool=v.prior.original.prepare('regression','five',meta)
        for family,rho in v.grid('five'):
            env=v.environment(pool,meta,'five',family,rho,np.arange(len(pool['y'])));base,info=v.prior.original.bank('regression','five',meta,family,rho)
            for budget in v.BUDGETS:
                path=OUT/'predictions'/str(budget)/str(meta['task_id'])/f'{family}_{rho}.npz';rec=json.loads(path.with_suffix('.json').read_text())
                A,b,mask=v.context(env,rate,budget);hashes={k:protocol.array_sha256(x) for k,x in (base|dict(X_source=A,y_source=b)).items()}
                if hashes!=rec['input_hashes'] or rec['frozen_parameters_sha256']!=fingerprint:bad.append([str(path),'input_or_parameters'])
                with np.load(path) as z:p=z['prediction'];t=z['y_target'];saved=z['repair_mask']
                if not np.array_equal(t,base['y_target']) or not np.array_equal(mask,saved):bad.append([str(path),'truth_or_mask'])
                actual=metrics(t,p,'regression',info['scale'])
                if any(not np.isclose(actual[k],x,atol=1e-12,rtol=0) for k,x in rec['metrics'].items()):bad.append([str(path),'metric'])
                count+=1
    atomic_json(OUT/'audit.json',dict(status='PASS' if not bad and count==2800 else 'FAIL',conditions=count,failures=bad,
        one_immutable_configuration=True,development_only=True,independent_test_claim=False));print('FIVE_QUERY_AUDIT',count,'failures',bad[:5])
    if bad or count!=2800:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','worker','report','audit']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else globals()[a.phase]()
