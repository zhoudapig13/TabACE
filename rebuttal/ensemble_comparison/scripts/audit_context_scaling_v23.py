"""Independent completeness, frozen-query and metric audit for context scaling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tabicl_context_scaling_v23 as run
from prepare_data import atomic_json
from run_tabicl import metrics, CHECKPOINTS


def main():
    records=[];missing=[];failures=[]
    for task,block,meta,grid in run.frozen.jobs():
        for family,rho in grid:
            origin=run.frozen.OUT/'input_banks'/task/block/str(meta['task_id'])/f'{family}_{rho}.json'
            original=json.loads(origin.read_text())
            for budget in run.BUDGETS:
                path=run.OUT/'predictions'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                if not path.exists() or not path.with_suffix('.json').exists():
                    missing.append(str(path.relative_to(run.OUT)));continue
                rec=json.loads(path.with_suffix('.json').read_text())
                for key in ['X_target','y_target']:
                    if rec['input_hashes'][key]!=original['input_hashes'][key]:failures.append([str(path),key])
                if rec['scale']!=original['scale']:failures.append([str(path),'metric_scale'])
                if rec['seed']!=0 or rec['n_estimators']!=1 or rec['kv_cache'] is not False:failures.append([str(path),'model_configuration'])
                with np.load(path) as z:
                    truth,pred=z['y_target'],z['prediction']
                if run.protocol.array_sha256(truth)!=original['input_hashes']['y_target']:failures.append([str(path),'saved_query_labels'])
                if not np.isfinite(pred).all():failures.append([str(path),'nonfinite_prediction'])
                pool_record=run.OUT/'source_pools'/task/('five' if block=='clean' else block)/f'{meta["task_id"]}.json'
                available=json.loads(pool_record.read_text())['available_rows']
                if rec['actual_context']!=min(budget,available):failures.append([str(path),'context_budget'])
                measured=metrics(truth,pred,task,rec['scale'])
                for key,value in measured.items():
                    if not np.isclose(value,rec['metrics'][key],atol=1e-12,rtol=0):failures.append([str(path),key])
                records.append([budget,task,block,int(meta['task_id']),family,rho])
    manifest=json.loads((run.OUT/'plan.json').read_text())
    if run.sha(run.frozen.CONFIG)!=manifest['frozen_v22_sha256']:failures.append(['v22','frozen_suite_changed'])
    pools=[]
    for task in ['classification','regression']:
        for track in ['five','seven']:
            for meta in run.frozen.cohort(task,track):
                path=run.OUT/'source_pools'/task/track/f'{meta["task_id"]}.npz'
                if not path.exists():continue
                with np.load(path) as z:
                    ids=np.r_[z['h_ids'],z['r_ids']] if 'h_ids' in z else z['source_ids']
                    query=z['query_ids']
                if len(np.unique(ids))!=len(ids) or np.intersect1d(ids,query).size:failures.append([str(path),'source_row_overlap'])
                audit=run.OUT/'source_audits'/task/track/f'{meta["task_id"]}.json'
                if not audit.exists() or not json.loads(audit.read_text())['canonical256_exact']:failures.append([str(path),'256_replay'])
                pools.append(dict(task=task,track=track,task_id=meta['task_id'],dataset=meta['dataset'],pool_rows=len(ids),
                                  **{f'context_{b}':min(b,len(ids)) for b in run.BUDGETS}))
    pd.DataFrame(pools).to_csv(run.OUT/'dataset_context_counts.csv',index=False)
    expected=sum(len(grid) for _,_,_,grid in run.frozen.jobs())*len(run.BUDGETS)
    result=dict(status='PASS' if not missing and not failures else 'INCOMPLETE_OR_FAILED',
                expected_conditions=expected,verified_conditions=len(records),missing_conditions=missing,failures=failures,
                source_pools_audited=len(pools),query_features_and_labels_identical_to_v22=True if not failures else None,
                metric_denominators_unchanged=True if not failures else None,pristine_v2_checkpoints={
                    kind:run.sha(CHECKPOINTS/f'tabicl-{kind}-v2-20260212.ckpt') for kind in ['classifier','regressor']},
                final_runner_sha256=run.sha(Path(run.__file__)),audit_script_sha256=run.sha(__file__),
                original_paper_replication=False,semisynthetic_regression_clean_five=True)
    atomic_json(run.OUT/'final_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['missing_conditions','failures']},indent=2))
    if missing or failures:print('MISSING',len(missing),'FAILURES',failures[:10]);raise SystemExit(1)


if __name__=='__main__':main()
