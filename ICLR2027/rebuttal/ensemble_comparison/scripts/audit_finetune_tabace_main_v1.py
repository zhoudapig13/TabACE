"""Audit exact inputs, weight changes, information partitions and paired costs."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import finetune_tabace_main_v1 as main
from prepare_data import atomic_json,protocol


def interval(values):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(0)
    means=np.mean(rng.choice(a,(10000,len(a)),replace=True),axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]


def run():
    main.report()
    root=main.OUT
    rows=[];cost=[];errors=[]
    for job in main.jobs():
        dest=main.folder(job)
        if not (dest/'completion.json').exists():continue
        try:
            with np.load(dest/'split.npz') as z:
                ids=z['ids'];sets=[set(ids[z[k]]) for k in ['train','val','cal']]
            pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
            query=set(pool['query_ids'])
            assert all(not (a & b) for i,a in enumerate(sets+[query]) for b in (sets+[query])[i+1:])
            train=json.loads((dest/'training_result.json').read_text())
            assert train['actual_backbone_weight_updates'] and train['changed_parameter_tensors']>0 and train['trainable_weight_delta_L2']>0
            assert main.sha(dest/'weights.pt')==train['checkpoint_file_sha256']
            policies=json.loads((dest/'policies.json').read_text())
            assert not policies['final_query_labels_used']
            for p in sorted((dest/'query').glob('*/*.json')):
                record=json.loads(p.read_text());block=record['block'];family=record['family'];rho=record['rho']
                origin=main.frozen.OUT/'input_banks'/job['task']/block/str(job['meta']['task_id'])/p.name
                old=json.loads(origin.read_text())
                assert record['input_hashes']==old['input_hashes']
                assert record['source_rows']==256 and max(record['member_rows'].values())<=256
                assert record['seed']==0 and len(record['members'])==10
                assert not record['original_query_labels_used_by_predictor_or_constructor']
                with np.load(p.with_suffix('.npz')) as z:
                    assert np.isfinite(z['base']).all() and np.isfinite(z['bank']).all()
                    assert z['bank'].shape[0]==10 and z['base'].shape[0]==len(z['y'])
                rows.append(dict(task=job['task'],track=job['track'],block=block,task_id=job['meta']['task_id'],family=family,rho=rho,
                    exact_frozen_inputs=True,weight_updates_verified=True,disjoint_supervision=True))
                common=dict(task=job['task'],block=block,task_id=job['meta']['task_id'],family=family,rho=rho,query_rows=record['query_rows'])
                for method,key,overhead,calls in [
                    ('FT','FT_inference_seconds',0.,1),
                    ('FT+Shared','FT_Shared_inference_seconds',record['shared_construction_seconds'],6),
                    ('FT+Specific','FT_Specific_inference_seconds',record['shared_construction_seconds']+record['specific_extra_construction_seconds'],11)]:
                    cost.append(common|dict(method=method,inference_seconds=record[key],context_construction_seconds=overhead,
                        nominal_TFM_calls=calls,seconds_per_query=(record[key]+overhead)/record['query_rows']))
        except Exception as e:errors.append(dict(job=job,error=repr(e)))
    pd.DataFrame(rows).to_csv(root/'input_information_audit.csv',index=False)
    if cost:
        costs=pd.DataFrame(cost);costs.to_csv(root/'inference_cost_detail.csv',index=False)
        costs.groupby(['task','block','method']).agg(conditions=('task_id','size'),mean_inference_seconds=('inference_seconds','mean'),
            median_inference_seconds=('inference_seconds','median'),mean_construction_seconds=('context_construction_seconds','mean'),
            mean_seconds_per_query=('seconds_per_query','mean'),nominal_TFM_calls=('nominal_TFM_calls','first')).reset_index().to_csv(root/'inference_cost_averages.csv',index=False)
    detail=pd.read_csv(root/'query_detail.csv');deltas=[]
    for (task,block),g in detail.groupby(['task','block']):
        for metric in (['accuracy','auc'] if task=='classification' else ['r2','nrmse']):
            table=g.groupby(['task_id','method'])[metric].mean().unstack('method')
            for a,b in [('FT','Base (cached, paired)'),('FT+Shared','FT'),('FT+Specific','FT'),('FT+Specific','FT+Shared')]:
                values=(table[a]-table[b]).dropna()
                ci=interval(values)
                deltas.append(dict(task=task,block=block,metric=metric,comparison=a+' minus '+b,datasets=len(values),
                    mean_difference=float(values.mean()),bootstrap95_lower=ci[0],bootstrap95_upper=ci[1],
                    dataset_bootstrap_seed=0,bootstrap_samples=10000,not_multi_seed_variance=True))
    pd.DataFrame(deltas).to_csv(root/'paired_differences.csv',index=False)
    atomic_json(root/'audit.json',dict(status='PASS' if not errors else 'FAIL',completed_jobs=len(set((r['task'],r['track'],r['task_id']) for r in rows)),
        audited_query_conditions=len(rows),errors=errors,exact_v22_inputs=True,final_query_information_not_used_for_training_or_selection=True,
        original_benchmark_was_development_selected=True,regression_clean_five_semisynthetic=True,classification_five_query_feature_perturbation=True,
        classification_inference_calls_may_internally_use_hierarchical_many_class_passes=True,
        costs_are_wall_time_per_case_for_all_query_rows=True,Markdown_written=False,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('AUDIT',len(rows),'conditions',len(errors),'errors',flush=True)


if __name__=='__main__':run()
