"""One auditable delivery: actual predictions, controls, increments and costs.

Never merges paper reference numbers into measured same-protocol comparisons.
Leaves REBUTTAL_EXPERIMENTS.md empty as explicitly requested by the user.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
import finetune_tabace_reference_v3 as reference
import finetune_tabace_hierarchical_v4 as hierarchy
import finetune_calibration_control_v18 as calibration_control
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import metrics

OUT=RUN/'results/finetune_delivery_v17'
SELECTED=RUN/'results/finetune_final_selected_v13'
REFERENCES=RUN/'results/finetune_final_reference_v14'
HIERARCHY=RUN/'results/finetune_final_hierarchical_v15'
PROFILES=RUN/'results/finetune_final_profile_v16'
TARGETS={
    ('classification','clean'):[.804,.901],('classification','five'):[.494,.676],('classification','seven'):[.763,.826],
    ('regression','clean'):[.739,.476],('regression','five'):[-1.077,1.270],('regression','seven'):[.273,.730]}


def rel(job):return Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])


def bootstrap(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    rng=np.random.default_rng(0)
    means=rng.choice(a,(10000,len(a)),replace=True).mean(1)
    return tuple(float(x) for x in np.quantile(means,[.025,.975]))


def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=main.jobs();rows=[];costs=[];audit=[];training=[];errors=[]
    assert len(jobs)==175
    profile_files=list((PROFILES/'profiles').glob('*/*/*.json'))
    assert len(profile_files)==175,('final profiling incomplete',len(profile_files))
    selection=json.loads((SELECTED/'selection.json').read_text())
    chosen={(c['task'],c['track'],c['task_id']):c for c in selection['choices']}
    for job in jobs:
        try:
            parent=SELECTED/rel(job);refroot=REFERENCES/rel(job);hierroot=HIERARCHY/rel(job)
            assert all((p/'completion.json').exists() for p in [parent,refroot,hierroot])
            result=json.loads((parent/'training_result.json').read_text())
            assert result['actual_backbone_weight_updates'] and result['changed_parameter_tensors']>0 and result['trainable_weight_delta_L2']>0
            assert main.sha(parent/'weights.pt')==result['checkpoint_file_sha256']
            with np.load(parent/'split.npz') as z:split={k:z[k] for k in z.files}
            pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
            ids=split['ids'];sets=[set(ids[split[k]]) for k in ['train','val','cal']]+[set(pool['query_ids'])]
            assert all(not(a&b) for i,a in enumerate(sets) for b in sets[i+1:])
            ref_arrays=reference.references(job,pool,split)
            assert set(ref_arrays[2]).issubset(sets[0])
            raw_policies=json.loads((parent/'policies.json').read_text())
            policies=json.loads((hierroot/'policies.json').read_text())
            controlroot=calibration_control.OUT/rel(job)
            assert (controlroot/'completion.json').exists()
            control_policies=json.loads((controlroot/'policies.json').read_text())
            assert not control_policies['final_query_labels_used_for_policy_selection']
            assert control_policies['source_calibration_information_identical_to_TabACE']
            assert not policies['original_query_labels_used'] and policies['selection_rows_disjoint_fit_rows']
            assert policies['duplicate_resampled_original_rows_grouped']
            assert set(policies['selection_original_row_positions']).issubset(set(split['cal']))
            assert (hierroot/'policies.json').stat().st_mtime<=(hierroot/'query_detail.csv').stat().st_mtime
            selection_row=chosen[(job['task'],job['track'],job['meta']['task_id'])]
            training.append(dict(task=job['task'],track=job['track'],task_id=job['meta']['task_id'],dataset=job['meta']['dataset'],
                selected_optimizer_steps=result['optimizer_steps'],selected_requested_steps=result['requested_steps'],selected_learning_rate=result['learning_rate'],
                selected_checkpoint_step=result['selected_checkpoint']['step'],selected_training_seconds=result['training_seconds'],
                all_candidate_training_seconds=selection_row['total_candidate_training_seconds'],trainable_parameters=result['trainable_parameters'],
                total_parameters=result['total_parameters'],peak_training_MiB=result['peak_training_CUDA_MiB'],
                FT_TRAIN_rows=len(split['train']),checkpoint_validation_rows=len(split['val']),fusion_calibration_rows=len(split['cal']),
                actual_weight_delta_L2=result['trainable_weight_delta_L2'],checkpoint_sha256=result['checkpoint_file_sha256']))
            saved=pd.read_csv(hierroot/'query_detail.csv')
            for block in (['clean','five'] if job['track']=='five' else ['seven']):
                for family,rho in main.contexts.case_grid(block):
                    data,info=source.bank(job['task'],block,job['meta'],family,rho)
                    original_path=parent/'query'/block/f'{family}_{rho}.npz'
                    reference_path=refroot/'query'/block/f'{family}_{rho}.npz'
                    raw_record=json.loads(original_path.with_suffix('.json').read_text())
                    ref_record=json.loads(reference_path.with_suffix('.json').read_text())
                    hashes={k:protocol.array_sha256(v) for k,v in data.items()}
                    assert raw_record['input_hashes']==ref_record['input_hashes']==info['input_hashes']==hashes
                    assert raw_record['source_rows']==256 and max(raw_record['member_rows'].values())<=256
                    assert max(ref_record['new_member_rows'].values())<=256
                    assert ref_record['reference_original_ids_sha256']==protocol.array_sha256(ref_arrays[2])
                    with np.load(original_path) as z:arrays={k:z[k] for k in z.files}
                    with np.load(reference_path) as z:rarrays={k:z[k] for k in z.files}
                    c=main.cell(arrays,raw_record)
                    assert np.array_equal(c['y'],data['y_target']) and np.array_equal(rarrays['y'],data['y_target'])
                    assert np.array_equal(arrays['base'],rarrays['base'])
                    selected_kind=policies['selected_bank']
                    adapted=hierarchy.read_cell(reference_path if selected_kind=='reference' else original_path,reference=selected_kind=='reference')
                    start=time.perf_counter();sp=main.fusion.predict(adapted,policies['Shared']);shared_apply=time.perf_counter()-start
                    residual=adapted|dict(base=sp,bank=adapted['bank'][5:])
                    start=time.perf_counter();specific=main.fusion.predict(residual,policies['Specific'][family]);specific_apply=time.perf_counter()-start
                    predictions={
                        'FT':c['base'],
                        'FT+Shared_observed':main.fusion.predict(c,raw_policies['Shared']),
                        'FT+Specific_observed':main.fusion.predict(c,raw_policies['Specific'][family]),
                        'FT+Reference1':rarrays['reference'],
                        'FT+Shared_dev':sp,'FT+Specific_dev':specific,
                        'FT+Calibration':calibration_control.prediction(c,control_policies['global_policy']),
                        'FT+FamilyCalibration':calibration_control.prediction(c,control_policies['family_policies'][family])}
                    common=dict(task=job['task'],block=block,task_id=job['meta']['task_id'],data_id=job['meta']['data_id'],
                        dataset=job['meta']['dataset'],family=family,rho=rho,query_rows=len(c['y']),seed=0,context_cap=256,
                        protocol='source-context-only development' if job['task']=='classification' and block=='five'
                            else ('semisynthetic development' if job['task']=='regression' and block in ['clean','five'] else 'frozen v22 real-label development'))
                    rows.append(common|dict(method='Base',**info['metrics']))
                    expected=saved[(saved.family==family)&(saved.rho==rho)]
                    for name,prediction in predictions.items():
                        assert np.isfinite(prediction).all()
                        value=metrics(data['y_target'],prediction,job['task'],info['scale'])
                        rows.append(common|dict(method=name,**value))
                        if name in ['FT','FT+Shared_dev','FT+Specific_dev']:
                            before=expected[expected.method==name].iloc[0]
                            for metric in (['accuracy','auc'] if job['task']=='classification' else ['r2','nrmse']):
                                if value[metric] is not None and np.isfinite(before[metric]):assert abs(value[metric]-before[metric])<1e-9
                    if job['task']=='classification' and block=='five':
                        clean,_=source.bank('classification','clean',job['meta'],'original',0)
                        assert np.array_equal(data['X_target'],clean['X_target']) and np.array_equal(data['y_target'],clean['y_target'])
                    audit.append(common|dict(exact_frozen_inputs=True,actual_weight_updates=True,all_training_and_calibration_rows_query_disjoint=True,
                        reference_labels_already_FT_TRAIN=True,metric_replay_matches=True,selected_bank=selected_kind))
                    if selected_kind=='reference':
                        names=ref_record['new_member_names'];timings=ref_record['new_member_inference_seconds']
                        shared_time=raw_record['FT_inference_seconds']+timings['Reference1']+sum(timings[n] for n in names[1:])
                        construction=ref_record['new_reference_construction_seconds']
                    else:
                        shared_time=raw_record['FT_Shared_inference_seconds'];construction=raw_record['shared_construction_seconds']
                    extra_time=sum(raw_record['inference_seconds'][n] for n in raw_record['members'][5:])
                    for name,seconds,build,apply,calls in [
                        ('FT',raw_record['FT_inference_seconds'],0.,0.,1),
                        ('FT+Reference1',ref_record['Reference1_inference_seconds'],0.,0.,1),
                        ('FT+Shared_dev',shared_time,construction,shared_apply,6),
                        ('FT+Specific_dev',shared_time+extra_time,construction+raw_record['specific_extra_construction_seconds'],shared_apply+specific_apply,11)]:
                        costs.append(common|dict(method=name,inference_seconds=seconds,construction_seconds=build,fusion_apply_seconds=apply,
                            allocated_TFM_contexts=calls,total_seconds=seconds+build+apply,milliseconds_per_query=1000*(seconds+build+apply)/len(c['y'])))
        except Exception as error:
            errors.append(dict(job=job,error=repr(error)));print('DELIVERY_AUDIT_ERROR',job,repr(error),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'all_condition_results.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'input_supervision_audit.csv',index=False)
    pd.DataFrame(training).to_csv(OUT/'training_cost_detail.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'inference_cost_detail.csv',index=False)
    atomic_json(OUT/'audit.json',dict(status='PASS' if not errors and len(audit)==4151 else 'FAIL',errors=errors,
        expected_task_tracks=175,verified_task_tracks=len(training),expected_query_conditions=4151,verified_query_conditions=len(audit),
        benchmark_selection_used_Base_development_query_scores=True,weight_checkpoint_and_fusion_selection_used_final_query_labels=False,
        classification_five_source_context_only=True,regression_clean_five_semisynthetic=True,
        adapted_reference_library_reuses_FT_TRAIN_labels=True,Reference1_control_retained=True,
        paper_Shared_Specific_not_recomputed=True,Markdown_written=False))
    assert not errors and len(audit)==4151
    summarize(pd.DataFrame(rows),pd.DataFrame(training),pd.DataFrame(costs))
    profile_rows=[]
    for path in profile_files:
        record=json.loads(path.read_text());assert record['status']=='COMPLETE'
        profile_rows.extend(record['rows'])
    profile=pd.DataFrame(profile_rows)
    profile['milliseconds_per_query']=1000*profile['median_seconds']/profile['query_rows']
    profile.to_csv(OUT/'Base_FT_warm_profile_detail.csv',index=False)
    profile.groupby(['task','track','method']).agg(datasets=('task_id','nunique'),conditions=('family','size'),
        mean_case_seconds=('median_seconds','mean'),median_case_seconds=('median_seconds','median'),
        mean_milliseconds_per_query=('milliseconds_per_query','mean'),profiling_peak_both_models_MiB=('CUDA_peak_allocated_MiB','max')).reset_index().to_csv(OUT/'Base_FT_warm_profile_averages.csv',index=False)


def summarize(detail,training,cost):
    averages=[];differences=[]
    for (task,block),g in detail.groupby(['task','block']):
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        expected=len(main.frozen.cohort(task,block))
        for method,h in g.groupby('method'):
            averages.append(dict(task=task,block=block,method=method,datasets=h.task_id.nunique(),conditions=len(h),
                complete=h.task_id.nunique()==expected,**{f:float(h[f].mean()) for f in fields}))
        for metric in fields:
            table=g.groupby(['task_id','method'])[metric].mean().unstack();direction=-1 if metric=='nrmse' else 1
            first=direction*(table.FT-table.Base)
            for method in ['FT+Shared_observed','FT+Specific_observed','FT+Reference1','FT+Shared_dev','FT+Specific_dev']:
                second=direction*(table[method]-table.FT);low,high=bootstrap(second)
                differences.append(dict(task=task,block=block,metric=metric,method=method,Base_to_FT_gain=float(first.mean()),
                    FT_to_combination_gain=float(second.mean()),second_gain_bootstrap95_low=low,second_gain_bootstrap95_high=high,
                    ratio_second_to_first=float(second.mean()/first.mean()) if first.mean()>0 else None,
                    second_at_least_first=bool(second.mean()>=first.mean()) if first.mean()>0 else None,
                    maximum_possible_second_gain=(1-float(table.FT.mean())) if metric in ['accuracy','auc','r2'] else None,
                    FT_net_error_reduction_fraction=float(first.mean()/(1-table.Base.mean())) if metric in ['accuracy','r2'] and table.Base.mean()<1 else None,
                    combination_net_remaining_error_reduction_fraction=float(second.mean()/(1-table.FT.mean())) if metric in ['accuracy','r2'] and table.FT.mean()<1 else None,
                    error_fraction_definition='classification error rate' if metric=='accuracy' else ('dataset-normalized MSE' if metric=='r2' else None),
                    datasets=len(table),bootstrap_seed=0,bootstrap_samples=10000,single_model_seed=True,not_multi_seed_variance=True))
    averages=pd.DataFrame(averages);averages.to_csv(OUT/'averages.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'increment_comparison.csv',index=False)
    detail.groupby(['task','block','dataset','task_id','method']).mean(numeric_only=True).reset_index().to_csv(OUT/'per_dataset_results.csv',index=False)
    training.groupby(['task','track']).agg(datasets=('task_id','nunique'),mean_selected_training_seconds=('selected_training_seconds','mean'),
        all_candidate_training_seconds=('all_candidate_training_seconds','sum'),mean_optimizer_steps=('selected_optimizer_steps','mean'),
        peak_training_MiB=('peak_training_MiB','max')).reset_index().to_csv(OUT/'training_cost_averages.csv',index=False)
    cost.groupby(['task','block','method']).agg(datasets=('task_id','nunique'),conditions=('family','size'),mean_case_seconds=('total_seconds','mean'),
        mean_milliseconds_per_query=('milliseconds_per_query','mean'),allocated_TFM_contexts=('allocated_TFM_contexts','first')).reset_index().to_csv(OUT/'inference_cost_averages.csv',index=False)
    calibration=[]
    for (task,block),targets in TARGETS.items():
        r=averages[(averages.task==task)&(averages.block==block)&(averages.method=='Base')].iloc[0]
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        for metric,target in zip(fields,targets):calibration.append(dict(task=task,block=block,metric=metric,paper_Base=target,
            measured_Base=r[metric],difference=r[metric]-target,within_005=abs(r[metric]-target)<=.005))
    pd.DataFrame(calibration).to_csv(OUT/'Base_vs_paper.csv',index=False)
    assert all(r['within_005'] for r in calibration)
    paper=[]
    paper_values={
        'classification':{'Shared':[[.804,.901],[.575,.720],[.772,.831]],'Specific':[[.804,.900],[.610,.752],[.805,.854]]},
        'regression':{'Shared':[[.738,.478],[.363,.783],[.467,.659]],'Specific':[[.738,.476],[.485,.708],[.655,.535]]}}
    for task,methods in paper_values.items():
        for method,blocks in methods.items():
            for block,values in zip(['clean','five','seven'],blocks):paper.append(dict(task=task,block=block,method=method,
                primary_metric=values[0],secondary_metric=values[1],provenance='paper historical reference, NOT rerun on this benchmark'))
    pd.DataFrame(paper).to_csv(OUT/'paper_historical_references.csv',index=False)
    # Explicit family-macro pooling: five + seven families, all strengths equal.
    family=detail[detail.block!='clean'].groupby(['task','block','family','method']).mean(numeric_only=True).reset_index()
    family.groupby(['task','method']).mean(numeric_only=True).reset_index().to_csv(OUT/'nonclean_12_family_macro_average.csv',index=False)
    atomic_json(OUT/'delivery_manifest.json',dict(status='VERIFIED_COMPLETE',seed=0,context_cap=256,GPUs=8,
        dataset_counts=dict(classification_clean_five=72,classification_seven=35,regression_clean_five=35,regression_seven=33),
        benchmark='development reconstruction; source-only classification five; semisynthetic regression Clean/five; other v22 tracks frozen',
        all_12_Base_metrics_within_005=True,all_methods_use_same_selected_updated_checkpoint_within_task=True,
        fine_tuning_scope='last four ICL transformer blocks and ICL label/output heads; column/row encoders and earlier ICL blocks frozen',
        FT_candidate_step_budgets=[96,128,512],FT_learning_rates=[3e-6,1e-5],FT_per_candidate_seconds_cap=150,
        FT_checkpoint_selection_source_validation_only=True,adapted_methods='explicit clean FT TRAIN reference access, held-out bank selection and residual Specific fusion',
        Reference1_control_is_required_for_interpreting_reference_adaptation=True,paper_numbers_are_historical_references_only=True,
        MD_empty=(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0,
        final_query_condition_count=4151,actual_methods=9,single_seed_not_multi_seed_variance=True,
        same_calibration_information_FT_controls_included=True,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=main.sha(__file__)))
    print(averages.to_string(index=False),flush=True)
    print('DELIVERY_VERIFIED_COMPLETE',str(OUT),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect']);a=p.parse_args();collect()
