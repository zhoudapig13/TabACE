"""Read-only recovery of fully audited endpoint recipes under the latest gains.

Historical predictions, settings and reports are never rewritten. Every newly
archived endpoint uses one complete recipe across its full dataset cohort;
different recipes and their information budgets are explicitly recorded.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import unseen_shared_robustness_v117 as source
from prepare_data import RUN,ROOT,atomic_json
from run_tabicl import metrics

OUT=RUN/'results/threshold_recheck_v362'
FROZEN=RUN/'results/frozen_endpoints_v363'
SPECS=[
    ('shared_clean_reference_contexts_v269','perturbation','classification','feature_quantization','Shared_extra_reference',64,
     'Original frozen backbone, source192 plus64 independent clean labeled references; original five Shared roles/gate; original full query evaluation, no target labels'),
    ('shared_expanded_pool_v291','perturbation','regression','cell_heavy_tail','Shared_adapted',35,
     'Original backbone, larger disjoint clean labeled reference pool, query-local256-row contexts with five Shared roles/gate; original full query, no target labels'),
    ('shared_expanded_pool_v291','perturbation','regression','column_desynchronization','Shared_adapted',35,
     'Same expanded reference-pool recipe291; full original query, no target labels'),
    ('shared_finetuned_inference_v309','shift','classification','cross_feature_rotation','Shared_finetuned',31,
     'Source-only per-dataset fine-tuned307 backbone, original five Shared roles/gate; target adaptation32 labels in each context, removed from evaluation'),
    ('shared_finetuned_inference_v309','shift','regression','interaction_sampling','Shared_finetuned',33,
     'Source-only fine-tuned307 backbone and original Shared bank; fixed quarter target adaptation cap64 labels in contexts, removed from evaluation'),
]


def main():
    OUT.mkdir(parents=True,exist_ok=True);contract_path=RUN/'results/acceptance_contract_v361/contract.json'
    contract=json.loads(contract_path.read_text())
    assert contract['perturbation_required_gain_strictly_greater_than']==.03 and contract['shift_required_gain_strictly_greater_than']==.02
    parent_path=RUN/'results/frozen_endpoints_v330/registry.json';parent=json.loads(parent_path.read_text())
    for item in parent['endpoints']:
        for p in item['conditions']:
            assert source.sha(p['prediction_file'])==p['prediction_sha256']
            assert source.sha(p['metadata_file'])==p['metadata_sha256']
    cases=json.loads((source.OUT/'case_manifest.json').read_text())
    entries=[];all_rows=[]
    for version,track,task,family,method,expected,annotation in SPECS:
        directory=RUN/'results'/version
        report=json.loads((directory/'independent_audit.json').read_text());assert report['status']=='PASS'
        selected=[c for c in cases if c['track']==track and c['task']==task and family in c['families']]
        assert len(selected)==expected
        conditions=[];rows=[]
        for c in selected:
            assert source.sha(c['truth_file'])==c['truth_sha256'] and source.sha(c['input_file'])==c['input_sha256']
            with np.load(c['truth_file']) as z:truth=z['y_target']
            for rho in source.novel.STRENGTHS:
                rel=(Path(track)/task/str(c['task_id']) if version.endswith('309') else Path(task)/str(c['task_id']))
                p=directory/'predictions'/rel/f'{family}_{rho}.npz'
                meta=json.loads(p.with_suffix('.json').read_text());assert source.sha(p)==meta['output_sha256']
                assert meta['task_id']==c['task_id'] and meta['data_id']==c['data_id'] and meta['rho']==rho and meta['family']==family
                assert meta['original_input_sha256']==c['input_sha256']
                if version.endswith('309'):
                    apos=np.array(meta['target_adaptation_positions'],dtype=int)
                    epos=np.array(meta['evaluation_positions_in_original_query'],dtype=int)
                    assert len(apos)==meta['target_labels_used_in_context'] and set(apos).isdisjoint(epos)
                    assert not meta['final_eval_labels_opened']
                    trained=json.loads(Path(meta['training_result_file']).read_text())
                    assert source.sha(trained['checkpoint_file'])==meta['checkpoint_sha256']==trained['checkpoint_sha256']
                else:
                    epos=np.arange(len(truth))
                    assert not meta.get('query_labels_opened',meta.get('query_labels_opened_for_inference',False))
                y=truth[epos]
                original=source.OUT/'predictions'/track/task/str(c['task_id'])/p.name
                assert source.sha(original)==meta['original_prediction_sha256']
                with np.load(p) as pred,np.load(original) as old:
                    np.testing.assert_array_equal(pred['Base'],old['Base'][epos])
                    for name in ['Base',method]:
                        assert len(pred[name])==len(y) and np.isfinite(pred[name]).all()
                        rows.append({'track':track,'task':task,'data_id':c['data_id'],'task_id':c['task_id'],
                            'family':family,'rho':rho,'method':name,**metrics(y,pred[name],task,max(float(np.std(y)),1e-12))})
                conditions.append({'prediction_file':str(p),'prediction_sha256':source.sha(p),
                    'metadata_file':str(p.with_suffix('.json')),'metadata_sha256':source.sha(p.with_suffix('.json'))})
        assert len(conditions)==4*expected
        df=pd.DataFrame(rows);metric='accuracy' if task=='classification' else 'r2'
        ds=df.groupby(['method','data_id'])[['accuracy','auc','r2','nrmse']].mean() if all(k in df for k in ['accuracy','auc','r2','nrmse']) else df.groupby(['method','data_id'])[[k for k in ['accuracy','auc','r2','nrmse'] if k in df]].mean()
        table=ds.groupby('method').mean()
        base=float(table.loc['Base',metric]);adapted=float(table.loc[method,metric]);gain=adapted-base
        oldtable=pd.read_csv(directory/'family_averages.csv');oldtable=oldtable[(oldtable.task==task)&(oldtable.family==family)]
        for name in ['Base',method]:np.testing.assert_allclose(float(oldtable.loc[oldtable.method==name,metric].iloc[0]),float(table.loc[name,metric]),atol=1e-10,rtol=0)
        threshold=contract['perturbation_required_gain_strictly_greater_than' if track=='perturbation' else 'shift_required_gain_strictly_greater_than']
        assert gain>threshold
        entry={'endpoint':[track,task,family],'directory':str(directory),'method':method,'datasets':expected,
            'annotation':annotation,'metric':metric,'Base':base,'adapted':adapted,'gain':gain,'required_gain_strictly_greater_than':threshold,
            'passed':True,'conditions':conditions,'upstream_full_audit_sha256':source.sha(directory/'independent_audit.json'),
            'original_plan_sha256':source.sha(directory/'plan.json'),'historical_predictions_modified':False}
        entries.append(entry);all_rows.extend(rows)
        print('ARCHIVED_ENDPOINT_VERIFIED',track,task,family,'datasets',expected,'gain',gain,flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'recomputed_condition_metrics.csv',index=False)
    atomic_json(OUT/'independent_recheck.json',{'status':'PASS','newly_verified_endpoints':5,'conditions':sum(len(e['conditions']) for e in entries),
        'all_predictions_and_Original_Base_rechecked_against_truth':True,'all_four_strengths_and_full_applicable_cohorts_checked':True,
        'earlier_three_frozen_endpoints_unchanged':True,'different_recipes_and_eval_protocols_explicit':True,
        'no_training_or_inference_rerun':True,'acceptance_contract_sha256':source.sha(contract_path),'script_sha256':source.sha(__file__)})
    FROZEN.mkdir(parents=True,exist_ok=True)
    registry={'status':'FROZEN_VERIFIED_ENDPOINTS_UNDER_LATEST_USER_THRESHOLDS','parent_registry_file':str(parent_path),
        'parent_registry_sha256':source.sha(parent_path),'unchanged_parent_endpoints':parent['endpoints'],'additional_archived_endpoints':entries,
        'frozen_endpoint_count':8,'acceptance_contract_file':str(contract_path),'acceptance_contract_sha256':source.sha(contract_path),
        'configuration_policy':'original three retained exactly; archival recipes fixed per whole endpoint; no per-condition or per-dataset cherry-picking',
        'these_are_separately_adapted_recipes_not_one_global_Shared_configuration':True,
        'active_endpoint':['perturbation','classification','cell_heavy_tail'],
        'unresolved_endpoints':[['perturbation','classification','cell_heavy_tail'],['perturbation','classification','column_desynchronization'],
            ['shift','classification','nonlinear_sensor'],['shift','classification','interaction_sampling']],
        'Markdown_written':False}
    p=FROZEN/'registry.json'
    if p.exists():assert json.loads(p.read_text())==registry
    else:atomic_json(p,registry)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    print('TOTAL_FROZEN_ENDPOINTS',8,flush=True)


if __name__=='__main__':main()
