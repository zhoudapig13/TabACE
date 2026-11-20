"""Compare all independently developed Specific extensions with frozen controls."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,metrics,paired,write_json,array_hash

TASKS=['sepsis_A_to_B','sepsis_B_to_A','readmission_other_to_ER']
VARIANTS=[('Specific_C','medical_specific_context_v21','specific_context_predictions.npz'),
          ('Specific_P','medical_specific_simple_v22','specific_simple_predictions.npz'),
          ('Specific_R','medical_specific_residual_v20','specific_residual_predictions.npz'),
          ('Specific_S','medical_selective_specific_v23','specific_selective_predictions.npz')]


def main():
    out=ROOT/'results/specific_extensions_evaluation';out.mkdir(exist_ok=True)
    plan=[];route={}
    for task in TASKS:
        snapshots={}
        for method,version,prediction in VARIANTS:
            path=ROOT/'results'/version/task/'frozen_selection.json'
            f=json.loads(path.read_text());assert f['test_labels_opened'] is False and f['base_shared_unchanged']
            snapshots[method]=f
            plan.append(dict(task=task,method=method,version=version,prediction=prediction,
                snapshot=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        route[task]={d:max([v[0] for v in VARIANTS],key=lambda m:snapshots[m]['development_accuracy'][m][d])
                     for d in ['id','ood']}
    write_json(out/'frozen_evaluation_plan.json',dict(
        frozen_before_outcome_read_at_utc=datetime.now(timezone.utc).isoformat(),experiments=plan,
        selected_specific_per_known_domain=route,selected_by='Development accuracy only',
        tie_priority=[v[0] for v in VARIANTS],test_filtering=False,previous_test_cohorts_observed=True))
    rows=[];audits=[]
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data']
        with np.load(data/'inputs.npz') as z:pack={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in pack.items()}==f['metadata']['input_hashes']
        with np.load(data/item['prediction']) as z:p={k:z[k] for k in z.files}
        with np.load(data/'expanded_probability_predictions.npz') as z:parent={k:z[k] for k in z.files}
        for domain in ['id','ood']:
            assert all(np.array_equal(p[domain+'_'+m],parent[domain+'_'+m]) for m in ['Base','Shared'])
        with np.load(ROOT/f['metadata']['original_inputs']/'test_truth.npz') as z:truth={k:z[k] for k in z.files}
        result=dict(task=item['task'],variant=item['method'],target_fitting_budget=64,known_domain_metadata=True,
                     base_shared_unchanged=True,source_auxiliary_or_retrieval_pool=len(pack['y_train']),domains={})
        for domain in ['id','ood']:
            threshold=f['configurations'][domain]['Base']['policy']['threshold']
            for split in ['development','test']:
                sl=slice(*f['query_layout'][domain+'_'+split])
                y=pack['y_'+domain+'_dev'] if split=='development' else truth['y_'+domain]
                probabilities={m:p[domain+'_'+m][sl] for m in ['Base','Shared',item['method']]}
                scores={m:metrics(y,x,threshold) for m,x in probabilities.items()}
                if split=='development':
                    assert all(abs(v['accuracy']-f['development_accuracy'][m][domain])<1e-12 for m,v in scores.items())
                changes=int(np.count_nonzero((probabilities['Shared']>=threshold)!=(probabilities[item['method']]>=threshold)))
                if item['method']=='Specific_S':assert changes<=int(np.floor(.045*len(y)))
                comparison=paired(y,probabilities['Shared']>=threshold,probabilities[item['method']]>=threshold)
                acc={m:v['accuracy'] for m,v in scores.items()}
                d1=100*(acc['Shared']-acc['Base']);d2=100*(acc[item['method']]-acc['Shared'])
                rows.append(dict(task=item['task'],split=split,domain=domain,variant=item['method'],n=len(y),
                    Base=acc['Base'],Shared=acc['Shared'],Specific=acc[item['method']],
                    Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,changed_predictions=changes,
                    edit_fraction=changes/len(y),specific_step_1to5pp=bool(1-1e-10<=d2<=5+1e-10),
                    all_steps_1to5pp=bool(1-1e-10<=d1<=5+1e-10 and 1-1e-10<=d2<=5+1e-10)))
                result['domains'][domain+'_'+split]=dict(n=len(y),methods=scores,changed_predictions=changes,
                    specific_vs_shared=comparison if split=='test' else dict(accuracy_delta=comparison['accuracy_delta'],postselection_only=True))
        write_json(out/item['method']/item['task']/'metrics.json',result)
        audits.append(dict(task=item['task'],method=item['method'],base_shared_exactly_unchanged=True,
                           input_hashes_match=True,selective_edit_budget_checked=item['method']=='Specific_S'))
    table=pd.DataFrame(rows);table.to_csv(out/'all_specific_extensions.csv',index=False)
    selected=table[[r['variant']==route[r['task']][r['domain']] for r in rows]]
    selected.to_csv(out/'development_selected_specific.csv',index=False)
    t=selected[selected.split=='test']
    write_json(out/'verification.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        audits=audits,selected_specific_steps_1to5pp=int(t.specific_step_1to5pp.sum()),
        selected_all_steps_1to5pp=int(t.all_steps_1to5pp.sum()),test_domains=len(t),
        standalone_auxiliary_controls='results/independent_specific_evaluation/all_variants_accuracy.csv',
        source_target_fit_disjoint_from_development=True,target_fitting_budget=64,
        source_and_target_test_labels_used_for_selection=False,all_versions_retained=True,
        previous_test_cohorts_observed=True,markdown_written=False,
        script_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}))
    print('FROZEN DEVELOPMENT-SELECTED SPECIFIC')
    print(t.to_string(index=False))
    print('SELECTIVE MODEL ALL FIXED TESTS')
    print(table[(table.split=='test')&(table.variant=='Specific_S')].to_string(index=False))


if __name__=='__main__':main()
