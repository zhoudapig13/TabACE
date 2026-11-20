"""Frozen evaluation of separately developed Specific-R and Specific-C variants."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,metrics,paired,write_json,array_hash
from summarize_public_shift import holm

TASKS=['sepsis_A_to_B','sepsis_B_to_A','readmission_other_to_ER']
VARIANTS=[('Specific_C','medical_specific_context_v21','specific_context_predictions.npz','Forest'),
          ('Specific_R','medical_specific_residual_v20','specific_residual_predictions.npz','Auxiliary')]


def main():
    out=ROOT/'results/independent_specific_evaluation';out.mkdir(exist_ok=True)
    plan=[];route={}
    for task in TASKS:
        route[task]={}
        snapshots={}
        for method,version,pred,reference in VARIANTS:
            path=ROOT/'results'/version/task/'frozen_selection.json';f=json.loads(path.read_text())
            assert f['test_labels_opened'] is False and f['base_shared_unchanged']
            snapshots[method]=f
            plan.append(dict(task=task,method=method,version=version,prediction=pred,reference=reference,
                snapshot=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for domain in ['id','ood']:
            # Ties favor C, which keeps the expert's final output entirely TFM based.
            route[task][domain]=max([x[0] for x in VARIANTS],
                key=lambda method:snapshots[method]['development_accuracy'][method][domain])
    write_json(out/'frozen_evaluation_plan.json',dict(
        frozen_before_final_outcomes_at_utc=datetime.now(timezone.utc).isoformat(),evaluations=plan,
        selected_specific_by_development_only=route,tie_rule='Prefer Specific-C on equal development Acc',
        domain_routing_metadata_available_to_all_methods=True,base_shared_predictions_will_be_checked_identical=True,
        no_test_filtering=True,test_cohorts_previously_observed=True))
    rows=[];details=[];tests={}
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data']
        with np.load(data/'inputs.npz') as z:pack={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in pack.items()}==f['metadata']['input_hashes']
        with np.load(data/item['prediction']) as z:p={k:z[k] for k in z.files}
        with np.load(data/'expanded_probability_predictions.npz') as z:parent={k:z[k] for k in z.files}
        for domain in ['id','ood']:
            for method in ['Base','Shared']:assert np.array_equal(p[domain+'_'+method],parent[domain+'_'+method])
        with np.load(ROOT/f['metadata']['original_inputs']/'test_truth.npz') as z:truth={k:z[k] for k in z.files}
        report=dict(task=item['task'],variant=item['method'],source_auxiliary_training_pool_n=len(pack['y_train']),
                    target_context_training_budget=32,target_calibration_budget=32,
                    baseline_shared_identical_to_parent=True,domains={})
        for domain in ['id','ood']:
            threshold=f['configurations'][domain]['Base']['policy']['threshold']
            reference_threshold=(f['forest_standalone_configurations'] if item['reference']=='Forest' else
                                 f['auxiliary_standalone_configurations'])[domain]['threshold']
            for split in ['development','test']:
                sl=slice(*f['query_layout'][domain+'_'+split])
                y=pack['y_'+domain+'_dev'] if split=='development' else truth['y_'+domain]
                methods=['Base','Shared',item['method'],item['reference']]
                probabilities={m:p[domain+'_'+m][sl] for m in methods}
                assert all(len(x)==len(y) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in probabilities.values())
                scores={m:metrics(y,v,reference_threshold if m==item['reference'] else threshold) for m,v in probabilities.items()}
                if split=='development':
                    for m,v in scores.items():assert abs(v['accuracy']-f['development_accuracy'][m][domain])<1e-12
                test=paired(y,probabilities['Shared']>=threshold,probabilities[item['method']]>=threshold)
                if split=='test':tests.setdefault(item['method'],[]).append((item['task']+'/'+domain,test))
                acc={m:v['accuracy'] for m,v in scores.items()}
                d1=100*(acc['Shared']-acc['Base']);d2=100*(acc[item['method']]-acc['Shared'])
                rows.append(dict(task=item['task'],split=split,domain=domain,variant=item['method'],n=len(y),
                    Base=acc['Base'],Shared=acc['Shared'],Specific=acc[item['method']],
                    reference_method=item['reference'],reference_accuracy=acc[item['reference']],
                    Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                    specific_step_1to5pp=bool(1-1e-10<=d2<=5+1e-10),
                    all_steps_1to5pp=bool(1-1e-10<=d1<=5+1e-10 and 1-1e-10<=d2<=5+1e-10)))
                report['domains'][domain+'_'+split]=dict(n=len(y),methods=scores,
                    specific_vs_shared=test if split=='test' else dict(accuracy_delta=test['accuracy_delta'],postselection_only=True))
        write_json(out/item['method']/item['task']/'metrics.json',report);details.append(report)
    for variant,tt in tests.items():holm(tt);write_json(out/variant/'paired_tests_exploratory.json',dict(tt))
    table=pd.DataFrame(rows);table.to_csv(out/'all_variants_accuracy.csv',index=False)
    selected=table[[row['variant']==route[row['task']][row['domain']] for row in rows]].copy()
    selected.to_csv(out/'development_selected_specific_accuracy.csv',index=False)
    tt=selected[selected.split=='test']
    write_json(out/'verification.json',dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(),all_control_predictions_unchanged=True,
        all_source_inputs_hash_verified=True,target_fitting_budget=64,
        selected_specific_step_domains_1to5pp=int(tt.specific_step_1to5pp.sum()),
        selected_all_steps_domains_1to5pp=int(tt.all_steps_1to5pp.sum()),domains_tested=len(tt),
        auxiliary_standalone_results_reported=True,all_previous_versions_preserved=True,
        original_paper_specific_reproduction_claimed=False,markdown_written=False,
        clinical_test_labels_used_for_selection=False,
        scripts_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}))
    print('SELECTED SPECIFIC (development routing frozen BEFORE final outcomes)')
    print(tt.to_string(index=False))
    print('ALL SPECIFIC VARIANTS')
    print(table[table.split=='test'].to_string(index=False))


if __name__=='__main__':main()
