"""Evaluate the frozen common Native-8 control on the unchanged ID/OOD cohorts."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,array_hash,metrics,paired,write_json
from summarize_public_shift import holm

TASKS=['sepsis_A_to_B','sepsis_B_to_A','readmission_other_to_ER']
VERSIONS=[('medical_common_native8_v24','medical_common_native8_predictions.npz')]


def main():
    out=ROOT/'results/medical_native8_evaluation';out.mkdir(exist_ok=True)
    plan=[]
    for version,prediction in VERSIONS:
        for task in TASKS:
            path=ROOT/'results'/version/task/'frozen_selection.json';f=json.loads(path.read_text())
            assert f['test_labels_opened'] is False and f['external_id_development']
            assert f['native_estimators_per_branch']==8 and f['native_estimator_count_matched_across_all_methods']
            plan.append(dict(version=version,task=task,prediction=prediction,
                snapshot=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(out/'frozen_evaluation_plan.json',dict(
        frozen_before_final_outcomes_at_utc=datetime.now(timezone.utc).isoformat(),experiments=plan,
        primary_protocol='Common Native-8 backbone for Base, Shared, and Specific; separate from Native-1 primary results',
        test_patients_unchanged=True,test_metrics_do_not_control_selection=True))
    rows=[];tests={};audits=[]
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data']
        with np.load(data/'inputs.npz') as z:pack={k:z[k] for k in z.files}
        meta=json.loads((data/'metadata.json').read_text())
        assert {k:array_hash(v) for k,v in pack.items()}==meta['input_hashes']
        with np.load(data/'target_fitting_membership.npz') as z:
            assert np.array_equal(np.sort(z['context']),np.sort(f['target_context_indices']))
            assert np.array_equal(np.sort(z['calibration']),np.sort(f['target_calibration_indices']))
            assert not set(z['additional_validation_from_old_fit']).intersection(np.r_[z['context'],z['calibration']])
        with np.load(data/item['prediction']) as z:p={k:z[k] for k in z.files}
        layout=f['query_layout'];n=layout['ood_test'][1]
        assert all(len(x)==n and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in p.values())
        # Final truth opened only after both candidate protocols were frozen.
        with np.load(ROOT/meta['original_inputs']/'test_truth.npz') as z:truth={k:z[k] for k in z.files}
        details=dict(task=item['task'],version=item['version'],known_domain_metadata=True,
                     target_fitting_budget=64,source_calibration_full=True,native_estimators_per_branch=8,
                     more_development_validation_labels_disclosed=True,domains={})
        for domain in ['id','ood']:
            threshold=f['configurations'][domain]['Base']['policy']['threshold']
            assert all(abs(f['configurations'][domain][m]['policy']['threshold']-threshold)<1e-10 for m in ['Base','Shared','Specific'])
            for split in ['development','test']:
                sl=slice(*layout[domain+'_'+split])
                y=pack['y_'+domain+'_dev'] if split=='development' else truth['y_'+domain]
                pp={m:p[domain+'_'+m][sl] for m in ['Base','Shared','Specific']}
                assert all(len(x)==len(y) for x in pp.values())
                scores={m:metrics(y,x,threshold) for m,x in pp.items()}
                if split=='development':
                    assert all(abs(s['accuracy']-f['development_accuracy'][m][domain])<1e-12 for m,s in scores.items())
                paired_tests={b+'_minus_'+a:paired(y,pp[a]>=threshold,pp[b]>=threshold)
                              for a,b in [('Base','Shared'),('Shared','Specific')]}
                if split=='test':tests.setdefault(item['version'],[]).extend(
                    (item['task']+'/'+domain+'/'+name,t) for name,t in paired_tests.items())
                else:paired_tests={k:dict(accuracy_delta=v['accuracy_delta'],postselection_descriptive_only=True) for k,v in paired_tests.items()}
                acc={m:v['accuracy'] for m,v in scores.items()}
                d1=100*(acc['Shared']-acc['Base']);d2=100*(acc['Specific']-acc['Shared'])
                rows.append(dict(version=item['version'],task=item['task'],split=split,domain=domain,n=len(y),**acc,
                    Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                    both_steps_1to5pp=bool(1-1e-10<=d1<=5+1e-10 and 1-1e-10<=d2<=5+1e-10)))
                details['domains'][domain+'_'+split]=dict(n=len(y),positive_count=int(y.sum()),methods=scores,comparisons=paired_tests)
        write_json(out/item['version']/item['task']/'metrics.json',details)
        audits.append(dict(task=item['task'],version=item['version'],input_hashes_verified=True,
                           exact_same64_target_fit_patients=True,extra_validation_excludes_fitting=True))
    for version,tt in tests.items():holm(tt);write_json(out/version/'paired_tests_exploratory.json',dict(tt))
    table=pd.DataFrame(rows);table.to_csv(out/'accuracy_all.csv',index=False)
    for version,_ in VERSIONS:table[table.version==version].to_csv(out/(version+'_accuracy.csv'),index=False)
    status={v:dict(domains_tested=6,domains_meeting_1to5pp_both_steps=int(
        table[(table.version==v)&(table.split=='test')].both_steps_1to5pp.sum())) for v,_ in VERSIONS}
    write_json(out/'verification.json',dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(),audits=audits,status=status,
        current_target_fit_labels=64,all_original_test_patients_preserved=True,
        test_filtering_or_test_training=False,known_domain_adaptation_explicit=True,
        previous_test_cohorts_observed=True,inference_native_estimators_per_branch=8,
        all_prior_outcomes_retained=True,markdown_written=False,
        scripts_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}))
    print(table[table.split=='test'].to_string(index=False))
    print('STATUS',status)


if __name__=='__main__':main()
