"""Evaluate all frozen ID/OOD protocols without feeding results back to selection."""
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,metrics,paired,write_json,array_hash
from summarize_public_shift import holm

TASKS=['sepsis_A_to_B','sepsis_B_to_A','readmission_other_to_ER']
VERSIONS=[('joint_medical_v14','joint_raw_predictions.npz',False),
          ('joint_medical_risk_v15','joint_risk_predictions.npz',False),
          ('known_domain_medical_v16','known_domain_raw_predictions.npz',True),
          ('known_domain_logit_v17','known_domain_logit_predictions.npz',True)]


def main():
    output=ROOT/'results/joint_domain_medical_evaluation';output.mkdir(exist_ok=True)
    plan=[];primary={}
    # Freeze all experiment files and choose common-input representation from
    # BASE development accuracy alone, BEFORE reading any final outcomes.
    for task in TASKS:
        candidates=[]
        for version in ['joint_medical_v14','joint_medical_risk_v15']:
            f=json.loads((ROOT/'results'/version/task/'frozen_selection.json').read_text())
            candidates.append((np.mean(list(f['development_accuracy']['Base'].values())),version))
        primary[task]=max(candidates,key=lambda x:x[0])[1]  # ties preserve raw first
        for version,prediction,known in VERSIONS:
            path=ROOT/'results'/version/task/'frozen_selection.json'
            snapshot=json.loads(path.read_text())
            assert snapshot['test_labels_opened'] is False
            plan.append(dict(task=task,version=version,predictions=prediction,known_domain=known,
                             selection_path=str(path.relative_to(ROOT)),
                             frozen_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(output/'evaluation_plan.json',dict(
        frozen_before_outcome_access_at_utc=datetime.now(timezone.utc).isoformat(),evaluations=plan,
        primary_representation_selected_by_strongest_base_development=primary,
        final_results_do_not_control_protocol_or_stop_selection=True,
        previous_test_cohorts_observed=True,primary_metric='accuracy'))
    wide=[];long=[];tests_by_version={};audits=[]
    for item in plan:
        f=json.loads((ROOT/item['selection_path']).read_text());data=ROOT/f['data']
        meta=json.loads((data/'metadata.json').read_text())
        with np.load(data/'inputs.npz') as z:pack={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in pack.items()}==meta['input_hashes']
        fit=np.asarray(f['source_calibration_fit_indices'],int)
        iddev=np.asarray(f['id_development_indices'],int)
        assert len(set(fit).intersection(iddev))==0
        ctx=f['target_context_indices'];cal=f['target_calibration_indices']
        assert len(ctx)==len(cal)==32 and not set(ctx).intersection(cal)
        with np.load(data/item['predictions']) as z:p={k:z[k] for k in z.files}
        ncal=len(pack['y_cal'])+32;nval=len(pack['y_dev_val']);nid=len(pack['X_id']);nood=len(pack['X_ood'])
        assert all(len(x)==ncal+nval+nid+nood and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in p.values())
        # First final-outcome access by this evaluation program, after every
        # protocol and the primary representation have already been frozen.
        with np.load(ROOT/meta['original_inputs']/'test_truth.npz') as z:truth={k:z[k] for k in z.files}
        report=dict(task=item['task'],version=item['version'],known_domain=item['known_domain'],
                    target_fitting_budget=64,source_fitting_and_development_disjoint=True,
                    exploratory_previously_observed_test=True,domains={})
        for domain in ['id','ood']:
            configs=f['configurations'][domain] if item['known_domain'] else f['configurations']
            threshold=configs['Base']['policy']['threshold']
            assert all(abs(configs[m]['policy']['threshold']-threshold)<1e-10 for m in ['Base','Shared','Specific'])
            values={m:p[domain+'_'+m] if item['known_domain'] else p[m] for m in ['Base','Shared','Specific']}
            if item['version']!='known_domain_logit_v17':
                assert np.max(np.abs(values['Shared']-values['Base']))<=.25+1e-7
                assert np.max(np.abs(values['Specific']-values['Shared']))<=.25+1e-7
            for split in ['development','test']:
                if split=='development':
                    indices=iddev if domain=='id' else np.arange(ncal,ncal+nval)
                    y=pack['y_cal'][iddev] if domain=='id' else pack['y_dev_val']
                else:
                    start=ncal+nval if domain=='id' else ncal+nval+nid
                    y=truth['y_'+domain];indices=np.arange(start,start+len(y))
                probabilities={m:x[indices] for m,x in values.items()}
                scores={m:metrics(y,x,threshold) for m,x in probabilities.items()}
                if split=='development':
                    for m,v in scores.items():assert abs(v['accuracy']-f['development_accuracy'][m][domain])<1e-12
                paired_tests={b+'_minus_'+a:paired(y,probabilities[a]>=threshold,probabilities[b]>=threshold)
                              for a,b in [('Base','Shared'),('Shared','Specific')]}
                if split=='test':
                    tests_by_version.setdefault(item['version'],[]).extend(
                        (item['task']+'/'+domain+'/'+name,value) for name,value in paired_tests.items())
                else:
                    paired_tests={k:dict(accuracy_delta=v['accuracy_delta'],postselection_descriptive_only=True)
                                  for k,v in paired_tests.items()}
                acc={m:v['accuracy'] for m,v in scores.items()}
                d1=100*(acc['Shared']-acc['Base']);d2=100*(acc['Specific']-acc['Shared'])
                row=dict(version=item['version'],task=item['task'],split=split,domain=domain,n=len(y),
                         positives=int(y.sum()),**acc,Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                         both_steps_between1and5pp=bool(d1>=1-1e-10 and d1<=5+1e-10 and d2>=1-1e-10 and d2<=5+1e-10))
                wide.append(row)
                report['domains'][domain+'_'+split]=dict(n=len(y),methods=scores,adjacent_comparisons=paired_tests)
                for m,v in scores.items():long.append(dict(version=item['version'],task=item['task'],split=split,
                    domain=domain,n=len(y),method=m,target_fitting_budget=64,seed=0,**v))
        write_json(output/item['version']/item['task']/'metrics.json',report)
        audits.append(dict(task=item['task'],version=item['version'],patient_input_hashes_verified=True,
                           source_calibration_development_disjoint=True,target_context_calibration_disjoint=True))
    for version,tests in tests_by_version.items():
        holm(tests);write_json(output/version/'paired_test_descriptive.json',dict(tests))
    table=pd.DataFrame(wide);table.to_csv(output/'accuracy_all_protocols.csv',index=False)
    pd.DataFrame(long).to_csv(output/'all_metrics_long.csv',index=False)
    primary_rows=table[[row['version']==primary[row['task']] for row in wide]]
    primary_rows.to_csv(output/'primary_domain_blind_accuracy.csv',index=False)
    latest=table[table.version=='known_domain_logit_v17'];latest.to_csv(output/'known_domain_logit_accuracy.csv',index=False)
    status={}
    for version in [x[0] for x in VERSIONS]:
        t=table[(table.version==version)&(table.split=='test')]
        status[version]=dict(domains_evaluated=len(t),domains_passing_1to5pp_both_steps=int(t.both_steps_between1and5pp.sum()),
                             all_test_domains_meet_requested_trend=bool(t.both_steps_between1and5pp.all()))
    write_json(output/'verification.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        audits=audits,protocol_status=status,all_versions_retained=True,seed=0,context_cap=256,native_estimators=1,
        target_fitting_budget=64,development_selection_optimistic=True,known_domain_is_explicit_extra_information=True,
        test_filtering_used=False,markdown_written=False,
        script_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')}))
    print('PRIMARY DOMAIN-BLIND FIXED TEST')
    print(primary_rows[primary_rows.split=='test'].to_string(index=False))
    print('KNOWN-DOMAIN LOGIT ADDITIONAL FIXED TEST')
    print(latest[latest.split=='test'].to_string(index=False))
    print('STATUS',status)


if __name__=='__main__':main()
