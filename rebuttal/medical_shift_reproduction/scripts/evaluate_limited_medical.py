"""Evaluate prespecified frozen low-label variants; do not feed outcomes to search."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_public_shift import ROOT,metrics,paired,write_json,array_hash
from summarize_public_shift import holm


def main(plan_name='final_medical_limited_selection.json',out_name='final_medical_limited'):
    assert Path(plan_name).name==plan_name and Path(out_name).name==out_name
    plan=json.loads((ROOT/'configs'/plan_name).read_text())
    output=ROOT/'results'/out_name
    all_rows=[];all_tests=[];details=[]
    for item in plan['selections']:
        task=item['task'];data=ROOT/item['data'];dest=output/task
        frozen_path=ROOT/item['frozen']
        frozen=json.loads(frozen_path.read_text());meta=json.loads((data/'metadata.json').read_text())
        with np.load(data/'inputs.npz') as z:pack={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in pack.items()}==meta['input_hashes']
        with np.load(data/item['predictions']) as z:p={m:z[m] for m in ['Base','Shared','Specific']}
        threshold=frozen['configurations']['Base']['policy']['threshold']
        assert all(abs(frozen['configurations'][m]['policy']['threshold']-threshold)<1e-10 for m in p)
        ncal=len(pack['y_cal'])+32
        sizes=[ncal,len(pack['y_dev_val']),len(pack['X_id']),len(pack['X_ood'])]
        assert all(len(x)==sum(sizes) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in p.values())
        assert max(abs(p['Shared']-p['Base']))<=.25+1e-8
        assert max(abs(p['Specific']-p['Shared']))<=.25+1e-8
        # Save exact frozen-selection hashes BEFORE evaluator opens final outcomes.
        write_json(dest/'evaluation_selection.json',dict(
            frozen_selection=str(frozen_path.relative_to(ROOT)),sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            evaluation_started_at_utc=datetime.now(timezone.utc).isoformat(),
            extra_target_fitting_labels=64,target_fitting_and_validation_disjoint=True,
            method_status='DAGGER conservative target-supervised development adapters',
            test_data_used_to_select_this_configuration=False,test_patients_previously_observed=True))
        with np.load(ROOT/meta['original_inputs']/'test_truth.npz') as z:
            truth={k:z[k] for k in z.files}
        report=dict(task=task,context_cap=256,native_estimators=1,seed=0,target_fitting_labels=64,
                    target_context_labels=32,target_calibration_labels=32,
                    separate_development_validation_labels=len(pack['y_dev_val']),
                    development_postselection=True,exploratory_previously_observed_test=True,domains={})
        offset=ncal
        for domain,y in [('development_validation',pack['y_dev_val']),('id',truth['y_id']),('ood',truth['y_ood'])]:
            values={m:x[offset:offset+len(y)] for m,x in p.items()};offset+=len(y)
            score={m:metrics(y,x,threshold) for m,x in values.items()}
            tests={b+'_minus_'+a:paired(y,values[a]>=threshold,values[b]>=threshold)
                   for a,b in [('Base','Shared'),('Shared','Specific')]}
            if domain=='development_validation':
                for m in score:assert abs(score[m]['accuracy']-frozen['development_validation_accuracy'][m])<1e-12
                # Post-selection development significance would be misleading.
                tests={k:dict(accuracy_delta=v['accuracy_delta'],postselection_descriptive_only=True) for k,v in tests.items()}
            if domain=='ood':all_tests.extend((task+'/'+name,value) for name,value in tests.items())
            report['domains'][domain]=dict(n=len(y),positives=int(y.sum()),methods=score,adjacent_comparisons=tests)
            for method,s in score.items():all_rows.append(dict(task=task,domain=domain,method=method,
                n=len(y),positives=int(y.sum()),target_fitting_labels=64,seed=0,**s))
        write_json(dest/'metrics.json',report);details.append(report)
    holm(all_tests)
    out=output;out.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out/'all_metrics.csv',index=False)
    write_json(out/'ood_paired_tests.json',dict(all_tests))
    wide=[]
    for result in details:
        for domain,d in result['domains'].items():
            acc={m:s['accuracy'] for m,s in d['methods'].items()}
            wide.append(dict(task=result['task'],domain=domain,n=d['n'],**acc,
                              Shared_minus_Base_pp=100*(acc['Shared']-acc['Base']),
                              Specific_minus_Shared_pp=100*(acc['Specific']-acc['Shared'])))
    pd.DataFrame(wide).to_csv(out/'accuracy_comparison.csv',index=False)
    history=[]
    for path in sorted((ROOT/'results').glob('*/*/frozen_development_selection.json')):
        d=json.loads(path.read_text())
        if 'development_validation_accuracy' in d:
            history.append(dict(version=path.parent.parent.name,task=path.parent.name,**d['development_validation_accuracy']))
    pd.DataFrame(history).to_csv(out/'development_history.csv',index=False)
    write_json(out/'verification.json',dict(
        verified_at_utc=datetime.now(timezone.utc).isoformat(),input_hashes_checked=True,
        same_threshold_all_methods=True,probability_update_cap_verified=.25,
        target_fitting_labels=64,postselection_validation_marked=True,all_previous_versions_retained=True,
        script_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')},
        original_natural_results_preserved=True,markdown_written=False))
    print(pd.DataFrame(wide).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='final_medical_limited_selection.json')
    parser.add_argument('--output',default='final_medical_limited');args=parser.parse_args()
    main(args.plan,args.output)
