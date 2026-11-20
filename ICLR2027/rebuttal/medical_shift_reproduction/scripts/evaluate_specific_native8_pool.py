"""Evaluate all frozen mixed-expert Specific adaptations without test selection."""
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from run_public_shift import ROOT, array_hash, metrics, paired, write_json

TASKS=['sepsis_A_to_B','sepsis_B_to_A','readmission_other_to_ER']


def main():
    out=ROOT/'results/medical_specific_native8_pool_evaluation'
    plan=[]
    for task in TASKS:
        path=ROOT/'results/medical_specific_native8_pool_v25'/task/'frozen_selection.json'
        f=json.loads(path.read_text())
        assert f['test_labels_opened'] is False and f['base_shared_unchanged']
        plan.append(dict(task=task,snapshot=str(path.relative_to(ROOT)),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(out/'frozen_evaluation_plan.json',dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),experiments=plan,
        test_metrics_used_for_selection=False,previous_test_cohorts_observed=True))
    rows=[];audits=[]
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data']
        with np.load(data/'inputs.npz') as z:pack={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in pack.items()}==f['metadata']['input_hashes']
        with np.load(data/'specific_native8_pool_predictions.npz') as z:p={k:z[k] for k in z.files}
        with np.load(data/'medical_common_native8_predictions.npz') as z:parent={k:z[k] for k in z.files}
        for domain in ['id','ood']:
            assert all(np.array_equal(p[domain+'_'+m],parent[domain+'_'+m]) for m in ['Base','Shared'])
        with np.load(ROOT/f['metadata']['original_inputs']/'test_truth.npz') as z:truth={k:z[k] for k in z.files}
        details=dict(task=item['task'],mixed_expert_adaptation=True,base_shared_frozen=True,domains={})
        for domain in ['id','ood']:
            threshold=f['configurations'][domain]['Base']['policy']['threshold']
            for split in ['development','test']:
                sl=slice(*f['query_layout'][domain+'_'+split])
                y=pack['y_'+domain+'_dev'] if split=='development' else truth['y_'+domain]
                probs={m:p[domain+'_'+m][sl] for m in ['Base','Shared','Specific_N8P']}
                assert all(len(x)==len(y) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in probs.values())
                scores={m:metrics(y,x,threshold) for m,x in probs.items()}
                if split=='development':
                    assert all(abs(s['accuracy']-f['development_accuracy'][m][domain])<1e-12 for m,s in scores.items())
                acc={m:s['accuracy'] for m,s in scores.items()}
                d1=100*(acc['Shared']-acc['Base']);d2=100*(acc['Specific_N8P']-acc['Shared'])
                rows.append(dict(task=item['task'],domain=domain,split=split,n=len(y),
                    **acc,Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                    specific_step_1to5pp=bool(1-1e-10<=d2<=5+1e-10),
                    all_steps_1to5pp=bool(1-1e-10<=d1<=5+1e-10 and 1-1e-10<=d2<=5+1e-10)))
                details['domains'][domain+'_'+split]=dict(methods=scores,
                    specific_vs_shared=paired(y,probs['Shared']>=threshold,probs['Specific_N8P']>=threshold)
                    if split=='test' else dict(postselection_descriptive_only=True))
        write_json(out/item['task']/'metrics.json',details)
        audits.append(dict(task=item['task'],input_hashes_match=True,base_shared_exactly_unchanged=True))
    table=pd.DataFrame(rows);table.to_csv(out/'accuracy_all.csv',index=False)
    test=table[table.split=='test']
    write_json(out/'verification.json',dict(audits=audits,test_domains=len(test),
        specific_steps_1to5pp=int(test.specific_step_1to5pp.sum()),all_steps_1to5pp=int(test.all_steps_1to5pp.sum()),
        target_fitting_budget=64,previous_test_cohorts_observed=True,markdown_written=False))
    print(test.to_string(index=False))


if __name__=='__main__':main()
