"""Combine both independently audited tracks under the current numerical contract."""
import json
import pandas as pd
from prepare_data import RUN,ROOT,atomic_json
import unseen_shared_robustness_v117 as source


def main():
    parents={'perturbation':RUN/'results/shared_expanded_pool_v291',
             'shift':RUN/'results/shared_shift_target_adaptation_v292'}
    out=RUN/'results/six_mechanism_supervised_summary_v296';out.mkdir(parents=True,exist_ok=True)
    frames=[];accept=[];evidence={}
    for track,path in parents.items():
        assert json.loads((path/'independent_audit.json').read_text())['status']=='PASS'
        t=pd.read_csv(path/'family_averages.csv');t=t[(t.family!='clean')&t.method.isin(['Base','Shared_adapted'])].copy()
        t['track']=track;t['model_directory']=str(path)
        if track=='perturbation':
            t['extra_source_labels']=t['extra_training_labels'];t['target_adaptation_labels']=0.
        frames.append(t)
        for (task,family),g in t.groupby(['task','family']):
            primary='accuracy' if task=='classification' else 'r2'
            b=g[g.method=='Base'].iloc[0];s=g[g.method=='Shared_adapted'].iloc[0]
            gain=float(s[primary]-b[primary]);threshold=.05 if track=='perturbation' else .02
            accept.append({'track':track,'task':task,'family':family,'datasets':int(s.datasets),
                'Base':float(b[primary]),'Shared':float(s[primary]),'gain':gain,'required_gain_strictly_greater_than':threshold,
                'passed':gain>threshold,'extra_source_labels_mean':float(s.extra_source_labels),
                'target_adaptation_labels_mean':float(s.target_adaptation_labels)})
        evidence[track]={'directory':str(path),'plan_sha256':source.sha(path/'plan.json'),
            'independent_audit_sha256':source.sha(path/'independent_audit.json'),
            'family_averages_sha256':source.sha(path/'family_averages.csv')}
    assert len(accept)==12
    combined=pd.concat(frames,ignore_index=True);assert len(combined)==24
    combined.to_csv(out/'base_shared_all_metrics.csv',index=False)
    atomic_json(out/'numerical_acceptance.json',{'family_results':accept,
        'all_twelve_pass':all(g['passed'] for g in accept),'passed_count':sum(g['passed'] for g in accept),
        'evidence':evidence,'supervised_adaptation_not_original_frozen_Shared':True,
        'shift_target_adaptation_rows_excluded_from_final_evaluation':True,
        'Base_and_Shared_same_evaluation_rows_within_each_track':True,
        'single_seed':0,'TFM_context_cap':256,'script_sha256':source.sha(__file__)})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    print(combined.to_string(index=False),flush=True);print(json.dumps(accept,indent=2),flush=True)


if __name__=='__main__':main()
