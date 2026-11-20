"""Consolidate all attempted correlated-error safeguards, including failures."""
import hashlib,json
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,write_json

VERSIONS=['correlated_risk_v4','calibrated_trust_v5','smooth_update_v6']
SCRIPTS=['correlated_risk_calibration.py','calibrated_trust_radius.py','smooth_update_guard.py']

def macro(df,columns):return df.groupby(['method','data_id'])[columns].mean().groupby('method').mean()

def main():
    output=OUT/'common_error_study_summary';output.mkdir(exist_ok=True)
    natural=[];effects=[];pooled=[];changes=[];selection=[]
    for bone in ['tabicl','tabpfn']:
        for track in ['five','seven_reimplemented']:
            original=None
            for version in VERSIONS:
                folder=OUT/version/bone/track
                df=pd.read_csv(folder/'performance.csv')
                candidate=df[df.method=='Original'].sort_values(['data_id','family','rho'])
                if original is None:original=candidate
                else:assert np.allclose(candidate.r2,original.r2,rtol=1e-8,atol=1e-9)
                for scope,d in [('clean',df[df.rho==0]),('nonclean',df[df.rho!=0])]:
                    a=macro(d,['r2','nrmse'])
                    for method,row in a.iterrows():
                        natural.append(dict(version=version,backbone=bone,track=track,scope=scope,method=method,
                            datasets=d.data_id.nunique(),r2=float(row.r2),nrmse=float(row.nrmse),
                            delta_r2=float(row.r2-a.loc['Original','r2']),delta_nrmse=float(row.nrmse-a.loc['Original','nrmse'])))
                    paired=d.groupby(['data_id','method']).r2.mean().unstack()
                    for method in paired.columns:
                        if method=='Original':continue
                        delta=(paired[method]-paired.Original).to_numpy();rng=np.random.default_rng(0)
                        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                        changes.append(dict(version=version,backbone=bone,track=track,scope=scope,method=method,
                            mean_delta_r2=float(delta.mean()),ci_lower=float(np.quantile(boot,.025)),ci_upper=float(np.quantile(boot,.975)),
                            improved_datasets=int((delta>1e-9).sum()),worsened_datasets=int((delta<-1e-9).sum()),
                            dataset_block_bootstrap_seed=0))
                sd=pd.read_csv(folder/'stress_by_condition.csv')
                for scope,d in [('all',sd),('nonclean',sd[sd.rho!=0])]:
                    for (pattern,magnitude),g in d.groupby(['pattern','magnitude_source_sd']):
                        r2=macro(g,['r2']);records=[]
                        for method,h in g.groupby('method'):
                            records.append(dict(method=method,mean_output_displacement_source_sd=float(np.average(h.mean_displacement_source_sd,weights=h.n)),
                                r2=float(r2.loc[method,'r2'])))
                        base=next(r for r in records if r['method']=='Original')
                        for record in records:
                            effects.append(dict(version=version,backbone=bone,track=track,scope=scope,pattern=pattern,
                                magnitude_source_sd=magnitude,**record,
                                displacement_reduction_percent=100*(1-record['mean_output_displacement_source_sd']/max(base['mean_output_displacement_source_sd'],1e-12)),
                                delta_stress_r2=record['r2']-base['r2']))
                p=pd.read_csv(folder/'stress_pooled.csv');p['version']=version;pooled.extend(p.to_dict('records'))
            for f in range(5):
                r=json.loads((OUT/'correlated_risk_v4'/bone/track/f'fold_{f}.json').read_text())
                s=json.loads((OUT/'calibrated_trust_v5'/bone/track/f'fold_{f}.json').read_text())
                choice=r['selected']
                selection.append(dict(backbone=bone,track=track,outer_fold=f,
                    risk_augmentation_weight=choice['augmentation_weight'] if choice else None,
                    risk_gamma=choice['gamma'] if choice else None,
                    risk_control_gamma=r['no_augmentation_control_gamma'],selected_trust_radius=s['selected_radius']))
    for name,records in [('natural_metrics',natural),('stress_effects',effects),('pooled_displacement',pooled),
                         ('paired_dataset_changes',changes),('source_selection',selection)]:
        pd.DataFrame(records).to_csv(output/f'{name}.csv',index=False)
    for version,script in zip(VERSIONS,SCRIPTS):
        plan=json.loads((OUT/version/'frozen_plan.json').read_text())
        assert hashlib.sha256((ROOT/'rebuttal/gate_sensitivity_replay/scripts'/script).read_bytes()).hexdigest()==plan['source_sha256']
        for item in plan.get('parent_gate_hashes',[]):
            assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    parent=json.loads((OUT/'frozen_plan.json').read_text())
    policies=[p for case in parent['cases'] for p in case['policies']]
    for p in policies:assert hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()==p['sha256']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    write_json(output/'experiment_log.json',dict(purpose='Resolve how correlated, jointly wrong alternative predictions affect the gate and test bounded-influence repairs',
        settings=dict(seed=0,backbones=['TabICL','TabPFN'],regression_datasets=dict(five=35,seven=33),
            data_and_context_banks_unchanged=True,backbone_calls=0),
        results=dict(all_three_attempts_retained=True,source_selection=selection),
        analysis='Risk-head source validation generally preferred zero attenuation; trust-radius calibration and fixed-radius controls expose the accuracy/robustness tradeoff. Smooth saturation directly bounds update influence and is evaluated as the predeclared final iteration.',
        next_steps='Choose and freeze one development variant only after reviewing clean, nonclean, and full severity curves; use separate data for confirmatory claims',
        evaluation_status='Reconstructed development evaluation with adaptive algorithm iteration; held-out labels are used for scoring but not parameter fitting or within-run selection',
        stress_protocol='Fixed seed signs per query, independent or shared; original and all variants get identical biases. Unlike the older stress replay, this is not worst-sign selection.',
        averaging='Natural R2 and stress R2: equal datasets then conditions. Mean displacement: query-condition weighted; pooled P95 includes clean.',
        frozen_original_policy_hashes_verified=len(policies),markdown_bytes=0))
    print('LATEST_NATURAL')
    nd=pd.DataFrame(natural)
    print(nd[(nd.scope=='nonclean')&nd.method.isin(['Original','Learned_calibrated_radius','Learned_radius6','Smooth_cap3','Smooth_cap6'])].to_string(index=False))
    print('LATEST_COMMON_BIAS')
    e=pd.DataFrame(effects)
    print(e[(e.scope=='nonclean')&(e.pattern=='shared_bias_5of5')&e.magnitude_source_sd.isin([1,3,5,10])&e.method.isin(['Smooth_cap3','Smooth_cap6','Learned_radius6'])].to_string(index=False))
    print('COMPLETE')

if __name__=='__main__':main()
