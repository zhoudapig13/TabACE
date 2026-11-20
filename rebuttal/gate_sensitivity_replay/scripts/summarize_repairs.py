"""Build auditable numerical tables for the three independent repairs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from algorithm_repairs import REPAIR
from correlated_repair import DEST
from replay_diagnostics import OUT, ROOT, write_json

def macro(frame,metrics):
    return frame.groupby(['method','data_id'])[metrics].mean().groupby('method').mean()

def main():
    performance=[];confidence=[];single=[];common=[];activation=[]
    for backbone in ['tabicl','tabpfn']:
        for track in ['five','seven_reimplemented']:
            cls=REPAIR/backbone/track/'classification'
            df=pd.read_csv(cls/'performance.csv')
            counts=pd.read_csv(cls/'confidence_counts.csv')
            for scope,subset in [('clean',df[df.rho==0]),('nonclean',df[df.rho!=0])]:
                averaged=macro(subset,['accuracy','log_loss'])
                for method,row in averaged.iterrows():
                    performance.append(dict(backbone=backbone,track=track,task='classification',variant='B_HC_repair',
                        scope=scope,method=method,datasets=subset.data_id.nunique(),**row.to_dict()))
            for threshold in [.9,.95]:
                subset=counts[(counts.rho!=0)&(counts.confidence_threshold==threshold)]
                summed=subset.groupby('method')[['n_queries','n_high_confidence','base_errors','corrected_errors','base_correct','new_errors']].sum()
                for method,row in summed.iterrows():
                    d=row.to_dict()
                    confidence.append(dict(backbone=backbone,track=track,threshold=threshold,method=method,**d,
                        correction_rate=float(row.corrected_errors/max(row.base_errors,1)),
                        harm_rate=float(row.new_errors/max(row.base_correct,1)),
                        hc_error_fraction_of_all_queries=float(row.base_errors/row.n_queries)))
            for variant,dest in [('C_Median_guard',REPAIR/backbone/track/'regression'),('A_Trust_gate',DEST/backbone/track)]:
                if not (dest/'performance.csv').exists():continue
                df=pd.read_csv(dest/'performance.csv')
                for scope,subset in [('clean',df[df.rho==0]),('nonclean',df[df.rho!=0])]:
                    for method,row in macro(subset,['r2','nrmse']).iterrows():
                        performance.append(dict(backbone=backbone,track=track,task='regression',variant=variant,
                            scope=scope,method=method,datasets=subset.data_id.nunique(),**row.to_dict()))
                if variant.startswith('C_'):
                    base=pd.read_csv(OUT/backbone/track/'regression'/'stress_pooled.csv')
                    before=base[(base.method=='Full')&base.magnitude_source_sd.isin([10,100])]
                    after=pd.read_csv(dest/'stress_pooled.csv')
                    after=after[after.magnitude_source_sd.isin([10,100])]
                    single.extend(pd.concat([before,after]).to_dict('records'))
                    triggered=pd.read_csv(dest/'clipping_counts.csv')
                    sums=triggered[['n','alternative_values','clipped_values','changed_final_predictions']].sum().to_dict()
                    activation.append(dict(backbone=backbone,track=track,variant=variant,**sums,
                        clipped_branch_fraction=sums['clipped_values']/sums['alternative_values'],
                        changed_final_fraction=sums['changed_final_predictions']/sums['n']))
                else:
                    common.extend(pd.read_csv(dest/'stress_pooled.csv').to_dict('records'))
                    triggered=pd.read_csv(dest/'activation_counts.csv')
                    sums=triggered[['n','trust_activated','changed']].sum().to_dict()
                    activation.append(dict(backbone=backbone,track=track,variant=variant,**sums,
                        trust_activated_fraction=sums['trust_activated']/sums['n']))
    dest=REPAIR/'summary';dest.mkdir(exist_ok=True)
    for name,records in [('natural_performance',performance),('high_confidence',confidence),
                         ('single_branch_stress',single),('correlated_branch_stress',common),('activation',activation)]:
        pd.DataFrame(records).to_csv(dest/f'{name}.csv',index=False)
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    policies=[p for case in plan['cases'] for p in case['policies']]
    import hashlib
    assert all(hashlib.sha256((ROOT/p['path']).read_bytes()).hexdigest()==p['sha256'] for p in policies)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    audit=dict(purpose='Address correlated regression errors, high-confidence classification errors, and single extreme regression branches with separate algorithm variants',
        seed=0,benchmark_unchanged=True,backbone_calls=0,source_policy_hashes_verified=len(policies),
        held_out_labels_used_for_training=False,markdown_bytes=0,
        input_scope='Appendix-rule reconstructed caches; not recovered original submission prediction archive',
        averaging='Natural metrics: equal datasets, then equal nonclean conditions within each dataset. Stress: pooled query-condition predictions including clean.',
        completed_common_mode_cases=len(common)//16 if common else 0,
        interpretation='Repairs are separate new methods. Report all cases, including negligible or negative accuracy changes. No result-based stopping or evaluation-label fitting.')
    write_json(dest/'experiment_log.json',audit)
    print('NATURAL_NONCLEAN')
    print(pd.DataFrame(performance).query('scope=="nonclean"').to_string(index=False))
    print('HC_0.9')
    print(pd.DataFrame(confidence).query('threshold==0.9').to_string(index=False))
    print('ACTIVATION')
    print(pd.DataFrame(activation).to_string(index=False))
    if common:
        print('COMMON100');print(pd.DataFrame(common).query('magnitude_source_sd==100 and biased_branches==5').to_string(index=False))
    print('SINGLE100');print(pd.DataFrame(single).query('magnitude_source_sd==100').to_string(index=False))

if __name__=='__main__':main()
