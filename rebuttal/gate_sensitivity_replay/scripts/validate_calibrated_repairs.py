"""Unchanged, label-free stress protocol for source-calibrated repairs."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,json
from collections import defaultdict
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,cells,regress,write_json
from calibrated_regression_repairs import DEST as V2,numpy_predict
from bounded_gain_gate import DEST as V3,predict

def run(backbone,track):
    pool=defaultdict(list);rows=[];heads={}
    for fold in range(5):
        root=V2/backbone/track
        heads[fold]={v:np.asarray(json.loads((root/v/f'fold_{fold}.json').read_text())['parameters'])
            for v in ['A_learned_trust','C_calibrated_guard']}
        heads[fold]['A_bounded_gain']=np.asarray(json.loads((V3/backbone/track/f'fold_{fold}.json').read_text())['parameters'])
    def apply(method,b,k,theta,head):
        if method=='Original':return regress(b,k.T,theta,0.,1.)[0]
        if method=='A_bounded_gain':return predict(b,k,theta,head[method])
        return numpy_predict(method,b,k,theta,head[method])
    for i,m,rawb,rawk,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T
        initial={v:apply(v,b,k,t,heads[fold]) for v in ['Original','A_learned_trust','A_bounded_gain','C_calibrated_guard']}
        for stress in ['common_bias','single_outlier']:
            methods=['Original','A_learned_trust','A_bounded_gain'] if stress=='common_bias' else ['Original','C_calibrated_guard']
            for magnitude in ([1.,5.,10.,100.] if stress=='common_bias' else [10.,100.]):
                worst={v:np.zeros(len(b)) for v in methods}
                median=np.median(k,axis=1)
                for branch in ([-1] if stress=='common_bias' else range(5)):
                    for sign in [-1.,1.]:
                        changed=k.copy()
                        if stress=='common_bias':changed+=sign*magnitude
                        else:changed[:,branch]=median+sign*magnitude
                        for method in methods:
                            p=apply(method,b,changed,t,heads[fold])
                            worst[method]=np.maximum(worst[method],np.abs(p-initial[method]))
                for method,d in worst.items():
                    pool[(stress,magnitude,method)].append(d)
                    rows.append(dict(backbone=backbone,track=track,stress=stress,magnitude_source_sd=magnitude,
                        method=method,data_id=m['data_id'],family=m['family'],rho=m['rho'],n=len(b),
                        mean_displacement_source_sd=float(d.mean()),p95_displacement_source_sd=float(np.quantile(d,.95))))
        if i%200==0:print('V2_V3_STRESS',backbone,track,i,flush=True)
    dest=V3/backbone/track
    pd.DataFrame(rows).to_csv(dest/'stress_by_condition.csv',index=False)
    pooled=[]
    for (stress,magnitude,method),parts in pool.items():
        d=np.concatenate(parts)
        pooled.append(dict(backbone=backbone,track=track,stress=stress,magnitude_source_sd=magnitude,method=method,
            n_query_conditions=len(d),mean_displacement_source_sd=float(d.mean()),
            p95_displacement_source_sd=float(np.quantile(d,.95)),max_displacement_source_sd=float(d.max())))
    pd.DataFrame(pooled).to_csv(dest/'stress_pooled.csv',index=False)
    # Cross-check the original stress against the preceding, frozen replay.
    reference=OUT/'algorithm_repairs_v1'/'correlated_gate'/backbone/track/'stress_pooled.csv'
    old=pd.read_csv(reference)
    for r in pooled:
        if r['method']=='Original' and r['stress']=='common_bias' and r['magnitude_source_sd'] in [1,5,10,100]:
            target=old[(old.method=='Original')&(old.biased_branches==5)&(old.magnitude_source_sd==r['magnitude_source_sd'])].iloc[0]
            assert abs(r['p95_displacement_source_sd']-target.p95_displacement_source_sd)<1e-7
    write_json(dest/'stress_audit.json',dict(original_stress_reproduced=True,labels_used_to_select_injection=False,
        own_unstressed_method_used_as_displacement_reference=True,scope='Pooled clean and nonclean query-condition predictions'))
    print('V2_V3_STRESS_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    run(a.backbone,a.track)
