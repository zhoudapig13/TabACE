"""Recompute every saved regression score; no version selection from queries."""
import json
import numpy as np
import pandas as pd
import calibrated_risk_fusion_v37 as f
from prepare_data import RUN, atomic_json

if __name__ == '__main__':
    for version, dirname in [('fixed_pool','tabace_context_adaptation_v29'),('full_pool','tabace_full_pool_v34')]:
        root=f.ROOT/version;source=RUN/'results'/dirname
        targets=json.loads((root/'paper_targets.json').read_text())['regression']
        rows=[];checks=[];verified=0
        for budget in [256,1024,4096,8192]:
            for block in ['clean','five','seven']:
                query=f.load(source,'query',budget,block)
                policy=root/'policies'/str(budget)/'regression'/block
                shared=json.loads((policy/'Shared.json').read_text())['model']
                specific={name:json.loads((policy/f'Specific_{name}.json').read_text())['model'] for name in {c['meta']['family'] for c in query}}
                group=[]
                for c in query:
                    path=root/'predictions'/str(budget)/c['key']
                    meta=json.loads(path.with_suffix('.json').read_text())
                    sh=f.predict(c,shared);model=specific[c['meta']['family']]
                    sp=sh if model is None else f.predict(c,model)
                    with np.load(path) as z:
                        assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
                        assert np.allclose(z['Shared'],sh,rtol=0,atol=1e-12)
                        assert np.allclose(z['Specific'],sp,rtol=0,atol=1e-12)
                        for method in ['Base','Shared','Specific']:
                            measured=f.metrics(z['y_target'],z[method],'regression',meta['scale'])
                            assert all(abs(measured[k]-meta['methods'][method][k])<1e-12 for k in measured)
                    assert meta['original_query_labels_used_for_fit'] is False
                    group.append(meta);verified+=1
                means={method:{key:float(np.mean([r['methods'][method][key] for r in group])) for key in ['r2','nrmse']} for method in ['Base','Shared','Specific']}
                for method in means:rows.append(dict(context=budget,block=block,method=method,datasets=len({c['meta']['task_id'] for c in query}),conditions=len(query),**means[method]))
                for j,key in enumerate(['r2','nrmse']):
                    sign=-1 if key=='nrmse' else 1;p=targets[block]
                    strict=sign*(means['Shared'][key]-means['Base'][key])>1e-9 and sign*(means['Specific'][key]-means['Shared'][key])>1e-9
                    gain=all(sign*(means[m][key]-means['Base'][key])>=max(0,sign*(p[m][j]-p['Base'][j]))-1e-9 for m in ['Shared','Specific'])
                    checks.append(dict(context=budget,block=block,metric=key,strict_order_pass=bool(strict),minimum_gain_pass=bool(gain),pass_all=bool(strict and gain)))
        pd.DataFrame(rows).to_csv(root/'averages.csv',index=False)
        atomic_json(root/'numerical_audit.json',dict(status='PREDICTIONS_VERIFIED',verified_conditions=verified,
            scope='regression only; not the full 48-check goal',required_metric_checks=24,
            passed_metric_checks=sum(r['pass_all'] for r in checks),checks=checks,
            clean_TRAIN_extra_supervision=True,original_paper_replication=False,
            original_query_labels_used_for_fit=False,Base_small_growth_still_unresolved=True))
        print('CALIBRATED_RISK_AUDIT',version,verified,sum(r['pass_all'] for r in checks),'/24',flush=True)
