import json,csv
import numpy as np
from prepare_data import RUN,atomic_json
from run_historical_feature_severity import CONFIGURATIONS

target=json.loads((RUN/'results/severity_calibration_summary.json').read_text())['target_on_calibration']
rows=[]
for cfg in CONFIGURATIONS:
    beta=cfg['beta'];m=cfg['feature_multiplier'];name=f'beta_{beta:g}' if m==1 else f'beta_{beta:g}_features{m}'
    files=list((RUN/'results/severity_calibration/historical_feature'/name).glob('*.json'))
    if not files:continue
    means=np.asarray([[np.mean([r[k] for r in json.loads(p.read_text())['rows']]) for k in ['accuracy','auc']] for p in files])
    a,u=means.mean(0);rows.append(cfg|{'name':name,'datasets':len(files),'complete':len(files)==63,'accuracy':float(a),'auc':float(u),
        'distance':float(((a-target['accuracy'])/.02)**2+((u-target['auc'])/.03)**2)})
complete=len(rows)==len(CONFIGURATIONS) and all(r['complete'] for r in rows)
result={'status':'COMPLETE' if complete else 'IN_PROGRESS','rows':rows,'target_on_calibration':target,
        'original_test_labels_used':False,'method_order_used':False,'source_labels_unchanged':True,
        'protocol_role':'additional historical feature-only variant, not exact original-paper restoration'}
if complete:
    best=min(rows,key=lambda r:r['distance']);result.update(selected_scores=best,
        selected_configuration={k:best[k] for k in ['beta','feature_multiplier','name']},
        within_tolerance=abs(best['accuracy']-target['accuracy'])<=.02 and abs(best['auc']-target['auc'])<=.03)
atomic_json(RUN/'results/historical_feature_severity_summary.json',result)
print(json.dumps(result,ensure_ascii=False),flush=True)
