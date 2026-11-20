import csv,json,math
import numpy as np
from prepare_data import ROOT,RUN,atomic_json
from run_seven_benchmark_severity import MULTIPLIERS

rows=[]
for s in MULTIPLIERS:
    files=list((RUN/'results/severity_calibration/seven_benchmark_development'/f'amplitude_{s:g}').glob('*.json'))
    if not files:continue
    values=np.asarray([[np.mean([r[k] for r in json.loads(p.read_text())['rows']]) for k in ['r2','nrmse']] for p in files])
    rows.append({'amplitude_multiplier':s,'datasets':len(files),'r2':float(values[:,0].mean()),'nrmse':float(values[:,1].mean()),'complete':len(files)==33})
result={'status':'COMPLETE' if len(rows)==len(MULTIPLIERS) and all(r['complete'] for r in rows) else 'IN_PROGRESS',
        'rows':rows,'benchmark_labels_used_for_selection':True,'results_role':'benchmark-development calibration','method_order_used':False}
if result['status']=='COMPLETE':
    with (ROOT/'PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/composite/regression/summary_oof.csv').open() as f:old=[r for r in csv.DictReader(f) if float(r['strength'])>0]
    target={k:math.fsum(float(r['mean_base_'+k]) for r in old)/len(old)+(-.005 if k=='r2' else .005) for k in ['r2','nrmse']}
    for r in rows:r['distance']=((r['r2']-target['r2'])/.02)**2+((r['nrmse']-target['nrmse'])/.02)**2
    best=min(rows,key=lambda r:r['distance']);result.update(target=target,selected_scores=best,
        selected_configuration={'name':f"amplitude_{best['amplitude_multiplier']:g}",'amplitude_multiplier':best['amplitude_multiplier']})
atomic_json(RUN/'results/seven_benchmark_severity_summary.json',result)
print(json.dumps(result,ensure_ascii=False),flush=True)
