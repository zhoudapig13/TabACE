"""Select a global regression-shift amplitude using reserved queries only."""
import csv,json,math
import numpy as np
from prepare_data import RUN,ROOT,atomic_json
from calibrated_seven_protocol import candidates


def main():
    manifest=json.loads((RUN/'configs/seven_severity_calibration_manifest.json').read_text());expected=manifest['ready_datasets']
    rows=[]
    for config in [{'name':'clean','amplitude_multiplier':0.}]+candidates():
        paths=list((RUN/'results/severity_calibration/seven_regression'/config['name']).glob('*.json'))
        observations=[json.loads(p.read_text()) for p in paths]
        if observations:
            values=np.asarray([[np.mean([r[k] for r in m['rows']]) for k in ['r2','nrmse']] for m in observations])
            rows.append({'configuration':config['name'],'datasets':len(observations),'complete':len(observations)==expected,
                         'r2':float(values[:,0].mean()),'nrmse':float(values[:,1].mean())})
    complete=len(rows)==1+len(candidates()) and all(r['complete'] for r in rows)
    result={'status':'COMPLETE' if complete else 'IN_PROGRESS','rows':rows,'calibration_datasets':expected,
            'selection_uses_original_test_labels':False,'selection_uses_TabACE_results':False}
    if complete:
        path=ROOT/'PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/composite/regression/summary_oof.csv'
        with path.open() as f:paper=[r for r in csv.DictReader(f) if float(r['strength'])>0]
        with (RUN/'results/summary_long.csv').open() as f:existing=list(csv.DictReader(f))
        clean=next(r for r in rows if r['configuration']=='clean');target={}
        for metric,offset in [('r2',-.005),('nrmse',.005)]:
            paper_mean=math.fsum(float(r['mean_base_'+metric]) for r in paper)/len(paper)
            main_clean=float(next(r['mean'] for r in existing if r['backbone']=='tabicl' and r['method']=='base_protocol_check'
                and r['task']=='regression' and r['block']=='clean_shifts_reimplemented' and r['metric']==metric))
            target[metric]=paper_mean+(clean[metric]-main_clean)+offset
        choices=[r for r in rows if r['configuration']!='clean']
        for r in choices:r['calibration_distance']=((r['r2']-target['r2'])/.03)**2+((r['nrmse']-target['nrmse'])/.03)**2
        selected=min(choices,key=lambda r:r['calibration_distance'])
        result.update(target_on_calibration=target,selected_configuration=next(c for c in candidates() if c['name']==selected['configuration']),
            selected_scores=selected,within_calibration_tolerance=abs(selected['r2']-target['r2'])<=.03 and abs(selected['nrmse']-target['nrmse'])<=.03)
    atomic_json(RUN/'results/seven_severity_calibration_summary.json',result)
    if rows:
        with (RUN/'results/seven_severity_calibration_summary.csv').open('w',newline='') as f:
            fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
