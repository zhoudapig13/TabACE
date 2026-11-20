"""Report every severity and select only from disjoint calibration-query scores."""
from collections import defaultdict
import csv,json,math
import numpy as np
from prepare_data import RUN,ROOT,atomic_json
from calibrated_five_protocol import candidates


def main():
    manifest=json.loads((RUN/'configs/severity_calibration_manifest.json').read_text());expected=manifest['ready_datasets']
    rows=[]
    for config in [{'name':'clean'}]+candidates():
        paths=list((RUN/'results/severity_calibration/five_classification'/config['name']).glob('*.json'))
        observations=[json.loads(p.read_text()) for p in paths]
        means=np.asarray([[np.mean([r[k] for r in m['rows']]) for k in ['accuracy','auc']] for m in observations])
        if len(observations):rows.append({'configuration':config['name'],'datasets':len(observations),'complete':len(observations)==expected,
            'accuracy':float(means[:,0].mean()),'auc':float(means[:,1].mean())})
    if not rows:raise RuntimeError('no completed calibration cases')
    with (RUN/'results/severity_calibration_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    complete=all(r['complete'] for r in rows) and len(rows)==1+len(candidates())
    result={'status':'COMPLETE' if complete else 'IN_PROGRESS','calibration_datasets':expected,'rows':rows,
        'selection_uses_only_disjoint_calibration_queries':True,'TabACE_used_for_selection':False}
    if complete:
        with (ROOT/'fair_budget_experiments/unified_context_ensemble/results/tabicl_origin_safe_final_v2/five_shift/classification/summary_oof.csv').open() as f:old=list(csv.DictReader(f))
        get=lambda clean,metric:math.fsum(float(r['mean_base_'+metric]) for r in old if (int(r['rho'])==0)==clean)/(5 if clean else 20)
        clean=next(r for r in rows if r['configuration']=='clean')
        target={'accuracy':clean['accuracy']-(get(True,'accuracy')-get(False,'accuracy'))-.005,
                'auc':clean['auc']-(get(True,'auc')-get(False,'auc'))-.005}
        options=[r for r in rows if r['configuration']!='clean']
        for r in options:r['calibration_distance']=((r['accuracy']-target['accuracy'])/.02)**2+((r['auc']-target['auc'])/.03)**2
        selected=min(options,key=lambda r:r['calibration_distance'])
        result.update(target_on_calibration=target,selected_configuration=next(c for c in candidates() if c['name']==selected['configuration']),
            selected_scores=selected,within_calibration_tolerance=abs(selected['accuracy']-target['accuracy'])<=.02 and abs(selected['auc']-target['auc'])<=.03,
            target_definition='match paper clean-to-perturbed degradation, with an additional 0.005 drop, on new calibration queries')
    atomic_json(RUN/'results/severity_calibration_summary.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
