"""Bounded, transparent Base-only difficulty calibration of seven regression shifts.

All candidate vectors use one fixed positive amplitude per family across every
dataset and nominal strength. Previous actual model predictions are reused only
at the exact same family/amplitude. This is benchmark development, not a claim
to recover the original paper's generator or its family-wise scores.
"""
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from prepare_data import RUN,ROOT,atomic_json,protocol
from seven_shifts import FAMILIES,registry,regression_clean
from calibrated_seven_protocol import materialize
from run_tabicl import metrics

BASE_VECTOR={'covariate':.55,'label':.45,'concept':.45,'selection':.1,'domain':1.25,'temporal':1.5,'support':.1}
CANDIDATES=[('positive_family_match',BASE_VECTOR),
    ('concept_0.50',BASE_VECTOR|{'concept':.50}),
    ('concept_0.55',BASE_VECTOR|{'concept':.55})]


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    entries=[m for m in registry() if m['task']=='regression'];assert len(entries)==33
    target_path=ROOT/'PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/composite/regression/summary_oof.csv'
    with target_path.open() as f:old=[r for r in csv.DictReader(f) if float(r['strength'])>0]
    target={k:math.fsum(float(r['mean_base_'+k]) for r in old)/len(old) for k in ['r2','nrmse']}
    cache={};candidate_rows=[]
    for name,vector in CANDIDATES:
        assert set(vector)==set(FAMILIES) and all(v>0 for v in vector.values())
        rows=[]
        for meta in entries:
            for family in FAMILIES:
                amplitude=vector[family];key=(meta['task_id'],amplitude)
                if key not in cache:
                    source=RUN/'results/severity_calibration/seven_benchmark_development'/f'amplitude_{amplitude:g}'/f"{meta['task_id']}.npz"
                    info=json.loads(source.with_suffix('.json').read_text());assert info['status']=='COMPLETE'
                    with np.load(source) as f:cache[key]=(f['predictions'].copy(),f['y_target'].copy(),source)
                predictions,truth,source=cache[key]
                for index,rho in enumerate([20,50,80,100]):
                    prediction=predictions[list(FAMILIES).index(family)*4+index]
                    rows.append({'task_id':meta['task_id'],'data_id':meta['data_id'],'family':family,'rho':rho,
                        'amplitude_multiplier':amplitude,'source_prediction':str(source.relative_to(RUN)),
                        **metrics(truth,prediction,'regression',max(float(truth.std()),1e-12))})
        assert len(rows)==33*7*4
        # The complete balanced grid gives every dataset/family/strength equal weight.
        averages={k:math.fsum(r[k] for r in rows)/len(rows) for k in ['r2','nrmse']}
        distance=sum(((averages[k]-target[k])/.01)**2 for k in target)
        candidate_rows.append({'name':name,'family_amplitude_multipliers':vector,'datasets':33,'conditions':len(rows),
            'metrics':averages,'distance':distance,'rows':rows})
    best=min(candidate_rows,key=lambda r:r['distance'])
    result={'status':'COMPLETE','target':target,'candidates':candidate_rows,'selected':{k:v for k,v in best.items() if k!='rows'},
        'selection_role':'benchmark-development aggregate difficulty matching using Base only',
        'adaptive_refinement_from_existing_Base_curves':True,'candidate_results_all_retained':True,
        'target_labels_modified':False,'datasets_dropped':False,'TabACE_scores_used_for_selection':False,
        'original_generator_recovered':False,'familywise_original_scores_reproduced':False}
    atomic_json(RUN/'results/seven_regression_recalibration.json',result)
    if a.freeze:
        configuration={'status':'FROZEN','track':'seven_calibrated_v2','task':'regression',
            'family_amplitude_multipliers':best['family_amplitude_multipliers'],'nominal_strengths':[.2,.5,.8,1.],
            'datasets':33,'seed':0,'context_budget':256,'history_reference_and_query_splits':'unchanged from seven_calibrated',
            'selection':'minimum aggregate Base distance among three documented positive family vectors',
            'benchmark_queries_are_development_data':True,'TabACE_scores_used_for_selection':False,
            'exact_original_paper_protocol':False,'target':target,'Base_metrics':best['metrics']}
        configuration['protocol_sha256']=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
        configuration['frozen_utc']=datetime.now(timezone.utc).isoformat()
        frozen=RUN/'configs/seven_regression_protocol_v2.json'
        if frozen.exists():assert json.loads(frozen.read_text())['protocol_sha256']==configuration['protocol_sha256']
        else:atomic_json(frozen,configuration)
        for meta in entries:
            clean=regression_clean(meta)
            for family in FAMILIES:
                amplitude=best['family_amplitude_multipliers'][family]
                predictions,truth,source=cache[(meta['task_id'],amplitude)]
                for index,rho in enumerate([20,50,80,100]):
                    data,info=materialize(meta,family,rho,clean,amplitude)
                    assert np.array_equal(truth,data['y_target'])
                    info.update(track='seven_calibrated_v2',condition_materialized=True,n_classes=None,
                        protocol_sha256=configuration['protocol_sha256'],severity_selection_data_role='benchmark development',
                        calibration_protocol='familywise amplitude selected using benchmark-development Base only',
                        original_paper_generator_recovered=False)
                    out=RUN/'cache/seven_calibrated_v2/regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if not out.with_suffix('.json').exists():
                        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
                        np.savez_compressed(tmp,**data);tmp.replace(out);atomic_json(out.with_suffix('.json'),info)
                    prediction=predictions[list(FAMILIES).index(family)*4+index]
                    destination=RUN/'results/calibrated_base/seven_calibrated_v2/regression'/str(meta['task_id'])/out.name
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.with_suffix('.json').exists():
                        tmp=destination.with_suffix('.tmp.npz');np.savez_compressed(tmp,prediction=prediction,y_target=truth);tmp.replace(destination)
                        atomic_json(destination.with_suffix('.json'),info|{'status':'COMPLETE','method':'Base',
                            'metrics':metrics(truth,prediction,'regression',max(float(truth.std()),1e-12)),
                            'actual_cached_backbone_prediction_reused':str(source.relative_to(RUN)),
                            'cached_prediction_family':family,'cached_prediction_amplitude':amplitude})
            print('V2_FROZEN_DATASET',meta['task_id'],flush=True)
    print(json.dumps({'target':target,'selected':result['selected'],'frozen':a.freeze}),flush=True)


if __name__=='__main__':main()
