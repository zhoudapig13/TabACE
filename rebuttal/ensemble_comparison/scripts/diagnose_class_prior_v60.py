"""TRAIN-only test of the source-calibration prior mismatch hypothesis.

The frozen query sampler balances a capped pool, then stratifies its split.
Existing calibration instead stratifies the expanded raw TRAIN pool. Measure
that discrepancy using only original TRAIN core/calibration labels; never use
query labels to estimate a target prior or to choose any weight.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import classification_risk_v58 as experiment
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_class_prior_diagnosis_v60'


def core_prior(meta,classes):
    pool=experiment.head.source.adaptation.raw.scaling.prepare('classification','five',meta)
    path=experiment.head.source.adaptation.raw.OUT/'validation_splits/classification/five'/f'{meta["task_id"]}.npz'
    with np.load(path) as z:held=z['validation_original_ids']
    assert not np.intersect1d(pool['source_ids'][:256],held).size
    prior=np.bincount(pool['y'][:256].astype(int),minlength=classes)/256.
    return prior


def weighted_score(cells,predictions,priors):
    scores=[]
    for c in cells:
        y=c['y'].astype(int);p=predictions[c['key']];prior=priors[str(c['meta']['task_id'])]
        empirical=np.bincount(y,minlength=p.shape[1])/len(y)
        w=prior[y]/empirical[y];aucs=[]
        for k in range(p.shape[1]):
            if np.any(y==k) and np.any(y!=k):aucs.append(roc_auc_score(y==k,p[:,k],sample_weight=w))
        scores.append(dict(accuracy=float(np.average(p.argmax(1)==y,weights=w)),auc=float(np.mean(aucs)) if aucs else None))
    return {key:float(np.mean([s[key] for s in scores if s[key] is not None])) for key in ['accuracy','auc']}


def run():
    rows=[];evaluations=[]
    for budget in [256,1024,4096,8192]:
        cells=experiment.head.load('validation',budget,'clean');priors={}
        for c in cells:
            tid=str(c['meta']['task_id']);prior=core_prior(c['meta'],c['base'].shape[1]);priors[tid]=prior
            empirical=np.bincount(c['y'],minlength=len(prior))/len(c['y'])
            if budget==256:rows.append(dict(task_id=c['meta']['task_id'],dataset=c['meta']['dataset'],core_prior=prior.tolist(),
                calibration_prior=empirical.tolist(),total_variation=float(.5*np.abs(prior-empirical).sum()),
                core_class_mass_represented_in_calibration=float(prior[empirical>0].sum()),
                source_calibration_rows=len(c['y'])))
        frozen=experiment.OUT/'policies'/str(budget)/'classification/clean/all_policies_frozen.json'
        policy=json.loads(frozen.read_text())['Shared'];base={c['key']:c['base'] for c in cells}
        if policy is None:pred=base
        else:
            raw=experiment.oof(cells,policy['model']);a=policy['blend']
            pred={c['key']:experiment.normalize((1-a)*base[c['key']]+a*raw[c['key']]) for c in cells}
        evaluations.append(dict(budget=budget,source_unweighted_Base=experiment.head.score(cells,base)[0],
            source_unweighted_Shared=experiment.head.score(cells,pred)[0],
            source_core_prior_weighted_Base=weighted_score(cells,base,priors),
            source_core_prior_weighted_Shared=weighted_score(cells,pred,priors)))
    summary=dict(status='DIAGNOSTIC_COMPLETE_NOT_A_PROTOCOL_CHANGE',datasets=len(rows),
        mean_core_calibration_prior_TV=float(np.mean([r['total_variation'] for r in rows])),
        median_core_calibration_prior_TV=float(np.median([r['total_variation'] for r in rows])),
        datasets_TV_over_010=sum(r['total_variation']>.1 for r in rows),
        incomplete_calibration_class_coverage=[r['task_id'] for r in rows if r['core_class_mass_represented_in_calibration']<.999999],
        rows=rows,source_evaluations=evaluations,query_labels_loaded=False,query_scores_used_for_weight_definition=False,
        benchmark_and_existing_predictions_unchanged=True,
        code_evidence='protocol._balanced_cap_indices/load_cc18_split versus tabace_validation_banks_v28.validation_split')
    atomic_json(OUT/'diagnosis.json',summary)
    print('TRAIN_PRIOR_MISMATCH',summary['mean_core_calibration_prior_TV'],summary['datasets_TV_over_010'],flush=True)
    print(json.dumps(evaluations,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
