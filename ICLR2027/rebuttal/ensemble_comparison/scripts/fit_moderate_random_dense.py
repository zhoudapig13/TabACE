"""Bounded source-view adaptation: at most two modified members among six.

All configurations are fixed; calibration excludes the evaluated dataset fold.
No target transport, feature repair, recent-source branch or score cap is used.
"""
import argparse
import json
import numpy as np
from scipy.stats import rankdata
from prepare_data import RUN,atomic_json
from fit_combiners import fold_mapping,sample_weights
from run_tabicl import metrics,normalize

PATTERNS = ['raw_six','five_raw_plus_source_quantiles','five_raw_plus_features75',
            'four_raw_plus_source_quantiles_and_features75']


def outputs(bank):
    mean=normalize(bank.mean(0))
    z=np.log(np.clip(bank,1e-12,1)).mean(0)
    return np.stack([mean,normalize(np.exp(z-z.max(1,keepdims=True))),normalize(bank.max(0))])


def prediction(c,pattern,rule):
    return c['outputs'][pattern,rule]


def clean(cells,pattern=None,rule=0):
    rows=[]
    for c in cells:
        if c['meta']['rho']==0:
            p=c['base'] if pattern is None else prediction(c,pattern,rule)
            rows.append(metrics(c['y'],p,'classification',1.))
    return {k:float(np.mean([r[k] for r in rows])) for k in ['accuracy','auc']}


def score(cells,pattern,rule):
    total=0.
    for c,w in zip(cells,sample_weights(cells)):
        p=prediction(c,pattern,rule);y=c['y'];aucs=[]
        for label in range(p.shape[1]):
            mask=y==label;n=int(mask.sum());other=len(y)-n
            if n and other:
                ranks=rankdata(p[:,label],method='average')
                aucs.append((float(ranks[mask].sum())-n*(n+1)/2)/(n*other))
        total+=float(w.sum())*.5*(float(np.mean(p.argmax(1)==y))+(float(np.mean(aucs)) if aucs else .5))
    return total


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--track',required=True,choices=['five_feature_calibrated','seven_reimplemented'])
    parser.add_argument('--fold',required=True,type=int)
    parser.add_argument('--subspace50',action='store_true')
    args=parser.parse_args();track=args.track
    patterns=PATTERNS + (['five_raw_plus_features50','four_raw_plus_features75_and_features50'] if args.subspace50 else [])
    result_root='moderate_random_dense_subspace50' if args.subspace50 else 'moderate_random_dense'
    mapping,source=fold_mapping('classification',track)
    paths=sorted((RUN/'results/robust_six_all/tabicl'/track/'classification').glob('*/*.npz'))
    assert len(paths)==(1512 if track.startswith('five') else 1015)
    cells=[]
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['roles'][1]=='source_gaussian_quantiles'
        assert meta['roles'][3]=='least_drift_features_0.75'
        with np.load(path) as f:
            raw=f['raw_bank'];adapt=f['bank']
            banks=[raw,np.concatenate([raw[:5],adapt[1:2]]),
                   np.concatenate([raw[:5],adapt[3:4]]),
                   np.concatenate([raw[:4],adapt[1:2],adapt[3:4]])]
            if args.subspace50:
                assert meta['roles'][4]=='least_drift_features_0.5'
                banks += [np.concatenate([raw[:5],adapt[4:5]]),
                          np.concatenate([raw[:4],adapt[3:5]])]
            assert all(b.shape==raw.shape for b in banks)
            cells.append({'meta':meta,'path':path,'outputs':np.stack([outputs(b) for b in banks]),
                'base':f['base'],'y':f['y_target'],'fold':mapping[meta['data_id']],
                'clean_weight':(5. if track.startswith('five') else 7.) if meta['rho']==0 else 1.})
    calibration=[c for c in cells if c['fold']!=args.fold];test=[c for c in cells if c['fold']==args.fold]
    cal_ids={c['meta']['data_id'] for c in calibration};test_ids={c['meta']['data_id'] for c in test}
    assert cal_ids.isdisjoint(test_ids)
    base_clean=clean(calibration)
    for method in ['Random','DENSE']:
        out=RUN/'results'/result_root/'combiners'/track/f'fold_{args.fold}_{method}.json'
        if out.exists():continue
        best=None;trials=[]
        for pattern in range(len(patterns)):
            for rule in ([0] if method=='Random' else range(3)):
                observed=clean(calibration,pattern,rule)
                feasible=all(observed[k]>=base_clean[k]-.005 for k in observed)
                value=score(calibration,pattern,rule)
                trials.append({'pattern':pattern,'rule':rule,'calibration_score':value,'Clean':observed,'Clean_feasible':feasible})
                if feasible and (best is None or value>best[0]):best=(value,pattern,rule)
        feasible=best is not None
        _,pattern,rule=best if best is not None else (None,0,0)
        rows=[]
        for c in test:
            p=prediction(c,pattern,rule)
            rows.append(c['meta'] | {'method':method+'-6 moderate source views','outer_fold':args.fold,
                'selected_pattern':pattern,'selected_rule':rule} | metrics(c['y'],p,'classification',1.))
            target=RUN/'results'/result_root/'predictions'/track/method/str(c['meta']['task_id'])/c['path'].name
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=p,y_target=c['y']);tmp.replace(target)
        atomic_json(out,{'status':'COMPLETE','seed':0,'track':track,'fold':args.fold,'method':method,
            'calibration_data_ids':sorted(cal_ids),'test_data_ids':sorted(test_ids),'fold_map_source':source,
            'selected_pattern':pattern,'patterns':patterns,'selected_rule':rule,
            'rule_names':['uniform_MoE_mean','uniform_PoE_geometric','uniform_Max'],
            'calibration_base_Clean':base_clean,'calibration_feasible_configuration_found':feasible,'trials':trials,
            'clean_absolute_decline_limit':.005,'context_count':6,'rows_per_context':256,
            'maximum_modified_members':2,'test_labels_used_for_config_selection':False,
            'Shared_or_requested_Acc_bound_used_for_selection':False,'Shared_run':False,
            'Base_fallback_used':False,'new_GPU_inference':False,'rows':rows})
        print('MODERATE_DONE',track,args.fold,method,pattern,rule,feasible,flush=True)


if __name__=='__main__':main()
