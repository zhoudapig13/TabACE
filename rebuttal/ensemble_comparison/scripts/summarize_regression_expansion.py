"""Chat/JSON/CSV-only progress: average complete dataset grids, never partial grids."""
import csv
import json
import math
from collections import defaultdict
from prepare_data import RUN,atomic_json

TRACKS={'five':(35,21),'seven_calibrated_v2':(33,29)}


def old_rows(bone,method,track):
    rows=[]
    for p in sorted((RUN/'results'/bone/track/'regression').glob('*/*.json')):
        meta=json.loads(p.read_text());score=meta.get('methods',{}).get(method)
        if score:rows.append(meta|{k:score[k] for k in ['r2','nrmse']})
    return rows


def aggregate(rows,track):
    expected,conditions=TRACKS[track]
    grouped=defaultdict(list)
    for r in rows:grouped[r['task_id']].append(r)
    valid=[];datasets=[]
    for task_id,rs in grouped.items():
        unique={(r['family'],r['rho']) for r in rs}
        if len(rs)==conditions and len(unique)==conditions and sum(r['rho']==0 for r in rs)==1:
            valid+=rs;datasets.append(task_id)
    result={'datasets':len(datasets),'expected_datasets':expected,'conditions_seen':len(rows),
            'conditions_complete_grid':len(valid),'complete':len(datasets)==expected}
    for block,nonzero in [('clean',False),('shift',True)]:
        selected=[r for r in valid if (r['rho']>0)==nonzero]
        result[block]=({k:math.fsum(r[k] for r in selected)/len(selected) for k in ['r2','nrmse']} if selected else None)
    return result


def main():
    bases={};records=[]
    for bone,key in [('tabicl','base_protocol_check'),('tabpfn','tabpfn_v3_base_check')]:
        bases[bone]={t:aggregate(old_rows(bone,key,t),t) for t in TRACKS}
        records.append({'backbone':bone,'method':'Base','label':'Base','role':'anchor','tracks':bases[bone]})
    def append(label,method,bone,role,loader):
        records.append({'backbone':bone,'method':method,'label':label,'role':role,
                        'tracks':{t:aggregate(loader(t),t) for t in TRACKS}})
    append('Native-6','native_6','tabicl','native ensemble; canonical source',lambda t:old_rows('tabicl','native_6',t))
    append('PHE-6（官方六配置）','PHE6','tabpfn','official implementation; controlled six-configuration portfolio',
           lambda t:old_rows('tabpfn','PHE_official_custom6_native_source_calibration',t))
    def gpu_rows(component,method,t):
        rows=[]
        for p in sorted((RUN/'results/regression_expansion'/component/t/'regression').glob('*/*.json')):
            m=json.loads(p.read_text());s=m.get('methods',{}).get(method)
            if s:rows.append(m|{k:s[k] for k in ['r2','nrmse']})
        return rows
    for label,method,component,role in [
        ('Native-6','Native_PFN6','pfn','native preprocessing ensemble'),
        ('IQR-6→1（OligoICP策略适配）','IQR6_keep1_TabPFN_adapted','pfn','six-candidate strategy adaptation, not original 400-candidate OligoICP'),
        ('RF-PFN-6（depth2、自适应）','RF_PFN6_depth2_adaptive','rf','official algorithm, fixed six trees and depth2; additional node/validation calls')]:
        append(label,method,'tabpfn',role,lambda t,c=component,m=method:gpu_rows(c,m,t))
    def policy_rows(root,method,t,key='rows'):
        rows=[]
        for p in sorted((RUN/'results'/root/'policies'/t).glob(f'fold_*_{method}.json')):
            m=json.loads(p.read_text());rows+=m[key]
        return rows
    methods=[('GES-6＋校准正则','GES6'),('Convex Stacking-6＋校准正则','ConvexStacking6'),
        ('Affine Ridge Stacking-6','RidgeStacking6'),('Median-6＋校准混合','Median6'),
        ('FeatureMean-6＋校准混合','FeatureMean6')]
    variants=[]
    for label,method in methods:
        append(label,method,'tabicl','fixed six source-feature views, regularized on calibration datasets',
               lambda t,m=method:policy_rows('regression_expansion_static_clean',m,t))
        for root in ['regression_expansion_static','regression_expansion_static_clean']:
            for key in ['rows','raw_rows']:
                variants.append({'backbone':'tabicl','method':method,'stage':root,'variant':key,
                    'tracks':{t:aggregate(policy_rows(root,method,t,key),t) for t in TRACKS}})
    for label,method in [('跨TFM Mean-6＋校准混合','CrossTFMMean6'),('跨TFM GES-6＋校准正则','CrossTFMGES6')]:
        append(label,method,'tabicl+tabpfn','three candidates per TFM; fixed six total, calibrated',
               lambda t,m=method:policy_rows('regression_cross_tfm',m,t))
        variants.append({'backbone':'tabicl+tabpfn','method':method,'stage':'cross_TFM','variant':'raw_rows',
            'tracks':{t:aggregate(policy_rows('regression_cross_tfm',method,t,'raw_rows'),t) for t in TRACKS}})
    for r in records:
        anchor='tabpfn' if r['backbone']=='tabpfn' else 'tabicl'
        r['complete']=all(v['complete'] for v in r['tracks'].values())
        for track,values in r['tracks'].items():
            base=bases[anchor][track]
            if values['clean'] and base['clean']:
                values['Clean_difference_vs_Base']={k:values['clean'][k]-base['clean'][k] for k in ['r2','nrmse']}
                values['Clean_within_005']=values['clean']['r2']>=base['clean']['r2']-.005 and values['clean']['nrmse']<=base['clean']['nrmse']+.005
            if values['shift'] and base['shift']:
                values['shift_difference_vs_Base']={k:values['shift'][k]-base['shift'][k] for k in ['r2','nrmse']}
                ref={'tabicl':{'five':(.363,.783),'seven_calibrated_v2':(.467,.659)},
                     'tabpfn':{'five':(.237,.834),'seven_calibrated_v2':(.460,.656)}}[anchor][track]
                values['between_Base_and_historical_Shared']=values['shift']['r2']>=base['shift']['r2'] and values['shift']['nrmse']<=base['shift']['nrmse'] and values['shift']['r2']<ref[0] and values['shift']['nrmse']>ref[1]
    assert not any(r['method'] in ['Random','bootstrap_mean_6','tabpfn_bootstrap_mean_6','Random_partial010'] for r in records)
    summary={'status':'COMPLETE' if all(r['complete'] for r in records) else 'RUNNING',
        'seed':0,'records':records,'retained_raw_and_previous_variants':variants,
        'main_ensemble_rows':sum(r['role']!='anchor' for r in records),'Random_excluded_from_main_table':True,
        'partial_averages_use_only_fully_completed_dataset_grids':True,'Shared_run':False,'historical_reference_only':True,
        'score_ordering_bounds_not_used_for_selection':True,'Markdown_written':False}
    atomic_json(RUN/'results/regression_expansion_latest_summary.json',summary)
    flat=[]
    for r in records:
        row={'backbone':r['backbone'],'method':r['label'],'role':r['role'],'complete':r['complete']}
        for track,name in [('five','five'),('seven_calibrated_v2','seven')]:
            v=r['tracks'][track];row[name+'_datasets']=v['datasets']
            for block in ['clean','shift']:
                for k in ['r2','nrmse']:row[name+'_'+block+'_'+k]=v[block][k] if v[block] else None
        flat.append(row)
    with (RUN/'results/regression_expansion_latest_averages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('STATUS',summary['status'],'ENSEMBLE_ROWS',summary['main_ensemble_rows'],flush=True)
    for r in records:
        def fmt(t):
            v=r['tracks'][t];m=v['shift']
            return ('—' if not m else f"{m['r2']:.4f}/{m['nrmse']:.4f}")+f" ({v['datasets']}/{v['expected_datasets']})"
        print(r['backbone'],r['label'],'five',fmt('five'),'seven',fmt('seven_calibrated_v2'),
              'CleanOK',[r['tracks'][t].get('Clean_within_005') for t in TRACKS],flush=True)


if __name__=='__main__':main()
