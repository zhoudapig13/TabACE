"""Independent all-row metric, probability, cohort-hash and coverage audit."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from prepare_data import RUN,atomic_json,protocol

ROOTS=['tabicl','tabpfn','tabm','reference/tabicl','reference/tabpfn',
       'reference_specific/tabicl','reference_specific/tabpfn',
       'feature_views/tabicl','dense_full_context/tabicl','oligo_extended/tabicl']
EXPECTED={'five':{'classification':1512,'regression':735},
          'five_corrected':{'classification':1512},
          'seven_reimplemented':{'classification':1015,'regression':957}}


def independent_metrics(y,pred,task,scale):
    if task=='regression':
        p=np.asarray(pred,dtype=np.float64).reshape(-1);error=p-y
        mse=float(np.dot(error,error)/len(y));denom=max(float(np.dot(y-y.mean(),y-y.mean())),1e-12)
        return {'r2':1-float(np.dot(error,error))/denom,'nrmse':np.sqrt(mse)/scale,'nmae':float(np.abs(error).mean())/scale}
    p=np.clip(np.asarray(pred,dtype=np.float64),1e-12,None);p/=p.sum(1,keepdims=True)
    labels=y.astype(int);ranks=rankdata(p,axis=0,method='average')
    positive=labels[:,None]==np.arange(p.shape[1])[None];n=positive.sum(0);negative=len(y)-n;valid=(n>0)&(negative>0)
    auc=(np.sum(ranks*positive,axis=0)-n*(n+1)/2)/np.maximum(n*negative,1)
    return {'accuracy':float(np.mean(p.argmax(1)==labels)),'auc':float(auc[valid].mean()) if valid.any() else None}


def check_one(filename):
    path=Path(filename);m=json.loads(path.read_text());issues=[];checked=0
    if m.get('status')!='COMPLETE':return str(path),0,['non-complete metadata'],None
    if m.get('backbone')=='tabm' and m.get('configuration',{}).get('preprocessing_revision')!=2:
        issues.append('TabM preprocessing revision 2 is required')
    cohort=None
    try:
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as raw:
            y=np.asarray(raw['y_target']);assert y.ndim==1 and np.isfinite(y).all()
            cohort={'key':(m['track'],m['task'],m['task_id'],m['family'] if m['rho'] else 'clean',m['rho']),
                    'y_hash':protocol.array_sha256(y),'source_context_hash':m.get('X_context_sha256')}
            if m.get('y_target_sha256') and protocol.array_sha256(y)!=m['y_target_sha256']:issues.append('target cohort hash mismatch')
            scale=(max(float(y.std()),1e-12) if m['task']=='regression' and m['track'].startswith('seven') else m['scale'])
            for name,record in m.get('methods',{}).items():
                p=raw['prediction__'+name]
                if p.shape[0]!=len(y):issues.append(name+': target row count mismatch');continue
                if not np.isfinite(p).all():issues.append(name+': nonfinite prediction');continue
                if m['task']=='classification':
                    if p.ndim!=2 or p.shape[1]<=int(y.max()):issues.append(name+': invalid class columns');continue
                    if np.any(p<0) or not np.allclose(p.sum(1),1.,atol=1e-7):issues.append(name+': invalid probability simplex')
                values=independent_metrics(y,p,m['task'],scale)
                for metric,expected in values.items():
                    if metric not in record:continue
                    actual=record[metric]
                    if expected is None:
                        if actual is not None:issues.append(name+': invalid single-class AUC')
                    elif actual is None or not np.isclose(actual,expected,rtol=2e-8,atol=2e-9):
                        issues.append(f'{name}: {metric} mismatch recorded={actual}, independent={expected}')
                checked+=1
            if 'oligo_extended' in str(path) and m.get('candidate_pool_size',0)>=400:
                bank=raw['bank'];iqr=raw['iqr'];assert bank.shape[0]>=400 and iqr.shape==bank.shape
                for total in [40,400]:
                    chosen=np.argsort(iqr[:total].mean(1),kind='stable')[:total//10]
                    if chosen.tolist()!=m['selected_candidates'][str(total)]:issues.append('IQR selection mismatch')
                    name=f'prediction__OligoICP_IQR{total}_keep{total//10}_adapted'
                    if not np.allclose(raw[name],bank[chosen].mean(0),atol=1e-10,rtol=1e-10):issues.append('selected IQR ensemble prediction mismatch')
    except Exception as exc:issues.append(repr(exc))
    return str(path.relative_to(RUN)),checked,issues,cohort


def coverage():
    rows=[]
    for root in ROOTS:
        for track,tasks in EXPECTED.items():
            for task,number in tasks.items():
                if root.startswith('reference_specific'):
                    number=(72 if task=='classification' else 35)*25 if track.startswith('five') else (35 if task=='classification' else 33)*35
                if root.startswith('dense_full_context') and task=='regression':continue
                if root.startswith('oligo_extended') and task=='classification':continue
                files=[p for p in (RUN/'results'/root/track/task).rglob('*.json') if '.tmp' not in p.name]
                valid=0
                if root.startswith('oligo_extended'):
                    valid=sum(json.loads(p.read_text()).get('candidate_pool_size',0)>=400 for p in files)
                else:valid=len(files)
                rows.append({'root':root,'track':track,'task':task,'completed':valid,'expected':number,'complete':valid==number})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    paths=[str(p) for root in ROOTS for p in (RUN/'results'/root).rglob('*.json') if '.tmp' not in p.name]
    failures=[];checked=0;cohorts=defaultdict(set);source_contexts=defaultdict(set)
    with ProcessPoolExecutor(args.workers) as pool:
        for i,(path,n,issues,cohort) in enumerate(pool.map(check_one,paths,chunksize=32)):
            checked+=n
            if issues:failures.append({'file':path,'issues':issues})
            if cohort:
                key=tuple(cohort['key']);cohorts[key].add(cohort['y_hash'])
                if cohort['source_context_hash']:source_contexts[key].add(cohort['source_context_hash'])
            if (i+1)%2000==0:print('VERIFIED_FILES',i+1,'metric_records',checked,'failed_files',len(failures),flush=True)
    for name,groups in [('target',cohorts),('source_context',source_contexts)]:
        for key,hashes in groups.items():
            if len(hashes)>1:failures.append({'condition':key,'issues':[name+' differs across methods/backbones']})
    covered=coverage();complete=all(r['complete'] for r in covered)
    result={'status':'PASS' if complete and not failures else 'IN_PROGRESS' if not complete and not failures else 'FAIL',
        'files_checked':len(paths),'metric_records_checked':checked,'independent_auc':'average-rank Mann-Whitney statistic, macro OVR',
        'coverage':covered,'failures':failures,'all_required_raw_runs_complete':complete,
        'cross_method_target_cohorts_checked':len(cohorts),'corrected_source_context_cohorts_checked':len(source_contexts),
        'original_main_table_bit_exact_restoration':False,'old_numeric_baseline_matching_used_for_selection':False}
    atomic_json(RUN/'results/raw_result_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['coverage','failures']}),flush=True)
    if failures or (args.require_complete and not complete):raise SystemExit(1)


if __name__=='__main__':main()
