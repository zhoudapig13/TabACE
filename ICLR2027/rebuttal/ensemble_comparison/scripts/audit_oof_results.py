"""Reconstruct every held-out ensemble and check folds and recorded metrics."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from fit_combiners import fold_mapping
from verify_raw_results import independent_metrics


@lru_cache(maxsize=128)
def raw(path):
    path=Path(path)
    with np.load(path) as f:return {k:f[k] for k in f.files if k!='metadata_json'}


def normalized(p):
    p=np.maximum(p,1e-12);return p/p.sum(-1,keepdims=True)


def reconstructed(j,row,kind):
    track,task,task_id,family,rho=[row[k] for k in ['track','task','task_id','family','rho']]
    name=f'{family}_{rho}.npz';backbone=j.get('backbone','tabicl')
    if kind in ['shared','specific']:
        root='reference' if kind=='shared' else 'reference_specific'
        d=raw(str(RUN/'results'/root/backbone/track/task/str(task_id)/name))
        bank,b=d['members'],d['base'];params=np.asarray(j['parameters']);mu=float(d['source_mean']);scale=float(d['source_scale'])
        if task=='classification':
            K=b.shape[1];a=np.clip(bank,1e-12,1)
            q=(1+(a*np.log(a)).sum(2)/np.log(K)).T
            s=(1-.5*np.abs(bank-bank.mean(0)[None]).sum(2)).T
            base=np.clip(b,1e-12,1);u=-(base*np.log(base)).sum(1)/np.log(K)
            z=params[:5]+q*params[5:10]+s*params[10:15];w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
            alpha=params[-1]*u*(w*q*s).sum(1)
            pred=normalized((1-alpha[:,None])*b+alpha[:,None]*np.einsum('nm,mnc->nc',w,bank))
        else:
            members=(bank-mu)/scale;base=(b-mu)/scale;tau=np.exp(params[-2])
            agreement=np.exp(-np.abs(members-np.median(members,axis=0)[None]).T/tau)
            z=params[:5]+agreement*params[5:10];w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
            aggregate=(w*members.T).sum(1)
            alpha=params[-1]*(1-np.exp(-np.abs(aggregate-base)/tau))*(w*agreement).sum(1)
            pred=((1-alpha)*base+alpha*aggregate)*scale+mu
        saved=RUN/'results'/('reference_predictions' if kind=='shared' else 'reference_specific_predictions')/backbone/track/task/str(task_id)/name
        if not np.allclose(raw(str(saved))['prediction'],pred,rtol=1e-10,atol=1e-10):raise AssertionError('saved calibrated prediction mismatch')
        return d['y_target'],pred
    if kind=='feature':
        d=raw(str(RUN/'results/feature_views/tabicl'/track/task/str(task_id)/name));weights=np.asarray(j['weights'])
        if task=='classification':
            p=normalized(np.einsum('m,mnc->nc',weights,d['bank']));z=np.log(p)/j['temperature'];pred=normalized(np.exp(z-z.max(1,keepdims=True)))
        else:pred=weights@(d['bank']-float(d['source_mean']))+float(d['source_mean'])
        return d['y_target'],pred
    if kind=='dense':
        d=raw(str(RUN/'results/dense_full_context/tabicl'/track/task/str(task_id)/name))
        return d['y_target'],d['candidates'][j['rules'].index(j['selected_rule'])]
    d=raw(str(RUN/'results/tabicl'/track/task/str(task_id)/name));bank=d['bootstrap_bank']
    if row['method'].startswith('GES'):
        return d['y_target'],np.einsum('m,m...->...',np.asarray(j['ges_weights']),bank)
    if kind=='convex':return d['y_target'],np.einsum('m,m...->...',np.asarray(j['stack_weights']),bank)
    beta=np.asarray(j['stack']['beta'])
    if task=='classification':
        z=np.einsum('m,mnc->nc',beta,np.log(np.clip(bank,1e-12,1)));pred=normalized(np.exp(z-z.max(1,keepdims=True)))
    else:
        mu=float(d['source_y_mean']);sigma=float(d['source_y_std'])
        pred=(((bank-mu)/sigma).T@beta+j['stack']['intercept'])*sigma+mu
    return d['y_target'],pred


def main():
    failures=[];checked=0;files=0;row_counts=Counter();seen=set();coverage=[]
    paths=sorted((RUN/'results/combiners').rglob('fold_*.json'))
    for path in paths:
        j=json.loads(path.read_text());files+=1
        try:
            assert j['status']=='COMPLETE';fold=j['fold'];sample=j['rows'][0];track=sample['track'];task=sample['task']
            train={int(x) for x in j['train_data_ids']};test={int(x) for x in j['test_data_ids']}
            assert train.isdisjoint(test)
            mapping,_=fold_mapping(task,track)
            expected_ids={int(m['data_id']) for p in (RUN/'cache/five'/task).glob('*.json') for m in [json.loads(p.read_text())]} if track.startswith('five') else set(mapping)
            assert train|test==expected_ids
            assert test=={x for x in expected_ids if mapping[x]==fold}
            assert train=={x for x in expected_ids if mapping[x]!=fold}
            kind=('specific' if '_tabace_specific_' in path.name else 'shared' if '_tabace_' in path.name else
                  'feature' if '_feature_nnls' in path.name else 'dense' if '_dense_full_context' in path.name else
                  'convex' if '_convex' in path.name else 'old_stack')
            if kind in ['shared','specific']:assert j['optimization']['optimizer_success']
            for row in j['rows']:
                assert int(row['data_id']) in test and row['outer_fold']==fold
                key=(row.get('backbone',j.get('backbone','tabicl')),row['track'],row['task'],row['method'],row['data_id'],row['family'],row['rho'])
                assert key not in seen;seen.add(key)
                y,pred=reconstructed(j,row,kind)
                assert np.isfinite(pred).all()
                scale=max(float(y.std()),1e-12) if task=='regression' and track=='seven_reimplemented' else row['scale']
                for metric,value in independent_metrics(y,pred,task,scale).items():
                    if value is None:assert row[metric] is None
                    else:assert np.isclose(value,row[metric],rtol=3e-8,atol=2e-9),(metric,row[metric],value)
                row_counts[key[:4]]+=1;checked+=1
        except Exception as exc:failures.append({'file':str(path.relative_to(RUN)),'error':repr(exc)})
        if files%25==0:print('OOF_AUDITED',files,'rows',checked,'failures',len(failures),flush=True)
    for track,tasks in [('five',['classification','regression']),('five_corrected',['classification']),('seven_reimplemented',['classification','regression'])]:
        for task in tasks:
            n=(72 if task=='classification' else 35) if track.startswith('five') else (35 if task=='classification' else 33)
            conditions=21 if track.startswith('five') else 29
            methods=[('tabicl',m) for m in ['GES_25','Stacking_OOF','GES_25_metric_normalized','Stacking_convex_OOF','TabFM_feature_views6_NNLS_adapted']]
            if task=='classification':methods.append(('tabicl','DENSE_fullcontext_OOF_selected_adapted'))
            for b in ['tabicl','tabpfn']:methods += [(b,'TabACE_Shared_R5_reimplemented'),(b,'TabACE_Specific_R5_reimplemented')]
            for b,m in methods:
                expected=n*((25 if track.startswith('five') else 35) if 'Specific' in m else conditions)
                actual=row_counts[(b,track,task,m)]
                coverage.append({'backbone':b,'track':track,'task':task,'method':m,'completed':actual,'expected':expected,'complete':actual==expected})
    result={'status':'FAIL' if failures else 'PASS' if all(x['complete'] for x in coverage) else 'IN_PROGRESS',
        'files_checked':files,'metric_rows_reconstructed':checked,'fold_partitions_checked':'all, compared against original dataset fold mapping',
        'held_out_target_labels_used_for_fit':False,'failures':failures,'coverage':coverage}
    atomic_json(RUN/'results/oof_result_audit.json',result)
    print({k:v for k,v in result.items() if k not in ['failures','coverage']},flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
