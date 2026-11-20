"""Context scaling on frozen v22 DEVELOPMENT tasks; never calibrates to scores.

Original 256-row context and fixed queries are retained. Additional distinct
source rows come from official TRAIN, excluding the frozen query rows. Regression
Clean/five remain explicitly semisynthetic, using the frozen v22 teacher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from prepare_data import RUN, LOCAL_CC18, atomic_json, protocol
import benchmark256_v22 as frozen

OUT = RUN/'results/tabicl_context_scaling_v23'
BUDGETS = [1024, 4096, 8192]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank(task, block, meta, family='original', rho=0):
    p = frozen.OUT/'input_banks'/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
    with np.load(p) as z:
        data = {k: z[k] for k in ['X_source','y_source','X_target','y_target']}
    return data, json.loads(p.with_suffix('.json').read_text())


def npwrite(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def extra_rows(eligible, used, query, task_id):
    remaining = np.setdiff1d(eligible, np.r_[used, query])
    return np.random.default_rng(np.random.SeedSequence([0, int(task_id), 2300])).permutation(remaining)


def classification_raw(meta):
    folder = LOCAL_CC18/str(meta['task_id'])
    frame = pd.read_parquet(folder/'X.parquet')
    labels = pd.read_parquet(folder/'y.parquet').iloc[:,0]
    valid = ~labels.isna().to_numpy()
    original = np.flatnonzero(valid)
    X, _ = protocol._numeric_matrix(frame.loc[valid])
    y = LabelEncoder().fit_transform(labels.loc[valid])
    with np.load(folder/'official_folds.npz') as z:
        eligible = np.intersect1d(z['train_0'], original)
    return X, y, original, eligible


def prepare(task, track, meta):
    """track=five covers Clean and five; track=seven has separate source/query."""
    path = OUT/'source_pools'/task/track/f'{meta["task_id"]}.npz'
    if path.exists():
        with np.load(path) as z: return {k:z[k] for k in z.files}
    tid = int(meta['task_id'])
    if task == 'classification':
        X, y, original, eligible = classification_raw(meta)
        pos = np.full(int(original.max())+1, -1, dtype=int)
        pos[original] = np.arange(len(original))
        if track == 'five':
            with np.load(RUN/'cache/five/classification'/f'{tid}.npz') as z:
                first, query = z['source_indices'], z['target_indices']
                assert np.array_equal(X[pos[first]], z['X_source'])
                assert np.array_equal(y[pos[first]], z['y_source'])
                assert np.array_equal(X[pos[query]], z['X_target'])
            rows = np.r_[first, extra_rows(eligible, first, query, tid)][:8192]
            arrays = dict(X=X[pos[rows]], y=y[pos[rows]], source_ids=rows, query_ids=query)
        else:
            ids = protocol._balanced_cap_indices(y,448,0)
            source, query_pos = train_test_split(ids,train_size=320,test_size=128,stratify=y[ids],random_state=0)
            hp, rp = train_test_split(source,train_size=256,test_size=64,stratify=y[source],random_state=0)
            query = original[query_pos]
            extras = extra_rows(eligible,original[source],query,tid)
            available = min(8192,len(source)+len(extras))
            h_count = available-available//4
            hids = np.r_[original[hp],extras[:h_count-256]]
            rids = np.r_[original[rp],extras[h_count-256:h_count-256+available//4-64]]
            arrays = dict(H=X[pos[hids]],hy=y[pos[hids]],R=X[pos[rids]],ry=y[pos[rids]],
                          H0=X[hp],hy0=y[hp],R0=X[rp],ry0=y[rp],
                          h_ids=hids,r_ids=rids,query_ids=query)
    else:
        from internal_regression_benchmark import raw_fold
        X,y,source,_,original = raw_fold(meta)
        pos = np.full(int(original.max())+1,-1,dtype=int);pos[original]=np.arange(len(original))
        eligible = original[source]
        if track == 'five':
            import tabicl_semisynthetic256_v22 as semi
            g = semi.prepare(meta)
            teacher = json.loads((semi.OUT/'generators'/f'{tid}.json').read_text())
            cfg = frozen.configuration()['regression_clean']
            sigma = cfg['mu']*np.exp(cfg['h']*float(g['dataset_noise_draw']))
            old_y = g['signal_source']+sigma*g['epsilon_source']
            first = g['context_indices']
            ordering = np.r_[first,np.setdiff1d(np.arange(len(old_y)),first)]
            old_ids = g['source_row_ids'][ordering]
            extras = extra_rows(eligible,g['source_row_ids'],g['query_row_ids'],tid)
            needed = max(0,8192-len(old_ids));extras=extras[:needed]
            A = X[pos[extras]].astype(np.float64)
            precision = RUN/'results/internal_regression_component_v1/clean_candidates/source11_target7_precision0.03'/f'{tid}.npz'
            with np.load(precision) as z:
                cols,center,scale = (z[k] for k in ['feature_precision_columns','feature_precision_center','feature_precision_scale'])
            A[:,cols]=center[cols]+np.round((A[:,cols]-center[cols])/scale[cols]/.03)*.03*scale[cols]
            # Same float32 measurement grid as the frozen teacher's input.
            A=A.astype(np.float32).astype(np.float64)
            Z=np.clip((A-np.asarray(teacher['original_feature_center']))/np.asarray(teacher['original_feature_scale']),-5,5)
            signal=(Z[:,teacher['teacher_active_original_columns']]@np.asarray(teacher['teacher_coefficients'])+teacher['teacher_intercept']-teacher['signal_center'])/teacher['signal_scale']
            noise=np.random.default_rng(np.random.SeedSequence([0,tid,2301])).normal(size=len(extras))
            rows=np.r_[old_ids,extras][:8192]
            arrays=dict(X=np.vstack([g['X_source_pool'][ordering],Z[:,teacher['selected_original_columns']].astype(np.float32)])[:8192],
                        y=np.r_[old_y[ordering],signal+sigma*noise][:8192],source_ids=rows,query_ids=g['query_row_ids'])
        else:
            from seven_shifts import regression_clean
            clean=regression_clean(meta)
            with np.load(RUN/'cache/five/regression'/f'{tid}.npz') as z:
                oldids=z['source_indices'];queryids=z['target_indices'];oldX=z['X_source'];oldy=z['y_source']
                rng=np.random.default_rng(2718)
                context=rng.permutation(len(oldids))[:min(len(oldids),1024)][:256]
                qids=rng.choice(len(queryids),min(256,len(queryids)),replace=False)
            ordering=np.r_[context,np.setdiff1d(np.arange(len(oldids)),context)]
            extra=extra_rows(eligible,oldids,queryids[qids],tid)[:max(0,8192-len(oldids))]
            arrays=dict(X=np.vstack([oldX[ordering],X[pos[extra]]])[:8192],y=np.r_[oldy[ordering],y[pos[extra]]][:8192],
                        source_ids=np.r_[oldids[ordering],extra][:8192],query_ids=queryids[qids],
                        H0=clean[0],hy0=clean[1],R0=clean[2],ry0=clean[3],X0=clean[4],y0=clean[5])
    ids=np.r_[arrays['h_ids'],arrays['r_ids']] if 'h_ids' in arrays else arrays['source_ids']
    assert len(np.unique(ids))==len(ids)
    assert not np.intersect1d(ids,arrays['query_ids']).size
    assert all(np.isfinite(v).all() for v in arrays.values())
    npwrite(path,**arrays)
    atomic_json(path.with_suffix('.json'),dict(task=task,track=track,task_id=tid,dataset=meta['dataset'],
                available_rows=len(ids),distinct_source_rows=len(np.unique(ids)),query_overlap=0,
                budgets={str(b):min(b,len(ids)) for b in BUDGETS},source_pool_sha256=sha(path),
                semisynthetic=task=='regression' and track=='five',
                expansion='original canonical source retained, then distinct official fold0 TRAIN rows excluding frozen queries'))
    return arrays


def class_seven(pool, family, rho, budget):
    from sklearn.linear_model import LogisticRegression
    from seven_shifts import standardize
    p=frozen.configuration()['classification_seven']['generator_parameters'];s=rho/100.
    n=min(budget,len(pool['hy'])+len(pool['ry']));nr=n//4;nh=n-nr
    H=pool['H'][:max(256,nh)].copy();hy=pool['hy'][:max(256,nh)].copy()
    R=pool['R'][:nr].copy();ry=pool['ry'][:nr].copy()
    H0,R0=pool['H0'],pool['R0'];reference=np.vstack([H0,R0]);j=int(np.argmax(reference.var(0)))
    def biased(score):
        logits=np.clip(p['x_weight_exponent']*s*score,-15,15);prob=np.exp(logits-logits.max());prob/=prob.sum()
        return np.random.default_rng(0).choice(len(score),size=len(score),replace=True,p=prob)
    if family in ['covariate','selection']:
        if family=='selection':
            learner=LogisticRegression(max_iter=500,random_state=0).fit(standardize(reference,reference),np.r_[pool['hy0'],pool['ry0']])
            majority=int(np.mean(np.r_[pool['hy0'],pool['ry0']])>=.5)
        for A,b in [(H,hy),(R,ry)]:
            score=standardize(A,reference)[:,j] if family=='covariate' else learner.predict_proba(standardize(A,reference))[:,majority]
            ids=biased(score);A[:],b[:]=A[ids].copy(),b[ids].copy()
    elif family in ['concept','temporal']:
        feature=int(np.argmax(H0.var(0)));ids=np.argsort(np.abs(R[:,feature]-np.median(H0[:,feature])),kind='stable')
        ids=ids[:int(np.ceil(p['max_boundary_flip_fraction']*s*len(ry)))];ry[ids]=1-ry[ids]
        if family=='temporal':
            mean=H0.mean(0);sd=np.maximum(H0.std(0),1e-6)
            R=mean+(R-mean)*(1+p['affine_scale_increment']*s)+p['affine_translation_sd']*s*sd
    return np.vstack([H[:nh],R]).astype(np.float32),np.r_[hy[:nh],ry]


def reg_seven(pool,family,rho,budget,Q):
    # Freeze severity statistics to the original v22 256-row environment.
    from calibrated_seven_protocol import amplitude
    from seven_shifts import standardize,biased_rows,affine,PARAMETERS
    X=pool['X'][:budget].copy();y=pool['y'][:budget].copy();s=rho/100.
    ref=np.vstack([pool['H0'],pool['R0'],pool['X0']]);j=int(np.argmax(pool['X0'].var(0)))
    z=standardize(X,ref)
    multiplier=frozen.configuration()['regression_seven']['family_amplitude_multipliers'][family]
    with amplitude(multiplier):
        if family=='covariate':
            affected=np.random.default_rng(2718).permutation(len(y))[:int(np.ceil(s*len(y)))]
            ids=biased_rows(z[:,j],len(y),1.,2718);X[affected],y[affected]=X[ids[affected]],y[ids[affected]]
        elif family=='label':
            score=(y-pool['y0'].mean())/max(float(pool['y0'].std()),1e-6)
            ids=biased_rows(score,len(y),s,2718);X,y=X[ids],y[ids]
        elif family in ['concept','temporal']:
            y=y+PARAMETERS['regression_response_drift_sd']*s*pool['y0'].std()*np.tanh(z[:,j])
            if family=='temporal':X=affine(X,ref,s)
        elif family in ['selection','support']:
            distance=np.mean(((X-Q.mean(0))/np.maximum(ref.std(0),1e-6))**2,axis=1)
            olddistance=np.mean(((pool['X0']-Q.mean(0))/np.maximum(ref.std(0),1e-6))**2,axis=1)
            score=distance/max(float(olddistance.std()),1e-6)
            ids=biased_rows(score if family=='selection' else -score,len(y),s,2718);X,y=X[ids],y[ids]
        elif family=='domain':X=affine(X,ref,s)
    return X.astype(np.float32),y


def materialize(task,block,meta,family,rho,budget,pool):
    data,info=bank(task,block,meta,family,rho)
    if block=='seven':
        if task=='classification':X,y=class_seven(pool,family,rho,budget)
        else:X,y=reg_seven(pool,family,rho,budget,data['X_target'])
        data.update(X_source=X,y_source=y)
    else:
        data.update(X_source=pool['X'][:budget],y_source=pool['y'][:budget])
        if task=='regression' and block=='five':
            from tabicl_semisynthetic_five_v22 import corrupt_data
            data=corrupt_data(data,family,rho,frozen.configuration()['regression_five']['families'][family]['parameters'])
    return data,info


def audit(meta,task,track,pool):
    from seven_shifts import FAMILIES
    blocks=['clean','five'] if track=='five' else ['seven']
    rows=[]
    for block in blocks:
        grid=[('original',0)] if block=='clean' else [(f,r) for f in (protocol.FAMILIES if block=='five' else FAMILIES) for r in [20,50,80,100]]
        for family,rho in grid:
            old,_=bank(task,block,meta,family,rho)
            new,_=materialize(task,block,meta,family,rho,256,pool)
            for key in old:
                assert np.array_equal(new[key],old[key]),(task,track,meta['task_id'],family,rho,key,float(np.max(np.abs(new[key]-old[key]))))
            rows.append([family,rho])
    atomic_json(OUT/'source_audits'/task/track/f'{meta["task_id"]}.json',dict(status='PASS',conditions=len(rows),canonical256_exact=True))


class Runner:
    def __init__(self,task,cache=False):
        from run_tabicl import Predictor
        self.predictor=Predictor(task,'cuda:0');self.task=task;self.cache=cache;self.key=None

    def __call__(self,data,n_classes):
        from run_tabicl import normalize,CHECKPOINTS
        from tabicl import TabICLClassifier,TabICLRegressor
        X,y,Q=(data[k] for k in ['X_source','y_source','X_target'])
        if not self.cache:return self.predictor(X,y,Q,members=1,random_state=0,n_classes=n_classes)[0]
        if X.shape[1]==0 or np.all(np.ptp(X.astype(np.float64),axis=0)==0) or len(np.unique(y))==1:
            return self.predictor(X,y,Q,members=1,random_state=0,n_classes=n_classes)[0]
        key=(protocol.array_sha256(X),protocol.array_sha256(y))
        if not hasattr(self,'model'):
            cls=TabICLClassifier if self.task=='classification' else TabICLRegressor
            self.model=cls(n_estimators=1,device='cuda:0',model_path=str(CHECKPOINTS/f'tabicl-{ "classifier" if self.task=="classification" else "regressor"}-v2-20260212.ckpt'),
                           allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0,kv_cache='repr')
        if key!=self.key:
            self.model.fit(X.astype(np.float32),y)
            self.model._load_model=lambda:None
            self.key=key
        if self.task=='classification':
            pred=self.model.predict_proba(Q.astype(np.float32));out=np.zeros((len(Q),n_classes))
            out[:,self.model.classes_.astype(int)]=pred
            return normalize(out)
        return np.asarray(self.model.predict(Q.astype(np.float32)),dtype=np.float64)


def plan():
    path=OUT/'plan.json'
    if path.exists():return
    atomic_json(path,dict(status='FROZEN_BEFORE_CONTEXT_SCALING_PREDICTIONS',budgets=BUDGETS,seed=0,n_estimators=1,
        frozen_v22_sha256=sha(frozen.CONFIG),development_benchmark=True,original_paper_replication=False,
        semisynthetic_regression_blocks=['clean','five'],query_arrays_and_metric_denominators='unchanged v22 exported arrays',
        context_expansion='nested distinct original source + official fold0 TRAIN excluding queries; cap when insufficient; no padding',
        seven_classification_history_reference_ratio='3:1; original reference statistics and source-only selection model frozen',
        seven_regression_severity_statistics='frozen original256 reference, feature choice, label scale and distance scale',
        perturbation_strengths=[.2,.5,.8,1.],new_benchmark_calibration=False,finetuning=False,
        Markdown_written=False,script_sha256=sha(__file__)))


def worker(shard,shards,only=None,cache=False):
    import warnings,torch
    from run_tabicl import metrics
    warnings.filterwarnings('ignore');torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    jobs=[]
    for task in ['classification','regression']:
        for track in ['five','seven']:
            for meta in frozen.cohort(task,track):jobs.append((task,track,meta))
    # Requested image datasets first; block conditions remain grouped for cache reuse.
    jobs.sort(key=lambda j:(int(j[2]['task_id']) not in [167124,167121],j[0],j[1],int(j[2]['task_id'])))
    if only:jobs=[j for j in jobs if str(j[2]['task_id']) in only.split(',')]
    runners={}
    for task,track,meta in jobs[shard::shards]:
        try:
            pool=prepare(task,track,meta);audit(meta,task,track,pool)
            if task not in runners:runners[task]=Runner(task,cache)
            runner=runners[task];nc=int(max(pool['hy'].max(),pool['ry'].max())+1) if 'hy' in pool else (int(pool['y'].max()+1) if task=='classification' else None)
            blocks=['clean','five'] if track=='five' else ['seven']
            for budget in BUDGETS:
                for block in blocks:
                    from seven_shifts import FAMILIES
                    grid=[('original',0)] if block=='clean' else [(f,r) for f in (protocol.FAMILIES if block=='five' else FAMILIES) for r in [20,50,80,100]]
                    for family,rho in grid:
                        out=OUT/'predictions'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                        if out.exists() and out.with_suffix('.json').exists():continue
                        data,info=materialize(task,block,meta,family,rho,budget,pool)
                        hashes={k:protocol.array_sha256(v) for k,v in data.items()}
                        actual=len(data['y_source']);start=time.monotonic()
                        # Identical capped inputs may reuse their actual inference, with hashes audited.
                        reused=None
                        for smaller in BUDGETS:
                            prior=OUT/'predictions'/str(smaller)/task/block/str(meta['task_id'])/out.name
                            if smaller>=budget:break
                            if prior.with_suffix('.json').exists():
                                rec=json.loads(prior.with_suffix('.json').read_text())
                                if rec['input_hashes']==hashes:reused=prior;break
                        if reused:
                            with np.load(reused) as z:pred=z['prediction']
                        else:pred=runner(data,nc)
                        assert np.isfinite(pred).all()
                        npwrite(out,prediction=pred,y_target=data['y_target'])
                        atomic_json(out.with_suffix('.json'),dict(task=task,block=block,task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                            requested_context=budget,actual_context=actual,seed=0,n_estimators=1,scale=info['scale'],
                            input_hashes=hashes,metrics=metrics(data['y_target'],pred,task,info['scale']),elapsed_seconds=time.monotonic()-start,
                            reused_exact_input_from=str(reused) if reused else None,kv_cache='repr' if cache else False,
                            semisynthetic=task=='regression' and block in ['clean','five'],development_only=True,original_paper_replication=False))
                    print('BLOCK_COMPLETE',task,meta['dataset'],budget,block,flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{task}_{track}_{meta["task_id"]}.json',dict(task=task,track=track,task_id=meta['task_id'],error=error))
            # A corrupt pool or an execution error is explicit, never silently omitted in reporting.
            if task in runners:del runners[task]
            import gc;gc.collect();torch.cuda.empty_cache()


def report():
    detail=[]
    for p in (OUT/'predictions').glob('*/*/*/*/*.json'):
        if p.name.endswith('.tmp.json'):continue
        r=json.loads(p.read_text());detail.append({k:v for k,v in r.items() if k not in ['input_hashes','metrics']}|r['metrics'])
    if not detail:return
    df=pd.DataFrame(detail)
    for key in ['accuracy','auc','r2','nrmse']:
        if key not in df:df[key]=np.nan
    df.to_csv(OUT/'detail.csv',index=False)
    per=df.groupby(['requested_context','task','block','task_id','dataset'],as_index=False).agg(
        actual_context=('actual_context','first'),conditions=('family','size'),
        **{k:(k,'mean') for k in ['accuracy','auc','r2','nrmse']})
    per['expected_conditions']=per['block'].map(dict(clean=1,five=20,seven=28))
    per['complete']=per['conditions']==per['expected_conditions'];per.to_csv(OUT/'per_dataset.csv',index=False)
    avg=per[per['complete']].groupby(['requested_context','task','block'],as_index=False).agg(
        completed_datasets=('task_id','size'),full_budget_datasets=('actual_context',lambda x:0),
        min_context=('actual_context','min'),max_context=('actual_context','max'),**{k:(k,'mean') for k in ['accuracy','auc','r2','nrmse']})
    # Groupby aggregation cannot see the context key; calculate attainment explicitly.
    for i,r in avg.iterrows():
        v=per[(per.requested_context==r.requested_context)&(per.task==r.task)&(per.block==r.block)&per.complete]
        avg.loc[i,'full_budget_datasets']=int((v.actual_context==r.requested_context).sum())
        avg.loc[i,'expected_datasets']=len(frozen.cohort(r.task,r.block))
    avg.to_csv(OUT/'averages.csv',index=False)
    special=per[per.task_id.isin([167124,167121])];special.to_csv(OUT/'requested_image_datasets.csv',index=False)
    print(avg.to_string(index=False));print('SPECIAL_DATASETS');print(special.to_string(index=False))
    expected=sum(len(grid) for _,_,_,grid in frozen.jobs())*3
    atomic_json(OUT/'progress.json',dict(completed_conditions=len(df),expected_conditions=expected,
        errors=len(list((OUT/'errors').glob('*.json'))),complete=len(df)==expected,markdown_bytes=(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--only');p.add_argument('--cache',action='store_true')
    a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards,a.only,a.cache)
    else:report()
