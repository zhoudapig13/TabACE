"""Metric-aligned, equal-grid ABCD refits; no ordering-based model selection.

Classification keeps the v99 target64 supervision and fixed v98 banks. CE
fits are selected by calibration Accuracy. Regression keeps v87 pools/banks;
normalized MSE fits are selected by other-dataset validation R2. Every A-D
uses the same candidate families and search budget. Old evaluated cohorts
are development data; all pairwise contrasts, including negative ones, stay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import softmax

import target_supervised_adaptation_v99 as cls
import abcd_concept_calibrated_v87 as reg
import abcd_mechanism_v84 as origin
import tabace_fusion_v28 as fusion
from prepare_data import RUN,atomic_json
from run_tabicl import metrics,normalize
from report_abcd_v84 import paired_statistics

OUT=RUN/'results/abcd_metric_aligned_v100'
REGS=(.0001,.001,.01,.1)
METHODS=[('A',True,True),('B',False,True),('C',True,False),('D',False,False)]
CONTRASTS=[('A','B'),('A','C'),('B','D'),('C','D')]


def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    plan={'seed':0,'stage':'bounded metric-aligned adaptation attempt1',
        'classification':'v97 inputs, target64, fixed v98 banks; same nested32/32 calibration',
        'classification_loss':'cross-entropy; prior-centered regularization',
        'classification_selection':'validation Accuracy, then validation negative log loss; never evaluated cohort',
        'regression':'fixed v87 banks, five dataset outer folds and an inner heldout dataset fold',
        'regression_loss':'calibration-response-variance-normalized squared error; cross-dataset calibration only',
        'regression_selection':'mean dataset validation R2',
        'same_grid_and_candidate_families_for_A_B_C_D':True,'regularizations':REGS,
        'extra_candidates':['identity Base','single branch','old prior policy','static convex stack'],
        'no_ordering_or_between_method_margin_in_fitting_or_selection':True,
        'pairwise_contrasts':CONTRASTS,'Holm_family':8,
        'C_D_keep_all_features':True,'B_D_exclude_equivalent_mechanisms':True,
        'old_query_cohorts_are_development_only':True,'original_paper_replication':False,
        'no_backbone_or_benchmark_change':True,'context_cap':256,'alternative_branches':5,
        'bounded_followup':'at most one further calibration-selected constructor-availability refinement before fresh confirmation',
        'Markdown_written':False,'code_sha256':origin.sha(__file__)}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def predict(c,p):
    if p['kind']=='member':return c['bank'][p['slot']]
    if p['kind']=='convex_stack':
        b=np.concatenate([c['base'][None],c['bank']],axis=0)
        out=np.tensordot(np.asarray(p['weights']),b,axes=(0,0))
        return normalize(out) if c['task']=='classification' else out
    return cls.predict(c,p) if c['task']=='classification' else fusion.predict(c,p)


def score(cells,p):
    grouped={}
    for c in cells:
        q=predict(c,p);y=c['y'];v=metrics(y,q,c['task'],max(float(y.std()),1e-12))
        key='accuracy' if c['task']=='classification' else 'r2'
        nll=-np.log(np.clip(q[np.arange(len(y)),y.astype(int)],1e-12,1.)).mean() if key=='accuracy' else -v[key]
        grouped.setdefault(c['meta']['data_id'],[]).append([v[key],-nll])
    values=np.array([np.mean(v,axis=0) for v in grouped.values()])
    return {'primary':float(values[:,0].mean()),'secondary':float(values[:,1].mean())}


def fit(cells,task,regularization,prior,kind):
    fusion.GATE_VARIANT='entropy_or_disagreement'
    # Calibration labels may normalize the training loss. Query/evaluation
    # spread never enters deployment or regularization selection.
    if task=='regression':
        for c in cells:c['meta']['scale']=max(float(c['y'].std()),1e-6)
    data=fusion.training_tensors(cells,5)
    if kind=='convex_stack':
        initial=np.r_[2.,np.zeros(5)];bounds=[(-8,8)]*6
        if task=='classification':
            stack=torch.cat([data['true_base'][:,None],data['true_members']],dim=1)
        else:stack=torch.cat([data['B'][:,None],data['M']],dim=1)
    else:
        initial=np.asarray(prior).copy()
        bounds=[(-8,8)]*5+([(-10,10)]*10+[(0,1)] if task=='classification' else [(-10,10)]*5+[(-4,4),(0,1)])
    def objective(values):
        p=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        if kind=='convex_stack':
            w=torch.softmax(p,dim=0);q=(stack*w).sum(1)
            penalty=(w-torch.tensor([1.,0,0,0,0,0],dtype=torch.float64)).square().mean()
        elif task=='classification':
            w=torch.softmax(p[:5]+data['Q']*p[5:10]+data['S']*p[10:15],dim=1)
            mixed=(w*data['true_members']).sum(1)
            alpha=p[-1]*data['U']*(w*data['S']).sum(1)
            q=(1-alpha)*data['true_base']+alpha*mixed
            penalty=(p-torch.as_tensor(prior)).square().mean()
        else:
            q=fusion.mapping(p,data,task,5,1.)
            penalty=(p-torch.as_tensor(prior)).square().mean()
        raw=-q.clamp(1e-12,1.).log() if task=='classification' else (q-data['Y']).square()
        loss=(raw*data['weights']).sum()+regularization*penalty
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    result=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=bounds,
                    options={'maxiter':240,'ftol':1e-9,'gtol':1e-6})
    assert np.isfinite(result.x).all() and np.isfinite(result.fun)
    model={'kind':'convex_stack' if kind=='convex_stack' else ('agreement_gate_adapted' if task=='classification' else 'gated_fusion'),
        'regularization':regularization,'parameters':result.x.tolist(),'training_objective':float(result.fun),
        'optimizer_success':bool(result.success),'prior_parameters':np.asarray(prior).tolist()}
    if kind=='convex_stack':model['weights']=softmax(result.x).tolist()
    elif task=='regression':model.update(m=5,gain=1.,gate_variant='entropy_or_disagreement')
    return model


def select(training,validation,refit,task,prior,old):
    trials=[]
    candidates=[('Base',{'kind':'base'}),('Prior',old)]
    candidates += [(f'Member_{slot}',{'kind':'member','slot':slot}) for slot in range(5)]
    for name,p in candidates:trials.append({'candidate':name,'model':p,'validation':score(validation,p)})
    for kind in ['gated','convex_stack']:
        for regularization in REGS:
            p=fit(training,task,regularization,prior,kind)
            trials.append({'candidate':kind,'regularization':regularization,'model':p,'validation':score(validation,p)})
    best=max(trials,key=lambda x:(x['validation']['primary'],x['validation']['secondary']))
    final=fit(refit,task,best['regularization'],prior,best['candidate']) if best['candidate'] in ('gated','convex_stack') else best['model']
    return final,best,trials


def class_worker(task_id):
    torch.set_num_threads(1);torch.manual_seed(0)
    inp=cls.banks.split.OUT/'inputs'/f'{task_id}.npz';meta=json.loads(inp.with_suffix('.json').read_text())
    with np.load(inp) as z:data={k:z[k] for k in z.files}
    data['data_id']=int(meta['data_id']);folder=OUT/'classification'/str(task_id)
    if (folder/'completion.json').exists():return {'task':'classification','id':task_id,'cached':True}
    for method,seen,matched in METHODS:
        train=cls.cells_for(task_id,'adapt',matched,seen,data)
        val=cls.cells_for(task_id,'validation',matched,seen,data)
        full=cls.cells_for(task_id,'refit',matched,seen,data)
        prior,origin_prior=cls.source_prior(meta,method)
        p,best,trials=select(train,val,full,'classification',prior,{'kind':'agreement_gate_adapted','parameters':prior.tolist()})
        atomic_json(folder/f'{method}_policy.json',{'policy':p,'trials':trials,'selected_candidate':best['candidate'],
            'selected_regularization':best.get('regularization'),'seen':seen,'matched':matched,
            'allowed_families':sorted({c['meta']['family'] for c in train}),
            'source_prior':origin_prior,'calibration_target_labels':64,'fresh_evaluation_labels_used':False})
    atomic_json(folder/'completion.json',{'status':'COMPLETE','all_four_policies_frozen':True})
    return {'task':'classification','id':task_id}


def regression_cells(matched):
    cells=origin.load('regression',matched)
    for c in cells:
        m=c['meta']
        if m['rho'] and m['family'] in ('concept','temporal'):
            path=reg.OUT/'banks'/str(m['task_id'])/f'{m["family"]}_{m["rho"]}.npz'
            with np.load(path) as z:
                assert np.array_equal(c['y'],z['y_target'])
                c['base'],c['bank']=z['base'].copy(),z['members'].copy()
                c['mu'],c['sigma']=float(z['source_mean']),float(z['source_scale'])
                if matched:c['bank'][0]=z['repair']
        c['meta']['scale']=max(float(c['y'].std()),1e-6)
    return cells


def reg_worker(fold):
    torch.set_num_threads(1);torch.manual_seed(0)
    folder=OUT/'regression'/str(fold)
    if (folder/'completion.json').exists():return {'task':'regression','id':fold,'cached':True}
    cells={m:regression_cells(m) for m in (True,False)}
    for method,seen,matched in METHODS:
        source=[c for c in cells[matched] if c['fold']!=fold and (seen or not origin.exposed(c,'regression'))]
        inner=min(c['fold'] for c in source)
        train=[c for c in source if c['fold']!=inner];val=[c for c in source if c['fold']==inner]
        oldfile=reg.OUT/'folds/regression'/str(fold)/f'{method}_policy.json'
        old=json.loads(oldfile.read_text())['policy']
        # Reusing a full-outer-train fit as an inner-validation candidate would
        # leak the inner labels. Initialize from a fixed neutral gate instead.
        prior=np.r_[np.zeros(5),np.ones(5),0.,.5]
        neutral={'kind':'gated_fusion','parameters':prior.tolist(),'m':5,'gain':1.,'gate_variant':'entropy_or_disagreement'}
        p,best,trials=select(train,val,source,'regression',prior,neutral)
        atomic_json(folder/f'{method}_policy.json',{'policy':p,'trials':trials,'selected_candidate':best['candidate'],
            'selected_regularization':best.get('regularization'),'seen':seen,'matched':matched,'inner_fold':inner,
            'training_data_ids':sorted({c['meta']['data_id'] for c in source}),
            'excluded_data_ids':sorted({c['meta']['data_id'] for c in cells[matched] if c['fold']==fold}),
            'calibration_families':sorted({c['meta']['family'] for c in source}),
            'heldout_query_labels_used':False})
    atomic_json(folder/'completion.json',{'status':'COMPLETE','all_four_policies_frozen':True})
    return {'task':'regression','id':fold}


def launch():
    freeze();jobs=[('classification',int(p.stem)) for p in (cls.banks.split.OUT/'inputs').glob('*.npz')]
    jobs += [('regression',i) for i in range(5)];records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(class_worker if task=='classification' else reg_worker,i) for task,i in jobs]
        for f in as_completed(futures):
            r=f.result();records.append(r);print(r,flush=True)
            atomic_json(OUT/'execution.json',{'status':'RUNNING','jobs':records})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','jobs':records,'classification_datasets':72,'regression_folds':5})


def report():
    assert json.loads((OUT/'execution.json').read_text())['status']=='COMPLETE'
    assert json.loads((OUT/'plan.json').read_text())['code_sha256']==origin.sha(__file__)
    # Seal predictions/policies before reading the old development evaluation labels.
    files=list((OUT/'classification').glob('*/*_policy.json'))+list((OUT/'regression').glob('*/*_policy.json'))
    seal={str(p.relative_to(OUT)):origin.sha(p) for p in files};assert len(seal)==308
    if not (OUT/'policy_seal.json').exists():atomic_json(OUT/'policy_seal.json',seal)
    else:assert json.loads((OUT/'policy_seal.json').read_text())==seal
    rows=[]
    for inp in (cls.banks.split.OUT/'inputs').glob('*.npz'):
        tid=int(inp.stem);meta=json.loads(inp.with_suffix('.json').read_text())
        with np.load(inp) as z:data={k:z[k] for k in z.files}
        data['data_id']=meta['data_id']
        truth=cls.banks.split.OUT/'evaluation_truth'/f'{tid}.npz'
        with np.load(truth) as z:y=z['y_evaluation']
        for method,seen,matched in METHODS:
            policy=json.loads((OUT/'classification'/str(tid)/f'{method}_policy.json').read_text())['policy']
            for c in cls.cells_for(tid,'evaluation',matched,seen,data):
                predictions={method:predict(c,policy)}
                if method=='A':predictions.update(Base=c['base'],Base_zero=c['zero'],Target_only=c['bank'][1],Single_matched=c['bank'][3])
                for name,q in predictions.items():
                    out=OUT/'development_predictions/classification'/str(tid)/f'{c["meta"]["rho"]}_{name}.npz'
                    out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,prediction=q)
                    rows.append({'task':'classification','task_id':tid,'data_id':meta['data_id'],'rho':c['meta']['rho'],
                        'method':name,**metrics(y,q,'classification',1.)})
    for matched in (True,False):
        for c in regression_cells(matched):
            m=c['meta']
            if not (m['track']=='seven' and (m['rho']==0 or m['family']=='concept')):continue
            for method,seen,match in METHODS:
                if match!=matched:continue
                policy=json.loads((OUT/'regression'/str(c['fold'])/f'{method}_policy.json').read_text())['policy']
                predictions={method:predict(c,policy)}
                if method=='A':predictions.update(Base=c['base'],Single_matched=c['bank'][0])
                for name,q in predictions.items():
                    out=OUT/'development_predictions/regression'/str(m['task_id'])/f'{m["rho"]}_{name}.npz'
                    out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,prediction=q)
                    rows.append({'task':'regression','task_id':m['task_id'],'data_id':m['data_id'],'rho':m['rho'],
                        'method':name,**metrics(c['y'],q,'regression',max(float(c['y'].std()),1e-12))})
    detail=pd.DataFrame(rows);detail.to_csv(OUT/'development_detail.csv',index=False)
    tables=[];tests=[]
    for task,group in detail.groupby('task'):
        first,second=('accuracy','auc') if task=='classification' else ('r2','nrmse')
        values=group[group.rho>0].groupby(['data_id','method'])[[first,second]].mean()
        clean=group[group.rho==0].set_index(['data_id','method'])[[first,second]]
        base=values.xs('Base',level='method')
        for method in group.method.unique():
            v=values.xs(method,level='method');cl=clean.xs(method,level='method')
            tables.append({'task':task,'method':method,'datasets':len(v),first:v[first].mean(),second:v[second].mean(),
                'clean_primary':cl[first].mean(),'gain_vs_Base':(v[first]-base[first]).mean()})
        for a,b in CONTRASTS:
            delta=(values.xs(a,level='method')[first]-values.xs(b,level='method')[first]).dropna()
            tests.append({'task':task,'contrast':a+'-'+b,'datasets':len(delta),**paired_statistics(delta)})
    table=pd.DataFrame(tables);stats=pd.DataFrame(tests);prev=0.
    for rank,i in enumerate(sorted(stats.index,key=lambda j:stats.loc[j,'p_two_sided'])):
        prev=max(prev,min(1.,(8-rank)*stats.loc[i,'p_two_sided']));stats.loc[i,'p_holm']=prev
    stats['positive_significant']=(stats.gain_ci_low>0)&(stats.p_holm<.05)
    table.to_csv(OUT/'development_tables.csv',index=False);stats.to_csv(OUT/'development_contrasts.csv',index=False)
    atomic_json(OUT/'development_status.json',{'status':'COMPLETE','old_evaluated_cohorts':True,
        'all_requested_contrast_means_positive':bool((stats.gain>0).all()),
        'all_requested_contrasts_positive_significant':bool(stats.positive_significant.all()),
        'ordering_used_for_training_or_selection':False,'Markdown_bytes':(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(table.to_string(index=False),flush=True);print(stats.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','report']);a=p.parse_args()
    launch() if a.phase=='launch' else report()
