"""Frozen-policy diagnostics for confidence repair and one-branch regression stress.

No fitting, benchmark modification, backbone inference, or outcome-based setting
selection. Inputs are explicitly labelled appendix-rule reconstructions.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'rebuttal/ensemble_comparison'
OUT=ROOT/'rebuttal/gate_sensitivity_replay/results'
THRESHOLDS=[.8,.9,.95,.99]
MAGNITUDES=[1.,2.,5.,10.,20.,50.,100.]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_paths(backbone,track,task):
    return sorted((RUN/'results/combiners'/track/task).glob(f'fold_*_tabace_{backbone}_reimplemented.json'))


def prepare():
    cases=[]
    for backbone in ['tabicl','tabpfn']:
        for track in ['five','seven_reimplemented']:
            for task in ['classification','regression']:
                paths=policy_paths(backbone,track,task);assert len(paths)==5
                ids=set();policies=[]
                for path in paths:
                    f=json.loads(path.read_text());train=set(f['train_data_ids']);test=set(f['test_data_ids'])
                    assert f['status']=='COMPLETE' and not train.intersection(test)
                    assert not ids.intersection(test);ids.update(test)
                    assert f['held_out_labels_used_for_fit'] is False
                    policies.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),fold=f['fold']))
                bankroot=RUN/'results/reference'/backbone/track/task
                banks=sorted(bankroot.glob('*/*.npz'));assert len(banks)==len(ids)*(21 if track=='five' else 29)
                cases.append(dict(backbone=backbone,track=track,task=task,datasets=len(ids),
                                  episodes=len(banks),bank_root=str(bankroot.relative_to(ROOT)),policies=policies))
    plan=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,cases=cases,
        confidence_thresholds=THRESHOLDS,regression_stress_source_sd_multipliers=MAGNITUDES,
        regression_stress='Replace exactly one alternative by original alternative median plus/minus k*source_std; cycle all five branches and both signs',
        stress_selection='Worst absolute output displacement across the ten branch/sign choices per query; no target labels used to select injection',
        regression_methods=['Full','Mean6','Median6','Full_clip10sd'],
        clipped_variant='Additional development safeguard: clip alternative predictions to source_mean +/-10*source_std before fusion; Base unchanged',
        classification_methods=['Full','No_entropy_gate'],
        classification_ablation='No_entropy_gate sets Base U=1 with all learned coefficients fixed; no refitting',
        parameters_frozen=True,training_on_evaluation_labels=False,backbone_calls=0,
        input_scope='Cached appendix-rule reconstruction, not recovered original paper prediction archive',
        current_benchmark_settings_unchanged=True,all_results_retained=True,markdown_written=False)
    path=OUT/'frozen_plan.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old['cases']==cases and old['confidence_thresholds']==THRESHOLDS
    else:write_json(path,plan)
    print('FROZEN_PLAN',[(c['backbone'],c['track'],c['task'],c['datasets'],c['episodes']) for c in cases],flush=True)


def softmax(z,axis):
    z=z-z.max(axis=axis,keepdims=True);p=np.exp(z);return p/p.sum(axis=axis,keepdims=True)


def classify(base,bank,theta,disable_entropy=False,disagreement_override=False):
    classes=base.shape[1]
    clipped=np.clip(bank,1e-12,1)
    q=(1+(clipped*np.log(clipped)).sum(2)/np.log(classes)).T
    agreement=(1-.5*np.abs(bank-bank.mean(0)[None]).sum(2)).T
    b=np.clip(base,1e-12,1)
    u=-(b*np.log(b)).sum(1)/np.log(classes)
    weights=softmax(theta[:5]+q*theta[5:10]+agreement*theta[10:15],1)
    aggregate=np.einsum('nm,mnc->nc',weights,bank)
    support=(weights*q*agreement).sum(1)
    gate=np.ones(len(u)) if disable_entropy else u
    if disagreement_override:
        disagreement=np.clip(.5*np.abs(base-aggregate).sum(1),0.,1.)
        gate=np.maximum(u,disagreement)
    alpha=theta[-1]*gate*support
    output=(1-alpha[:,None])*base+alpha[:,None]*aggregate
    output=np.clip(output,1e-12,None);output/=output.sum(1,keepdims=True)
    assert np.all(alpha>=-1e-7) and np.all(alpha<=1+1e-7)
    return output,alpha


def regress(base,bank,theta,source_mean,source_scale,clip=False):
    if clip:
        width=10. if clip is True else float(clip)
        bank=np.clip(bank,source_mean-width*source_scale,source_mean+width*source_scale)
    members=(bank-source_mean)/source_scale;b=(base-source_mean)/source_scale
    tau=np.exp(theta[-2]);median=np.median(members,axis=0)
    agreement=np.exp(-np.abs(members-median[None])/tau)
    weights=softmax(theta[:5,None]+agreement*theta[5:10,None],0)
    aggregate=(weights*members).sum(0)
    support=(weights*agreement).sum(0)
    alpha=theta[-1]*(1-np.exp(-np.abs(aggregate-b)/tau))*support
    result=((1-alpha)*b+alpha*aggregate)*source_scale+source_mean
    assert np.isfinite(result).all() and np.all(alpha>=-1e-9) and np.all(alpha<=1+1e-9)
    low=np.minimum(base,bank.min(0));high=np.maximum(base,bank.max(0))
    assert np.all(result>=low-1e-7*np.maximum(1,np.abs(low)))
    assert np.all(result<=high+1e-7*np.maximum(1,np.abs(high)))
    return result,weights,alpha


def cells(backbone,track,task):
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    case=next(c for c in plan['cases'] if (c['backbone'],c['track'],c['task'])==(backbone,track,task))
    lookup={};row_lookup={}
    for record in case['policies']:
        path=ROOT/record['path'];assert sha(path)==record['sha256']
        f=json.loads(path.read_text());theta=np.asarray(f['parameters'],dtype=float)
        for data_id in f['test_data_ids']:lookup[int(data_id)]=(theta,int(f['fold']))
        for row in f['rows']:row_lookup[(int(row['data_id']),row['family'],int(row['rho']))]=row
    for i,path in enumerate(sorted((ROOT/case['bank_root']).glob('*/*.npz'))):
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            base=z['base'];bank=z['members'];y=z['y_target'];ss=float(z['source_scale']);sm=float(z['source_mean'])
        assert bank.shape[0]==5 and ss>0 and np.isfinite(base).all() and np.isfinite(bank).all()
        theta,fold=lookup[int(meta['data_id'])]
        reference=row_lookup[(int(meta['data_id']),meta['family'],int(meta['rho']))]
        yield i,meta,base,bank,y,sm,ss,theta,fold,reference,path


def confidence(backbone,track):
    rows=[];audit=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'classification'):
        original,_=classify(b,k,t)
        error=abs(float(np.mean(original.argmax(1)==y))-ref['accuracy']);assert error<1e-12
        confidence=b.max(1);before=b.argmax(1);correct=before==y
        for method in ['Full','No_entropy_gate']:
            p,alpha=classify(b,k,t,disable_entropy=method=='No_entropy_gate')
            after=p.argmax(1);right=after==y
            for threshold in THRESHOLDS:
                selected=confidence>=threshold;wrong=selected&~correct;good=selected&correct
                fixed=wrong&right;harmed=good&~right
                rows.append(dict(backbone=backbone,track=track,data_id=m['data_id'],task_id=m['task_id'],
                    dataset=m['dataset'],family=m['family'],rho=m['rho'],outer_fold=fold,method=method,
                    confidence_threshold=threshold,n_queries=len(y),n_high_confidence=int(selected.sum()),
                    base_errors=int(wrong.sum()),corrected_errors=int(fixed.sum()),
                    base_correct=int(good.sum()),new_errors=int(harmed.sum()),
                    changed_predictions=int((selected&(before!=after)).sum()),
                    alpha_sum_high_confidence=float(alpha[selected].sum()),
                    max_corrected_base_confidence=float(confidence[fixed].max()) if fixed.any() else None))
        audit.append(dict(data_id=m['data_id'],family=m['family'],rho=m['rho'],replayed_accuracy_difference=error,
                          bank_source=str(path.relative_to(ROOT)),source_context_rows=m['source_rows'],
                          reference_status=m.get('reference_status',m.get('protocol_status'))))
        if i%200==0:print('CONFIDENCE',backbone,track,i,flush=True)
    dest=OUT/backbone/track/'classification';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'condition_counts.csv',index=False)
    write_json(dest/'audit.json',dict(cells= len(audit),all_frozen_policy_metric_replays_exact=True,
        input_scope='Appendix-rule reconstructed banks',episodes=audit,backbone_calls=0))
    print('CONFIDENCE_COMPLETE',backbone,track,len(audit),flush=True)


def regression(backbone,track):
    metrics=[];stress=[];samples=defaultdict(list);audits=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        original,_,_=regress(b,k,t,sm,ss)
        eval_scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        nrmse=float(np.sqrt(np.mean((original-y)**2))/eval_scale)
        assert abs(nrmse-ref['nrmse'])<1e-8*max(1,nrmse)
        fullset=np.r_[b[None],k]
        baseline=dict(Full=original,Mean6=fullset.mean(0),Median6=np.median(fullset,axis=0),
                      Full_clip10sd=regress(b,k,t,sm,ss,True)[0])
        for method,p in baseline.items():
            metrics.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method=method,n=len(y),nrmse=float(np.sqrt(np.mean((p-y)**2))/eval_scale),
                r2=1-float(np.sum((p-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                changed_from_full=int(np.count_nonzero(np.abs(p-original)>1e-10*ss))))
        # Uniform alternative replacement: the same hypothetical values are
        # presented to every aggregation method. Outcome labels select nothing.
        median=np.median(k,axis=0)
        for magnitude in MAGNITUDES:
            worst={name:np.full(len(y),-1.) for name in baseline}
            selected_error={name:np.empty(len(y)) for name in baseline}
            selected_weight=np.zeros(len(y));selected_alpha=np.zeros(len(y))
            for branch in range(5):
                for sign in [-1.,1.]:
                    modified=k.copy();modified[branch]=median+sign*magnitude*ss
                    full,w,alpha=regress(b,modified,t,sm,ss)
                    six=np.r_[b[None],modified]
                    outputs=dict(Full=full,Mean6=six.mean(0),Median6=np.median(six,axis=0),
                                 Full_clip10sd=regress(b,modified,t,sm,ss,True)[0])
                    for name,p in outputs.items():
                        displacement=np.abs(p-baseline[name])/ss;select=displacement>worst[name]
                        worst[name][select]=displacement[select]
                        selected_error[name][select]=(p[select]-y[select])/eval_scale
                        if name=='Full':
                            selected_weight[select]=w[branch,select];selected_alpha[select]=alpha[select]
            for name,delta in worst.items():
                samples[(magnitude,name)].append(delta)
                stress.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                    magnitude_source_sd=magnitude,method=name,n=len(y),
                    mean_displacement_source_sd=float(delta.mean()),p95_displacement_source_sd=float(np.quantile(delta,.95)),
                    max_displacement_source_sd=float(delta.max()),
                    worst_displacement_selected_nrmse=float(np.sqrt(np.mean(selected_error[name]**2))),
                    mean_attacked_branch_weight=float(selected_weight.mean()) if name=='Full' else None,
                    mean_alpha=float(selected_alpha.mean()) if name=='Full' else None))
        audits.append(dict(data_id=m['data_id'],family=m['family'],rho=m['rho'],
            replayed_nrmse_difference=abs(nrmse-ref['nrmse']),source_context_rows=m['source_rows'],
            source_scale=ss,clip_changed_alternative_values=int(np.count_nonzero(k!=np.clip(k,sm-10*ss,sm+10*ss))),
            alternative_values=k.size,reference_status=m.get('reference_status',m.get('protocol_status')),
            bank_source=str(path.relative_to(ROOT))))
        if i%100==0:print('REGRESSION_STRESS',backbone,track,i,flush=True)
    dest=OUT/backbone/track/'regression';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(dest/'unmodified_performance.csv',index=False)
    pd.DataFrame(stress).to_csv(dest/'stress_by_condition.csv',index=False)
    pooled=[]
    for (magnitude,name),values in samples.items():
        delta=np.concatenate(values)
        pooled.append(dict(backbone=backbone,track=track,magnitude_source_sd=magnitude,method=name,n_query_conditions=len(delta),
            mean_displacement_source_sd=float(delta.mean()),median_displacement_source_sd=float(np.median(delta)),
            p95_displacement_source_sd=float(np.quantile(delta,.95)),p99_displacement_source_sd=float(np.quantile(delta,.99)),
            max_displacement_source_sd=float(delta.max())))
    pd.DataFrame(pooled).to_csv(dest/'stress_pooled.csv',index=False)
    write_json(dest/'audit.json',dict(cells=len(audits),all_frozen_policy_metric_replays_verified=True,
        source_only_fixed_clipping_rule=True,stress_choices_do_not_use_target_labels=True,
        input_scope='Appendix-rule reconstructed banks',episodes=audits,backbone_calls=0))
    print('REGRESSION_STRESS_COMPLETE',backbone,track,len(audits),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--stage',choices=['classification','regression'])
    p.add_argument('--backbone',choices=['tabicl','tabpfn']);p.add_argument('--track',choices=['five','seven_reimplemented'])
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='classification':confidence(a.backbone,a.track)
    elif a.stage=='regression':regression(a.backbone,a.track)
    else:p.error('Specify --prepare or a stage/backbone/track')
