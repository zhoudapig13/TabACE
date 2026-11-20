"""Audit the one active endpoint, including cross-fit member axes and selection."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import heavy_source_retention_v338 as run
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import metrics


def main(partial=False):
    source=run.source;plan=json.loads((run.OUT/'plan.json').read_text())
    for module,key in [(run,'script_sha256'),(run.retention,'retention_sha256'),(source.novel,'generator_sha256'),
                       (source.constructors,'registry_sha256'),(source.fusion,'fusion_sha256')]:
        assert source.sha(module.__file__)==plan[key]
    cases=run.endpoint.endpoint_cases();assert len(cases)==64
    rows=[];count=0;contexts=0;policy_rows=[]
    for c in cases:
        tid=c['task_id'];assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:X0,y0,Q0,num,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert set(apos).isdisjoint(epos) and set(sid).isdisjoint(qid);folds=np.arange(len(apos))%4
        cp=json.loads((run.endpoint.oldbank.training.OUT/'checkpoints'/'perturbation'/'classification'/str(tid)/'result.json').read_text())
        assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json';gate=json.loads(pfile.read_text());assert c['data_id'] not in gate['training_data_ids']
        for rho in source.novel.STRENGTHS:
            path=run.OUT/'predictions'/str(tid)/f'cell_heavy_tail_{rho}.npz'
            if not path.with_suffix('.json').exists():
                assert partial,(tid,rho,'missing')
                continue
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['input_sha256']==c['input_sha256'] and meta['adaptation_sha256']==c['adaptation_sha256'] and meta['checkpoint_sha256']==cp['checkpoint_sha256']
            assert meta['adaptation_positions']==apos.tolist() and meta['evaluation_positions']==epos.tolist() and meta['adaptation_folds']==folds.tolist()
            assert meta['policy_sha256']==source.sha(pfile) and meta['actual_GPU'] in range(6) and not meta['evaluation_labels_opened']
            X,y,Q,_=source.novel.generate(X0,y0,Q0,num,'classification','cell_heavy_tail',rho,tid)
            views=[('canonical',X,y,Q)]+source.constructors.shared_contexts(X,y,Q,len(y),'classification',None)
            order=source.constructors.specific.importance(X,y.astype(float));columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/path.name
            oldpath=run.endpoint.oldbank.OUT/'predictions'/'perturbation'/'classification'/str(tid)/path.name
            assert source.sha(original)==meta['original_prediction_sha256'] and source.sha(oldpath)==meta['old_bank_sha256']
            with np.load(path) as p,np.load(original) as original_np,np.load(oldpath) as old:
                np.testing.assert_array_equal(p['Base'],original_np['Base'][epos])
                sets=meta['prediction_sets'];expected=[]
                def check_set(mode,fold,requested,train_positions,train_labels,wanted):
                    nonlocal contexts
                    matching=[r for r in sets if r['mode']==mode and r['fold']==fold and r['requested_source_rows']==requested]
                    assert len(matching)==1
                    r=matching[0];expected.append((mode,fold,requested))
                    assert r['training_positions']==train_positions.tolist() and r['query_positions']==wanted.tolist()
                    assert set(train_positions).isdisjoint(wanted) and set(qid[train_positions]).isdisjoint(qid[wanted])
                    assert len(r['contexts'])==6
                    for (name,A,b,Tall),saved in zip(views,r['contexts']):
                        T=Tall[wanted]
                        V,labels,record=run.retention.context(name,A,b,T,Q[train_positions],train_labels,requested,columns)
                        assert name==saved['name'] and len(labels)<=256 and record['retained_candidate_positions']==saved['retained_candidate_positions']
                        assert record['target_rows']==saved['target_rows']==len(train_positions) and record['source_rows']==saved['source_rows']
                        if requested==0:assert record['source_rows']==0
                        else:np.testing.assert_array_equal(np.unique(labels[:record['source_rows']]),np.unique(b))
                        for value,key in [(V,'X_sha256'),(labels,'y_sha256'),(T,'query_X_sha256')]:assert protocol.array_sha256(value)==saved[key]
                        np.testing.assert_array_equal(labels[-len(train_labels):],train_labels)
                        if name=='canonical':mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                        contexts+=1
                    np.testing.assert_allclose([mu,sigma],[r['source_mean'],r['source_scale']],atol=1e-12,rtol=0)
                    if mode=='crossfit':
                        j=list(run.retention.COUNTS).index(requested);va=np.flatnonzero(folds==fold)
                        native=p['cv_native'][j,va];members=p['cv_members'][j][:,va];shared=p['cv_shared'][j,va]
                    else:native,members,shared=p[f'final_native_{requested}'],p[f'final_members_{requested}'],p[f'final_shared_{requested}']
                    replay=source.fusion.predict({'task':'classification','base':native,'bank':members,'mu':mu,'sigma':sigma},gate['Shared'])
                    np.testing.assert_allclose(replay,shared,atol=1e-12,rtol=0)
                for requested in run.retention.COUNTS:
                    for fold in range(4):
                        tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                        check_set('crossfit',fold,requested,apos[tr],ay[tr],apos[va])
                for name,key in [('Native','cv_native'),('Shared','cv_shared')]:
                    candidates={'warm_source':old[f'{name}_finetuned'][apos]}|{str(n):p[key][j] for j,n in enumerate(run.retention.COUNTS)}
                    policy=run.retention.select(ay,original_np['Base'][apos],candidates)
                    saved=meta[f'{name}_policy'];assert policy['candidate']==saved['candidate'] and policy['alpha']==saved['alpha']
                    np.testing.assert_allclose(policy['score'],saved['score'],atol=1e-12,rtol=0)
                chosen=sorted({int(meta[f'{name}_policy']['candidate']) for name in ['Native','Shared'] if meta[f'{name}_policy']['candidate'] not in ['Base','warm_source']})
                for requested in chosen:check_set('final',None,requested,apos,ay,epos)
                assert len(expected)==len(sets)
                def final_candidate(policy,native=False):
                    if policy['candidate']=='Base':return original_np['Base'][epos]
                    if policy['candidate']=='warm_source':return old['Native_finetuned' if native else 'Shared_finetuned'][epos]
                    return p[f'final_{"native" if native else "shared"}_{policy["candidate"]}']
                for name in ['Native','Shared']:
                    policy=meta[f'{name}_policy'];result=run.retention.apply(policy,original_np['Base'][epos],final_candidate(policy,name=='Native'))
                    np.testing.assert_allclose(result,p[f'{name}_adapted'],atol=1e-12,rtol=0)
                policy=meta['Shared_policy'];matched=run.retention.apply(policy,original_np['Base'][epos],final_candidate(policy,True))
                np.testing.assert_allclose(matched,p['Native_matched_Shared_policy'],atol=1e-12,rtol=0)
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as truth_np:truth=truth_np['y_target'];np.testing.assert_array_equal(truth_np['target_indices'],qid[epos])
                for name in ['Base','Native_adapted','Shared_adapted','Native_matched_Shared_policy']:
                    assert np.isfinite(p[name]).all() and len(p[name])==len(truth)
                    rows.append({'task_id':tid,'data_id':c['data_id'],'rho':rho,'method':name,**metrics(truth,p[name],'classification',1.)})
                policy_rows.append({'task_id':tid,'rho':rho,'Native_candidate':meta['Native_policy']['candidate'],'Native_alpha':meta['Native_policy']['alpha'],
                    'Shared_candidate':meta['Shared_policy']['candidate'],'Shared_alpha':meta['Shared_policy']['alpha']})
            count+=1
        print('HEAVY_RETENTION_AUDIT_CASE',tid,flush=True)
    assert count>0
    if not partial:assert count==256
    prefix='partial_' if count<256 else ''
    df=pd.DataFrame(rows);df.to_csv(run.OUT/f'{prefix}condition_metrics.csv',index=False)
    valid=df.groupby(['data_id','method']).rho.nunique().reset_index(name='strengths');valid=valid[valid.strengths==4][['data_id','method']]
    ds=df.merge(valid,on=['data_id','method']).groupby(['data_id','method'])[['accuracy','auc']].mean().reset_index()
    table=ds.groupby('method')[['accuracy','auc']].mean().reset_index();table['datasets']=ds.data_id.nunique()
    table.to_csv(run.OUT/f'{prefix}family_averages.csv',index=False);pd.DataFrame(policy_rows).to_csv(run.OUT/f'{prefix}selected_policies.csv',index=False)
    get=lambda name:float(table.loc[table.method==name,'accuracy'].iloc[0])
    gain=get('Shared_adapted')-get('Base')
    atomic_json(run.OUT/f'{prefix}numerical_acceptance.json',{'active_endpoint':list(run.endpoint.ENDPOINT),'conditions':count,
        'datasets':int(table.datasets.iloc[0]),'Base':get('Base'),'Shared_adapted':get('Shared_adapted'),'gain':gain,
        'gain_vs_Native_same_Shared_policy':get('Shared_adapted')-get('Native_matched_Shared_policy'),
        'required_gain_strictly_greater_than':.05,'active_endpoint_pass':count==256 and table.datasets.iloc[0]==64 and gain>.05,
        'all_twelve_pass':False,'extra_source_and_test_subset_supervision':True})
    assert run.frozen_check.verify_frozen()==plan['frozen_registry_sha256']
    registry=json.loads((RUN/'results/frozen_endpoints_v330/registry.json').read_text())
    previous=RUN/'results/task_guarded_adaptation_v318'
    assert source.sha(previous/'numerical_acceptance.json')==registry['source_acceptance_sha256']
    assert source.sha(previous/'independent_audit.json')==registry['source_audit_sha256']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(run.OUT/f'{prefix}independent_audit.json',{'status':'PASS','complete':count==256,'conditions':count,'contexts_reconstructed':contexts,
        'source_and_adaptation_and_evaluation_row_ids_checked':True,'crossfit_heldout_labels_not_in_contexts':True,
        'every_Shared_gate_output_replayed':True,'source_retention_and_blend_policies_reselected_from_adaptation_labels_only':True,
        'Base_bitwise_preserved_on_fixed_evaluation_rows':True,'all_three_frozen_endpoint_predictions_and_metadata_and_acceptance_files_unchanged':True,
        'only_first_unresolved_endpoint_run':True,'Markdown_bytes':0,'script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print('GAIN',gain,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
