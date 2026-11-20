"""Reference-pool Shared roles with cross-fitted test-subset context labels.

The original evaluation subset stays fixed. All six TFM contexts use the same
permitted pool and target-label budget; Native is a supervision-matched control.
No held-out evaluation label is opened by this program.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
from tabicl import TabICLClassifier,TabICLRegressor
import task_adaptation_bank_v312 as oldbank
import reference_context_geometry_v321 as geometry
import crossfit_context_selector_v322 as selector
from shared_finetuned_inference_v309 import reserve_target
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import Predictor

source=oldbank.source
OUT=RUN/'results/reference_target_contexts_v323'
PILOT_POSITIONS=[0,30,72,90,107,120,142,160]


def prepare():
    assert json.loads((oldbank.split.OUT/'audit.json').read_text())['status']=='PASS'
    assert json.loads((oldbank.training.OUT/'independent_training_audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((oldbank.split.OUT/'case_manifest.json').read_text())
    policy=json.loads(oldbank.training.implementation.GPU_POLICY.read_text())
    assert policy['allowed_GPUs']==list(range(6)) and policy['reserved_GPUs']==[6,7]
    pilot_count=sum(1+4*len(cases[i]['families']) for i in PILOT_POSITIONS)
    plan={'seed':0,'cases':175,'conditions':2175,'pilot_case_positions':PILOT_POSITIONS,'pilot_conditions':pilot_count,
        'context_cap':256,'members':'canonical plus unchanged original five v28 Shared constructors',
        'membership':'four unlabeled query clusters; nearest pool rows with class coverage, no source proportion quota',
        'permitted_pool':'observed source256 plus existing disjoint clean labeled source references',
        'extra_clean_source_labels_used_for_contexts_and_finetuned_weights':True,
        'target_labels':'existing v311 fixed quarter query, cap64, excluded from evaluation',
        'context_target_labels':'all adaptation labels at deployment; only three folds of them when predicting fourth-fold validation',
        'adaptation_validation':'four fixed label-independent folds',
        'per_dataset_family_strength_parameters':'convex weights on frozen Base, source-finetuned method, reference+target adapted method',
        'same_label_and_reference_budget_for_Native_and_Shared':True,'TFM_calls_not_compute_matched_to_Base':True,
        'existing_evaluation_positions_preserved':True,'benchmark_generator_unchanged':True,
        'final_evaluation_labels_used_for_selection':False,'not_original_zero_shot_Shared':True,
        'previously_scored_development_cohort':True,'GPU_policy':policy,
        'manifest_sha256':source.sha(oldbank.split.OUT/'case_manifest.json'),
        'source_training_audit_sha256':source.sha(oldbank.training.OUT/'independent_training_audit.json'),
        'script_sha256':source.sha(__file__),'geometry_sha256':source.sha(geometry.__file__),
        'selector_sha256':source.sha(selector.__file__),'generator_sha256':source.sha(source.novel.__file__),
        'registry_sha256':source.sha(source.constructors.__file__),'fusion_sha256':source.sha(source.fusion.__file__),
        'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def run_case(c,gpu):
    track,task,tid=c['track'],c['task'],c['task_id']
    assert source.sha(c['input_file'])==c['input_sha256']
    with np.load(c['input_file']) as z:
        assert 'y_target' not in z.files
        X0,y0,Q0,numeric,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
    assert source.sha(c['reference_file'])==c['reference_sha256']
    with np.load(c['reference_file']) as z:RX,Ry,rids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices'))
    assert set(rids).isdisjoint(qids) and set(rids).isdisjoint(sids)
    assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
    with np.load(c['adaptation_file']) as z:
        assert 'y_target' not in z.files
        apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
    assert set(apos).isdisjoint(epos)
    folds=np.arange(len(apos))%4
    checkpointdir=oldbank.training.OUT/'checkpoints'/track/task/str(tid)
    checkpoint=json.loads((checkpointdir/'result.json').read_text())
    assert source.sha(checkpoint['checkpoint_file'])==checkpoint['checkpoint_sha256']
    pfile=source.OUT/'policies'/task/f'{c["fold"]}.json';gate=json.loads(pfile.read_text())
    assert c['data_id'] not in gate['training_data_ids']
    K=len(np.unique(y0)) if task=='classification' else None
    predictor=Predictor(task,'cuda:0');cls=TabICLClassifier if task=='classification' else TabICLRegressor
    predictor.models[1]=cls(n_estimators=1,model_path=checkpoint['checkpoint_file'],allow_auto_download=False,
        device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
    for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
        dest=OUT/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
        if dest.with_suffix('.json').exists():
            assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
        if rho:X,y,Q,genmeta=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
        else:X,y,Q,genmeta=X0,y0,Q0,{}
        poolX,pooly=np.r_[X,RX],np.r_[y,Ry]
        observed_ids=sids[np.array(genmeta['selected_source_positions'])] if genmeta.get('source_rows_resampled') else sids
        poolids=np.r_[observed_ids,rids]
        assert set(poolids).isdisjoint(qids)
        groups=geometry.construct(poolX,pooly,Q,task)
        prefix=(K,) if task=='classification' else ()
        make=lambda n:np.full((n,)+prefix,np.nan,dtype=float)
        reference_native,reference_shared=make(len(Q)),make(len(Q));reference_members=np.stack([make(len(Q)) for _ in range(5)])
        final_native,final_shared=make(len(epos)),make(len(epos));final_members=np.stack([make(len(epos)) for _ in range(5)])
        cv_native,cv_shared=make(len(apos)),make(len(apos));cv_members=np.stack([make(len(apos)) for _ in range(5)])
        oldpath=oldbank.OUT/'predictions'/track/task/str(tid)/dest.name
        oldmeta=json.loads(oldpath.with_suffix('.json').read_text());assert source.sha(oldpath)==oldmeta['output_sha256']
        with np.load(oldpath) as z:oldnative,oldshared=z['Native_finetuned'],z['Shared_finetuned']
        original=source.OUT/'predictions'/track/task/str(tid)/dest.name
        om=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==om['output_sha256']
        with np.load(original) as z:original_base=z['Base']
        cache={};records=[]
        def infer(A,b,T):
            assert len(b)<=256
            key=tuple(protocol.array_sha256(v) for v in (A,b,T))
            if key not in cache:
                for estimator in predictor.models.values():
                    root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                    if root is not None and hasattr(root,'root'):root.root=None
                torch.cuda.empty_cache()
                cache[key]=predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
            return cache[key]
        for group in groups:
            qi,rows=group['query_positions'],group['pool_positions']
            CX,Cy=poolX[rows],pooly[rows]
            fullviews=source.constructors.shared_contexts(CX,Cy,Q[qi],len(Cy),task,None)
            assert len(fullviews)==5
            columns=None
            if task=='classification':
                order=source.constructors.specific.importance(CX,Cy.astype(float))
                columns=np.array([j for j in range(CX.shape[1]) if j!=int(order[0])])
                if not len(columns):columns=np.array([0])
            def predict_set(mode,fold,train_positions,train_labels,wanted,output_positions):
                if not len(wanted):return
                assert set(train_positions).isdisjoint(wanted)
                local={int(v):i for i,v in enumerate(qi)}
                take=np.array([local[int(v)] for v in wanted],dtype=int)
                AX=Q[train_positions]
                fitted=[];member=[]
                for name,A,b,Tall in [('canonical',CX,Cy,Q[qi])]+fullviews:
                    T=Tall[take]
                    V,labels,record=reserve_target(name,A,b,T,AX,train_labels,task,columns)
                    fitted.append(record|{'name':name,'X_sha256':protocol.array_sha256(V),
                        'y_sha256':protocol.array_sha256(labels),'query_X_sha256':protocol.array_sha256(T)})
                    member.append(infer(V,labels,T))
                    if name=='canonical':mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                native=member[0];members=np.stack(member[1:])
                shared=source.fusion.predict({'task':task,'base':native,'bank':members,'mu':mu,'sigma':sigma},gate['Shared'])
                if mode=='reference':
                    reference_native[output_positions]=native;reference_shared[output_positions]=shared;reference_members[:,output_positions]=members
                elif mode=='final':
                    final_native[output_positions]=native;final_shared[output_positions]=shared;final_members[:,output_positions]=members
                else:
                    cv_native[output_positions]=native;cv_shared[output_positions]=shared;cv_members[:,output_positions]=members
                records.append({'mode':mode,'fold':fold,'group':group['group'],'pool_positions':rows.tolist(),
                    'query_positions':wanted.tolist(),'output_positions':output_positions.tolist(),
                    'target_training_positions':train_positions.tolist(),'source_mean':mu,'source_scale':sigma,'contexts':fitted})
            predict_set('reference',None,np.array([],dtype=int),ay[:0],qi,qi)
            for fold in range(4):
                val_positions=np.flatnonzero((folds==fold)&np.isin(apos,qi))
                train_positions=np.flatnonzero(folds!=fold)
                predict_set('crossfit',fold,apos[train_positions],ay[train_positions],apos[val_positions],val_positions)
            final_positions=np.flatnonzero(np.isin(epos,qi))
            predict_set('final',None,apos,ay,epos[final_positions],final_positions)
        for p in [reference_native,reference_shared,reference_members,final_native,final_shared,final_members,cv_native,cv_shared,cv_members]:assert np.isfinite(p).all()
        scale=max(float(y.std()),1e-12)
        native_policy=selector.select(ay,np.stack([original_base[apos],oldnative[apos],cv_native]),task,scale)
        shared_policy=selector.select(ay,np.stack([original_base[apos],oldshared[apos],cv_shared]),task,scale)
        native_selected=selector.apply(native_policy,np.stack([original_base[epos],oldnative[epos],final_native]),task)
        shared_selected=selector.apply(shared_policy,np.stack([original_base[epos],oldshared[epos],final_shared]),task)
        source.npwrite(dest,Base=original_base[epos],Native_finetuned=oldnative[epos],Shared_finetuned=oldshared[epos],
            Native_reference_only=reference_native[epos],Shared_reference_only=reference_shared[epos],
            Native_target_context=final_native,Shared_target_context=final_shared,
            Native_adapted=native_selected,Shared_adapted=shared_selected,
            reference_native_full=reference_native,reference_shared_full=reference_shared,reference_members_full=reference_members,
            cv_native=cv_native,cv_shared=cv_shared,cv_members=cv_members,final_members=final_members)
        atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('track','task','task_id','data_id','fold')}|
            {'family':family,'rho':rho,'output_sha256':source.sha(dest),'actual_GPU':gpu,
             'observed_source_X_sha256':protocol.array_sha256(X),'observed_source_y_sha256':protocol.array_sha256(y),
             'observed_query_X_sha256':protocol.array_sha256(Q),'input_sha256':c['input_sha256'],
             'reference_sha256':c['reference_sha256'],'reference_rows':len(Ry),'adaptation_sha256':c['adaptation_sha256'],
             'pool_row_ids_sha256':protocol.array_sha256(poolids),'source_training_checkpoint_sha256':checkpoint['checkpoint_sha256'],
             'previous_bank_sha256':oldmeta['output_sha256'],'original_prediction_sha256':om['output_sha256'],
             'policy_sha256':source.sha(pfile),'target_adaptation_positions':apos.tolist(),'evaluation_positions':epos.tolist(),
             'adaptation_folds':folds.tolist(),'Native_policy':native_policy,'Shared_policy':shared_policy,
             'final_eval_labels_opened':False,'final_eval_scores_used_for_selection':False,'actual_TFM_calls':len(cache),'predicted_context_sets':records})
        print('REFERENCE_TARGET_CONDITION',track,task,tid,family,rho,'calls',len(cache),flush=True)
    del predictor;gc.collect();torch.cuda.empty_cache()
    print('REFERENCE_TARGET_CASE_COMPLETE',track,task,tid,flush=True)


def worker(shard,shards,pilot):
    torch.set_num_threads(1);torch.manual_seed(0)
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);plan=json.loads((OUT/'plan.json').read_text())
    assert gpu in plan['GPU_policy']['allowed_GPUs'] and gpu not in plan['GPU_policy']['reserved_GPUs']
    assert source.sha(__file__)==plan['script_sha256']
    cases=json.loads((oldbank.split.OUT/'case_manifest.json').read_text())
    positions=PILOT_POSITIONS if pilot else np.random.default_rng(323).permutation(len(cases))
    for i,index in enumerate(positions):
        if i%shards==shard:run_case(cases[int(index)],gpu)


def launch(pilot):
    prepare();plan=json.loads((OUT/'plan.json').read_text());suffix='pilot' if pilot else 'full'
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i)
        command=[sys.executable,__file__,'worker','--shard',str(i),'--shards','6']+(['--pilot'] if pilot else [])
        log=OUT/f'{suffix}_worker_{i}.log'
        with log.open('a') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'GPU':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(OUT/f'{suffix}_execution.json',{'status':'RUNNING','GPU_policy':plan['GPU_policy']})
    done=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(6)]):
            done.append(future.result());atomic_json(OUT/f'{suffix}_execution.json',{'status':'RUNNING','workers':done})
    cases=json.loads((oldbank.split.OUT/'case_manifest.json').read_text())
    selected=[cases[i] for i in PILOT_POSITIONS] if pilot else cases
    expected=plan['pilot_conditions'] if pilot else plan['conditions']
    count=sum(len(list((OUT/'predictions'/c['track']/c['task']/str(c['task_id'])).glob('*.json'))) for c in selected)
    success=all(r['returncode']==0 for r in done) and count==expected
    atomic_json(OUT/f'{suffix}_execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':done,
        'conditions':count,'expected':expected,'GPU_policy':plan['GPU_policy']})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','launch','worker']);p.add_argument('--pilot',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='launch':launch(a.pilot)
    else:worker(a.shard,a.shards,a.pilot)
