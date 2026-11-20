"""Reference-pool contexts with honest cross-fitted target-label insertion.

Same64 target labels and64 evaluation rows as448. Five original Shared rules,
source-only307 weights,256-row cap and seed0 remain fixed. Evaluation labels
are not opened. This is a supervised extension, not original zero-shot Shared.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
from tabicl import TabICLClassifier
import task_adaptation_bank_v312 as oldbank
import interaction_sampling_probe_features_v394 as features
import interaction_sampling_nested_labels_v447 as split64
import reference_context_geometry_v321 as geometry
import exact_base_anchor_gate_v439 as gate
from shared_finetuned_inference_v309 import reserve_target
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import Predictor
from serial_endpoint_state_v371 import verify_registry

source=oldbank.source
OUT=RUN/'results/interaction_sampling_context_adaptation_v455'


def prepare():
    split_plan=json.loads((split64.OUT/'plan.json').read_text())
    assert source.sha(split64.OUT/'case_manifest.json')==split_plan['manifest_sha256']
    assert source.sha(split64.__file__)==split_plan['script_sha256']
    policy=json.loads(oldbank.training.implementation.GPU_POLICY.read_text())
    assert policy['allowed_GPUs']==list(range(6)) and policy['reserved_GPUs']==[6,7]
    plan={'seed':0,'active_endpoint':['shift','classification','interaction_sampling'],
        'datasets':35,'conditions':140,'target_adaptation_rows':64,'evaluation_rows':64,
        'context_cap':256,'context_pool':'observed source256 plus existing disjoint clean labeled source references',
        'membership':'unchanged321 four unlabeled query groups, nearest class-covering256 pool rows',
        'context_rules':'canonical plus unchanged original five Shared constructors28',
        'target_context_labels':'crossfit48 labels when predicting other16; all64 when predicting remaining evaluation64',
        'backbone':'same source-only307 weights; no further gradient updates',
        'Native_and_Shared_same_permitted_pool_and_target_labels':True,
        'TFM_calls_not_compute_matched_to_original_Base':True,
        'fusion':'unchanged dataset-OOF77 Shared gate, followed by calibration-only exact original-Base anchor439',
        'final_eval_labels_used_for_fit_or_selection':False,'previously_scored_development_cohort':True,
        'not_original_zero_shot_Shared':True,'script_sha256':source.sha(__file__),
        'geometry_sha256':source.sha(geometry.__file__),'gate_sha256':source.sha(gate.__file__),
        'constructor_sha256':source.sha(source.constructors.__file__),'fusion_sha256':source.sha(source.fusion.__file__),
        'generator_sha256':source.sha(source.novel.__file__),'split_plan_sha256':source.sha(split64.OUT/'plan.json'),
        'frozen_registry_sha256':verify_registry(RUN/'results/frozen_endpoints_v391/registry.json')['registry_sha256'],
        'GPU_policy':policy,'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text())==plan
    else:
        atomic_json(p,plan)


def run_case(c,gpu):
    tid=c['task_id']
    assert source.sha(c['input_file'])==c['input_sha256']
    with np.load(c['input_file']) as z:
        assert 'y_target' not in z.files
        X0,y0,Q0,num,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
    assert source.sha(c['reference_file'])==c['reference_sha256']
    with np.load(c['reference_file']) as z:
        RX,Ry,rids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices'))
    assert set(rids).isdisjoint(np.r_[sids,qids])
    assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
    with np.load(c['adaptation_file']) as z:
        apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
    assert len(apos)==len(epos)==64 and set(apos).isdisjoint(epos)
    folds=np.arange(len(apos))%4
    cpdir=oldbank.training.OUT/'checkpoints'/'shift'/'classification'/str(tid)
    cp=json.loads((cpdir/'result.json').read_text())
    assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
    assert not cp['target_adaptation_labels_used_for_weights']
    pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json'
    snapshot=json.loads(pfile.read_text())
    assert c['data_id'] not in snapshot['training_data_ids']
    K=len(np.unique(y0))
    predictor=Predictor('classification','cuda:0')
    predictor.models[1]=TabICLClassifier(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
        device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
    for rho in source.novel.STRENGTHS:
        dest=OUT/'predictions'/str(tid)/f'interaction_sampling_{rho}.npz'
        if dest.with_suffix('.json').exists():
            assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256']
            continue
        X,y,Q,genmeta=source.novel.generate(X0,y0,Q0,num,'classification','interaction_sampling',rho,tid)
        poolX,pooly=np.r_[X,RX],np.r_[y,Ry]
        observed_ids=sids[np.asarray(genmeta['selected_source_positions'])] if genmeta.get('source_rows_resampled') else sids
        poolids=np.r_[observed_ids,rids]
        assert set(poolids).isdisjoint(qids)
        groups=geometry.construct(poolX,pooly,Q,'classification')
        make=lambda n:np.full((n,K),np.nan)
        cv_native,cv_shared,final_native,final_shared=[make(64) for _ in range(4)]
        cv_members,final_members=np.stack([make(64) for _ in range(5)]),np.stack([make(64) for _ in range(5)])
        cache={};records=[]
        def infer(A,b,T):
            assert len(b)<=256
            key=tuple(protocol.array_sha256(v) for v in (A,b,T))
            if key not in cache:
                for estimator in predictor.models.values():
                    root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                    if root is not None and hasattr(root,'root'):
                        root.root=None
                torch.cuda.empty_cache()
                cache[key]=predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
            return cache[key]
        for group in groups:
            qi,rows=group['query_positions'],group['pool_positions']
            CX,Cy=poolX[rows],pooly[rows]
            views=source.constructors.shared_contexts(CX,Cy,Q[qi],len(Cy),'classification',None)
            order=source.constructors.specific.importance(CX,Cy.astype(float))
            columns=np.asarray([j for j in range(CX.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            def predict_set(mode,fold,train,wanted,output):
                if not len(wanted):return
                train_positions=apos[train]
                assert set(train_positions).isdisjoint(wanted)
                local={int(v):i for i,v in enumerate(qi)}
                take=np.asarray([local[int(v)] for v in wanted],dtype=int)
                predictions=[];contexts=[]
                for name,A,b,Tall in [('canonical',CX,Cy,Q[qi])]+views:
                    T=Tall[take]
                    V,labels,record=reserve_target(name,A,b,T,Q[train_positions],ay[train],'classification',columns)
                    predictions.append(infer(V,labels,T))
                    contexts.append(dict(record,name=name,X_sha256=protocol.array_sha256(V),
                        y_sha256=protocol.array_sha256(labels),query_X_sha256=protocol.array_sha256(T)))
                    if name=='canonical':mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                native,bank=predictions[0],np.stack(predictions[1:])
                shared=source.fusion.predict({'task':'classification','base':native,'bank':bank,'mu':mu,'sigma':sigma},snapshot['Shared'])
                if mode=='crossfit':
                    cv_native[output]=native;cv_shared[output]=shared;cv_members[:,output]=bank
                else:
                    final_native[output]=native;final_shared[output]=shared;final_members[:,output]=bank
                records.append({'mode':mode,'fold':fold,'group':group['group'],'pool_positions':rows.tolist(),
                    'query_positions':wanted.tolist(),'output_positions':output.tolist(),
                    'target_training_positions':train_positions.tolist(),'source_mean':mu,'source_scale':sigma,'contexts':contexts})
            for fold in range(4):
                val=np.flatnonzero((folds==fold)&np.isin(apos,qi))
                train=np.flatnonzero(folds!=fold)
                predict_set('crossfit',fold,train,apos[val],val)
            output=np.flatnonzero(np.isin(epos,qi))
            predict_set('final',None,np.arange(64),epos[output],output)
        for a in (cv_native,cv_shared,final_native,final_shared,cv_members,final_members):assert np.isfinite(a).all()
        original=source.OUT/'predictions'/'shift'/'classification'/str(tid)/dest.name
        om=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==om['output_sha256']
        with np.load(original) as z:base=z['Base']
        npolicy,spolicy=gate.select(ay,base[apos],cv_native),gate.select(ay,base[apos],cv_shared)
        source.npwrite(dest,Base=base[epos],Native_probe=gate.apply(npolicy,base[epos],final_native),
            Shared_probe=gate.apply(spolicy,base[epos],final_shared),Native_context_raw=final_native,
            Shared_context_raw=final_shared,Native_cv_probe=cv_native,Shared_cv_probe=cv_shared,
            cv_members=cv_members,members=final_members,Original_Base_cv=base[apos])
        atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('track','task','task_id','data_id','fold')}|
            {'family':'interaction_sampling','rho':rho,'output_sha256':source.sha(dest),'actual_GPU':gpu,
             'original_input_sha256':c['input_sha256'],'reference_sha256':c['reference_sha256'],
             'adaptation_sha256':c['adaptation_sha256'],'adaptation_positions':apos.tolist(),'evaluation_positions':epos.tolist(),
             'pool_X_sha256':protocol.array_sha256(poolX),'pool_y_sha256':protocol.array_sha256(pooly),
             'pool_row_ids_sha256':protocol.array_sha256(poolids),'original_prediction_sha256':om['output_sha256'],
             'checkpoint_sha256':cp['checkpoint_sha256'],'training_result_file':str(cpdir/'result.json'),
             'policy_sha256':source.sha(pfile),'evaluation_labels_opened':False,
             'Native_outer_gate':npolicy,'Shared_outer_gate':spolicy,'context_records':records,'actual_TFM_calls':len(cache)})
        print('TARGET_CONTEXT',tid,rho,'calls',len(cache),flush=True)
    del predictor;gc.collect();torch.cuda.empty_cache()


def worker(shard):
    torch.set_num_threads(1);torch.manual_seed(0)
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in range(6)
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    for i,c in enumerate(split64.endpoint_cases()):
        if i%6==shard:run_case(c,gpu)


def launch():
    prepare()
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i)
        log=OUT/f'GPU_worker_{i}.log'
        with log.open('a') as handle:
            p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i)],env=env,stdout=handle,stderr=subprocess.STDOUT)
        return {'GPU':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(OUT/'execution.json',{'status':'RUNNING','GPU_workers':6,'reserved_GPUs':[6,7],'expected_conditions':140})
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(execute,range(6)))
    count=len(list((OUT/'predictions').glob('*/*.json')))
    ok=count==140 and all(r['returncode']==0 for r in results)
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if ok else 'FAILED','conditions':count,'workers':results})
    assert ok and (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['launch','worker']);parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args();launch() if args.phase=='launch' else worker(args.shard)
