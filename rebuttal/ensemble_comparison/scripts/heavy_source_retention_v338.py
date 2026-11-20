"""Sequential active-endpoint-only source-retention adaptation, unchanged eval."""
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
import serial_heavy_classification_v331 as endpoint
import heavy_accuracy_finetuning_v334 as frozen_check
import source_retention_context_v337 as retention
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import Predictor

source=endpoint.source
OUT=RUN/'results/heavy_source_retention_v338'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'active_endpoint':list(endpoint.ENDPOINT),'seed':0,'datasets':64,'conditions':256,
        'source_row_choices':list(retention.COUNTS),'target_context_labels':'same32 v311 adaptation rows, excluded from evaluation',
        'actual_source_rows':'class coverage can raise nonzero requested count; bounded by available rows and256 total context',
        'weights':'unchanged source-only fine-tuned307; target labels never in gradient weights',
        'registry':'same original five Shared source constructors, then source thinning and target append',
        'four_fold_context_training':'other three adaptation folds only; no held-out fold label in context',
        'selection':'cross-fit NLL then accuracy; candidate-to-original-Base blend alpha .25,.5,.75,1; identity included',
        'members':6,'context_cap':256,'n_estimators_each_context':1,'norm_methods':['none'],
        'Native_and_Shared_tuned_independently_with_same_label_budget':True,
        'extra_control':'Native with Shared chosen source retention and blend weight',
        'benchmark_and_eval_rows_unchanged':True,'Base_predictions_reused_exactly':True,
        'other_endpoints_not_run':True,'extra_source_and_test_subset_supervision_explicit':True,
        'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'frozen_registry_sha256':frozen_check.verify_frozen(),'script_sha256':source.sha(__file__),
        'retention_sha256':source.sha(retention.__file__),'generator_sha256':source.sha(source.novel.__file__),
        'registry_sha256':source.sha(source.constructors.__file__),'fusion_sha256':source.sha(source.fusion.__file__),
        'GPU_policy':json.loads(endpoint.oldbank.training.implementation.GPU_POLICY.read_text()),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards):
    torch.set_num_threads(1);torch.manual_seed(0);gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in range(6)
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    cases=endpoint.endpoint_cases()
    for index,pos in enumerate(np.random.default_rng(338).permutation(len(cases))):
        if index%shards!=shard:continue
        c=cases[int(pos)];tid=c['task_id']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:X0,y0,Q0,num=(z[k] for k in ('X_source','y_source','X_target','numeric_columns'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert set(apos).isdisjoint(epos);folds=np.arange(len(apos))%4;K=len(np.unique(y0))
        cp=json.loads((endpoint.oldbank.training.OUT/'checkpoints'/'perturbation'/'classification'/str(tid)/'result.json').read_text())
        assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
        predictor=Predictor('classification','cuda:0')
        predictor.models[1]=TabICLClassifier(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
            device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json';gate=json.loads(pfile.read_text())
        assert c['data_id'] not in gate['training_data_ids']
        for rho in source.novel.STRENGTHS:
            family='cell_heavy_tail';dest=OUT/'predictions'/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            X,y,Qfull,_=source.novel.generate(X0,y0,Q0,num,'classification',family,rho,tid)
            views=[('canonical',X,y,Qfull)]+source.constructors.shared_contexts(X,y,Qfull,len(y),'classification',None)
            order=source.constructors.specific.importance(X,y.astype(float));columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/dest.name
            oldfile=endpoint.oldbank.OUT/'predictions'/'perturbation'/'classification'/str(tid)/dest.name
            om=json.loads(original.with_suffix('.json').read_text());oldm=json.loads(oldfile.with_suffix('.json').read_text())
            assert source.sha(original)==om['output_sha256'] and source.sha(oldfile)==oldm['output_sha256']
            with np.load(original) as z:base=z['Base']
            with np.load(oldfile) as z:oldnative,oldshared=z['Native_finetuned'],z['Shared_finetuned']
            records=[];cache={}
            def infer(A,b,T):
                key=tuple(protocol.array_sha256(v) for v in (A,b,T))
                if key not in cache:
                    for est in predictor.models.values():
                        root=getattr(getattr(est,'model_',None),'icl_predictor',None)
                        if root is not None and hasattr(root,'root'):root.root=None
                    torch.cuda.empty_cache()
                    cache[key]=predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
                return cache[key]
            def predict_set(mode,fold,count,train_positions,train_labels,wanted):
                assert set(train_positions).isdisjoint(wanted)
                predicted=[];meta=[]
                for name,A,b,Tall in views:
                    T=Tall[wanted]
                    V,labels,record=retention.context(name,A,b,T,Qfull[train_positions],train_labels,count,columns)
                    predicted.append(infer(V,labels,T))
                    meta.append(record|{'name':name,'X_sha256':protocol.array_sha256(V),'y_sha256':protocol.array_sha256(labels),'query_X_sha256':protocol.array_sha256(T)})
                    if name=='canonical':mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                native,members=predicted[0],np.stack(predicted[1:])
                shared=source.fusion.predict({'task':'classification','base':native,'bank':members,'mu':mu,'sigma':sigma},gate['Shared'])
                records.append({'mode':mode,'fold':fold,'requested_source_rows':count,'training_positions':train_positions.tolist(),
                    'query_positions':wanted.tolist(),'source_mean':mu,'source_scale':sigma,'contexts':meta})
                return native,shared,members
            cv_native=np.full((len(retention.COUNTS),len(apos),K),np.nan)
            cv_shared=np.full_like(cv_native,np.nan);cv_members=np.full((len(retention.COUNTS),5,len(apos),K),np.nan)
            for j,count in enumerate(retention.COUNTS):
                for fold in range(4):
                    tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                    native,shared,members=predict_set('crossfit',fold,count,apos[tr],ay[tr],apos[va])
                    cv_native[j,va]=native;cv_shared[j,va]=shared;cv_members[j,:,va,:]=members.transpose(1,0,2)
            assert np.isfinite(cv_native).all() and np.isfinite(cv_shared).all() and np.isfinite(cv_members).all()
            native_candidates={'warm_source':oldnative[apos]}|{str(n):cv_native[j] for j,n in enumerate(retention.COUNTS)}
            shared_candidates={'warm_source':oldshared[apos]}|{str(n):cv_shared[j] for j,n in enumerate(retention.COUNTS)}
            native_policy=retention.select(ay,base[apos],native_candidates)
            shared_policy=retention.select(ay,base[apos],shared_candidates)
            chosen=sorted({int(p['candidate']) for p in [native_policy,shared_policy] if p['candidate'] not in ['Base','warm_source']})
            final={}
            for n in chosen:final[n]=predict_set('final',None,n,apos,ay,epos)
            def candidate(policy,which):
                if policy['candidate']=='Base':return base[epos]
                if policy['candidate']=='warm_source':return (oldnative if which==0 else oldshared)[epos]
                return final[int(policy['candidate'])][which]
            native=retention.apply(native_policy,base[epos],candidate(native_policy,0))
            shared=retention.apply(shared_policy,base[epos],candidate(shared_policy,1))
            matched=retention.apply(shared_policy,base[epos],candidate(shared_policy,0))
            arrays={'Base':base[epos],'Native_adapted':native,'Shared_adapted':shared,'Native_matched_Shared_policy':matched,
                'cv_native':cv_native,'cv_shared':cv_shared,'cv_members':cv_members}
            for n,(core,p,members) in final.items():arrays.update({f'final_native_{n}':core,f'final_shared_{n}':p,f'final_members_{n}':members})
            source.npwrite(dest,**arrays)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task_id','data_id','task','track','fold')}|
                {'family':family,'rho':rho,'output_sha256':source.sha(dest),'actual_GPU':gpu,
                 'input_sha256':c['input_sha256'],'adaptation_sha256':c['adaptation_sha256'],'checkpoint_sha256':cp['checkpoint_sha256'],
                 'original_prediction_sha256':om['output_sha256'],'old_bank_sha256':oldm['output_sha256'],'policy_sha256':source.sha(pfile),
                 'adaptation_positions':apos.tolist(),'evaluation_positions':epos.tolist(),'adaptation_folds':folds.tolist(),
                 'Native_policy':native_policy,'Shared_policy':shared_policy,'prediction_sets':records,
                 'evaluation_labels_opened':False,'actual_TFM_calls':len(cache)})
            print('HEAVY_RETENTION_CONDITION',tid,rho,'native',native_policy['candidate'],native_policy['alpha'],'shared',shared_policy['candidate'],shared_policy['alpha'],flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()


def launch():
    prepare()
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards','6'],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'GPU':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(OUT/'execution.json',{'status':'RUNNING','active_endpoint':list(endpoint.ENDPOINT)})
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(execute,range(6)))
    count=len(list((OUT/'predictions').glob('*/*.json')));ok=all(r['returncode']==0 for r in results) and count==256
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if ok else 'FAILED','conditions':count,'workers':results,'active_endpoint':list(endpoint.ENDPOINT)})
    frozen_check.verify_frozen();assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    assert ok


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='launch':launch()
    else:worker(a.shard,a.shards)
