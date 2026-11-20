"""Label-free TFM representations for a supervised linear adaptation head.

Only the 256 observed source labels enter TFM contexts. Reference/calibration
and original-query features are appended as unlabeled prediction rows. Target
labels are not opened by this feature extraction program.
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
import task_adaptation_bank_v312 as representation
import interaction_sampling_schedule_v393 as schedule
from prepare_data import RUN,ROOT,atomic_json,protocol

source=schedule.source
OUT=RUN/'results/interaction_sampling_probe_features_v394'


def projection(phi):
    phi=np.asarray(phi,dtype=np.float32)
    phi=phi/np.maximum(np.linalg.norm(phi,axis=1,keepdims=True),1e-12)
    if phi.shape[1]>256:
        matrix=np.random.default_rng(0).normal(size=(phi.shape[1],256)).astype(np.float32)/16.
        phi=phi@matrix
    return phi


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    schedule.prepare()
    training_plan=json.loads((schedule.OUT/'plan.json').read_text())
    plan={'active_endpoint':list(schedule.endpoint.ENDPOINT),'seed':0,'datasets':35,'conditions':140,
        'pilot_task_ids':training_plan['pilot_task_ids'],'pilot_selection':'minimum, median and maximum observable source/reference class-prior discrepancy',
        'checkpoint':'unchanged source-only v307 weights; no target labels in weights',
        'TFM_context':'original observed source256, no appended target labels',
        'prediction_rows':'independent reference train features, reserved source-calibration features, original query features',
        'TFM_prediction_rows_labels_not_used':True,'target_labels_opened':False,
        'Native_features':'canonical decoder input, projected to256 dimensions, plus log canonical probabilities',
        'Shared_features':'same canonical representation, plus log Native/Shared/five-member probabilities',
        'registry_and_gate':'unchanged five Shared contexts and dataset-OOF v77 gate',
        'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'frozen_registry_sha256':schedule.verify_frozen(),'script_sha256':source.sha(__file__),
        'representation_sha256':source.sha(representation.__file__),'generator_sha256':source.sha(source.novel.__file__),
        'registry_sha256':source.sha(source.constructors.__file__),'fusion_sha256':source.sha(source.fusion.__file__),
        'GPU_policy':training_plan['GPU_policy'],'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards,pilot):
    torch.set_num_threads(1);torch.manual_seed(0);gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in range(6)
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    cases=schedule.endpoint.endpoint_cases()
    if pilot:cases=[c for c in cases if c['task_id'] in plan['pilot_task_ids']]
    for i,c in enumerate(cases):
        if i%shards!=shard:continue
        tid=c['task_id'];assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0,y0,Q0,num,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:RX,CX,rids,cids=(z[k] for k in ('X_train_reference','X_calibration','train_reference_indices','calibration_indices'))
        assert set(np.r_[rids,cids]).isdisjoint(qids) and set(np.r_[rids,cids]).isdisjoint(sids) and set(rids).isdisjoint(cids)
        K=len(np.unique(y0))
        cp=json.loads((schedule.endpoint.oldbank.training.OUT/'checkpoints'/'shift'/'classification'/str(tid)/'result.json').read_text())
        assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
        predictor=representation.LatentPredictor('classification','cuda:0')
        predictor.models[1]=TabICLClassifier(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
            device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json';gate=json.loads(pfile.read_text());assert c['data_id'] not in gate['training_data_ids']
        for rho in source.novel.STRENGTHS:
            dest=OUT/'features'/str(tid)/f'interaction_sampling_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            X,y,Q,_=source.novel.generate(X0,y0,Q0,num,'classification','interaction_sampling',rho,tid)
            allQ=np.r_[RX,CX,Q]
            # Reference/calibration rows retain their source-domain features;
            # target rows follow the unchanged generator output; source rows are resampled.
            # No oracle mapping or target labels enter representation extraction.
            native,phi=predictor.core(X,y,allQ,K)
            embedding=projection(phi);raw_dimension=phi.shape[1];del phi
            views=source.constructors.shared_contexts(X,y,Q,len(y),'classification',None)
            order=source.constructors.specific.importance(X,y.astype(float));columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            members=[];records=[]
            for name,A,b,T in views:
                prediction_X=allQ[:,columns] if name=='feature_subspace' else allQ
                for estimator in predictor.models.values():
                    root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                    if root is not None and hasattr(root,'root'):root.root=None
                torch.cuda.empty_cache()
                p=predictor(A,b,prediction_X,members=1,random_state=0,n_classes=K)[0]
                members.append(p);records.append({'name':name,'rows':len(b),'X_sha256':protocol.array_sha256(A),
                    'y_sha256':protocol.array_sha256(b),'prediction_X_sha256':protocol.array_sha256(prediction_X)})
            bank=np.stack(members);mu,sigma=float(y.mean()),max(float(y.std()),1e-12)
            shared=source.fusion.predict({'task':'classification','base':native,'bank':bank,'mu':mu,'sigma':sigma},gate['Shared'])
            log=lambda p:np.log(np.clip(p,1e-12,1.))
            Fnative=np.c_[embedding,log(native)]
            Fshared=np.c_[embedding,log(native),log(shared),log(bank).transpose(1,0,2).reshape(len(allQ),-1)]
            source.npwrite(dest,Native_features=Fnative.astype(np.float32),Shared_features=Fshared.astype(np.float32),
                Native_raw=native,Shared_raw=shared,members=bank)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task_id','data_id','task','track','fold')}|
                {'family':'interaction_sampling','rho':rho,'output_sha256':source.sha(dest),'input_sha256':c['input_sha256'],
                 'reference_sha256':c['reference_sha256'],'checkpoint_sha256':cp['checkpoint_sha256'],'policy_sha256':source.sha(pfile),
                 'reference_rows':len(RX),'calibration_rows':len(CX),'original_query_rows':len(Q),
                 'prediction_row_ids_sha256':protocol.array_sha256(np.r_[rids,cids,qids]),'raw_representation_dimension':raw_dimension,
                 'projected_representation_dimension':embedding.shape[1],'observed_source_X_sha256':protocol.array_sha256(X),
                 'observed_source_y_sha256':protocol.array_sha256(y),'unlabeled_prediction_X_sha256':protocol.array_sha256(allQ),
                 'source_mean':mu,'source_scale':sigma,'contexts':records,'actual_GPU':gpu,
                 'target_labels_opened':False,'target_labels_in_context':0,'context_rows':len(y)})
            print('HEAVY_PROBE_FEATURES',tid,rho,'reference',len(RX),'raw_dimension',raw_dimension,flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()


def launch(pilot):
    prepare();n=3 if pilot else 6;prefix='pilot' if pilot else 'full'
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'{prefix}_worker_{i}.log'
        command=[sys.executable,__file__,'worker','--shard',str(i),'--shards',str(n)]+(['--pilot'] if pilot else [])
        with log.open('a') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'GPU':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(OUT/f'{prefix}_execution.json',{'status':'RUNNING','active_endpoint':list(schedule.endpoint.ENDPOINT)})
    with ThreadPoolExecutor(max_workers=n) as pool:results=list(pool.map(execute,range(n)))
    ok=all(r['returncode']==0 for r in results)
    atomic_json(OUT/f'{prefix}_execution.json',{'status':'COMPLETE' if ok else 'FAILED','workers':results})
    schedule.verify_frozen();assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    assert ok


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','launch','worker']);p.add_argument('--pilot',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='launch':launch(a.pilot)
    else:worker(a.shard,a.shards,a.pilot)
