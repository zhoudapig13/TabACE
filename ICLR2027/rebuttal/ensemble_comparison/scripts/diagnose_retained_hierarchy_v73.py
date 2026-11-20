"""Isolate retained many-class state and automatic batching on TRAIN only.

Different execution batches under AMP may change rounding. Test the exact
last physical Devnagari context preceding CIFAR, then a fixed-batch control.
No scores choose a model/recipe and no earlier prediction is overwritten.
"""
import json
import os
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import torch
import balanced_clean_calibration_v71 as source
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor
from tabicl._model.inference import InferenceManager

OUT=RUN/'results/tabace_retained_hierarchy_diagnosis_v73'
TRACE=[]
ORIGINAL_ESTIMATE=InferenceManager.estimate_safe_batch_size


def estimate(self,*args,**kwargs):
    answer=ORIGINAL_ESTIMATE(self,*args,**kwargs)
    TRACE.append(dict(component=self.enc_name,sequence_length=int(args[0] if args else kwargs['seq_len']),
                      available_MB=float(answer[0]),estimated_batch=int(answer[1]),use_amp=self.use_amp))
    return answer


def inputs(tid,budget):
    meta=next(m for m in source.cohort() if int(m['task_id'])==tid)
    pool=source.full.reference.scaling.prepare('classification','five',meta);split=source.partition(meta,pool)
    path=source.OUT/'source'/str(budget)/f'{tid}.npz';info=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:base=z['base']
    data=dict(X_source=pool['X'][split['train_positions']][:budget],y_source=pool['y'][split['train_positions']][:budget],X_target=split['X_validation'])
    assert {k:protocol.array_sha256(v) for k,v in data.items()}==info['source_input_hashes']
    return meta,pool,split,data,base,info


def retained(model):
    root=getattr(model.models[1].model_.icl_predictor,'root',None)
    seen=set();shapes=[]
    def visit(node):
        if node is None:return
        for name,value in vars(node).items():
            if torch.is_tensor(value) and value.is_cuda:
                pointer=value.untyped_storage().data_ptr()
                if pointer not in seen:
                    seen.add(pointer);shapes.append(dict(name=name,shape=list(value.shape),bytes=value.untyped_storage().nbytes()))
        for child in getattr(node,'child_nodes',[]):visit(child)
    visit(root)
    return dict(present=root is not None,unique_cuda_storage_bytes=sum(s['bytes'] for s in shapes),tensor_shapes=shapes,
                allocated_cuda_bytes=torch.cuda.memory_allocated())


def run():
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    InferenceManager.estimate_safe_batch_size=estimate
    meta,pool,split,data,saved,info=inputs(167124,4096)
    dm,dp,ds,dd,db,di=inputs(167121,8192)
    views,_=source.construct(dm,dp,ds,8192,db)
    last=max(i for i,alias in enumerate(di['aliases']) if not alias)
    name,A,b,Q=views[last];A,b,Q=np.asarray(A,dtype=np.float32),np.asarray(b),np.asarray(Q,dtype=np.float32)
    assert [protocol.array_sha256(v) for v in [A,b,Q]]==di['member_input_hashes'][last]
    records=[];predictions={}
    def call(label,model,X,y,T,k,compare=True):
        TRACE.clear();started=time.monotonic()
        p,_=model(X,y,T,members=1,random_state=0,n_classes=k)
        row=dict(label=label,seconds=time.monotonic()-started,traces=list(TRACE),retained_state=retained(model),
                 max_error_vs_archived=float(np.max(np.abs(p-saved))) if compare else None)
        if compare:
            row['max_errors_vs_other_CIFAR_calls']={key:float(np.max(np.abs(p-value))) for key,value in predictions.items()}
            predictions[label]=p.copy()
        records.append(row);source.full.reference.scaling.npwrite(OUT/f'{label}.npz',prediction=p)
        atomic_json(OUT/'diagnosis.json',dict(status='RUNNING',task='TRAIN-only numerical execution diagnosis',
            archived_predictions_modified=False,query_labels_used=False,Devnagari_last_physical_view=dict(index=last,name=name),records=records))
        print('HIERARCHY_DIAGNOSTIC',label,row['max_error_vs_archived'],row.get('max_errors_vs_other_CIFAR_calls'),
              'retained_bytes',row['retained_state']['unique_cuda_storage_bytes'],'batches',[(t['component'],t['estimated_batch']) for t in TRACE],flush=True)
    predictor=Predictor('classification','cuda:0')
    call('fresh_CIFAR',predictor,data['X_source'],data['y_source'],data['X_target'],10)
    call('warm_Devnagari_exact_last_view',predictor,A,b,Q,46,False)
    call('after_Devnagari_CIFAR',predictor,data['X_source'],data['y_source'],data['X_target'],10)
    predictor.models[1].model_.icl_predictor.root=None;torch.cuda.empty_cache()
    call('cleared_hierarchy_CIFAR',predictor,data['X_source'],data['y_source'],data['X_target'],10)
    predictor.models[1].inference_config={key:dict(safety_factor=0.,min_batch_size=size,use_async=False)
        for key,size in [('COL_CONFIG',64),('ROW_CONFIG',128),('ICL_CONFIG',1)]}
    call('fixed_batches_CIFAR',predictor,data['X_source'],data['y_source'],data['X_target'],10)
    call('fixed_batches_warm_Devnagari',predictor,A,b,Q,46,False)
    call('fixed_batches_after_Devnagari_CIFAR',predictor,data['X_source'],data['y_source'],data['X_target'],10)
    result=json.loads((OUT/'diagnosis.json').read_text());result.update(status='COMPLETE_REQUIRES_ANALYSIS',time_unix=time.time(),
        fixed_batch_replay_error=float(np.max(np.abs(predictions['fixed_batches_CIFAR']-predictions['fixed_batches_after_Devnagari_CIFAR']))))
    atomic_json(OUT/'diagnosis.json',result)


if __name__=='__main__':run()
