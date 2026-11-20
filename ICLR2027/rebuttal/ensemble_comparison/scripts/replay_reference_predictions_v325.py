"""Independent GPU replay of core and transformed Shared branch predictions."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
from tabicl import TabICLClassifier,TabICLRegressor
import reference_target_contexts_v323 as run
from prepare_data import atomic_json
from run_tabicl import Predictor


def main(pilot=False):
    torch.set_num_threads(1);torch.manual_seed(0)
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in range(6)
    allcases=json.loads((run.oldbank.split.OUT/'case_manifest.json').read_text())
    cases=[allcases[i] for i in run.PILOT_POSITIONS] if pilot else allcases
    reports=[]
    for c in cases:
        track,task,tid=c['track'],c['task'],c['task_id']
        with np.load(c['input_file']) as z:X0,y0,Q0,num=(z[k] for k in ('X_source','y_source','X_target','numeric_columns'))
        with np.load(c['reference_file']) as z:RX,Ry=z['X_train_reference'],z['y_train_reference']
        with np.load(c['adaptation_file']) as z:apos,ay=z['adaptation_positions'],z['adaptation_y']
        cp=json.loads((run.oldbank.training.OUT/'checkpoints'/track/task/str(tid)/'result.json').read_text())
        assert run.source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
        model=Predictor(task,'cuda:0');cls=TabICLClassifier if task=='classification' else TabICLRegressor
        model.models[1]=cls(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
            device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
        for family,rho in [('clean',0),(c['families'][0],100)]:
            p=run.OUT/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            meta=json.loads(p.with_suffix('.json').read_text());assert run.source.sha(p)==meta['output_sha256']
            if rho:X,y,Q,_=run.source.novel.generate(X0,y0,Q0,num,task,family,rho,tid)
            else:X,y,Q=X0,y0,Q0
            poolX,pooly=np.r_[X,RX],np.r_[y,Ry]
            record=max([r for r in meta['predicted_context_sets'] if r['mode']=='final'],key=lambda r:len(r['query_positions']))
            rows=np.asarray(record['pool_positions']);wanted=np.asarray(record['query_positions']);output=np.asarray(record['output_positions'])
            group=next(g for g in run.geometry.construct(poolX,pooly,Q,task) if g['group']==record['group'])
            qi=group['query_positions'];CX,Cy=poolX[rows],pooly[rows]
            views=[('canonical',CX,Cy,Q[qi])]+run.source.constructors.shared_contexts(CX,Cy,Q[qi],len(Cy),task,None)
            local={int(v):i for i,v in enumerate(qi)};take=np.array([local[int(v)] for v in wanted])
            columns=None
            if task=='classification':
                rank=run.source.constructors.specific.importance(CX,Cy.astype(float))
                columns=np.array([j for j in range(CX.shape[1]) if j!=int(rank[0])])
                if not len(columns):columns=np.array([0])
            for member in [0,4]:
                name,A,b,T=views[member];T=T[take]
                V,labels,_=run.reserve_target(name,A,b,T,Q[apos],ay,task,columns)
                for estimator in model.models.values():
                    root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                    if root is not None and hasattr(root,'root'):root.root=None
                actual=model(V,labels,T,members=1,random_state=0,n_classes=len(np.unique(y0)) if task=='classification' else None)[0]
                with np.load(p) as z:saved=z['Native_target_context'][output] if member==0 else z['final_members'][member-1,output]
                error=float(np.max(np.abs(actual-saved)))
                np.testing.assert_allclose(actual,saved,atol=2e-6,rtol=2e-6)
                reports.append({'track':track,'task':task,'task_id':tid,'family':family,'rho':rho,'member':name,'max_abs_difference':error})
        del model;torch.cuda.empty_cache()
        print('NEURAL_REPLAY_CASE',track,task,tid,flush=True)
    atomic_json(run.OUT/f'{"pilot_" if pilot else ""}neural_spot_replay.json',{'status':'PASS','pilot_only':pilot,
        'contexts_independently_recomputed':len(reports),'sampling':'clean and first family at100; largest final query group; canonical and transformed branch',
        'coverage':'spot checks, not every neural prediction','results':reports,'script_sha256':run.source.sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
