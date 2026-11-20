"""Fixed feature-location perturbation candidates; source labels stay intact."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

from prepare_data import RUN,atomic_json,protocol
from tabicl_base256_calibration_v22 import OUT,sha

PROFILES=[(4.5,0.),(4.5,-.2),(4.5,.2),(5.,-.2),(5.,.2)]


def tag(beta,delta):return f'beta{beta:g}_location{delta:g}'


def declare():
    p=OUT/'class_location_plan.json';assert not p.exists()
    atomic_json(p,{'status':'DECLARED_BEFORE_ADDITIONAL_PREDICTIONS','profiles':PROFILES,
        'feature_multiplier':8,'outer_seed':0,'context_rows':256,'n_estimators':1,
        'additional_location_rule':'Q + delta * rho * source_centroid_direction',
        'binary_direction':'class1 centroid minus class0 centroid',
        'multiclass_direction':'most frequent source class centroid minus remaining source centroid',
        'source_labels_changed':False,'query_labels_used_by_generator':False,
        'original_queries_and_strength_grid_retained':True,'Clean_unchanged':True,
        'all_trials_retained':True,'Base_development_metrics_select_generator':True,
        'Shared_Specific_not_used':True,'original_profile_trials_sha256':sha(OUT/'classification_five_expanded_selection.json')})


def worker(shard,shards):
    import torch
    import warnings
    from sklearn.ensemble import ExtraTreesClassifier
    from run_tabicl import Predictor,metrics
    from scaled_historical_feature_protocol import apply_feature_scaled_shift
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('classification','cuda:0')
    tasks=[json.loads(p.read_text()) for p in sorted((RUN/'cache/five/classification').glob('*.json'))]
    for m in tasks[shard::shards]:
        with np.load(RUN/'cache/five/classification'/f'{m["task_id"]}.npz') as z:
            X,y,Q,t=(z[k] for k in ['X_source','y_source','X_target','y_target'])
        classes,counts=np.unique(y,return_counts=True);assert len(y)==256
        positive=classes[1] if len(classes)==2 else classes[int(np.argmax(counts))]
        direction=X[y==positive].mean(0)-X[y!=positive].mean(0)
        pseudo=ExtraTreesClassifier(n_estimators=128,random_state=0,n_jobs=1).fit(X,y).predict(Q)
        for beta,delta in PROFILES:
            for family in protocol.FAMILIES:
                for rho in [20,50,80,100]:
                    p=OUT/'classification_location_profiles'/tag(beta,delta)/str(m['task_id'])/f'{family}_{rho}.npz'
                    if p.exists() and p.with_suffix('.json').exists():continue
                    kind='aucband' if family=='band' else family
                    op=f'publicshift_d{m["data_id"]}_{kind}_rho{rho}_beta{int(beta*100)}'
                    shifted,_=apply_feature_scaled_shift(op,X,y,Q,pseudo,0,feature_multiplier=8)
                    shifted=shifted+delta*(rho/100.)*direction
                    pred,_=predictor(X,y,shifted,members=1,random_state=0,n_classes=len(classes))
                    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,prediction=pred,y_target=t);tmp.replace(p)
                    atomic_json(p.with_suffix('.json'),{'task_id':m['task_id'],'family':family,'rho':rho,
                        'beta':beta,'feature_multiplier':8,'location_delta':delta,'seed':0,'context_rows':256,
                        'n_estimators':1,'metrics':metrics(t,pred,'classification',1.),
                        'source_labels_changed':False,'generator_reads_query_labels':False,
                        'input_hashes':{k:protocol.array_sha256(v) for k,v in
                            [('X_source',X),('y_source',y),('X_target',shifted),('y_target',t)]}})
        print('CLASS_LOCATION_DATASET_COMPLETE',m['task_id'],flush=True)


def launch():
    def one(gpu):
        env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        with (RUN/'logs'/f'tabicl_class_location_v22_gpu{gpu}.log').open('a') as f:
            p=subprocess.run([str(RUN/'.venv/bin/python'),'-u',str(Path(__file__)),'worker',
                              '--shard',str(gpu),'--shards','8'],env=env,stdout=f,stderr=subprocess.STDOUT)
        r={'gpu':gpu,'exit_code':p.returncode};print(json.dumps(r),flush=True);return r
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,range(8)))
    atomic_json(OUT/'class_location_execution.json',rows)
    if any(r['exit_code'] for r in rows):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','launch','worker'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='declare':declare()
    elif a.phase=='launch':launch()
    else:worker(a.shard,a.shards)
