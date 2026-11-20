"""Development-only context/fusion search; final test labels opened once after freezing.

These dagger variants use extra labeled target-development patients and are NOT
the paper's source-only medical-transfer configuration. All three methods have
the same source and target labeled pools and calibration information.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import torch
from tabicl import TabICLClassifier
import run_public_shift as exp

ROOT=exp.ROOT
DATA=ROOT/'data/processed/target_development_v2'
OUT=ROOT/'results/target_development_v2'
PLAN_PATH=ROOT/'configs/target_development_v2.json'
EVALUATE_TEST=True


def select_rows(y, n, weights, seed):
    rng=np.random.default_rng(seed)
    n=min(n,len(y))
    if n==0:return np.array([],int)
    positive=int(np.clip(round(n*y.mean()),1,max(1,n-1)))
    ids=[]
    for label,count in [(0,n-positive),(1,positive)]:
        available=np.flatnonzero(y==label)
        count=min(count,len(available))
        p=np.maximum(weights[available],1e-9);p/=p.sum()
        ids.extend(rng.choice(available,count,replace=False,p=p).tolist())
    return np.array(ids,int)


class Worker:
    def __init__(self,task,gpu,context_budget=None,target_calibration_budget=None):
        self.task=task;self.meta=json.loads((DATA/task/'metadata.json').read_text())
        with np.load(DATA/task/'inputs.npz') as z:self.pack={k:z[k] for k in z.files}
        assert {k:exp.array_hash(v) for k,v in self.pack.items()}==self.meta['input_hashes']
        p=self.pack
        context,cal=train_test_split(np.arange(len(p['y_dev_fit'])),test_size=1/3,random_state=0,stratify=p['y_dev_fit'])
        if context_budget is not None and len(context)>context_budget:
            context,_=train_test_split(context,train_size=context_budget,random_state=0,stratify=p['y_dev_fit'][context])
        if target_calibration_budget is not None and len(cal)>target_calibration_budget:
            cal,_=train_test_split(cal,train_size=target_calibration_budget,random_state=0,stratify=p['y_dev_fit'][cal])
        self.target_context_indices=context
        self.target_calibration_indices=cal
        assert not set(context).intersection(cal)
        self.X=np.r_[p['X_train'],p['X_dev_fit'][context]]
        self.y=np.r_[p['y_train'],p['y_dev_fit'][context]]
        self.ns=len(p['y_train']);self.nt=len(context)
        self.cal_y=np.r_[p['y_cal'],p['y_dev_fit'][cal]]
        self.n_source_cal=len(p['y_cal']);self.n_target_cal=len(cal)
        self.val_y=p['y_dev_val']
        self.sizes=[len(self.cal_y),len(self.val_y),len(p['X_id']),len(p['X_ood'])]
        self.queries=np.r_[p['X_cal'],p['X_dev_fit'][cal],p['X_dev_val'],p['X_id'],p['X_ood']]
        self.slices={name:slice(start,end) for name,start,end in zip(
            ['cal','validation','id','ood'],np.cumsum([0]+self.sizes[:-1]),np.cumsum(self.sizes))}
        self.A,self.B,_,_=exp.geometry(self.X,p['X_ood'])
        targetdist=NearestNeighbors(n_neighbors=min(10,len(self.B))).fit(self.B).kneighbors(self.A)[0].mean(1)
        outlier=np.minimum(self.A*self.A,100).mean(1)
        neighbors=NearestNeighbors(n_neighbors=11).fit(self.A).kneighbors(self.A,return_distance=False)
        neighbors=np.stack([x[x!=i][:10] for i,x in enumerate(neighbors)])
        vote=self.y[neighbors].mean(1)
        # TRAINING labels alone compute local inconsistency; calibration rows are absent.
        unstable=np.abs(self.y-vote)
        self.scores=[targetdist,targetdist+.35*outlier,unstable+.1*targetdist,
                     np.zeros(len(self.y)),unstable+.2*outlier]
        self.match=targetdist
        domain=LogisticRegression(C=.1,max_iter=300,random_state=0)
        domain.fit(np.r_[self.A,self.B],np.r_[np.zeros(len(self.A)),np.ones(len(self.B))])
        probability=np.clip(domain.predict_proba(self.A)[:,1],1e-6,1-1e-6)
        self.ratio=np.minimum(probability/(1-probability)*len(self.A)/len(self.B),10)**.5
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        self.subspace=np.flatnonzero(np.arange(self.X.shape[1])!=association.argmax())
        self.model=TabICLClassifier(n_estimators=1,device=f'cuda:{gpu}',model_path=str(exp.CHECKPOINT),
            allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0)
        cache_name='member_cache' if context_budget is None and target_calibration_budget is None else f'member_cache_limited_{context_budget}_{target_calibration_budget}'
        self.cache_dir=DATA/task/cache_name;self.cache_dir.mkdir(exist_ok=True)
        self.predict_count=0

    def weights(self,target_share):
        return np.r_[np.full(self.n_source_cal,(1-target_share)/self.n_source_cal),
                     np.full(self.n_target_cal,target_share/self.n_target_cal)]

    def rows(self,fraction,weights,seed):
        target_n=min(round(256*fraction),self.nt)
        source_n=min(256-target_n,self.ns)
        a=select_rows(self.y[:self.ns],source_n,weights[:self.ns],seed)
        b=select_rows(self.y[self.ns:],target_n,weights[self.ns:],seed+100)+self.ns
        rows=np.r_[a,b];np.random.default_rng(seed).shuffle(rows)
        assert len(rows)<=256 and len(np.unique(rows))==len(rows)
        return rows

    def predict(self,rows,columns=None,align=0):
        columns=np.arange(self.X.shape[1]) if columns is None else columns
        context=self.X[rows].copy()
        if align:
            for col in self.pack['align_columns']:
                ms,mt=np.median(self.X[:self.ns,col]),np.median(self.pack['X_ood'][:,col])
                ss=max(float(np.subtract(*np.quantile(self.X[:self.ns,col],[.75,.25]))),1)
                st=max(float(np.subtract(*np.quantile(self.pack['X_ood'][:,col],[.75,.25]))),1)
                # Only SOURCE rows are transported. Real target development rows are unchanged.
                mask=rows<self.ns
                context[mask,col]=(1-align)*context[mask,col]+align*((context[mask,col]-ms)*st/ss+mt)
        context=context[:,columns]
        key=hashlib.sha256(context.tobytes()+self.y[rows].tobytes()+columns.tobytes()).hexdigest()
        path=self.cache_dir/(key+'.npz')
        if path.exists():
            with np.load(path) as z:return z['p']
        root=getattr(getattr(self.model,'model_',None),'icl_predictor',None)
        if root is not None and hasattr(root,'root'):root.root=None
        self.model.fit(context,self.y[rows])
        self.model._load_model=lambda:None
        probabilities=[]
        for begin in range(0,len(self.queries),1024):
            probabilities.append(self.model.predict_proba(self.queries[begin:begin+1024][:,columns])[:,1])
        probability=np.concatenate(probabilities)
        assert np.isfinite(probability).all()
        np.savez_compressed(path,p=probability,rows=rows,columns=columns,alignment=align)
        self.predict_count+=1
        return probability

    def bank(self,fraction,strength):
        predictions=[]
        for i,score in enumerate(self.scores):
            rank=np.argsort(np.argsort(score,kind='stable'),kind='stable')/max(1,len(score)-1)
            weight=np.exp(-6*strength*rank)
            rows=self.rows(fraction,weight,10+i)
            predictions.append(self.predict(rows,self.subspace if i==3 else None))
        return np.asarray(predictions)

    def extras(self,fraction,alignment):
        rank=np.argsort(np.argsort(self.match,kind='stable'),kind='stable')/max(1,len(self.match)-1)
        a=self.rows(fraction,np.exp(-6*rank),101)
        b=self.rows(fraction,self.ratio,102)
        c=self.rows(fraction,np.ones(len(self.y)),103)
        return np.asarray([self.predict(a),self.predict(b),self.predict(c,align=alignment)])

    def score(self,p,threshold):
        return float(np.mean((p[self.slices['validation']]>=threshold)==self.val_y))

    def fuse(self,anchor,bank,fallback,fallback_configuration):
        best_probability=anchor
        best_configuration=fallback_configuration
        best_score=self.score(anchor,fallback['threshold'])
        trials=[]
        for gain in [1.,3.]:
            for regularization in [.01,.1]:
                for target_share in [.25,.5,1.]:
                    weight=self.weights(target_share)
                    policy=exp.fit_gate(anchor[self.slices['cal']],bank[:,self.slices['cal']],
                                        self.cal_y,weight,len(bank),gain,regularization)
                    probability=exp.probability(anchor,bank,policy)
                    t=exp.threshold(probability[self.slices['cal']],self.cal_y,weight)
                    policy['threshold']=t
                    score=self.score(probability,t)
                    trials.append(dict(gain=gain,regularization=regularization,target_share=target_share,
                                       threshold=t,validation_accuracy=score))
                    if score>best_score+1e-12:
                        best_score=score;best_probability=probability
                        best_configuration=dict(policy=policy,target_calibration_share=target_share)
        return best_probability,best_configuration,best_score,trials

    def run(self):
        dest=OUT/self.task;dest.mkdir(parents=True,exist_ok=True)
        best_score=-1;base_p=None;base_configuration=None
        base_trials=[]
        for fraction in [0.,.25,.5,1.]:
            rows=self.rows(fraction,np.ones(len(self.y)),0)
            p=self.predict(rows)
            for share in [.25,.5,1.]:
                t=exp.threshold(p[self.slices['cal']],self.cal_y,self.weights(share))
                score=self.score(p,t)
                base_trials.append(dict(target_fraction=fraction,target_calibration_share=share,
                                        threshold=t,validation_accuracy=score))
                if score>best_score+1e-12:
                    best_score=score;base_p=p;base_configuration=dict(
                        policy=dict(kind='base',threshold=t),target_fraction=fraction,target_calibration_share=share)
        print('DEVELOPMENT_BASE',self.task,best_score,flush=True)
        chosen_shared=base_p;shared_config=base_configuration.copy();shared_score=best_score;chosen_bank=None
        shared_trials=[]
        for fraction in [0.,.25,.5,1.]:
            for strength in [0.,.5,1.]:
                bank=self.bank(fraction,strength)
                p,config,score,trials=self.fuse(base_p,bank,base_configuration['policy'],base_configuration)
                shared_trials.append(dict(target_fraction=fraction,sampling_strength=strength,fusion_trials=trials))
                if score>shared_score+1e-12:
                    chosen_shared=p;shared_config=dict(config,context_fraction=fraction,sampling_strength=strength)
                    shared_score=score;chosen_bank=bank
                print('DEVELOPMENT_SHARED_TRIAL',self.task,fraction,strength,round(score,6),flush=True)
        if chosen_bank is None:
            # This bank is used only in later candidates; selected Shared remains the Base fallback.
            chosen_bank=self.bank(base_configuration['target_fraction'],.5)
        shared_config.setdefault('policy',base_configuration['policy'])
        chosen_specific=chosen_shared;specific_config=shared_config.copy();specific_score=shared_score
        specific_trials=[]
        for fraction in [0.,.25,.5,1.]:
            for alignment in [0.,.25,.5]:
                extras=self.extras(fraction,alignment)
                bank=np.r_[chosen_bank,extras]
                p,config,score,trials=self.fuse(chosen_shared,bank,shared_config['policy'],shared_config)
                specific_trials.append(dict(target_fraction=fraction,alignment=alignment,fusion_trials=trials))
                if score>specific_score+1e-12:
                    chosen_specific=p;specific_config=dict(config,context_fraction=fraction,alignment=alignment,
                                                          anchor='selected Shared')
                    specific_score=score
                print('DEVELOPMENT_SPECIFIC_TRIAL',self.task,fraction,alignment,round(score,6),flush=True)
        configurations=dict(Base=base_configuration,Shared=shared_config,Specific=specific_config)
        predictions=dict(Base=base_p,Shared=chosen_shared,Specific=chosen_specific)
        development_scores=dict(Base=best_score,Shared=shared_score,Specific=specific_score)
        exp.write_json(dest/'frozen_development_selection.json',dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            configurations=configurations,development_validation_accuracy=development_scores,
            protocol=json.loads(PLAN_PATH.read_text()),
            metadata=self.meta,extra_target_labels=True,source_and_target_label_access_matched=True,
            trials=dict(Base=base_trials,Shared=shared_trials,Specific=specific_trials),
            selected_by='target-development validation Acc; test outcomes never used'))
        np.savez_compressed(DATA/self.task/'selected_predictions.npz',**predictions)
        print('FINAL_DEVELOPMENT_SELECTED',self.task,development_scores,flush=True)
        if not EVALUATE_TEST:
            print('FINAL_TEST_LABELS_NOT_OPENED',self.task,flush=True)
            return
        # First opening of previous test truth during this NEW tuning attempt.
        original=ROOT/self.meta['original_inputs']
        with np.load(original/'test_truth.npz') as z:truth={k:z[k] for k in z.files}
        result=dict(task=self.task,seed=0,extra_target_labels=True,development_validation_accuracy=development_scores,
                    method_status='DAGGER: target-supervised medical development variants; frozen backbone',domains={})
        for domain in ['id','ood']:
            y=truth['y_'+domain]
            scores={m:exp.metrics(y,p[self.slices[domain]],configurations[m]['policy']['threshold'])
                    for m,p in predictions.items()}
            tests={b+'_minus_'+a:exp.paired(y,
                   predictions[a][self.slices[domain]]>=configurations[a]['policy']['threshold'],
                   predictions[b][self.slices[domain]]>=configurations[b]['policy']['threshold'])
                   for a,b in [('Base','Shared'),('Shared','Specific')]}
            result['domains'][domain]=dict(n=len(y),positives=int(y.sum()),methods=scores,adjacent_comparisons=tests)
        exp.write_json(dest/'metrics.json',result)
        print('FINAL_OOD',self.task,{m:s['accuracy'] for m,s in result['domains']['ood']['methods'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    Worker(args.task,args.gpu).run()
