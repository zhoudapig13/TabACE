"""Specific-only local TFM experts on the v29 readmission supervision budget."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
from readmission_development_adaptation import DevelopmentWorker,NAME
from tune_readmission_focus import ReadmissionWorker


class LocalDevelopmentWorker(DevelopmentWorker):
    calibrated=ReadmissionWorker.calibrated
    mapped=ReadmissionWorker.mapped
    consider_expert=ReadmissionWorker.consider_expert
    parent_shared_oof=ReadmissionWorker.parent_shared_oof

    def __init__(self,gpu):
        super().__init__(gpu)
        self.parent_path=exp.ROOT/'results/readmission_development_adaptation_v29'/NAME/'frozen_selection.json'
        self.focus_parent=json.loads(self.parent_path.read_text())
        assert self.focus_parent['configurations']['Shared']['kind']=='five-context probability gate'
        with np.load(self.data/'readmission_development_predictions_v29.npz') as z:
            self.pp={k:z[k] for k in z.files}
        self.threshold=self.focus_parent['configurations']['Base']['threshold']
        self.shared=self.pp['Shared'].copy();self.shared_oof=self.parent_shared_oof()
        self.specific=self.pp['Specific'].copy();self.specific_score=self.score(self.specific,self.threshold)
        self.specific_config=dict(kind='Frozen v29 Specific',threshold=self.threshold,
                                  full_configuration=self.focus_parent['configurations']['Specific'])
        self.standalone=self.pp['Auxiliary'].copy();self.standalone_config=self.focus_parent['configurations']['Auxiliary']
        self.standalone_best=self.score(self.standalone,self.standalone_config['threshold']);self.trials=[]

    def execute_local(self):
        x=np.c_[self.X,self.fit_domains];q=np.c_[self.queries,self.query_domains]
        target=self.fit_domains.astype(bool)
        weight=np.where(target,.3/target.sum(),.7/(~target).sum())*len(self.y)
        forest=RandomForestClassifier(n_estimators=64,max_depth=4,min_samples_leaf=8,n_jobs=1,random_state=0)
        forest.fit(x,self.y,sample_weight=weight);pl=forest.apply(x);ql=forest.apply(q)
        anchors=[];nearest=np.zeros(len(x));candidate=int(np.random.default_rng(0).integers(len(x)))
        for _ in range(4):
            anchors.append(candidate);nearest=np.maximum(nearest,(pl==pl[candidate]).mean(1))
            distance=1-nearest;distance[anchors]=-1;candidate=int(distance.argmax())
        al=pl[anchors];pool_sim=(pl[:,None,:]==al[None,:,:]).mean(2)
        query_sim=(ql[:,None,:]==al[None,:,:]).mean(2);groups=query_sim.argmax(1)
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        views=[np.arange(self.X.shape[1]),np.sort(np.argsort(-association,kind='stable')[:64])]
        for target_count in [48,96]:
            for view,cols in enumerate(views):
                prediction=np.empty(len(q));rowsets=[]
                for j in range(4):
                    source=np.argsort(-pool_sim[:self.ns,j],kind='stable')[:256-target_count]
                    target_candidates=np.arange(self.ns,self.ns+self.nt)
                    chosen=target_candidates[np.argsort(-pool_sim[target_candidates,j],kind='stable')[:target_count-len(self.mandatory)]]
                    rows=np.r_[source,chosen,self.mandatory]
                    for label in [0,1]:
                        shortage=32-int((self.y[rows]==label).sum())
                        if shortage<=0:continue
                        available=np.flatnonzero((self.y[:self.ns]==label)&~np.isin(np.arange(self.ns),rows))
                        take=available[np.argsort(-pool_sim[available,j],kind='stable')[:shortage]]
                        replace=np.flatnonzero((rows<self.ns)&(self.y[rows]!=label))
                        replace=replace[np.argsort(pool_sim[rows[replace],j],kind='stable')[:len(take)]]
                        rows[replace]=take
                    assert len(rows)==len(np.unique(rows))==256 and set(self.mandatory).issubset(rows)
                    qi=np.flatnonzero(groups==j);rowsets.append(rows.tolist())
                    if len(qi):prediction[qi]=self.predict(rows,cols)[qi]
                self.consider_expert(prediction,dict(expert='forest-guided local Native-8 TFM',
                    target_context_rows=target_count,source_context_rows=256-target_count,
                    view=view,columns=cols.tolist(),anchor_count=4,anchors=anchors,
                    context_rows=rowsets,context_size=256,forest_parameters=forest.get_params(),extra_label_budget=0))
                print('READMISSION_NEW_POOL_LOCAL',target_count,view,
                      dict(Specific=self.specific_score,standalone=self.standalone_best),flush=True)
        output=dict(Base=self.pp['Base'],Shared=self.shared,Specific=self.specific,
                    Auxiliary=self.standalone,Mean5=self.pp['Mean5'])
        assert np.array_equal(output['Base'],self.pp['Base']) and np.array_equal(output['Shared'],self.pp['Shared'])
        scores={m:self.score(p,self.standalone_config['threshold'] if m=='Auxiliary' else
                  (self.focus_parent['configurations']['Mean5']['threshold'] if m=='Mean5' else self.threshold))
                for m,p in output.items()}
        report=dict(self.focus_parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_labels_opened=False,
            development_accuracy=scores,local_specific_trials=self.trials,
            configurations=dict(self.focus_parent['configurations'],Specific=self.specific_config,Auxiliary=self.standalone_config),
            base_shared_predictions_bitwise_unchanged=True,parent_snapshot=str(self.parent_path.relative_to(exp.ROOT)),
            parent_sha256=hashlib.sha256(self.parent_path.read_bytes()).hexdigest(),new_local_context_configurations=4,
            context_rows_exactly256=True,target_fitting_budget_unchanged=192,all_other_tasks_unchanged=True,
            focus_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        exp.write_json(exp.ROOT/'results/readmission_local_development_v30/frozen_selection.json',report)
        np.savez_compressed(self.data/'readmission_local_development_predictions_v30.npz',**output)
        print('READMISSION_LOCAL_FROZEN',scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):LocalDevelopmentWorker(a.gpu).execute_local()
