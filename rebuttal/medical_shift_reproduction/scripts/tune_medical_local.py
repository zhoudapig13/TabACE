"""Specific development variants: query-local and reduced-feature contexts.

Frozen v2 Base and Shared are the anchors. Only development-validation outcomes
rank candidates; this stage writes NO final test metrics.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import torch
import run_public_shift as exp
from tune_medical_development import Worker,ROOT,DATA,select_rows
BASE_RESULTS=ROOT/'results/target_development_v2'
LOCAL_OUT=ROOT/'results/target_development_local_v3'


class LocalWorker(Worker):
    def local(self,columns,size):
        context_space,query_space,_,_=exp.geometry(self.X[:,columns],self.queries[:,columns])
        clusters=MiniBatchKMeans(n_clusters=8,random_state=0,n_init=3,batch_size=256)
        clusters.fit(context_space)
        assignment=clusters.predict(query_space)
        values=np.empty(len(self.queries));rowsets=[]
        for cluster in range(8):
            query_ids=np.flatnonzero(assignment==cluster)
            distance=((context_space-clusters.cluster_centers_[cluster])**2).sum(1)
            nt=min(size//2,self.nt);ns=min(size-nt,self.ns)
            rows=np.r_[np.argsort(distance[:self.ns],kind='stable')[:ns],
                       self.ns+np.argsort(distance[self.ns:],kind='stable')[:nt]]
            if len(np.unique(self.y[rows]))<2:
                opposite=np.flatnonzero(self.y!=self.y[rows[0]])
                rows[-1]=opposite[np.argmin(distance[opposite])]
            assert len(rows)<=256 and len(np.unique(rows))==len(rows)
            rowsets.append(rows.tolist())
            if not len(query_ids):continue
            key=hashlib.sha256(rows.tobytes()+columns.tobytes()+query_ids.tobytes()).hexdigest()
            cache=self.cache_dir/('local_'+key+'.npz')
            if cache.exists():
                with np.load(cache) as z:values[query_ids]=z['p']
                continue
            predictor=getattr(getattr(self.model,'model_',None),'icl_predictor',None)
            if predictor is not None and hasattr(predictor,'root'):predictor.root=None
            self.model.fit(self.X[rows][:,columns],self.y[rows]);self.model._load_model=lambda:None
            probability=[]
            for start in range(0,len(query_ids),1024):
                probability.append(self.model.predict_proba(self.queries[query_ids[start:start+1024]][:,columns])[:,1])
            probability=np.concatenate(probability);values[query_ids]=probability
            np.savez_compressed(cache,p=probability,rows=rows,query_ids=query_ids,columns=columns)
        return values,rowsets

    def run_local(self):
        old=BASE_RESULTS/self.task
        frozen=json.loads((old/'frozen_development_selection.json').read_text())
        configurations=frozen['configurations'];shared_conf=configurations['Shared']
        with np.load(DATA/self.task/'selected_predictions.npz') as z:
            base=z['Base'];shared=z['Shared'];previous_specific=z['Specific']
        shared_fraction=shared_conf.get('context_fraction',configurations['Base']['target_fraction'])
        shared_bank=self.bank(shared_fraction,shared_conf.get('sampling_strength',.5))
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-association,kind='stable')
        original=ROOT/self.meta['original_inputs']
        numeric=json.loads((original/'metadata.json').read_text())['numeric_features']
        if self.task.startswith('sepsis'):
            phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int)
        else:
            phenotype=np.sort(ranked[:32])
        columns_list=[np.sort(ranked[:32]),np.sort(ranked[:64]),np.sort(ranked[:128]),phenotype]
        best=previous_specific;best_conf=configurations['Specific'].copy()
        best_score=self.score(best,best_conf['policy']['threshold'])
        trials=[]
        for ci,columns in enumerate(columns_list):
            for size in [64,128,256]:
                local,rows=self.local(columns,size)
                fractions=[configurations['Base']['target_fraction'],shared_fraction]
                uniform=np.ones(len(self.y))
                branch2=self.predict(self.rows(fractions[0],uniform,201),columns)
                branch3=self.predict(self.rows(fractions[1],uniform,202),phenotype)
                bank=np.r_[shared_bank,np.array([local,branch2,branch3])]
                p,config,score,ft=self.fuse(shared,bank,shared_conf['policy'],shared_conf)
                trials.append(dict(feature_view=ci,feature_count=len(columns),local_context_size=size,
                                   local_context_rows=rows,fusion_trials=ft,validation_accuracy=score))
                if score>best_score+1e-12:
                    best=p;best_conf=dict(config,feature_view=ci,columns=columns.tolist(),
                                          local_context_size=size,anchor='v2 Shared',variant='query-local reduced-feature Specific')
                    best_score=score
                print('LOCAL_DEVELOPMENT_TRIAL',self.task,'view',ci,'context',size,'accuracy',score,flush=True)
        dest=LOCAL_OUT/self.task
        scores=dict(Base=frozen['development_validation_accuracy']['Base'],
                    Shared=frozen['development_validation_accuracy']['Shared'],Specific=best_score)
        exp.write_json(dest/'frozen_development_selection.json',dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            configurations=dict(Base=configurations['Base'],Shared=shared_conf,Specific=best_conf),
            development_validation_accuracy=scores,trials=trials,extra_target_labels=True,
            selected_by='Development-validation accuracy only',
            no_final_test_metrics_written=True,previously_observed_test_cohort=True,
            method_status='DAGGER additional query-local medical adaptation, not archived paper method'))
        np.savez_compressed(DATA/self.task/'selected_local_predictions.npz',Base=base,Shared=shared,Specific=best)
        print('LOCAL_DEVELOPMENT_SELECTED',self.task,scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    LocalWorker(args.task,args.gpu).run_local()
