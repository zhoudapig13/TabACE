"""Specific-C: forest-guided retrieval with frozen TFM expert predictions."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from scipy.special import logit,expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from develop_specific_residual import ResidualWorker,calibrate,fit_residual,apply_residual


class ContextWorker(ResidualWorker):
    def retrieve(self,pool_leaves,query_leaves,anchors,size,domain):
        anchor_leaves=pool_leaves[anchors]
        sim_pool=(pool_leaves[:,None,:]==anchor_leaves[None,:,:]).mean(2)
        sim_query=(query_leaves[:,None,:]==anchor_leaves[None,:,:]).mean(2)
        group=sim_query.argmax(1);prediction=np.empty(len(self.queries));rowsets=[]
        for j in range(len(anchors)):
            qi=np.flatnonzero(group==j)
            target_n=min(32,size//4) if domain=='ood' else 0
            source_n=min(size-target_n,self.ns)
            source=np.argsort(-sim_pool[:self.ns,j],kind='stable')[:source_n]
            target=self.ns+np.argsort(-sim_pool[self.ns:,j],kind='stable')[:target_n]
            rows=np.r_[source,target]
            minimum=max(1,len(rows)//8)
            for cls in [0,1]:
                need=minimum-int((self.y[rows]==cls).sum())
                if need<=0:continue
                available=np.flatnonzero((self.y==cls)&~np.isin(np.arange(len(self.y)),rows))
                if domain=='id':available=available[available<self.ns]
                take=available[np.argsort(-sim_pool[available,j],kind='stable')[:need]]
                replace=np.flatnonzero(self.y[rows]!=cls)
                replace=replace[np.argsort(sim_pool[rows[replace],j],kind='stable')[:len(take)]]
                rows[replace]=take
            assert len(np.unique(rows))==len(rows) and len(rows)<=256 and len(np.unique(self.y[rows]))==2
            rowsets.append(rows.tolist())
            if not len(qi):continue
            key=hashlib.sha256(rows.tobytes()+qi.tobytes()).hexdigest()
            cache=self.context_cache/(key+'.npz')
            if cache.exists():
                with np.load(cache) as z:prediction[qi]=z['p']
                continue
            predictor=getattr(getattr(self.model,'model_',None),'icl_predictor',None)
            if predictor is not None and hasattr(predictor,'root'):predictor.root=None
            self.model.fit(self.X[rows],self.y[rows]);self.model._load_model=lambda:None
            p=np.r_[tuple(self.model.predict_proba(self.queries[qi[start:start+1024]])[:,1]
                          for start in range(0,len(qi),1024))]
            prediction[qi]=p
            temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,p=p,rows=rows,query_indices=qi);temp.replace(cache)
        return prediction,rowsets

    def run_context(self):
        parent=json.loads((main.ROOT/'results/expanded_medical_probability_v18'/self.task/'frozen_selection.json').read_text())
        with np.load(main.DATA/self.task/'expanded_probability_predictions.npz') as z:old={k:z[k] for k in z.files}
        ncal=len(self.cal_y);qdomain=np.zeros(len(self.queries));qdomain[self.n_source_cal:ncal]=1
        qdomain[self.slices['validation']]=1;qdomain[self.slices['ood']]=1
        x=np.c_[self.X,np.r_[np.zeros(self.ns),np.ones(self.nt)]];q=np.c_[self.queries,qdomain]
        forest=RandomForestClassifier(n_estimators=64,max_depth=4,min_samples_leaf=8,n_jobs=1,random_state=0)
        weight=np.r_[np.full(self.ns,.8/self.ns),np.full(self.nt,.2/self.nt)]*len(self.y)
        forest.fit(x,self.y,sample_weight=weight)
        pl=forest.apply(x);ql=forest.apply(q)
        anchors=[];near=np.zeros(len(x));candidate=int(np.random.default_rng(0).integers(len(x)))
        for _ in range(16):
            anchors.append(candidate)
            similarity=(pl==pl[candidate]).mean(1);near=np.maximum(near,similarity)
            dissimilarity=1-near;dissimilarity[anchors]=-1;candidate=int(dissimilarity.argmax())
        self.context_cache=main.DATA/self.task/'forest_context_cache_v21';self.context_cache.mkdir(exist_ok=True)
        self.cache_dir=main.DATA/self.task/'member_cache_known_domain_raw'
        reference={d:self.shared_oof(d,parent,old) for d in ['id','ood']}
        predictions={};configs={};scores={m:{} for m in ['Base','Shared','Specific_C']};trials=[]
        for domain in ['id','ood']:
            shared=old[domain+'_Shared'];threshold=parent['configurations'][domain]['Base']['policy']['threshold']
            best=shared;best_conf=dict(kind='Shared fallback',threshold=threshold)
            best_score=self.domain_score(best,threshold,domain)
            for nanchors in [8,16]:
                for size in [64,128,256]:
                    raw,rowsets=self.retrieve(pl,ql,anchors[:nanchors],size,domain)
                    rawlogit=logit(np.clip(raw,1e-6,1-1e-6));choices=[]
                    strata=self.cal_y+2*(np.arange(ncal)>=self.n_source_cal)
                    for share in ([0.,.25] if domain=='id' else [.5,.75]):
                        weights=self.fit_weights(share);oof=np.zeros(ncal)
                        for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),strata):
                            a,b=calibrate(rawlogit[fit],self.cal_y[fit],weights[fit]);oof[val]=expit(a*rawlogit[val]+b)
                        a,b=calibrate(rawlogit[:ncal],self.cal_y,weights);expert=expit(a*rawlogit+b)
                        tau=exp.threshold(oof,self.cal_y,weights)
                        s=logit(np.clip(reference[domain],1e-6,1-1e-6))-logit(threshold)
                        t=logit(np.clip(oof,1e-6,1-1e-6))-logit(tau)
                        for cap in [.15,.25]:
                            for reg in [.1,.01]:
                                gate=fit_residual(s,t,self.cal_y,weights,cap,reg)
                                pred=apply_residual(shared,expert,threshold,tau,gate)
                                value=self.domain_score(pred,threshold,domain)
                                choices.append(dict(target_calibration_share=share,alpha_cap=cap,
                                                    regularization=reg,development_accuracy=value))
                                if value>best_score+1e-12:
                                    best=pred;best_score=value;best_conf=dict(threshold=threshold,
                                        anchors=anchors[:nanchors],context_rows=rowsets,context_size=size,
                                        platt_a=float(a),platt_b=float(b),expert_threshold=tau,gate=gate,
                                        target_calibration_share=share,forest_parameters=forest.get_params(),
                                        final_expert_is_frozen_tfm=True)
                    trials.append(dict(domain=domain,anchor_count=nanchors,context_size=size,choices=choices))
                    print('SPECIFIC_CONTEXT_TRIAL',self.task,domain,nanchors,size,best_score,flush=True)
            configs[domain]=best_conf
            for m,p in [('Base',old[domain+'_Base']),('Shared',shared),('Specific_C',best)]:
                predictions[domain+'_'+m]=p;scores[m][domain]=self.domain_score(p,threshold,domain)
        report=dict(parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
            parent_snapshot='results/expanded_medical_probability_v18/'+self.task+'/frozen_selection.json',
            base_shared_unchanged=True,context_specific_configurations=configs,development_accuracy=scores,
            specific_context_trials=trials,protocol=json.loads((main.ROOT/'configs/medical_specific_context_v21.json').read_text()),
            supervised_retrieval_explicit=True,tfm_backbone_frozen=True,
            method_status='Specific-C DAGGER supervised forest-guided context retrieval; not archived original Specific')
        exp.write_json(main.ROOT/'results/medical_specific_context_v21'/self.task/'frozen_selection.json',report)
        np.savez_compressed(main.DATA/self.task/'specific_context_predictions.npz',**predictions)
        print('SPECIFIC_CONTEXT_FROZEN',self.task,scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):ContextWorker(args.task,args.gpu,'raw').run_context()
