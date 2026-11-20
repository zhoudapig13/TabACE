"""Fresh-patient Specific development, with fixed stronger preceding controls."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
from tune_targeted_medical import TargetedWorker
from tune_targeted_medical_paired import PairedWorker
from tune_readmission_focus import ReadmissionWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES
from develop_specific_residual import new_learner
from prepare_readmission_fresh import NAME,DEST


class FreshWorker(TargetedWorker):
    calibrated=ReadmissionWorker.calibrated
    mapped=ReadmissionWorker.mapped
    consider_expert=ReadmissionWorker.consider_expert

    def __init__(self,gpu):
        super().__init__('readmission_other_to_ER','ood',gpu,data_root=DEST.parent)
        self.previous=json.loads((exp.ROOT/'results/medical_test_assisted_paired_v27'/NAME/'frozen_selection.json').read_text())
        self.old=PairedWorker(self.task,self.domain,gpu)
        self.cache=self.data/'member_cache_native8_fixed_evaluation_batch';self.cache.mkdir(exist_ok=True)
        self.threshold=self.previous['configurations']['Base']['threshold']
        self.pooling_strengths=[.2,.4,.6,.8];self.residual_caps=[.4,.8]
        self.trials=[];self.standalone_best=-1;self.standalone=None;self.standalone_config=None

    def predict(self,rows,columns=None):
        # TabICL can depend on the other unlabelled queries in its inference
        # batch. Preserve the entire previous reference batches for evaluation;
        # fresh calibration/validation queries are predicted in a separate pass.
        columns=np.arange(self.X.shape[1]) if columns is None else columns
        context=self.X[rows][:,columns]
        key=hashlib.sha256(context.tobytes()+self.y[rows].tobytes()+columns.tobytes()+
            self.query_hash.encode()+exp.array_hash(self.old.queries).encode()+b'fixed-reference-batches-native8').hexdigest()
        path=self.cache/(key+'.npz')
        if path.exists():
            with np.load(path) as z:return z['p']
        root=getattr(getattr(self.model,'model_',None),'icl_predictor',None)
        if root is not None and hasattr(root,'root'):root.root=None
        self.model.fit(context,self.y[rows]);self.model._load_model=lambda:None
        def forward(q):
            return np.concatenate([self.model.predict_proba(q[a:a+1024][:,columns])[:,1]
                                   for a in range(0,len(q),1024)])
        reference=forward(self.old.queries);fresh=forward(self.queries)
        cut=self.nsc+32
        result=np.r_[reference[:cut],fresh[cut:cut+64],reference[cut:self.old.ncal],
                     fresh[self.dev],reference[self.old.evaluation]]
        assert len(result)==len(self.queries) and np.isfinite(result).all()
        np.savez_compressed(path,p=result,rows=rows,columns=columns,
            evaluation_reference_batches_preserved=True)
        print('FIXED_BATCH_TFM',len(rows),len(columns),flush=True)
        return result

    def map_rows(self,rows):
        new=rows.copy();mask=new>=self.old.ns+self.old.nt
        new[mask]+=self.nt-self.old.nt
        assert np.array_equal(self.X[new],self.old.X[rows]) and np.array_equal(self.y[new],self.old.y[rows])
        return new

    def anchors(self):
        conf=self.previous['configurations']['Base']
        rows=self.old.rows(conf['use_original_target_context'],np.ones(len(self.old.y)),0)
        self.base=self.predict(self.map_rows(rows))
        shared_conf=self.previous['configurations']['Shared'];bank=[]
        for i,score in enumerate(self.old.old.scores):
            rank=np.argsort(np.argsort(score,kind='stable'),kind='stable')/max(1,len(score)-1)
            r=self.old.old.rows(float(shared_conf['use_original_target_context']),
                                np.exp(-6*shared_conf['sampling_strength']*rank),10+i)
            r=self.map_rows(self.old.pair(r,10+i))
            bank.append(self.predict(r,self.old.subspace if i==3 else None))
        bank=np.asarray(bank);exp.features=lambda b,k,m:floor_features(b,k,m,.25)
        try:full=exp.probability(self.base,bank,shared_conf['policy'])
        finally:exp.features=ORIGINAL_FEATURES
        cap=shared_conf['update_cap'];self.shared=(1-cap)*self.base+cap*full
        with np.load(self.old.data/'paired_frozen_predictions_v27.npz') as z:old={k:z[k] for k in z.files}
        old_eval=slice(*self.previous['query_layout']['evaluation'])
        self.anchor_audit={}
        for m,p in [('Base',self.base),('Shared',self.shared)]:
            before=old[m][old_eval];actual=p[self.evaluation]
            delta=float(np.max(np.abs(actual-before)))
            mismatches=int(np.count_nonzero((actual>=self.threshold)!=(before>=self.threshold)))
            self.anchor_audit[m]=dict(max_probability_difference=delta,classification_disagreements=mismatches)
            print('ANCHOR_CONSISTENCY',m,self.anchor_audit[m],flush=True)
            assert delta<1e-5 and mismatches==0
        with np.load(self.old.data/'readmission_focused_calibration_oof_v28.npz') as z:old_oof=z['Shared']
        cut=self.nsc+32
        self.shared_oof=np.r_[old_oof[:cut],self.shared[cut:cut+64],old_oof[cut:]]
        assert len(self.shared_oof)==self.ncal
        self.specific=self.shared.copy();self.specific_score=self.score(self.shared,self.threshold)
        self.specific_config=dict(kind='Fixed preceding Shared fallback',threshold=self.threshold)
        # Reconstruct the previously selected Specific, then assess it on the
        # fresh validation cohort rather than its repeatedly reused old cohort.
        pc=self.previous['configurations']['Specific'];assert pc['kind']=='constant logit pool'
        model=new_learner(pc['learner']);d=self.old.fit_domains.astype(bool);mass=pc['target_training_mass']
        weight=np.where(d,mass/d.sum(),(1-mass)/(~d).sum())*len(d)
        model.fit(np.c_[self.old.X,self.old.fit_domains],self.old.y,sample_weight=weight)
        raw=model.predict_proba(np.c_[self.queries,self.query_domains])[:,1]
        teacher=expit(pc['platt_a']*logit(np.clip(raw,1e-6,1-1e-6))+pc['platt_b'])
        centered=logit(np.clip(self.shared,1e-6,1-1e-6))-logit(self.threshold)
        expert=logit(np.clip(teacher,1e-6,1-1e-6))-logit(pc['expert_threshold'])
        prior=expit((1-pc['alpha'])*centered+pc['alpha']*expert+logit(self.threshold))
        assert np.max(np.abs(prior[self.evaluation]-old['Specific'][old_eval]))<1e-5
        value=self.score(prior,self.threshold)
        if value>self.specific_score+1e-12:
            self.specific=prior;self.specific_score=value
            self.specific_config=dict(kind='Frozen v27 Specific',threshold=self.threshold,full_configuration=pc)
        print('FRESH_ANCHORS',dict(Base=self.score(self.base,self.threshold),Shared=self.score(self.shared,self.threshold),
                                   prior_Specific=value,selected_Specific=self.specific_score),self.anchor_audit,flush=True)

    def execute(self):
        self.anchors();target=self.fit_domains.astype(bool)
        x=np.c_[self.X,self.fit_domains];q=np.c_[self.queries,self.query_domains]
        specs=[dict(kind='logistic',C=.03),dict(kind='logistic',C=.3),dict(kind='forest',depth=4),
               dict(kind='forest',depth=8),dict(kind='lightgbm',depth=3),dict(kind='lightgbm',depth=4)]
        for i,spec in enumerate(specs):
            for mass in [.5,1.]:
                selected=target if mass==1 else np.ones(len(target),dtype=bool)
                weight=np.where(target,mass/target.sum(),(1-mass)/(~target).sum())*selected.sum()
                model=new_learner(spec);model.fit(x[selected],self.y[selected],sample_weight=weight[selected])
                self.consider_expert(model.predict_proba(q)[:,1],dict(expert='fresh-target supervised learner',
                    learner=spec,target_training_mass=mass,target_training_patients=304,
                    fresh_target_fit=256,fresh_target_calibration=64))
                print('FRESH_SUPERVISED',i,mass,dict(Specific=self.specific_score,standalone=self.standalone_best),flush=True)
        targets=np.flatnonzero(target);optional=np.setdiff1d(targets,self.mandatory)
        xc=self.X[targets]-self.X[targets].mean(0);yc=self.y[targets]-self.y[targets].mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        views=[np.arange(self.X.shape[1]),np.sort(np.argsort(-association,kind='stable')[:64])]
        for source_count in [0,64]:
            for view,columns in enumerate(views):
                members=[];rowsets=[]
                for seed in [0,1,2]:
                    rng=np.random.default_rng(seed);rows=list(self.mandatory)
                    for label in [0,1]:
                        if source_count:rows.extend(rng.choice(np.flatnonzero(self.y[:self.ns]==label),source_count//2,replace=False))
                        choices=optional[self.y[optional]==label]
                        rows.extend(rng.choice(choices,(256-source_count-16)//2,replace=False))
                    rows=np.asarray(rows,dtype=int);rng.shuffle(rows)
                    assert len(rows)==len(np.unique(rows))==256
                    members.append(self.predict(rows,columns));rowsets.append(rows.tolist())
                self.consider_expert(np.asarray(members).mean(0),dict(expert='fresh-target Native-8 context ensemble',
                    source_context_rows=source_count,target_context_rows=256-source_count,context_rows=rowsets,
                    view=view,columns=columns.tolist(),members=3,native_estimators_per_member=8,experiment_seed=0))
                print('FRESH_TFM',source_count,view,dict(Specific=self.specific_score,standalone=self.standalone_best),flush=True)
        output=dict(Base=self.base,Shared=self.shared,Specific=self.specific,Auxiliary=self.standalone)
        scores={m:self.score(p,self.standalone_config['threshold'] if m=='Auxiliary' else self.threshold) for m,p in output.items()}
        report=dict(task=self.task,domain='ood',data=str(self.data.relative_to(exp.ROOT)),metadata=self.meta,
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_labels_opened=False,
            query_layout=dict(calibration=[0,self.ncal],development=[self.dev.start,self.dev.stop],
                              evaluation=[self.evaluation.start,self.evaluation.stop]),
            configurations=dict(Base=self.previous['configurations']['Base'],Shared=self.previous['configurations']['Shared'],
                Specific=self.specific_config,Auxiliary=self.standalone_config),development_accuracy=scores,trials=self.trials,
            anchors_match_preceding_models=self.anchor_audit,original780_evaluation_patients_preserved=True,
            fresh_target_fit_labels=256,fresh_target_calibration_labels=64,fresh_validation_labels=512,
            specific_target_fitting_labels=416,fixed_controls_target_fitting_labels=96,
            supervision_and_compute_matched=False,all_other_tasks_frozen=True,context_rows_exactly256=True,
            experiment_seed=0,previous_evaluation_cohorts_observed=True,
            method_status='Extra-target-supervised component development; not original TabACE or a matched-budget rebuttal comparison',
            original_evaluation_query_batches_preserved=True,fresh_queries_predicted_in_separate_pass=True,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        exp.write_json(exp.ROOT/'results/readmission_fresh_specialist_v32/frozen_selection.json',report)
        np.savez_compressed(self.data/'fresh_specialist_predictions_v32.npz',**output)
        print('FRESH_READMISSION_FROZEN',scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):FreshWorker(a.gpu).execute()
