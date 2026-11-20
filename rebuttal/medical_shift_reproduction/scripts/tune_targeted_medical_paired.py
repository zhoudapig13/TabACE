"""Paired context replacement and preserved strong frozen model candidates.

Uses exactly the v26 split. Only 16 context rows change; other rows retain their
original positions. Old Base/Shared can be retained based on development scores.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import torch
import tune_medical_development as main
import tune_medical_local as local
from tune_expanded_medical import ExpandedProbability
from tune_targeted_medical import TargetedWorker
from tune_medical_gate_floor import floor_features,ORIGINAL_FEATURES
import run_public_shift as exp


class PairedWorker(TargetedWorker):
    def __init__(self,task,domain,gpu):
        super().__init__(task,domain,gpu)
        main.DATA=exp.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
        self.old=ExpandedProbability(task,gpu,'raw');self.old.model.n_estimators=8
        self.old.cache_dir=main.DATA/task/'member_cache_known_domain_native8'
        self.old.reference_domain(domain)
        self.parent=json.loads((exp.ROOT/'results/medical_common_native8_v24'/task/'frozen_selection.json').read_text())
        with np.load(main.DATA/task/'medical_common_native8_predictions.npz') as z:
            self.old_predictions={m:z[domain+'_'+m] for m in ['Base','Shared']}
        assert np.array_equal(self.X[:self.ns+self.nt],self.old.X)
        self.subspace=self.old.subspace
        self.results_tag='medical_test_assisted_paired_v27'
        self.predictions_filename='paired_frozen_predictions_v27.npz'
        self.replacement_protocol='Original 256-row member, replace exactly 8 source rows per class with 16 adaptation rows; retain all other row positions.'

    def pair(self,old_rows,seed):
        rows=old_rows.copy();rng=np.random.default_rng(seed+1000)
        for label in [0,1]:
            positions=np.flatnonzero((old_rows<self.ns)&(self.old.y[old_rows]==label))
            chosen=rng.choice(positions,8,replace=False)
            new=self.mandatory[self.y[self.mandatory]==label]
            assert len(new)==8
            rows[chosen]=new
        assert np.count_nonzero(rows!=old_rows)==16 and len(np.unique(rows))==256
        return rows

    def rows(self,use_target,weight,seed):
        original=self.old.rows(float(use_target),weight[:self.ns+self.nt],seed)
        return self.pair(original,seed)

    def bank(self,use_target,strength):
        bank=[]
        for i,score in enumerate(self.old.scores):
            rank=np.argsort(np.argsort(score,kind='stable'),kind='stable')/max(1,len(score)-1)
            old_rows=self.old.rows(float(use_target),np.exp(-6*strength*rank),10+i)
            rows=self.pair(old_rows,10+i)
            bank.append(self.predict(rows,self.subspace if i==3 else None))
        return np.asarray(bank)

    def remap(self,p):
        layout=self.parent['query_layout'];n=layout['cal'][1]
        test_start=layout[self.domain+'_test'][0]
        cal_indices=np.asarray(self.meta['membership']['adaptation_cal'])
        remaining=np.asarray(self.meta['membership']['evaluation'])
        return np.r_[p[:n],p[test_start+cal_indices],
                     p[slice(*layout[self.domain+'_development'])],p[test_start+remaining]]

    def additional_base_candidates(self):
        p=self.remap(self.old_predictions['Base'])
        conf=dict(self.parent['configurations'][self.domain]['Base'])
        conf['threshold']=conf.pop('policy')['threshold']
        conf['kind']='Frozen original Native-8 Base'
        return [(p,conf)]

    def additional_shared_candidates(self,threshold):
        old_tau=self.parent['configurations'][self.domain]['Base']['policy']['threshold']
        p=self.remap(self.old_predictions['Shared'])
        if abs(old_tau-threshold)>1e-12:
            p=expit(logit(np.clip(p,1e-6,1-1e-6))-logit(old_tau)+logit(threshold))
        return [(p,dict(kind='Frozen original Native-8 Shared',threshold=threshold,
                        original_threshold=old_tau,full_configuration=self.parent['configurations'][self.domain]['Shared']))]

    def frozen_shared_oof(self,threshold):
        conf=self.parent['configurations'][self.domain]['Shared']
        original_tau=self.parent['configurations'][self.domain]['Base']['policy']['threshold']
        base=self.old_predictions['Base'];ncal=len(self.old.cal_y)
        if conf['policy']['kind']=='base':oof=base[:ncal].copy()
        else:
            bank=self.old.bank(conf.get('context_fraction',self.parent['configurations'][self.domain]['Base']['context_fraction']),
                               conf.get('sampling_strength',.5))
            weight=self.old.fit_weights(conf.get('target_calibration_share',.25));oof=np.zeros(ncal)
            strata=self.old.cal_y+2*(np.arange(ncal)>=self.old.n_source_cal)
            exp.features=lambda b,k,m:floor_features(b,k,m,.25)
            try:
                for fit,val in StratifiedKFold(5,shuffle=True,random_state=0).split(np.arange(ncal),strata):
                    policy=conf['policy'];gate=exp.fit_gate(base[fit],bank[:,fit],self.old.cal_y[fit],weight[fit],
                                                          len(bank),policy['gain'],policy['regularization'])
                    full=exp.probability(base[val],bank[:,val],gate)
                    oof[val]=(1-conf['update_cap'])*base[val]+conf['update_cap']*full
            finally:exp.features=ORIGINAL_FEATURES
        newcal=self.remap(self.old_predictions['Shared'])[ncal:self.ncal]
        p=np.r_[oof,newcal]
        if abs(original_tau-threshold)>1e-12:
            p=expit(logit(np.clip(p,1e-6,1-1e-6))-logit(original_tau)+logit(threshold))
        return p


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True)
    p.add_argument('--domain',choices=['id','ood'],required=True);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):PairedWorker(a.task,a.domain,a.gpu).run()
