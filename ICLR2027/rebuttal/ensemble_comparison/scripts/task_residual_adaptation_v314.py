"""Independent task/family/strength residual policies; adaptation labels only.

The output evaluator is a separate program. This program never opens final
evaluation labels, including when choosing kernel features and hyperparameters.
"""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import task_adaptation_bank_v312 as bank
import task_residual_kernel_v313 as kernel
from prepare_data import RUN,ROOT,atomic_json,protocol

source=bank.source
OUT=RUN/'results/task_residual_adaptation_v314'


def unit_rows(F):
    F=np.asarray(F,dtype=np.float64)
    return F/np.maximum(np.linalg.norm(F,axis=1,keepdims=True),1e-12)


def features(z,task,shared):
    latent=unit_rows(z['latent'])
    core=z['Native_finetuned']
    raw=np.concatenate([core[None],z['members']],axis=0) if shared else core[None]
    if task=='classification':
        output=np.sqrt(np.maximum(raw,0)).transpose(1,0,2).reshape(len(core),-1)
    else:
        output=np.tanh((raw-float(z['source_mean']))/float(z['source_scale'])).T
    output=unit_rows(output)
    return {'latent':latent,'prediction':output,'latent_and_prediction':unit_rows(np.concatenate([latent,output],axis=1))}


def select_and_predict(candidates,p,ay,apos,epos,task):
    selected=None
    for name,F in candidates.items():
        model=kernel.fit(F[apos],p[apos],ay,task)
        if selected is None or tuple(model['score'])>tuple(selected['score']):
            selected=dict(model,feature=name)
    F=candidates[selected['feature']]
    pred=kernel.predict(F[epos],p[epos],F[apos],selected,task)
    return pred,selected


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'seed':0,'cases':175,'conditions':2175,'task_specific_parameters':True,
        'independent_policy_per':['dataset','track','mechanism','strength','method'],
        'selection':'accuracy then log-likelihood for classification; MSE for regression, exact adaptation-subset LOO',
        'feature_candidates':['latent','prediction','latent_and_prediction'],
        'bandwidth_multipliers':[.5,1.,2.],'ridge_penalties':[.01,.1,1.,10.],
        'identity_candidate_included':True,'source_weights_and_contexts_unchanged_from_v312':True,
        'target_supervision':'quarter original query, cap64, removed from evaluation',
        'target_labels_enter_TFM_context':False,'final_eval_labels_used_for_selection':False,
        'same_adaptation_budget_for_Native_and_Shared':True,'unlabeled_query_features_used_transductively':True,
        'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'bank_plan_sha256':source.sha(bank.OUT/'plan.json'),
        'split_manifest_sha256':source.sha(bank.split.OUT/'case_manifest.json'),
        'script_sha256':source.sha(__file__),'kernel_sha256':source.sha(kernel.__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def main(require_complete=False):
    prepare();plan=json.loads((OUT/'plan.json').read_text())
    assert source.sha(__file__)==plan['script_sha256'] and source.sha(kernel.__file__)==plan['kernel_sha256']
    complete=0
    for c in json.loads((bank.split.OUT/'case_manifest.json').read_text()):
        track,task,tid=c['track'],c['task'],c['task_id']
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as labels:
            assert 'y_target' not in labels.files
            apos,epos,ay=(labels[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert set(apos).isdisjoint(epos)
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
            path=bank.OUT/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            if not path.with_suffix('.json').exists():continue
            dest=OUT/'predictions'/track/task/str(tid)/path.name
            if dest.with_suffix('.json').exists():
                meta=json.loads(dest.with_suffix('.json').read_text());assert source.sha(dest)==meta['output_sha256']
                complete+=1;continue
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            with np.load(path) as z:
                native,native_policy=select_and_predict(features(z,task,False),z['Native_finetuned'],ay,apos,epos,task)
                shared,shared_policy=select_and_predict(features(z,task,True),z['Shared_finetuned'],ay,apos,epos,task)
                old=source.OUT/'predictions'/track/task/str(tid)/path.name
                original_meta=json.loads(old.with_suffix('.json').read_text());assert source.sha(old)==original_meta['output_sha256']
                with np.load(old) as original:
                    original_base=original['Base'][epos]
                source.npwrite(dest,Base=original_base,Native_finetuned=z['Native_finetuned'][epos],
                    Shared_finetuned=z['Shared_finetuned'][epos],Native_adapted=native,Shared_adapted=shared)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('track','task','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'output_sha256':source.sha(dest),'bank_sha256':meta['output_sha256'],
                 'adaptation_sha256':c['adaptation_sha256'],'target_adaptation_rows':len(apos),
                 'evaluation_positions':epos.tolist(),'adaptation_positions':apos.tolist(),
                 'final_eval_labels_opened':False,'final_eval_scores_used_for_selection':False,
                 'Native_policy':native_policy,'Shared_policy':shared_policy})
            complete+=1
        print('TASK_RESIDUAL_CASE',track,task,tid,'conditions',complete,flush=True)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if complete==2175 else 'PARTIAL','conditions':complete,'expected':2175})
    if require_complete:assert complete==2175


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();main(args.require_complete)
