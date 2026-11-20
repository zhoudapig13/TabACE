"""Pair response repair and no repair on identical bootstrap memberships.

Protocol correction before v104 query metrics were scored. Reuses v104's
generic/source-quality predictions, and replaces matched slot0 by repaired
labels on the EXACT SAME seed0/seed1 bootstrap rows. Source64 quality labels,
search counts and deployed context counts are identical for all A-D.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import subprocess
import sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import numpy as np
import torch
import abcd_source_quality_v104 as algorithm
import abcd_reference_repair_v102 as repair
import abcd_fresh_confirmation_v101 as fresh
import abcd_metric_aligned_v100 as old
import seven_shifts as seven
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor

PREVIOUS=algorithm.OUT
OUT=RUN/'results/abcd_paired_quality_v105'


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'code_sha256':old.origin.sha(__file__),'algorithm_sha256':old.origin.sha(algorithm.__file__),
        'source_quality_and_calibration_protocol':str(PREVIOUS/'plan.json'),
        'protocol_correction':'same bootstrap row identities and unique-row counts for matched/unmatched candidates',
        'repair_strength':1.,'two_candidate_seeds':[0,1],'all_methods_candidate_counts':2,
        'TFM_fits_per_method':7,'deployed_alternative_contexts':5,'context_cap':256,
        'source_reference_quality_labels':64,'extra_target_query_supervision':False,
        'v104_query_metrics_not_scored_before_correction':True,
        'A_C_family_specialized_calibration':True,'B_D_exclude_concept_temporal_offline':True,
        'no_method_ordering_in_loss_or_selection':True,'previous_query_cohorts_are_development':True,
        'original_paper_replication':False,'independent_confirmation':False,'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def prepare_worker(shard):
    torch.set_num_threads(1);seven.PARAMETERS['regression_response_drift_sd']=1.
    items=[m for m in seven.registry() if m['task']=='regression'];model=Predictor('regression','cuda:0')
    for i,m in enumerate(items):
        if i%8!=shard:continue
        tid=int(m['task_id']);clean=seven.regression_clean(m);Qextra=None
        extra=fresh.OUT/'inputs/regression'/f'{tid}.npz'
        if extra.exists():
            with np.load(extra) as z:Qextra=z['X_evaluation']
        for previous in (PREVIOUS/'banks'/str(tid)).glob('*.npz'):
            out=OUT/'banks'/str(tid)/previous.name
            if out.with_suffix('.json').exists():continue
            meta=json.loads(previous.with_suffix('.json').read_text());family,rho=meta['family'],meta['rho']
            data,_=seven.materialize(m,family,rho,clean)
            X,y,P,py,Q=(data[k] for k in ('X_source','y_source','X_pool','y_pool','X_target'))
            held=np.arange(320,384);keep=np.setdiff1d(np.arange(len(py)),held)
            T,t=P[held],py[held];P,py=P[keep],py[keep]
            use_extra=Qextra is not None and (rho==0 or family=='concept')
            queries=np.vstack([T,Q,Qextra]) if use_extra else np.vstack([T,Q])
            _,repaired,details=repair.differential_repair(X,y,P,py,1.)
            repaired_pool=py.copy();repaired_pool[:len(y)]=repaired
            candidates=[];mse=[];row_hashes=[];unique=[]
            scale=max(float(t.std()),1e-6)
            for seed in (0,1):
                ids=np.random.default_rng(seed).choice(len(py),256,replace=True)
                pred=model(P[ids],repaired_pool[ids],queries)[0]
                candidates.append(pred);mse.append(float(np.mean((pred[:64]-t)**2))/scale**2)
                row_hashes.append(protocol.array_sha256(ids));unique.append(len(np.unique(ids)))
            choice=int(np.argmin(mse));pred=candidates[choice]
            with np.load(previous) as z:pack={k:z[k] for k in z.files}
            pack['matched_members']=pack['generic_members'].copy();pack['matched_members'][0]=pred[64:64+len(Q)]
            pack['matched_quality']=pack['generic_quality'].copy();pack['matched_quality'][1]=-np.log1p(np.clip(mse[choice],0,1e6))
            if use_extra:
                pack['matched_extra_members']=pack['generic_extra_members'].copy();pack['matched_extra_members'][0]=pred[64+len(Q):]
            out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,**pack)
            meta.update(matched_repair_strength=1.,matched_bootstrap_seed=choice,
                matched_source_quality_mse_choices=mse,paired_candidate_row_hashes=row_hashes,
                paired_candidate_unique_rows=unique,matched_and_generic_candidates_use_identical_rows=True,
                repair_details=details,source_quality_labels_excluded_from_all_contexts=True,
                previous_bank_sha256=old.origin.sha(previous),generic_bank_reused_unchanged=True)
            atomic_json(out.with_suffix('.json'),meta)
        print('PAIRED_QUALITY_BANK',tid,flush=True)


def prepare():
    freeze()
    def run(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i)
        with (OUT/f'prepare_{i}.log').open('a') as f:
            p=subprocess.run([sys.executable,__file__,'prepare-worker','--shard',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode}
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(run,i) for i in range(8)]):
            r=f.result();records.append(r);print(r,flush=True)
    atomic_json(OUT/'preparation.json',{'status':'COMPLETE' if not any(r['returncode'] for r in records) else 'FAILED','workers':records})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','prepare-worker','train','report']);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='prepare-worker':prepare_worker(a.shard)
    else:
        algorithm.OUT=OUT
        algorithm.train() if a.phase=='train' else algorithm.report()
