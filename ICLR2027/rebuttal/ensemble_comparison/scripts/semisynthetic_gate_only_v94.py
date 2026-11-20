"""Source-selected gate-only adaptation; strict C/D constructors stay intact.

Uses the v92 density/row bank, never the generic feature-partition proposal.
This preserves absence of feature suppression in the C/D negative controls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch

import semisynthetic_abcd_v92 as source
from semisynthetic_feature_adapter_v93 import fit, predict, loss
from prepare_data import RUN, atomic_json
from run_tabicl import metrics

OUT = RUN / 'results/semisynthetic_gate_only_v94'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text())['code_sha256'] == source.original.sha(__file__)
        return
    atomic_json(path, {'status': 'FROZEN_BEFORE_GATE_ONLY_FITTING', 'seed': 0,
        'benchmark_and_Base_unchanged': True, 'Base_selection': source.selection(),
        'bank': 'v92 density_adapted, unchanged observed source rows and all feature columns in C/D',
        'feature_partition_proposal_used': False,
        'reason_feature_partitions_excluded': 'generic partitions can implicitly suppress the attacked coordinate; they are unsuitable as a strict no-feature-repair control',
        'gate_candidates': ['paper', 'disagreement_agreement_evidence'],
        'adapted_gate': 'learned quality/agreement coefficients may be negative; Base entropy OR branch disagreement; agreement-only support, alpha_max0to1',
        'same_TFM_branch_predictions_reused': True, 'new_backbone_finetuning': False,
        'new_raw_feature_supervised_model': False, 'context_budget': 256, 'alternative_contexts': 5,
        'selection': 'inner-dataset validation CE, both gate choices refitted and reported',
        'regularization_grid': source.original.REGS,
        'B_D_excludes_all_positive_five_source_perturbation_episodes': True,
        'fitting_uses_evaluated_dataset_query_labels': False,
        'previous_development_outcomes_motivated_gate_hypothesis': True,
        'original_paper_replication': False, 'independent_confirmation': False,
        'target_labels_only_used_in_declared_semisynthetic_generation_and_evaluation': True,
        'code_sha256': source.original.sha(__file__),
        'gate_implementation_sha256': source.original.sha(__import__('semisynthetic_feature_adapter_v93').__file__),
        'Markdown_written': False})


def worker(fold):
    torch.set_num_threads(1); torch.manual_seed(0)
    folder = OUT / 'folds/classification' / str(fold); rows = []
    for method, seen, matched in [('A',True,True),('B',False,True),('C',True,False),('D',False,False)]:
        cells = source.load(matched, 'density_adapted')
        train = [c for c in cells if c['fold']!=fold and (seen or not source.original.exposed(c,'classification'))]
        test = [c for c in cells if c['fold']==fold and c['meta']['track']=='five_semisynthetic']
        train_ids={c['meta']['data_id'] for c in train}; test_ids={c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert seen or not any(source.original.exposed(c,'classification') for c in train)
        inner=min(c['fold'] for c in train); first=[c for c in train if c['fold']!=inner]; val=[c for c in train if c['fold']==inner]
        candidates=[]
        for kind in ('paper','disagreement_agreement_evidence'):
            if kind=='paper':
                path=source.OUT/'density_adapted/folds/classification'/str(fold)/f'{method}_policy.json'
                old=json.loads(path.read_text()); assert old['training_data_ids']==sorted(train_ids)
                candidates.append({'gate':kind,'inner_CE':min(old['trials'])[0], 'trials':old['trials'],
                    'model':old['policy'],'reused_policy_sha256':source.original.sha(path)})
                continue
            trials=[]
            for reg in source.original.REGS:
                p=fit(first,kind,reg); trials.append([loss(val,p),reg])
            value,reg=min(trials)
            candidates.append({'gate':kind,'inner_CE':value,'trials':trials,'model':fit(train,kind,reg)})
        selected=min(candidates,key=lambda p:p['inner_CE'])
        atomic_json(folder/f'{method}_policy.json',{'method':method,'seen':seen,'matched':matched,
            'training_data_ids':sorted(train_ids),'excluded_data_ids':sorted(test_ids),'inner_fold':inner,
            'candidates':candidates,'selected_gate':selected['gate'],'selection_uses_outer_test_metrics':False})
        for candidate in candidates:
            name=method if candidate['gate']==selected['gate'] else method+'_unselected'
            for c in test:
                m=c['meta']; p=predict(c,candidate['model'])
                path=folder/'predictions'/str(m['task_id'])/f'{m["rho"]}_{name}.npz'
                path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=p,y=c['y'])
                rows.append({'task':'classification','method':name,'gate':candidate['gate'],'data_id':m['data_id'],
                    'task_id':m['task_id'],'dataset':m['dataset'],'rho':m['rho'],'fold':fold,**metrics(c['y'],p,'classification',1.)})
        if method=='A':
            for c in test:
                m=c['meta']
                for name,p in [('Base',c['base']),('Single-matched',c['bank'][3])]:
                    path=folder/'predictions'/str(m['task_id'])/f'{m["rho"]}_{name}.npz';np.savez_compressed(path,prediction=p,y=c['y'])
                    rows.append({'task':'classification','method':name,'gate':'control','data_id':m['data_id'],
                        'task_id':m['task_id'],'dataset':m['dataset'],'rho':m['rho'],'fold':fold,**metrics(c['y'],p,'classification',1.)})
        print('GATE_ONLY_POLICY',fold,method,selected['gate'],flush=True)
    pd.DataFrame(rows).to_csv(folder/'detail.csv',index=False)
    atomic_json(folder/'completion.json',{'status':'COMPLETE','rows':len(rows)})


def launch():
    freeze()
    def execute(fold):
        log=OUT/f'worker_{fold}.log'
        with log.open('a') as handle:
            p=subprocess.run([sys.executable,__file__,'worker','--fold',str(fold)],stdout=handle,stderr=subprocess.STDOUT)
        return {'fold':fold,'returncode':p.returncode,'log':str(log)}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(execute,i) for i in range(5)]):
            r=f.result();results.append(r);print(r,flush=True)
            atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':results})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if all(r['returncode']==0 for r in results) else 'FAILED','workers':results})
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.phase=='worker':worker(a.fold)
    else:launch()
