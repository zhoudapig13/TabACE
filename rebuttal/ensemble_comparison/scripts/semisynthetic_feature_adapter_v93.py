"""Generic complementary feature views and a Source-selected gate adaptation.

No view receives the synthetic/attacked coordinate identity. Two complementary
feature subsets guarantee coverage of all columns while keeping five TFM
alternative passes. C/D still remove the dedicated correlation-ranked drop.
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
from scipy.optimize import minimize

import semisynthetic_abcd_v92 as prior
import tabace_fusion_v28 as fusion
from prepare_data import RUN, atomic_json
from run_tabicl import Predictor, metrics, normalize
from shared_reference import load_condition

OUT = RUN / 'results/semisynthetic_feature_adapter_v93'


def feature_views(X, Q, task_id):
    order = np.random.default_rng(np.random.SeedSequence([0, int(task_id)])).permutation(X.shape[1])
    if len(order) == 1:
        groups = (order, order)
    else:
        groups = (order[:len(order)//2], order[len(order)//2:])
    return [(X[:, np.sort(ids)], Q[:, np.sort(ids)], np.sort(ids)) for ids in groups]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == prior.original.sha(__file__)
        return
    atomic_json(p, {'status': 'FROZEN_BEFORE_FEATURE_VIEW_INFERENCE', 'seed': 0,
        'Base_selection': prior.selection(), 'Base_and_benchmark_unchanged_from_v92': True,
        'context_rows': 256, 'alternative_slots': 5, 'estimators_per_context': 1,
        'slots_1_2': 'two complementary random feature groups, same observed current256 source labels',
        'feature_partition_seed': 'SeedSequence([global_seed0, task_id]); constant across conditions',
        'feature_partition_uses_labels_or_synthetic_column_identity': False,
        'C_D_slot3': 'uniform source bootstrap; dedicated correlation-ranked deletion absent',
        'generic_feature_suppression_remains_in_C_D': True,
        'interpretation': 'no dedicated matched constructor, but generic random feature partitioning allowed; not an ablation of all feature suppression',
        'gate_candidates': ['paper', 'disagreement_agreement_evidence'],
        'adapted_gate': 'unrestricted learned quality/agreement coefficients; max(Base entropy, branch disagreement) gate; agreement-only evidence; alpha_max0to1',
        'gate_choice': 'inner-dataset validation CE on allowed calibration episodes only',
        'regularization_grid': prior.original.REGS, 'all_gate_candidates_retained': True,
        'B_D_excludes_positive_five_episodes': True, 'own_dataset_query_labels_used_for_fit': False,
        'backbone_finetuning': False, 'additional_raw_feature_supervised_predictor': False,
        'prior_development_results_informed_design': True, 'adaptation_dagger': True,
        'original_paper_replication': False, 'independent_confirmation': False,
        'target_labels_only_used_for_declared_semisynthetic_generation_and_evaluation': True,
        'Markdown_written': False, 'code_sha256': prior.original.sha(__file__),
        'upstream_plan_sha256': prior.original.sha(prior.OUT / 'plan.json')})


def prepare(shard, shards):
    torch.set_num_threads(1); torch.manual_seed(0)
    paths = sorted((prior.OUT / 'banks').glob('*/*/*.npz'))
    tasks = sorted({p.parent.name for p in paths}, key=int)
    assigned = {identifier for i, identifier in enumerate(tasks) if i % shards == shard}
    model = Predictor('classification', 'cuda:0'); latent = {}; n = 0
    for path in paths:
        if path.parent.name not in assigned:
            continue
        meta = json.loads(path.with_suffix('.json').read_text())
        out = OUT / 'branches' / path.relative_to(prior.OUT / 'banks')
        if out.with_suffix('.json').exists():
            continue
        if meta['new_semisynthetic_condition']:
            if meta['task_id'] not in latent:
                latent[meta['task_id']] = prior.generator.latent(meta)
            case = prior.generator.materialize(meta, prior.selection()['configuration'], meta['rho'], latent[meta['task_id']])
        else:
            original_meta = json.loads(__import__('pathlib').Path(meta['original_bank_path']).with_suffix('.json').read_text())
            case, _ = load_condition(original_meta, original_meta['family'], original_meta['rho'])
        with np.load(path) as z:
            k = z['base'].shape[1]; assert np.array_equal(z['y_target'], case['y_target'])
        X, y, Q = case['X_source'], case['y_source'], case['X_target']
        arrays = {}; groups = []
        for i, (A, T, columns) in enumerate(feature_views(X, Q, meta['task_id'])):
            for estimator in model.models.values():
                root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                if root is not None and hasattr(root, 'root'):
                    root.root = None
            torch.cuda.empty_cache()
            arrays[f'prediction_{i}'] = model(A, y, T, n_classes=k)[0]
            arrays[f'columns_{i}'] = columns
            groups.append(columns.tolist())
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp.npz'); np.savez_compressed(temp, **arrays); temp.replace(out)
        atomic_json(out.with_suffix('.json'), {'status': 'COMPLETE', 'upstream_bank': str(path),
            'upstream_bank_sha256': prior.original.sha(path), 'feature_groups': groups,
            'source_rows': len(y), 'source_labels_modified': False, 'constructor_receives_target_labels': False,
            'constructor_receives_synthetic_column_identity': False})
        n += 1
        if n % 50 == 0:
            print('COMPLEMENTARY_VIEWS', shard, n, flush=True)


def load(matched):
    cells = prior.load(matched, 'original_registry')
    for c in cells:
        path = OUT / 'branches' / c['path'].relative_to(prior.OUT / 'banks')
        with np.load(path) as z:
            c['bank'][1] = z['prediction_0']; c['bank'][2] = z['prediction_1']
    return cells


def fit(cells, kind, regularization):
    fusion.GATE_VARIANT = 'entropy' if kind == 'paper' else 'entropy_or_disagreement'
    if kind == 'paper':
        return fusion.fit(cells, 'classification', 5, 1., regularization)
    tensors = fusion.training_tensors(cells, 5)
    def objective(values):
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        w = torch.softmax(p[:5] + tensors['Q'] * p[5:10] + tensors['S'] * p[10:15], 1)
        mixed = (w * tensors['true_members']).sum(1)
        alpha = p[-1] * tensors['U'] * (w * tensors['S']).sum(1)
        prediction = (1-alpha) * tensors['true_base'] + alpha * mixed
        penalty = (p[:5]-p[:5].mean()).square().mean() + (p[5:15]-1).square().mean()
        loss = -(tensors['weights'] * prediction.clamp(1e-12, 1.).log()).sum() + regularization * penalty
        loss.backward(); return float(loss.detach()), p.grad.detach().numpy()
    initial = np.r_[np.zeros(5), np.ones(10), .5]
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
        bounds=[(-8,8)]*5 + [(-10,10)]*10 + [(0,1)], options={'maxiter':280,'ftol':1e-9,'gtol':1e-6})
    assert np.isfinite(result.x).all()
    return {'kind': 'agreement_gate_adapted', 'parameters': result.x.tolist(), 'regularization': regularization,
            'gate_variant': 'entropy_or_disagreement', 'optimizer_success': bool(result.success)}


def predict(c, model):
    if model['kind'] != 'agreement_gate_adapted':
        return fusion.predict(c, model)
    p = np.asarray(model['parameters']); f = fusion.features(c, 5, 'entropy_or_disagreement')
    score = p[:5] + f['Q'] * p[5:10] + f['S'] * p[10:15]
    w = np.exp(score-score.max(1,keepdims=True)); w /= w.sum(1,keepdims=True)
    mixed = np.einsum('nr,rnc->nc', w, c['bank'])
    alpha = p[-1] * f['U'] * (w * f['S']).sum(1)
    return normalize((1-alpha[:,None])*c['base'] + alpha[:,None]*mixed)


def loss(cells, model):
    values = {}
    for c in cells:
        p = predict(c, model); y = c['y'].astype(int)
        ce = -np.log(p[np.arange(len(y)), y].clip(1e-12,1.)).mean()
        values.setdefault(c['meta']['data_id'], []).append(ce)
    return float(np.mean([np.mean(v) for v in values.values()]))


def worker(fold):
    torch.set_num_threads(1); torch.manual_seed(0)
    folder = OUT / 'folds/classification' / str(fold); rows = []
    for method, seen, matched in [('A',True,True),('B',False,True),('C',True,False),('D',False,False)]:
        cells = load(matched)
        train = [c for c in cells if c['fold'] != fold and (seen or not prior.original.exposed(c,'classification'))]
        test = [c for c in cells if c['fold']==fold and c['meta']['track']=='five_semisynthetic']
        train_ids={c['meta']['data_id'] for c in train}; test_ids={c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert seen or not any(prior.original.exposed(c,'classification') for c in train)
        inner=min(c['fold'] for c in train); first=[c for c in train if c['fold']!=inner]; val=[c for c in train if c['fold']==inner]
        candidates=[]
        for kind in ('paper','disagreement_agreement_evidence'):
            trials=[]
            for reg in prior.original.REGS:
                p=fit(first,kind,reg);trials.append([loss(val,p),reg])
            value,reg=min(trials)
            candidates.append({'gate':kind,'inner_CE':value,'trials':trials,'model':fit(train,kind,reg)})
        selected=min(candidates,key=lambda t:t['inner_CE'])
        atomic_json(folder/f'{method}_policy.json',{'method':method,'seen':seen,'matched':matched,
            'training_data_ids':sorted(train_ids),'excluded_data_ids':sorted(test_ids),'inner_fold':inner,
            'candidates':candidates,'selected_gate':selected['gate'],'selection_uses_outer_test_metrics':False})
        for candidate in candidates:
            name=method if candidate['gate']==selected['gate'] else method+'_unselected'
            for c in test:
                m=c['meta'];p=predict(c,candidate['model'])
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
        print('FEATURE_GATE_POLICY',fold,method,selected['gate'],flush=True)
    pd.DataFrame(rows).to_csv(folder/'detail.csv',index=False)
    atomic_json(folder/'completion.json',{'status':'COMPLETE','rows':len(rows)})


def launch(phase,shards):
    freeze()
    def execute(i):
        args=['prepare-worker','--shard',str(i),'--shards',str(shards)] if phase=='prepare' else ['fit-worker','--fold',str(i)]
        env=os.environ.copy()
        if phase=='prepare':env['CUDA_VISIBLE_DEVICES']=str(i)
        log=OUT/f'{phase}_{i}.log'
        with log.open('a') as handle:p=subprocess.run([sys.executable,__file__,*args],env=env,stdout=handle,stderr=subprocess.STDOUT)
        return {'job':i,'returncode':p.returncode,'log':str(log)}
    records=[]
    with ThreadPoolExecutor(max_workers=shards if phase=='prepare' else 4) as pool:
        for f in as_completed([pool.submit(execute,i) for i in range(shards if phase=='prepare' else 5)]):
            r=f.result();records.append(r);print(r,flush=True)
            atomic_json(OUT/f'{phase}_execution.json',{'status':'RUNNING','workers':records})
    atomic_json(OUT/f'{phase}_execution.json',{'status':'COMPLETE' if all(r['returncode']==0 for r in records) else 'FAILED','workers':records})
    if any(r['returncode'] for r in records):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','prepare-worker','fit','fit-worker'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.phase=='prepare-worker':prepare(a.shard,a.shards)
    elif a.phase=='fit-worker':worker(a.fold)
    else:launch(a.phase,a.shards)
