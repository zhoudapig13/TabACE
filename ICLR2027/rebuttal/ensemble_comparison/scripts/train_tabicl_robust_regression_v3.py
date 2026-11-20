"""Actual task-specific weight updates using authorized clean TRAIN labels.

Training/validation queries are drawn only from the original source training
partition. Frozen benchmark query labels never enter this trainer. Each updated
checkpoint is shared by all ensemble algorithms; an updated single model is
also evaluated to separate backbone adaptation from ensemble gains.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from internal_regression_benchmark import OUT as OLD,manifest
from recalibrate_tabicl_regression_v2 import OUT as BENCH
from run_tabicl import CHECKPOINTS

OUT=RUN/'results/tabicl_regression_weight_adaptation_v3'
PLAN=RUN/'configs/tabicl_regression_weight_adaptation_v3.json'
CKPT=CHECKPOINTS/'tabicl-regressor-v2-20260212.ckpt'


def init():
    assert not PLAN.exists()
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(PLAN,{'status':'DECLARED_BEFORE_WEIGHT_UPDATES','benchmark':'internal_regression_benchmark_v2',
        'benchmark_changes_allowed':False,'model_seed':0,'strengths':[.2,.5,.8,1.],
        'target_R2_gain_five_and_seven':.02,'other_metrics':'report actual values; no gain floor',
        'desired_Clean_R2_band':[.71,.74],'Clean_band_is_soft_not_score_manipulation':True,
        'additional_supervision_user_authorized':'Clean labels from the same official TRAIN partition before corruption',
        'training_rows_cap':2048,'source_train_validation_split':[.8,.2],
        'benchmark_query_labels_used_for_training_selection_or_stopping':False,
        'context_augmentation':'same five/seven physical operators and frozen coefficients; source-only corruption',
        'training_context_cap':256,'training_query_cap':64,'validation_query_cap':128,
        'training_context_mix':{'clean':.2,'five_perturbations':.4,'seven_shifts':.4},
        'initial_training_steps':128,'learning_rate':1e-5,'weight_decay':.01,'grad_clip':1.,
        'trainable_modules':'last four ICL transformer blocks plus ICL decoder/y encoder/layer norm',
        'column_and_row_modules_frozen':True,'checkpoint_selection':'best source-validation proxy among updated steps',
        'pretrained_weights_not_candidate_for_updated_checkpoint':True,
        'ensemble_candidate_budget':6,'original_Base_prediction_fallback_allowed':False,
        'mandatory_extra_control':'fine-tuned single-estimator backbone',
        'checkpoint_shared_by_all_ensemble_algorithms':True,
        'annotations':{'†':'Context/input adaptation','‡':'Combiner calibration','§':'Verified backbone weight updates'},
        'Shared_score_used_for_model_selection':False,'internal_development_only':True,'Markdown_written':False})
    lock={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BENCH/f'frozen_{t}.json' for t in ['clean','five','seven']]}
    atomic_json(OUT/'benchmark_lock.json',lock)
    print('WEIGHT_ADAPTATION_V3_DECLARED',flush=True)


def source_data(m):
    frozen=json.loads((OLD/'frozen_clean.json').read_text())
    p=OLD/'clean_candidates'/frozen['candidate']/f"{m['task_id']}.npz"
    raw=OLD/'clean_candidates/source11_target7'/f"{m['task_id']}.npz"
    with np.load(p) as z:
        X=z['X_source'];y=z['y_source'];ids=z['source_indices'];query_ids=z['target_indices']
    with np.load(raw) as z:
        Xraw=z['X_source'];assert np.array_equal(z['y_source'],y) and np.array_equal(z['source_indices'],ids)
    assert not np.intersect1d(ids,query_ids).size
    order=np.random.default_rng(0).permutation(len(y));nv=max(32,int(np.ceil(.2*len(y))))
    val=order[:nv];train=order[nv:]
    assert not np.intersect1d(train,val).size
    return {'five':X,'seven':Xraw,'y':y,'source_indices':ids,'train':train,'val':val}


def augment(m,source,ctx,qry,remaining,track,family,rho):
    X=source[track][ctx];y=source['y'][ctx];Q=source[track][qry];t=source['y'][qry]
    if not rho:return X.copy(),y.copy(),Q.copy(),t.copy()
    if track=='five':
        settings=json.loads((BENCH/'frozen_five.json').read_text())['generator_parameters']
        active=settings['family_parameters'][family]
        changed=protocol.manipulate_source_context('regression',X,y,family=family,rho=rho/100.,seed=1729+rho,
            shift_scale=active['shift_scale'],label_scale=active['label_scale'])
        contrast=active.get('label_contrast',1.)
        if active.get('label_contrast_tail_power',0.):
            ratio=float(np.subtract(*np.percentile(y,[75,25]))/1.349/max(float(y.std()),1e-12))
            contrast*=float(np.clip(ratio,.05,2.))**active['label_contrast_tail_power']
        out=changed.y_context if contrast==1. else y+contrast*(changed.y_context-y)
        return changed.X_context,out,Q,t
    from calibrated_seven_protocol import materialize
    coeff=json.loads((BENCH/'frozen_seven.json').read_text())['generator_config']['family_amplitude_multipliers'][family]
    ref=remaining[:128];hist=remaining[128:384]
    clean=(source[track][hist],source['y'][hist],source[track][ref],source['y'][ref],X,y,Q,t)
    data,_=materialize(m,family,rho,clean,coeff)
    return tuple(data[k] for k in ['X_source','y_source','X_target','y_target'])


def batch(m,source,track,family,rho,rng,validation=False):
    from tabicl._finetune.data import _build_ensemble_generator
    train=source['train'];order=rng.permutation(train)
    if validation:
        ctx=order[:min(256,len(order))];qry=source['val'][:128];remaining=order[len(ctx):]
    else:
        nq=min(64,max(16,len(order)//6));qry=order[:nq];ctx=order[nq:nq+min(256,len(order)-nq)]
        remaining=order[nq+len(ctx):]
    X,y,Q,t=augment(m,source,ctx,qry,remaining,track,family,rho)
    mean=float(y.mean());sd=max(float(y.std()),1e-8);target_sd=max(float(source['y'][source['train']].std()),1e-6)
    yn=(y-mean)/sd;tn=(t-mean)/sd
    if X.shape[1]==0 or np.all(np.ptp(X,axis=0)==0):
        return {'degenerate':True,'t':torch.from_numpy(tn.copy()).float(),'mean':mean,'sd':sd,
            'query_variance':max(float(t.var()),1e-12),'track':track,'family':family,'rho':rho}
    gen=_build_ensemble_generator(classification=False,n_estimators=1,norm_methods=None,
        feat_shuffle_method='latin',class_shuffle_method='shift',outlier_threshold=4.,random_state=0)
    gen.fit(X,yn);variants=gen.transform(Q,mode='both')
    A=np.concatenate([v[0] for v in variants.values()],axis=0)
    b=np.concatenate([v[1] for v in variants.values()],axis=0)
    return {'X':torch.from_numpy(A).float(),'y':torch.from_numpy(b).float(),
        't':torch.from_numpy(np.broadcast_to(tn,(len(A),len(tn))).copy()).float(),
        'ratio':sd/target_sd,'track':track,'family':family,'rho':rho,
        'query_variance':max(float(t.var()),1e-12),'mean':mean,'sd':sd}


def train_mode(model):
    model.train();model.col_embedder.eval();model.row_interactor.eval()


def forward(model,b):
    device=next(model.parameters()).device
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['X'].to(device),b['y'].to(device))
    return out.float()


def validation(model,batches):
    train_mode(model);rows=[]
    with torch.no_grad():
        for b in batches:
            if b.get('degenerate',False):
                mse=float(torch.mean(b['t']**2))*b['sd']**2
                rows.append((b['track'],b['rho'],1-mse/b['query_variance']))
                continue
            pred=forward(model,b).mean(-1);target=b['t'].to(pred.device)
            mse=float(torch.mean((pred-target)**2).item())*b['sd']**2
            rows.append((b['track'],b['rho'],1-mse/b['query_variance']))
    five=float(np.mean([v for t,r,v in rows if t=='five' and r>0]))
    seven=float(np.mean([v for t,r,v in rows if t=='seven' and r>0]))
    clean=float(np.mean([v for t,r,v in rows if r==0]))
    return {'five_proxy_R2':five,'seven_proxy_R2':seven,'Clean_proxy_R2':clean,
            'selection_score':.5*five+.5*seven}


def fit(m,steps,lr,tag):
    from tabicl import TabICLRegressor
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    dest=OUT/'checkpoints'/tag/str(m['task_id']);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'training_result.json').exists():return
    source=source_data(m)
    atomic_json(dest/'source_split.json',{'train_original_row_ids':source['source_indices'][source['train']].tolist(),
        'validation_original_row_ids':source['source_indices'][source['val']].tolist(),
        'benchmark_queries_disjoint':True,'additional_clean_TRAIN_label_supervision':True})
    loader=TabICLRegressor(n_estimators=1,device='cuda:0',model_path=str(CKPT),allow_auto_download=False,n_jobs=1,random_state=0)
    loader._resolve_device();loader._load_model();model=loader.model_.to('cuda:0')
    # Frozen inference code returns a no-grad view. The trainable ICL y encoder
    # updates that representation in place; clone at the boundary to make the
    # same values a regular tensor created in the surrounding training context.
    model.row_interactor.register_forward_hook(lambda _module,_args,output:output.clone())
    for p in model.parameters():p.requires_grad=False
    for p in model.icl_predictor.parameters():p.requires_grad=True
    for block in model.icl_predictor.tf_icl.blocks[:-4]:
        for p in block.parameters():p.requires_grad=False
    initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
    val=[]
    from seven_shifts import FAMILIES as SEVEN
    for track,families in [('five',protocol.FAMILIES),('seven',SEVEN)]:
        for f,r in [('original' if track=='five' else 'covariate',0)]+[(f,80) for f in families]:
            val.append(batch(m,source,track,f,r,np.random.default_rng(1234),True))
    baseline=validation(model,val);history=[];best=None;start=time.monotonic()
    atomic_json(dest/'training_config.json',{'steps':steps,'learning_rate':lr,'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameter_count':sum(p.numel() for p in model.parameters()),'seed':0,'baseline_validation':baseline,
        'checkpoint_selection_uses_only_source_validation':True,'benchmark_query_labels_used':False})
    log=(dest/'steps.jsonl').open('a',buffering=1);updates=0;skipped=0
    for step in range(1,steps+1):
        rng=np.random.default_rng(step);track='five' if rng.random()<.375 else 'seven'
        families=protocol.FAMILIES if track=='five' else SEVEN
        clean=rng.random()<.2;family=('original' if track=='five' else 'covariate') if clean else families[int(rng.integers(len(families)))]
        rho=0 if clean else int(rng.choice([20,50,80,100]));b=batch(m,source,track,family,rho,rng)
        if b.get('degenerate',False):
            skipped+=1;log.write(json.dumps({'step':step,'skipped_degenerate_training_episode':True,'track':track,'family':family,'rho':rho})+'\n')
            continue
        train_mode(model);optimizer.zero_grad(set_to_none=True);raw=forward(model,b);target=b['t'].to(raw.device)
        pred=raw.mean(-1);ratio=b['ratio']
        mse=torch.mean(((pred-target)*ratio)**2)
        alphas=torch.linspace(0.,1.,raw.shape[-1]+2,device=raw.device)[1:-1]
        diff=target.unsqueeze(-1)-raw
        pinball=torch.maximum(alphas*diff,(alphas-1)*diff).mean()*ratio
        loss=mse+.05*pinball
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fine-tuning loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();updates+=1;log.write(json.dumps({'step':step,'loss':float(loss),'gradient_norm':float(grad),'track':track,'family':family,'rho':rho})+'\n')
        if step%32==0 or step==steps:
            scores=validation(model,val)
            gain5=scores['five_proxy_R2']-baseline['five_proxy_R2'];gain7=scores['seven_proxy_R2']-baseline['seven_proxy_R2']
            scores['five_proxy_R2_gain']=gain5;scores['seven_proxy_R2_gain']=gain7
            scores['selection_score']=min(gain5,gain7)+.05*(gain5+gain7)
            history.append({'step':step,**scores})
            checkpoint={'config':loader.model_config_,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'curr_step':step,
                'adaptation':'supervised robust fine-tuning on clean official TRAIN labels; no benchmark query labels'}
            if best is None or scores['selection_score']>best['selection_score']:
                torch.save(checkpoint,dest/'best.ckpt');best={'step':step,**scores}
            torch.save(checkpoint,dest/'last.ckpt')
            print('TRUE_FT_STEP',m['dataset'],step,scores,flush=True)
    log.close()
    selected=torch.load(dest/'best.ckpt',map_location='cpu',weights_only=False)['state_dict']
    delta=sum(float(torch.sum((selected[k]-v)**2)) for k,v in initial.items())
    changed=sum(not torch.equal(selected[k],v) for k,v in initial.items());assert delta>0 and changed>0
    result={'status':'COMPLETE','dataset':m['dataset'],'task_id':m['task_id'],'steps':steps,'best':best,'history':history,
        'optimizer_steps_performed':updates,'degenerate_training_episodes_skipped':skipped,
        'trainable_weight_delta_L2':delta**.5,'changed_parameter_tensors':changed,'actual_backbone_weights_updated':True,
        'additional_clean_training_supervision':True,'benchmark_query_labels_used_for_training_or_selection':False,
        'annotation':'§','baseline_source_validation':baseline,'seconds':time.monotonic()-start}
    atomic_json(dest/'training_result.json',result);print('TRUE_FT_COMPLETE',m['dataset'],result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','train']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--steps',type=int,default=128);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--tag',default='initial128')
    a=p.parse_args()
    if a.phase=='init':init()
    else:
        jobs=manifest()[a.shard::a.shards]
        if a.limit:jobs=jobs[:a.limit]
        for m in jobs:fit(m,a.steps,a.lr,a.tag)
