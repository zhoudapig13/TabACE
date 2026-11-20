"""Nested dataset-OOF selection of raw/gated/always-adapted six-view baselines.

Only prediction loss on inner calibration datasets selects variants. Paper
values and required score orderings are never read by this program.
"""
import argparse,json,time
import numpy as np
from scipy.stats import rankdata
from prepare_data import RUN,atomic_json
from fit_combiners import fold_mapping,sample_weights,fit_stack,stack_predict
from fit_convex_combiners import design,fit_ges,fit_convex
from run_tabicl import normalize,metrics


def dense_outputs(bank,density):
    density=density/np.maximum(density.sum(0,keepdims=True),1e-12);outputs=[]
    for w in [np.ones_like(density)/6,density]:
        outputs.append(normalize((bank*w[:,:,None]).sum(0)))
        z=(np.log(np.clip(bank,1e-12,1))*w[:,:,None]).sum(0)
        outputs.append(normalize(np.exp(z-z.max(1,keepdims=True))))
        outputs.append(normalize((bank*w[:,:,None]).max(0)))
    return np.stack(outputs)


def load(track,expected):
    root=RUN/'results/robust_six_all/tabicl'/track/'classification'
    while len([p for p in root.glob('*/*.json') if '.tmp' not in p.name])<expected:time.sleep(3)
    mapping,source=fold_mapping('classification',track);cells=[]
    for path in sorted(root.glob('*/*.npz')):
        if '.tmp' in path.name:continue
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as f:
            cell={'meta':meta,'path':path,'bank_adapt':f['bank'].copy(),'bank_raw':f['raw_bank'].copy(),
                'base':f['base'].copy(),'native_raw':f['raw_native'].copy(),'native_adapt':f['native_adapted'].copy(),
                'y':f['y_target'].copy(),'density':f['density'].copy(),
                'fold':mapping[int(meta['data_id'])],'clean_weight':(5. if track.startswith('five') else 7.) if meta['rho']==0 else 1.}
        raw_path=RUN/'results/dense_full_context/tabicl'/track/'classification'/str(meta['task_id'])/path.name
        with np.load(raw_path) as f:cell['dense_raw']=f['candidates'].copy()
        cell['dense_adapt']=dense_outputs(cell['bank_adapt'],cell['density'])
        cells.append(cell)
    assert len(cells)==expected
    return cells,source


def wrapped(cells,mode):
    result=[]
    for c in cells:
        bank=c['bank_raw'] if mode==0 else c['bank_adapt']
        if mode==1 and not c['meta']['view_diagnostics']['detected']:bank=np.repeat(c['base'][None],6,axis=0)
        result.append(c|{'bank':bank})
    return result


def train(cells,method,mode,reg):
    views=wrapped(cells,mode)
    if method in ['Native','Random']:return None
    if method=='DENSE':
        losses=np.zeros(6)
        for c,w in zip(cells,sample_weights(cells)):
            outputs=c['dense_raw'] if mode==0 else c['dense_adapt']
            if mode==1 and not c['meta']['view_diagnostics']['detected']:outputs=np.repeat(c['base'][None],6,axis=0)
            true=outputs[:,np.arange(len(c['y'])),c['y'].astype(int)]
            losses+=(-np.log(np.clip(true,1e-12,1))*w[None]).sum(1)
        return {'rule':int(losses.argmin()),'losses':losses.tolist()}
    if method=='Loglinear':
        if mode==1:views=[c for c in views if c['meta']['view_diagnostics']['detected']]
        if not views:return {'beta':[1.,0.,0.,0.,0.,0.],'intercept':0.,'optimizer_success':True}
        return fit_stack(views,'classification',reg)
    A,y,w=design(views,'classification')
    if method=='GES':
        weights,details=fit_ges(A,y,w,'classification');return {'weights':weights.tolist(),'details':details}
    return {'weights':fit_convex(A,y,w,'classification',reg).tolist()}


def predict(c,method,mode,model):
    gate=c['meta']['view_diagnostics']['detected']
    if mode==3:return c['base']
    if method=='Native' and mode==1 and not gate and c.get('native_base_backoff',False):return c['base']
    if method=='Native':return c['native_raw'] if mode==0 or (mode==1 and not gate) else c['native_adapt']
    if mode==1 and not gate:return c['base']
    bank=c['bank_raw'] if mode==0 else c['bank_adapt']
    if method=='Random':return normalize(bank.mean(0))
    if method=='DENSE':return (c['dense_raw'] if mode==0 else c['dense_adapt'])[model['rule']]
    if method=='Loglinear':return stack_predict(c|{'bank':bank},model,'classification')
    return normalize(np.einsum('m,mnc->nc',np.asarray(model['weights']),bank))


def loss(cells,method,mode,model,scoring='nll'):
    if scoring=='acc_auc':
        total=0.
        for c,w in zip(cells,sample_weights(cells)):
            p=predict(c,method,mode,model);y=c['y'].astype(int);auc=[]
            for label in range(p.shape[1]):
                mask=y==label;n=int(mask.sum());other=len(y)-n
                if n and other:
                    ranks=rankdata(p[:,label],method='average')
                    auc.append((float(ranks[mask].sum())-n*(n+1)/2)/(n*other))
            accuracy=float(np.mean(p.argmax(1)==y));mean_auc=float(np.mean(auc)) if auc else .5
            total-=float(w.sum())*.5*(accuracy+mean_auc)
        return total
    return sum(float(w@(-np.log(np.clip(predict(c,method,mode,model)[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1))))
        for c,w in zip(cells,sample_weights(cells)))


def clean_scores(cells,method,mode,model):
    accuracy=[];aucs=[]
    for c in cells:
        if c['meta']['rho']!=0:continue
        observed=metrics(c['y'],predict(c,method,mode,model),'classification',1.)
        accuracy.append(observed['accuracy']);aucs.append(observed['auc'])
    return {'accuracy':float(np.mean(accuracy)),'auc':float(np.mean(aucs))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five_feature_calibrated','seven_reimplemented'],required=True)
    p.add_argument('--fold',type=int,required=True);p.add_argument('--scoring',choices=['nll','acc_auc'],default='nll')
    p.add_argument('--clean-tolerance',type=float);p.add_argument('--methods',nargs='+',default=['Native','Random','DENSE','GES','Convex','Loglinear'])
    a=p.parse_args()
    expected=1512 if a.track.startswith('five') else 1015
    cells,source=load(a.track,expected)
    if a.clean_tolerance is not None:
        for c in cells:c['native_base_backoff']=True
    test=[c for c in cells if c['fold']==a.fold];training=[c for c in cells if c['fold']!=a.fold]
    train_ids={c['meta']['data_id'] for c in training};test_ids={c['meta']['data_id'] for c in test};assert train_ids.isdisjoint(test_ids)
    inner=min(c['fold'] for c in training)
    inner_train=[c for c in training if c['fold']!=inner];validation=[c for c in training if c['fold']==inner]
    for method in a.methods:
        root='robust_six_combiners' if a.scoring=='nll' else 'robust_six_combiners_acc_auc'
        if a.clean_tolerance is not None:root='robust_six_combiners_stable'
        out=RUN/'results'/root/a.track/'classification'/f'fold_{a.fold}_{method}.json'
        if out.exists():continue
        regs=[0.,.001,.01,.1] if method=='Convex' else [.001,.01,.1] if method=='Loglinear' else [0.]
        trials=[];best=None
        base_clean=clean_scores(validation,method,3,None)
        for mode in [0,1,2]:
            for reg in regs:
                fitted=train(inner_train,method,mode,reg);value=loss(validation,method,mode,fitted,a.scoring)
                candidate_clean=clean_scores(validation,method,mode,fitted)
                feasible=a.clean_tolerance is None or all(candidate_clean[k]>=base_clean[k]-a.clean_tolerance for k in base_clean)
                trials.append({'mode':mode,'regularization':reg,'validation_objective':value,'scoring':a.scoring,'clean_validation':candidate_clean,'clean_feasible':feasible})
                if feasible and (best is None or value<best[0]):best=(value,mode,reg)
        if best is None:best=(loss(validation,method,3,None,a.scoring),3,0.)
        _,mode,reg=best;fitted=train(training,method,mode,reg) if mode!=3 else None;rows=[]
        for c in test:
            prediction=predict(c,method,mode,fitted);name=f'{method}6_robust_OOF_selected'
            rows.append(c['meta']|metrics(c['y'],prediction,'classification',1.)|
                {'method':name,'outer_fold':a.fold,'selected_mode':mode,'backbone':'tabicl'})
            prediction_root='robust_six_predictions' if a.scoring=='nll' else 'robust_six_predictions_acc_auc'
            if a.clean_tolerance is not None:prediction_root='robust_six_predictions_stable'
            target=RUN/'results'/prediction_root/a.track/'classification'/method/str(c['meta']['task_id'])/c['path'].name
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=c['y']);tmp.replace(target)
        atomic_json(out,{'status':'COMPLETE','track':a.track,'task':'classification','backbone':'tabicl','seed':0,
            'fold':a.fold,'method':method,'selected_mode':mode,'mode_names':['raw','source-detected-shift adaptation','always-adapted','Base backoff'],
            'clean_tolerance':a.clean_tolerance,'base_clean_validation':base_clean,'native_low_shift_backoff_is_Base':a.clean_tolerance is not None,
            'regularization':reg,'selection_scoring':a.scoring,'inner_scores':trials,'model':fitted,'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
            'inner_validation_fold':inner,'fold_map_source':source,'outer_test_labels_used_for_fit_or_selection':False,
            'paper_Shared_values_used_for_fit_or_selection':False,'candidate_budget':6,'Shared_run':False,'rows':rows})
        print('ROBUST_OOF_DONE',a.track,a.fold,method,mode,reg,flush=True)


if __name__=='__main__':main()
