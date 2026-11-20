"""Choose raw/source-clipping variants using nested dataset calibration only."""
import argparse
import json
import time
import numpy as np
from scipy.stats import rankdata
from prepare_data import RUN,atomic_json
from fit_combiners import fold_mapping,sample_weights
from run_tabicl import metrics


def prediction(c,method,mode,rule=0):
    if method=='Random':
        return c['raw_bank' if mode==0 else 'bank'].mean(0)
    return c['raw_dense' if mode==0 else 'dense'][rule]


def fit_rule(cells,mode):
    total=np.zeros(6)
    for c,w in zip(cells,sample_weights(cells)):
        outputs=c['raw_dense' if mode==0 else 'dense']
        true=outputs[:,np.arange(len(c['y'])),c['y'].astype(int)]
        total+=(-np.log(np.clip(true,1e-12,1))*w[None]).sum(1)
    return int(total.argmin()),total.tolist()


def clean_scores(cells,method=None,mode=0,rule=0):
    rows=[]
    for c in cells:
        if c['meta']['rho']==0:
            p=c['base'] if method is None else prediction(c,method,mode,rule)
            rows.append(metrics(c['y'],p,'classification',1.))
    return {k:float(np.mean([r[k] for r in rows])) for k in ['accuracy','auc']}


def objective(cells,method,mode,rule):
    total=0.
    for c,w in zip(cells,sample_weights(cells)):
        p=prediction(c,method,mode,rule);y=c['y'];aucs=[]
        for label in range(p.shape[1]):
            mask=y==label;n=int(mask.sum());other=len(y)-n
            if n and other:
                ranks=rankdata(p[:,label],method='average')
                aucs.append((float(ranks[mask].sum())-n*(n+1)/2)/(n*other))
        acc=float(np.mean(p.argmax(1)==y));auc=float(np.mean(aucs)) if aucs else .5
        total-=float(w.sum())*.5*(acc+auc)
    return total


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--track',required=True,choices=['five_feature_calibrated','seven_reimplemented'])
    parser.add_argument('--fold',required=True,type=int)
    parser.add_argument('--joint-rules',action='store_true',help='Validate each fixed fusion rule under the same Clean constraint')
    args=parser.parse_args()
    track=args.track;expected=1512 if track.startswith('five') else 1015
    root=RUN/'results/light_random_dense/tabicl'/track/'classification'
    paths=sorted(root.glob('*/*.npz'))
    assert len(paths)==expected,(len(paths),expected)
    mapping,source=fold_mapping('classification',track);cells=[]
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as f:
            cell={k:f[k] for k in ['bank','dense','base','raw_bank','raw_dense']}
            cell.update(y=f['y_target'],meta=meta,path=path,fold=mapping[meta['data_id']],
                clean_weight=(5. if track.startswith('five') else 7.) if meta['rho']==0 else 1.)
        cells.append(cell)
    train=[c for c in cells if c['fold']!=args.fold];test=[c for c in cells if c['fold']==args.fold]
    train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test}
    assert train_ids.isdisjoint(test_ids)
    inner=min(c['fold'] for c in train)
    tr=[c for c in train if c['fold']!=inner];val=[c for c in train if c['fold']==inner]
    base_clean=clean_scores(val)
    for method in ['Random','DENSE']:
        if args.joint_rules and method!='DENSE':continue
        root_name='combiners_joint_rules' if args.joint_rules else 'combiners'
        out=RUN/'results/light_random_dense'/root_name/track/f'fold_{args.fold}_{method}.json'
        if out.exists():continue
        trials=[];best=None
        for mode in [0,1]:
            trained_rule,losses=fit_rule(tr,mode) if method=='DENSE' else (0,None)
            rules=range(6) if args.joint_rules else [trained_rule]
            for rule in rules:
                value=objective(val,method,mode,rule);clean=clean_scores(val,method,mode,rule)
                feasible=all(clean[k]>=base_clean[k]-.005 for k in clean)
                trials.append({'mode':mode,'rule':rule,'rule_training_losses':losses,
                    'validation_objective':value,'clean_validation':clean,'clean_feasible':feasible})
                if feasible and (best is None or value<best[0]):best=(value,mode,rule)
        feasible_choice=best is not None
        # Preserve a genuine six-member baseline if neither config is feasible;
        # flag the unmet constraint, rather than silently collapsing to Base.
        mode=best[1] if best is not None else 0
        if args.joint_rules:
            rule=best[2] if best is not None else 0
            losses=None
        else:rule,losses=fit_rule(train,mode) if method=='DENSE' else (0,None)
        rows=[]
        for c in test:
            pred=prediction(c,method,mode,rule)
            rows.append(c['meta'] | {'method':method+'-6 light calibration','outer_fold':args.fold,
                'selected_mode':mode,'selected_rule':rule} | metrics(c['y'],pred,'classification',1.))
            prediction_root='predictions_joint_rules' if args.joint_rules else 'predictions'
            target=RUN/'results/light_random_dense'/prediction_root/track/method/str(c['meta']['task_id'])/c['path'].name
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=pred,y_target=c['y']);tmp.replace(target)
        atomic_json(out,{'status':'COMPLETE','seed':0,'track':track,'method':method,'fold':args.fold,
            'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'fold_map_source':source,
            'inner_validation_fold':inner,'selected_mode':mode,'mode_names':['original bootstrap','source-only clipping 1%-99%'],
            'selected_rule':rule,'rule_training_losses':losses,'trials':trials,
            'joint_rule_selection_on_inner_validation':args.joint_rules,
            'base_clean_validation':base_clean,'feasible_calibration_configuration_found':feasible_choice,
            'clean_tolerance':.005,'context_count':6,'rows_per_context':256,
            'Base_fallback_used':False,'outer_test_labels_used_for_fit_or_selection':False,
            'requested_Acc_bound_0565_used_for_selection':False,'paper_Shared_used_for_selection':False,
            'Shared_run':False,'rows':rows})
        print('LIGHT_FIT_DONE',track,args.fold,method,mode,rule,feasible_choice,flush=True)


if __name__=='__main__':main()
