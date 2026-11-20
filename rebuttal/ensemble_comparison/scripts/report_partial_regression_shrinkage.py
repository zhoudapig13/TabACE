"""A single fixed regularization check for canonical partial-resampling fusion.

Use 75% uniform weights plus 25% previously fitted OOF weights. The purpose is
Clean stability; no historical score bound enters fitting or the prediction map.
"""
from report_io import read_markdown,write_markdown
import csv
import json
import math
import numpy as np
from prepare_data import RUN,ROOT,atomic_json
from fit_combiners import fold_mapping
from run_tabicl import metrics


def main():
    root=RUN/'results/canonical_partial_010'
    original=json.loads((RUN/'results/canonical_partial_010_regression_summary.json').read_text())
    base=next(r for r in original['averages'] if r['method']=='base_protocol_check')
    raw_random=next(r for r in original['averages'] if r['method']=='bootstrap_mean_6')
    mapping,_=fold_mapping('regression','seven_calibrated_v2')
    rows=[];checked=0
    for fold in range(5):
        path=root/'combiners/seven_calibrated_v2/regression'/f'fold_{fold}.json'
        policy=json.loads(path.read_text())
        assert set(policy['train_data_ids']).isdisjoint(policy['test_data_ids'])
        assert not policy['outer_test_labels_used_for_fit_or_selection']
        fitted={}
        for method,key in [('GES-6 + 收缩','ges_weights'),('Convex-6 + 收缩','convex_weights')]:
            beta=.75*np.full(6,1/6)+.25*np.asarray(policy[key])
            assert np.all(beta>=0) and abs(beta.sum()-1)<1e-9
            fitted[method]=beta.tolist()
            for raw in sorted((root/'tabicl/seven_calibrated_v2/regression').glob('*/*.npz')):
                meta=json.loads(raw.with_suffix('.json').read_text())
                if mapping[meta['data_id']]!=fold:continue
                assert meta['data_id'] in policy['test_data_ids']
                with np.load(raw) as f:
                    bank,y=f['bootstrap_bank'],f['y_target']
                pred=beta@bank
                assert np.isfinite(pred).all()
                row=meta | {'method':method,'outer_fold':fold} | metrics(y,pred,'regression',max(float(y.std()),1e-12))
                rows.append(row);checked+=1
                out=root/'shrinkage025/predictions'/method/str(meta['task_id'])/raw.name
                out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,prediction=pred,y_target=y);tmp.replace(out)
        atomic_json(root/'shrinkage025/policies'/f'fold_{fold}.json',{
            'status':'COMPLETE','seed':0,'fold':fold,'context_count':6,'rows_per_context':256,
            'resampling_fraction':.1,'learned_weight_fraction':.25,'uniform_weight_fraction':.75,
            'train_data_ids':policy['train_data_ids'],'test_data_ids':policy['test_data_ids'],
            'weights':fitted,'outer_test_labels_used_for_fit_or_selection':False,
            'paper_Shared_or_requested_score_bounds_used_for_selection':False,
            'Shared_run':False,'unshrunk_versions_retained':True})
    assert checked==1914
    averages=[{'method':'Random-6（10%行替换）','clean':raw_random['clean'],'seven':raw_random['seven']}]
    for method in ['GES-6 + 收缩','Convex-6 + 收缩']:
        result={'method':method}
        for block,nonzero,count in [('clean',False,33),('seven',True,924)]:
            selected=[r for r in rows if r['method']==method and (r['rho']>0)==nonzero]
            assert len(selected)==count
            result[block]={k:math.fsum(r[k] for r in selected)/count for k in ['r2','nrmse']}
        averages.append(result)
    for r in averages:
        r['Clean_difference_vs_Base']={k:r['clean'][k]-base['clean'][k] for k in ['r2','nrmse']}
        r['Clean_within_005']=r['clean']['r2']>=base['clean']['r2']-.005 and r['clean']['nrmse']<=base['clean']['nrmse']+.005
        r['seven_both_metrics_better_than_Base']=r['seven']['r2']>base['seven']['r2'] and r['seven']['nrmse']<base['seven']['nrmse']
        r['seven_both_metrics_worse_than_historical_Shared']=r['seven']['r2']<.467 and r['seven']['nrmse']>.659
    summary={'status':'COMPLETE','scope':'Three canonical-input regression controls, not all requested baselines',
        'comparison_role':'Additional partial-resampling and fixed weight-shrinkage controls; not default Random/GES/Convex configurations',
        'seed':0,'datasets':33,'nonzero_conditions':924,'source_rows':256,'context_count':6,
        'context_resampling_fraction':.1,'learned_weight_fraction':.25,'Shared_run':False,
        'previous_full_pool_and_unshrunk_results_retained':True,
        'requested_score_bounds_used_for_prediction_or_fitting':False,
        'development_history':'Fixed shrinkage check added after previous aggregate Clean declined; not untouched confirmatory evaluation',
        'averages':averages,'Base':base,'prediction_metrics_computed':checked}
    atomic_json(RUN/'results/canonical_partial_regression_current_summary.json',summary)
    flat=[{'method':r['method']} | {b+'_'+k:v for b in ['clean','seven'] for k,v in r[b].items()} for r in averages]
    with (RUN/'results/canonical_partial_regression_current_averages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['<!-- CANONICAL_REGRESSION_SHRINKAGE -->','',
        '七偏移回归的同输入、保守重采样控制：固定源池256行，与Base相同；每成员仅替换10%的行，共6成员，种子0。',
        'GES/Convex再采用固定75%均匀权重＋25%原OOF权重；无额外历史/参考训练样本，无特征或标签适配。原始标准Bootstrap、完整训练池与未收缩融合结果全部保留。','',
        '| 方法 | 本组Clean R²↑ / nRMSE↓ | 七偏移 R²↑ / nRMSE↓ | Clean降幅≤0.005 |','|---|---:|---:|---|',
        f"| Base | {base['clean']['r2']:.4f} / {base['clean']['nrmse']:.4f} | {base['seven']['r2']:.4f} / {base['seven']['nrmse']:.4f} | 是 |"]
    for r in averages:
        c,s=r['clean'],r['seven']
        lines.append(f"| {r['method']} | {c['r2']:.4f} / {c['nrmse']:.4f} | {s['r2']:.4f} / {s['nrmse']:.4f} | {'是' if r['Clean_within_005'] else '否'} |")
    lines+=['| Shared（论文历史参考） | — | 0.4670 / 0.6590 | — |','',
        'Clean列为本组33个回归数据集的Clean控制，不与五扰动35数据集的Clean混合。当前规则属于公开标注的保守适配，不作为原方法默认配置；历史Shared不是同次协议评测。',
        '<!-- END_CANONICAL_REGRESSION_SHRINKAGE -->','']
    doc=ROOT/'REBUTTAL_EXPERIMENTS.md';text=read_markdown(doc);start='<!-- CANONICAL_REGRESSION_SHRINKAGE -->';end='<!-- END_CANONICAL_REGRESSION_SHRINKAGE -->'
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+'\n'.join(lines)+after
    else:
        title,rest=text.split('\n',1);text=title+'\n\n'+'\n'.join(lines)+'\n'+rest
    write_markdown(text,doc)
    print(json.dumps(averages,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
