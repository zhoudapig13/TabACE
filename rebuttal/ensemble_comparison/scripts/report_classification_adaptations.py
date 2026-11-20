"""Report all adaptation attempts; freeze complete primary calibration policies."""
from report_io import read_markdown,write_markdown
import csv,json,math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from prepare_data import RUN,ROOT,atomic_json

METHODS=['Native','Random','DENSE','GES','Convex','Loglinear']
LABELS={'Native':'Native-6 + 适配','Random':'Random-6 + 适配','DENSE':'DENSE-6 + 适配',
    'GES':'GES-6 + 适配','Convex':'Convex Stacking-6 + 适配','Loglinear':'Log-linear Stacking-6 + 适配'}
RAW={'Native':'native_6','Random':'bootstrap_mean_6','DENSE':'DENSE_fullcontext_OOF_selected_adapted',
    'GES':'GES_25_metric_normalized','Convex':'Stacking_convex_OOF','Loglinear':'Stacking_OOF'}


def average(rows,nonzero,expected):
    rows=[r for r in rows if (r['rho']>0)==nonzero];assert len(rows)==expected,(len(rows),expected)
    return {k:math.fsum(r[k] for r in rows)/len(rows) for k in ['accuracy','auc']}


def main():
    old=json.loads((RUN/'results/final_ensemble_summary.json').read_text())['records'];raw={}
    for r in old:
        if r['backbone']=='tabicl' and r['task']=='classification':raw[r['method']]=r
    records=[];all_variants=[];policy_count=0
    for method in METHODS:
        result={'method':method,'label':LABELS[method],'seed':0,'candidate_budget':6}
        for track,n,nf,block in [('five_feature_calibrated',72,5,'five'),('seven_reimplemented',35,7,'seven')]:
            # Same user-requested Clean-stability rule for every method.
            # Historical Shared scores do not select configurations.
            root='robust_six_combiners_stable'
            files=sorted((RUN/'results'/root/track/'classification').glob(f'fold_*_{method}.json'))
            assert len(files)==5,(method,track,len(files))
            policies=[json.loads(p.read_text()) for p in files];rows=[r for p in policies for r in p['rows']]
            assert len(rows)==n*(1+4*nf)
            for p in policies:
                assert set(p['train_data_ids']).isdisjoint(p['test_data_ids'])
                assert not p['outer_test_labels_used_for_fit_or_selection'] and not p['paper_Shared_values_used_for_fit_or_selection']
                policy_count+=1
            observed=average(rows,True,n*4*nf);clean=average(rows,False,n)
            result[block]=observed
            if block=='five':result['clean']=clean
            result[block+'_clean']=clean
            base_clean_rows=[]
            for path in (RUN/'results/tabicl'/track/'classification').glob('*/*.json'):
                m=json.loads(path.read_text())
                if m['rho']==0:base_clean_rows.append(m['methods']['base_protocol_check'])
            assert len(base_clean_rows)==n
            base_clean={k:math.fsum(r[k] for r in base_clean_rows)/n for k in clean}
            result[block+'_clean_difference_vs_backbone']={k:clean[k]-base_clean[k] for k in clean}
            result[block+'_clean_decline_at_most_005']=all(clean[k]>=base_clean[k]-.005 for k in clean)
            result[block+'_calibration_root']=root
            result[block+'_selected_modes']=[p['selected_mode'] for p in policies]
            base=raw['base_protocol_check'][block]['metrics'];reference={'five':{'accuracy':.575,'auc':.720},'seven':{'accuracy':.772,'auc':.831}}[block]
            result[block+'_difference_vs_backbone']={k:observed[k]-base[k] for k in observed}
            result[block+'_strictly_below_paper_Shared']=all(observed[k]<reference[k] for k in observed)
            result[block+'_at_least_backbone']=all(observed[k]>=base[k] for k in observed)
            result[block+'_near_or_above_backbone_005']=all(observed[k]>=base[k]-.005 for k in observed)
            for variant_root in ['robust_six_combiners','robust_six_combiners_acc_auc','robust_six_combiners_stable']:
                variants=sorted((RUN/'results'/variant_root/track/'classification').glob(f'fold_*_{method}.json'))
                if len(variants)!=5:continue
                vr=[r for path in variants for r in json.loads(path.read_text())['rows']]
                all_variants.append({'method':method,'track':track,'variant':variant_root,
                    'clean':average(vr,False,n),'shift':average(vr,True,n*4*nf)})
        records.append(result)
    summary={'status':'COMPLETE','scope':'classification adaptation phase','seed':0,'Shared_run':False,
        'paper_Shared_reference_only':True,'protocol_changed':False,'primary':records,'all_variants':all_variants,
        'checked_primary_OOF_policies':policy_count,'near_tolerance_reported_separately':.005,
        'selection_uses_Shared_bounds':False,'all_variant_results_retained':True,
        'primary_policy_description':'All methods use inner-calibration mean(Acc,AUC), subject to validation Clean Acc and AUC >= validation Base minus 0.005. Outer outcomes checked separately.',
        'user_Clean_absolute_decline_limit':.005,
        'unconstrained_versions_can_exceed_paper_reference':True}
    atomic_json(RUN/'results/classification_adaptation_summary.json',summary)
    table=[]
    for r in records:
        row={'method':r['method'],'label':r['label']}
        for block in ['clean','five','seven']:
            for key,value in r[block].items():row[block+'_'+key]=value
        table.append(row)
    with (RUN/'results/classification_adaptation_averages.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    all_flat=[]
    for r in all_variants:all_flat.append({k:r[k] for k in ['method','track','variant']}|
        {block+'_'+metric:value for block in ['clean','shift'] for metric,value in r[block].items()})
    with (RUN/'results/classification_adaptation_all_variants.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_flat[0]));w.writeheader();w.writerows(all_flat)
    def fmt(d):return f"{d['accuracy']:.3f} / {d['auc']:.3f}"
    parts=['<!-- CLASSIFICATION_ADAPTED_SIX -->','',
        '早先的TabICL重适配平均（归档；'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')+'；单种子0；Acc / AUC）。Random/DENSE当前有限子空间版见文档开头，以下分数原样保留。',
        '数据与扰动保持冻结，成员上限6。采用无标签目标特征预处理、特征子空间、源数据回退和嵌套数据集OOF校准；Shared未运行。','',
        '各方法统一按内层校准集的平均Acc/AUC选配置，要求校准集Clean Acc和AUC相对Base下降≤0.005；最终实测降幅另行核验。不以低于论文Shared作为选择条件。该协议属于缺失代码后的重建与开发实验，论文Shared只作历史参考。','',
        '| 方法 | Clean | 五扰动 | 七偏移 |','|---|---:|---:|---:|',
        '| Base | '+ ' | '.join(fmt(raw['base_protocol_check'][b]['metrics']) for b in ['clean','five','seven'])+' |']
    for r in records:parts.append('| '+r['label']+' | '+' | '.join(fmt(r[b]) for b in ['clean','five','seven'])+' |')
    parts+=['| Shared（论文参考） | 0.804 / 0.901 | 0.575 / 0.720 | 0.772 / 0.831 |','',
        'DENSE在部分校准折选择均值融合，因此可能与Random同分。DENSE七偏移AUC略低于Base，不标为全面提升。部分无Clean约束变体的五扰动成绩超过论文Shared历史参考，不能隐去或解释为原协议复现。','',
        '取消Clean稳定性约束的适配变体（完整保留；部分超过论文Shared历史参考）：','',
        '| 方法 | Clean | 五扰动 | 七偏移 |','|---|---:|---:|---:|']
    for method in METHODS:
        variants=[r for r in all_variants if r['method']==method and r['variant']=='robust_six_combiners_acc_auc']
        five=next(r for r in variants if r['track']=='five_feature_calibrated')
        seven=next(r for r in variants if r['track']=='seven_reimplemented')
        parts.append('| '+LABELS[method]+'（无Clean约束） | '+' | '.join(fmt(v) for v in [five['clean'],five['shift'],seven['shift']])+' |')
    parts+=['',
        '[当前适配均值](rebuttal/ensemble_comparison/results/classification_adaptation_averages.csv) · '
        '[全部适配版本均值](rebuttal/ensemble_comparison/results/classification_adaptation_all_variants.csv)。',
        '<!-- END_CLASSIFICATION_ADAPTED_SIX -->','']
    section='\n'.join(parts);doc=ROOT/'REBUTTAL_EXPERIMENTS.md';text=read_markdown(doc);start='<!-- CLASSIFICATION_ADAPTED_SIX -->';end='<!-- END_CLASSIFICATION_ADAPTED_SIX -->'
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+section+after
    else:
        first,rest=text.split('\n',1);text=first+'\n\n'+section+'\n'+rest
    write_markdown(text,doc)
    print(json.dumps({'status':'COMPLETE','primary':records},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
