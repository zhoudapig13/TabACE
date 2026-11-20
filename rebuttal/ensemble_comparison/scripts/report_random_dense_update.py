"""Report and audit every reduced-adaptation attempt; preserve earlier results."""
from report_io import read_markdown,write_markdown
import csv
import hashlib
import json
import math
from collections import Counter
import numpy as np
from prepare_data import RUN,ROOT,atomic_json
from fit_combiners import fold_mapping
from run_tabicl import metrics


def average(rows,nonzero,n):
    selected=[r for r in rows if (r['rho']>0)==nonzero]
    assert len(selected)==n
    assert len({(r['task_id'],r['family'],r['rho']) for r in selected})==n
    return {k:math.fsum(r[k] for r in selected)/n for k in ['accuracy','auc']}


def main():
    raw=json.loads((RUN/'results/final_ensemble_summary.json').read_text())['records']
    base=next(r for r in raw if r['backbone']=='tabicl' and r['task']=='classification' and r['method']=='base_protocol_check')
    stages=[('light_source_clipping','light_random_dense','combiners','predictions'),
            ('moderate_features75','moderate_random_dense','combiners','predictions'),
            ('moderate_features50','moderate_random_dense_subspace50','combiners','predictions')]
    records=[];checked_policies=checked_metrics=checked_accuracy=0
    for stage,result_root,policy_root,prediction_root in stages:
        for method in ['Random','DENSE']:
            item={'stage':stage,'method':method,'seed':0,'context_count':6}
            for track,n,families,block in [('five_feature_calibrated',72,5,'five'),('seven_reimplemented',35,7,'seven')]:
                proot='combiners_joint_rules' if stage=='light_source_clipping' and method=='DENSE' else policy_root
                droot='predictions_joint_rules' if stage=='light_source_clipping' and method=='DENSE' else prediction_root
                paths=sorted((RUN/'results'/result_root/proot/track).glob(f'fold_*_{method}.json'))
                assert len(paths)==5,(stage,method,track,len(paths))
                policies=[json.loads(p.read_text()) for p in paths]
                rows=[r for p in policies for r in p['rows']]
                assert len(rows)==n*(1+families*4)
                mapping,_=fold_mapping('classification',track)
                for policy in policies:
                    train=policy.get('calibration_data_ids',policy.get('train_data_ids'))
                    assert set(train).isdisjoint(policy['test_data_ids'])
                    assert all(mapping[i]!=policy['fold'] for i in train)
                    assert all(mapping[i]==policy['fold'] for i in policy['test_data_ids'])
                    assert policy['context_count']==6 and not policy['Base_fallback_used']
                    assert not policy['Shared_run']
                    if 'test_labels_used_for_config_selection' in policy:
                        assert not policy['test_labels_used_for_config_selection']
                        assert not policy['Shared_or_requested_Acc_bound_used_for_selection']
                    else:
                        assert not policy['outer_test_labels_used_for_fit_or_selection']
                        assert not policy['requested_Acc_bound_0565_used_for_selection']
                    prows=policy['rows']
                    sample={0,len(prows)//2,len(prows)-1}
                    for idx,row in enumerate(prows):
                        path=RUN/'results'/result_root/droot/track/method/str(row['task_id'])/f"{row['family']}_{row['rho']}.npz"
                        with np.load(path) as f:
                            p,y=f['prediction'],f['y_target']
                        assert np.isfinite(p).all() and (p>=0).all()
                        assert np.max(np.abs(p.sum(1)-1))<1e-9
                        assert abs(float(np.mean(p.argmax(1)==y))-row['accuracy'])<1e-12
                        checked_accuracy+=1
                        if idx in sample:
                            observed=metrics(y,p,'classification',1.)
                            assert abs(observed['auc']-row['auc'])<1e-10
                            checked_metrics+=1
                    checked_policies+=1
                item[block]=average(rows,True,n*4*families)
                clean=average(rows,False,n)
                item[block+'_Clean']=clean
                if block=='five':item['clean']=clean
                base_clean=[]
                for p in (RUN/'results/tabicl'/track/'classification').glob('*/*.json'):
                    m=json.loads(p.read_text())
                    if m['rho']==0:base_clean.append(m['methods']['base_protocol_check'])
                assert len(base_clean)==n
                base_clean={k:math.fsum(r[k] for r in base_clean)/n for k in clean}
                item[block+'_Clean_difference_vs_Base']={k:clean[k]-base_clean[k] for k in clean}
                item[block+'_Clean_within_005']=all(clean[k]>=base_clean[k]-.005 for k in clean)
                item[block+'_difference_vs_Base']={k:item[block][k]-base[block]['metrics'][k] for k in clean}
                item[block+'_patterns']=dict(Counter(str(p.get('selected_pattern',p.get('selected_mode'))) for p in policies))
                item[block+'_rules']=dict(Counter(str(p['selected_rule']) for p in policies))
                item[block+'_calibration_infeasible_folds']=[p['fold'] for p in policies if not p.get('calibration_feasible_configuration_found',p.get('feasible_calibration_configuration_found',False))]
            records.append(item)
    current=[r for r in records if r['stage']=='moderate_features50']
    for r in current:
        r['five_in_user_preferred_range']=.510<=r['five']['accuracy']<=.540
        r['both_blocks_improve_Acc_and_AUC_vs_current_Base']=all(v>0 for b in ['five','seven'] for v in r[b+'_difference_vs_Base'].values())
        r['both_blocks_below_historical_Shared']=all(r[b][k]<{'five':{'accuracy':.575,'auc':.720},'seven':{'accuracy':.772,'auc':.831}}[b][k] for b in ['five','seven'] for k in ['accuracy','auc'])
    assert all(r['five_in_user_preferred_range'] and r['both_blocks_improve_Acc_and_AUC_vs_current_Base']
               and r['both_blocks_below_historical_Shared'] and r['five_Clean_within_005'] and r['seven_Clean_within_005'] for r in current)
    summary={'status':'COMPLETE','scope':'Random/DENSE classification adaptation degree, not all ensemble tasks',
        'seed':0,'Shared_run':False,'current':current,'reduced_adaptation_attempts':records,
        'earlier_strong_attempts_retained_in':'results/classification_adaptation_summary.json',
        'requested_range_not_used_for_per_fold_fitting_or_selection':True,
        'development_benchmark':'Candidate families were expanded after observing earlier aggregate scores; this is not untouched confirmatory testing.',
        'audits':{'policies_with_disjoint_folds':checked_policies,'all_saved_accuracy_metrics_recomputed':checked_accuracy,
                  'sampled_AUC_metrics_recomputed':checked_metrics},
        'code_sha256':{n:hashlib.sha256((RUN/'scripts'/n).read_bytes()).hexdigest() for n in ['fit_moderate_random_dense.py','report_random_dense_update.py','robust_six_views.py']}}
    atomic_json(RUN/'results/random_dense_current_summary.json',summary)
    flattened=[]
    for r in records:
        flattened.append({k:r[k] for k in ['stage','method','seed','context_count']} | {
            b+'_'+k:v for b in ['clean','five','seven'] for k,v in r[b].items()})
    with (RUN/'results/random_dense_reduced_adaptation_averages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    def fmt(d):return f"{d['accuracy']:.4f} / {d['auc']:.4f}"
    lines=['<!-- RANDOM_DENSE_CURRENT -->','',
        'Random / DENSE 最新的有限子空间适配（种子0；Acc / AUC；每个方法固定6个成员，每成员256行；无Base回退）。',
        '从6个普通Bootstrap成员出发，最多替换2个为源分位变换或75%/50%低漂移特征子空间成员。特征漂移仅用未标注查询特征估计；不使用目标分布对齐、协方差修复、近邻修复或近期样本分支。',
        '按数据集五折留出，每折在其余四折的固定候选中选配置，Clean Acc/AUC优先满足Base−0.005，再最大化平均Acc/AUC；无可行配置时保留原始六成员并标注，最终Clean限额单独核验。未使用测试分数区间或Shared分数选配置。','',
        '| 方法 | Clean | 五扰动 | 七偏移 |','|---|---:|---:|---:|',
        '| Base（本轮同协议） | '+' | '.join(fmt(base[b]['metrics']) for b in ['clean','five','seven'])+' |']
    for r in current:lines.append('| '+r['method']+'-6 + 有限子空间适配 | '+' | '.join(fmt(r[b]) for b in ['clean','five','seven'])+' |')
    lines+=['| Shared（论文历史参考；未重跑） | 0.8040 / 0.9010 | 0.5750 / 0.7200 | 0.7720 / 0.8310 |','',
        '本轮两个方法的五扰动Acc均在用户希望的0.510–0.540之间，五扰动及七偏移的Acc/AUC均高于本轮Base，Clean各指标下降均≤0.005。历史Shared并非同次协议评测，不能据此证明同协议优劣。',
        '此前强适配、仅截断及75%特征版本均保留。候选族曾在观察开发集汇总后扩展，因此本结果属于开发实验，不视为未触碰测试集的确认结果。',
        '[各次适配平均表](rebuttal/ensemble_comparison/results/random_dense_reduced_adaptation_averages.csv)。',
        '<!-- END_RANDOM_DENSE_CURRENT -->','']
    doc=ROOT/'REBUTTAL_EXPERIMENTS.md';text=read_markdown(doc);start='<!-- RANDOM_DENSE_CURRENT -->';end='<!-- END_RANDOM_DENSE_CURRENT -->'
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+'\n'.join(lines)+after
    else:
        title,rest=text.split('\n',1);text=title+'\n\n'+'\n'.join(lines)+'\n'+rest
    write_markdown(text,doc)
    print(json.dumps({'current':current,'audit':summary['audits']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
