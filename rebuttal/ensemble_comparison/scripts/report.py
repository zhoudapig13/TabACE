"""Publish observed numbers, coverage, provenance and full metric rows in one log."""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from prepare_data import ROOT,RUN,atomic_json

MARK='<!-- ENSEMBLE_RUN_RESULTS_START -->'


def number(x):
    return '—' if x is None or not math.isfinite(float(x)) else f'{float(x):.6f}'


def read_new():
    rows=[]
    for backbone in ['tabicl','tabpfn','tabm']:
        roots=[RUN/'results'/backbone,RUN/'results/reference'/backbone,RUN/'results/feature_views'/backbone,RUN/'results/oligo_extended'/backbone,RUN/'results/dense_full_context'/backbone,RUN/'results/reference_specific'/backbone]
        paths=[p for root in roots for p in root.rglob('*.json')]
        paths=[p for p in paths if '.tmp' not in p.name]
        with ThreadPoolExecutor(max_workers=24) as pool:
            entries=list(pool.map(lambda p:json.loads(p.read_text()),paths))
        for path,j in zip(paths,entries):
            if j.get('status')!='COMPLETE' or 'methods' not in j:continue
            for method,values in j['methods'].items():
                if j['task']=='regression' and j.get('bootstrap_revision',1)<2 and method in ['bootstrap_mean_6','oligoicp_iqr_6_keep1_adapted']:
                    continue
                rows.append({k:j.get(k) for k in ['track','task','task_id','data_id','dataset','family','rho','seed']}|
                            {'backbone':backbone,'method':method,'source':str(path.relative_to(ROOT))}|values)
    for path in (RUN/'results/combiners').rglob('fold_*.json'):
        j=json.loads(path.read_text())
        for r in j['rows']:
            if r['method']=='Stacking_OOF' and r['task']=='regression':
                r=r|{'method':'Stacking_OOF_UNSTABLE_source_scale_debug'}
            rows.append(r|{'backbone':r.get('backbone',j.get('backbone','tabicl')),'seed':0,'source':str(path.relative_to(ROOT))})
    return rows


def aggregate(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['track']=='five':block='clean' if int(r['rho'])==0 else 'perturbations_available_variant'
        elif r['track']=='five_corrected':block='clean_polarity_corrected' if int(r['rho'])==0 else 'perturbations_polarity_corrected'
        else:block='clean_shifts_reimplemented' if int(r['rho'])==0 else 'shifts_reimplemented'
        for metric in ['accuracy','auc','r2','nrmse']:
            if r.get(metric) is not None:
                grouped[(r['backbone'],r['method'],r['task'],block,metric)][r['data_id']].append(float(r[metric]))
    result=[]
    for key,groups in sorted(grouped.items()):
        backbone,method,task,block,metric=key
        # Seven clean anchors belong to that reimplemented protocol, not main-table Clean.
        values=[float(np.mean(x)) for x in groups.values()]
        five_block=block in ['clean','perturbations_available_variant','clean_polarity_corrected','perturbations_polarity_corrected']
        expected_datasets=(72 if task=='classification' else 35) if five_block else (35 if task=='classification' else 33)
        expected_per_dataset=1 if block.startswith('clean') else 20 if five_block else 28
        if block.startswith('clean') and 'Specific' in method:expected_per_dataset=5 if five_block else 7
        result.append({'backbone':backbone,'method':method,'task':task,'block':block,
                       'metric':metric,'mean':float(np.mean(values)),'datasets':len(groups),
                       'conditions':sum(map(len,groups.values())),'complete':
                       len(groups)==expected_datasets and all(len(v)==expected_per_dataset for v in groups.values())})
    return result


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def strength_summary(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['task']=='classification' and r['track']=='five':continue
        for metric in ['accuracy','auc','r2','nrmse']:
            if r.get(metric) is not None:
                key=(r['backbone'],r['method'],r['track'],r['task'],int(r['rho']),metric)
                groups[key][r['data_id']].append(float(r[metric]))
    result=[]
    for key,datasets in sorted(groups.items()):
        b,m,track,task,rho,metric=key
        result.append({'backbone':b,'method':m,'track':track,'task':task,'rho':rho/100.,'metric':metric,
                       'mean':float(np.mean([np.mean(x) for x in datasets.values()])),
                       'datasets':len(datasets),'conditions':sum(map(len,datasets.values()))})
    return result


def old_table():
    # Read the existing table; these are explicitly existing rounded reference values.
    lines=(ROOT/'paper/tables/overall_results.tex').read_text().splitlines()
    import re
    current=None;rows=[]
    for i,line in enumerate(lines,1):
        if '\\' not in line or '&' not in line:continue
        first=line.split('&')[0].strip()
        if first=='TabM':label='TabM (existing)'
        elif first in ['TabICL-v2','TabPFN-v3']:
            current=first;label=current+' Base (existing)'
        elif r'\method{}-shared' in first and current:label=current+' TabACE-Shared (existing)'
        elif r'\method{}-specific' in first and current:label=current+' TabACE-Specific (existing)'
        elif first in ['LimiX-2M','TabDPT v1.2']:
            current=None;continue
        else:continue
        values=[]
        for c in line.split('&')[1:]:
            matches=re.findall(r'(?<![\w.])-?\d+\.\d+',c)
            if matches:values.append(float(matches[0]))
        if len(values)==12:rows.append({'method':label,'values':values,'source_line':i})
    return rows


def primary_matrix(summary):
    """Main-table shape with the new protocol blocks explicitly identified."""
    lookup={(r['backbone'],r['method'],r['task'],r['block'],r['metric']):r for r in summary}
    methods=[
      ('tabicl','base_protocol_check','Base (1)'),
      ('tabicl','native_6','Native (6)'),('tabicl','native_8','Native (8)'),
      ('tabicl','bootstrap_mean_6','Bootstrap (6)'),
      ('tabicl','DENSE_fullcontext_OOF_selected_adapted','DENSE full-context / OOF rule (6, adapted)'),
      ('tabicl','oligoicp_iqr_6_keep1_adapted','OligoICP IQR (6 → 1, adapted)'),
      ('tabicl','OligoICP_IQR40_keep4_adapted','OligoICP IQR (40 → 4, adapted)'),
      ('tabicl','Bootstrap_mean40_extended_budget','Bootstrap (40)'),
      ('tabicl','OligoICP_IQR400_keep40_adapted','OligoICP IQR (400 → 40, adapted)'),
      ('tabicl','Bootstrap_mean400_extended_budget','Bootstrap (400)'),
      ('tabicl','GES_25_metric_normalized','GES (6, OOF)'),
      ('tabicl','Stacking_convex_OOF','Convex stacking (6, OOF)'),
      ('tabicl','Stacking_OOF','Log-linear / ridge stacking (6, OOF; debug rows excluded)'),
      ('tabicl','TabFM_feature_views6_NNLS_adapted','TabFM feature views / NNLS (6, adapted)'),
      ('tabicl','divide_conquer_feature_adapted','Feature divide-and-conquer (≤6, adapted)'),
      ('tabicl','Shared_context_mean_reimplemented','Shared-context mean (5, reimplemented)'),
      ('tabicl','TabACE_Shared_R5_reimplemented','TabACE Shared (Base + 5, reimplemented)'),
      ('tabicl','Specific_context_mean_reimplemented','Specific-context mean (5, reimplemented)'),
      ('tabicl','TabACE_Specific_R5_reimplemented','TabACE Specific (Base + 5, reimplemented)'),
      ('tabpfn','tabpfn_v3_base_check','Base (1)'),
      ('tabpfn','tabpfn_native_6','Native (6)'),('tabpfn','tabpfn_native_8','Native (8)'),
      ('tabpfn','tabpfn_bootstrap_mean_6','Bootstrap (6)'),
      ('tabpfn','PHE_official_custom6_native_source_calibration','PHE (6 configs, official custom portfolio)'),
      ('tabpfn','RF_PFN_official6_no_adaptive_tree','RF-PFN (6, no adaptive tree)'),
      ('tabpfn','BETA_bootstrap_component_only_no_adaptation','BETA bootstrap component (6, no adapter)'),
      ('tabpfn','MixturePFN_MICP_no_CaPFN','MixturePFN MICP (no CaPFN)'),
      ('tabpfn','TabACE_Shared_R5_reimplemented','TabACE Shared (Base + 5, reimplemented)'),
      ('tabpfn','TabACE_Specific_R5_reimplemented','TabACE Specific (Base + 5, reimplemented)'),
      ('tabm','TabM_official_source_refit','TabM (official defaults, source fit)')]
    columns=[('classification','clean_polarity_corrected','accuracy'),('classification','clean_polarity_corrected','auc'),
      ('regression','clean','r2'),('regression','clean','nrmse'),
      ('classification','perturbations_polarity_corrected','accuracy'),('classification','perturbations_polarity_corrected','auc'),
      ('regression','perturbations_available_variant','r2'),('regression','perturbations_available_variant','nrmse'),
      ('classification','shifts_reimplemented','accuracy'),('classification','shifts_reimplemented','auc'),
      ('regression','shifts_reimplemented','r2'),('regression','shifts_reimplemented','nrmse')]
    parts=['','### 10.2a 本轮同条件对照：主表格式','',
      '列顺序与主表相同。分类五扰动为方向修复后的 standalone 协议，回归五扰动为现存 standalone 协议；七偏移为附录重建协议。它们均未被验证为原主表精确复现，因此仅在本轮同列内比较。',
      '每项为单种子 0，数据集内平均全部非零强度再按数据集宏平均；不支持的任务标为 —，尚未跑齐标为运行中。Specific 的 Clean 对五个家族各自零强度策略取平均。候选数不同的 OligoICP 与对应随机平均并列；其成本包括所有候选推理。',
      '', '| 骨干 | 方法 / 推理预算 | Clean Acc | Clean AUC | Clean R² | Clean nRMSE | 扰动 Acc | 扰动 AUC | 扰动 R² | 扰动 nRMSE | 新偏移 Acc | 新偏移 AUC | 新偏移 R² | 新偏移 nRMSE |',
      '| --- | --- |'+' ---: |'*12]
    records=[]
    for backbone,method,label in methods:
        values=[];record={'backbone':backbone,'method':method,'label':label}
        for task,block,metric in columns:
            r=lookup.get((backbone,method,task,block,metric))
            values.append('—' if r is None else f"{r['mean']:.3f}" if r['complete'] else '运行中')
            record[f'{task}_{block}_{metric}']=None if r is None or not r['complete'] else r['mean']
        parts.append('| '+backbone+' | '+label+' | '+' | '.join(values)+' |');records.append(record)
    write_csv(RUN/'results/main_table_current_protocol.csv',records)
    return parts


def observed_comparisons(summary):
    lookup={(r['backbone'],r['method'],r['task'],r['block'],r['metric']):r for r in summary if r['complete']}
    candidates={'tabicl':['native_6','bootstrap_mean_6','DENSE_fullcontext_OOF_selected_adapted',
        'oligoicp_iqr_6_keep1_adapted','GES_25_metric_normalized','Stacking_convex_OOF',
        'Stacking_OOF','TabFM_feature_views6_NNLS_adapted','divide_conquer_feature_adapted'],
        'tabpfn':['tabpfn_native_6','tabpfn_bootstrap_mean_6','PHE_official_custom6_native_source_calibration',
        'RF_PFN_official6_no_adaptive_tree','BETA_bootstrap_component_only_no_adaptation','MixturePFN_MICP_no_CaPFN']}
    parts=['','### 10.2b 已观测的主指标差值','',
        '以各列实际观测值列出最多 6 个 TFM 成员预算组中的最强对照；此处只作结果描述，不用于选择协议、参数或丢弃结果。40/400 候选的 OligoICP 和随机平均单列于上表。Δ 为 TabACE Specific 重实现减去该对照，Acc/R² 均为越大越好；不声称单种子差异具有统计显著性。','',
        '| 骨干 | 协议块 | 任务 | 指标 | 最强对照（本列观测） | 对照值 | TabACE Specific 重实现 | Δ |',
        '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for backbone in ['tabicl','tabpfn']:
        for task,metric in [('classification','accuracy'),('regression','r2')]:
            for block in [('perturbations_polarity_corrected' if task=='classification' else 'perturbations_available_variant'),'shifts_reimplemented']:
                reference=lookup.get((backbone,'TabACE_Specific_R5_reimplemented',task,block,metric))
                values=[lookup[(backbone,m,task,block,metric)] for m in candidates[backbone] if (backbone,m,task,block,metric) in lookup]
                if reference is None or not values:continue
                best=max(values,key=lambda x:x['mean']);delta=reference['mean']-best['mean']
                parts.append('| '+' | '.join([backbone,block,task,metric,best['method'],number(best['mean']),number(reference['mean']),number(delta)])+' |')
    return parts


def main():
    rows=read_new();summary=aggregate(rows);strengths=strength_summary(rows)
    write_csv(RUN/'results/all_detail.csv',rows)
    write_csv(RUN/'results/summary_long.csv',summary)
    write_csv(RUN/'results/summary_strength.csv',strengths)
    counts={}
    for backbone in ['tabicl','tabpfn','tabm']:
        for track in ['five','five_corrected','seven_reimplemented']:
            for task in ['classification','regression']:
                if track=='five_corrected' and task!='classification':continue
                counts[f'{backbone}_{track}_{task}']=len([p for p in (RUN/'results'/backbone/track/task).rglob('*.json') if '.tmp' not in p.name])
    audits={}
    for name in ['raw_result_audit','oof_result_audit','classification_source_fix_audit','seven_protocol_audit']:
        p=RUN/'results'/f'{name}.json'
        if p.exists():audits[name]=json.loads(p.read_text()).get('status')
    complete=all(audits.get(k)=='PASS' for k in ['raw_result_audit','oof_result_audit','classification_source_fix_audit','seven_protocol_audit'])
    status={'updated_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'seed':0,'five_original_operator_match':False,'five_main_table_comparable':False,'cell_counts':counts,'expected_five_cells':{'classification':1512,'regression':735},
            'expected_seven_cells':{'classification':1015,'regression':957},
            'seven_original_source_available':False,'seven_reimplementation_authorized':True,
            'detail_rows':len(rows),'audits':audits,'status':'COMPLETE_CURRENT_PROTOCOL' if complete else 'IN_PROGRESS'}
    atomic_json(RUN/'results/status.json',status)
    primary_matrix(summary)
    from write_average_tables import write_report
    write_report(RUN/'results/main_table_current_protocol.csv', ROOT/'REBUTTAL_EXPERIMENTS.md')
    print(json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
