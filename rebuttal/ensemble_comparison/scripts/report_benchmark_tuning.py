"""Compact averages for development tuning and disjoint-query validation."""
from report_io import read_markdown,write_markdown
import csv, json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from prepare_data import RUN, ROOT, atomic_json

BLOCKS=[('five_calibrated','classification',72,'五扰动：source-label stress'),
    ('five_feature_calibrated','classification',72,'五扰动：feature-only'),
    ('five','regression',35,'五扰动：回归'),
    ('seven_reimplemented','classification',35,'七偏移：分类'),
    ('seven_calibrated','regression',33,'七偏移：回归（幅度0.59）')]


def aggregate(rows,keys):
    # Each dataset receives equal weight; family and strength means are nested.
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get('rho',0)>0:groups[int(r['task_id'])][r['family']].append(r)
    if not groups:return None
    return {k:float(np.mean([np.mean([np.mean([r[k] for r in conditions]) for conditions in families.values()])
        for families in groups.values()])) for k in keys}


def main():
    entries=[]
    for bone in ['tabicl','tabpfn']:
        for track,task,expected,label in BLOCKS:
            keys=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
            raw=[];base_key='base_protocol_check' if bone=='tabicl' else 'tabpfn_v3_base_check'
            for f in (RUN/'results'/bone/track/task).glob('*/*.json'):
                m=json.loads(f.read_text())
                if base_key in m.get('methods',{}):raw.append(m|m['methods'][base_key])
            if not raw:
                for f in (RUN/'results/reference'/bone/track/task).glob('*/*.npz'):
                    m=json.loads(f.with_suffix('.json').read_text())
                    if m['rho']==0:continue
                    from run_tabicl import metrics
                    with np.load(f) as bank:values=metrics(bank['y_target'],bank['base'],task,
                        max(float(bank['y_target'].std()),1e-12) if task=='regression' else 1.)
                    raw.append(m|values)
            streams={'Base':raw}
            for variant in ['shared','specific']:
                values=[]
                for f in (RUN/'results/benchmark_tuning'/bone/track/task).glob(f'{variant}_*.json'):
                    values.extend(json.loads(f.read_text())['rows'])
                streams['Shared' if variant=='shared' else 'Specific_R5']=values
            unlimited=[];pool_counts={}
            pool_root='benchmark_specific_uncapped'
            full_root=RUN/'results/benchmark_specific_uncapped_full'/bone/track/task
            full_files=list(full_root.glob('specific_*.json'))
            if len(full_files)==(5 if track.startswith('five') else 7):pool_root='benchmark_specific_uncapped_full'
            for f in (RUN/'results'/pool_root/bone/track/task).glob('specific_*.json'):
                info=json.loads(f.read_text());unlimited.extend(info['rows'])
                pool_counts[info['family']]=info['deployed_alternative_branches']
            streams['Specific']=unlimited
            records={name:{'conditions':sum(r.get('rho',0)>0 for r in rows),
                'datasets':len({r['task_id'] for r in rows if r.get('rho',0)>0}),
                'metrics':aggregate(rows,keys)} for name,rows in streams.items()}
            required=expected*(20 if track.startswith('five') else 28)
            entries.append({'backbone':bone,'track':track,'task':task,'label':label,
                'expected_conditions':required,'expected_datasets':expected,
                'complete':all(v['conditions']==required for v in records.values()),
                'specific_pool_root':pool_root,'specific_context_cap':None,'specific_deployed_context_counts':pool_counts,
                'full_registry_pending':track in ['five','five_feature_calibrated'] and pool_root!='benchmark_specific_uncapped_full',
                'results_role':'benchmark development; Shared/Specific fitted on these query labels',**records})
    heldout=[]
    for bone in ['tabicl','tabpfn']:
        for track,task,expected,label in BLOCKS:
            if track not in ['five_calibrated','five_feature_calibrated','seven_calibrated']:continue
            rows=defaultdict(list)
            for f in (RUN/'results/benchmark_tuning_holdout'/bone/track/task).glob('*/*.json'):
                m=json.loads(f.read_text());assert not m['holdout_query_labels_used_for_fit_or_selection']
                for name,scores in m['methods'].items():rows[name].append(m|scores)
            keys=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
            expected=25 if task=='regression' else 56
            records={name:{'conditions':len(values),'datasets':len({r['task_id'] for r in values}),
                'metrics':aggregate(values,keys)} for name,values in rows.items()}
            required=expected*(28 if task=='regression' else 20)
            heldout.append({'backbone':bone,'track':track,'task':task,'label':label,'expected_datasets':expected,
                'expected_conditions':required,'complete':bool(records) and all(v['conditions']==required for v in records.values()),
                'methods':records,'results_role':'independent queries on seen benchmark datasets; not unseen datasets'})
    summary={'development':entries,'independent_queries':heldout,
        'exact_original_paper_protocol_restored':False,'target_ranking_enforced':False}
    atomic_json(RUN/'results/benchmark_tuning_summary.json',summary)
    def fmt(record,task):
        if not record or not record.get('metrics'):return '待完成'
        keys=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        return ' / '.join(f"{record['metrics'][k]:.3f}" for k in keys)
    parts=['','<!-- BENCHMARK_TUNING_TABLES -->','',
        'Benchmark 调参平均结果（'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M')+'）。',
        '以下 Shared/Specific 使用 benchmark 查询标签拟合，属于开发集分数；与上方未使用这些标签拟合的表分开解释。分类为 Acc / AUC，回归为 R² / nRMSE。',
        '本轮使用重建协议；接近原数值不等于恢复原主表实验设置。','',
        'Specific 不设 context 数量上限；R5 只保留为等预算诊断。当前池为已实现 Shared 与家族候选的并集，完整范围补跑时单独标记。','',
        '| 骨干 | 条件 | Base | Shared（调参） | Specific（无上限） | Specific R5（诊断） | 实际 R | 完整性 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for row in entries:
        coverage='完整' if row['complete'] else '进行中：'+','.join(f"{k}={row[k]['conditions']}/{row['expected_conditions']}" for k in ['Base','Shared','Specific'])
        if row['full_registry_pending']:coverage+='；完整候选范围计算中'
        counts=row['specific_deployed_context_counts'];branches=','.join(f'{f}:{v}' for f,v in counts.items()) or '待完成'
        parts.append('| '+' | '.join([row['backbone'],row['label'],fmt(row['Base'],row['task']),
            fmt(row['Shared'],row['task']),fmt(row['Specific'],row['task']),fmt(row['Specific_R5'],row['task']),branches,coverage])+' |')
    parts+=['','新查询验证平均结果（参数冻结后评估；同一批数据集的新样本）。','',
        '| 骨干 | 条件 | 数据集数 | Base | Shared | Specific R5（诊断） | 完整性 |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in heldout:
        m=row['methods'];count=max([v['datasets'] for v in m.values()],default=0)
        parts.append('| '+' | '.join([row['backbone'],row['label'],f"{count}/{row['expected_datasets']}",fmt(m.get('Base'),row['task']),
            fmt(m.get('TabACE_Shared_benchmark_tuned_R5'),row['task']),fmt(m.get('TabACE_Specific_benchmark_tuned_R5'),row['task']),
            '完整' if row['complete'] else '进行中'])+' |')
    parts+=['','[完整精度与覆盖情况](rebuttal/ensemble_comparison/results/benchmark_tuning_summary.json)。失败变体与全部调参结果均保留。','']
    document=ROOT/'REBUTTAL_EXPERIMENTS.md';old=read_markdown(document).split('<!-- BENCHMARK_TUNING_TABLES -->')[0].rstrip()
    if '上方为早先重建协议的对照' not in old:
        old=old.replace('**本表属于本轮修复／重建协议，尚未复现原主表：TabICL 五扰动 Base Acc 为 0.657，原论文为 0.494。**',
            '**上方为早先重建协议的对照；下方另列本轮 benchmark 调参及新查询验证。不同协议的分数分别解释。**')
    write_markdown(old+'\n'+'\n'.join(parts),document)
    print(json.dumps({'development_complete':sum(r['complete'] for r in entries),'holdout_complete':sum(r['complete'] for r in heldout)}),flush=True)


if __name__=='__main__':main()
