"""Report only frozen-protocol observed baselines; Shared is a historical reference."""
from report_io import read_markdown,write_markdown
import csv,json,math
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from prepare_data import ROOT,RUN,atomic_json

COHORTS=[('five_feature_calibrated','classification',72,20),('five','regression',35,20),
    ('seven_reimplemented','classification',35,28),('seven_calibrated_v2','regression',33,28)]
METHODS=[('tabicl','base_protocol_check','Base','1','both'),
 ('tabicl','native_6','Native-6','6','both'),('tabicl','bootstrap_mean_6','Random-6','6','both'),
 ('tabicl','DENSE_fullcontext_OOF_selected_adapted','DENSE-6*','6','classification'),
 ('tabicl','GES_25_metric_normalized','GES-6','≤6','both'),
 ('tabicl','Stacking_convex_OOF','Convex Stacking-6','≤6','both'),
 ('tabicl','Stacking_OOF','Log-linear Stacking-6','6','classification'),
 ('tabicl','TabFM_feature_views6_NNLS_adapted','TabFM-6*','≤6','classification'),
 ('tabicl','Feature_views6_convex_OOF_adapted','Feature views-6（凸融合适配）*','≤6','regression'),
 ('tabicl','TabFM_feature_views6_NNLS_adapted','TabFM-6（原NNLS诊断）*','≤6','regression'),
 ('tabicl','divide_conquer_feature_adapted','Divide-and-conquer*','≤6','both'),
 ('tabicl','oligoicp_iqr_6_keep1_adapted','OligoICP 6→1*','6→1','regression'),
 ('tabicl','OligoICP_IQR60_keep6_adapted','OligoICP 60→6*','60→6','regression'),
 ('tabicl','Bootstrap_mean60_extended_budget','Random-60（筛选成本对照）','60','regression'),
 ('tabpfn','tabpfn_v3_base_check','Base','1','both'),
 ('tabpfn','tabpfn_bootstrap_mean_6','Random-6','6','both'),
 ('tabpfn','PHE_official_custom6_native_source_calibration','PHE-6（官方六配置）','≤6','both'),
 ('tabpfn','RF_PFN_official6_no_adaptive_tree','RF-PFN-6*','6','classification'),
 ('tabpfn','MixturePFN_MICP6_no_CaPFN','MixturePFN MICP-6（无CaPFN）*','6专家','classification')]


def collect():
    allowed={(b,m) for b,m,*_ in METHODS};rows=[]
    for track,task,_,_ in COHORTS:
        for bone in ['tabicl','tabpfn']:
            for root in [RUN/'results'/bone/track/task,RUN/'results/micp6'/bone/track/task,
                RUN/'results/oligo_selected60'/bone/track/task,RUN/'results/oligo_extended'/bone/track/task]:
                for path in root.glob('*/*.json'):
                    if '.tmp' in path.name:continue
                    info=json.loads(path.read_text())
                    if info.get('status')!='COMPLETE':continue
                    for method,values in info.get('methods',{}).items():
                        if (bone,method) in allowed:rows.append(info|values|{'backbone':bone,'method':method,'source':str(path.relative_to(RUN))})
        for path in (RUN/'results/combiners'/track/task).glob('fold_*.json'):
            if '.tmp' in path.name:continue
            info=json.loads(path.read_text())
            for row in info.get('rows',[]):
                bone=row.get('backbone',info.get('backbone','tabicl'));method=row['method']
                if (bone,method) not in allowed:continue
                if method=='Stacking_OOF' and task=='regression':continue
                rows.append(row|{'backbone':bone,'method':method,'source':str(path.relative_to(RUN))})
    unique={}
    for row in rows:
        key=(row['backbone'],row['method'],row['track'],row['task'],row['task_id'],row['family'],row['rho'])
        if key in unique:
            for metric in ['accuracy','auc','r2','nrmse']:
                if metric in row:assert abs(row[metric]-unique[key][metric])<1e-10,key
        unique[key]=row
    return list(unique.values())


def average(rows,keys,expected_datasets,expected_conditions):
    by=defaultdict(lambda:defaultdict(list))
    for row in rows:by[row['data_id']][row['family']].append(row)
    count=sum(len(v) for families in by.values() for v in families.values())
    complete=len(by)==expected_datasets and count==expected_datasets*expected_conditions
    if not by:return {'complete':False,'datasets':0,'conditions':0,'metrics':None}
    values={k:math.fsum(math.fsum(math.fsum(r[k] for r in group)/len(group) for group in families.values())/len(families)
        for families in by.values())/len(by) for k in keys}
    assert all(math.isfinite(v) for v in values.values())
    return {'complete':complete,'datasets':len(by),'conditions':count,'metrics':values}


def main():
    rows=collect();records=[]
    for bone,method,label,budget,applicability in METHODS:
        for task in ['classification','regression']:
            if applicability not in ['both',task]:continue
            keys=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
            selected=[r for r in rows if r['backbone']==bone and r['method']==method and r['task']==task]
            five='five_feature_calibrated' if task=='classification' else 'five'
            seven='seven_reimplemented' if task=='classification' else 'seven_calibrated_v2'
            n5=72 if task=='classification' else 35;n7=35 if task=='classification' else 33
            record={'backbone':bone,'method':method,'label':label,'budget':budget,'task':task}
            for block,track,nonzero,n,c in [('clean',five,False,n5,1),('five',five,True,n5,20),('seven',seven,True,n7,28)]:
                record[block]=average([r for r in selected if r['track']==track and (r['rho']>0)==nonzero],keys,n,c)
            record['complete']=all(record[k]['complete'] for k in ['clean','five','seven'])
            records.append(record)
    summary={'seed':0,'Shared_is_paper_reference_only':True,'same_protocol_Shared_computed':False,
        'all_results_reported':True,'records':records,'complete':all(r['complete'] for r in records)}
    atomic_json(RUN/'results/final_ensemble_summary.json',summary)
    flat=[]
    for r in records:
        entry={k:r[k] for k in ['backbone','method','label','budget','task','complete']}
        for block in ['clean','five','seven']:
            entry[block+'_datasets']=r[block]['datasets'];entry[block+'_conditions']=r[block]['conditions']
            for metric,value in (r[block]['metrics'] or {}).items():entry[block+'_'+metric]=value
        flat.append(entry)
    fields=list(dict.fromkeys(k for row in flat for k in row))
    with (RUN/'results/final_ensemble_averages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    references={('tabicl','classification'):[[.804,.901],[.575,.720],[.772,.831]],
        ('tabicl','regression'):[[.738,.478],[.363,.783],[.467,.659]],
        ('tabpfn','classification'):[[.792,.893],[.553,.697],[.770,.817]],
        ('tabpfn','regression'):[[.739,.480],[.237,.834],[.460,.656]]}
    def fmt(block,task):
        if not block['complete']:return f"进行中（{block['conditions']}）"
        keys=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        return ' / '.join(f"{block['metrics'][k]:.3f}" for k in keys)
    parts=['<!-- FINAL_ENSEMBLE_FROZEN -->','',
        '新协议 ensemble 平均结果（'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')+'；单种子0）。',
        'Shared 仅引用论文，未运行；其分数属于历史协议参考。所有实测结果均保留。主要方法使用六候选／成员上限；选择型方法的非零成员数可能更少。',
        'OligoICP 6→1 使用六次候选预测；60→6 组使用60次候选预测，额外筛选成本不与六次预测组混同。','']
    for task,title in [('classification','分类：Acc↑ / AUC↑'),('regression','回归：R²↑ / nRMSE↓')]:
        parts += [title,'','| 骨干 | 方法 | Clean | 五扰动 | 七偏移 |','|---|---|---:|---:|---:|']
        for bone in ['tabicl','tabpfn']:
            for r in records:
                if r['task']==task and r['backbone']==bone:
                    parts.append('| '+' | '.join([bone,r['label']]+[fmt(r[k],task) for k in ['clean','five','seven']])+' |')
            ref=references[(bone,task)]
            parts.append('| '+' | '.join([bone,'Shared（论文参考）']+[' / '.join(f'{v:.3f}' for v in pair) for pair in ref])+' |')
        parts.append('')
    parts += ['* 为策略适配／组件变体；MixturePFN 未运行 CaPFN 适配，RF-PFN 未使用自适应树。DENSE 与 OligoICP 为 TabICL 策略适配；TabFM 使用六视图而非原完整实现。',
        '[完整精度平均表](rebuttal/ensemble_comparison/results/final_ensemble_averages.csv)。','<!-- END_FINAL_ENSEMBLE_FROZEN -->','']
    section='\n'.join(parts);document=ROOT/'REBUTTAL_EXPERIMENTS.md';old=read_markdown(document);start='<!-- FINAL_ENSEMBLE_FROZEN -->';end='<!-- END_FINAL_ENSEMBLE_FROZEN -->'
    if start in old:
        before,rest=old.split(start,1);_,after=rest.split(end,1);result=before+section+after
    else:
        first,rest=old.split('\n',1);result=first+'\n\n'+section+'\n'+rest
    write_markdown(result,document)
    print(json.dumps({'complete_rows':sum(r['complete'] for r in records),'expected_rows':len(records),'complete':summary['complete']}),flush=True)


if __name__=='__main__':main()
