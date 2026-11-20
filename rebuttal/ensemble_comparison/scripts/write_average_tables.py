"""Write only compact macro-average tables to the user-facing Markdown file."""
from __future__ import annotations
from report_io import read_markdown,write_markdown
import csv
from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[1]
LABELS={
    'base_protocol_check':'Base','tabpfn_v3_base_check':'Base',
    'native_6':'Native-6','native_8':'Native-8','tabpfn_native_6':'Native-6','tabpfn_native_8':'Native-8',
    'bootstrap_mean_6':'Random-6','tabpfn_bootstrap_mean_6':'Random-6',
    'DENSE_fullcontext_OOF_selected_adapted':'DENSE-6*',
    'oligoicp_iqr_6_keep1_adapted':'OligoICP 6→1*',
    'OligoICP_IQR40_keep4_adapted':'OligoICP 40→4*','OligoICP_IQR400_keep40_adapted':'OligoICP 400→40*',
    'Bootstrap_mean40_extended_budget':'Random-40','Bootstrap_mean400_extended_budget':'Random-400',
    'GES_25_metric_normalized':'GES-6','Stacking_convex_OOF':'Convex Stacking-6',
    'Stacking_OOF':'Log-linear Stacking-6','TabFM_feature_views6_NNLS_adapted':'TabFM-6*',
    'divide_conquer_feature_adapted':'Divide-and-conquer*',
    'Shared_context_mean_reimplemented':'Shared-context Mean†',
    'Specific_context_mean_reimplemented':'Specific-context Mean†',
    'TabACE_Shared_R5_reimplemented':'**TabACE-Shared†**',
    'TabACE_Specific_R5_reimplemented':'**TabACE-Specific†**',
    'PHE_official_custom6_native_source_calibration':'PHE（六配置）',
    'RF_PFN_official6_no_adaptive_tree':'RF-PFN（无自适应树）',
    'BETA_bootstrap_component_only_no_adaptation':'BETA（仅 bootstrap）',
    'MixturePFN_MICP_no_CaPFN':'MixturePFN（无 CaPFN）','TabM_official_source_refit':'TabM'}
BLOCKS=[
    ('分类：每格为 Acc↑ / AUC↑。',[
        ('classification_clean_polarity_corrected_accuracy','classification_clean_polarity_corrected_auc'),
        ('classification_perturbations_polarity_corrected_accuracy','classification_perturbations_polarity_corrected_auc'),
        ('classification_shifts_reimplemented_accuracy','classification_shifts_reimplemented_auc')]),
    ('回归：每格为 R²↑ / nRMSE↓。',[
        ('regression_clean_r2','regression_clean_nrmse'),
        ('regression_perturbations_available_variant_r2','regression_perturbations_available_variant_nrmse'),
        ('regression_shifts_reimplemented_r2','regression_shifts_reimplemented_nrmse')])]


def render(rows):
    parts=['# Ensemble 对照平均结果','',
        '更新时间：'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')+'（北京时间）。',
        '单种子 0；数据集内平均各家族及 0.2、0.5、0.8、1.0 四个非零强度，再对数据集取宏平均；保留三位小数。',
        '**本表属于本轮修复／重建协议，尚未复现原主表：TabICL 五扰动 Base Acc 为 0.657，原论文为 0.494。**','']
    for title,columns in BLOCKS:
        parts += [title,'','| 骨干 | 方法 | Clean | 五扰动平均 | 七偏移平均 |',
                  '|---|---|---:|---:|---:|']
        for row in rows:
            if not any(row[k] for pair in columns for k in pair):continue
            values=[' / '.join(f'{float(row[k]):.3f}' if row[k] else '—' for k in pair) for pair in columns]
            backbone={'tabicl':'TabICL','tabpfn':'TabPFN','tabm':'独立模型'}[row['backbone']]
            parts.append('| '+' | '.join([backbone,LABELS.get(row['method'],row['label'])]+values)+' |')
        parts.append('')
    parts += [r'\* 策略适配版。† 本轮 context／TabACE 重实现；加粗仅便于定位。OligoICP 箭头表示候选数→保留数，40/400 候选与六成员预算分别解释。',
              '旧 ridge 数值失稳行未列为有效对照。TabM 七偏移 R² 保留全部条件，包括少数严重外推失败。','',
              '[完整精度平均表](rebuttal/ensemble_comparison/results/main_table_current_protocol.csv) · '
              '[逐数据集明细 CSV](rebuttal/ensemble_comparison/results/all_detail.csv)','']
    return '\n'.join(parts)


def write_report(source=None,destination=None):
    source=Path(source) if source else RUN/'results/main_table_current_protocol.csv'
    destination=Path(destination) if destination else ROOT/'REBUTTAL_EXPERIMENTS.md'
    with source.open(newline='') as stream:rows=list(csv.DictReader(stream))
    result=render(rows)
    written=write_markdown(result,destination)
    return {'methods':len(rows),'table_rows':result.count('\n| ')-2,'bytes':len(result.encode()),'markdown_written':written}


if __name__=='__main__':print(write_report(),flush=True)
