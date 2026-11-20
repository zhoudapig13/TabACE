"""Report every reference-budget condition with paired measured FT controls."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'results'
OUT = ROOT / 'reference_adaptation_summary_v29'


def main():
    OUT.mkdir(exist_ok=True)
    roots = [ROOT / 'few_reference_contexts_v26'] + [
        ROOT / f'few_reference_contexts_v27_ref{k}' for k in [16, 32, 128]]
    frames = []
    for r in roots:
        for p in sorted(r.glob('averages_*.csv')):
            if 'rawscale' in p.name:
                continue
            df = pd.read_csv(p)
            df['reference_examples'] = json.loads((r / 'plan.json').read_text())['source_reference_pool_cap']
            df['adaptation'] = 'pure_reference_contexts'
            frames.append(df)
    for name, adaptation in [('reference_repair_contexts_v28', 'reference_guided_source_label_repair'),
                             ('reference_repair_mixed_contexts_v30', 'reference_guided_repair_plus005')]:
        for p in sorted((ROOT / name).glob('averages_*.csv')):
            df = pd.read_csv(p)
            df['reference_examples'] = 64
            df['adaptation'] = adaptation + ('_canonical128' if 'canonical128' in p.name else '')
            frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(OUT / 'all_averages.csv', index=False)
    ft = pd.read_csv(ROOT / 'few_reference_contexts_v26/verified_frozen_FT_averages.csv')
    lines = [
        '# 固定环境下的参考适配与 FT 对照', '',
        '所有条件保留冻结的 benchmark、Base、查询行和强度网格 {0.2,0.5,0.8,1.0}，seed=0。'
        '分类 Clean/五扰动为72个数据集、七偏移35个；回归分别为35、33个。'
        '既有 FT 的4151个条件已逐一核对输入哈希和查询标签，指标由缓存的实测预测重新计算。', '',
        '这是开发性配置重建与适配分析。回归 Clean/五扰动沿用冻结的半合成任务。'
        '原主表完整 Specific 部署列表仍未恢复；本轮使用明确记录的五类 Shared 与五类家族定向构造。'
        '所有资源档均报告，不按查询分数挑选某一档作为“原论文复现”。', '',
        'K 表示实际输入参考池的干净 TRAIN 样本数。分类按训练标签分层抽取，类别数超过K时保证类别覆盖；'
        '回归使用固定 TRAIN 顺序的前K条。这是额外干净参考监督；K不是标签盲选样的标注成本声明。'
        '参考行与校准、最终查询行分离。融合参数使用五折数据集级OOF校准，保留原门控形式。', '',
    ]
    for task, title, a, b in [('classification', '分类：Acc / AUC', 'accuracy', 'auc'),
                               ('regression', '回归：R² / nRMSE', 'r2', 'nrmse')]:
        lines += ['## ' + title, '', '| 配置 | 方法 | Clean | 五扰动平均 | 七偏移平均 |', '|---|---|---|---|---|']
        base = all_rows[(all_rows.task == task) & (all_rows.method == 'Base')].drop_duplicates('block')
        items = [('冻结输入', 'Base', base), ('既有同输入实测', 'FT', ft[ft.task == task])]
        for budget in [16, 32, 64, 128]:
            for method in ['Shared', 'Specific']:
                subset = all_rows[(all_rows.task == task) & (all_rows.reference_examples == budget) &
                                  (all_rows.method == method) & (all_rows.adaptation == 'pure_reference_contexts')]
                items.append((f'干净参考池 K={budget}', method, subset))
        for adaptation, label, methods in [
            ('reference_guided_source_label_repair', '64参考辅助修复原256行', ['Shared', 'Specific']),
            ('reference_guided_repair_plus005', '修复＋最多5%参考行', ['Shared', 'Specific']),
            ('reference_guided_repair_plus005_canonical128', '修复＋5%＋单个128行参考专家', ['Specific']),
        ]:
            for method in methods:
                subset = all_rows[(all_rows.task == task) & (all_rows.method == method) &
                                  (all_rows.adaptation == adaptation)]
                items.append((label, method, subset))
        for config, method, subset in items:
            values = []
            for block in ['clean', 'five', 'seven']:
                cell = subset[subset.block == block]
                values.append('运行中' if cell.empty else f'{cell.iloc[0][a]:.4f} / {cell.iloc[0][b]:.4f}')
            lines.append('| ' + ' | '.join([config, method] + values) + ' |')
        lines += ['']
    lines += [
        '纯参考池版的每个上下文最多K行（上限256）；Source-label repair版保留原始源特征与256行容量，仅修复明显不一致标签。'
        'Shared/Specific分别使用5/10个替代上下文，加Base为6/11次常规查询预测。'
        '修复版另需一次参考TFM对源上下文的检查；同监督 Native-1 是独立对照。', '',
        '单个128行参考专家对照在Specific的10个替代上下文上额外增加一个上下文，因此使用128条参考样本、'
        '11个替代上下文，加Base为12次常规查询预测；它与标准10替代上下文版本分开报告。', '',
        '原有 Table 28、rebuttal 文稿和 PDF 未被覆盖。'
    ]
    (OUT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n')
    print(OUT / 'RESULTS_ZH.md')


if __name__ == '__main__':
    main()
