"""Audit and report the full finite Clean-stability experiment, without censoring."""
from report_io import read_markdown,write_markdown
import json
from fit_stable_regression_references import load,predict
from prepare_data import RUN,ROOT,atomic_json
from run_tabicl import metrics


def main():
    summary = json.loads((RUN/'results/regression_Clean_stable_summary.json').read_text())
    count_policy = count_metrics = 0
    for track,regime in [('five','canonical'),('seven_calibrated_v2','canonical'),('seven_calibrated_v2','full_pool')]:
        cells = load(track,'full_pool' if track=='five' else regime)
        index = {(c['meta']['task_id'],c['meta']['family'],c['meta']['rho']):c for c in cells}
        for path in (RUN/'results/regression_Clean_stable'/regime/track).glob('fold_*.json'):
            p = json.loads(path.read_text())
            assert set(p['train_data_ids']).isdisjoint(p['test_data_ids'])
            assert p['inner_validation_fold'] != p['fold']
            assert not p['outer_test_labels_used_for_fit_or_selection']
            assert not p['paper_Shared_used_for_selection'] and not p['Shared_run']
            assert any(t['clean_feasible'] and t['regularization']==p['selected_regularization'] for t in p['trials'])
            rows = p['rows']
            for idx in sorted({0,len(rows)//2,len(rows)-1}):
                row = rows[idx]
                c = index[(row['task_id'],row['family'],row['rho'])]
                assert c['meta']['data_id'] in p['test_data_ids']
                scale = max(float(c['y'].std()),1e-12) if track.startswith('seven') else c['meta']['scale']
                observed = metrics(c['y'],predict(c,p['method'],p['model']),'regression',scale)
                for k in observed:
                    assert abs(observed[k]-row[k]) < 1e-10
                count_metrics += 1
            count_policy += 1
    assert count_policy == 75
    atomic_json(RUN/'results/regression_Clean_stable_audit.json',{
        'status':'PASS','OOF_policies_checked':count_policy,'saved_metrics_recomputed':count_metrics,
        'all_primary_Clean_within_005':all(r['outer_Clean_within_005'] for r in summary['averages']),
        'Shared_run':False,'ranking_bounds_used_for_fit':False,
        'unconstrained_and_all_access_regimes_retained':True})
    lines = ['<!-- REGRESSION_CLEAN_STABLE -->','',
        '回归的Clean稳定性控制（种子0；R²↑ / nRMSE↓）：校准集Clean的R²下降与nRMSE增加各≤0.005，再按校准预测损失选配置。',
        '该控制允许校准后回退Base，回退时实际只用一次预测；所以它属于模型选择控制，不能当作固定六成员的纯集成结果。原始六成员结果继续完整保留。','',
        '| 协议 / 输入 | 方法 | 本协议Clean | 扰动或偏移平均 | Base回退比例 |',
        '|---|---|---:|---:|---:|']
    for r in summary['averages']:
        group = '五扰动 / 原源集465–2048行' if r['track']=='five' else ('七偏移 / 原256行' if r['access_regime']=='canonical' else '七偏移 / 完整历史池')
        clean,shift = r['clean'],r['shift']
        lines.append(f"| {group} | {r['method']} + 校准回退 | {clean['r2']:.3f} / {clean['nrmse']:.3f} | {shift['r2']:.3f} / {shift['nrmse']:.3f} | {100*r['backoff_condition_fraction']:.1f}% |")
    lines += ['', '五扰动与七偏移分别使用35/33个回归数据集，Clean列也对应各自数据集，不混合平均。',
              '完整历史池的Random、GES、Convex仍高于论文Shared的R²、低于其nRMSE；如实保留，不能为满足预设排序压低分数。Shared仅作历史参考且本轮未运行。',
              '<!-- END_REGRESSION_CLEAN_STABLE -->','']
    doc = ROOT/'REBUTTAL_EXPERIMENTS.md'
    text = read_markdown(doc)
    start,end = '<!-- REGRESSION_CLEAN_STABLE -->','<!-- END_REGRESSION_CLEAN_STABLE -->'
    section = '\n'.join(lines)
    if start in text:
        before,rest = text.split(start,1)
        _,after = rest.split(end,1)
        text = before+section+after
    else:
        title,rest = text.split('\n',1)
        text = title+'\n\n'+section+'\n'+rest
    write_markdown(text,doc)
    print('PASS',count_policy,count_metrics,'Clean within limit',all(r['outer_Clean_within_005'] for r in summary['averages']),flush=True)


if __name__=='__main__':main()
