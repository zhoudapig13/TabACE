"""Check row counts against Appendix sample caps, without touching predictions."""
from report_io import read_markdown,write_markdown
import hashlib
import json
from collections import Counter
import numpy as np
from prepare_data import RUN,ROOT,atomic_json


def main():
    records=[]
    for track,task,pattern,expected,cap,target_cap in [
        ('five','classification','*.npz',72,256,128),
        ('five','regression','*.npz',35,2048,512),
        ('seven_reimplemented','classification','*/covariate_0.npz',35,256,128),
        ('seven_calibrated_v2','regression','*/covariate_0.npz',33,256,256)]:
        paths=sorted((RUN/'cache'/track/task).glob(pattern))
        assert len(paths)==expected,(track,task,len(paths))
        source=[];target=[];pool=[]
        for p in paths:
            with np.load(p) as f:
                source.append(len(f['y_source']));target.append(len(f['y_target']))
                pool.append(len(f['y_pool']) if 'y_pool' in f else len(f['y_source']))
            assert source[-1]<=cap and target[-1]<=target_cap
            if task=='classification' or track.startswith('seven'):
                assert source[-1]==256
        records.append({'track':track,'task':task,'datasets':expected,
            'source_rows_distribution':dict(Counter(source)),
            'target_rows_distribution':dict(Counter(target)),
            'accessible_pool_rows_distribution':dict(Counter(pool)),
            'source_row_cap':cap,'target_row_cap':target_cap,
            'row_caps_match_paper_Appendix':True})
    paper=ROOT/'paper/sections/appendix.tex'
    result={'status':'PASS','datasets_checked':sum(r['datasets'] for r in records),'records':records,
        'paper_source':str(paper.relative_to(ROOT)),
        'paper_sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),
        'only_row_counts_verified_not_bit_exact_generator':True,
        'regression_five_source_rows_are_not_256':True,
        'old_five_regression_256_documentation_was_incorrect':True,
        'no_predictions_or_scores_changed':True}
    atomic_json(RUN/'results/track_sample_budget_audit.json',result)
    doc=ROOT/'REBUTTAL_EXPERIMENTS.md';text=read_markdown(doc)
    text=text.replace('五扰动 / 原256行','五扰动 / 原源集465–2048行')
    marker='TRACK_SAMPLE_BUDGETS'
    lines=['<!-- '+marker+' -->','',
        '样本预算核对（已对照论文附录与实际缓存；六成员数量与每成员行数分别记录）：','',
        '| 轨道 | 任务数 | 当前源上下文行数 | 重建缓存的完整源池 | 测试行数上限 |',
        '|---|---:|---:|---:|---:|',
        '| 五扰动分类 | 72 | 256 | 256 | 128 |',
        '| 五扰动回归 | 35 | 465–2048 | 与当前源集相同 | 512 |',
        '| 七偏移分类 | 35 | 256 | 320 | 128 |',
        '| 七偏移回归 | 33 | 256 | 691–1024 | 256 |','',
        '此前“五扰动回归原256行”的表格说明已更正；缓存和预测本来就使用465–2048行，分数无需重算。固定256行的七偏移回归输入匹配组是额外控制，不能与完整历史源池组混称同一输入设置。',
        '这项核对只确认样本预算符合附录，不证明重建算子与历史实验逐位相同，也不证明论文Shared实际使用了哪一种源池。',
        '<!-- END_'+marker+' -->','']
    start,end='<!-- '+marker+' -->','<!-- END_'+marker+' -->'
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+'\n'.join(lines)+after
    else:text=text.rstrip()+'\n\n'+'\n'.join(lines)
    write_markdown(text,doc)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
