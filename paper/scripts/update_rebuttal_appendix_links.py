"""Resolve author-response appendix pointers against the compiled LaTeX aux file."""
from pathlib import Path
import json
import re

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
aux = (PAPER / 'main_iclr.aux').read_text(errors='replace')
locations = {k: {'number': n, 'page': int(p)}
             for k, n, p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}', aux)
             if p.isdigit()}


def ref(section, tables=(), zh=False):
    s = locations[section]
    text = ('附录 ' if zh else 'Appendix ') + s['number']
    if tables:
        nums = [locations[t]['number'] for t in tables]
        text += ('，表 ' if zh else ', Table' + ('s ' if len(nums) > 1 else ' '))
        text += '–'.join(nums) if len(nums) < 3 else ', '.join(nums)
    pages = sorted({locations[t]['page'] for t in tables} or {s['page']})
    page = str(pages[0]) if len(pages) == 1 else f'{pages[0]}–{pages[-1]}'
    text += f'，第 {page} 页' if zh else f", {'pp.' if len(pages)>1 else 'p.'} {page}"
    return text


def pointer(which, zh):
    spec = {
        'ensemble': ('app:revision-ensembles', ('tab:revision-tabicl-cls','tab:revision-tabpfn-reg')),
        'large': ('app:revision-large-context', ('tab:revision-large-cls','tab:revision-large-reg')),
        'matching': ('app:revision-matching', ('tab:revision-matching',)),
        'unseen': ('app:revision-unseen', ('tab:revision-unseen-cls','tab:revision-unseen-reg')),
        'folk': ('app:revision-folktables-comparison', ('tab:revision-folktables',)),
        'contexts': ('app:revision-folktables-contexts', ()),
        'variance': ('app:revision-folktables-variance', ('tab:revision-folktables-variance',)),
        'medical': ('app:revision-medical', ('tab:revision-medical',)),
        'ft': ('app:revision-finetuning', ('tab:revision-finetuning',)),
        'hc': ('app:revision-high-confidence', ('tab:revision-high-confidence',)),
        'reg': ('app:revision-regression-robustness', ('tab:revision-joint-errors','tab:revision-regression-extension')),
    }
    return ref(*spec[which], zh=zh)


def refresh_pages(doc, zh):
    """Keep existing explicit citations correct when preceding layout changes."""
    by_section = {v['number']: v for k,v in locations.items() if k.startswith('app:')}
    by_table = {v['number']: v for k,v in locations.items() if k.startswith('tab:')}
    pattern = (r'附录 (E(?:\.\d+)*)(?:，表 (\d+(?:–\d+)?))?，第 (\d+(?:–\d+)?) 页'
               if zh else r'Appendix (E(?:\.\d+)*)(?:, Tables? (\d+(?:–\d+)?))?, pp?\. (\d+(?:–\d+)?)')
    def replace(m):
        sec, table_range, _ = m.groups()
        if sec not in by_section:
            return m.group()
        if table_range:
            nums=table_range.split('–')
            pages=[by_table[n]['page'] for n in nums if n in by_table]
        else:
            pages=[by_section[sec]['page']]
        if not pages:
            return m.group()
        lo,hi=min(pages),max(pages)
        pg=str(lo) if lo==hi else f'{lo}–{hi}'
        if zh:
            return f'附录 {sec}'+(f'，表 {table_range}' if table_range else '')+f'，第 {pg} 页'
        return f'Appendix {sec}'+(f", Table{'s' if '–' in table_range else ''} {table_range}" if table_range else '')+f", {'p.' if lo==hi else 'pp.'} {pg}"
    return re.sub(pattern,replace,doc)


def add_after_first_paragraph(text, sentence):
    # A response paragraph follows all opening blockquotes; skip headings/quotes.
    for para in text.split('\n\n'):
        if para.startswith(('Thank you', '感谢您')):
            return text.replace(para, para + ' ' + sentence, 1)
    raise ValueError('No response introduction found')


def process(name, zh):
    path = ROOT / name
    doc = refresh_pages(path.read_text(), zh)
    marker = '# 对审稿人 ' if zh else '# Response to Reviewer '
    review_matches = list(re.finditer('^' + re.escape(marker) + r'(\w+)', doc, re.M))
    pieces = []
    for i, m in enumerate(review_matches):
        text = doc[m.start():review_matches[i+1].start() if i+1<len(review_matches) else len(doc)]
        reviewer = m.group(1)
        def p(which): return pointer(which, zh)
        if reviewer == 'SbVz':
            if zh:
                replacements = {
                    '**校准与构造器匹配。**': '**校准与构造器匹配（'+p('matching')+'）。**',
                    '**未见机制。**': '**未见机制（'+p('unseen')+'）。**',
                    'DistPFN 使用 TabICL v2 作为骨干。': 'DistPFN 使用 TabICL v2 作为骨干（'+p('folk')+'）。',
                    '分别得到 FT+Shared 和 FT+Specific。': '分别得到 FT+Shared 和 FT+Specific（'+p('ft')+'）。',
                    '**我们将在更新后的 PDF 附录中补充完整基线表格与实验配置**': '**完整基线表格与预算说明已列于'+p('ensemble')+'**',
                    '以急诊入院来源作为 OOD。': '以急诊入院来源作为 OOD。我们报告 1:1 病例对照评估的准确率（'+p('medical')+'）。',
                }
            else:
                replacements = {
                    '**Calibration and constructor matching.**': '**Calibration and constructor matching ('+p('matching')+').**',
                    '**Unseen mechanisms.**': '**Unseen mechanisms ('+p('unseen')+').**',
                    'DistPFN uses TabICL v2 as its backbone.': 'DistPFN uses TabICL v2 as its backbone ('+p('folk')+').',
                    'yielding FT+Shared and FT+Specific.': 'yielding FT+Shared and FT+Specific ('+p('ft')+').',
                    '**We will include the complete baseline tables and experimental configurations in the updated PDF appendix**': '**The complete baseline tables and budget conventions are provided in '+p('ensemble')+'**',
                    'while emergency-room admissions form the OOD domain.': 'while emergency-room admissions form the OOD domain. We report Accuracy on 1:1 case–control evaluations ('+p('medical')+').',
                }
        elif reviewer == 'BV6f':
            if zh:
                replacements = {
                    '完整结果、适配配置及选择开销将在更新后的 PDF 附录中补充。': '完整结果、适配标记与候选／部署预算说明见'+p('ensemble')+'。',
                    '我们将在修订附录中明确这些定义。': '我们已在附录 D.3 明确这些定义。',
                    '**超出校准与构造器匹配的迁移。**': '**超出校准与构造器匹配的迁移（'+p('matching')+'）。**',
                    '完整结果将在修订 PDF 附录中补充。': '完整结果见'+p('unseen')+'。',
                    '完整结果将在修订版 PDF 附录中展示。': '完整结果见'+p('unseen')+'。',
                    '下表报告 OOD Acc。': '下表报告 1:1 病例对照评估的 OOD Acc（'+p('medical')+'）。',
                    '**高置信 Base 错误影响的样本范围较小：**': '**高置信 Base 错误影响的样本范围较小：**（'+p('hc')+'）',
                    '**共同错误与有害更新。': '**共同错误与有害更新（'+p('reg')+'）。',
                }
                old='受正文篇幅限制，我们将在修订版附录中按任务、家族和外层折补充选中的上下文、构造参数及实际分支数量，并在总前向计算次数中计入 Base。'
                text=text.replace(old,'Folktables 中实际部署的上下文列表、构造参数与分支数量已列于'+p('contexts')+'。主基准的候选选择流程仍见附录 B.4.6。')
            else:
                replacements = {
                    'Complete results, adaptation settings, and selection costs will be included in the updated PDF appendix.': 'Complete results, adaptation markers, and candidate/deployment budget conventions are provided in '+p('ensemble')+'.',
                    'We will make these definitions explicit in the revised appendix.': 'We have made these definitions explicit in Appendix D.3.',
                    '**Transfer beyond calibration and constructor matching.**': '**Transfer beyond calibration and constructor matching ('+p('matching')+').**',
                    'The complete results will be included in the revised PDF appendix.': 'The complete results are provided in '+p('unseen')+'.',
                    'The table reports OOD Acc.': 'The table reports OOD Acc for the 1:1 case–control evaluation ('+p('medical')+').',
                    '**High-confidence Base errors affect a small fraction of the evaluated queries:**': '**High-confidence Base errors affect a small fraction of the evaluated queries:** ('+p('hc')+')',
                    '**Correlated errors and harmful updates.': '**Correlated errors and harmful updates ('+p('reg')+').',
                }
                old='Given the space constraints in the main text, we will add a deployment table to the revised appendix listing the selected contexts, construction parameters, and actual branch counts for each task, family, and outer fold, including Base in the total forward-pass count.'
                text=text.replace(old,'The deployed Folktables context list, construction parameters, and branch counts are provided in '+p('contexts')+'. Candidate selection for the main benchmark is described in Appendix B.4.6.')
        else:
            if zh:
                replacements = {
                    '**分类（Acc / AUC）。**': '**分类（Acc / AUC；'+p('large')+'）。**',
                    '**完整基线表格与配置将在更新后的 PDF 附录中补充**': '**完整基线表格与预算说明已列于'+p('ensemble')+'**',
                    '**区分校准覆盖与构造器匹配。**': '**区分校准覆盖与构造器匹配（'+p('matching')+'）。**',
                    '**Shared 构造库之外的机制。**': '**Shared 构造库之外的机制（'+p('unseen')+'）。**',
                    '完整医疗结果将在更新后的 PDF 附录中补充。': '完整的 1:1 病例对照医疗结果见'+p('medical')+'。',
                    '**上下文构造。**': '**上下文构造（'+p('contexts')+'）。**',
                    '**与偏移方法对照。**': '**与偏移方法对照（'+p('folk')+'）。**',
                    '**随机种子与方差。**': '**随机种子与方差（'+p('variance')+'）。**',
                    '**共同错误与有害更新。': '**共同错误与有害更新（'+p('reg')+'）。',
                    '**高置信 Base 错误影响的样本范围较小：**': '**高置信 Base 错误影响的样本范围较小：**（'+p('hc')+'）',
                }
            else:
                replacements = {
                    '**Classification (Acc / AUC).**': '**Classification (Acc / AUC; '+p('large')+').**',
                    '**The complete baseline tables and configurations will be included in the updated PDF appendix**': '**The complete baseline tables and budget conventions are provided in '+p('ensemble')+'**',
                    '**Separating calibration coverage from constructor matching.**': '**Separating calibration coverage from constructor matching ('+p('matching')+').**',
                    '**Mechanisms outside the Shared bank.**': '**Mechanisms outside the Shared bank ('+p('unseen')+').**',
                    'Complete clinical results will be included in the updated PDF appendix.': 'Complete clinical results for the 1:1 case–control evaluation are provided in '+p('medical')+'.',
                    '**Context construction.**': '**Context construction ('+p('contexts')+').**',
                    '**Comparison with shift-specific methods.**': '**Comparison with shift-specific methods ('+p('folk')+').**',
                    '**Seeds and variance.**': '**Seeds and variance ('+p('variance')+').**',
                    '**Correlated errors and harmful updates.': '**Correlated errors and harmful updates ('+p('reg')+').',
                    '**High-confidence Base errors affect a small fraction of the evaluated queries:**': '**High-confidence Base errors affect a small fraction of the evaluated queries:** ('+p('hc')+')',
                }
        for old,new in replacements.items():
            if new in text:
                continue
            if old in text:
                text=text.replace(old,new)
            else:
                print('Unmatched optional phrase:', reviewer, repr(old[:65]))
        # Normalize duplicate pointers if an earlier updater run inserted a
        # reference after an otherwise unchanged heading or sentence.
        text=re.sub(r'(\(Appendix [^()]+\))\s+\1',r'\1',text)
        text=re.sub(r'(（附录 [^（）]+）)\s*\1',r'\1',text)
        for sentence in [
            'We report Accuracy on 1:1 case–control evaluations ('+p('medical')+').',
            '我们报告 1:1 病例对照评估的准确率（'+p('medical')+'）。',
        ]:
            text=text.replace(sentence+' '+sentence,sentence).replace(sentence+sentence,sentence)
        pieces.append(text)
    path.write_text(''.join(pieces))


process('REBUTTAL_EXPERIMENTS.md',False)
process('REBUTTAL_EXPERIMENTS_ZH.md',True)
revision={k:v for k,v in locations.items() if k.startswith(('app:revision','app:rebuttal','tab:revision'))}
(PAPER/'revision_data/appendix_locations.json').write_text(json.dumps(revision,indent=2)+'\n')
index=['# Rebuttal appendix locations / 回复附录定位','',
       '| LaTeX label | Section/table | PDF page |','|---|---|---:|']
for k,v in revision.items():
    index.append(f"| `{k}` | {v['number']} | {v['page']} |")
index += ['', 'Original protocol and ablation references remain Appendix B.1.1 (Table 4), B.1.2, B.4 (Tables 8–10), B.4.6, B.5 (Table 11), and D.1–D.4.',
          '', 'The detailed per-family, per-outer-fold deployed lists for the original main-benchmark Specific runs are not reconstructed by the supplied Folktables registry. Appendix E.5.1 records the actual supplemental Folktables configuration only.', '']
(PAPER/'REVISION_LOCATION_INDEX.md').write_text('\n'.join(index))
print('Updated both responses and wrote the compiled appendix location index.')
