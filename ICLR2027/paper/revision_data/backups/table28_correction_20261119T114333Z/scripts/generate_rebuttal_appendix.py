"""Build revision tables from author-supplied tables and bilingual response data.

This does not rerun, recalibrate, or change any experiment. The JSON snapshot
records the source of every numeric table; revised tables retain protocol scope.
"""
from pathlib import Path
import json
import re

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
# Keep appendix evidence independent of the author-response posting format.
# This frozen source reproduces the existing 16 tables and 119 numeric rows.
MD = (PAPER / 'revision_data/appendix_evidence_source.md').read_text()
OUT = PAPER / 'tables/revision'
OUT.mkdir(parents=True, exist_ok=True)


def section(start, end, after=''):
    a = MD.index(start, MD.index(after) if after else 0)
    return MD[a:MD.index(end, a + len(start))]


def md_tables(text):
    groups = re.findall(r'(?:^\|.*\n)+', text, re.M)
    result = []
    for group in groups:
        lines = group.strip().splitlines()
        if len(lines) < 3:
            continue
        result.append([[c.strip() for c in line.strip('|').split('|')]
                       for line in [lines[0], *lines[2:]]])
    return result


def escape(text):
    out = text
    for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
        out = out.replace(a, b)
    for a, b in [('−', '-'), ('–', '--'), ('—', '---'),
                 ('→', r'\(\rightarrow\)'), ('±', r'\(\pm\)'),
                 ('†', r'\textsuperscript{\(\dagger\)}')]:
        out = out.replace(a, b)
    return out


def inline(text):
    parts = re.split(r'(\$[^$]*\$)', text)
    result = ''
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            result += part
        else:
            bold_parts = part.split('**')
            result += ''.join((r'\textbf{' + escape(v) + '}') if i % 2 else escape(v)
                              for i, v in enumerate(bold_parts))
    return result


DATA = {}


def table(key, label, caption, headers, rows, source, note=''):
    assert rows and all(len(r) == len(headers) for r in rows), key
    DATA[key] = dict(label=label, headers=headers, rows=rows, source=source,
                     caption=caption, note=note)
    spec = '@{}' + 'l' + 'r' * (len(headers)-1) + '@{}'
    lines = [r'\begin{table}[H]', r'\centering\footnotesize',
             r'\color{revisionblue}', r'\setlength{\tabcolsep}{4pt}',
             r'\renewcommand{\arraystretch}{1.12}',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{' + spec + '}', r'\toprule',
             ' & '.join(inline(h) for h in headers) + r' \\', r'\midrule']
    for row in rows:
        lines.append(' & '.join(inline(str(v)) for v in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}']
    if note:
        lines += [r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize', note,
                  r'\end{minipage}']
    lines += [r'\end{table}', '']
    (OUT / (key + '.tex')).write_text('\n'.join(lines))
    return r'\input{tables/revision/' + key + '}\n'


cls_icl = [
['TabICL-v2 Base','0.804 / 0.901','0.494 / 0.676','0.763 / 0.826'],
['Native-6','0.803 / 0.901','0.507 / 0.683','0.763 / 0.825'],
['Random-6','0.800 / 0.897','0.523 / 0.696','0.764 / 0.828'],
['DENSE-6','0.798 / 0.897','0.531 / 0.695','0.764 / 0.826'],
['GES-6','0.801 / 0.899','0.539 / 0.694','0.765 / 0.827'],
['Convex Stacking-6','0.800 / 0.898','0.559 / 0.704','0.766 / 0.827'],
['Log-linear Stacking-6','0.803 / 0.901','0.535 / 0.692','0.763 / 0.824'],
['TabFM feature-view NNLS-6','0.805 / 0.901','0.506 / 0.672','0.767 / 0.827'],
['Divide-and-conquer','0.805 / 0.902','0.503 / 0.672','0.763 / 0.825'],
['Shared','0.804 / 0.901','**0.575 / 0.720**','**0.772 / 0.831**']]
reg_icl = [
['TabICL-v2 Base','0.739 / 0.476','−1.077 / 1.270','0.273 / 0.730'],
['Native-6','0.740 / 0.487','−0.597 / 1.145','0.287 / 0.721'],
['Random-6','0.742 / 0.505','−0.397 / 1.089','0.351 / 0.702'],
['GES-6','0.731 / 0.501','−0.434 / 1.103','0.397 / 0.684'],
['Convex Stacking-6','0.733 / 0.499','−0.446 / 1.104','0.401 / 0.682'],
['Ridge Stacking-6','0.732 / 0.500','−0.457 / 1.111','0.416 / 0.683'],
['Median-6†','0.734 / 0.496','−0.505 / 1.121','0.421 / 0.675'],
['Feature Mean-6','0.734 / 0.496','−0.461 / 1.109','0.421 / 0.677'],
['Trimmed Mean-6','0.735 / 0.492','−0.483 / 1.114','0.419 / 0.676'],
['IQR-selection','0.728 / 0.501','−0.446 / 1.102','0.353 / 0.702'],
['Shared','0.738 / 0.478','**0.363 / 0.783**','**0.467 / 0.659**']]
cls_pfn = [
['TabPFN Base','0.793 / 0.893','0.457 / 0.636','0.758 / 0.809'],
['Native-6','0.799 / 0.899','0.467 / 0.637','0.766 / 0.814'],
['PHE-6','0.802 / 0.900','0.464 / 0.626','0.763 / 0.816'],
['RF-PFN-6','0.806 / 0.901','0.470 / 0.629','0.760 / 0.817'],
['MICP component*','0.783 / 0.892','0.461 / 0.626','0.751 / 0.816'],
['Shared','0.792 / 0.893','**0.553 / 0.697**','**0.770** / 0.817']]
reg_pfn = [
['TabPFN Base','0.739 / 0.476','−1.253 / 1.313','0.426 / 0.669'],
['Native-6†','0.737 / 0.508','−0.405 / 1.116','0.444 / 0.640'],
['PHE-6†','0.728 / 0.516','−0.387 / 1.111','0.424 / 0.659'],
['RF-PFN-6†','0.737 / 0.509','−0.400 / 1.116','0.444 / 0.660'],
['OligoICP 60→6*†','0.703 / 0.549','−0.370 / 1.120','0.333 / 0.703'],
['OligoICP 400→40*†','0.697 / 0.549','−0.325 / 1.104','0.355 / 0.695'],
['Shared','0.739 / 0.480','**0.237 / 0.834**','**0.460** / 0.656']]

blocks = [r'''% Generated from revision_data/rebuttal_tables.json and author-provided response tables.
\clearpage
\begingroup
\providecolor{revisionblue}{RGB}{25,55,135}
\color{revisionblue}
\section{Additional Experiments and Clarifications during Discussion}
\label{app:rebuttal-additions}

This section collects the additional evaluations discussed in the author
responses. Newly added material is shown in blue to help locate revisions.
These supplementary evaluations complement the original results; configuration
adaptations, development-task changes, and repeated-run statistics are identified
where applicable rather than replacing the original benchmark definitions.

\subsection{Ensemble Comparisons}
\label{app:revision-ensembles}

Tables~\ref{tab:revision-tabicl-cls}--\ref{tab:revision-tabpfn-reg} give
the supplementary ensemble comparisons. Classification cells report
Accuracy/AUROC and regression cells report $R^2$/nRMSE. Each robustness block
averages its nonzero strengths over the five perturbation or seven shift families.
Native-6 uses the backbone's native ensemble; Random-6 averages six sampled
contexts. GES selects members greedily; convex and log-linear stacking learn
fixed fusion rules. Median, trimmed-mean, and IQR variants provide robust
aggregation or uncertainty-based selection controls. DENSE follows demonstration
ensembling~\citep{khalifa2023dense}; PHE and RF-PFN follow the TabPFN
ensemble components~\citep{hollmann2025tabpfn}. OligoICP uses low-IQR member
selection and averaging~\citep{revision_li2025oligo}.

The suffix ``6'' denotes six deployed predictions; Shared uses Base and five
alternatives. OligoICP $60\!\rightarrow\!6$ and $400\!\rightarrow\!40$
denote candidate/deployed counts. The latter has a larger deployment budget and
is reported separately from six-member comparisons. Candidate generation and
selection are additional preparation costs, not counted as deployed predictions.
An asterisk marks a component-level or budget-adapted implementation; a dagger
marks an adapted supplementary configuration, not necessarily backbone fine-tuning.
''']

heads = ['Method','Clean','Five perturbations','Seven shifts']
for key, label, title, data in [
('ensemble_tabicl_cls','tab:revision-tabicl-cls','TabICL-v2 classification ensemble comparison (Accuracy/AUROC).',cls_icl),
('ensemble_tabicl_reg','tab:revision-tabicl-reg','TabICL-v2 regression ensemble comparison ($R^2$/nRMSE).',reg_icl),
('ensemble_tabpfn_cls','tab:revision-tabpfn-cls','TabPFN classification ensemble comparison (Accuracy/AUROC).',cls_pfn),
('ensemble_tabpfn_reg','tab:revision-tabpfn-reg','TabPFN regression ensemble comparison ($R^2$/nRMSE).',reg_pfn)]:
    blocks.append(table(key,label,title,heads,data,'Author-provided ensemble tables in conversation, 2026-11-12'))

blocks.append(r'''
\subsection{Larger Context Budgets}
\label{app:revision-large-context}
We evaluate TabICL-v2 with row caps of 1,024, 4,096, and 8,192 for Base,
Shared, and Specific. The sample counts are 72 classification/35 regression
datasets for clean and perturbed evaluation, and 35/33 for shifts.
These are supplementary calibrated development-benchmark evaluations; the
regression clean and perturbed tasks include semisynthetic reconstruction.
For the regression recovery protocol, the original 256-row perturbed core is
retained and added context rows come from clean training data. Thus these
recovery results include the contribution of extra clean evidence as the budget
increases, rather than keeping the contamination fraction fixed.
Tables~\ref{tab:revision-large-cls} and~\ref{tab:revision-large-reg}
report the additional results. The compact 256-row budget is also examined in
TabDPT's inference ablation~\citep{ma2025tabdpt}; it is not its capacity limit.
''')
larger = md_tables(section('## Part 1: Context size','## Part 2:', '# Response to Reviewer bPdS'))
for key,label,title,t in [
('large_context_cls','tab:revision-large-cls','Larger-context classification (Accuracy/AUROC); columns cover 72/72/35 datasets.',larger[0]),
('large_context_reg','tab:revision-large-reg','Larger-context regression ($R^2$/nRMSE); columns cover 35/35/33 datasets.',larger[1])]:
    blocks.append(table(key,label,title,['Context cap','Method','Clean','Five perturbations','Seven shifts'],t[1:],'Author-provided larger-context tables, reproduced in bPdS response Part 1'))

blocks.append(r'''
\subsection{Calibration Coverage and Constructor Matching}
\label{app:revision-matching}
The $2\times2$ Shortcut experiment independently varies whether calibration
includes Shortcut and whether the matched feature-removal constructor is present.
B/D exclude Shortcut from policy calibration; C/D use nonmatching constructions.
The comparisons use the same available supervision. This diagnostic separately
tests calibration transfer and the need for a matched constructor.
Problem-aware construction is useful when domain knowledge is available: SPN
constructs strategic contexts for post-manipulation inputs~\citep{lv2026strategic}.
Shared instead operates without a mechanism identifier.
''')
t = md_tables(section('## Part 1: Generalization','## Part 2:'))[0]
blocks.append(table('matching_ablation','tab:revision-matching','Shortcut calibration/constructor-matching ablation.',t[0],t[1:],'Author-provided 2x2 results in SbVz Part 1'))

blocks.append(r'''
\subsection{Unseen Perturbations and Distribution Shifts}
\label{app:revision-unseen}
All six mechanisms below are excluded from policy calibration and have no
dedicated matching constructors added to the Shared bank. Strengths are
$s\in\{0.2,0.5,0.8,1.0\}$; results average strengths and datasets.
Use source-feature median $m$ and IQR $a$ (falling back to the standard
deviation for zero IQR) as location and scale. Heavy-tailed corruption selects
numeric cells with probability $0.1s$ and adds
$2a\,\operatorname{clip}(t_2,-20,20)$, with $t_2$ Student-$t$ noise.
Column misalignment independently permutes approximately half of nonconstant
columns within approximately $0.5s$ of source rows. Quantization rounds numeric
source features to a grid with step $sa$.

For sensor transformation let $z=(x-m)/a$ and set
$\widetilde x=m+a\operatorname{sign}(z)\log(1+s|z|)/s$.
Rotation randomly pairs numeric features and rotates standardized pairs by
$s\pi/4$ before restoring scale. These two maps act on target-query features for
classification and source-context features for regression. Interaction-dependent
sampling selects up to eight nonconstant feature pairs, computes the average
product of their standardized, $\tanh$-compressed values $g_i$, and resamples
complete source rows with $w_i\propto\exp(3s g_i)$. Associated labels travel
with each row; target labels are not used to generate these changes.
''')
unseen=md_tables(section('## Part 1: Generalization','## Part 2:'))[1]
counts=[('64','35'),('72','35'),('64','35'),('31','31'),('32','33'),('35','33')]
for reg in [False,True]:
    rows=[]
    for i,r in enumerate(unseen[1:]):
        rows.append([r[0], counts[i][int(reg)],r[4] if reg else r[2],r[5] if reg else r[3]])
    key='unseen_reg' if reg else 'unseen_cls'
    blocks.append(table(key,'tab:revision-'+key.replace('_','-'),
        ('Regression under unseen mechanisms ($R^2$/nRMSE).' if reg else 'Classification under unseen mechanisms (Accuracy/AUROC).'),
        ['Unseen mechanism','Datasets','Base','Shared'],rows,'Author-provided six-mechanism tables in SbVz Part 1'))

blocks.append(r'''
\subsection{Natural Shifts and Repeated Runs}
\label{app:revision-natural}
\subsubsection{Folktables Context Lists}
\label{app:revision-folktables-contexts}
Geographic transfer uses California to Michigan (2018); temporal transfer uses
California 2014 to 2018, on Employment, Income, and TravelTime. Specific uses
source-to-target domain-shift contexts: the five Shared alternatives plus three
additional constructions, each capped at 512 rows (six/nine predictions including
Base). The following is the deployed list for this supplementary Folktables study,
not a uniform branch-count claim for every main-benchmark family.
For this deployed list, $R_{\mathrm{Shared}}=5$ and
$R_{\mathrm{Specific}}=8$; including Base gives
$N_{\mathrm{Shared}}=6$ and $N_{\mathrm{Specific}}=9$ predictions per query.
The five shared types and three additional Specific types are enumerated below.
''')
natpart=section('## Part 4: Folktables configuration','## Part 5:', '# Response to Reviewer bPdS')
intro=natpart[natpart.index('**Context construction'):natpart.index('**Comparison with shift-specific methods')]
intro=re.sub(r'^\*\*Context construction.*?\*\*\s*', '', intro, count=1)
before= intro.split('\n- ')[0].strip()
blocks.append(inline(before)+'\n\n'+r'\begin{itemize}'+'\n')
for line in intro.splitlines():
    if line.startswith('- '):blocks.append(r'\item '+inline(line[2:])+'\n')
blocks.append(r'\end{itemize}'+'\n')
blocks.append(r'''
\subsubsection{Folktables Accuracy Comparisons}
\label{app:revision-folktables-comparison}
DistPFN uses the TabICL-v2 backbone~\citep{lee2026distpfn}; Drift-Resilient
TabPFN uses its own model~\citep{helli2024drifttabpfn}. Table~\ref{tab:revision-folktables}
reports the accuracy comparisons and baseline dispersions. The separate
repeated-run configuration below has its own statistics.
''')
nt=md_tables(natpart)
shortrows=[]
for r in nt[0][1:]:
    r=list(r);r[0]={'Geographic':'Geog.','Temporal':'Temp.'}.get(r[0],r[0]);shortrows.append(r)
blocks.append(table('folktables_comparison','tab:revision-folktables',
    'Folktables accuracy; Geog. and Temp. denote geographic and temporal transfer.',
    ['Shift','Task','Domain','Base','DistPFN','Drift-res.','Shared','Specific'],shortrows,
    'Author-provided Folktables comparison table; not merged with the separate variance-run means'))
blocks.append(r'''
\subsubsection{Seed Variability}
\label{app:revision-folktables-variance}
For a separate source-supervised adapted configuration, repeat seeds 0--4 with
fixed data partitions and hyperparameters. Table~\ref{tab:revision-folktables-variance}
reports sample standard deviation and sample variance
$s^2=\frac14\sum_{k=0}^4(a_k-\bar a)^2$. These quantify context sampling,
model randomness, and adaptation under the fixed split; they are not attached
to the different point estimates in Table~\ref{tab:revision-folktables}.
''')
varrows=[]
for r in nt[1][1:]:
    r=list(r);r[0]={'Geographic':'Geog.','Temporal':'Temp.'}.get(r[0],r[0]);varrows.append(r)
blocks.append(table('folktables_variance','tab:revision-folktables-variance',
    r'Five-seed Folktables variability: $\pm$ sample standard deviation (sample variance), Accuracy scale.',
    ['Shift','Task','Domain','Base','Shared','Specific'],varrows,
    'Author-provided dispersion table; multi_seed_natural_v13/summary_mean_std_variance.csv (ddof=1)'))
blocks.append(r'''
\subsubsection{Additional Clinical Shifts}
\label{app:revision-medical}
PhysioNet Sepsis 2019 uses bidirectional hospital-system A/B transfer. UCI
Diabetes 130-US Hospitals uses non-emergency-room to emergency-room admission
source transfer. ID and OOD denote held-out source and target domains.
The table reports the case--control component evaluation with balanced classes
(1:1); this changes prevalence relative to the natural clinical population.
''')
medical=md_tables(section('## Part 2: Robustness','## Part 3:'))[1]
medrows=[]
for r in medical[1:]:
    r=list(r);r[0]=r[0].replace('PhysioNet Sepsis 2019','Sepsis 2019').replace('UCI Diabetes 130-US Hospitals','Diabetes 130-US');medrows.append(r)
blocks.append(table('medical_shifts','tab:revision-medical','Clinical domain-shift component evaluation (Accuracy; 1:1 case--control).',medical[0],medrows,'Author-provided medical table in SbVz Part 2'))

blocks.append(r'''
\subsection{Robust Backbone Fine-Tuning}
\label{app:revision-finetuning}
FT updates TabICL-v2 on 20\% clean and 80\% corruption-augmented training
episodes at strengths $\{0.2,0.5,0.8,1.0\}$. AdamW updates the last four ICL
blocks and prediction components. Held-out training data select checkpoints;
final evaluation queries do not train or select the models. FT+Shared/Specific
freeze the fine-tuned backbone and apply context fusion at inference.
All compared methods use the same queries within this supplementary benchmark.
Shared/Specific use context-and-fusion-adapted configurations; regression clean
and five-perturbation results include supplementary semisynthetic tasks.
''')
ft=md_tables(section('## Part 3: Comparison and complementarity','## Part 4:'))[0]
blocks.append(table('finetuning','tab:revision-finetuning','Fine-tuning and inference-time context adaptation; classification Accuracy/AUROC and regression $R^2$/nRMSE.',ft[0],ft[1:],'Author-provided FT comparison in SbVz Part 3'))
blocks.append(r'''
The recorded run configuration uses eight RTX 5090 GPUs (32 GB each), one job
per GPU, for 175 dataset--track jobs. Selected-model training averages
41.4 seconds/model; all recorded candidates total 147.8 GPU-minutes. The latter
is summed device time, not elapsed wall-clock time. For context, the original
inference-cost experiment reports median Shared latencies of 872 ms for
classification and 141 ms for regression. Direct inference-time use avoids
backbone optimization; these timing measurements describe distinct stages.

\subsection{Gate Diagnostics and Robustness Extensions}
\label{app:revision-gate}
\subsubsection{High-Confidence Classification Errors}
\label{app:revision-high-confidence}
On the reconstructed five-perturbation benchmark, high-confidence Base errors
(maximum class probability at least 0.9) occupy 2.06--2.44\% of all
query--condition pairs. An extended update-strength estimator adds relative
Base/context confidence, disagreement, and branch consensus; branch aggregation
is unchanged and the added score is calibrated on training folds.
''')
gate=md_tables(section('## Part 4: High-confidence errors','## Part 5:'))
blocks.append(table('high_confidence','tab:revision-high-confidence',
    r'High-confidence error corrections on reconstructed five-perturbation data (original $\rightarrow$ extended).',
    ['Backbone','HC errors/all','Correction rate','Corrected','Overall Acc'],gate[0][1:],
    'gate_sensitivity_replay/results/algorithm_repairs_v1/summary/high_confidence.csv; author response BV6f Part 4',
    'Correction rates condition on high-confidence Base errors; overall Accuracy is macro-averaged across datasets and conditions.'))
blocks.append(r'''
\subsubsection{Single-Branch Extremes and Joint Errors}
\label{app:revision-regression-robustness}
The original scalar predictions have no extra clipping. For the single-branch
extension, replace alternative $\widehat y_i^{(r)}$ by
\[
\widetilde y_i^{(r)}=m_i+\operatorname{clip}
 (\widehat y_i^{(r)}-m_i,-3s_{\mathrm{src}},3s_{\mathrm{src}}),
\]
where $m_i$ is the median alternative and $s_{\mathrm{src}}$ the source-label
standard deviation; Base is unchanged. Exactly one alternative lies outside
this interval on 0.109--0.379\% of queries. Clipping changes ordinary mean
$R^2$ by less than 0.0007 in absolute value in all four backbone/track settings.
In the TabICL five-perturbation single-branch stress test, output-displacement
P95 falls from 24.264 to 0.996 source standard deviations.

For correlated errors, a severe event means all five alternatives make
same-direction errors exceeding three evaluation-target standard deviations.
It is materially harmful when the original fused squared loss exceeds Base's
by more than 0.01 times target variance. Rates use all nonzero-strength
query--condition pairs; target labels define these post-hoc events only.
''')
blocks.append(table('joint_error_frequency','tab:revision-joint-errors','Severe joint errors and material fusion harm, per backbone and track.',
    ['Backbone','Track','Severe joint error','Joint error + fusion harm'],gate[1][1:],
    'gate_sensitivity_replay/results/shared_error_rebuttal_summary/final_incidence_all_thresholds.csv, threshold 3'))
blocks.append(r'''
A separate composite extension combines source-calibrated residual correction
with a learned harm-risk score, contracts risky updates toward Base, and bounds
their magnitude using the source scale. This differs from single-branch clipping.
Table~\ref{tab:revision-regression-extension} macro-averages within each track
and then equally weights five-perturbation and seven-shift tracks.
''')
blocks.append(table('regression_extension','tab:revision-regression-extension',r'Composite regression extension on reconstructed tasks (original $\rightarrow$ extended).',
    ['Backbone','Clean $R^2$','Perturb./shift $R^2$','Perturb./shift nRMSE','Loss recovered'],gate[2][1:],
    'gate_sensitivity_replay/results/shared_error_rebuttal_summary/final_natural_metrics.csv and final_error_budgets.csv',
    r'Excess loss is $\max\{\ell(\widehat y_{\mathrm{TabACE}},y)-\ell(\widehat y_{\mathrm{Base}},y),0\}$. Recovery uses its original total in the fixed severe-joint-error cohort, not total prediction error.'))
blocks += [r'\FloatBarrier', r'\endgroup', '']

(PAPER / 'sections/rebuttal_additions.tex').write_text('\n'.join(blocks))
(PAPER / 'revision_data/rebuttal_tables.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2)+'\n')
print(f'Generated {len(DATA)} labelled revision tables and appendix section.')
