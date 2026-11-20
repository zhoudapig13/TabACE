# Before-Submit Report — `main_iclr.tex`

_生成时间：2026-09-25 23:53 +08:00 · 投稿：ICLR 2027 main track · 版本：双盲评审稿 · 规则：官方实时规则_

## Summary

- 🔴 Desk-reject risks：1
- 🟠 Reviewers will frown：15
- 🔵 Optional polish：8
- 审计范围：15 个组装后的 TeX 文件、1 个 BibTeX 文件、54 条书目记录、主要结果表及 11 张定量图。
- 正向结果：官方模板文件匹配；未发现匿名信息泄露、缺失引用键、重复 citation key 或明确撤稿文献。

## 🔴 Desk-reject risk

### Mandatory AI-use disclosure

- **`sections/08_conclusion.tex:31--36`** — 当前声明仅明确披露“写作润色”和“起草部分稿件”，并用含糊的 `support tasks not separately listed` 概括其他用途。结合本次稿件工作记录，AI 还参与了数学证明/理论修订、方法与实验反馈、实现和结果解释，以及图表与参考文献处理。ICLR 2027 要求对理论框架、数学主张/证明、方法/实验、实现和结果解释等用途作具体披露；论文声明与 OpenReview 表单也应一致。_修正：_ 如实逐类列明实际用途，并同步更新表单勾选；保留作者审核和最终负责声明。
  ```latex
  \section*{AI Use Statement}

  The authors used generative AI tools to aid and polish writing, draft portions
  of the manuscript, and support tasks not separately listed in the submission
  form. All AI-assisted material was reviewed and verified by the authors, who
  take responsibility for the final manuscript.
  ```

## 🟠 Reviewers will frown

### Numerical and protocol consistency

- **`sections/00_abstract_iclr.tex:15--16` ↔ `sections/06_results.tex:133--139`** — 摘要称缓存带来 “up to fivefold” 加速，但正文报告约 $5.7\times$，图中标注为 $5.8\times$ 和 $5.7\times$。“up to” 表示上限，因此 5 倍不能作为 5.7--5.8 倍的近似上限。_修正：_ 核实最终采用的图中数值后，统一写成 “up to $5.8\times$” 或 “nearly sixfold”。
  ```latex
  % sections/00_abstract_iclr.tex:15--16
  caching reusable context computations accelerates repeated inference by up to
  fivefold.
  ```
  ```latex
  % sections/06_results.tex:133--139
  \paragraph{Inference cost.}
  For the fixed five-context Shared bank, Figure~\ref{fig:inference-cost}
  decomposes wall-clock latency into six backbone passes (Base plus five
  alternatives) and context construction/fusion. Caching reusable propensity
  quantities yields about $5.7\times$ faster repeated inference. After including
  the one-time cache construction, it retains about $3.9\times$ amortized speedup
  over ten batches.
  ```

- **`sections/context_construction_details.tex:272--278` ↔ `sections/appendix.tex:1291--1295` ↔ `sections/06_results.tex:97--103`** — Specific 一处被描述为由冻结后的成员列表 $\Phi_z$ 决定部署分支数，另一处又说每个配置保留完整候选池；时延图同时给出 Shared/Specific，却只笼统写“五个 alternative contexts”。因此读者无法判断 Specific 的实际前向次数。_修正：_ 统一候选池与部署成员列表的关系，并明确时延实验中 Shared 和 Specific 各自使用的分支数；若两者都固定为 5，应直接写明。
  ```latex
  % sections/context_construction_details.tex:272--278
  Specific retains all Shared contexts and activates the complete registered pool
  for family $z$. The pool has no protocol-level candidate-count cap. Each range,
  repair strength, subspace, and transport level below defines a separate
  candidate. This pool is the outer-fold selection space, not the deployed branch
  count: the selected member list $\Phi_z$ is frozen before held-out inference,
  which uses one Base pass and $R_z=|\Phi_z|$ alternative passes. The latency
  experiment fixes $R_z=5$ for Shared.
  ```
  ```latex
  % sections/appendix.tex:1291--1295
  Candidate selection
    & Each configuration retains its complete registered pool. Five-fold
      dataset-level OOF fits source-side policy parameters on the other four
      folds and freezes them for the held-out dataset, using cross-entropy for
      classification and squared error for regression. \\
  ```
  ```latex
  % sections/06_results.tex:97--103
  \caption{Inference-cost ablation with five alternative contexts and a
  separately evaluated Base branch. Panels (a--b) decompose median latency;
  (c) compares original and cached inference; (d) includes the one-time cache
  setup in the amortized speedup.}
  ```

- **`sections/appendix.tex:481--485` ↔ `sections/appendix.tex:1399--1405,1423--1428,1445--1446`** — 协议称两个 track 的归一化分母不同，因此 nMAE/nRMSE 只在 track 内比较、跨 track 汇总使用 $R^2$；但三个消融图又对两条 track 的 nMAE/nRMSE 做等权汇总，形成直接冲突。_修正：_ 优先按既定协议将跨轨回归消融改为 $R^2$；若必须汇总 nMAE/nRMSE，则需先定义统一归一化方式，并相应改写协议。
  ```latex
  % sections/appendix.tex:481--485
  Both are population standard deviations (\texttt{ddof=0}).  Because the two
  tracks use different normalization scales, nMAE and nRMSE are compared only
  within a track; cross-track regression summaries use $R^2$.  In the shift
  track, $\operatorname{std}(y_{\mathrm{target}})$ is computed only after all
  predictions have been produced and is used solely for metric evaluation.
  ```
  ```latex
  % sections/appendix.tex:1423--1428
  \caption{Aggregation-rule ablation at $R=5$, equally weighting the
  five-perturbation and seven-shift tracks. Panels compare classification
  Accuracy and regression nRMSE.}
  ```

### Bibliography — metadata

- **`references.bib:267`** — `lee2026distpfn` 只列 Seunghan Lee；作者维护的 accepted-paper repository 与 LG AI Research publication record 列出 10 位作者。arXiv 摘要页目前只显示第一作者，不能据此覆盖 accepted version。_修正：_ 按 ICML 接收版本补齐完整作者列表。
  ```latex
  @inproceedings{lee2026distpfn,
    title={Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment},
    author={Seunghan Lee},
  ```

- **`references.bib:353`** — `lv2026strategic` 的四位中间作者顺序与当前官方 arXiv 记录不一致；官方顺序为 `Jinxuan Yang, Kun Kuang, Yuanlong Chen, Mingyang Geng, Wanrong Huang, Shixuan Liu`。_修正：_ 按版本记录重排作者。
  ```latex
  author={Xinpeng Lv and Yunxin Mao and Renzhe Xu and Chunyuan Zheng and Yikai Chen and Haoxuan Li and Jinxuan Yang and Yuanlong Chen and Kun Kuang and Mingyang Geng and Shixuan Liu and Wanrong Huang and Shaowu Yang and Wenjing Yang and Zhouchen Lin and Haotian Wang},
  ```

- **`references.bib:594`** — `fischer2023openmlctr23` 漏掉共同作者 Liana Harutyunyan；官方记录作者为 S. F. Fischer、L. Harutyunyan、M. Feurer、B. Bischl。_修正：_ 在第二位补入 `Liana Harutyunyan`，并以引用的 OpenReview 版本核对拼写。
  ```latex
  author={Sebastian Felix Fischer and Matthias Feurer and Bernd Bischl},
  ```

- **`references.bib:527`** — `F Thielmann, Anton` 会被 BibTeX 解析成姓氏 `F Thielmann`，从而生成错误的引用标签。_修正：_ 写成 `Thielmann, Anton Frederik` 或 `Thielmann, Anton F.`。
  ```latex
  author =       {F Thielmann, Anton and Samiee, Soheila},
  ```

- **`references.bib:744`** — `zhou2025ftat` 将第四作者写成 `Yufeng Li`，而 AAAI 官方记录为 `Yu-Feng Li`。_修正：_ 使用 `Li, Yu{-}Feng`。
  ```latex
  author       = {Zhi Zhou and
                  Kun{-}Yang Yu and
                  Lan{-}Zhe Guo and
                  Yufeng Li},
  ```

### Bibliography — usage

- **`references.bib:165`** — `huang2026lara` 存在于 Bib 文件中，但未被任何活动的 `\cite...{}` 或 `\nocite{}` 引用。_修正：_ 若正文不需要则删除；若确实支撑相关工作论述，则在对应位置引用。
  ```latex
  @inproceedings{huang2026lara,
    title={Divide, Reweight, and Conquer: {A} Logit Arithmetic Approach for In-Context
           Learning},
  ```

- **`references.bib:188`** — `liu2025beta` 同样未被正文或附录引用。_修正：_ 删除未用条目，或只在确实相关的论述处加入引用。
  ```latex
  @inproceedings{liu2025beta,
    title = {{T}ab{PFN} Unleashed: A Scalable and Effective Solution to Tabular Classification Problems},
  ```

### Acronyms and terminology

- **`sections/06_results.tex:50`** — `DGP` 首次出现时未定义。_修正：_ 首次文字提及时写为 `data-generating processes (DGPs)`。
  ```latex
  \caption{Pairwise stability over 180 aligned seed--DGP--scenario units.
  ```

- **`tables/overall_results.tex:41`** — 主结果表首次使用 `HPO`，但正文和表题均未定义。_修正：_ 首次写为 `hyperparameter optimization (HPO)`。
  ```latex
  LogReg/Ridge + HPO & 0.754 & 0.867 & 0.543 & 0.684 & 0.418 & 0.582 & -0.559 & 1.175 & 0.719 & 0.775 & -0.490 & 0.851 \\
  ```

- **`sections/06_results.tex:89--93`** — `ID` 与 `OOD` 未定义。_修正：_ 首次写为 `in-distribution (ID)` 和 `out-of-distribution (OOD)`。
  ```latex
  California-to-Michigan geographic shift and a 2014-to-2018 temporal shift
  within California. Table~\ref{tab:natural-shifts} reports ID and OOD Accuracy
  for Employment, Income, and TravelTime.
  ```

- **`sections/01_introduction.tex:33--35` ↔ `sections/06_results.tex:74`** — 变体名在 `TabACE-shared`/`TabACE-specific` 与 `TabACE-Shared`/`TabACE-Specific` 之间混用。_修正：_ 全文统一为一种正式写法，建议 `TabACE-Shared`/`TabACE-Specific`。
  ```latex
  % sections/01_introduction.tex:33--35
  target labels. \method{}-shared uses a fixed context bank across predefined
  changes without receiving their family or strength. Given a change family before
  inference, \method{}-specific adds corresponding candidates but receives no
  ```
  ```latex
  % sections/06_results.tex:74
  aligned units. TabACE-Shared ranks first in both classification and regression,
  ```

- **`tables/tabicl_strength_details.tex:13,66`** — 普通名词 `context` 在两个表题中被不必要地大写，与全文用法不一致。_修正：_ 改为 `controlled context perturbations`。
  ```latex
  \caption{TabICL-v2 Accuracy/AUC across the five controlled Context perturbations.
  ```

## 🔵 Optional polish

### Table semantics

- **`tables/overall_results.tex:61--65`** — Shared/Specific 的全部数值都被加粗，包括并非组内最优的单元格；表题也未解释粗体含义，容易让读者误以为粗体代表最佳结果。_修正：_ 只对真实最优值/并列最优值加粗，或明确说明粗体仅用于标识 TabACE 行、不表示最优。
  ```latex
  TabICL-v2 & 0.804 & 0.901 & 0.739 & 0.476 & 0.494 & 0.676 & -1.077 & 1.270 & 0.763 & 0.826 & 0.273 & 0.730 \\
  \rowcolor{tabaceSharedRow}
  \textcolor{tabaceSharedText}{\textbf{+ \method{}-shared}} & \textcolor{tabaceSharedText}{\textbf{0.804}} & \textcolor{tabaceSharedText}{\textbf{0.901}} & \textcolor{tabaceSharedText}{\textbf{0.738}} & \textcolor{tabaceSharedText}{\textbf{0.478}} & \textcolor{tabaceSharedText}{\textbf{0.575}} & \textcolor{tabaceSharedText}{\textbf{0.720}} & \textcolor{tabaceSharedText}{\textbf{0.363}} & \textcolor{tabaceSharedText}{\textbf{0.783}} & \textcolor{tabaceSharedText}{\textbf{0.772}} & \textcolor{tabaceSharedText}{\textbf{0.831}} & \textcolor{tabaceSharedText}{\textbf{0.467}} & \textcolor{tabaceSharedText}{\textbf{0.659}} \\
  ```

### Bibliography — BibTeX hygiene

- **`references.bib:360`** — `cox1958binary` 的页码范围使用单连字符，且把普通 PDF URL 放在 `eprint`。_修正：_ 改为 `pages={215--232}` 并删除 `eprint`；保留 DOI/URL 即可。
  ```latex
  pages = {215-232},
  eprint = {https://academic.oup.com/jrsssb/article-pdf/20/2/215/49096552/jrsssb_20_2_215.pdf}
  ```

- **`references.bib:377`** — `hoerl1970ridge` 在 `eprint` 中重复存放 DOI URL。_修正：_ 删除 `eprint`，保留 `doi` 与 `url`。
  ```latex
  eprint = {https://doi.org/10.1080/00401706.1970.10488634}
  ```

- **`references.bib:544`** — `ahamed2024mambatab` 的数字页码范围使用单连字符。_修正：_ 改为 `369--375`。
  ```latex
  pages={369-375},
  ```

- **`references.bib:456,580`** — `prokhorenkova2018catboost` 与 `bischl2021openml` 保留空的 `pages={}`。_修正：_ 删除空字段，不要猜测页码。
  ```latex
  pages = {},
  ```

### Citation spacing

- **`sections/01_introduction.tex:6--7`** — 引用与前文之间是普通换行空格，可能把引用拆到下一行。_修正：_ 将 `updates` 后的空格改成 `~\citep{...}`。
  ```latex
  learning without dataset-specific parameter updates
  \citep{muller2023tabpfn,hollmann2025tabpfn,qu2025tabicl,ma2025tabdpt}.
  ```

- **`sections/01_introduction.tex:9--10`** — `fixed` 与引用之间是普通空格。_修正：_ 改成 `fixed~\citep{...}`。
  ```latex
  Changing context examples, features, or labels can still alter the
  output with model and query fixed \citep{abdullah2026contextsampling}.
  ```

- **`sections/01_introduction.tex:14--16`** — `alignment` 与引用之间是普通换行空格。_修正：_ 改成 `alignment~\citep{...}`。
  ```latex
  selection, feature or response transformation, and distribution alignment
  \citep{feuer2024tunetables,thomas2024localpfn,xu2025mixturepfn,
  treerath2026tabaicl,hollmann2025tabpfn,khalifa2023dense}. These constructions
  ```

## What I checked / skipped

- ✅ BeforeSubmitSkill — 已从 `thinkwee/BeforeSubmitSkill` 安装到 `C:\Users\26435\.codex\skills\before-submit`，并按其并行审计流程执行。
- ✅ Source assembly — 解析 `main_iclr.tex` 及其 14 个活动的输入文件；未发现组装警告。
- ✅ ICLR compliance — 以官方 ICLR 2027 author guidelines 与 AI policy 为准，检查双盲、样式文件、必需声明、参考文献与附录位置。
- ✅ Template integrity — 本地 `iclr2027_conference.sty`、`.bst`、`natbib.sty`、`fancyhdr.sty` 与官方模板包哈希一致。
- ✅ Anonymization — 未发现作者身份、个人/机构仓库 URL、致谢或 PDF author metadata 泄露。
- ✅ Bibliography — 54 条记录；52 条被引；0 个缺失 citation key；0 个重复 key；34 个自动异常均进行了联网复核。
- ✅ Citation relevance and status — 未确认到明确不相关引用、撤稿记录或应当替换却未替换的正式版本。
- ✅ Faithfulness — 检查正文、表格与主要定量图之间的数字、协议和措辞一致性。
- ⏭️ Clean compile + authoritative page count — 跳过：当前可用的是 MiKTeX `pdflatex`，而该 skill 的严格复现规则要求 TeX Live/TinyTeX/Overleaf 等价工具链；未用其他引擎冒充官方编译结果。提交前请在 Overleaf 或官方 TeX Live 环境做最后一次 clean build 与页数核验。
- ⏭️ HTML report — 未生成；仅在用户需要时由 Markdown 报告转换。

## Venue rules used

- Source: [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)；[ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
- Review main-text limit：9 页；参考文献不计入；附录可在参考文献之后不限页数。
- Double-blind：是；当前 review source 保持 `Anonymous Authors` 且未启用 `\iclrfinalcopy`。
- Mandatory：论文中必须有 AI Use Statement；所有实际适用的 AI 使用类别还应与投稿表单保持一致。
