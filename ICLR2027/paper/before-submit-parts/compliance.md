## 🔴 Desk-reject risk

### Required AI-use disclosure
- **sections/08_conclusion.tex:31--36** — 当前声明只明说“写作润色”和“起草部分稿件”，并以含糊的 `support tasks not separately listed` 代替其他用途。这与本次稿件中实际的 AI 参与不匹配：还包括数学证明/理论修订、方法与实验反馈、实现与结果解释；图表和参考文献处理也属于实际用途。ICLR 2027 将前述理论、证明、方法/实验、实现和结果解释列为必须具体披露的类别；笼统的“其他支持任务”不足以表明使用方式，也与当前表单勾选不完全一致。 _Fix:_ 在论文声明中逐类明确披露上述实际用途，并同步更新 OpenReview 表单勾选；保留作者审核与负责声明。
  ```latex
\section*{AI Use Statement}

The authors used generative AI tools to aid and polish writing, draft portions
of the manuscript, and support tasks not separately listed in the submission
form. All AI-assisted material was reviewed and verified by the authors, who
take responsibility for the final manuscript.
  ```

## Compliance checks passed

- The review source uses `\author{Anonymous Authors}` and leaves `\iclrfinalcopy` commented out.
- The local `iclr2027_conference.sty` SHA-256 hash exactly matches the official ICLR 2027 style archive downloaded from the live author-guidelines page.
- References precede the appendix; the appendix begins on a new page and may be unlimited under the live ICLR 2027 rules.
- The mandatory unnumbered `AI Use Statement` is present before the references; an unnumbered reproducibility statement is also present.
- No identity-revealing author names, profile/repository URLs, acknowledgments, or PDF-author metadata were found, including in comments.
