## 🟡 Reviewers will frown

### Acronyms
- **sections/06_results.tex:50** — `DGP` 在 ICLR 版图注中首次出现时未定义，后文也没有给出“data-generating process (DGP)”的明确展开形式。 _Fix:_ 在本节第 38--39 行的首次文字提及处写为 `data-generating processes (DGPs)`，后续再用 `DGP`。
  ```latex
  \caption{Pairwise stability over 180 aligned seed--DGP--scenario units.
  ```

- **tables/overall_results.tex:41** — 主结果表首次使用 `HPO` 但表题和正文均未定义。 _Fix:_ 在表题或首个表项中展开为 `hyperparameter optimization (HPO)`。
  ```latex
    LogReg/Ridge + HPO & 0.754 & 0.867 & 0.543 & 0.684 & 0.418 & 0.582 & -0.559 & 1.175 & 0.719 & 0.775 & -0.490 & 0.851 \\
  ```

- **sections/06_results.tex:89--93** — `ID` 和 `OOD` 在自然偏移结果中未定义。 _Fix:_ 首次写为 `in-distribution (ID)` 和 `out-of-distribution (OOD)`。
  ```latex
Folktables~\citep{ding2021folktables} provides a 2018
California-to-Michigan geographic shift and a 2014-to-2018 temporal shift
within California. Table~\ref{tab:natural-shifts} reports ID and OOD Accuracy
for Employment, Income, and TravelTime. Specific achieves the highest score in
every setting.
  ```

### Terminology consistency
- **sections/01_introduction.tex:33--35 → sections/06_results.tex:74** — 变体名在 `TabACE-shared`/`TabACE-specific` 和 `TabACE-Shared`/`TabACE-Specific` 之间混用大小写；主表也使用小写形式。 _Fix:_ 选定一种作为正式变体名并全文统一（建议 `TabACE-Shared`/`TabACE-Specific`）。
  ```latex
target labels. \method{}-shared uses a fixed context bank across predefined
changes without receiving their family or strength. Given a change family before
inference, \method{}-specific adds corresponding candidates but receives no
  ```
  ```latex
aligned units. TabACE-Shared ranks first in both classification and regression,
  ```

- **tables/tabicl_strength_details.tex:13** — 普通名词 `context` 在句中被不必要地大写，与全文的通用用法不一致；同样的情况还出现在该文件第 66 行。 _Fix:_ 两处均改为 `controlled context perturbations`。
  ```latex
  \caption{TabICL-v2 Accuracy/AUC across the five controlled Context perturbations.
  ```

## 🔵 Optional polish

### Citation spacing
- **sections/01_introduction.tex:6--7** — 引用与前文之间使用普通换行空格，可能在排版时将引用拆到下一行。 _Fix:_ 在 `updates` 后改用不间断空格 `~\citep{...}`。
  ```latex
learning without dataset-specific parameter updates
\citep{muller2023tabpfn,hollmann2025tabpfn,qu2025tabicl,ma2025tabdpt}.
  ```

- **sections/01_introduction.tex:9--10** — 引用前使用普通空格。 _Fix:_ 改为 `fixed~\citep{...}`。
  ```latex
Changing context examples, features, or labels can still alter the
output with model and query fixed \citep{abdullah2026contextsampling}.
  ```

- **sections/01_introduction.tex:14--16** — 引用与前文之间使用普通换行空格。 _Fix:_ 在 `alignment` 后改用 `~\citep{...}`。
  ```latex
selection, feature or response transformation, and distribution alignment
\citep{feuer2024tunetables,thomas2024localpfn,xu2025mixturepfn,
treerath2026tabaicl,hollmann2025tabpfn,khalifa2023dense}. These constructions
  ```
