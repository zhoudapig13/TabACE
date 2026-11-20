## 🟠 Reviewers will frown

### Bibliography — metadata

- **references.bib:267** — `lee2026distpfn` contains only Seunghan Lee, but the accepted ICML 2026 paper's author-maintained repository and LG AI Research publication record list ten authors: Seunghan Lee, Jaehoon Lee, Jun Seo, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, SoonYoung Lee, and Wonbin Ahn. (The arXiv abstract metadata itself is inconsistent and currently exposes only the first author, so it should not be copied blindly.) _Fix:_ replace the one-author field with the complete ICML author list.
  ```latex
  @inproceedings{lee2026distpfn,
    title={Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment},
    author={Seunghan Lee},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=8N2kUoknvy}
  }
  ```

- **references.bib:353** — `lv2026strategic` lists four middle authors in an order that disagrees with the current official arXiv record: the official order is `... Jinxuan Yang, Kun Kuang, Yuanlong Chen, Mingyang Geng, Wanrong Huang, Shixuan Liu, ...`. _Fix:_ reorder those authors to match the version of record.
  ```latex
  @inproceedings{lv2026strategic,
    title={When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach},
    author={Xinpeng Lv and Yunxin Mao and Renzhe Xu and Chunyuan Zheng and Yikai Chen and Haoxuan Li and Jinxuan Yang and Yuanlong Chen and Kun Kuang and Mingyang Geng and Shixuan Liu and Wanrong Huang and Shaowu Yang and Wenjing Yang and Zhouchen Lin and Haotian Wang},
  ```

- **references.bib:594** — `fischer2023openmlctr23` omits coauthor Liana Harutyunyan. The official MCML record lists `S. F. Fischer, L. Harutyunyan, M. Feurer, B. Bischl`. _Fix:_ add `Liana Harutyunyan` in second position and verify the preferred spelling against the cited OpenReview version.
  ```latex
  @inproceedings{fischer2023openmlctr23,
    title={Open{ML}-{CTR}23 {\textendash} A curated tabular regression benchmarking suite},
    author={Sebastian Felix Fischer and Matthias Feurer and Bernd Bischl},
    booktitle={AutoML Conference 2023 (Workshop)},
    year={2023},
    url={https://openreview.net/forum?id=HebAOoMm94}
  }
  ```

- **references.bib:527** — the first author of `thielmann2024tabularnn` is parsed by BibTeX as family name `F Thielmann`, producing the compiled citation label `F Thielmann & Samiee`; the paper and project identify the author as Anton Frederik Thielmann. _Fix:_ encode the author as `Thielmann, Anton Frederik` (or `Thielmann, Anton F.`).
  ```latex
  @inproceedings{thielmann2024tabularnn,
    title = 	 {On the Efficiency of {NLP}-Inspired Methods for Tabular Deep Learning},
    author =       {F Thielmann, Anton and Samiee, Soheila},
  ```

- **references.bib:744** — `zhou2025ftat` gives the fourth author as `Yufeng Li`, whereas the official AAAI record uses `Yu-Feng Li`. _Fix:_ preserve the official hyphenated given name, e.g. `Li, Yu{-}Feng`.
  ```latex
  @inproceedings{zhou2025ftat,
    author       = {Zhi Zhou and
                    Kun{-}Yang Yu and
                    Lan{-}Zhe Guo and
                    Yufeng Li},
  ```

### Bibliography — usage

- **references.bib:165** — `huang2026lara` is present in the assembled bibliography but is not cited by any active `\cite...{}` or `\nocite{}` in `main_iclr.tex` and its included files. _Fix:_ remove the entry if it is not needed, or cite it at the relevant claim.
  ```latex
  @inproceedings{huang2026lara,
    author       = {Chengsong Huang and
                    Langlin Huang and
                    Jiaxin Huang},
    editor       = {Vera Demberg and
                    Kentaro Inui and
                    Llu{\'{\i}}s Marquez},
    title        = {Divide, Reweight, and Conquer: {A} Logit Arithmetic Approach for In-Context
                    Learning},
  ```

- **references.bib:188** — `liu2025beta` is present in the assembled bibliography but is not cited by any active `\cite...{}` or `\nocite{}` in `main_iclr.tex` and its included files. _Fix:_ remove the entry if it is not needed, or cite it at the relevant claim.
  ```latex
  @inproceedings{liu2025beta,
    title = 	 {{T}ab{PFN} Unleashed: A Scalable and Effective Solution to Tabular Classification Problems},
    author =       {Liu, Siyang and Ye, Han-Jia},
    booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  ```

## 🔵 Optional polish

### Bibliography — BibTeX hygiene

- **references.bib:360** — `cox1958binary` uses a single hyphen for its page range and stores an ordinary PDF URL in `eprint`; these fields are noncanonical BibTeX and can render inconsistently across styles. _Fix:_ use `pages = {215--232}` and remove the `eprint` field (the DOI/URL already identifies the article).
  ```latex
  @article{cox1958binary,
    author = {Cox, D. R.},
    title = {The Regression Analysis of Binary Sequences},
    journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
    volume = {20},
    number = {2},
    pages = {215-232},
    year = {1958},
    month = {07},
    abstract = {A sequence of 0’s and 1’s is observed and it is suspected that the chance that a particular trial is a 1 depends on the value of one or more independent variables. Tests and estimates for such situations are considered, dealing first with problems in which the independent variable is preassigned and then with independent variables that are functions of the sequence. There is a considerable amount of earlier work, which is reviewed.},
    issn = {0035-9246},
    doi = {10.1111/j.2517-6161.1958.tb00292.x},
    url = {https://doi.org/10.1111/j.2517-6161.1958.tb00292.x},
    eprint = {https://academic.oup.com/jrsssb/article-pdf/20/2/215/49096552/jrsssb_20_2_215.pdf}
  }
  ```

- **references.bib:377** — `hoerl1970ridge` duplicates its DOI URL in `eprint`, although it is not a preprint identifier. _Fix:_ remove `eprint`; retain `doi` and `url`.
  ```latex
  @article{hoerl1970ridge,
    author = {Arthur E. Hoerl and Robert W. Kennard},
    title = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
    journal = {Technometrics},
    volume = {12},
    number = {1},
    pages = {55--67},
    year = {1970},
    publisher = {Taylor \& Francis},
    doi = {10.1080/00401706.1970.10488634},
    URL = {https://doi.org/10.1080/00401706.1970.10488634},
    eprint = {https://doi.org/10.1080/00401706.1970.10488634}
  }
  ```

- **references.bib:544** — `ahamed2024mambatab` uses a single hyphen in a numeric page range. _Fix:_ change `369-375` to BibTeX's range form `369--375`.
  ```latex
  @inproceedings{ahamed2024mambatab,
    author={Ahamed, Md Atik and Cheng, Qiang},
    booktitle={2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR)}, 
    title={{MambaTab}: A Plug-and-Play Model for Learning Tabular Data}, 
    year={2024},
    pages={369-375},
    doi={10.1109/MIPR62202.2024.00065}
  }
  ```

- **references.bib:456, references.bib:580** — two entries retain empty `pages = {}` fields. _Fix:_ remove the empty fields; do not invent page ranges if the proceedings do not assign them.
  ```latex
  % references.bib:456
  @inproceedings{prokhorenkova2018catboost,
    author = {Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
    pages = {},
  % references.bib:580
  @inproceedings{bischl2021openml,
    author = {Bischl, Bernd and Casalicchio, Giuseppe and Feurer, Matthias and Gijsbers, Pieter and Hutter, Frank and Lang, Michel and Gomes Mantovani, Rafael and van Rijn, Jan and Vanschoren, Joaquin},
    booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
    editor = {J. Vanschoren and S. Yeung},
    pages = {},
  ```
