ARXIV_MAIN := main
ICLR_MAIN := main_iclr
ARXIV_ZIP := dist/tabace-arxiv-source.zip
SECTIONS := $(wildcard sections/*.tex)

.PHONY: all arxiv iclr check-arxiv arxiv-package clean

all: arxiv

arxiv:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(ARXIV_MAIN).tex

iclr:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(ICLR_MAIN).tex

check-arxiv:
	@! rg -n --fixed-strings '\TODO{' main.tex macros.tex sections || \
		(printf '%s\n' 'Refusing release: resolve every \TODO before uploading to arXiv.'; exit 1)
	@! rg -n --fixed-strings 'Anonymous Authors' main.tex || \
		(printf '%s\n' 'Refusing release: replace the anonymous author placeholder.'; exit 1)

arxiv-package: check-arxiv arxiv
	@mkdir -p dist
	@rm -f $(ARXIV_ZIP)
	zip -q $(ARXIV_ZIP) main.tex macros.tex references.bib main.bbl $(SECTIONS)
	@printf '%s\n' 'Created $(ARXIV_ZIP)'

clean:
	latexmk -C $(ARXIV_MAIN).tex
	latexmk -C $(ICLR_MAIN).tex
