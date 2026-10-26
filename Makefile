# Reproduce the results of "Imputation of missing locations in time-use surveys".
# make test needs no restricted data. make results needs the HETUS 2010 scientific use
# files, placed as described in data/README.md.

PYTHON ?= python3
NML ?= data/TUS_Eurostat.nml
OUT ?= data/restricted/imputation_out
VERSION ?= 4.1
RESULTS ?= results/v$(VERSION)

.PHONY: help test results figures paper clean

help:
	@echo "make test      end-to-end test on synthetic data, about seven minutes"
	@echo "make results   full run on the files listed in $(NML), then export of the aggregates"
	@echo "make figures   figures and tables of the manuscript from $(RESULTS)"
	@echo "make paper     compile manuscript/build/paper.pdf"
	@echo "make clean     remove LaTeX build files"

test:
	$(PYTHON) tests/run_tests.py

results:
	$(PYTHON) src/location_imputation.py $(NML) --out $(OUT)
	$(PYTHON) src/export_results.py $(OUT)

figures:
	$(PYTHON) src/make_results_figures.py $(RESULTS) manuscript

paper:
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex

clean:
	cd manuscript && latexmk -C -outdir=build paper.tex
