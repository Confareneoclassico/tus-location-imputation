# Reproduce the results of "Imputation of missing locations in time-use surveys".
# make test needs no restricted data. make results needs the HETUS 2010 scientific use
# files, placed as described in data/README.md.

PYTHON ?= python3
NML ?= data/TUS_Eurostat.nml
OUT ?= data/restricted/imputation_out
VERSION ?= 4.1
RESULTS ?= results/v$(VERSION)
FIGURES ?= output

.PHONY: help test results figures

help:
	@echo "make test      end-to-end test on synthetic data, about five minutes"
	@echo "make results   full run on the files listed in $(NML), then export of the aggregates"
	@echo "make figures   figures and tables of the paper from $(RESULTS), in $(FIGURES)/"

test:
	$(PYTHON) tests/run_tests.py

results:
	$(PYTHON) src/location_imputation.py $(NML) --out $(OUT)
	$(PYTHON) src/export_results.py $(OUT)

figures:
	$(PYTHON) src/make_results_figures.py $(RESULTS) $(FIGURES)
