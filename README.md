# Imputation of missing locations in time-use surveys

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23671806.svg)](https://doi.org/10.5281/zenodo.23671806)

Code and aggregate results for the paper "Imputation of missing locations in time-use surveys" by Samuele Lo Piano.

The German contribution to the 2010 round of the Harmonised European Time Use Surveys (HETUS) records activities and trips but not where activities took place. The pipeline in this repository imputes these locations from the 13 HETUS countries that record all of them, with a Markov chain over activity episodes that conditions on the activity, the time of day, the previous location, intervening trips and age. It tests the method by imputing each of these countries from the others, and a global sensitivity analysis over the full factorial of modelling choices, coding assumptions, random draws and transfer errors apportions the variance of the evaluation metrics and of the German estimates among them with exact Sobol' indices, and measures with optimal-transport indices how far each choice moves the distribution of each metric.

## Repository layout

```
tus-location-imputation/
├── README.md                     this file
├── CITATION.cff                  how to cite
├── LICENSE                       MIT licence (code)
├── Makefile                      make test, make results, make figures
├── requirements.txt              Python packages
├── .github/workflows/tests.yml   runs the end-to-end test on every push
├── src/
│   ├── location_imputation.py    the pipeline, version 4.1
│   ├── export_results.py         copies the aggregate outputs into results/
│   ├── make_results_figures.py   figures and tables of the paper, from results/
│   └── hetus_inventory.py        inventory of the country files (aggregate counts only)
├── tests/
│   ├── make_synthetic.py         synthetic files with the layout of the HETUS files
│   ├── test_indices.py           exact Sobol' and optimal-transport indices against known values
│   ├── test_disclosure.py        disclosure control of the per-place scores
│   └── run_tests.py              end-to-end test on synthetic data
├── data/
│   ├── README.md                 how to obtain and place the HETUS files
│   ├── TUS_Eurostat.nml.example  template of the settings file
│   └── restricted/               HETUS files and microdata outputs (never committed)
└── results/
    └── v4.1/                     aggregate outputs of the run behind the paper
```

## Reproducing the results

The code needs Python 3.10 or later with NumPy, pandas and Matplotlib (`pip install -r requirements.txt`). The end-to-end test gives identical results with the oldest versions allowed there (Python 3.10, NumPy 1.24, pandas 2.0, Matplotlib 3.7) and with Python 3.13, NumPy 2.5, pandas 3.0 and Matplotlib 3.11. Tables are written as LaTeX (booktabs and siunitx).

The end-to-end test needs no restricted data. It writes synthetic country files with the layout and quirks of the HETUS scientific use files to a temporary folder, runs the full pipeline on them, exports the aggregate outputs and builds the figures and tables from them, in about five minutes. GitHub Actions runs it on every push.

```
make test
```

With access to the scientific use files, place them as described in [data/README.md](data/README.md) and run

```
make results     # the pipeline (about 30 minutes on a laptop), then export_results.py
make figures     # output/figures and output/tables from results/v4.1
```

`make results` writes every output, including the imputed German diaries, to data/restricted/imputation_out/, and then copies the aggregate files into results/v<version>/. If the country files and a settings file already exist elsewhere, point to that settings file instead: `make results NML=/path/to/TUS_Eurostat.nml`. A full run needs about 1.7 GB of memory. Without make, the same steps are

```
python src/location_imputation.py data/TUS_Eurostat.nml --out data/restricted/imputation_out
python src/export_results.py data/restricted/imputation_out
python src/make_results_figures.py results/v4.1 output
python tests/run_tests.py
```

All random draws are seeded (the seed is recorded in run_settings.json), so a run on the same files with the same package versions reproduces the results.

## Data

The HETUS scientific use files are available from Eurostat to recognised research entities for approved research projects and cannot be redistributed. Nothing derived from them at the level of individual diaries may leave data/restricted/ either. This includes the imputed German diaries, which the pipeline writes to files whose names start with PRIVATE_. The only route from data/restricted/ into the repository is src/export_results.py, which copies the aggregate tables of the latest run and never a PRIVATE_ file, and .gitignore excludes data/restricted/, PRIVATE_ files and .npz files as a second safeguard. The aggregate outputs report statistics for whole countries and for German groups of at least 3,706 diaries. Per-place scores of a country, and the location shares of an activity, are withheld where fewer than 50 respondents stand behind them, in line with Eurostat's guidelines for publishing results from the scientific use files, which bar results based on fewer than 20 respondents and require results based on 20 to 49 to be flagged.

This work is based on data from Eurostat, Harmonised European Time Use Surveys, 2010 round. The responsibility for all conclusions drawn from the data lies entirely with the author.

## Results

results/v4.1/ holds the aggregate outputs of version 4.1 of the pipeline, run on 25 October 2026, from which the figures and tables of the paper are built. The README.txt written by the pipeline in the folder describes each file. checks.csv lists the data and consistency checks: 320 passed and 6 gave warnings. The files for Finland, France, Poland and the United Kingdom hold 27, 204, 2,102 and 40 fewer diaries than the documentation of the scientific use files lists, 0.02% of the British diaries lack the day of the week, and two by-slot files that an earlier run had written to the same output folder were kept out of the export. The local folder names in run_settings.json were removed by export_results.py.

## Licence and citation

The code is released under the MIT licence (LICENSE). The figures and aggregate results are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). If you use the code or the results, please cite the paper and this repository as given in CITATION.cff.

Version 4.1, the version behind the paper, is archived on Zenodo with the DOI 10.5281/zenodo.23671806 (https://doi.org/10.5281/zenodo.23671806).

Correspondence: Samuele Lo Piano, Faculty of Management and Economics, Gdansk University of Technology (samuele.lo.piano@pg.edu.pl).
