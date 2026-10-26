# Hold-out test on the German time use survey of 2022

The German diaries of 2012/13, imputed in the paper, record no places, so the imputation cannot be checked on them. The German time use survey of 2022 (Zeitverwendungserhebung, ZVE 2022) records a place or a mode of transport for every ten-minute slot. Setting these aside, imputing them from the 13 HETUS 2010 donors exactly as the 2012/13 diaries were imputed, and comparing the result with what was recorded measures the transfer error for German respondents directly. Two scripts do this:

1. `src/zve2022_to_hetus.py` converts the scientific use file (SUF) of ZVE 2022 to the layout of the HETUS 2010 SUF (MACT001 to MACT144, WHEREP001 to WHEREP144, INC2, DDV1, WGHT1, HID, PID, DIARY), with the crosswalks of this folder.
2. `src/zve2022_holdout.py` reads the 13 donors, the converted diaries and the aggregate results of the main run (`results/v4.1`), and writes aggregate tables:
   - the seven evaluation metrics, recall and precision by place, hours per day and share at home by slot for the nine variants of the paper, each estimated on all 13 donors;
   - the coverage of plain intervals from 20 repeated imputations, as multiple imputation would report them;
   - for each of the 144 method combinations of the global analysis, Germany's metrics and its errors (imputed minus recorded), set against the 13 donors' errors under the same combination: is Germany's transfer error one of theirs?
   - part B for 2022: the 2,880 imputations of the 2022 diaries, each corrected by the 13 donors' transfer errors, give 37,440 values of every estimate; does their 90% range, over all combinations and over those with Pareto-optimal methods, contain the value recorded in 2022?

The donors' errors, the Pareto-optimal combinations and the home activities are read from the results of the main run, so the 2022 diaries choose and calibrate nothing; they are only scored. The comparison spans a decade and a pandemic as well as the border, so it is a harder test than the paper's application: a transfer error within the donors' range would support the uncertainty statement for 2012/13, one outside it would show what the donors miss.

## Access and permissions

- **Data.** The ZVE 2022 SUF (DOI 10.21242/63911.2022.00.00.3.1.1) is supplied by the research data centres of the statistical offices (FDZ) as CSV files (households, persons, ten-minute slots, summaries), a 95% subsample of about 9,300 households. Institutions with their seat in the EU may use SUFs at their own premises; first-time applicants go through an eligibility check of several weeks, every user must be committed to statistical confidentiality (§ 16 (7) BStatG, at any statistical office), and a fee applies.
- **External information.** The FDZ require external information combined with their data to be agreed in advance, preferably in the application. Here no records are linked: an imputation model estimated on the HETUS 2010 SUF (aggregated location probabilities) is applied to the 2022 diaries, and only aggregate results leave the analysis.
- **Eurostat.** The HETUS 2010 SUF contract binds researchers "not to match the data set with other sources". Applying a model estimated on it to another survey links no records, but a confirmation from Eurostat's microdata access team before the run is advisable. If either provider asks for it, the model can be restricted to cells with at least 50 respondents, so that what is applied to the 2022 diaries is itself a publishable aggregate.
- **Outputs.** Aggregates only, with per-place scores withheld below 50 respondents, as for HETUS. Any result from the SUF must also meet the FDZ's rules for publication.

## Crosswalks

`activity_crosswalk.csv` maps the 175 activity codes of the 2022 list (FDZ, Kodeliste Aktivitäten) to the HETUS ACL 2008 codes of the 2010 SUF. It follows the FDZ key between the codes of 2022 and 2012/13 (Gegenüberstellung der Aktivitätscodes 2022 zu 2012/13), so that the 2022 diaries are coded as the 2012/13 diaries were: online shopping (462) goes with household planning (492 in 2012/13), supervision of a child (478) with physical care (471), visits to the cemetery (533) with religious activities and grave care (534) with gardening. Every non-travel code goes to a code that the 13 donors use; trips go to a travel code, whose purpose the model does not use. Entries marked CHECK in the note column are judgement calls with a stated alternative, to be reviewed against the ACL 2008 definitions.

`location_crosswalk.csv` maps the two-digit codes of the 2022 list of places and modes (FDZ, Liste Ort/Verkehrsmittel), which follows the HETUS 2018 list, to the WHEREP codes of the 2010 SUF: places 11 to 17 to 1 to 7, other place (19) and unspecified place (10, and 00 when the activity is not a trip) to 13, which in the 2010 SUF combines the two, and modes 20 to 39 to 8 to 12. Every slot with a mode counts as transport, as in the donors.

`columns.json` holds the names of the variables in the SUF files. The entries marked TO_CONFIRM must be filled in from the product metadata report (Metadatenreport Teil II) before the converter runs.

## Steps

```
python src/zve2022_to_hetus.py --slots <slot file> --persons <person file> [--days <diary-day file>] \
    --columns data/zve2022/columns.json --out data/restricted/DE22.csv
python src/zve2022_holdout.py data/restricted/DE22.csv --results results/v4.1 --out data/restricted/zve2022_out
python src/export_results.py data/restricted/zve2022_out results/zve2022
```

The converted diaries are microdata and stay under `data/restricted/`; `export_results.py` copies only the aggregate tables. The end-to-end test (`tests/run_tests.py`) runs both scripts on synthetic files (`tests/make_synthetic_zve2022.py`), whose variable names are invented.
