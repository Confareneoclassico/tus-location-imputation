# Data

The pipeline reads the scientific use files of the 2010 round of the Harmonised European Time Use Surveys (HETUS). Eurostat makes them available to recognised research entities for research projects it has approved, through its procedure for access to European microdata (https://ec.europa.eu/eurostat/web/microdata/harmonised-european-time-use-surveys). The files cannot be redistributed, so this folder holds only this description and a template of the settings file.

## Placing the files

Put the country files in data/restricted/hetus2010/ and copy TUS_Eurostat.nml.example to TUS_Eurostat.nml in this folder. The settings file maps each country to its file with a path relative to this folder, so the files can keep the names they were delivered with. If the files already live elsewhere, for instance in a folder shared with other projects, write absolute paths into TUS_Eurostat.nml instead of moving them. Both data/restricted/ and data/TUS_Eurostat.nml are excluded from version control. Keep the scientific use files and data/restricted/ out of folders synchronised with cloud services, unless the conditions of access allow it.

The pipeline expects one CSV file per country, with one row per diary day, for the 17 countries of the scientific use files: AT, BE, DE, EE, EL, ES, FI, FR, HU, IT, LU, NL, NO, PL, RO, RS and UK. It reads the main activity (MACT1 to MACT144) and the location or mode of transport (WHEREP1 to WHEREP144) of each ten-minute slot, or of 96 fifteen-minute slots for Austria, together with the age band (INC2), the day of the week (DDV1), the combined individual and day weight (WGHT1) and the identifiers HID, PID and DIARY. Column names may be in upper or lower case.

Before the first full run, `python src/hetus_inventory.py data/TUS_Eurostat.nml --out data/restricted/inventory_out` lists the columns, codes and counts of each file, and the pipeline itself checks every file against the documentation of the scientific use files and reports the results in checks.csv. The number of diary days per country should match Annex 1 of that documentation, as recorded in ANNEX1_RECORDS in src/location_imputation.py.

| Country | Diary days (Annex 1) | Country | Diary days (Annex 1) |
|---|---:|---|---:|
| AT | 8,234 | IT | 38,234 |
| BE | 11,118 | LU | 4,162 |
| DE | 27,143 | NL | 14,035 |
| EE | 9,946 | NO | 7,882 |
| EL | 14,274 | PL | 78,759 |
| ES | 19,295 | RO | 56,514 |
| FI | 7,480 | RS | 6,835 |
| FR | 27,903 | UK | 16,118 |
| HU | 8,391 | | |

The files used for the results hold 27, 204, 2,102 and 40 fewer diaries than listed for FI, FR, PL and UK.

## What may leave data/restricted/

The pipeline writes all its outputs to the folder given with --out, by default data/restricted/imputation_out/ when run through the Makefile. Files whose names start with PRIVATE_ hold the imputed German diaries and their identifiers. They are microdata derived from the scientific use files and stay in data/restricted/. Under the conditions of access they must be deleted, together with the scientific use files, when the research project ends, and the principal researcher signs a declaration of destruction; the aggregate results reported in publications are exempt. The other files are aggregates, and src/export_results.py copies them, and only them, into results/.

## Publishing results

Eurostat's guidelines for publishing results based on the scientific use files (Access to confidential data for scientific purposes: guidelines for publication, https://ec.europa.eu/eurostat/documents/203647/771732/Guidelines+for+publication) apply the rules of EU-SILC to HETUS: results based on fewer than 20 respondents (unweighted) must not be published, results based on 20 to 49 respondents must be flagged, and a publication may state only that a result rests on fewer than 20 or on 20 to 49 respondents, not their number. The aggregate outputs of the pipeline rest on whole countries or on German groups of thousands of diaries, except the per-place scores of a country and the location shares of single activities, which are withheld where fewer than 50 respondents stand behind them. Eurostat also asks for a copy of every report produced with the data and for references to the research results in the project's space.
