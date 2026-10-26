#!/usr/bin/env python3
"""Inventory of the HETUS 2010-round country files used by location_imputation.py.

Writes aggregate counts and rates only. No row of microdata is written, and counts
from 1 to MIN_CELL - 1 are suppressed. The small settings CSVs named in the .nml
file (age groups, location mapping, activity mapping and the like) are copied too.

Usage: save the script anywhere inside the project folder and run
    python hetus_inventory.py
It looks for TUS_Eurostat.nml in its own folder, the current folder and their
parents (and in a Data subfolder of each), then resolves the file paths written
in the .nml. Optional arguments:
    python hetus_inventory.py /path/to/TUS_Eurostat.nml
    python hetus_inventory.py /path/to/folder/with/the/country/csv/files
    python hetus_inventory.py --out /path/to/output_folder
The output goes to inventory_out next to the script, or to the folder given with --out.
It holds aggregates only, but where_files_were_found.txt lists local paths.
"""
import argparse
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 10       # counts from 1 to MIN_CELL - 1 are suppressed; zeros are kept
CHUNK_ROWS = 20000  # diaries read at a time, to keep memory low
MAX_SETTINGS_MB = 5  # settings files larger than this are not copied

# Provisional reading of the 13 location codes, inferred from the November 2025
# probability table: 1-7 and 13 are places, 8-12 are modes of transport.
# The raw code counts are written as well, so nothing depends on these guesses.
PLACE_CODES = [1, 2, 3, 4, 5, 6, 7, 13]
TRANSPORT_CODES = [8, 9, 10, 11, 12]
HOME_CODE = 1
TRAVEL_ACTIVITY = (900, 990)  # activity codes 900 to 989 are treated as travel

COVARIATES = ["DDV1", "DDV3", "DDV6", "DDV7", "HHC1", "HHC3", "HHC4", "HHC5",
              "HHQ5", "HHQ9_1", "INC1", "INC2", "INC4_1", "IND6_1", "IND22_1",
              "IND42_1", "IND44", "IND44_1", "WGHT1", "WGHT2"]
WEIGHTS = {"WGHT1", "WGHT2"}  # only their availability is reported, not their values

# Records (diary days) per country in Annex 1 of the HETUS 2010 SUF documentation.
ANNEX1_RECORDS = {"AT": 8234, "BE": 11118, "DE": 27143, "EE": 9946, "EL": 14274,
                  "ES": 19295, "FI": 7480, "FR": 27903, "HU": 8391, "IT": 38234,
                  "LU": 4162, "NL": 14035, "NO": 7882, "PL": 78759, "RO": 56514,
                  "RS": 6835, "UK": 16118}

NML_COUNTRY = re.compile(r"^\s*TUS_([A-Za-z]{2})_data\s*=\s*['\"](.+?)['\"]", re.MULTILINE)
NML_SETTING = re.compile(r"^\s*(age_groups|energy_mapping|location_code_mapping|subarea_mapping|"
                         r"weekday_mapping|kindOfDay_mapping)\s*=\s*['\"](.+?)['\"]", re.MULTILINE)
# BE.csv, BE_2013.csv and similar: two letters, then the extension or a non-letter.
COUNTRY_FILE = re.compile(r"^([A-Za-z]{2})(?:[^A-Za-z][^/]*)?\.csv$", re.IGNORECASE)
SLOT_COLUMN = re.compile(r"^(MACT|WHEREP)(\d+)$")


def is_small(count):
    return 0 < count < MIN_CELL


def suppressed_rows(counts, keys):
    """Turn a {value: count} mapping into rows, blanking small counts."""
    rows = []
    for value, count in sorted(counts.items(), key=lambda kv: str(kv[0])):
        small = is_small(count)
        rows.append({**keys, "value": value,
                     "count": np.nan if small else int(count), "suppressed": small})
    return rows


def code_counts(values):
    """Counts of each code in an array, missing values reported as 'NA'."""
    flat = values.ravel()
    missing = np.isnan(flat)
    codes, counts = np.unique(flat[~missing].astype(np.int64), return_counts=True)
    out = Counter({int(c): int(n) for c, n in zip(codes, counts)})
    if missing.any():
        out["NA"] = int(missing.sum())
    return out


def count_newlines(path):
    """Raw line breaks in a file, to tell a short file from rows lost when parsing."""
    n = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            n += block.count(b"\n")
    return n


def locate_nml(starts):
    """First TUS_Eurostat.nml found in, or in a Data folder of, the start folders and their parents."""
    for start in starts:
        for folder in [start, *start.parents][:6]:
            for candidate in (folder / "TUS_Eurostat.nml", folder / "Data" / "TUS_Eurostat.nml"):
                if candidate.is_file():
                    return candidate.resolve()
    return None


def resolve(written, bases):
    """Turn a path written in the .nml into an existing file, trying several base folders.

    The .nml paths are relative to the folder the notebooks run from, which may not be
    the folder this script runs from. As a last resort the bare file name is looked up
    in each base folder (for example CSV files kept next to the script)."""
    written = Path(written)
    if written.is_absolute():
        return written if written.is_file() else None
    for base in bases:
        candidate = Path(os.path.normpath(base / written))
        if candidate.is_file():
            return candidate
    for base in bases:
        candidate = base / written.name
        if candidate.is_file():
            return candidate
    return None


def folder_files(folder):
    """Country files in a folder, keyed by country code."""
    found = {}
    for p in sorted(folder.iterdir()):
        match = COUNTRY_FILE.match(p.name)
        if match:
            code = match.group(1).upper()
            found[code if code not in found else p.stem] = p
    return found


def gather_inputs(source, script_dir):
    """Country files, settings files and a printable report of where each was found."""
    cwd = Path.cwd()
    nml, folder = None, None
    if source is not None:
        source = Path(source).expanduser()
        if source.is_dir():
            folder = source.resolve()
        elif source.is_file() and source.suffix.lower() == ".nml":
            nml = source.resolve()
        else:
            print(f"Note: {source} was not found, searching for TUS_Eurostat.nml instead.")
    if nml is None:
        nml = locate_nml([script_dir, cwd] + ([folder] if folder else []))
    bases = [cwd, script_dir] + ([nml.parent] if nml else []) + ([folder] if folder else [])
    countries, settings, report = {}, [], []
    if nml:
        report.append(f"Settings file: {nml}")
        text = nml.read_text()
        for code, written in NML_COUNTRY.findall(text):
            countries[code.upper()] = (written, resolve(written, bases))
        for _, written in NML_SETTING.findall(text):
            settings.append((written, resolve(written, bases)))
    else:
        report.append("Settings file TUS_Eurostat.nml not found: settings CSVs will not be copied.")
    if folder:
        for code, path in folder_files(folder).items():
            if code not in countries or countries[code][1] is None:
                countries[code] = (path.name, path)
    for code in sorted(countries):
        written, path = countries[code]
        report.append(f"  {code}: {path if path else 'NOT FOUND, written as ' + written}")
    return countries, settings, report


def describe_file(path):
    """Column names, slot columns and covariates of one file (header only)."""
    header = list(pd.read_csv(path, nrows=0).columns)
    by_upper = {}
    for col in header:
        by_upper.setdefault(col.strip().upper(), col)
    slots = {"MACT": {}, "WHEREP": {}}
    for upper, original in by_upper.items():
        match = SLOT_COLUMN.match(upper)
        if match:
            slots[match.group(1)][int(match.group(2))] = original
    covariates = [c for c in COVARIATES if c in by_upper]
    wanted = (list(slots["MACT"].values()) + list(slots["WHEREP"].values())
              + [by_upper[c] for c in covariates])
    info = {
        "n_columns": len(header),
        "n_slots_activity": len(slots["MACT"]),
        "n_slots_location": len(slots["WHEREP"]),
        "covariates_found": " ".join(covariates),
        "covariates_missing": " ".join(c for c in COVARIATES if c not in by_upper),
        "names_not_upper_case": " ".join(by_upper[c] for c in covariates if by_upper[c] != c),
    }
    return header, sorted(slots["MACT"]), sorted(slots["WHEREP"]), covariates, wanted, info


def sequence_counts(activity, location):
    """Additive counts that decide whether locations are drawn per slot, episode or stay."""
    n, n_slots = location.shape
    act = np.nan_to_num(activity, nan=-1).astype(np.int32).ravel()
    loc = np.nan_to_num(location, nan=-1).astype(np.int32).ravel()
    diary = np.repeat(np.arange(n, dtype=np.int32), n_slots)

    place = np.isin(loc, PLACE_CODES)
    transport = np.isin(loc, TRANSPORT_CODES)
    travel_activity = (act >= TRAVEL_ACTIVITY[0]) & (act < TRAVEL_ACTIVITY[1])
    travel = transport | travel_activity

    # Consecutive place slots of the same diary, possibly with non-place slots between.
    idx = np.flatnonzero(place)
    same = diary[idx[:-1]] == diary[idx[1:]]
    i, j = idx[:-1][same], idx[1:][same]
    change = loc[i] != loc[j]
    before = np.concatenate([[0], np.cumsum(travel, dtype=np.int32)])
    travel_between = (before[j] - before[i + 1]) > 0
    adjacent = (j - i) == 1
    same_activity = act[i] == act[j]

    # Activity episodes: runs of the same activity code within a diary.
    starts = np.r_[True, (act[1:] != act[:-1]) | (diary[1:] != diary[:-1])]
    episode = np.cumsum(starts, dtype=np.int64)
    pairs = np.unique(episode[place] * 100 + loc[place])  # distinct (episode, place)
    places_per_episode = np.bincount(pairs // 100)
    places_per_episode = places_per_episode[places_per_episode > 0]

    # First and last place of each diary.
    d = diary[idx]
    first = loc[idx[np.r_[True, d[1:] != d[:-1]]]] if len(idx) else np.array([])
    last = loc[idx[np.r_[d[1:] != d[:-1], True]]] if len(idx) else np.array([])

    return Counter({
        "diaries": n,
        "slots": n * n_slots,
        "slots_at_place_codes": int(place.sum()),
        "slots_at_transport_codes": int(transport.sum()),
        "slots_with_travel_activity": int(travel_activity.sum()),
        "travel_activity_slots_at_place_codes": int((travel_activity & place).sum()),
        "slots_at_other_or_missing_codes": int((~place & ~transport).sum()),
        "place_to_place_pairs": int(len(i)),
        "place_changes": int(change.sum()),
        "place_changes_with_travel_between": int((change & travel_between).sum()),
        "place_changes_without_travel_between": int((change & ~travel_between).sum()),
        "place_changes_between_adjacent_slots": int((change & adjacent).sum()),
        "place_changes_inside_activity_episode": int((change & adjacent & same_activity).sum()),
        "returns_to_same_place_after_travel": int((~change & travel_between).sum()),
        "episodes_with_place_slots": int(len(places_per_episode)),
        "episodes_with_more_than_one_place": int((places_per_episode > 1).sum()),
        "diaries_with_place_slots": int(len(first)),
        "diaries_first_place_home": int((first == HOME_CODE).sum()),
        "diaries_last_place_home": int((last == HOME_CODE).sum()),
    })


def sequence_rows(counts):
    rows = [{"statistic": k, "value": (np.nan if is_small(v) else v), "suppressed": is_small(v)}
            for k, v in counts.items()]
    safe = {k: v for k, v in counts.items() if not is_small(v)}

    def rate(num, den):
        if num in safe and den in safe and safe[den] > 0:
            return round(safe[num] / safe[den], 4)
        return np.nan

    rows += [
        {"statistic": "rate_place_changes_per_diary", "value": rate("place_changes", "diaries")},
        {"statistic": "share_changes_without_travel",
         "value": rate("place_changes_without_travel_between", "place_changes")},
        {"statistic": "share_changes_inside_episode",
         "value": rate("place_changes_inside_activity_episode", "place_changes")},
        {"statistic": "share_episodes_with_more_than_one_place",
         "value": rate("episodes_with_more_than_one_place", "episodes_with_place_slots")},
        {"statistic": "share_first_place_home",
         "value": rate("diaries_first_place_home", "diaries_with_place_slots")},
        {"statistic": "share_last_place_home",
         "value": rate("diaries_last_place_home", "diaries_with_place_slots")},
    ]
    return rows


def inventory(country, path):
    """All aggregate outputs for one country file."""
    header, act_slots, loc_slots, covariates, wanted, info = describe_file(path)
    n_rows, any_valid_location = 0, False
    activity_counts, location_counts, seq = Counter(), Counter(), Counter()
    covariate_counts = {v: Counter() for v in covariates if v not in WEIGHTS}
    weights_present = {v: 0 for v in covariates if v in WEIGHTS}
    if wanted:
        reader = pd.read_csv(path, usecols=wanted, chunksize=CHUNK_ROWS, low_memory=False)
    else:
        reader = pd.read_csv(path, usecols=[0], chunksize=CHUNK_ROWS, low_memory=False)
    for chunk in reader:
        n_rows += len(chunk)
        if not wanted:
            continue
        chunk = chunk.rename(columns={c: c.strip().upper() for c in chunk.columns})
        chunk = chunk.apply(pd.to_numeric, errors="coerce").astype("float32")
        activity = chunk[[f"MACT{t}" for t in act_slots]].to_numpy(dtype=float)
        activity_counts.update(code_counts(activity))
        for var in covariate_counts:
            covariate_counts[var].update(
                {("NA" if pd.isna(k) else int(k)): int(v)
                 for k, v in chunk[var].value_counts(dropna=False).items()})
        for var in weights_present:
            weights_present[var] += int((chunk[var] > 0).sum())
        if loc_slots:
            location = chunk[[f"WHEREP{t}" for t in loc_slots]].to_numpy(dtype=float)
            location_counts.update(code_counts(location))
            valid = location[~np.isnan(location)]
            if valid.size and (valid >= 0).any():
                any_valid_location = True
                if len(act_slots) == len(loc_slots):
                    seq.update(sequence_counts(activity, location))  # update keeps zeros

    expected = ANNEX1_RECORDS.get(country)
    info.update({
        "country": country, "file": path.name, "n_rows": n_rows,
        "raw_lines_minus_header": count_newlines(path) - 1,
        "annex1_records": expected,
        "rows_minus_annex1": (n_rows - expected) if expected else None,
        "location_all_missing_or_negative": not any_valid_location,
        "rows_with_positive_weight": " ".join(f"{k}:{v}" for k, v in weights_present.items()),
    })
    columns = [{"country": country, "position": k, "column": c} for k, c in enumerate(header)]
    activity_rows = suppressed_rows(activity_counts, {"country": country})
    location_rows = suppressed_rows(location_counts, {"country": country})
    covariate_rows = []
    for var, counts in covariate_counts.items():
        covariate_rows += suppressed_rows(counts, {"country": country, "variable": var})
    seq_rows = [{"country": country, **r} for r in sequence_rows(seq)] if seq else []
    return info, columns, location_rows, activity_rows, covariate_rows, seq_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=None,
                        help="TUS_Eurostat.nml, or a folder with the country CSV files (optional)")
    parser.add_argument("--out", default=None, help="output folder (default: inventory_out next to the script)")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    out = Path(args.out).expanduser() if args.out else script_dir / "inventory_out"
    out.mkdir(parents=True, exist_ok=True)

    found, settings_found, report = gather_inputs(args.source, script_dir)
    print("\n".join(report))
    countries = {c: p for c, (_, p) in found.items() if p is not None}
    missing = {c: w for c, (w, p) in found.items() if p is None}
    settings = [p for _, p in settings_found if p is not None]
    if not countries:
        raise SystemExit("\nNo country files found. Pass the folder that holds BE.csv, PL.csv and the "
                         "others, for example:\n    python hetus_inventory.py /path/to/that/folder")

    results = [[] for _ in range(6)]
    for country in sorted(countries):
        print(f"Reading {country} ({countries[country].name}) ...", flush=True)
        info, *tables = inventory(country, countries[country])
        results[0].append(info)
        for k, table in enumerate(tables, start=1):
            results[k] += table

    order = ["country", "file", "n_rows", "raw_lines_minus_header", "annex1_records",
             "rows_minus_annex1",
             "n_columns", "n_slots_activity", "n_slots_location",
             "location_all_missing_or_negative", "covariates_found", "covariates_missing",
             "names_not_upper_case", "rows_with_positive_weight"]
    names = ["01_files", "02_columns", "03_location_codes", "04_activity_codes",
             "05_covariates", "06_sequences"]
    pd.DataFrame(results[0])[order].to_csv(out / f"{names[0]}.csv", index=False)
    for name, rows in zip(names[1:], results[1:]):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)

    copied = []
    for p in settings:
        if p.exists() and p.stat().st_size <= MAX_SETTINGS_MB * 1e6:
            (out / "settings").mkdir(exist_ok=True)
            shutil.copy2(p, out / "settings" / p.name)
            copied.append(p.name)
    (out / "where_files_were_found.txt").write_text("\n".join(report) + "\n")
    if missing:
        print("\nNot found:", " ".join(sorted(missing)))
    (out / "README.txt").write_text(
        f"Aggregate inventory of the HETUS country files. Counts from 1 to {MIN_CELL - 1} "
        "are blank and flagged as suppressed. No microdata rows are included.\n"
        "01 files, row counts against Annex 1 and covariate availability; 02 column names; "
        "03 location code counts; 04 activity code counts; 05 covariate value counts; "
        "06 sequence statistics; settings/ copies of the settings CSVs named in the .nml: "
        + (", ".join(copied) if copied else "none") + ".\n")
    print(f"Done. Inventory written to {out.resolve()}")


if __name__ == "__main__":
    main()
