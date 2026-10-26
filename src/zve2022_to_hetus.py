#!/usr/bin/env python3
"""Convert the diaries of the German time use survey of 2022 (ZVE 2022, scientific use file of
the research data centres of the statistical offices, FDZ) to the layout of the HETUS 2010
scientific use files, so that zve2022_holdout.py can impute and score them like a HETUS country.

  - Main activity, 175 codes of the 2022 activity list -> HETUS ACL 2008 codes of the 2010 SUF,
    through data/zve2022/activity_crosswalk.csv (built from the FDZ code list and the FDZ key
    between the activity codes of 2022 and 2012/13). Every non-travel code goes to a code that
    the 13 donors use; trips (921 to 992) go to a travel code (900 to 989).
  - Place or mode of transport, two-digit codes of the 2022 list -> WHEREP codes 1 to 13,
    through data/zve2022/location_crosswalk.csv: places 11 to 17 -> 1 to 7, other place (19)
    and unspecified place (10, 00) -> 13, modes 20 to 39 -> 8 to 12.
  - Age in years -> the 15 age bands of INC2 (10-17, 18-19, 20-24, ..., 75-79, 80+), or a
    table of age groups given in the column map.
  - Day of the week -> DDV1 (1 Sunday, 2 Monday, ..., 7 Saturday).
  - Diary-day weight -> WGHT1; household, person and diary-day numbers -> HID, PID, DIARY.

The names of the variables in the SUF files are read from a column map (JSON); see
data/zve2022/columns.json, whose entries marked TO_CONFIRM must be checked against the
product metadata report of the SUF (Metadatenreport Teil II). The script stops and lists the
columns of a file when a mapped variable is missing.

Writes the converted diaries (microdata: keep them under data/restricted/) and
conversion_summary.csv next to them, with aggregate counts only (share of slots by converted
code, codes that could not be converted).

Usage:
    python src/zve2022_to_hetus.py --slots path/to/slot_file.csv --persons path/to/person_file.csv \\
        [--days path/to/diary_day_file.csv] --columns data/zve2022/columns.json \\
        --out data/restricted/DE22.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = 144
REPO = Path(__file__).resolve().parent.parent
CROSSWALK_DIR = REPO / "data" / "zve2022"
AGE_EDGES = [10, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]   # INC2 bands 1 to 15


def need(frame, columns, what):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"{what}: column(s) {', '.join(missing)} not found. Columns in the file: "
                         + ", ".join(map(str, frame.columns[:80])) + (" ..." if len(frame.columns) > 80 else ""))


def read(path, sep):
    return pd.read_csv(path, sep=sep, low_memory=False, encoding_errors="replace")


def slot_matrix(frame, spec, keys, what):
    """Diary-day x 144 matrix of one diary item, from a wide layout (one column per slot,
    matched by spec['wide_pattern'], slot numbers 1 to 144 or 0 to 143) or a long layout
    (spec['column'] with the slot number in spec['slot_column'])."""
    if "wide_pattern" in spec:
        pat = re.compile(spec["wide_pattern"], re.IGNORECASE)
        cols = sorted(((int(m.group(1)), c) for c in frame.columns for m in [pat.match(str(c))] if m))
        if len(cols) != SLOTS:
            raise SystemExit(f"{what}: {len(cols)} columns match {spec['wide_pattern']!r}, expected {SLOTS}")
        return frame[keys].reset_index(drop=True), frame[[c for _, c in cols]].to_numpy()
    need(frame, [spec["column"], spec["slot_column"]] + keys, what)
    slot = pd.to_numeric(frame[spec["slot_column"]], errors="coerce")
    slot = slot - slot.min()                      # 1..144 or 0..143 -> 0..143
    wide = frame.assign(_slot=slot).pivot_table(index=keys, columns="_slot", values=spec["column"], aggfunc="first")
    if wide.shape[1] != SLOTS:
        raise SystemExit(f"{what}: {wide.shape[1]} slots per diary day, expected {SLOTS}")
    return wide.index.to_frame(index=False), wide.to_numpy()


def age_bands(values, spec):
    if spec.get("kind", "years") == "years":
        years = pd.to_numeric(values, errors="coerce").to_numpy()
        band = np.searchsorted(AGE_EDGES, years, side="right")      # 10-17 -> 1, ..., 80+ -> 15
        return np.where(np.isfinite(years) & (years >= 10), band, 0).astype(int)
    table = {str(k): int(v) for k, v in spec["groups"].items()}
    return values.astype(str).map(table).fillna(0).astype(int).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slots", required=True, help="SUF file with the diary items by ten-minute slot")
    parser.add_argument("--persons", required=True, help="SUF file of persons (age)")
    parser.add_argument("--days", default=None, help="SUF file of diary days, if day of the week or weight are there")
    parser.add_argument("--columns", default=str(CROSSWALK_DIR / "columns.json"), help="column map (JSON)")
    parser.add_argument("--out", required=True, help="output CSV in the HETUS layout (keep under data/restricted/)")
    args = parser.parse_args()
    cmap = json.loads(Path(args.columns).read_text())
    todo = [k for k, v in cmap.items() if "TO_CONFIRM" in json.dumps(v)]
    if todo:
        raise SystemExit(f"{args.columns}: confirm the entries {', '.join(todo)} against the product metadata report")
    sep = cmap.get("separator", ";")
    hh, pp, dd = cmap["household_id"], cmap["person_id"], cmap["diary_day"]
    keys = [hh, pp, dd]

    acts = pd.read_csv(CROSSWALK_DIR / "activity_crosswalk.csv", dtype={"hetus_acl2008": str})
    act_map = dict(zip(acts.code_2022.astype(int), acts.hetus_acl2008.astype(int)))
    locs = pd.read_csv(CROSSWALK_DIR / "location_crosswalk.csv", dtype={"code_2022": str})
    loc_map = dict(zip(locs.code_2022.astype(int), locs.wherep.astype(int)))

    wk, wt = cmap["weekday"], cmap["weight"]
    slots = read(args.slots, sep)
    need(slots, keys, args.slots)
    ids, A22 = slot_matrix(slots, cmap["main_activity"], keys, "main activity")
    ids2, L22 = slot_matrix(slots, cmap["place_or_mode"], keys, "place or mode of transport")
    if not ids.equals(ids2):
        raise SystemExit("the activity and place items do not cover the same diary days")
    # day of the week and weight: in the slot file or in the file of diary days
    day_vars = [wk["column"], wt]
    day_frame = read(args.days, sep) if args.days else None
    for var in day_vars:
        if var in slots.columns:
            source = slots
        elif day_frame is not None and var in day_frame.columns:
            source = day_frame
        else:
            raise SystemExit(f"{var} found neither in {args.slots} nor in the file of diary days (--days)")
        ids = ids.merge(source[keys + [var]].drop_duplicates(keys), on=keys, how="left")
    del slots, day_frame
    A22 = pd.DataFrame(A22).apply(pd.to_numeric, errors="coerce").to_numpy()
    L22 = pd.DataFrame(L22).apply(pd.to_numeric, errors="coerce").to_numpy()

    # activities and places through the crosswalks; anything else stays missing (-1, -9)
    A = np.full(A22.shape, -1, np.int16)
    for code, h in act_map.items():
        A[A22 == code] = h
    L = np.full(L22.shape, -9, np.int16)
    for code, w in loc_map.items():
        L[L22 == code] = w

    # diary-day and person variables
    weekday = pd.to_numeric(ids[wk["column"]], errors="coerce").to_numpy()
    first = int(wk.get("monday", 1))              # code of Monday in the SUF (Monday to Sunday in order)
    ddv1 = np.where(np.isfinite(weekday), ((weekday - first) % 7 + 1) % 7 + 1, 0).astype(int)   # Monday -> 2, Sunday -> 1
    persons = read(args.persons, sep)
    age_spec = cmap["age"]
    need(persons, [hh, pp, age_spec["column"]], args.persons)
    persons = persons[[hh, pp, age_spec["column"]]].drop_duplicates([hh, pp])
    ids = ids.merge(persons, on=[hh, pp], how="left")
    inc2 = age_bands(ids[age_spec["column"]], age_spec)

    out = pd.DataFrame({f"MACT{s + 1:03d}": A[:, s] for s in range(SLOTS)})
    out = pd.concat([out, pd.DataFrame({f"WHEREP{s + 1:03d}": L[:, s] for s in range(SLOTS)})], axis=1)
    out["INC2"] = inc2
    out["DDV1"] = ddv1
    out["WGHT1"] = pd.to_numeric(ids[wt], errors="coerce").to_numpy()
    out["HID"] = pd.factorize(ids[hh])[0] + 1           # anonymous running numbers
    out["PID"] = pd.to_numeric(ids[pp], errors="coerce").to_numpy()
    out["DIARY"] = pd.to_numeric(ids[dd], errors="coerce").to_numpy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # aggregate summary of the conversion
    rows = []
    n_slots = A22.size
    unmapped_a = sorted(set(np.unique(A22[np.isfinite(A22)]).astype(int)) - set(act_map))
    unmapped_l = sorted(set(np.unique(L22[np.isfinite(L22)]).astype(int)) - set(loc_map))
    rows.append(dict(item="diary days", value=len(out)))
    rows.append(dict(item="persons", value=int(ids[[hh, pp]].drop_duplicates().shape[0])))
    rows.append(dict(item="share of slots with an activity that could not be converted",
                     value=float((A < 0).sum() / n_slots)))
    rows.append(dict(item="activity codes that could not be converted", value=" ".join(map(str, unmapped_a))))
    rows.append(dict(item="share of slots with no place or mode", value=float((L < 0).sum() / n_slots)))
    rows.append(dict(item="place codes that could not be converted", value=" ".join(map(str, unmapped_l))))
    for w in sorted(set(loc_map.values())):
        rows.append(dict(item=f"share of slots with WHEREP {w}", value=float((L == w).sum() / n_slots)))
    rows.append(dict(item="share of diary days without age band", value=float((inc2 == 0).mean())))
    rows.append(dict(item="share of diary days without day of the week", value=float((ddv1 == 0).mean())))
    rows.append(dict(item="share of diary days without a positive weight", value=float((~(out.WGHT1 > 0)).mean())))
    pd.DataFrame(rows).to_csv(out_path.with_name("conversion_summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"written {out_path} ({len(out)} diary days)")


if __name__ == "__main__":
    sys.exit(main())
