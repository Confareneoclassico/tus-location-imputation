"""Synthetic files with the content of the ZVE 2022 scientific use file, for testing
zve2022_to_hetus.py and zve2022_holdout.py. Not real data, and not the real layout: the
variable names are invented, and the column map written with the files says which they are.

Diaries come from the generator of tests/make_synthetic.py, as for a donor country (places
recorded, modes also during other activities), with the codes of the 2022 lists: every HETUS
code is written as a 2022 code that the crosswalk maps back to it, other places partly as
unspecified (10) and partly as other (19). Each person keeps three diary days, two on
weekdays and one at the weekend, and age is in years.

Usage:
    python tests/make_synthetic_zve2022.py OUTPUT_FOLDER [SCALE]
writes OUTPUT_FOLDER/zve2022/{slots,persons}.csv and OUTPUT_FOLDER/zve2022/columns.json.
SCALE multiplies 17,000 persons (default 0.05), with at least 200.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synthetic import country_params, diary   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PLACE_2022 = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 21, 9: 23, 10: 25, 11: 32, 12: 20}
AGE_OF_BAND = {1: (10, 17), 2: (18, 19)}


def main(root, scale=0.05):
    folder = Path(root) / "zve2022"
    folder.mkdir(parents=True, exist_ok=True)
    cw = pd.read_csv(REPO / "data" / "zve2022" / "activity_crosswalk.csv", dtype={"hetus_acl2008": str})
    to_2022 = {}
    for code, h in zip(cw.code_2022, cw.hetus_acl2008.astype(int)):
        to_2022.setdefault(h, code)            # the first 2022 code that maps to each HETUS code
    rng = np.random.default_rng(22)
    P = country_params("DE22", rng)
    n_persons = max(200, int(17000 * scale))
    slots, persons = [], []
    for i in range(n_persons):
        hh, pnr = i // 2 + 1, i % 2 + 1
        band = int(rng.integers(1, 15))
        lo, hi = AGE_OF_BAND.get(band, (5 * band + 5, 5 * band + 9))    # band 3 is 20-24, ..., 13 is 70-74
        age = int(rng.integers(lo, hi + 1)) if band < 14 else int(rng.integers(75, 90))
        persons.append(dict(hhnr=hh, pnr=pnr, alter=age))
        days = rng.choice([1, 2, 3, 4, 5], 2, replace=False).tolist() + [int(rng.choice([6, 7]))]
        for tag, wtag in enumerate(days, start=1):
            a, p = diary(rng, P, band, wtag >= 6)
            akt = np.array([to_2022[int(x)] for x in a])
            ort = np.array([PLACE_2022.get(int(x), 19) for x in p])
            other = p == 13
            ort[other & (rng.random(len(p)) < 0.3)] = 10      # some other places left unspecified
            row = dict(hhnr=hh, pnr=pnr, tagnr=tag, wtag=wtag, gew_tag=float(rng.uniform(500, 1500)))
            row.update({f"akt_{s + 1:03d}": akt[s] for s in range(144)})
            row.update({f"ort_{s + 1:03d}": f"{ort[s]:02d}" for s in range(144)})
            slots.append(row)
    pd.DataFrame(slots).to_csv(folder / "slots.csv", sep=";", index=False)
    pd.DataFrame(persons).to_csv(folder / "persons.csv", sep=";", index=False)
    cmap = dict(separator=";", household_id="hhnr", person_id="pnr", diary_day="tagnr",
                main_activity=dict(wide_pattern=r"^akt_(\d+)$"), place_or_mode=dict(wide_pattern=r"^ort_(\d+)$"),
                weekday=dict(column="wtag", monday=1), weight="gew_tag", age=dict(column="alter", kind="years"))
    (folder / "columns.json").write_text(json.dumps(cmap, indent=2) + "\n")
    print(f"{n_persons} persons, {len(slots)} diary days written to {folder}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
