"""Synthetic HETUS-like country files with the layout and quirks of the HETUS 2010 SUF.

Not real data. Diaries follow a simple generative model with known structure so the
pipeline's behaviour can be checked: places depend on activity, time, age and country,
trips usually separate place changes but not always, and activity episodes keep one place
except for a small share of work episodes. Some donors read on the train, a mode during
another activity, and some record television at home as other or unspecified place. The
files reproduce the quirks of the real ones: Austria's 15-minute grid, partial location
coding in AT, FR and LU, no places in DE, lower-case column names in RO, and the diary
counts of the files used for the paper.

Usage:
    python tests/make_synthetic.py OUTPUT_FOLDER [SCALE]
writes OUTPUT_FOLDER/hetus2010/*.csv and OUTPUT_FOLDER/TUS_Eurostat.nml. SCALE (default
0.02) multiplies the diary counts, with at least 200 diaries per country.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ANNEX1 = {"AT": 8234, "BE": 11118, "DE": 27143, "EE": 9946, "EL": 14274, "ES": 19295,
          "FI": 7453, "FR": 27699, "HU": 8391, "IT": 38234, "LU": 4162, "NL": 14035,
          "NO": 7882, "PL": 76657, "RO": 56514, "RS": 6835, "UK": 16078}
MODES = [8, 9, 10, 11, 12]


def country_params(code, rng):
    r = np.random.default_rng(zlib.crc32(code.encode()))   # same parameters in every run
    return dict(telework=r.uniform(0.03, 0.2), lunch_out=r.uniform(0.1, 0.45),
                walk=r.uniform(0.2, 0.5), visit=r.uniform(0.1, 0.35),
                modes=r.dirichlet([3, 1 if code != "NL" else 4, 5, 2, 0.5]),
                no_trip_lunch=r.uniform(0.3, 0.7), read_on_train=r.uniform(0.05, 0.25))


def diary(rng, P, age, weekend, german=False):
    seq = []  # (activity, place) per slot

    def add(act, place, n):
        seq.extend([(act, place)] * int(n))

    def trip(act, n):
        mode = rng.choice(MODES, p=P["modes"])
        add(act, mode, n)

    add(11, 1, rng.integers(12, 25))
    add(31, 1, rng.integers(2, 5))
    add(21, 1, rng.integers(2, 4))
    worker = (2 <= age <= 11) and not weekend and rng.random() < 0.65
    student = age == 1 and not weekend and rng.random() < 0.9
    if worker or student:
        main, travel_act = (111, 910) if worker else (211, 920)
        home_office = worker and rng.random() < P["telework"]
        place = 1 if home_office else 3
        if not home_office:
            trip(travel_act, rng.integers(2, 6))
            if rng.random() < P["read_on_train"]:    # reading on the train: a mode during another activity,
                add(travel_act if german else 831, 11, rng.integers(1, 4))   # which German diaries record as a trip
                if rng.random() < 0.3:               # still reading on arrival: the same activity on both
                    add(831, place, rng.integers(1, 3))   # sides of a transport boundary, two episodes
        first = rng.integers(18, 30)
        if worker and rng.random() < 0.02:          # a meeting elsewhere inside the episode
            add(main, place, first // 2); add(main, 13, 3); add(main, place, first - first // 2 - 3)
        else:
            add(main, place, first)
        if rng.random() < P["lunch_out"] and not home_office:
            if rng.random() < P["no_trip_lunch"]:
                add(21, 5, rng.integers(3, 6))       # place change without a recorded trip
            else:
                trip(950, 1); add(21, 5, rng.integers(3, 6)); trip(950, 1)
        else:
            add(21, place, rng.integers(2, 4))
        add(main, place, rng.integers(14, 26))
        if not home_office:
            trip(travel_act, rng.integers(2, 6))
    else:
        add(311, 1, rng.integers(4, 10)); add(321, 1, rng.integers(3, 9))
        if rng.random() < 0.5:
            trip(936, rng.integers(1, 4)); add(361, 6, rng.integers(3, 9)); trip(936, rng.integers(1, 4))
        if rng.random() < P["walk"] * (1.3 if age >= 12 else 1):
            add(611, 13, rng.integers(3, 10))        # walk from home, no trip recorded
        add(21, 1, rng.integers(3, 6))
        if rng.random() < P["visit"]:
            trip(950, rng.integers(1, 4)); add(512, 4, rng.integers(6, 18)); trip(950, rng.integers(1, 4))
        add(381 if age in (4, 5, 6) else 821, 1, rng.integers(6, 16))
    add(21, 1, rng.integers(3, 6))
    if rng.random() < 0.15:
        trip(960, 2); add(521, 13, rng.integers(9, 15)); trip(960, 2)
    add(821, 1, rng.integers(6, 18))
    seq = seq[:144]
    seq += [(11, 1)] * (144 - len(seq))
    a, p = map(np.array, zip(*seq))
    if not german and rng.random() < 0.05:
        p[(a == 821) & (p == 1)] = 13   # television at home recorded as other or unspecified place
    u = rng.random()
    if u < 0.02:
        p[p == 1] = 2          # a day at the second home
    elif u < 0.03:
        p[p == 1] = 7          # a day away, at a hotel
    return a, p


def make_country(code, n, rng):
    P = country_params(code, rng)
    age = rng.integers(1, 16 if code != "DE" else 15, n)
    ddv1 = rng.integers(1, 8, n)
    weekend = np.isin(ddv1, [1, 7])
    A = np.empty((n, 144), np.int16); L = np.empty((n, 144), np.int16)
    for i in range(n):
        A[i], L[i] = diary(rng, P, age[i], weekend[i], german=code == "DE")
    places = ~np.isin(L, MODES)
    if code == "FR":
        L[np.isin(L, [6, 7])] = 13
    if code == "LU":
        L[np.isin(L, [2, 6, 7])] = 13
    if code == "AT":
        L[places & (L != 1)] = 13
    if code == "DE":
        L[places] = 13
    cols = {"YEAR": 2010, "HID": np.arange(n) // 2 + 1, "PID": 1, "DIARY": np.arange(n) % 2 + 1,
            "COUNTRY": code, "DDV1": ddv1, "DDV3": rng.integers(1, 13, n), "INC2": age,
            "WGHT1": rng.uniform(0.5, 1.5, n), "WGHT2": rng.uniform(0.5, 1.5, n)}
    df = pd.DataFrame(cols)
    if code == "AT":   # 96 fifteen-minute slots: take the 10-minute slot holding each 15-minute midpoint
        idx = (15 * np.arange(96) + 7) // 10
        A, L = A[:, idx], L[:, idx]
    T = A.shape[1]
    df = pd.concat([df, pd.DataFrame(A, columns=[f"Mact{t + 1}" for t in range(T)]),
                    pd.DataFrame(L, columns=[f"Wherep{t + 1}" for t in range(T)])], axis=1)
    if code == "RO":
        df = df.rename(columns={c: c.lower() for c in ["DDV1", "DDV3", "WGHT1", "WGHT2"]})
    return df


def main(root, scale=0.02):
    root = Path(root)
    folder = root / "hetus2010"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    lines = ["&settings"]
    for code, n in ANNEX1.items():
        n = max(200, int(n * scale))
        df = make_country(code, n, rng)
        name = "TUS_SUF_A_DE_2010.csv" if code == "DE" else f"{code}.csv"
        df.to_csv(folder / name, index=False)
        lines.append(f"TUS_{code}_data = 'hetus2010/{name}'")
        print(code, n, flush=True)
    lines.append("/")
    (root / "TUS_Eurostat.nml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.02)
